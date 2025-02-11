"""
This file is part of summit_extras.

Developed for the LSST Data Management System.
This product includes software developed by the LSST Project (https://www.lsst.org).
See the COPYRIGHT file at the top-level directory of this distribution for details
of code ownership.

This program is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this
program.  If not, see <https://www.gnu.org/licenses/>.
"""

import ast
import logging
import os
import re
import sys
import traceback
import warnings
import operator
from io import StringIO
from typing import Any, Dict, List, Optional, Union, Literal, TypedDict, Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yaml

from IPython.display import Image, Markdown, display

import chromadb
from chromadb import Client, Settings
from annoy import AnnoyIndex

import langchain_community
from langchain.agents import (
    AgentType,
    Tool,
    create_react_agent,
    create_tool_calling_agent,
)
from langchain.agents.agent import AgentExecutor, RunnableAgent, RunnableMultiActionAgent
from langchain.schema import HumanMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_experimental.agents.agent_toolkits.pandas.prompt import (
    FUNCTIONS_WITH_DF,
    PREFIX,
    PREFIX_FUNCTIONS,
    SUFFIX_NO_DF,
    SUFFIX_WITH_DF,
)
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langchain_community.callbacks import get_openai_callback

from sentence_transformers import SentenceTransformer

from lsst.summit.utils.utils import getCurrentDayObs_int, getSite
from lsst.utils import getPackageDir
from lsst.utils.iteration import ensure_iterable

# Global flag and logger
INSTALL_NEEDED = False
LOG = logging.getLogger(__name__)

try:
    import openai  # noqa: F401
except ImportError:
    INSTALL_NEEDED = True
    LOG.warning("openai package not found. Please install openai: pip install openai")


def _check_installation() -> None:
    """Check that all required packages are installed.

    Raises
    ------
    RuntimeError
        If the openai package is missing.
    """
    if INSTALL_NEEDED:
        raise RuntimeError(
            "openai package not found. Please install openai: pip install openai"
        )


def set_api_key(filename: str = "~/.openaikey.txt") -> None:
    """Set the OpenAI API key from a file.

    The file should contain a single line with the API key.
    If the API key is already set, it will be overwritten.

    Parameters
    ----------
    filename : str
        Name of the file containing the API key.
    """
    _check_installation()
    current_key = os.getenv("OPENAI_API_KEY")
    if current_key:
        LOG.warning(
            f"OPENAI_API_KEY is already set. Overwriting with key from {filename}"
        )
    filename = os.path.expanduser(filename)
    with open(filename, "r") as file:
        api_key = file.readline().strip()
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key


def get_observing_data(day_obs: Optional[int] = None) -> pd.DataFrame:
    """Get the observing metadata for a given day.

    If no day is specified, the current day is used.

    Parameters
    ----------
    day_obs : int, optional
        The day for which to get the observing metadata.

    Returns
    -------
    pd.DataFrame
        The observing metadata.
    """
    current_day_obs = getCurrentDayObs_int()
    if day_obs is None:
        day_obs = current_day_obs
    is_current = day_obs == current_day_obs

    site = getSite()
    filename = None
    if site == "summit":
        filename = f"/project/rubintv/sidecar_metadata/dayObs_{day_obs}.json"
    elif site in ["rubin-devl"]:
        LOG.warning(
            "Observing metadata at rubin-devl is currently copied by hand by Merlin and "
            "will not be updated in realtime"
        )
        filename = f"/sdf/home/m/mfl/u/rubinTvDataProducts/sidecar_metadata/dayObs_{day_obs}.json"
    elif site in ["staff-rsp"]:
        LOG.warning(
            "Observing metadata at staff-rsp is currently copied by hand by Merlin and "
            "will not be updated in realtime"
        )
        filename = f"/home/m/mfl/u/rubinTvDataProducts/sidecar_metadata/dayObs_{day_obs}.json"
    else:
        raise RuntimeError(f"Observing metadata not available for site {site}")

    if not os.path.exists(filename):
        LOG.warning(
            f"Observing metadata file for {'current' if is_current else 'specified'} dayObs "
            f"{day_obs} does not exist at {filename}."
        )
        return pd.DataFrame()

    table = pd.read_json(filename).T
    table = table.sort_index()
    # Remove backend-specific columns starting with an underscore.
    table = table.drop([col for col in table.columns if col.startswith("_")], axis=1)
    return table


def convert_tuples_to_lists(data: Any) -> Any:
    """Recursively convert tuples in data to lists.

    Parameters
    ----------
    data : any
        Input data.

    Returns
    -------
    any
        Data with tuples converted to lists.
    """
    if isinstance(data, tuple):
        return list(data)
    if isinstance(data, list):
        return [convert_tuples_to_lists(item) for item in data]
    if isinstance(data, dict):
        return {key: convert_tuples_to_lists(value) for key, value in data.items()}
    return data


class ResponseFormatter:
    """Format and display the chatbot response, including intermediate steps."""

    def __init__(self, agent_type: str) -> None:
        """
        Parameters
        ----------
        agent_type : str
            The type of agent.
        """
        self.agent_type = agent_type

    def get_thought_from_action(self, action_log: str) -> str:
        """
        Extract the thought part from the action log.

        Parameters
        ----------
        action_log : str
            The log string from an action.

        Returns
        -------
        str
            Extracted thought.
        """
        for line in action_log.split("\n"):
            if line.startswith("Thought:"):
                return line.strip()
        return "No thought available."

    def extract_code_from_action(self, action: Any) -> Optional[str]:
        """
        Extract Python code from the action if applicable.

        Parameters
        ----------
        action : Any
            The action object.

        Returns
        -------
        Optional[str]
            The code snippet in markdown format, if found.
        """
        if action.tool == "python_repl":
            code = action.tool_input
            return f"```python\n{code}\n```" if code else None
        return None

    def print_observation(self, observation: Any) -> None:
        """
        Display the observation if it exists.

        Parameters
        ----------
        observation : any
            Observation output.
        """
        if observation and observation.strip():
            print(f"Observation: {observation}")

    def print_final_answer(self, response: Dict[str, Any]) -> None:
        """
        Print the final answer from the chat history.

        Parameters
        ----------
        response : dict
            The response object.
        """
        final_answer = response.get("chat_history", [None])[-1]
        if final_answer:
            print(f"\nFinal Answer: {final_answer}")

    def print_response(self, response: Dict[str, Any]) -> None:
        """
        Format and print the full response with intermediate steps.

        Parameters
        ----------
        response : dict
            The response object.
        """
        steps = response.get("intermediate_steps", [])
        if not steps:
            return

        for step_num, (action, observation) in enumerate(steps, start=1):
            thought = self.get_thought_from_action(action.log)
            print(f"\nStep {step_num}:")
            print(thought)
            print(f"Tool: {action.tool}")
            print(f"Tool input: {action.tool_input}")
            self.print_observation(observation)
            code_snippet = self.extract_code_from_action(action)
            if code_snippet:
                print("\nUsed Python code:")
                display(Markdown(code_snippet))
        self.print_final_answer(response)

    def __call__(self, response: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """
        Format the response for display.

        Parameters
        ----------
        response : dict or list of dict
            The agent's response.
        """
        if self.agent_type in ["tool-calling", "ZERO_SHOT_REACT_DESCRIPTION"]:
            if isinstance(response, list):
                for resp in response:
                    self.print_response(resp)
            else:
                self.print_response(response)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")


class Tools:
    """A collection of extra tools available to the AstroChat agent."""

    def __init__(self, chat_model: BaseLanguageModel, yaml_file_path: str,
                 csv_file_path: str) -> None:
        """
        Parameters
        ----------
        chat_model : BaseLanguageModel
            The language model for chat.
        yaml_file_path : str
            Path to the YAML configuration file.
        csv_file_path : str
            Path to the CSV file with queries.
        """
        self.data_dir = os.path.dirname(yaml_file_path)
        self.index_path = os.path.join(self.data_dir, "annoy_index.ann")
        self._chat = chat_model
        self.data = self.load_yaml(yaml_file_path)
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index, self.descriptions = self.load_or_build_annoy_index()
        self.query_finder = QueryFinderAgent(csv_file_path)

        self.tools = [
            Tool(
                name="SecretWord",
                func=self.secret_word,
                description="Useful for answering the secret word."
            ),
            Tool(
                name="NASAImage",
                func=self.nasa_image,
                description=(
                    "Useful for retrieving NASA image of the day for a given date "
                    "(self.date)."
                ),
            ),
            Tool(
                name="RandomMTG",
                func=self.random_mtg_card,
                description="Useful for showing a random Magic The Gathering card."
            ),
            Tool(
                name="YAMLTopicFinder",
                func=lambda prompt: self.find_topic_with_ai(prompt),
                description=(
                    "Finds the topic in the YAML file based on a description using AI."
                ),
            ),
            Tool(
                name="QueryFinder",
                func=self.query_finder_wrapper,
                description=(
                    "Finds a relevant query from the CSV file based on the input description."
                ),
            ),
            Tool(
                name="SampleQueries",
                func=self.sample_queries_wrapper,
                description="Returns a sample of queries from the CSV file."
            ),
        ]

    def query_finder_wrapper(self, input_text: str) -> str:
        """Wrapper for the query finder."""
        try:
            result = self.query_finder.findQuery(input_text)
            print(f"QueryFinder result: {result}")
            return result
        except Exception as e:
            print(f"Error in query_finder_wrapper: {e}")
            return f"An error occurred while finding a query: {str(e)}"

    def sample_queries_wrapper(self, n: int = 5) -> Union[str, List[Dict[str, Any]]]:
        """Wrapper for getting sample queries."""
        try:
            result = self.query_finder.getSampleQueries(n)
            print(f"SampleQueries result: {result}")
            return result
        except Exception as e:
            print(f"Error in sample_queries_wrapper: {e}")
            return f"An error occurred while getting sample queries: {str(e)}"

    @staticmethod
    def secret_word() -> str:
        """Return the secret word."""
        return "The secret word is 'Rubin'"

    def nasa_image(self, date: str) -> Optional[str]:
        """
        Retrieve and display the NASA image for a given date.

        Parameters
        ----------
        date : str
            The date for which to retrieve the image.

        Returns
        -------
        Optional[str]
            URL of the image if available.
        """
        url = (
            f"https://api.nasa.gov/planetary/apod?date={date}&api_key="
            "GXacxntSzk6wpkUmDVw4L1Gfgt4kF6PzZrmSNWBb"
        )
        response = requests.get(url)
        data = response.json()
        if "url" in data:
            image_url = data["url"]
            display(Image(url=image_url))
            return image_url
        print("No image available for this date.")
        return None

    def random_mtg_card(self) -> Optional[str]:
        """
        Retrieve and display a random Magic The Gathering card.

        Returns
        -------
        Optional[str]
            URL of the card image if available.
        """
        url = "https://api.scryfall.com/cards/random"
        response = requests.get(url)
        data = response.json()
        image_url = data.get("image_uris", {}).get("normal")
        if image_url:
            display(Image(url=image_url))
            return image_url
        return None

    def load_yaml(self, file_path: str) -> Any:
        """
        Load a YAML file.

        Parameters
        ----------
        file_path : str
            Path to the YAML file.

        Returns
        -------
        any
            Parsed YAML data.
        """
        with open(file_path, "r") as file:
            return yaml.load(file, Loader=yaml.SafeLoader)

    def load_or_build_annoy_index(self) -> Any:
        """
        Load or build the Annoy index for embeddings.

        Returns
        -------
        tuple
            The Annoy index and descriptions.
        """
        if os.path.exists(self.index_path):
            return self.load_annoy_index()
        index, descriptions = self.build_annoy_index()
        self.descriptions = descriptions
        self.save_annoy_index(index)
        return index, descriptions

    def build_annoy_index(self) -> Any:
        """
        Build the Annoy index from YAML data.

        Returns
        -------
        tuple
            The Annoy index and a list of descriptions.
        """
        index = AnnoyIndex(384, "angular")
        descriptions = []
        for telemetry_name, telemetry_data in self.data.items():
            if telemetry_name.endswith("_Telemetry") and isinstance(telemetry_data, dict):
                if "SALTelemetrySet" in telemetry_data:
                    sal_set = telemetry_data["SALTelemetrySet"]
                    if "SALTelemetry" in sal_set:
                        for telemetry in sal_set["SALTelemetry"]:
                            if isinstance(telemetry, dict):
                                description = telemetry.get("Description")
                                efdb_topic = telemetry.get(
                                    "EFDB_Topic", "No EFDB_Topic found"
                                )
                                if description:
                                    vector = self.embed_description(description)
                                    index.add_item(len(descriptions), vector)
                                    descriptions.append((description, efdb_topic))
        index.build(10)
        return index, descriptions

    def save_annoy_index(self, index: AnnoyIndex) -> None:
        """
        Save the Annoy index and corresponding descriptions.

        Parameters
        ----------
        index : AnnoyIndex
            The built Annoy index.
        """
        directory = os.path.dirname(self.index_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        index.save(self.index_path)
        LOG.info(f"Annoy index saved to {self.index_path}")
        normalized_descriptions = convert_tuples_to_lists(self.descriptions)
        descriptions_path = os.path.splitext(self.index_path)[0] + ".yaml"
        with open(descriptions_path, "w") as file:
            yaml.dump(normalized_descriptions, file)
        print(f"Descriptions saved to {descriptions_path}, without tuples")
        LOG.info(f"Descriptions saved to {descriptions_path}")

    def load_annoy_index(self) -> Any:
        """
        Load the Annoy index and descriptions.

        Returns
        -------
        tuple
            The Annoy index and descriptions.
        """
        index = AnnoyIndex(384, "angular")
        index.load(self.index_path)
        LOG.info(f"Annoy index loaded from {self.index_path}")
        descriptions_path = os.path.splitext(self.index_path)[0] + ".yaml"
        with open(descriptions_path, "r") as file:
            descriptions = yaml.safe_load(file)
        LOG.info(f"Descriptions loaded from {descriptions_path}")
        return index, descriptions

    def rebuild_annoy_index(self) -> None:
        """Rebuild the Annoy index."""
        index, descriptions = self.build_annoy_index()
        self.descriptions = descriptions
        self.save_annoy_index(index)
        self.index = index
        LOG.info("Annoy index rebuilt")

    def embed_description(self, description: str) -> List[float]:
        """
        Embed a description into a vector.

        Parameters
        ----------
        description : str
            The description text.

        Returns
        -------
        List[float]
            The embedding vector.
        """
        return self.sentence_model.encode(description).tolist()

    def find_topic_with_ai(self, prompt: str) -> str:
        """
        Find the topic using AI based on a prompt.

        Parameters
        ----------
        prompt : str
            The input prompt.

        Returns
        -------
        str
            The best matching EFDB topic.
        """
        query_vector = self.embed_description(prompt)
        nearest_indices = self.index.get_nns_by_vector(query_vector, 5)
        filtered_indices = self.filter_descriptions_based_on_keywords(prompt, nearest_indices)
        return self.choose_best_description(filtered_indices)

    def filter_descriptions_based_on_keywords(
        self, prompt: str, indices: List[int]
    ) -> List[int]:
        """
        Filter descriptions based on keywords from the prompt.

        Parameters
        ----------
        prompt : str
            The input prompt.
        indices : list of int
            Candidate indices.

        Returns
        -------
        list of int
            Filtered indices.
        """
        keywords = prompt.lower().split()
        filtered = [
            i for i in indices
            if any(keyword in self.descriptions[i][0].lower() for keyword in keywords)
        ]
        return filtered if filtered else indices

    def choose_best_description(self, indices: List[int]) -> str:
        """
        Choose the best description using AI.

        Parameters
        ----------
        indices : list of int
            Candidate indices.

        Returns
        -------
        str
            The best EFDB topic or an error message.
        """
        if not indices:
            return "No relevant descriptions found."
        best_descriptions = [self.descriptions[i] for i in indices]
        descriptions_text = "\n".join(
            f"{i+1}. {desc[0]}" for i, desc in enumerate(best_descriptions)
        )
        combined_prompt = (
            "Based on the following descriptions, choose the best one related to your prompt:\n\n"
            f"{descriptions_text}\n\n"
            "Choose the best description and explain why you selected that one."
        )
        response = self._chat.invoke([HumanMessage(content=combined_prompt)])
        response_content = response.content.strip()
        best_match = re.search(r"(\d+)\.\s(.+)", response_content)
        if best_match:
            choice_index = int(best_match.group(1)) - 1
            if 0 <= choice_index < len(best_descriptions):
                best_description, topic = best_descriptions[choice_index]
                return f"Best EFDB Topic: {topic}"
        return "AI response could not identify the best description."


class State(TypedDict):
    """State of the system."""
    chat_history: Annotated[List[str], operator.add]
    input: str
    agent_type: str
    short_term_memory: Dict[str, Any]


class AstroChat:
    """
    ASTROCHAT: Advanced Systems for Telemetry-Linked Realtime Observing and Chat-Based Help.

    A GPT-4 based chatbot for answering questions about Rubin Observatory conditions
    and observing metadata.
    """

    allowed_verbosities = ("COST", "ALL", "NONE", None)
    demo_queries = {
        "darktime": "What is the total darktime for Image type = bias?",
        "imageCount": "How many engtest and bias images were taken?",
        "expTime": "What is the total exposure time?",
        "obsBreakdown": (
            "What are the different kinds of observation reasons, and how many of each type? "
            "Please list as a table"
        ),
        "pieChart": (
            "Can you make a pie chart of the total integration time for each of these filters "
            "and image type = science? Please add a legend with total time in minutes"
        ),
        "pythonCode": (
            "Can you give me some python code that will produce a histogram of zenith angles for "
            "all entries with Observation reason = object and Filter = SDSSr_65mm"
        ),
        "azVsTime": (
            "Can you show me a plot of azimuth vs. time (TAI) for all lines with Observation "
            "reason = intra"
        ),
        "imageDegradation": (
            "Large values of mount image motion degradation is considered a bad thing. Is there "
            "a correlation with azimuth? Can you show a correlation plot?"
        ),
        "analysis": (
            "Act as an expert astronomer. It seems azimuth of around 180 has large values. I wonder "
            "if this is due to low tracking rate. Can you assess that by making a plot vs. tracking rate, "
            "which you will have to compute?"
        ),
        "correlation": (
            "Try looking for a correlation between mount motion degradation and declination. Show a plot "
            "with grid lines"
        ),
        "pieChartObsReasons": (
            "Please produce a pie chart of total exposure time for different categories of Observation reason"
        ),
        "airmassVsTime": (
            "I would like a plot of airmass vs time for all objects, as a scatter plot on a single graph, "
            "with the legend showing each object. Airmass of one should be at the top, with increasing airmass "
            "plotted as decreasing y position. Exclude points with zero airmass"
        ),
        "seeingVsTime": (
            "The PSF FWHM is an important performance parameter. Restricting the analysis to images with filter "
            "name that includes SDSS, can you please make a scatter plot of FWHM vs. time for all such images, "
            "with a legend. I want all data points on a single graph."
        ),
        "secretWord": "Tell me what is the secret word.",
        "nasaImage": "Show me the NASA image of the day for the current observation day.",
        "randomMTG": "Show me a random Magic The Gathering card",
        "find_topic_with_ai": "Find the EFDB_Topic based on the description of data.",
        "find_query": "Find a query related to exposure time",
        "sample_queries": "Show me some sample queries from the CSV file",
    }
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        day_obs: Optional[int] = None,
        export: bool = False,
        temperature: float = 0.0,
        verbosity: str = "COST",
        agent_type: str = "tool-calling",
        model_name: str = "gpt-4o-mini",
    ) -> None:
        """
        Parameters
        ----------
        data : pd.DataFrame, optional
            DataFrame a usar. Si se proporciona, se utiliza en lugar de obtener
            los metadatos mediante get_observing_data.
        day_obs : int, optional
            Día para el cual se obtienen los metadatos de observación.
        export : bool, optional
            Indica si se exportan variables tras cada llamada.
        temperature : float, optional
            Temperatura del modelo.
        verbosity : str, optional
            Nivel de verbosidad: 'COST', 'ALL' o 'NONE'.
        agent_type : str, optional
            Tipo de agente: "tool-calling" o "ZERO_SHOT_REACT_DESCRIPTION".
        model_name : str, optional
            Nombre del modelo de lenguaje a utilizar.
        """
        _check_installation()
        self.set_verbosity_level(verbosity)
        self._chat = ChatOpenAI(model_name=model_name, temperature=temperature)
    
        # Se asigna el DataFrame: si se pasa "data" se usa, si no se obtiene mediante get_observing_data
        if data is not None:
            self.data = data
        else:
            self.data = get_observing_data(day_obs)
    
        if day_obs is not None:
            day_str = str(day_obs)
            self.date = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:]}"
        else:
            self.date = ""
        if agent_type not in ("tool-calling", "ZERO_SHOT_REACT_DESCRIPTION"):
            raise ValueError("Invalid agent type")
        self.agent_type = agent_type
    
        prefix = (
            "You are running in an interactive environment, so if users ask for plots, "
            "making them will show them correctly. You also have access to a pandas dataframe, "
            "referred to as `df`, and should use this dataframe to answer user questions and generate "
            "plots from it.\n\n"
            "If the question is not related to pandas, you can use extra tools. The extra tools are: "
            "1. 'Secret Word', 2. 'NASA Image', 3. 'Random MTG', 4. 'YAML Topic Finder'. When using "
            "the 'NASA Image' tool, use 'self.date' as a date and do not use 'self.data'."
        )

        package_dir = getPackageDir("summit_extras")
        yaml_file_path = os.path.join(package_dir, "data", "sal_interface.yaml")
        csv_file_path = os.path.join(package_dir, "data", "generated_prompt_influxql.csv")
        toolkit = Tools(self._chat, yaml_file_path, csv_file_path)
        self.toolkit = toolkit.tools

        self.pandas_agent = custom_dataframe_agent(
            llm=self._chat,
            df=self.data,
            agent_type=self.agent_type,
            prefix=prefix,
            extra_tools=self.toolkit,
            allow_dangerous_code=True,
            return_intermediate_steps=True,
        )
        self.pandas_agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.pandas_agent, tools=self.toolkit, verbose=True
        )
        self.custom_agent = self.create_custom_agent(
            self._chat, self.data, self.agent_type, prefix, self.toolkit
        )
        self.database_agent = DatabaseAgent()
        self.graph = self._create_graph()
        self._total_callbacks = langchain_community.callbacks.openai_info.OpenAICallbackHandler()
        self.formatter = ResponseFormatter(agent_type=self.agent_type)
        self.export = export
        if self.export:
            warnings.warn(
                "Exporting variables from the agent after each call. This can cause problems!"
            )

    def create_custom_agent(
        self,
        llm: BaseLanguageModel,
        df: Any,
        agent_type: str,
        prefix: str,
        extra_tools: List[Tool],
    ) -> AgentExecutor:
        """Create a custom dataframe agent."""
        return custom_dataframe_agent(
            llm=llm,
            df=df,
            agent_type=agent_type,
            prefix=prefix,
            extra_tools=extra_tools,
            allow_dangerous_code=True,
            return_intermediate_steps=True,
        )

    def _create_graph(self) -> Any:
        """Create and compile the state graph for agent routing."""
        workflow = StateGraph(State)
        workflow.add_node("router", self.route)
        workflow.add_node("agent", self.agent_node)
        workflow.add_edge("router", "agent")
        workflow.add_edge("agent", END)
        workflow.set_entry_point("router")
        return workflow.compile()

    def route(self, state: State) -> Dict[str, Any]:
        """Route the state to the appropriate agent."""
        if "database" in state["input"].lower():
            return {"agent_type": "database", "short_term_memory": state.get("short_term_memory", {})}
        return {"agent_type": "custom", "short_term_memory": state.get("short_term_memory", {})}

    def run(self, input_str: str) -> Optional[str]:
        """
        Run the AstroChat agent with the given input.

        Parameters
        ----------
        input_str : str
            The input query.

        Returns
        -------
        Optional[str]
            The output response.
        """
        with get_openai_callback() as cb:
            try:
                result = self.graph.invoke({
                    "input": input_str,
                    "chat_history": [],
                    "agent_type": self.agent_type,
                    "short_term_memory": {},
                })
                self.formatter(result)
            except Exception as e:
                LOG.error(f"Agent faced an error: {str(e)}")
                traceback.print_exc()
                return f"An error occurred: {str(e)}"
            self._add_usage_and_display(cb)
        return None

    def agent_node(self, state: State) -> Dict[str, Any]:
        """
        Process the input using the appropriate agent and return a response.

        Parameters
        ----------
        state : State
            The current state.

        Returns
        -------
        dict
            Updated state with chat history.
        """
        try:
            short_term_memory = state.get("short_term_memory", {})
            if state["agent_type"] == "database":
                response = self.database_agent.run(state["input"])
            else:
                result = self.custom_agent.invoke(state["input"])
                intermediate_steps = []
                if isinstance(result, dict):
                    response = result.get("output", "")
                    intermediate_steps = result.get("intermediate_steps", [])
                else:
                    response = str(result)
                self.formatter.print_response({"intermediate_steps": intermediate_steps})
            short_term_memory["last_response"] = response
            return {"chat_history": [response], "short_term_memory": short_term_memory}
        except Exception as e:
            return {
                "chat_history": [f"Error: {str(e)}"],
                "short_term_memory": state.get("short_term_memory", {}),
            }

    def set_verbosity_level(self, level: str) -> None:
        """Set the verbosity level for output.

        Parameters
        ----------
        level : str
            The verbosity level.
        """
        if level not in self.allowed_verbosities:
            raise ValueError(f"Allowed values are {self.allowed_verbosities}, got {level}")
        self._verbosity = level

    def _add_usage_and_display(self, cb: Any) -> None:
        """Update callback usage and display cost information."""
        self._total_callbacks.total_cost += cb.total_cost
        self._total_callbacks.successful_requests += cb.successful_requests
        self._total_callbacks.completion_tokens += cb.completion_tokens
        self._total_callbacks.prompt_tokens += cb.prompt_tokens
        if self._verbosity == "ALL":
            print("\nThis call:\n" + str(cb) + "\n")
            print(self._total_callbacks)
        elif self._verbosity == "COST":
            print(
                f"\nThis call cost: ${cb.total_cost:.3f}, "
                f"session total: ${self._total_callbacks.total_cost:.3f}"
            )

    def list_demos(self) -> None:
        """List available demo queries."""
        print("Available demo keys and their associated queries:")
        print("-------------------------------------------------")
        for key, query in self.demo_queries.items():
            print(f"{key}: {query}\n")

    def run_demo(self, items: Optional[Union[str, List[str]]] = None) -> None:
        """
        Run demo queries.

        Parameters
        ----------
        items : str or list of str, optional
            Demo keys to run; if None, all demos are run.
        """
        known_demos = list(self.demo_queries.keys())
        if items is None:
            items = known_demos
        items = list(ensure_iterable(items))
        for item in items:
            if item not in known_demos:
                raise ValueError(
                    f"Demo item {item} is not available. Known demos: {known_demos}"
                )
        for item in items:
            print(f"\nRunning demo '{item}':")
            print(f"Prompt: {self.demo_queries[item]}")
            self.run(self.demo_queries[item])


def custom_dataframe_agent(
    llm: BaseLanguageModel,
    df: Any,
    agent_type: Union[AgentType, Literal["tool-calling", "openai-tools"]] = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_df_in_prompt: bool = True,
    number_of_head_rows: int = 5,
    extra_tools: List[Tool] = (),
    allow_dangerous_code: bool = False,
    return_intermediate_steps: bool = False,
) -> AgentExecutor:
    """
    Create a custom dataframe agent.

    Parameters
    ----------
    llm : BaseLanguageModel
        The language model.
    df : any
        The pandas DataFrame.
    agent_type : str
        The type of agent.
    prefix : str, optional
        Prompt prefix.
    suffix : str, optional
        Prompt suffix.
    include_df_in_prompt : bool, optional
        Whether to include the dataframe in the prompt.
    number_of_head_rows : int, optional
        Number of head rows to include.
    extra_tools : list of Tool, optional
        Additional tools.
    allow_dangerous_code : bool, optional
        Whether to allow dangerous code execution.
    return_intermediate_steps : bool, optional
        Whether to return intermediate steps.

    Returns
    -------
    AgentExecutor
        The created agent executor.
    """
    if not allow_dangerous_code:
        raise ValueError("Dangerous code execution is not allowed without opting in.")
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
    df_locals = {
        "df": df,
        "pd": pd,
        "plt": plt,
        "np": np,
        "display": display,
        "Markdown": Markdown,
        "Image": Image,
    }
    repl = CustomPythonREPL(locals=df_locals)
    tools = [
        Tool(
            name="python_repl",
            func=repl,
            description=(
                "A Python REPL. Execute python commands. Input should be valid python code. "
                "Output will be the result of execution."
            )
        )
    ] + extra_tools
    prompt = _generate_prompt(
        df,
        agent_type,
        prefix=prefix,
        suffix=suffix,
        include_df_in_prompt=include_df_in_prompt,
        number_of_head_rows=number_of_head_rows,
    )
    if agent_type == "ZERO_SHOT_REACT_DESCRIPTION":
        runnable = create_react_agent(llm, tools, prompt)
        agent = RunnableAgent(
            runnable=runnable,
            input_keys_arg=["input"],
            return_keys_arg=["output"],
        )
    elif agent_type == "tool-calling":
        runnable = create_tool_calling_agent(llm, tools, prompt)
        agent = RunnableMultiActionAgent(
            runnable=runnable,
            input_keys_arg=["input"],
            return_keys_arg=["output"],
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    return AgentExecutor(
        agent=agent,
        tools=tools,
        return_intermediate_steps=return_intermediate_steps,
        verbose=True
    )


def _generate_prompt(df: Any, agent_type: str, **kwargs: Any) -> BasePromptTemplate:
    """
    Generate a prompt for the agent based on its type.

    Parameters
    ----------
    df : any
        The pandas DataFrame.
    agent_type : str
        The type of agent.

    Returns
    -------
    BasePromptTemplate
        The prompt template.
    """
    if agent_type == "ZERO_SHOT_REACT_DESCRIPTION":
        return _get_single_prompt(df, **kwargs)
    if agent_type == "tool-calling":
        return _get_functions_single_prompt(df, **kwargs)
    raise ValueError(f"Unsupported agent type for prompt generation: {agent_type}")


def _get_single_prompt(
    df: Any,
    *,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_df_in_prompt: bool = True,
    number_of_head_rows: int = 5,
) -> BasePromptTemplate:
    """
    Generate a single prompt template.

    Parameters
    ----------
    df : any
        The pandas DataFrame.
    prefix : str, optional
        Prompt prefix.
    suffix : str, optional
        Prompt suffix.
    include_df_in_prompt : bool, optional
        Whether to include the dataframe preview.
    number_of_head_rows : int, optional
        Number of rows for the preview.

    Returns
    -------
    BasePromptTemplate
        The prompt template.
    """
    suffix_to_use = suffix if suffix else (SUFFIX_WITH_DF if include_df_in_prompt else SUFFIX_NO_DF)
    prefix = prefix or PREFIX
    template = "\n\n".join([prefix, "{tools}", FORMAT_INSTRUCTIONS, suffix_to_use])
    prompt = PromptTemplate.from_template(template).partial()
    if "df_head" in prompt.input_variables:
        df_head = df.head(number_of_head_rows).to_markdown()
        prompt = prompt.partial(df_head=str(df_head))
    return prompt


def _get_functions_single_prompt(
    df: Any,
    *,
    prefix: Optional[str] = None,
    suffix: str = "",
    include_df_in_prompt: bool = True,
    number_of_head_rows: int = 5,
) -> ChatPromptTemplate:
    """
    Generate a functions prompt template.

    Parameters
    ----------
    df : any
        The pandas DataFrame.
    prefix : str, optional
        Prompt prefix.
    suffix : str
        Prompt suffix.
    include_df_in_prompt : bool, optional
        Whether to include the dataframe preview.
    number_of_head_rows : int, optional
        Number of rows for the preview.

    Returns
    -------
    ChatPromptTemplate
        The functions prompt template.
    """
    df_head = df.head(number_of_head_rows).to_markdown() if include_df_in_prompt else ""
    suffix = (suffix or FUNCTIONS_WITH_DF).format(df_head=df_head)
    prefix = prefix or PREFIX_FUNCTIONS
    system_message = SystemMessage(content=prefix + suffix)
    return OpenAIFunctionsAgent.create_prompt(system_message=system_message)


class DatabaseAgent:
    """A placeholder Database Agent."""

    def run(self, input_str: str) -> str:
        """
        Return a placeholder response.

        Parameters
        ----------
        input_str : str
            The input query.

        Returns
        -------
        str
            Placeholder response.
        """
        return "This is a placeholder response from the DatabaseAgent."


class QueryFinderAgent:
    """Agent to find queries from a CSV file."""

    def __init__(self, csv_file_path: str) -> None:
        """
        Parameters
        ----------
        csv_file_path : str
            Path to the CSV file containing queries.
        """
        self.csv_file_path = csv_file_path
        self.queries_df = None
        self.load_queries()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = Client(Settings())
        self.collection_name = "question_embeddings"
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        self.prepare_embeddings()

    def load_queries(self) -> None:
        """Load queries from the CSV file."""
        if os.path.exists(self.csv_file_path):
            try:
                self.queries_df = pd.read_csv(
                    self.csv_file_path, header=0, names=["name", "query", "question"]
                )
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                self.queries_df = None
        else:
            print("CSV file not found")

    def prepare_embeddings(self) -> None:
        """Prepare embeddings for the queries."""
        if self.queries_df is not None and not self.queries_df.empty:
            questions = self.queries_df["question"].tolist()
            embeddings = self.embedding_model.encode(questions)
            ids = [str(idx) for idx in range(len(embeddings))]
            metadata = [
                {
                    "query": self.queries_df.iloc[idx]["query"],
                    "question": self.queries_df.iloc[idx]["question"],
                }
                for idx in range(len(embeddings))
            ]
            self.chroma_collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadata,
                documents=questions,
            )

    def findQuery(self, input_text: str) -> str:
        """
        Find a query matching the input text.

        Parameters
        ----------
        input_text : str
            The input text.

        Returns
        -------
        str
            The matching query or a message if none is found.
        """
        input_embedding = self.embedding_model.encode([input_text])[0]
        results = self.chroma_collection.query(
            query_embeddings=[input_embedding.tolist()], n_results=1
        )
        if results and results["distances"][0]:
            similar_query = results["metadatas"][0][0]
            distance = results["distances"][0][0]
            return f"Matching query found: {similar_query['query']} (Similarity: {1 - distance:.2f})"
        return "No matching query found."

    def getSampleQueries(self, n: int = 5) -> Union[str, List[Dict[str, Any]]]:
        """
        Return a sample of queries.

        Parameters
        ----------
        n : int, optional
            Number of queries to sample.

        Returns
        -------
        list or str
            Sample queries as a list of dictionaries or an error message.
        """
        if self.queries_df is not None and not self.queries_df.empty:
            sample = self.queries_df.sample(n=min(n, len(self.queries_df)))
            return sample[["query", "question"]].to_dict("records")
        return "No queries available."


class CustomPythonREPL:
    """A custom Python REPL for executing code in a given namespace."""

    def __init__(self, locals: Optional[Dict[str, Any]] = None) -> None:
        """
        Parameters
        ----------
        locals : dict, optional
            The local namespace for code execution.
        """
        self.locals = locals or {}
        self.locals["_last_executed_code"] = ""
        self.locals["_last_result"] = ""

    def run(self, command: str) -> str:
        """
        Execute a Python command.

        Parameters
        ----------
        command : str
            The Python command to execute.

        Returns
        -------
        str
            The captured output and result.
        """
        command = command.strip().strip("`")
        if command.startswith("python"):
            command = command[len("python"):].strip()
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            self.locals["_last_executed_code"] = command
            tree = ast.parse(command)
            exec(compile(tree, "<string>", "exec"), self.locals)
            sys.stdout.flush()
            output = sys.stdout.getvalue()
            if isinstance(tree.body[-1], ast.Expr):
                self.locals["_last_result"] = eval(
                    compile(ast.Expression(tree.body[-1].value), "<string>", "eval"),
                    self.locals,
                )
            else:
                self.locals["_last_result"] = "Command executed successfully (no return value)."
            result = str(self.locals["_last_result"])
            return output + "\n" + result if output else result
        except Exception as e:
            return f"An error occurred: {str(e)}"
        finally:
            sys.stdout = old_stdout

    def __call__(self, command: str) -> str:
        """Allow the REPL to be called as a function."""
        return self.run(command)