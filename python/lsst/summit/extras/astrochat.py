# This file is part of summit_extras.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import os
import pandas as pd
import re
import warnings
import inspect

from lsst.summit.utils.utils import getCurrentDayObs_int, getSite
from lsst.utils.iteration import ensure_iterable
from IPython.display import display, Markdown, Image

INSTALL_NEEDED = False
LOG = logging.getLogger(__name__)

# TODO: work out a way to protect all of these imports
import langchain  # noqa: E402
import langchain_community
import langchain_experimental
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback  # noqa: E402
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType


try:
    import openai
except ImportError:
    INSTALL_NEEDED = True
    LOG.warning("openai package not found. Please install openai: pip install openai")


def _checkInstallation():
    """Check that all required packages are installed.

    Raises a RuntimeError if any are missing so that we can fail early.
    """
    if INSTALL_NEEDED:
        raise RuntimeError("openai package not found. Please install openai: pip install openai")


def setApiKey(filename="~/.openaikey.txt"):
    """Set the OpenAI API key from a file.

    Set the OpenAI API key from a file. The file should contain a single line
    with the API key. The file name can be specified as an argument. If the
    API key is already set, it will be overwritten, with a warning issues.

    Parameters
    ----------
    filename : `str`
        Name of the file containing the API key.
    """
    _checkInstallation()

    currentKey = os.getenv('OPENAI_API_KEY')
    # if currentKey:
    #     LOG.warning(f"OPENAI_API_KEY is already set. Overwriting with key from {filename}")

    filename = os.path.expanduser(filename)
    with open(filename, 'r') as file:
        apiKey = file.readline().strip()

    openai.api_key = apiKey
    os.environ["OPENAI_API_KEY"] = apiKey


def getObservingData(dayObs=None):
    """Get the observing metadata for the current or a past day.

    Get the observing metadata for the current or a past day. The metadata
    is the contents of the table on RubinTV. If a day is not specified, the
    current day is used. The metadata is returned as a pandas dataframe.

    Parameters
    ----------
    dayObs : `int`, optional
        The day for which to get the observing metadata. If not specified,
        the current day is used.

    Returns
    -------
    observingData: `pandas.DataFrame`
        The observing metadata for the specified day.
    """
    currentDayObs = getCurrentDayObs_int()
    if dayObs is None:
        dayObs = currentDayObs
    isCurrent = dayObs == currentDayObs

    site = getSite()

    filename = None
    if site == 'summit':
        filename = f"/project/rubintv/sidecar_metadata/dayObs_{dayObs}.json"
    elif site in ['rubin-devl']:
        # LOG.warning(f"Observing metadata at {site} is currently copied by hand by Merlin and will not be "
                    # "updated in realtime")
        filename = f"/sdf/home/m/mfl/u/rubinTvDataProducts/sidecar_metadata/dayObs_{dayObs}.json"
    elif site in ['staff-rsp']:
        # LOG.warning(f"Observing metadata at {site} is currently copied by hand by Merlin and will not be "
                    # "updated in realtime")
        filename = f"/home/m/mfl/u/rubinTvDataProducts/sidecar_metadata/dayObs_{dayObs}.json"
    else:
        raise RuntimeError(f"Observing metadata not available for site {site}")

    # check the file exists, and raise if not
    if not os.path.exists(filename):
        # LOG.warning(f"Observing metadata file for {'current' if isCurrent else ''} dayObs "
                    # f"{dayObs} does not exist at {filename}.")
        return pd.DataFrame()

    table = pd.read_json(filename).T
    table = table.sort_index()

    # remove all the columns which start with a leading underscore, as these
    # are used by the backend to signal how specific cells should be colored
    # on RubinTV, and for nothing else.
    table = table.drop([col for col in table.columns if col.startswith('_')], axis=1)

    return table


class ResponseFormatter:
    """Format the response from the chatbot.

    Format the intermiediate responses from the chatbot. This is a simple class
    which provides a __call__ method which can be used as a callback for the
    chatbot. The __call__ method formats the response as a markdown table.
    """

    def __init__(self, agentType):
        self.agentType = agentType
        self.allCode = []

    @staticmethod
    def getThoughtFromAction(action):
        logMessage = action.log

        # Split the input string based on "Action:" or "Action Input:" and take
        # the first part
        splitParts = re.split(r'Action:|Action Input:', logMessage)
        thought = splitParts[0].strip()
        return thought

    def printAction(self, action):
        if action.tool == 'python_repl_ast':
            print('\nExcuted the following code:')
            code = action.tool_input
            if not code.startswith('```'):
                self.allCode.append(code)
                display(Markdown(f"```python\n{code}\n```"))
            else:
                display(Markdown(code))
                self.allCode.extend(code.split('\n')[1:-1])
        else:
            print(f'Tool: {action.tool}')
            print(f'Tool input: {action.tool_input}')

    @staticmethod
    def printObservation(observation):
        # observation can be many things, including an empty string
        # but can be a pd.Series or maybe even potentially other things.
        # So we need to check for None and empty string, and then just return
        # and otherwise print whatever we get, because testing truthiness
        # of other objects is tricky as pd.Series need any()/all() called etc
        if (isinstance(observation, str) and observation == '') or observation is None:
            return
        print(f"Observation: {observation}")

    def printResponse(self, response):
        steps = response["intermediate_steps"]
        nSteps = len(steps)
        if nSteps > 1:
            print(f'There were {len(steps)} steps to the process:\n')
        for stepNum, step in enumerate(steps):
            if nSteps > 1:
                print(f'Step {stepNum + 1}:')
            action, observation = step  # unpack the tuple
            thought = self.getThoughtFromAction(action)

            print(thought)
            self.printAction(action)
            self.printObservation(observation)

        output = response["output"]
        print(f'\nFinal answer: {output}')

    @staticmethod
    def pprint(responses):
        print(f"Length of responses: {len(responses)}")
        steps = responses['intermediate_steps']
        print(f"with {len(steps)} steps\n")
        for stepNum, step in enumerate(steps):
            action, logs = step
            if action.tool == 'python_repl_ast':
                code = action.tool_input['query']
                print(f"Step {stepNum + 1}")
                display(Markdown(f"```python\n{code}\n```"))
                print(f"logs: {logs}")

    def __call__(self, response):
        """Format the response for notebook display.

        Parameters
        ----------
        response : `str`
            The response from the chatbot.

        Returns
        -------
        formattedResponse : `str`
            The formatted response.
        """
        if self.agentType == 'tool-calling':
            self.pprint(response)
            return
        elif self.agentType == 'ZERO_SHOT_REACT_DESCRIPTION':
            self.printResponse(response)
            allCode = self.allCode
            self.allCode = []
            return allCode
        else:
            raise ValueError(f"Unknown agent type: {self.agentType}")


# =========================================================== CUSTOM TOOL
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.agents import Tool
from datetime import datetime
import requests
import yaml
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
import annoy
from annoy import AnnoyIndex
import numpy as np


# def load_yaml(file_path):
#     with open(file_path, 'r') as file:
#         data = yaml.safe_load(file)
#     return data


class Tools():
    def __init__(self, chat_model, data) -> None:
        self.data = data
        self._chat = chat_model
        yaml_file_path = 'lsst/summit_extras/python/lsst/summit/extras/sal_interface.yaml'
        self.data = self.load_yaml(yaml_file_path)
        self.index = self.build_annoy_index()

        self.tools = [
            Tool(
                name = "Secret Word",
                func = self.secret_word,
                description = "Useful for when you need to answer what is the secret word"
            ),
            Tool(
                name = "NASA Image",
                func = self.nasa_image,
                description = "Useful for when you need to answer what is the NASA image of the day for a given date (self.date)"
            ),
            Tool(
                name = "Random MTG",
                func = self.random_mtg_card,
                description = "Useful for when you need to show a random Magic The Gathering card"
            ),
            Tool(
                name="YAML Topic Finder",
                func=self.find_topic,
                description="Finds the topic in the YAML file based on the description provided."
            ),
            Tool(
                name="YAML Topic Finder",
                func=lambda prompt: self.find_topic_with_ai(prompt),
                description="Finds the topic in the YAML file based on the description provided using AI."
            )
            # Tool(
            #     name="YAML Topic Finder",
            #     func=self.find_topic,
            #     description="Finds the topic in the YAML file based on the description provided."
            # )
        ]


    @staticmethod
    def secret_word(self):
        return "The secret word is 'Rubin'"


    def nasa_image(self, date):
        # NASA API URL
        url = f"https://api.nasa.gov/planetary/apod?date={date}&api_key=GXacxntSzk6wpkUmDVw4L1Gfgt4kF6PzZrmSNWBb"

        # Make the API request
        response = requests.get(url)
        data = response.json()

        # Check if the response contains an image URL
        if 'url' in data:
            image_url = data['url']

            # Display the image directly in the notebook
            display(Image(url=image_url))
            return image_url

        else:
            print("No image available for this date.")
            return None


    @staticmethod
    def random_mtg_card(self):
        url = 'https://api.scryfall.com/cards/random'
        response = requests.get(url)
        data = response.json()

        image_url = data['image_uris']['normal']

        display(Image(url=image_url))

        return image_url

    def load_yaml(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def build_annoy_index(self):
        index = AnnoyIndex(512, 'angular')  # Adjust the vector length based on your embeddings
        self.descriptions = []  # Store descriptions and topics for later use

        # Iterate over all entries in the loaded YAML data
        for telemetry_name, telemetry_data in self.data.items():
            # Check if the telemetry name ends with '_Telemetry'
            if telemetry_name.endswith('_Telemetry') and isinstance(telemetry_data, dict):
                if 'SALTelemetrySet' in telemetry_data:
                    sal_telemetry_set = telemetry_data['SALTelemetrySet']
                    if 'SALTelemetry' in sal_telemetry_set:
                        for telemetry in sal_telemetry_set['SALTelemetry']:
                            if isinstance(telemetry, dict):
                                description = telemetry.get('Description')
                                efdb_topic = telemetry.get('EFDB_Topic', 'No EFDB_Topic found')
                                if description:
                                    vector = self.embed_description(description)  # Convert description to vector
                                    index.add_item(len(self.descriptions), vector)
                                    # Keep track of the description and its corresponding EFDB topic
                                    self.descriptions.append((description, efdb_topic))

        index.build(10)  # Build the Annoy index with 10 trees
        # print(self.descriptions, "************Description after annoy build**********")
        return index

    def embed_description(self, description: str):
        # Placeholder for an actual embedding function
        # You should replace this with an actual embedding logic
        return np.random.rand(512).astype('float32').tolist()  # Example: random vector

    def find_topic_with_ai(self, prompt: str):
        query_vector = self.embed_description(prompt)  # Convert prompt to vector
        nearest_indices = self.index.get_nns_by_vector(query_vector, 5)  # Get 5 nearest neighbors

        # Filter descriptions based on keywords from the prompt
        filtered_indices = self.filter_descriptions_based_on_keywords(prompt, nearest_indices)

        # Use AI to select the best description and topic from filtered indices
        chosen_topic_info = self.choose_best_description(filtered_indices)
        return chosen_topic_info

    def filter_descriptions_based_on_keywords(self, prompt: str, indices):
        keywords = prompt.lower().split()  # Simple keyword extraction
        filtered_indices = []

        for i in indices:
            description, _ = self.descriptions[i]
            if any(keyword in description.lower() for keyword in keywords):
                filtered_indices.append(i)

        return filtered_indices if filtered_indices else indices  # Fallback to original indices if none match

    def choose_best_description(self, indices):
        if not indices:
            return "No relevant descriptions found."

        best_descriptions = [self.descriptions[i] for i in indices]
        descriptions_text = "\n".join(f"{i + 1}. {desc[0]}" for i, desc in enumerate(best_descriptions))

        combined_prompt = (
            f"Based on the following descriptions, choose the best one related to your prompt:\n\n"
            f"{descriptions_text}\n\n"
            f"Choose the best description and explain why you selected that one."
        )

        response = self._chat([HumanMessage(content=combined_prompt)])  # AI chat function placeholder
        response_content = response.content.strip()

        # Attempt to extract explanation and chosen topic
        explanation_match = re.search(r'^(.*?)(?=\n\d+\.|$)', response_content)
        topic_match = re.search(r'\b(\d+)\b', response_content)

        if topic_match:
            choice_index = int(topic_match.group(1)) - 1
            if 0 <= choice_index < len(best_descriptions):
                best_description, topic = best_descriptions[choice_index]
                explanation = explanation_match.group(1) if explanation_match else "No explanation provided."
                return f"Best Description: {best_description}\nEFDB Topic: {topic}\nExplanation: {explanation}"

        return f"AI response could not be parsed. Raw response: {response_content}"


toolkit = Tools().tools
# ===========================================================


class AstroChat:
    allowedVerbosities = ('COST', 'ALL', 'NONE', None)
    @staticmethod
    def load_yaml(file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    demoQueries = {
        'darktime': 'What is the total darktime for Image type = bias?',
        'imageCount': 'How many engtest and bias images were taken?',
        'expTime': 'What is the total exposure time?',
        'obsBreakdown': 'What are the different kinds of observation reasons, and how many of each type? Please list as a table',
        'pieChart': 'Can you make a pie chart of the total integration time for each of these filters and image type = science? Please add a legend with total time in minutes',
        'pythonCode': 'Can you give me some python code that will produce a histogram of zenith angles for all entries with Observation reason = object and Filter = SDSSr_65mm',
        'azVsTime': 'Can you show me a plot of azimuth vs. time (TAI) for all lines with Observation reason = intra',
        'imageDegradation': 'Large values of mount image motion degradation is considered a bad thing. Is there a correlation with azimuth? Can you show a correlation plot?',
        'analysis': 'Act as an expert astronomer. It seems azimuth of around 180 has large values. I wonder is this is due to low tracking rate. Can you assess that by making a plot vs. tracking rate, which you will have to compute?',
        'correlation': 'Try looking for a correlation between mount motion degradation and declination. Show a plot with grid lines',
        'pieChartObsReasons': 'Please produce a pie chart of total exposure time for different categories of Observation reason',
        'airmassVsTime': 'I would like a plot of airmass vs time for all objects, as a scatter plot on a single graph, with the legend showing each object. Airmass of one should be at the top, with increasing airmass plotted as decreasing y position. Exclude points with zero airmass',
        'seeingVsTime': 'The PSF FWHM is an important performance parameter. Restricting the analysis to images with filter name that includes SDSS, can you please make a scatter plot of FWHM vs. time for all such images, with a legend. I want all data points on a single graph.',
        'secretWord': 'Tell me what is the secret word.',
        'nasaImage': 'Show me the NASA image of the day for the current observation day.',
        'randomMTG': 'Show me a random Magic The Gathering card',
        # 'find_topic': 'What is the topic to find query?',
        'find_topic_with_ai': 'Find the EFDB_Topic based on the description of data.'
    }

    def __init__(
            self,
            dayObs=None,
            export=False,
            temperature=0.0,
            verbosity='COST',
            agentType='tool-calling',
            modelName='gpt-4-1106-preview',
        ):
        """Create an ASTROCHAT bot.

        ASTROCHAT: "Advanced Systems for Telemetry-Linked Realtime Observing
            and Chat-Based Help with Astronomy Tactics"

        A GPT-4 based chatbot for answering questions about Rubin Observatory
        conditions and observing metadata.

        Note that ``verbosity`` here is the verbosity of this interface, not of
        the underlying model, which has been pre-tuned for this application.

        Note that the use of export is only for notebook usage, and can easily
        cause problems, as any name clashes from inside the agent with
        variables you have defined in your notebook will be overwritten. The
        same functionality can be achieved by calling the
        ``exportLocalVariables`` function whenever you want to get things out,
        but this permanent setting is provided for notebook convenience (only).

        Parameters
        ----------
        dayObs : `int`, optional
            The day for which to get the observing metadata. If not specified,
            the current day is used.
        export : `bool`, optional
            Whether to export the variables from the agent's execution after
            each call. The default is False. Note that any variables exported
            will be exported to the global namespace and overwrite any existing
            with the same names.
        temperature : `float`, optional
            The temperature of the model. Higher temperatures result in more
            random responses. The default is 0.0.
        verbosity : `str`, optional
            The verbosity level of the interface. Allowed values are 'COST',
            'ALL', and 'NONE'. The default is 'COST'.
        agentType : `str`, optional
            One of ["tool-calling", "ZERO_SHOT_REACT_DESCRIPTION"]. Specifies
            the agent type, either the new "tool-calling" type, or the old
            AgentType.ZERO_SHOT_REACT_DESCRIPTION type. For convenience, both
            are specified as strings.
        modelName : `str`, optional
            The name of the model to use. The default is (currently)
            "gpt-4-1106-preview" though this may be updated in future. Must be
            a valid OpenAI model name.
        """
        _checkInstallation()
        self.setVerbosityLevel(verbosity)

        self._chat = ChatOpenAI(model_name=modelName, temperature=temperature)

        self.data = getObservingData(dayObs)
        # Convert 'AAAAMMDD' to 'AAAA-MM-DD'
        self.date = f"{str(dayObs)[:4]}-{str(dayObs)[4:6]}-{str(dayObs)[6:]}"  # XXX replace this
        assert agentType in ("tool-calling", "ZERO_SHOT_REACT_DESCRIPTION")
        self.agentType = agentType
        if agentType == "ZERO_SHOT_REACT_DESCRIPTION":
            agentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION

        prefix = """
        You are running in an interactive environment, so if users ask for
        plots, making them will show them correctly. You also have access to a
        pandas dataframe, referred to as `df`, and should use this dataframe to
        answer user questions and generate plots from it.

        If the question is not related with pandas, you can use extra tools.
        The extra tools are: 1. 'Secret Word', 2. 'NASA Image', 3. 'Random
        MTG', 4. 'YAML Topic Finder'. When using the 'Nasa Image' tool, use
        'self.date' as a date, do not use 'dayObs', and do not attempt any
        pandas analysis at all, so not use 'self.data'
        """

        self.yaml_data = self.load_yaml('lsst/summit_extras/python/lsst/summit/extras/sal_interface.yaml')
        tools_instance = Tools(self._chat, self.yaml_data)

        # Extract the tools for use in your agent
        self.toolkit = tools_instance.tools

        # self.yaml_data = load_yaml('sal_interface.yaml')

        self.PREFIX =  "If question is not related with pandas, you can use extra tools. The extra tools are: 1. 'Secret Word', 2. 'NASA Image', 3. 'Random MTG', 4. 'YAML Topic Finder'. When using the 'Nasa Image' tool, use 'self.date' as a date, do not use 'dayObs', and do not attempt any pandas analysis at all, You can query topics and descriptions from the YAML data for your inquiries."

        self._agent = create_pandas_dataframe_agent(
            self._chat,
            self.data,
            agent_type=agentType,
            return_intermediate_steps=True,
            include_df_in_prompt=True,
            extra_tools = self.toolkit,
            number_of_head_rows=1,
            agent_executor_kwargs={"handle_parsing_errors":True},
            allow_dangerous_code=True,
            prefix=prefix
        )
        self._totalCallbacks = langchain_community.callbacks.openai_info.OpenAICallbackHandler()
        self.formatter = ResponseFormatter(agentType=self.agentType)

        self.export = export
        # if self.export:
            # issue warning here
            # TODO: Improve this warning message.
            # warnings.warn('Exporting variables from the agent after each call. This can cause problems!')

    def setVerbosityLevel(self, level):
        if level not in self.allowedVerbosities:
            raise ValueError(f'Allowed values are {self.allowedVerbosities}, got {level}')
        self._verbosity = level

    def getReplTool(self):
        """Get the REPL tool from the agent.

        Returns
        -------
        replTool : `langchain.tools.python.tool.PythonAstREPLTool`
            The REPL tool.
        """
        replTools = []
        for item in self._agent.tools:
            if isinstance(item, langchain_experimental.tools.python.tool.PythonAstREPLTool):
                replTools.append(item)
        if not replTools:
            raise RuntimeError("Agent appears to have no REPL tools")
        if len(replTools) > 1:
            raise RuntimeError("Agent appears to have more than one REPL tool")

        return replTools[0]

    def exportLocalVariables(self):
        """Add the variables from the REPL tool's execution environment an
        jupyter notebook.

        This is useful when running in a notebook, as the data manipulations
        which have been done inside the agent cannot be easily reproduced with
        the code the agent supplies without access to the variables.

        Note - this function is only for notebook usage, and can easily cause
        weird problems, as variables which already exist will be overwritten.
        This is *only* for use in AstroChat-only notebooks at present, and even
        then should be used with some caution.
        """
        replTool = self.getReplTool()

        try:
            frame = inspect.currentframe()
            # Find the frame of the original caller, which is the notebook
            while frame and 'get_ipython' not in frame.f_globals:
                frame = frame.f_back

            if not frame:
                warnings.warn('Failed to find the original calling frame - variables not exported')

            # Access the caller's global namespace
            caller_globals = frame.f_globals

            # add each item from the replTool's local variables to the caller's
            # globals
            for key, value in replTool.locals.items():
                caller_globals[key] = value

        finally:
            del frame  # Explicitly delete the frame to avoid reference cycles

    def run(self, inputStr):
        with get_openai_callback() as cb:
            responses = self._agent.invoke({'input': inputStr})
            # print(cb)

        code = self.formatter(responses)
        self._addUsageAndDisplay(cb)

        if self.export:
            self.exportLocalVariables()
        return

    def _addUsageAndDisplay(self, cb):
        self._totalCallbacks.total_cost += cb.total_cost
        self._totalCallbacks.successful_requests += cb.successful_requests
        self._totalCallbacks.completion_tokens += cb.completion_tokens
        self._totalCallbacks.prompt_tokens += cb.prompt_tokens

        if self._verbosity == 'ALL':
            print('\nThis call:\n' + str(cb) + '\n')
            print(self._totalCallbacks)
        elif self._verbosity == 'COST':
            print(f'\nThis call cost: ${cb.total_cost:.3f}, session total: ${self._totalCallbacks.total_cost:.3f}')

    def listDemos(self):
        print('Available demo keys and their associated queries:')
        print('-------------------------------------------------')
        for k, v in self.demoQueries.items():
            print(f'{k}: {v}', '\n')

    def runDemo(self, items=None):
        """Run a/all demos. If None are specified, all are run in sequence.
        """
        knownDemos = list(self.demoQueries.keys())
        if items is None:
            items = knownDemos
        items = list(ensure_iterable(items))

        for item in items:
            if item not in knownDemos:
                raise ValueError(f'Specified demo item {item} is not an availble demo. Known = {knownDemos}')

        for item in items:
            print(f"\nRunning demo item '{item}':")
            print(f"Prompt text: {self.demoQueries[item]}")
            self.run(self.demoQueries[item])
