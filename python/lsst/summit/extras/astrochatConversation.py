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
import re
import warnings
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
from io import StringIO
from typing import Dict, Any, Optional, TypedDict, List
from openai import OpenAI
from langgraph.graph import StateGraph, END
from lsst.summit.utils.utils import getCurrentDayObs_int, getSite
from lsst.utils import getPackageDir
from lsst.utils.iteration import ensure_iterable

INSTALL_NEEDED = False
LOG = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    INSTALL_NEEDED = True
    LOG.warning("openai package not found. Please install openai: pip install openai")

def _checkInstallation():
    """Check that all required packages are installed."""
    if INSTALL_NEEDED:
        raise RuntimeError("openai package not found. Please install openai: pip install openai")

def setApiKey(filename="~/.openaikey.txt"):
    """Set the OpenAI API key from a file."""
    _checkInstallation()
    currentKey = os.getenv("OPENAI_API_KEY")
    if currentKey:
        LOG.warning(f"OPENAI_API_KEY is already set. Overwriting with key from {filename}")
    filename = os.path.expanduser(filename)
    with open(filename, "r") as file:
        apiKey = file.readline().strip()
    openai.api_key = apiKey
    os.environ["OPENAI_API_KEY"] = apiKey

def getObservingData(dayObs=None):
    """Get the observing metadata for the specified day."""
    currentDayObs = getCurrentDayObs_int()
    if dayObs is None:
        dayObs = currentDayObs
    isCurrent = dayObs == currentDayObs
    site = getSite()
    filename = None
    if site == "summit":
        filename = f"/project/rubintv/sidecar_metadata/dayObs_{dayObs}.json"
    elif site in ["rubin-devl"]:
        LOG.warning(f"Observing metadata at {site} is currently copied by hand by Merlin and will not be updated in realtime")
        filename = f"/sdf/home/m/mfl/u/rubinTvDataProducts/sidecar_metadata/dayObs_{dayObs}.json"
    elif site in ["staff-rsp"]:
        LOG.warning(f"Observing metadata at {site} is currently copied by hand by Merlin and will not be updated in realtime")
        filename = f"/home/m/mfl/u/rubinTvDataProducts/sidecar_metadata/dayObs_{dayObs}.json"
    else:
        raise RuntimeError(f"Observing metadata not available for site {site}")
    if not os.path.exists(filename):
        LOG.warning(f"Observing metadata file for {'current' if isCurrent else ''} dayObs {dayObs} does not exist at {filename}.")
        return pd.DataFrame()
    table = pd.read_json(filename).T
    table = table.sort_index()
    table = table.drop([col for col in table.columns if col.startswith("_")], axis=1)
    return table

# Define state schema using TypedDict
class AgentState(TypedDict):
    question: str
    chain_of_thought: List[str]
    code: str
    local_vars: Dict[str, Any]
    output: Optional[Any]
    error: Optional[str]
    previous_response: Optional[Any]
    client: Any  # OpenAI client
    next_agent: Optional[str]

# OpenAI LLM response function with temperature
def openai_response(client, prompt: str, temperature: float = 0.0, previous_code: Optional[str] = None, error: Optional[str] = None, context: Optional[str] = None) -> str:
    full_prompt = (
        "You are a Python coding assistant. Generate concise Python code to answer the request. "
        "Use the provided DataFrame 'df' and its columns. "
        "Return only the necessary code without explanations or print statements. "
        "The code should evaluate to the answer or store it in a 'result' variable. "
        "If a required column is not available, set 'result' to an explanatory string.\n\n"
        f"Request: {prompt}\n"
    )
    if previous_code:
        full_prompt += f"Previous code:\n```python\n{previous_code}\n```\n"
    if error:
        full_prompt += f"Error encountered:\n{error}\n"
    if context:
        full_prompt += f"Context:\n{context}\n"
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=200,
        temperature=temperature
    )
    code = response.choices[0].message.content.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code

# Execute code, capturing both printed output and errors
def execute_code(code: str, local_vars: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    try:
        exec(code, local_vars)
        sys.stdout = old_stdout
        output = redirected_output.getvalue().strip()
        return output if output else local_vars.get("result"), None
    except Exception as e:
        sys.stdout = old_stdout
        return None, traceback.format_exc()

# Human-in-the-loop
def human_review(step_description: str, data: str) -> str:
    print(f"\nHuman Review Required: {step_description}")
    print(f"Current Data:\n{data}")
    response = input("Approve (y), Modify (type new code/answer), or Reject (n): ").strip()
    if response.lower() == 'y':
        return data
    elif response.lower() == 'n':
        return None
    else:
        return response
    
def select_agent(client, question: str, temperature: float = 0.0) -> str:
    prompt = (
        "Given the following question, determine which agent should handle it:\n"
        "- 'pandas_agent' for data analysis, calculations, observation data questions, "
        "questions about image counts, or questions involving specific dates.\n"
        "- 'database_agent' for explicit database queries.\n"
        "- 'query_agent' for follow-up questions or generic queries.\n"
        "Return only the agent name (e.g., 'pandas_agent') without additional text.\n\n"
        f"Question: {question}"
    )
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=temperature
    )
    agent = response.choices[0].message.content.strip()
    # Validate the response to ensure it's a valid agent
    valid_agents = {"pandas_agent", "database_agent", "query_agent"}
    return agent if agent in valid_agents else "query_agent"

class AstroChat:
    allowedVerbosities = ("COST", "ALL", "NONE", None)

    def __init__(
        self,
        dayObs=None,
        export=False,
        temperature=0.0,
        verbosity="COST",
        modelName="gpt-4-1106-preview",
    ):
        """Create an ASTROCHAT bot."""
        _checkInstallation()
        self.setVerbosityLevel(verbosity)
        self.dayObs = dayObs
        self.export = export
        self.temperature = temperature
        self.client = OpenAI()
        self.data = getObservingData(dayObs)
        self.date = f"{str(dayObs)[:4]}-{str(dayObs)[4:6]}-{str(dayObs)[6:]}"
        self.workflow = self._build_workflow()
        self.state = None

    def _build_workflow(self):
        graph = StateGraph(AgentState)
        graph.add_node("orchestrate_agent", self.orchestrate_agent)
        graph.add_node("pandas_agent", self.pandas_dataframe_agent)
        graph.add_node("database_agent", self.database_agent)
        graph.add_node("query_agent", self.query_generator_agent)
        
        graph.add_conditional_edges(
            "orchestrate_agent",
            lambda state: state["next_agent"],
            {"pandas_agent": "pandas_agent", "query_agent": "query_agent", "database_agent": "database_agent"}
        )
        graph.add_conditional_edges("query_agent", lambda state: state["next_agent"], {END: END})
        graph.add_conditional_edges("pandas_agent", lambda state: state["next_agent"], {END: END})
        graph.add_conditional_edges("database_agent", lambda state: state["next_agent"], {END: END})
        
        graph.set_entry_point("orchestrate_agent")
        return graph.compile()

    def orchestrate_agent(self, state: AgentState) -> Dict[str, Any]:
        question = state["question"]
        chain_of_thought = state.get("chain_of_thought", [])
        chain_of_thought.append(f"Orchestrate Agent: Analyzing question - '{question}'")
        
        # Use AI to determine the appropriate agent
        next_agent = select_agent(state["client"], question, self.temperature)
        chain_of_thought.append(f"AI routed to: {next_agent}")
        
        # Fallback mechanism for specific keywords
        if next_agent != "pandas_agent" and any(keyword in question.lower() for keyword in ["images", "taken", "how many", str(self.dayObs)]):
            chain_of_thought.append("Fallback: Overriding AI decision to route to Pandas DataFrame Agent for observation data or count operations.")
            next_agent = "pandas_agent"
        
        return {"next_agent": next_agent, "chain_of_thought": chain_of_thought}
    # Query Generator Agent (HITL for follow-ups)
    def query_generator_agent(self, state: AgentState) -> Dict[str, Any]:
        chain_of_thought = state.get("chain_of_thought", [])
        chain_of_thought.append("Query Generator Agent: Processing request.")
        if any(keyword in state["question"].lower() for keyword in ["last", "previous", "explain", "why"]):
            chain_of_thought.append("Detected follow-up; generating response with previous context.")
            response = openai_response(state["client"], state["question"], self.temperature, context=str(state.get("previous_response")))
            reviewed_response = human_review("Response to follow-up question", response) or response
            chain_of_thought.append(f"Response after review: {reviewed_response}")
            output = reviewed_response
        else:
            chain_of_thought.append("Generic request; generating basic response.")
            output = "No specific action identified."
        next_agent = END
        return {
            "chain_of_thought": chain_of_thought,
            "output": output,
            "previous_response": output,
            "next_agent": next_agent
        }

    # Pandas DataFrame Agent (HITL for code)
    def pandas_dataframe_agent(self, state: AgentState) -> Dict[str, Any]:
        chain_of_thought = state.get("chain_of_thought", [])
        local_vars = state.get("local_vars", {})
        chain_of_thought.append("Pandas DataFrame Agent: Initializing.")
        
        # Initialize df with the observing data
        local_vars["df"] = self.data
        
        chain_of_thought.append("Generating initial code with OpenAI LLM.")
        
        # Provide context about the DataFrame structure
        context = (
            "You have access to a DataFrame 'df' containing observing data. "
            "The available columns are: " + ", ".join(self.data.columns) + ". "
            "Use this DataFrame to answer the question. If a required column is not available, "
            "explain that in the result."
        )
        
        code = openai_response(state["client"], f"{context}\n\nQuestion: {state['question']}", self.temperature)
        code = human_review("Initial code generation", code) or code
        chain_of_thought.append(f"Initial code after review:\n{code}")

        max_iterations = 5
        output = None
        for iteration in range(max_iterations):
            if not code:
                chain_of_thought.append("Code rejected by human; stopping.")
                output = None
                break
            
            output, error = execute_code(code, local_vars)
            print(f"Debug: Iteration {iteration + 1} - Output: {output}, Error: {error}")  # Debugging
            if error:
                chain_of_thought.append(f"Iteration {iteration + 1}: Failed with error:\n{error}")
                new_code = openai_response(state["client"], f"{context}\n\nError: {error}\n\nQuestion: {state['question']}", self.temperature, code, error)
                code = human_review(f"Code refinement after error (Iteration {iteration + 1})", new_code) or new_code
                chain_of_thought.append(f"Iteration {iteration + 1}: Refined code after review:\n{code}")
            else:
                chain_of_thought.append(f"Iteration {iteration + 1}: Success! Output: {output}")
                break
        else:
            chain_of_thought.append("Max iterations reached without success.")
        
        next_agent = END
        return {
            "chain_of_thought": chain_of_thought,
            "code": code,
            "local_vars": local_vars,
            "output": output,
            "previous_response": output,
            "next_agent": next_agent
        }
    
    # Database Agent (placeholder)
    def database_agent(self, state: AgentState) -> Dict[str, Any]:
        chain_of_thought = state.get("chain_of_thought", [])
        chain_of_thought.append("Database Agent: No database specified.")
        output = "Database operation not implemented."
        next_agent = END
        return {
            "chain_of_thought": chain_of_thought,
            "output": output,
            "previous_response": output,
            "next_agent": next_agent
        }

    def run_single(self, question: str) -> Any:
        """Run a single question through the workflow and return the output."""
        initial_state = {
            "question": question,
            "chain_of_thought": [],
            "code": "",
            "local_vars": self.state["local_vars"] if self.state and "local_vars" in self.state else {},
            "output": None,
            "error": None,
            "previous_response": self.state["output"] if self.state else None,  # Preserve previous response
            "client": self.client,
            "next_agent": None
        }
        self.state = self.workflow.invoke(initial_state)
        
        if self.verbosity != 'NONE':
            print("\nChain of Thought:")
            for step in self.state["chain_of_thought"]:
                print(step)
            print("\nFinal Code:")
            print(self.state["code"])
            print(f"Final Output: {self.state['output']}")
        
        if self.export:
            with open(f"astrochat_output_{self.dayObs}.txt", "a") as f:  # Append mode to log all interactions
                f.write("Chain of Thought:\n")
                for step in self.state["chain_of_thought"]:
                    f.write(f"{step}\n")
                f.write("\nFinal Code:\n")
                f.write(f"{self.state['code']}\n")
                f.write(f"Final Output: {self.state['output']}\n\n")
        
        return self.state["output"]

    def run_interactive(self):
        """Run an interactive session allowing follow-up questions."""
        print("Welcome to AstroChat! Ask a question or type 'exit' to quit.")
        while True:
            question = input("Your question: ").strip()
            if question.lower() == "exit":
                print("Goodbye!")
                break
            if not question:
                print("Please enter a valid question.")
                continue
            
            output = self.run_single(question)
            print(f"\nAnswer: {output}\n")
            print("Feel free to ask a follow-up question or type 'exit' to quit.")

    def run(self, question: str = None):
        """Main entry point: run interactively if no question provided, otherwise run a single question."""
        if question is None:
            self.run_interactive()
        else:
            return self.run_single(question)
    def setVerbosityLevel(self, level):
        if level not in self.allowedVerbosities:
            raise ValueError(f"Allowed values are {self.allowedVerbosities}, got {level}")
        self.verbosity = level
