# This file is part of summit_extras.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
DSPy Project with Specialized Workers.

This file illustrates:
- The basic configuration of a DSPy project.
- Various Workers (e.g., Python code executor) orchestrated by a Boss.
- Capabilities for conversation memory and Python code execution.

Notes
-----
Docstrings are in English for clarity. Following LSST conventions,
comment lines should not exceed 79 characters, and code lines should not
exceed 110 characters. Adjust paths, credentials, and endpoints according
to your environment.
"""

__all__ = [
    "UserInteraction",
    "ConversationContext",
    "ResponseWithContext",
    "respond_cot",
    "ProgramOfThought",
    "Tool",
    "python_tool",
    "dummy_tool",
    "Worker",
    "python_worker",
    "Boss",
]

import logging  # Used for logging error and debug messages
import os       # Used to load environment variables and file paths
import re       # Used for string manipulation
import yaml     # Used to load configuration files

from typing import List, Optional, Callable  # For type hints
from pathlib import Path  # Used in credential loading
from pydantic import BaseModel  # Used to define data models

import dspy  # Used to define workflows and data models
from dspy import (
    Signature,
    ChainOfThought,
    Module,
    settings
)
from dspy.primitives.python_interpreter import CodePrompt, PythonInterpreter  # For Python code execution

###############################################################################
# LOGGING CONFIGURATION AND LANGUAGE MODEL SETUP
###############################################################################
# Logging is fundamental for debugging and monitoring applications.
# Here we set the basic logging level to INFO so that all info,
# warning, and error messages will be recorded.

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

###############################################################################
# CONFIGURATION MANAGER AND PATHS
###############################################################################
class ConfigManager:
    """Configuration Manager implementing the Singleton pattern.
    
    This class demonstrates:
    1. Use of Optional[] types.
    2. Handling configuration from multiple sources (env vars, files).
    3. Fallbacks if one source fails.
    
    Parameters
    ----------
    config_path : `str`, optional
        Path to the configuration file.
    """
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or '/home/c/carlosm/lsst/summit_extras/python/lsst/summit/extras/data/config.yaml'
        
    def load_credentials(self, key_name: str) -> str:
        """Load credentials using multiple fallback methods.
        
        Parameters
        ----------
        key_name : `str`
            The name of the credential key.
        
        Returns
        -------
        value : `str`
            The credential value.
        """
        # Try environment variables first
        if value := os.getenv(key_name):
            return value
            
        # Try configuration file
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                if value := config.get(key_name):
                    return value
                    
        raise ValueError(f"Could not find credentials for {key_name}")
    
# Initialize configuration manager
config = ConfigManager()

# Load credentials (only the OpenAI API key is required now)
try:
    api_key = config.load_credentials('OPENAI_API_KEY')
    api_base = config.load_credentials('OLLAMA_SERVER')
except ValueError as e:
    LOG.error(f"Failed to load credentials: {e}")
    raise

###############################################################################
# LANGUAGE MODEL CONFIGURATION
###############################################################################
lm = dspy.LM(
    'ollama_chat/tulu3:70b',
    api_base=api_base,
    api_key=''  # API key is empty as per configuration
)

settings.configure(lm=lm)

###############################################################################
# USER INTERACTION AND CONVERSATION MEMORY CLASSES
###############################################################################
class UserInteraction:
    """Represents a user-assistant interaction.
    
    Demonstrates:
    1. Well-documented classes following the NumPy style.
    2. Documentation of parameters and types.
    3. Encapsulation of related data.
    
    Parameters
    ----------
    message : `str`
        The user's message.
    response : `str`
        The assistant's response.
    """
    def __init__(self, message: str, response: str):
        self.message = message
        self.response = response

    def serialize_by_role(self) -> List[dict]:
        """Return the interaction formatted for prompts.
        
        Returns
        -------
        serialized : `list` of `dict`
            Roles and content for the user and assistant.
        """
        return [
            {"role": "user", "content": self.message},
            {"role": "assistant", "content": self.response}
        ]


class ConversationContext:
    """Implements a circular buffer to maintain conversation context.
    
    Demonstrates:
    1. Use of data structures (list as a buffer).
    2. Handling of internal state.
    3. Methods to update and render the conversation.
    
    Parameters
    ----------
    window_size : `int`, optional
        Number of interactions to retain (default is 5).
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.content: List[UserInteraction] = []

    def update(self, interaction: UserInteraction) -> None:
        """Adds a new interaction and trims the buffer to `window_size`.
        
        Parameters
        ----------
        interaction : `UserInteraction`
            The new user-assistant interaction.
        """
        self.content.append(interaction)
        self.content = self.content[-self.window_size:]

    def render(self) -> str:
        """Returns a string with recent interactions.
        
        Returns
        -------
        context_str : `str`
            Text representing the conversation history.
        """
        out = []
        for c in self.content:
            out.append(
                f"User: {c.message}\n\nAssistant: {c.response}\n\n"
            )
        return "".join(out)

###############################################################################
# SIGNATURE AND RESPONSE WITH CONTEXT
###############################################################################
class ResponseWithContext(Signature):
    """Defines a signature for responses that include conversation context.
    
    Attributes
    ----------
    context : `str`
        The conversation context.
    message : `str`
        The user's message.
    response : `str`
        The contextualized response produced by the assistant.
    """
    context = dspy.InputField(desc="Conversation context")
    message = dspy.InputField(desc="User message")
    response = dspy.OutputField(desc="Contextualized response")

respond_cot = ChainOfThought(ResponseWithContext)

###############################################################################
# PROGRAM OF THOUGHT FOR PYTHON CODE EXECUTION
###############################################################################
class ProgramOfThought(Module):
    """Advanced example for safe Python code execution.
    
    Demonstrates:
    1. Use of decorators and inheritance.
    2. Safe code execution handling.
    3. Retry logic implementation.
    4. Use of whitelists for security.
    
    Parameters
    ----------
    signature : `dspy.Signature`
        Defines the expected input/output.
    max_iters : `int`, optional
        Maximum number of attempts for code generation/execution.
    """
    def __init__(self, signature: Signature, max_iters: int = 3):
        super().__init__()
        self.signature = signature
        self.max_iters = max_iters
        self.interpreter = PythonInterpreter(
            action_space={"print": print},
            import_white_list=["numpy", "astropy", "sympy", "matplotlib", "matplotlib.pyplot"]
        )

    def forward(self, question: str, context_str: str = "", variables: dict = None) -> dict:
        """Generates and executes Python code for a given question.
        
        Parameters
        ----------
        question : `str`
            The question or instruction to resolve.
        context_str : `str`, optional
            Conversation context.
        variables : `dict`, optional
            Additional variables.
        
        Returns
        -------
        result : `dict`
            A dictionary with the key "answer" containing the result of the execution
            or None if it fails.
        """
        if variables is None:
            variables = {}

        for i in range(self.max_iters):
            generated_code = self._generate_code(question, context_str)
            print(f"\nIteration {i + 1}: Generated code:\n{generated_code}")

            try:
                result, _ = CodePrompt(generated_code).execute(
                    self.interpreter,
                    user_variable=variables
                )
                print("Execution result:", result)
                return {"answer": result}
            except Exception as exc:
                LOG.error("Execution error: %s", exc)

        LOG.error("Failed to generate valid code after several attempts.")
        return {"answer": None}

    def _generate_code(self, question: str, context_str: str) -> str:
        """Generates Python code using the language model.
        
        Parameters
        ----------
        question : `str`
            The user's question.
        context_str : `str`
            The conversation context.
        
        Returns
        -------
        code : `str`
            Clean Python code ready for execution.
        """
        code_signature = dspy.Signature("question -> generated_code")
        predict = dspy.Predict(code_signature)

        prompt = (
            "You are an AI that writes Python code to solve problems.\n"
            "Context:\n"
            f"{context_str}\n\n"
            f"Question: {question}\n\n"
            "Code:\n"
        )
        completion = predict(question=prompt)
        code_block = re.sub(r"```[^\n]*\n", "", completion.generated_code)
        code_block = re.sub(r"```", "", code_block)
        return code_block

###############################################################################
# TOOL DEFINITION
###############################################################################
class Tool(BaseModel):
    """Implementation of the Strategy pattern using Pydantic.
    
    Demonstrates:
    1. Use of Pydantic for data validation.
    2. Static type hinting in Python.
    3. Use of Callables as types.
    
    Parameters
    ----------
    name : `str`
        The name of the tool.
    description : `str`
        A brief description of its functionality.
    requires : `str`
        The required parameter(s) for the tool.
    func : `Callable`
        The function that implements the tool's logic.
    """
    name: str
    description: str
    requires: str
    func: Callable

code_tool_signature = dspy.Signature("question -> answer")
code_program = ProgramOfThought(signature=code_tool_signature, max_iters=3)

python_tool = Tool(
    name="python executor",
    description=("Generates and executes Python code with context using "
                 "ProgramOfThought."),
    requires="question",
    func=lambda question_dict: code_program(
        question=question_dict.get("question", ""),
        context_str=question_dict.get("context", ""),
        variables=question_dict.get("variables", {})
    )
)

dummy_tool = Tool(
    name="dummy",
    description="Example tool for future extensions.",
    requires="some_argument",
    func=lambda arg: f"Dummy tool called with argument: {arg}"
)

###############################################################################
# GENERIC WORKER: PYTHON
###############################################################################
class Worker(Module):
    """Generic Worker that plans and executes tasks.
    
    Demonstrates:
    1. Object-oriented design.
    2. Separation of responsibilities.
    3. Task planning and execution.
    
    Parameters
    ----------
    role : `str`
        The role or identifier of the worker (e.g., "python_worker").
    tools : `List[Tool]`
        A list of available tools.
    """
    def __init__(self, role: str, tools: List[Tool]):
        self.role = role
        self.tools = {t.name: t for t in tools}
        self._plan = ChainOfThought("task, context -> proposed_plan")
        self._use_tool = ChainOfThought("task, context -> tool_name, tool_argument")

    def plan(self, task: str, feedback: str = "") -> str:
        """Generates a simplified textual plan.
        
        Parameters
        ----------
        task : `str`
            The description or goal of the task.
        feedback : `str`, optional
            Additional comments.
        
        Returns
        -------
        plan_text : `str`
            The proposed plan.
        """
        context = (
            f"Worker role: {self.role}; Tools: {list(self.tools.keys())}; "
            f"Feedback: {feedback}"
        )
        plan_result = self._plan(task=task, context=context)
        return plan_result.proposed_plan

    def execute(self, task: str, use_tool: bool, context: str = "") -> str:
        """Executes the task, optionally using a tool.
        
        Parameters
        ----------
        task : `str`
            The task or instruction.
        use_tool : `bool`
            Indicates whether a tool should be used.
        context : `str`, optional
            Additional text (e.g., plan information).
        
        Returns
        -------
        outcome : `str`
            The result of execution or an error message.
        """
        print(f"[{self.role}] Executing: {task}")
        if not use_tool:
            return f"Task '{task}' completed without using any tool."

        tool_decision = self._use_tool(task=task, context=context)
        tool_name = tool_decision.tool_name
        # Map "python" to "python executor" if necessary
        if tool_name == "python":
            tool_name = "python executor"
            
        arg = tool_decision.tool_argument
        # Ensure arg is a dictionary
        if not isinstance(arg, dict):
            arg = {"question": arg, "context": context, "variables": {}}

        if tool_name in self.tools:
            return self.tools[tool_name].func(arg)
        return f"Tool '{tool_name}' not found."

python_worker = Worker("python_worker", tools=[python_tool, dummy_tool])

###############################################################################
# BOSS FOR ORCHESTRATION
###############################################################################
class Boss(Module):
    """Mediator/Orchestrator that manages multiple workers.
    
    Demonstrates:
    1. Management of multiple workers.
    2. Decision making based on context.
    3. Advanced design patterns.
    
    Parameters
    ----------
    workers : `list` of `Module`
        List of available workers.
    """
    def __init__(self, workers: List[Module]):
        self.workers = {w.role: w for w in workers}
        self._assign = ChainOfThought("task -> who")
        self._approve = ChainOfThought("task, plan -> approval")

    def plan_and_execute(self, task: str, worker_hint: str = "", use_tool: bool = True) -> str:
        """Selects a worker, generates and approves a plan, and executes the task.
        
        Parameters
        ----------
        task : `str`
            The user's objective.
        worker_hint : `str`, optional
            A suggestion of which worker to use.
        use_tool : `bool`, optional
            Indicates if the worker should use a tool.
        
        Returns
        -------
        exec_result : `str`
            The result of the worker's execution.
        """
        if worker_hint and worker_hint in self.workers:
            assignee = worker_hint
        else:
            assign_result = self._assign(task=task)
            assignee = assign_result.who

        if assignee not in self.workers:
            return f"No worker named '{assignee}' exists."

        plan_text = self.workers[assignee].plan(task)
        print(f"Proposed plan: {plan_text}")

        approval_res = self._approve(task=task, plan=plan_text)
        is_approved = "yes" in approval_res.approval.lower()
        if not is_approved:
            LOG.info("The plan was not approved. It may be refined further.")

        context = f"Plan: {plan_text}; Approved"
        result = self.workers[assignee].execute(task, use_tool=use_tool, context=context)
        return result