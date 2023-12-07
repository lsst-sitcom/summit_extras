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
from IPython.display import display, Markdown

INSTALL_NEEDED = False
LOG = logging.getLogger(__name__)

# TODO: work out a way to protect all of these imports
import langchain  # noqa: E402
from langchain.agents import create_pandas_dataframe_agent  # noqa: E402
from langchain.chat_models import ChatOpenAI  # noqa: E402
from langchain.callbacks import get_openai_callback  # noqa: E402
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType
from langchain.agents import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents.agent import AgentExecutor

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
    if currentKey:
        LOG.warning(f"OPENAI_API_KEY is already set. Overwriting with key from {filename}")

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
    elif site in ['staff-rsp', 'rubin-devl']:
        LOG.warning(f"Observing metadata at {site} is currently copied by hand by Merlin and will not be "
                    "updated in realtime")
        filename = f"/sdf/home/m/mfl/u/rubinTvDataProducts/sidecar_metadata/dayObs_{dayObs}.json"
    else:
        raise RuntimeError(f"Observing metadata not available for site {site}")

    # check the file exists, and raise if not
    if not os.path.exists(filename):
        LOG.warning(f"Observing metadata file for {'current' if isCurrent else ''} dayObs "
                    f"{dayObs} does not exist.")
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

    def __init__(self):
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
#         steps = response["intermediate_steps"]
#         nSteps = len(steps)
#         if nSteps > 1:
#             print(f'There were {len(steps)} steps to the process:\n')
#         for stepNum, step in enumerate(steps):
#             if nSteps > 1:
#                 print(f'Step {stepNum + 1}:')
#             action, observation = step  # unpack the tuple
#             thought = self.getThoughtFromAction(action)

#             print(thought)
#             self.printAction(action)
#             self.printObservation(observation)

        output = response["output"]
        print(f'\nFinal answer: {output}')

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
        self.printResponse(response)
        allCode = self.allCode
        self.allCode = []
        return allCode


class AstroChat:
    allowedVerbosities = ('COST', 'ALL', 'NONE', None)

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
        'seeingVsTime': 'The PSF FWHM is an important performance parameter. Restricting the analysis to images with filter name that includes SDSS, can you please make a scatter plot of FWHM vs. time for all such images, with a legend. I want all data points on a single graph.'
    }

    memory = ConversationBufferMemory(memory_key="chat_history")

    PREFIX = """
            You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
            You should use the tools below to answer the question posed of you:"""

    SUFFIX = """
            This is the result of `print(df.head())`:
            {df}
            Begin!
            {chat_history}
            Question: {input}
            {agent_scratchpad}"""

    def create_pandas_dataframe_agent_test(
        self,
        llm,
        df,
        verbose: bool = False,
        max_iterations: Optional[int] = 15,
        max_execution_time: Optional[float] = None,
        early_stopping_method: str = "force",
        return_intermediate_steps: bool = False,
        callback_manager: Optional[BaseCallbackManager] = None,
        input_variables: Optional[List[str]] = None,  # Moved before default arguments
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
    ) -> AgentExecutor:
        """Construct a pandas agent from an LLM and dataframe."""

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected pandas object, got {type(df)}")
        if input_variables is None:
            input_variables = ["df", "input", "agent_scratchpad", "intermediate_steps"]
        tools = [PythonAstREPLTool(locals={"df": df})]

        PREFIX = """
            You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
            You should use the tools below to answer the question posed of you:"""

        SUFFIX = """
                This is the result of `print(df.head())`:
                {df}
                Begin!
                {chat_history}
                Question: {input}
                {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=PREFIX,
            suffix=SUFFIX,
            input_variables=["df", "input", "chat_history", "agent_scratchpad"]
        )

        partial_prompt = prompt.partial(df=str(df.head()))

        llm_chain = LLMChain(
            llm=llm,
            prompt=partial_prompt,
            callback_manager=callback_manager,
        )

        tool_names = [tool.name for tool in tools]

        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            callback_manager=callback_manager,
        )

        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=verbose,
            return_intermediate_steps=return_intermediate_steps,
            max_iterations=max_iterations,
            max_execution_time=max_execution_time,
            early_stopping_method=early_stopping_method,
            callback_manager=callback_manager,
            memory = self.memory
        )

    def __init__(self,
                 dayObs=None,
                 export=False,
                 temperature=0.0,
                 verbosity='COST'):
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
        """
        _checkInstallation()
        self.setVerbosityLevel(verbosity)

        self._chat = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=temperature)

        self.data = getObservingData(dayObs)

        self._agent = self.create_pandas_dataframe_agent_test(
            self._chat,
            self.data,
            return_intermediate_steps=False,
            # include_df_in_prompt=True,
            # number_of_head_rows=1,
        )
        self._totalCallbacks = langchain.callbacks.openai_info.OpenAICallbackHandler()
        self.formatter = ResponseFormatter()

        self.export = export
        if self.export:
            # issue warning here
            # TODO: Improve this warning message.
            warnings.warn('Exporting variables from the agent after each call. This can cause problems!')


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
            if isinstance(item, langchain.tools.python.tool.PythonAstREPLTool):
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
            responses = self._agent({'input': inputStr})

        code = self.formatter(responses)
        self._addUsageAndDisplay(cb)

        if self.export:
            self.exportLocalVariables()
        return code

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
