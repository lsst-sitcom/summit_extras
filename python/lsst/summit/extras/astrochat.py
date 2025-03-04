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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# [License header and imports remain unchanged]

# [License header and imports unchanged]
import ast
import logging
import os
import re
import sys
import traceback
import warnings
import operator
from io import StringIO
from typing import Dict, Any, List, Optional, Union, Literal, TypedDict, Annotated

import chromadb
import langchain_community
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yaml
from annoy import AnnoyIndex
from chromadb import Client, Settings
from IPython.display import Image, Markdown, display
from langchain.agents import AgentType, Tool, create_react_agent, create_tool_calling_agent
from langchain.schema import HumanMessage
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_experimental.agents.agent_toolkits.pandas.prompt import (
    FUNCTIONS_WITH_DF, PREFIX, PREFIX_FUNCTIONS, SUFFIX_NO_DF, SUFFIX_WITH_DF
)
from langchain_openai import ChatOpenAI
from langchain.agents.agent import AgentExecutor, RunnableAgent, RunnableMultiActionAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

from lsst.summit.utils.utils import getCurrentDayObs_int, getSite
from lsst.utils import getPackageDir
from lsst.utils.iteration import ensure_iterable

installNeeded = False
log = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    installNeeded = True
    log.warning("openai package not found. Please install openai: pip install openai")

def checkInstallation():
    """Check that all required packages are installed."""
    if installNeeded:
        raise RuntimeError("openai package not found. Please install openai: pip install openai")

def setApiKey(fileName="~/.openaikey.txt"):
    """Set the OpenAI API key from a file."""
    checkInstallation()
    currentKey = os.getenv("OPENAI_API_KEY")
    if currentKey:
        log.warning(f"OPENAI_API_KEY is already set. Overwriting with key from {fileName}")
    
    fileName = os.path.expanduser(fileName)
    with open(fileName, "r") as file:
        apiKey = file.readline().strip()
    
    openai.api_key = apiKey
    os.environ["OPENAI_API_KEY"] = apiKey

def getObservingData(dayObs=None):
    """Get observing metadata for the current or a past day."""
    currentDayObs = getCurrentDayObs_int()
    if dayObs is None:
        dayObs = currentDayObs
    isCurrent = dayObs == currentDayObs
    
    site = getSite()
    fileName = None
    if site == "summit":
        fileName = f"/project/rubintv/sidecar_metadata/dayObs_{dayObs}.json"
    elif site in ["rubin-devl"]:
        log.warning(
            f"Observing metadata at {site} is currently copied by hand by Merlin and will "
            "not be updated in realtime"
        )
        fileName = f"/sdf/home/m/mfl/u/rubinTvDataProducts/sidecar_metadata/dayObs_{dayObs}.json"
    elif site in ["staff-rsp"]:
        log.warning(
            f"Observing metadata at {site} is currently copied by hand by Merlin and will "
            "not be updated in realtime"
        )
        fileName = f"/home/m/mfl/u/rubinTvDataProducts/sidecar_metadata/dayObs_{dayObs}.json"
    else:
        raise RuntimeError(f"Observing metadata not available for site {site}")
    
    if not os.path.exists(fileName):
        log.warning(
            f"Observing metadata file for {'current' if isCurrent else ''} dayObs "
            f"{dayObs} does not exist at {fileName}."
        )
        return pd.DataFrame()
    
    table = pd.read_json(fileName).T
    table = table.sort_index()
    table = table.drop([col for col in table.columns if col.startswith("_")], axis=1)
    return table

class ResponseFormatter:
    """Format chatbot responses, ensuring Python code is shown before observation."""
    
    def __init__(self, agentType):
        self.agentType = agentType
    
    def getThoughtFromAction(self, action):
        """Extract thought from action, adapting to agent type."""
        if self.agentType == "tool-calling":
            # For tool-calling, thought might be implicit in tool selection
            if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                return f"Decided to use tool '{action.tool}' with input: {action.tool_input}"
            return "No explicit thought available."
        else:
            # Existing logic for ReAct-style agents
            if not hasattr(action, 'log') or not isinstance(action.log, str):
                return "No thought available."
            lines = action.log.split('\n')
            for line in lines:
                if line.startswith("Thought:"):
                    return line.strip()
            return "No thought available."
    
    def extractCodeFromAction(self, action):
        """Extract Python code from action if applicable."""
        if hasattr(action, 'tool') and action.tool == "PythonRepl":
            code = action.tool_input.strip() if isinstance(action.tool_input, str) else str(action.tool_input)
            return f"```python\n{code}\n```" if code else None
        return None
    
    def printObservation(self, observation):
        """Display observation if it exists."""
        if observation and observation.strip():
            print(f"Observation: {observation}")
    
    def printFinalAnswer(self, response):
        """Print the final answer from the response."""
        finalAnswer = response.get("chatHistory", [None])[-1]
        if finalAnswer and finalAnswer.strip():
            print(f"\nFinal Answer: {finalAnswer}")
        else:
            print("\nNo final answer provided.")
    
    def printResponse(self, response):
        """Format and print response with intermediate steps, code before observation."""
        steps = response.get("intermediateSteps", [])
        print("> Entering new AgentExecutor chain...")
        
        if not steps:
            print("No intermediate steps received.")
            self.printFinalAnswer(response)
            print("> Finished chain.")
            return
        
        seenSteps = set()
        for stepNum, (action, observation) in enumerate(steps, start=1):
            stepKey = (action.tool if hasattr(action, 'tool') else "unknown", 
                       str(action.tool_input) if hasattr(action, 'tool_input') else "no_input")
            if stepKey in seenSteps:
                continue
            seenSteps.add(stepKey)
            
            thought = self.getThoughtFromAction(action)
            tool = action.tool if hasattr(action, 'tool') else "Unknown"
            toolInput = action.tool_input if hasattr(action, 'tool_input') else "No input"
            
            print(f"\nStep {stepNum}:")
            print(f"Thought: {thought}")
            print(f"Tool: {tool}")
            print(f"Tool input: {toolInput}")
            
            codeSnippet = self.extractCodeFromAction(action)
            if codeSnippet:
                print("\nUsed Python code:")
                display(Markdown(codeSnippet))
            
            self.printObservation(observation)
        
        self.printFinalAnswer(response)
        print("> Finished chain.")
    
    def __call__(self, response):
        """Format response for notebook display."""
        if self.agentType in ["tool-calling", "ZERO_SHOT_REACT_DESCRIPTION"]:
            if isinstance(response, list):
                for resp in response:
                    self.printResponse(resp)
            else:
                self.printResponse(response)
        else:
            raise ValueError(f"Unknown agent type: {self.agentType}")
class CodeSafetyClassifier:
    """AI-based malicious code classifier using a language model."""
    def __init__(self, chatModel: ChatOpenAI):
        self.chatModel = chatModel
    
    def analyzeCode(self, code: str) -> tuple[bool, Optional[str]]:
        """Analyze code for malicious intent using an LLM."""
        prompt = (
            "You are a security expert. Analyze the following Python code and determine if it is potentially malicious. "
            "Malicious code may attempt to delete files, access the system, execute arbitrary commands, or cause harm. "
            "Return your response in the format: [is_malicious: true/false, reason: explanation].\n\n"
            f"Code:\n```python\n{code}\n```"
        )
        response = self.chatModel.invoke([HumanMessage(content=prompt)])
        responseText = response.content.strip()
        
        try:
            # Parse response assuming it's in the expected format
            isMalicious = "is_malicious: true" in responseText.lower()
            reasonStart = responseText.find("reason:") + 7
            reason = responseText[reasonStart:].strip() if reasonStart > 6 else "No reason provided"
            return isMalicious, reason
        except Exception as e:
            log.error(f"Failed to parse classifier response: {e}")
            return True, "Failed to analyze code safely"
class CustomPythonRepl:
    def __init__(self, locals: Optional[Dict] = None, safetyClassifier: Optional[CodeSafetyClassifier] = None):
        self.locals = locals or {}
        self.locals['_lastExecutedCode'] = ''
        self.locals['_lastResult'] = ''
        self.safetyClassifier = safetyClassifier or CodeSafetyClassifier(ChatOpenAI(model_name="gpt-4o", temperature=0.0))
    
    def checkMaliciousCode(self, command: str) -> tuple[bool, Optional[str]]:
        """Check if the code is malicious using the AI classifier."""
        command = command.strip().strip('`')
        if command.startswith('python'):
            command = command[len('python'):].strip()
        return self.safetyClassifier.analyzeCode(command)
    
    def run(self, command: str) -> str:
        """Execute Python code after safety check."""
        command = command.strip().strip('`')
        if command.startswith('python'):
            command = command[len('python'):].strip()
        
        isMalicious, reason = self.checkMaliciousCode(command)
        if isMalicious:
            raise ValueError(f"Potentially malicious code detected: {reason}")
        
        oldStdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            self.locals['_lastExecutedCode'] = command
            tree = ast.parse(command)
            exec(compile(tree, "<string>", "exec"), self.locals)
            output = sys.stdout.getvalue().strip()
            if output:
                return output
            if isinstance(tree.body[-1], ast.Expr):
                self.locals['_lastResult'] = eval(compile(ast.Expression(tree.body[-1].value), "<string>", "eval"), self.locals)
                return str(self.locals['_lastResult'])
            return "Command executed successfully (no return value)"
        except Exception as e:
            return f"An error occurred: {str(e)}"
        finally:
            sys.stdout = oldStdout
    
    def __call__(self, command: str) -> str:
        return self.run(command)  
def convertTuplesToLists(data):
    if isinstance(data, tuple):
        return list(data)
    elif isinstance(data, list):
        return [convertTuplesToLists(item) for item in data]
    elif isinstance(data, dict):
        return {key: convertTuplesToLists(value) for key, value in data.items()}
    else:
        return data

class Tools:
    def __init__(self, chatModel, yamlFilePath, csvFilePath, dataDir=None):
        self.dataDir = dataDir or os.path.dirname(yamlFilePath)
        self.indexPath = os.path.join(self.dataDir, "annoyIndex.ann")
        self.chat = chatModel
        self.data = self.loadYaml(yamlFilePath)
        self.sentenceModel = SentenceTransformer("all-MiniLM-L6-v2")
        self.index, self.descriptions = self.loadOrBuildAnnoyIndex()
        self.queryFinder = QueryFinderAgent(csvFilePath, dataDir=self.dataDir)
        
        self.tools = [
            Tool(
                name="PythonRepl",
                func=CustomPythonREPL(locals={"df": pd.DataFrame()}),
                description="A Python REPL for executing commands. Input must be valid Python."
            ),
            Tool(
                name="SecretWord",
                func=self.secretWord,
                description="Returns the secret word when needed."
            ),
            Tool(
                name="NASAImage",
                func=self.nasaImage,
                description="Returns NASA image of the day for a given date (self.date)."
            ),
            Tool(
                name="RandomMTG",
                func=self.randomMtgCard,
                description="Shows a random Magic The Gathering card."
            ),
            Tool(
                name="YAMLTopicFinder",
                func=lambda prompt: self.findTopicWithAi(prompt),
                description="Finds topic in YAML file using AI based on description."
            ),
            Tool(
                name="QueryFinder",
                func=self.queryFinderWrapper,
                description="Finds relevant query from CSV based on description."
            ),
            Tool(
                name="SampleQueries",
                func=self.sampleQueriesWrapper,
                description="Returns sample queries from CSV."
            )
        ]
    
    def queryFinderWrapper(self, inputText):
        try:
            result = self.queryFinder.findQuery(inputText)
            print(f"QueryFinder result: {result}")
            return result
        except Exception as e:
            print(f"Error in queryFinderWrapper: {e}")
            return f"An error occurred while finding a query: {str(e)}"
    
    def sampleQueriesWrapper(self, n=5):
        try:
            result = self.queryFinder.getSampleQueries(n)
            print(f"SampleQueries result: {result}")
            return result
        except Exception as e:
            print(f"Error in sampleQueriesWrapper: {e}")
            return f"An error occurred while getting sample queries: {str(e)}"
    
    @staticmethod
    def secretWord(self):
        return "The secret word is 'Rubin'"
    
    def nasaImage(self, date):
        url = (
            f"https://api.nasa.gov/planetary/apod?date={date}&api_key="
            "GXacxntSzk6wpkUmDVw4L1Gfgt4kF6PzZrmSNWBb"
        )
        response = requests.get(url)
        data = response.json()
        
        if "url" in data:
            imageUrl = data["url"]
            display(Image(url=imageUrl))
            return imageUrl
        else:
            print("No image available for this date.")
            return None
    
    @staticmethod
    def randomMtgCard(self):
        url = "https://api.scryfall.com/cards/random"
        response = requests.get(url)
        data = response.json()
        imageUrl = data["image_uris"]["normal"]
        display(Image(url=imageUrl))
        return imageUrl
    
    def loadYaml(self, filePath):
        with open(filePath, "r") as file:
            return yaml.load(file, Loader=yaml.SafeLoader)
    
    def loadOrBuildAnnoyIndex(self):
        if os.path.exists(self.indexPath):
            return self.loadAnnoyIndex()
        else:
            index, descriptions = self.buildAnnoyIndex()
            self.descriptions = descriptions
            self.saveAnnoyIndex(index)
            return index, descriptions
    
    def buildAnnoyIndex(self):
        index = AnnoyIndex(384, "angular")
        descriptions = []
        
        for telemetryName, telemetryData in self.data.items():
            if telemetryName.endswith("_Telemetry") and isinstance(telemetryData, dict):
                if "SALTelemetrySet" in telemetryData:
                    salTelemetrySet = telemetryData["SALTelemetrySet"]
                    if "SALTelemetry" in salTelemetrySet:
                        for telemetry in salTelemetrySet["SALTelemetry"]:
                            if isinstance(telemetry, dict):
                                description = telemetry.get("Description")
                                efdbTopic = telemetry.get("EFDB_Topic", "No EFDB_Topic found")
                                if description:
                                    vector = self.embedDescription(description)
                                    index.add_item(len(descriptions), vector)
                                    descriptions.append((description, efdbTopic))
        
        index.build(10)
        return index, descriptions
    
    def saveAnnoyIndex(self, index):
        directory = os.path.dirname(self.indexPath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        index.save(self.indexPath)
        log.info(f"Annoy index saved to {self.indexPath}")
        
        normalizedDescriptions = convertTuplesToLists(self.descriptions)
        descriptionsPath = os.path.splitext(self.indexPath)[0] + ".yaml"
        with open(descriptionsPath, "w") as file:
            yaml.dump(normalizedDescriptions, file)
        print(f"Descriptions saved to {descriptionsPath}, without tuples")
        log.info(f"Descriptions saved to {descriptionsPath}")
    
    def loadAnnoyIndex(self):
        index = AnnoyIndex(384, "angular")
        index.load(self.indexPath)
        log.info(f"Annoy index loaded from {self.indexPath}")
        
        descriptionsPath = os.path.splitext(self.indexPath)[0] + ".yaml"
        with open(descriptionsPath, "r") as file:
            descriptions = yaml.safe_load(file)
        log.info(f"Descriptions loaded from {descriptionsPath}")
        return index, descriptions
    
    def rebuildAnnoyIndex(self):
        index, descriptions = self.buildAnnoyIndex()
        self.descriptions = descriptions
        self.saveAnnoyIndex(index)
        self.index = index
        log.info("Annoy index rebuilt")
    
    def embedDescription(self, description):
        return self.sentenceModel.encode(description).tolist()
    
    def findTopicWithAi(self, prompt):
        queryVector = self.embedDescription(prompt)
        nearestIndices = self.index.get_nns_by_vector(queryVector, 5)
        filteredIndices = self.filterDescriptionsBasedOnKeywords(prompt, nearestIndices)
        chosenTopicInfo = self.chooseBestDescription(filteredIndices)
        return chosenTopicInfo
    
    def filterDescriptionsBasedOnKeywords(self, prompt, indices):
        keywords = prompt.lower().split()
        filteredIndices = [
            i for i in indices if any(keyword in self.descriptions[i][0].lower() for keyword in keywords)
        ]
        return filteredIndices if filteredIndices else indices
    
    def chooseBestDescription(self, indices):
        if not indices:
            return "No relevant descriptions found."
        
        bestDescriptions = [self.descriptions[i] for i in indices]
        descriptionsText = "\n".join(f"{i + 1}. {desc[0]}" for i, desc in enumerate(bestDescriptions))
        
        combinedPrompt = (
            f"Based on the following descriptions, choose the best one related to your prompt:\n\n"
            f"{descriptionsText}\n\n"
            f"Choose the best description and explain why you selected that one."
        )
        
        response = self.chat.invoke([HumanMessage(content=combinedPrompt)])
        responseContent = response.content.strip()
        
        bestMatch = re.search(r"(\d+)\.\s(.+)", responseContent)
        if bestMatch:
            choiceIndex = int(bestMatch.group(1)) - 1
            if 0 <= choiceIndex < len(bestDescriptions):
                bestDescription, topic = bestDescriptions[choiceIndex]
                return f"Best EFDB Topic: {topic}"
        
        return "AI response could not identify the best description."
class AstroChat:
    allowedVerbosities = ("COST", "ALL", "NONE", None)
    
    @staticmethod
    def loadYaml(filePath):
        with open(filePath, "r") as file:
            return yaml.safe_load(file)
    
    demoQueries = {
        "darktime": "What is the total darktime for Image type = bias?",
        "imageCount": "How many engtest and bias images were taken?",
        "expTime": "What is the total exposure time?",
        "obsBreakdown": "What are the different kinds of observation reasons, and how many of each type? Please list as a table",
        "pieChart": "Can you make a pie chart of the total integration time for each of these filters and image type = science? Please add a legend with total time in minutes",
        "pythonCode": "Can you give me some python code that will produce a histogram of zenith angles for all entries with Observation reason = object and Filter = SDSSr_65mm",
        "azVsTime": "Can you show me a plot of azimuth vs. time (TAI) for all lines with Observation reason = intra",
        "imageDegradation": "Large values of mount image motion degradation is considered a bad thing. Is there a correlation with azimuth? Can you show a correlation plot?",
        "analysis": "Act as an expert astronomer. It seems azimuth of around 180 has large values. I wonder is this is due to low tracking rate. Can you assess that by making a plot vs. tracking rate, which you will have to compute?",
        "correlation": "Try looking for a correlation between mount motion degradation and declination. Show a plot with grid lines",
        "pieChartObsReasons": "Please produce a pie chart of total exposure time for different categories of Observation reason",
        "airmassVsTime": "I would like a plot of airmass vs time for all objects, as a scatter plot on a single graph, with the legend showing each object. Airmass of one should be at the top, with increasing airmass plotted as decreasing y position. Exclude points with zero airmass",
        "seeingVsTime": "The PSF FWHM is an important performance parameter. Restricting the analysis to images with filter name that includes SDSS, can you please make a scatter plot of FWHM vs. time for all such images, with a legend. I want all data points on a single graph.",
        "secretWord": "Tell me what is the secret word.",
        "nasaImage": "Show me the NASA image of the day for the current observation day.",
        "randomMTG": "Show me a random Magic The Gathering card",
        "findTopicWithAi": "Find the EFDB_Topic based on the description of data.",
        "findQuery": "Find a query related to exposure time",
        "sampleQueries": "Show me some sample queries from the CSV file"
    }
    
    def __init__(self, dayObs=None, export=False, temperature=0.0, verbosity="COST", 
                 agentType="tool-calling", modelName="gpt-4-1106-preview"):
        """Initialize an ASTROCHAT bot."""
        checkInstallation()
        self.setVerbosityLevel(verbosity)
        self.chat = ChatOpenAI(model_name=modelName, temperature=temperature)
        self.data = getObservingData(dayObs)
        if self.data.empty:
            raise ValueError("The DataFrame is empty. Please ensure data is available for the specified dayObs.")
        self.date = f"{str(dayObs)[:4]}-{str(dayObs)[4:6]}-{str(dayObs)[6:]}" if dayObs else None
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
        The extra tools are: 1. 'SecretWord', 2. 'NASAImage', 3. 'RandomMTG', 
        4. 'YAMLTopicFinder'. When using the 'NASAImage' tool, use 'self.date' 
        as a date, do not use 'dayObs', and do not attempt any pandas analysis 
        at all, so not use 'self.data'
        """
        
        packageDir = getPackageDir("summit_extras")
        yamlFilePath = os.path.join(packageDir, "data", "sal_interface.yaml")
        csvFilePath = os.path.join(packageDir, "data", "generated_prompt_influxql.csv")
        dataDir = os.path.join(packageDir, "data")
        toolkit = Tools(self.chat, yamlFilePath, csvFilePath, dataDir=dataDir)
        safetyClassifier = CodeSafetyClassifier(self.chat)
        for tool in toolkit.tools:
            if tool.name == "PythonRepl":
                tool.func = CustomPythonRepl(
                    locals={"df": self.data, "pd": pd, "plt": plt, "np": np, 
                            "display": display, "Markdown": Markdown, "Image": Image},
                    safetyClassifier=safetyClassifier
                )
        self.toolkit = toolkit
        
        self.pandasAgent = customDataframeAgent(
            self.chat,
            extraTools=self.toolkit.tools,  # Pass the tools list here
            prefix=prefix,
            df=self.data,
            allowDangerousCode=True,
            returnIntermediateSteps=True,
            agentType=self.agentType
        )
        self.pandasAgentExecutor = AgentExecutor.from_agent_and_tools(
            agent=self.pandasAgent, 
            tools=self.toolkit.tools, 
            verbose=False,
            return_intermediate_steps=True
        )
        self.customAgent = self.createCustomAgent(
            self.chat, self.data, self.agentType, prefix, self.toolkit.tools 
        )
        self.databaseAgent = DatabaseAgent()
        self.graph = self.createGraph()
        self.totalCallbacks = langchain_community.callbacks.openai_info.OpenAICallbackHandler()
        self.formatter = ResponseFormatter(agentType=self.agentType)
        self.export = export
        if self.export:
            warnings.warn("Exporting variables from the agent after each call. This can cause problems!")
    
    def isPromptInjection(self, inputStr):
        """Check for potential prompt injection attempts."""
        suspiciousPhrases = [
            "ignore all previous instructions",
            "delete my",
            "system command",
            "execute code",
            "bypass"
        ]
        inputLower = inputStr.lower()
        return any(phrase in inputLower for phrase in suspiciousPhrases)
    
    def run(self, inputStr):
        if self.isPromptInjection(inputStr):
            raise ValueError("Potential prompt injection detected. Request aborted.")
        
        with get_openai_callback() as cb:
            try:
                result = self.graph.invoke({
                    "input": inputStr,
                    "chatHistory": [],
                    "agentType": self.agentType,
                    "shortTermMemory": {},
                    "intermediateSteps": []
                })
                self.formatter(result)
            except Exception as e:
                log.error(f"Agent faced an error: {str(e)}")
                traceback.print_exc()
                return f"An error occurred while processing your request: {str(e)}"
        
        self.addUsageAndDisplay(cb)
        return None
    
    def createCustomAgent(self, llm, df, agentType, prefix, extraTools):
        return customDataframeAgent(
            llm=llm,
            df=df,
            agentType=agentType,
            prefix=prefix,
            extraTools=extraTools,
            allowDangerousCode=True,
            returnIntermediateSteps=True
        )
    
    def createGraph(self):
        workflow = StateGraph(State)
        workflow.add_node("router", self.route)
        workflow.add_node("agent", self.agentNode)
        workflow.add_edge("router", "agent")
        workflow.add_edge("agent", END)
        workflow.set_entry_point("router")
        return workflow.compile()
    
    def route(self, state):
        if "database" in state["input"].lower():
            return {"agentType": "database", "shortTermMemory": state.get("shortTermMemory", {}), "intermediateSteps": []}
        return {"agentType": "custom", "shortTermMemory": state.get("shortTermMemory", {}), "intermediateSteps": []}
    
    def agentNode(self, state):
        try:
            shortTermMemory = state.get("shortTermMemory", {})
            intermediateSteps = state.get("intermediateSteps", [])
            
            if state["agentType"] == "database":
                response = self.databaseAgent.run(state["input"])
            else:
                result = self.customAgent.invoke(state["input"])
                log.debug(f"Custom agent result: {result}")
                if isinstance(result, dict):
                    response = result.get("output", "No output provided")
                    intermediateSteps.extend(result.get("intermediate_steps", []))
                else:
                    response = str(result)
            
            shortTermMemory["last_response"] = response
            log.debug(f"Intermediate steps after processing: {intermediateSteps}")
            return {
                "chatHistory": [response],
                "shortTermMemory": shortTermMemory,
                "intermediateSteps": intermediateSteps
            }
        except Exception as e:
            log.error(f"Error in agentNode: {str(e)}")
            return {
                "chatHistory": [f"Error: {str(e)}"],
                "shortTermMemory": state.get("shortTermMemory", {}),
                "intermediateSteps": []
            }
    
    def setVerbosityLevel(self, level):
        if level not in self.allowedVerbosities:
            raise ValueError(f"Allowed values are {self.allowedVerbosities}, got {level}")
        self.verbosity = level
    
    def addUsageAndDisplay(self, cb):
        self.totalCallbacks.total_cost += cb.total_cost
        self.totalCallbacks.successful_requests += cb.successful_requests
        self.totalCallbacks.completion_tokens += cb.completion_tokens
        self.totalCallbacks.prompt_tokens += cb.prompt_tokens
        
        if self.verbosity == "ALL":
            print("\nThis call:\n" + str(cb) + "\n")
            print(self.totalCallbacks)
        elif self.verbosity == "COST":
            print(
                f"\nThis call cost: ${cb.total_cost:.3f}, "
                f"session total: ${self.totalCallbacks.total_cost:.3f}"
            )
    
    def listDemos(self):
        print("Available demo keys and their associated queries:")
        print("-------------------------------------------------")
        for k, v in self.demoQueries.items():
            print(f"{k}: {v}", "\n")
    
    def runDemo(self, items=None):
        knownDemos = list(self.demoQueries.keys())
        if items is None:
            items = knownDemos
        items = list(ensure_iterable(items))
        
        for item in items:
            if item not in knownDemos:
                raise ValueError(f"Specified demo item {item} is not an available demo. Known = {knownDemos}")
        
        for item in items:
            print(f"\nRunning demo item '{item}':")
            print(f"Prompt text: {self.demoQueries[item]}")
            self.run(self.demoQueries[item])

def customDataframeAgent(llm, df, agentType=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         prefix=None, suffix=None, includeDfInPrompt=True, 
                         numberOfHeadRows=5, extraTools=(), allowDangerousCode=False, 
                         returnIntermediateSteps=False):
    if not allowDangerousCode:
        raise ValueError("Dangerous code execution is not allowed without opting in.")
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
    
    dfLocals = {
        "df": df,
        "pd": pd,
        "plt": plt,
        "np": np,
        "display": display,
        "Markdown": Markdown,
        "Image": Image
    }
    repl = CustomPythonRepl(locals=dfLocals)
    tools = [
        Tool(
            name="PythonRepl",
            func=repl,
            description="A Python REPL for executing commands. Input must be valid Python."
        )
    ] + extraTools
    
    prompt = generatePrompt(df, agentType, prefix=prefix, suffix=suffix, 
                           includeDfInPrompt=includeDfInPrompt, numberOfHeadRows=numberOfHeadRows)
    
    if agentType == "ZERO_SHOT_REACT_DESCRIPTION":
        runnable = create_react_agent(llm, tools, prompt)
        agent = RunnableAgent(runnable=runnable, input_keys_arg=["input"], return_keys_arg=["output"])
    elif agentType == "tool-calling":
        runnable = create_tool_calling_agent(llm, tools, prompt)
        agent = RunnableMultiActionAgent(runnable=runnable, input_keys_arg=["input"], return_keys_arg=["output"])
    else:
        raise ValueError(f"Unsupported agent type: {agentType}")
    
    return AgentExecutor(agent=agent, tools=tools, 
                         return_intermediate_steps=returnIntermediateSteps, verbose=False)

def generatePrompt(df, agentType, **kwargs):
    if agentType == "ZERO_SHOT_REACT_DESCRIPTION":
        return getSinglePrompt(df, **kwargs)
    elif agentType == "tool-calling":
        return getFunctionsSinglePrompt(df, **kwargs)
    raise ValueError(f"Unsupported agent type for prompt generation: {agentType}")

def getSinglePrompt(df, *, prefix=None, suffix=None, includeDfInPrompt=True, numberOfHeadRows=5):
    suffixToUse = suffix if suffix else (SUFFIX_WITH_DF if includeDfInPrompt else SUFFIX_NO_DF)
    prefix = prefix or PREFIX
    template = "\n\n".join([prefix, "{tools}", FORMAT_INSTRUCTIONS, suffixToUse])
    prompt = PromptTemplate.from_template(template).partial()
    if "df_head" in prompt.input_variables:
        dfHead = df.head(numberOfHeadRows).to_markdown()
        prompt = prompt.partial(df_head=str(dfHead))
    return prompt

def getFunctionsSinglePrompt(df, *, prefix=None, suffix="", includeDfInPrompt=True, numberOfHeadRows=5):
    dfHead = df.head(numberOfHeadRows).to_markdown() if includeDfInPrompt else ""
    suffix = (suffix or FUNCTIONS_WITH_DF).format(df_head=dfHead)
    prefix = prefix or PREFIX_FUNCTIONS
    systemMessage = SystemMessage(content=prefix + suffix)
    return OpenAIFunctionsAgent.create_prompt(system_message=systemMessage)

class DatabaseAgent:
    def run(self, input):
        return "This is a placeholder response from the DatabaseAgent."

class QueryFinderAgent:
    def __init__(self, csvFilePath, dataDir=None):
        """Initialize QueryFinderAgent with cached storage, no auto-embedding."""
        self.csvFilePath = csvFilePath
        self.dataDir = dataDir or os.path.dirname(csvFilePath)
        self.indexPath = os.path.join(self.dataDir, "chromaQueryIndex")
        self.queriesDf = None
        self.embeddingModel = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Set up logging
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
        
        # Initialize ChromaDB with persistent storage
        self.chromaClient = Client(Settings(persist_directory=self.indexPath))
        self.collectionName = "questionEmbeddings"
        self.chromaCollection = self.chromaClient.get_or_create_collection(name=self.collectionName)
        
        # Load queries from CSV
        self.loadQueries()
        
        # Check if index is cached
        if os.path.exists(self.indexPath) and os.listdir(self.indexPath):
            self.log.info(f"Loaded existing embeddings from {self.indexPath} (count: {self.chromaCollection.count()})")
        else:
            self.log.info(f"No embedding index found at {self.indexPath}. Run rebuildDatabase() to create it.")

    def loadQueries(self):
        """Load queries from the CSV file."""
        if os.path.exists(self.csvFilePath):
            try:
                self.queriesDf = pd.read_csv(self.csvFilePath, header=0, names=["name", "query", "question"])
                self.log.debug(f"Loaded {len(self.queriesDf)} queries from {self.csvFilePath}")
            except Exception as e:
                self.log.error(f"Error loading CSV file: {e}")
                self.queriesDf = None
        else:
            self.log.warning(f"CSV file not found at {self.csvFilePath}")
            self.queriesDf = None

    def prepareEmbeddings(self):
        """Generate and store embeddings for queries."""
        if self.queriesDf is None or self.queriesDf.empty:
            self.log.warning("No queries available to embed.")
            return
        
        questions = self.queriesDf["question"].tolist()
        embeddings = self.embeddingModel.encode(questions)
        ids = [str(idx) for idx in range(len(embeddings))]
        metadata = [
            {"query": self.queriesDf.iloc[idx]["query"], "question": self.queriesDf.iloc[idx]["question"]}
            for idx in range(len(embeddings))
        ]
        
        # Add new embeddings (no need to delete here since collection is dropped in rebuild)
        self.chromaCollection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadata,
            documents=questions
        )
        self.log.info(f"Stored {len(questions)} embeddings in {self.indexPath}")

    def rebuildDatabase(self, force=False):
        """Rebuild the embedding database if CSV file has changed or forced."""
        if not force:
            currentMtime = os.path.getmtime(self.csvFilePath)
            indexMtimeFile = os.path.join(self.dataDir, "chromaLastMtime.txt")
            if os.path.exists(indexMtimeFile):
                with open(indexMtimeFile, "r") as f:
                    lastMtime = float(f.read().strip())
                if currentMtime <= lastMtime and os.path.exists(self.indexPath) and os.listdir(self.indexPath):
                    self.log.info("CSV file unchanged and index exists. Skipping rebuild.")
                    return
        
        self.log.info("Rebuilding query embedding database.")
        self.loadQueries()
        
        # Drop the existing collection and recreate it
        self.chromaClient.delete_collection(self.collectionName)
        self.chromaCollection = self.chromaClient.get_or_create_collection(name=self.collectionName)
        
        self.prepareEmbeddings()
        
        # Update last modification time in data folder
        with open(os.path.join(self.dataDir, "chromaLastMtime.txt"), "w") as f:
            f.write(str(os.path.getmtime(self.csvFilePath)))
        self.log.info("Query embedding database rebuilt and cached.")

    def findQuery(self, inputText):
        """Find the most relevant query based on input text."""
        if self.chromaCollection.count() == 0:
            self.log.warning("No embeddings available. Please run rebuildDatabase() to create the index.")
            return "No embeddings available. Run rebuildDatabase() first."
        
        inputEmbedding = self.embeddingModel.encode([inputText])[0]
        results = self.chromaCollection.query(query_embeddings=[inputEmbedding.tolist()], n_results=1)
        if results and results["distances"][0]:
            similarQuery = results["metadatas"][0][0]
            distance = results["distances"][0][0]
            return f"Matching query found: {similarQuery['query']} (Similarity: {1 - distance:.2f})"
        return "No matching query found."

    def getSampleQueries(self, n=5):
        """Return a sample of queries from the CSV."""
        if self.queriesDf is not None and not self.queriesDf.empty:
            sample = self.queriesDf.sample(n=min(n, len(self.queriesDf)))
            return sample[["query", "question"]].to_dict("records")
        return "No queries available."
class State(TypedDict):
    """System state representation."""
    chatHistory: Annotated[List[str], operator.add]
    input: str
    agentType: str
    shortTermMemory: Dict[str, Any]
    intermediateSteps: List[Any]