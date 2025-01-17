# IMPORTACIÓN DE MÓDULOS
# ---------------------
# Los imports se organizan en grupos lógicos:
# 1. Módulos estándar de Python (built-in)
# 2. Módulos de terceros
# 3. Módulos locales del proyecto
# Esta organización es una buena práctica de Python (PEP 8)

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

"""Proyecto DSPy con integración JIRA y Workers especializados.

Este archivo ilustra:
- La configuración base de un proyecto DSPy.
- Varios Workers (e.g., RAG, Python, JIRA) bajo la orquestación de un Boss.
- Capacidades de memoria de conversación y generación de código en Python.
- Extracción de JQL (usando un ChainOfThought y DSPy) para realizar
  consultas a la API de JIRA.

Notas
-----
Las docstrings están en español para mayor claridad. Para alinear con
las convenciones LSST, las líneas de comentarios y docstrings no deben
exceder 79 caracteres. Las líneas de código no deben exceder 110
caracteres. Ajusta rutas, credenciales y endpoints según tu entorno.
"""

__all__ = [
    "UserInteraction",
    "ConversationContext",
    "ResponseWithContext",
    "respond_cot",
    "getObservingData",
    "retrieve_observing_metadata",
    "ProgramOfThought",
    "Tool",
    "rag_tool",
    "python_tool",
    "dummy_tool",
    "Worker",
    "rag_worker",
    "python_worker",
    "Boss",
    "call_jira_with_jql",
    "ExtractJQL",
    "extract_jql_cot",
    "JiraWorker",
    "jira_query_tool",
    "WorkerWithJira"
]

import logging # se utiliza para registrar mensajes de error y depuración
import os # se utiliza para cargar variables de entorno y rutas
import re # se utiliza para manipulación de cadenas
import requests # se utiliza para hacer llamadas HTTP a la API de JIRA
import json # se utiliza para manipulación de datos JSON
import yaml # se utiliza para cargar archivos de configuración
import dotenv # se utiliza para cargar variables de entorno desde archivos .env
import pandas as pd # se utiliza para manipulación de datos tabulares

from urllib.parse import quote # se utiliza para escapar caracteres en URLs
from typing import List, Optional, Callable # se utiliza para tipado estático
from pathlib import Path # se utiliza en la carga de credenciales 
from pydantic import BaseModel # se utiliza para definir modelos de datos

import dspy # se utiliza para definir flujos de trabajo y modelos de datos
from dspy import (
    Signature,
    ChainOfThought,
    Module,
    settings
)
from dspy.primitives.python_interpreter import CodePrompt, PythonInterpreter # se utiliza para ejecutar código Python

###############################################################################
# CONFIGURACIÓN DE LOGGING Y MODELO LLM
###############################################################################
# El logging es fundamental para depurar y monitorear aplicaciones
# Configuramos el nivel básico de logging a INFO, lo que significa que
# registrará todos los mensajes informativos, warnings y errores

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

###############################################################################
# CONFIGURACIÓN DE CREDENCIALES Y RUTAS
###############################################################################
class ConfigManager:
    """Gestor de configuración que implementa el patrón Singleton
    
    Esta clase demuestra:
    1. Uso de tipos opcionales con Optional[]
    2. Manejo de configuración con múltiples fuentes (env vars, archivos)
    3. Implementación de fallbacks (si una fuente falla, prueba la siguiente)
    """
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or '/Users/cmorales/Documents/summit_extras/data/config.yaml'
        
    def load_credentials(self, key_name: str) -> str:
        """Load credentials using multiple fallback methods"""
        
        # Try environment variables first
        if value := os.getenv(key_name):
            return value
            
        # Try config file
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                if value := config.get(key_name):
                    return value
                    
        raise ValueError(f"Could not find credentials for {key_name}")
    
# Initialize config manager
config = ConfigManager()

# Load credentials
try:
    api_key = config.load_credentials('OPENAI_API_KEY')
    jira_token = config.load_credentials('JIRA_API_TOKEN')
    jira_hostname = config.load_credentials('JIRA_API_HOSTNAME')
    jira_username = config.load_credentials('JIRA_API_USERNAME')
except ValueError as e:
    LOG.error(f"Failed to load credentials: {e}")
    raise

###############################################################################
# CONFIGURACIÓN DE MODELO DE LENGUAJE
###############################################################################

lm = dspy.LM(# 'ollama_chat/tulu3:70b', 
             # 'ollama_chat/olmo2:13b-1124-instruct-fp16',
             # 'ollama_chat/olmo2:13b',
             # 'ollama_chat/olmo2:7b',
             # 'ollama_chat/phi4:14b-fp16',
             # 'ollama_chat/dolphin3:latest',
             # 'ollama_chat/llama3.3:70b-instruct-q2_K',
             # 'ollama_chat/ALIENTELLIGENCE/cybersecuritymonitoring:latest',
             # 'ollama_chat/ALIENTELLIGENCE/whiterabbitv2:latest',
             # 'ollama_chat/TULU_131k_T08_70b:latest',
             # 'ollama_chat/COT_32k_T03_qwq:latest',
             # 'ollama_chat/qwq:latest',
             # 'ollama_chat/qwq:32b',
             # 'ollama_chat/tulu3:8b',
             'ollama_chat/tulu3:70b',
             # 'ollama_chat/hermes3:3b',
             # 'ollama_chat/hermes3:8b',
             # 'ollama_chat/hermes3:70b',
             # 'ollama_chat/hermes3:405b',
             # 'ollama_chat/llama3.3:70b',
             # 'ollama_chat/llama3.2-vision:90b',
             # 'ollama_chat/llama3.2:3b',
             # 'ollama_chat/qwen2.5:0.5b',
             # 'ollama_chat/qwen2.5:32b',
             # 'ollama_chat/qwen2.5:72b',
             # 'ollama_chat/qwen2.5-coder:32b',
             # 'ollama_chat/nemotron:70b',
             # 'ollama_chat/snowflake-arctic-embed:latest',
             # 'ollama_chat/nomic-embed-text:latest',
             
             api_base='https://5bbc-200-29-147-35.ngrok-free.app', api_key='') #api_key should be empty
# lm = dspy.LM('ollama_chat/tulu3', api_base='http://localhost:11434', api_key='') #api_key should be empty
# lm = LM("openai/gpt-4o", api_key=api_key)

settings.configure(lm=lm)

###############################################################################
# CLASES DE INTERACCIÓN Y MEMORIA DE CONVERSACIÓN
###############################################################################
class UserInteraction:
    """Ejemplo de una clase bien documentada que sigue el estilo NumPy
    
    Demuestra:
    1. Uso de docstrings con formato NumPy
    2. Documentación de parámetros y tipos
    3. Encapsulamiento de datos relacionados
    """
    """Representa una interacción (usuario - asistente).

    Parameters
    ----------
    message : `str`
        Mensaje del usuario.
    response : `str`
        Respuesta del asistente.
    """

    def __init__(self, message: str, response: str):
        self.message = message
        self.response = response

    def serialize_by_role(self) -> List[dict]:
        """Devuelve esta interacción en formato apto para prompts.

        Returns
        -------
        serialized : `list` de `dict`
            Roles y contenidos para usuario y asistente.
        """
        return [
            {"role": "user", "content": self.message},
            {"role": "assistant", "content": self.response}
        ]


class ConversationContext:
    """Implementación de un buffer circular para mantener contexto
    
    Demuestra:
    1. Uso de estructuras de datos (lista como buffer)
    2. Manejo de estado interno
    3. Métodos para manipular y presentar datos
    """
    """Almacena las interacciones recientes como contexto de conversación.

    Parameters
    ----------
    window_size : `int`, optional
        Cantidad de interacciones a mantener (por defecto 5).
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.content: List[UserInteraction] = []

    def update(self, interaction: UserInteraction) -> None:
        """Agrega una nueva interacción y recorta para mantener `window_size`.

        Parameters
        ----------
        interaction : `UserInteraction`
            La nueva interacción usuario-asistente.
        """
        self.content.append(interaction)
        self.content = self.content[-self.window_size:]

    def render(self) -> str:
        """Devuelve un string con las interacciones recientes.

        Returns
        -------
        context_str : `str`
            Texto que representa el historial de conversación.
        """
        out = []
        for c in self.content:
            out.append(
                f"User: {c.message}\n\nAssistant: {c.response}\n\n"
            )
        return "".join(out)

###############################################################################
# FIRMA Y CLASE PARA RESPUESTAS CON CONTEXTO
###############################################################################
class ResponseWithContext(Signature):
    """Solicita 'context', 'message' y 'rag_context'; devuelve 'response'.

    Attributes
    ----------
    context : `str`
        Contexto de conversación.
    message : `str`
        Mensaje del usuario.
    rag_context : `str`
        Información relevante proveniente de RAG u observaciones.
    response : `str`
        Respuesta contextualizada producida por el asistente.
    """
    context = dspy.InputField(desc="Contexto de la conversación")
    message = dspy.InputField(desc="Mensaje del usuario")
    rag_context = dspy.InputField(
        desc="Información relevante (observaciones u otras fuentes)"
    )
    response = dspy.OutputField(desc="Respuesta contextualizada")

respond_cot = ChainOfThought(ResponseWithContext)

###############################################################################
# FUNCIONES DE RAG: OBTENCIÓN DE DATOS DE OBSERVACIÓN
###############################################################################
def getObservingData(dayObs: Optional[int] = None) -> pd.DataFrame:
    """Obtiene un DataFrame con datos de observación (ejemplo dummy).

    Parameters
    ----------
    dayObs : `int` or `None`, optional
        Un identificador de día de observación. Si es None, se usa un
        valor por defecto.

    Returns
    -------
    df : `pandas.DataFrame`
        DataFrame con la información de observación (puede estar vacío).
    """
    currentDayObs = 20230913  # archivo local de ejemplo
    if dayObs is None:
        dayObs = currentDayObs

    filename = f"/Users/cmorales/Documents/summit_extras/data/dayObs_{dayObs}.json"
    if not os.path.exists(filename):
        LOG.warning("Archivo no encontrado: %s", filename)
        return pd.DataFrame()

    try:
        df = pd.read_json(filename).T
        df = df.drop(
            [col for col in df.columns if col.startswith("_")],
            axis=1,
            errors="ignore"
        )
        return df.sort_index()
    except Exception as exc:
        LOG.error("Error leyendo %s: %s", filename, exc)
        return pd.DataFrame()


def retrieve_observing_metadata(query: str,
                                dayObs: Optional[int] = None) -> str:
    """Filtra un DataFrame con lógica simple, devolviendo resultados en texto.

    Parameters
    ----------
    query : `str`
        Substring a buscar en el DataFrame.
    dayObs : `int` or `None`, optional
        Día de observación. Si None, se usa un valor ficticio.

    Returns
    -------
    result_str : `str`
        Cadena con resultados o mensaje si no se halla nada.
    """
    df = getObservingData(dayObs)
    if df.empty:
        return "No observing data available."

    result = df[df.apply(
        lambda row: query.lower() in row.to_string().lower(), axis=1
    )]
    if result.empty:
        return "No relevant data found."
    return result.to_string(index=False)

###############################################################################
# PROGRAM OF THOUGHT PARA EJECUTAR CÓDIGO PYTHON
###############################################################################
class ProgramOfThought(Module):
    """Ejemplo avanzado de ejecución segura de código Python
    
    Demuestra:
    1. Uso de decoradores y herencia
    2. Manejo seguro de ejecución de código
    3. Implementación de retry logic
    4. Uso de white lists para seguridad
    """
    """Genera y ejecuta código Python hasta `max_iters` iteraciones.

    Utiliza PythonInterpreter con un import_white_list para controlar
    paquetes permitidos.

    Parameters
    ----------
    signature : `dspy.Signature`
        Define el input/output esperado.
    max_iters : `int`, optional
        Número máximo de intentos de generación/ejecución de código.
    """

    def __init__(self, signature: Signature, max_iters: int = 3):
        super().__init__()
        self.signature = signature
        self.max_iters = max_iters
        self.interpreter = PythonInterpreter(
            action_space={"print": print},
            import_white_list=["numpy", "astropy", "sympy", "matplotlib"]
        )

    def forward(self, question: str, context_str: str = "",
                rag_context: str = "", variables: dict = None) -> dict:
        """Genera y ejecuta código Python para `question`.

        Parameters
        ----------
        question : `str`
            Pregunta o instrucción a resolver.
        context_str : `str`, optional
            Contexto de conversación.
        rag_context : `str`, optional
            Datos provenientes de RAG.
        variables : `dict`, optional
            Variables adicionales.

        Returns
        -------
        resultado : `dict`
            Diccionario con clave "answer" que contiene la respuesta
            de la ejecución o None si falla.
        """
        if variables is None:
            variables = {}

        for i in range(self.max_iters):
            generated_code = self._generate_code(
                question, context_str, rag_context
            )
            print(f"\nIteración {i + 1}: Código generado:\n{generated_code}")

            try:
                result, _ = CodePrompt(generated_code).execute(
                    self.interpreter,
                    user_variable=variables
                )
                print("Resultado de ejecución:", result)
                return {"answer": result}
            except Exception as exc:
                LOG.error("Error de ejecución: %s", exc)

        LOG.error("No se pudo generar código válido tras varios intentos.")
        return {"answer": None}

    def _generate_code(self, question: str, context_str: str,
                       rag_context: str) -> str:
        """Genera el código Python usando el modelo LLM.

        Parameters
        ----------
        question : `str`
            Pregunta del usuario.
        context_str : `str`
            Contexto conversacional.
        rag_context : `str`
            Datos de RAG.

        Returns
        -------
        code : `str`
            Código Python "limpio" listo para ejecución.
        """
        code_signature = dspy.Signature("question -> generated_code")
        predict = dspy.Predict(code_signature)

        prompt = (
            "Eres una IA que escribe código Python para resolver problemas.\n"
            "Contexto de conversación:\n"
            f"{context_str}\n\n"
            "Datos de observación (RAG):\n"
            f"{rag_context}\n\n"
            "Genera solo el código Python, sin explicaciones ni markdown.\n"
            f"Pregunta: {question}\n\n"
            "Código:\n"
        )
        completion = predict(question=prompt)
        code_block = re.sub(r"```[^\n]*\n", "", completion.generated_code)
        code_block = re.sub(r"```", "", code_block)
        return code_block

###############################################################################
# DEFINICIÓN DE TOOLS
###############################################################################
class Tool(BaseModel):
    """Implementación del patrón Strategy usando Pydantic
    
    Demuestra:
    1. Uso de Pydantic para validación de datos
    2. Tipado estático en Python
    3. Uso de Callables como tipos
    """
    """Plantilla base para herramientas (Tools) que usan los Workers.

    Parameters
    ----------
    name : `str`
        Nombre de la herramienta.
    description : `str`
        Descripción breve de la funcionalidad.
    requires : `str`
        Parámetro(s) requerido(s) por la herramienta.
    func : `Callable`
        Función que implementa la lógica de la herramienta.
    """
    name: str
    description: str
    requires: str
    func: Callable


rag_tool = Tool(
    name="rag retrieval",
    description="Busca datos de observación según un query y los devuelve.",
    requires="query",
    func=lambda q: retrieve_observing_metadata(q)
)

code_tool_signature = dspy.Signature("question -> answer")
code_program = ProgramOfThought(signature=code_tool_signature, max_iters=3)

python_tool = Tool(
    name="python executor",
    description=("Genera y ejecuta código Python con contexto usando "
                 "ProgramOfThought."),
    requires="question",
    func=lambda question_dict: code_program(
        question=question_dict.get("question", ""),
        context_str=question_dict.get("context", ""),
        rag_context=question_dict.get("rag_context", ""),
        variables=question_dict.get("variables", {})
    )
)

dummy_tool = Tool(
    name="dummy",
    description="Herramienta de ejemplo para futuras extensiones.",
    requires="some_argument",
    func=lambda arg: f"Dummy tool called with argument: {arg}"
)

###############################################################################
# WORKERS GENÉRICOS: RAG Y PYTHON
###############################################################################
class Worker(Module):
    """Implementación del patrón Worker
    
    Demuestra:
    1. Diseño orientado a objetos
    2. Separación de responsabilidades
    3. Planificación y ejecución de tareas
    """
    """Worker genérico que elabora un plan y ejecuta herramientas.

    Parameters
    ----------
    role : `str`
        Rol o identificación del worker (e.g., "rag_worker").
    tools : `List[Tool]`
        Lista de herramientas disponibles.
    """

    def __init__(self, role: str, tools: List[Tool]):
        self.role = role
        self.tools = {t.name: t for t in tools}
        self._plan = ChainOfThought("task, context -> proposed_plan")
        self._use_tool = ChainOfThought("task, context -> tool_name, tool_argument")

    def plan(self, task: str, feedback: str = "") -> str:
        """Genera un plan textual simplificado.

        Parameters
        ----------
        task : `str`
            Descripción o meta de la tarea.
        feedback : `str`, optional
            Comentarios adicionales.

        Returns
        -------
        plan_text : `str`
            Plan propuesto.
        """
        context = (
            f"Worker role: {self.role}; Tools: {list(self.tools.keys())}; "
            f"Feedback: {feedback}"
        )
        plan_result = self._plan(task=task, context=context)
        return plan_result.proposed_plan

    def execute(self, task: str, use_tool: bool, context: str = "") -> str:
        """Ejecuta la tarea, con o sin usar herramienta.

        Parameters
        ----------
        task : `str`
            Tarea o instrucción.
        use_tool : `bool`
            Indica si se usará alguna herramienta.
        context : `str`, optional
            Texto adicional (plan, etc.).

        Returns
        -------
        outcome : `str`
            Resultado o mensaje de error.
        """
        print(f"[{self.role}] Ejecutando: {task}")
        if not use_tool:
            return f"Tarea '{task}' completada sin herramientas."

        tool_decision = self._use_tool(task=task, context=context)
        tool_name = tool_decision.tool_name
        arg = tool_decision.tool_argument

        if tool_name in self.tools:
            return self.tools[tool_name].func(arg)
        return f"Herramienta '{tool_name}' no encontrada."

rag_worker = Worker("rag_worker", tools=[rag_tool, dummy_tool])
python_worker = Worker("python_worker", tools=[python_tool, dummy_tool])

###############################################################################
# BOSS PARA ORQUESTACIÓN
###############################################################################
class Boss(Module):
    """Implementación del patrón Mediator/Orchestrator
    
    Demuestra:
    1. Gestión de múltiples workers
    2. Toma de decisiones basada en contexto
    3. Patrones de diseño avanzados
    """
    """Orquesta y asigna tareas a los Workers.

    Parameters
    ----------
    workers : `list` de `Module`
        Lista de workers disponibles.
    """

    def __init__(self, workers: List[Module]):
        self.workers = {w.role: w for w in workers}
        self._assign = ChainOfThought("task -> who")
        self._approve = ChainOfThought("task, plan -> approval")

    def plan_and_execute(self, task: str, worker_hint: str = "",
                         use_tool: bool = True) -> str:
        """Decide qué worker usar, genera y aprueba el plan, y ejecuta.

        Parameters
        ----------
        task : `str`
            Objetivo del usuario.
        worker_hint : `str`, optional
            Sugerencia de qué worker usar.
        use_tool : `bool`, optional
            Si el worker usará herramientas.

        Returns
        -------
        exec_result : `str`
            Resultado de la ejecución del worker.
        """
        if worker_hint and worker_hint in self.workers:
            assignee = worker_hint
        else:
            assign_result = self._assign(task=task)
            assignee = assign_result.who

        if assignee not in self.workers:
            return f"No existe un worker '{assignee}'."

        plan_text = self.workers[assignee].plan(task)
        print(f"Plan propuesto: {plan_text}")

        approval_res = self._approve(task=task, plan=plan_text)
        is_approved = "yes" in approval_res.approval.lower()
        if not is_approved:
            LOG.info("El plan no fue aprobado. Se podría iterar o ajustar.")
            # Aquí podrías refinar el plan si lo deseas.

        context = f"Plan: {plan_text}; Aprobado"
        result = self.workers[assignee].execute(task, use_tool=use_tool,
                                                context=context)
        return result

###############################################################################
# NUEVO: LÓGICA PARA JIRA (EXTRACCIÓN DE JQL Y LLAMADA A LA API DE JIRA)
###############################################################################
def call_jira_with_jql(jql: str) -> str:
    """Ejemplo de integración con API externa
    
    Demuestra:
    1. Manejo de requests HTTP
    2. Procesamiento de respuestas JSON
    3. Manejo de errores y casos borde
    """
    """Llama la API de JIRA para ejecutar el JQL y devuelve resultados.

    Parameters
    ----------
    jql : `str`
        Cadena JQL para la búsqueda en JIRA.

    Returns
    -------
    response_str : `str`
        Texto con los issues hallados o mensaje de error.
    """
    try:
        # Use ConfigManager instead of direct os.getenv
        token = config.load_credentials('JIRA_API_TOKEN')
        hostname = config.load_credentials('JIRA_API_HOSTNAME')
        username = config.load_credentials('JIRA_API_USERNAME')
        
        # Create base64 encoded credentials
        credentials = base64.b64encode(
            f"{username}:{token}".encode()
        ).decode()
        
        headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/json"
        }
        
        # Construct proper JIRA REST API URL
        api_path = "rest/api/3/search"
        base_url = hostname.rstrip('/')  # Remove trailing slash if present
        url = f"{base_url}/{api_path}"
        
        # Add JQL to params
        params = {
            'jql': jql,
            'maxResults': 50  # Limit results
        }
        
        LOG.info(f"Calling JIRA API with URL: {url}")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        return response.json()
        
    except ValueError as e:
        LOG.error(f"Configuration error: {e}")
        return f"Error in JIRA configuration: {str(e)}"
    except requests.exceptions.RequestException as e:
        LOG.error(f"JIRA API request failed: {e}")
        return f"Error calling JIRA API: {str(e)}"
class ExtractJQL(Signature):
    """Firma para extraer la JQL de una consulta en lenguaje natural.

    Attributes
    ----------
    user_input : `str`
        Texto del usuario. Ej: "Muéstrame todos los tickets del proyecto OBS."
    jql_query : `str`
        La JQL final para la búsqueda en JIRA.
    """
    user_input = dspy.InputField(desc="Consulta del usuario en lenguaje natural.")
    jql_query = dspy.OutputField(desc="La consulta JQL extraída.")


extract_jql_cot = ChainOfThought(ExtractJQL)


class JiraWorker(Module):
    """Worker especializado en consultas a JIRA.

    1) Usa extract_jql_cot para extraer la JQL.
    2) Llama call_jira_with_jql(jql) y retorna los resultados.
    """

    def __init__(self, role: str):
        self.role = role
        self.extract_jql = extract_jql_cot

    def plan(self, user_input: str) -> str:
        """Genera un plan básico en texto."""
        return ("Plan: 1) Extraer JQL. 2) Llamar a la API de JIRA con la JQL.")

    def execute(self, user_input: str) -> str:
        """Extrae la JQL y llama a JIRA para obtener resultados."""
        jql_result = self.extract_jql(user_input=user_input)
        jql_query = jql_result.jql_query

        if not jql_query:
            return "No se pudo extraer una JQL válida."

        response = call_jira_with_jql(jql_query)
        return response


###############################################################################
# TOOL Y WORKER PARA CONSULTAS JIRA
###############################################################################
jira_query_tool = Tool(
    name="jira query",
    description="Extrae JQL de la consulta del usuario y busca en JIRA.",
    requires="user_input",
    func=lambda user_input: JiraWorker("jira_worker").execute(user_input)
)


class WorkerWithJira(Module):
    """Worker que puede utilizar la herramienta JIRA (entre otras).

    Parameters
    ----------
    role : `str`
        Rol/nombre del worker.
    tools : `list` de `Tool`
        Lista de herramientas que puede usar este worker.
    """

    def __init__(self, role: str, tools: List[Tool]):
        self.role = role
        self.tools = {t.name: t for t in tools}
        self._plan = ChainOfThought("task, context -> proposed_plan")
        self._use_tool = ChainOfThought("task, context -> tool_name, tool_argument")

    def plan(self, task: str, feedback: str = "") -> str:
        """Genera un plan textual sencillo."""
        context = (
            f"Worker role: {self.role}; Tools: {list(self.tools.keys())}; "
            f"Feedback:{feedback}"
        )
        plan_result = self._plan(task=task, context=context)
        return plan_result.proposed_plan

    def execute(self, task: str, use_tool: bool, context: str = "") -> str:
        """Decide si usa una herramienta y la ejecuta."""
        if not use_tool:
            return f"Tarea '{task}' completada sin usar herramienta."

        decision = self._use_tool(task=task, context=context)
        tool_name = decision.tool_name
        tool_arg = decision.tool_argument

        if tool_name in self.tools:
            return self.tools[tool_name].func(tool_arg)
        return f"Herramienta '{tool_name}' no encontrada."


###############################################################################
# EJEMPLO PRINCIPAL
###############################################################################
def main() -> None:
    """Función principal de ejemplo."""
    # Creamos un Boss con un JiraWorker, o con WorkerWithJira usando la tool.
    my_jira_worker = JiraWorker("jira_worker")
    boss = Boss(workers=[
        my_jira_worker,
        rag_worker,
        python_worker
    ])

    # Ejemplo de tarea RAG
    print("\n=== EJEMPLO 1: RAG ===")
    result_1 = boss.plan_and_execute(
        task="genera un grafico con matplotlib tomando los datos de observación de RAG.",
        worker_hint="rag_worker",
        use_tool=True
    )
    print("Resultado final (RAG):", result_1)

    # Ejemplo de ejecución de código Python
    print("\n=== EJEMPLO 2: Código Python ===")
    result_2 = boss.plan_and_execute(
        task="Calcula la energía potencial de un objeto con masa=10kg a 5m.",
        worker_hint="python_worker",
        use_tool=True
    )
    print("Resultado final (Código):", result_2)

    # Ejemplo de consulta JIRA
    print("\n=== EJEMPLO 3: Consulta JIRA ===")
    jira_task = ("Muéstrame todos los tickets del proyecto OBS "
                 "creados hoy y asignados a carlos morales")
    result_3 = boss.plan_and_execute(
        task=jira_task,
        worker_hint="jira_worker",
        use_tool=True
    )
    print("Resultado final (JIRA):", result_3)


if __name__ == "__main__":
    main()