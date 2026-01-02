import streamlit as st
import google.generativeai as genai
import json
import re
import traceback
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
# Default to empty, user can input in UI if not in secrets
API_KEY = ""
try:
    if "gemini_api_key" in st.secrets:
        API_KEY = st.secrets["gemini_api_key"]
except:
    # Secrets not found/accessible, API_KEY remains empty
    pass

# --- 1. THE SCHEMA (ZERO HALLUCINATION LAYER) ---

class ToolConfig(BaseModel):
    name: str
    description: str
    is_custom: bool = Field(..., description="True if this tool needs to be written from scratch in Python")
    python_logic: Optional[str] = Field(None, description="If custom, provide the Python code inside a @tool decorator")

class KnowledgeConfig(BaseModel):
    title: str
    source_type: Literal["text", "pdf", "csv", "excel", "json", "url"]
    content_or_path: str = Field(..., description="The content string or file path/URL")
    description: str

class AgentConfig(BaseModel):
    role: str
    goal: str
    backstory: str
    tools: List[ToolConfig] = []
    knowledge_sources: List[KnowledgeConfig] = []
    allow_delegation: bool = False
    verbose: bool = True

class TaskConfig(BaseModel):
    name: str
    description: str
    expected_output: str
    agent_role: str = Field(..., description="Must match one of the Agent roles exactly")
    context_from: List[str] = Field(default_factory=list, description="List of previous task names this task depends on")
    async_execution: bool = False

class CrewStructure(BaseModel):
    project_name: str
    agents: List[AgentConfig]
    tasks: List[TaskConfig]
    process_type: Literal["sequential", "hierarchical"] = "sequential"
    manager_agent_role: Optional[str] = Field(None, description="Role of the manager agent if hierarchical")

# --- 2. THE ARCHITECT (LLM ENGINE) ---
class CrewArchitect:
    def __init__(self, api_key: str, model_name: str = 'gemini-flash-latest'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def design_crew(self, prompt: str) -> CrewStructure:
        # Generate the JSON schema from the Pydantic model
        schema_json = json.dumps(CrewStructure.model_json_schema(), indent=2)
        
        system_prompt = f"""
You are a Principal Software Architect specializing in Agentic AI using the CrewAI framework.
Your goal is to translate a high-level natural language description of a workflow into a detailed, executable CrewAI architecture.

### RESPONSIBILITIES:
1. **Decompose** the workflow into specialized Agents with distinct roles, goals, and backstories.
2. **Define** precise Tasks with clear descriptions and expected outputs. Assign them to the most suitable Agent.
3. **Equip** Agents with Tools. 
   - Use standard CrewAI tools where possible (e.g., 'SerperDevTool' for search, 'ScrapeWebsiteTool' for scraping).
   - If a specific capability is needed (e.g., "Analyze CSV", "Query Custom API"), mark `is_custom=True` and provide the ACTUAL Python implementation using the `@tool` decorator.
   - **CRITICAL**: The 'tools' field must be a LIST of ToolConfig OBJECTS, not just strings.
4. **Identify** Knowledge sources. If the user mentions "using the company handbook" or "reading a website", define a Knowledge source.
5. **Structure** the Flow. Determine dependencies between tasks (`context_from`).

### SCHEMA (STRICT ENFORCEMENT):
You must output a single JSON object that strictly adheres to the following JSON Schema:
{schema_json}

### OUTPUT FORMAT:
Return ONLY valid JSON. Do not include markdown code blocks.
"""
        
        try:
            response = self.model.generate_content(
                f"{system_prompt}\n\n### USER REQUEST:\n{prompt}",
                generation_config={"response_mime_type": "application/json"}
            )
            return CrewStructure(**json.loads(response.text))
        except Exception as e:
            st.error(f"Architect Error: {e}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            return None

    def refine_crew(self, current_structure: CrewStructure, prompt: str) -> CrewStructure:
        schema_json = json.dumps(CrewStructure.model_json_schema(), indent=2)
        current_json = current_structure.model_dump_json(indent=2)
        
        system_prompt = f"""
You are a Principal Software Architect. You are REFINING an existing CrewAI architecture based on user feedback.

### CURRENT ARCHITECTURE (JSON):
{current_json}

### USER CHANGE REQUEST:
{prompt}

### INSTRUCTIONS:
1. Analyze the request and the current structure.
2. Apply the requested changes (add/remove/modify agents, tasks, tools, etc.).
3. Maintain the integrity of the rest of the structure.
4. Ensure 'tools' remains a list of ToolConfig objects.

### SCHEMA (STRICT ENFORCEMENT):
You must output a single JSON object that strictly adheres to the following JSON Schema:
{schema_json}

### OUTPUT FORMAT:
Return ONLY valid JSON.
"""
        try:
            response = self.model.generate_content(
                system_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return CrewStructure(**json.loads(response.text))
        except Exception as e:
            st.error(f"Refinement Error: {e}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            return None

# --- 3. THE VISUALIZER (DIAGRAM ENGINE) ---
def render_crew_diagram(crew: CrewStructure):
    """Generates a high-quality Graphviz diagram representing the CrewAI Flow."""
    dot = "digraph G {\n"
    # Global Graph Settings for "Opal-like" clarity
    dot += '  rankdir=LR;\n' # Left to Right flow
    dot += '  splines=ortho;\n' # Orthogonal lines for neatness
    dot += '  nodesep=0.8;\n'
    dot += '  ranksep=1.2;\n'
    dot += '  fontname="Sans-Serif";\n'
    
    # Defaults
    dot += '  node [fontname="Sans-Serif", fontsize=10, shape=box, style="rounded,filled", penwidth=1.5];\n'
    dot += '  edge [fontname="Sans-Serif", fontsize=9, color="#5f6368"];\n'
    
    # Legend / Subgraph for Agents
    with_knowledge = False
    
    # 1. Define Agents (Blue Ellipses)
    for agent in crew.agents:
        agent_id = re.sub(r'\W+', '_', agent.role).lower()
        label = f"🧑‍💻 {agent.role}\n<I>{agent.goal[:30]}...</I>"
        dot += f'  "{agent_id}" [label=<{label}>, shape=ellipse, fillcolor="#E8F0FE", color="#1967D2"];\n'
        
        # Tools (Orange Hexagons) attached to Agents
        for tool in agent.tools:
            tool_id = f"tool_{agent_id}_{tool.name}"
            dot += f'  "{tool_id}" [label="🔧 {tool.name}", shape=hexagon, fillcolor="#FEF7E0", color="#F9AB00", fontsize=8];\n'
            dot += f'  "{tool_id}" -> "{agent_id}" [style=dotted, arrowbeat=none, len=0.5];\n'
            
        # Knowledge (Green Cylinders) attached to Agents
        for knowledge in agent.knowledge_sources:
            with_knowledge = True
            k_id = f"know_{agent_id}_{knowledge.title}"
            dot += f'  "{k_id}" [label="🧠 {knowledge.title}", shape=cylinder, fillcolor="#E6F4EA", color="#137333", fontsize=8];\n'
            dot += f'  "{k_id}" -> "{agent_id}" [style=dashed, color="#137333", arrowhead=none];\n'

    # 2. Define Tasks (Grey Boxes) and Flow
    previous_task_id = None
    
    # Create a subgraph for the Process Flow to keep it aligned
    dot += '  subgraph cluster_flow {\n'
    dot += '    label="Workflow Process";\n'
    dot += '    style=dashed;\n'
    dot += '    color="#dadce0";\n'
    
    for i, task in enumerate(crew.tasks):
        task_id = f"task_{i}"
        agent_id = re.sub(r'\W+', '_', task.agent_role).lower()
        
        # Task Node
        task_label = f"📋 {task.name}\n<B>Out:</B> {task.expected_output[:40]}..."
        dot += f'    "{task_id}" [label=<{task_label}>, fillcolor="#FFFFFF", color="#5f6368"];\n'
        
        # Edge: Agent executes Task
        # We use 'constraint=false' to avoid messing up the left-right flow of tasks
        dot += f'    "{agent_id}" -> "{task_id}" [label="executes", style=bold, color="#1967D2", constraint=false];\n'
        
        # Edge: Dependency (Task Flow)
        if task.context_from:
            # Explicit dependencies defined by LLM
            for ctx_task_name in task.context_from:
                # Find the index/id of that task - simple lookup
                for j, prev_t in enumerate(crew.tasks):
                    if prev_t.name == ctx_task_name:
                        prev_id = f"task_{j}"
                        dot += f'    "{prev_id}" -> "{task_id}" [label="context"];\n'
        elif previous_task_id and crew.process_type == "sequential":
            # Implicit sequential flow
            dot += f'    "{previous_task_id}" -> "{task_id}" [label="next"];\n'
        
        previous_task_id = task_id
        
    dot += '  }\n'
    dot += "}"
    return dot

# --- 4. THE CODE GENERATOR ---
def generate_crew_code(crew: CrewStructure):
    # 1. Imports
    code = """import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
# from crewai_tools import SerperDevTool, ScrapeWebsiteTool # Uncomment if used
"""
    
    # 2. Custom Tools
    code += "\n# --- CUSTOM TOOLS ---\n"
    has_custom_tools = False
    for agent in crew.agents:
        for tool in agent.tools:
            if tool.is_custom and tool.python_logic:
                code += f"{tool.python_logic}\n\n"
                has_custom_tools = True
    
    if not has_custom_tools:
        code += "# No custom tools defined.\n"

    # 3. Class Definition
    code += f"""
# --- CREW DEFINITION ---
@CrewBase
class {crew.project_name.replace(" ", "")}Crew():
    \"\"\"{crew.project_name} crew\"\"\"
    
    # Define file paths for config (optional, but good practice)
    # agents_config = 'config/agents.yaml'
    # tasks_config = 'config/tasks.yaml'
"""

    # 4. Agents
    for i, agent in enumerate(crew.agents):
        tool_names = [f"'{t.name}'" for t in agent.tools if not t.is_custom]
        # For custom tools, we need the function name. Assuming logic gives a function decorated with @tool
        # We'll try to extract the name or just use the tool name reference if it was defined above.
        # For simplicity in this prototype, we'll append the list of custom tool functions if they exist.
        custom_tool_refs = []
        for t in agent.tools:
             if t.is_custom:
                 # Heuristic: assume the tool function name is the snake_case version of the tool name
                 custom_tool_refs.append(t.name.lower().replace(" ", "_"))

        all_tools = tool_names + custom_tool_refs
        
        knowledge_setup = ""
        if agent.knowledge_sources:
            knowledge_setup = "knowledge_sources=[\n"
            for k in agent.knowledge_sources:
                if k.source_type == "text":
                    knowledge_setup += f"            StringKnowledgeSource(content='{k.content_or_path}', metadata={{'description': '{k.description}'}}),\n"
                elif k.source_type == "url":
                    # Placeholder for URL knowledge
                    knowledge_setup += f"            # URL Source: {k.content_or_path} ({k.description})\n"
            knowledge_setup += "        ],"

        code += f"""

    @agent
    def {agent.role.lower().replace(" ", "_").replace("-", "_")}(self) -> Agent:
        return Agent(
            role="{agent.role}",
            goal="{agent.goal}",
            backstory="{agent.backstory}",
            tools=[{', '.join(all_tools)}],
            {knowledge_setup}
            allow_delegation={agent.allow_delegation},
            verbose={agent.verbose}
        )
"""

    # 5. Tasks
    for i, task in enumerate(crew.tasks):
        agent_func = task.agent_role.lower().replace(" ", "_").replace("-", "_")
        context_list = [f"self.{t.lower().replace(' ', '_').replace('-', '_')}()" for t in task.context_from]
        
        context_arg = ""
        if context_list:
             context_arg = f",\n            context=[{', '.join(context_list)}]"
             
        code += f"""

    @task
    def {task.name.lower().replace(" ", "_").replace("-", "_")}(self) -> Task:
        return Task(
            description="{task.description}",
            expected_output="{task.expected_output}",
            agent=self.{agent_func}(){context_arg},
            async_execution={task.async_execution}
        )
"""

    # 6. Crew
    agent_funcs = [a.role.lower().replace(" ", "_").replace("-", "_") for a in crew.agents]
    task_funcs = [t.name.lower().replace(" ", "_").replace("-", "_") for t in crew.tasks]
    
    code += f"""

    @crew
    def crew(self) -> Crew:
        \"\"\"Creates the {crew.project_name} crew\"\"\"
        return Crew(
            agents=self.agents, # Automatically collected by @agent decorator
            tasks=self.tasks,   # Automatically collected by @task decorator
            process=Process.{crew.process_type},
            verbose=True,
        )
"""

    # 7. Flow (Entry Point)
    code += f"""
# --- FLOW ENTRY POINT ---
from crewai.flow.flow import Flow, start, listen

class {crew.project_name.replace(" ", "")}Flow(Flow):
    
    @start()
    def start_crew(self):
        print("Starting the {crew.project_name} Flow")
        # Instantiate the Crew class
        my_crew = {crew.project_name.replace(" ", "")}Crew().crew()
        result = my_crew.kickoff()
        print("Crew Finished")
        return result

def kick():
    flow = {crew.project_name.replace(" ", "")}Flow()
    flow.kickoff()

if __name__ == "__main__":
    kick()
"""
    return code

# --- UI LAYOUT ---
st.set_page_config(layout="wide", page_title="TauPal: CrewAI Architect")

# Sidebar for Config
with st.sidebar:
    st.title("🧩 TauPal")
    st.markdown("Your AI Architect for CrewAI Systems.")
    
    user_api_key = st.text_input("Gemini API Key", value=API_KEY, type="password")
    if user_api_key:
        API_KEY = user_api_key
        
    model_option = st.text_input(
        "Gemini Model", 
        value="gemini-flash-latest"
    )
        
    st.info("Tip: Describe your workflow in detail. Mention roles, specific tools needed, and any documents/knowledge the agents should have.")

st.markdown("## 💠 TauPal: Workflow to CrewAI Architecture")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Describe Workflow")
    user_prompt = st.text_area(
        "Natural Language Description", 
        "I want to create a newsletter generator. \n"
        "1. A 'Researcher' searches specifically for the latest AI news on TechCrunch and VentureBeat. Tools: Search.\n"
        "2. An 'Analyst' filters the news for high-impact stories and summarizes them. Knowledge: 'Guidelines.txt' containing style rules.\n"
        "3. A 'Writer' compiles the final markdown newsletter.",
        height=300
    )
    
    generate_btn = st.button("Generate Architecture", type="primary", use_container_width=True)

    if generate_btn:
        if not API_KEY:
            st.error("Please provide a Gemini API Key in the sidebar.")
        else:
            with st.spinner("🧠 Architecting your Crew..."):
                architect = CrewArchitect(API_KEY, model_name=model_option)
                structure = architect.design_crew(user_prompt)
                if structure:
                    st.session_state['structure'] = structure
                else:
                    st.error("Failed to generate structure. Please try again.")

with col2:
    if 'structure' in st.session_state:
        struct = st.session_state['structure']
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Visual Diagram", "💬 Refine / Edit", "💾 Crew Code", "📋 JSON Structure"])
        
        with tab1:
            st.subheader(f"Blueprint: {struct.project_name}")
            viz_code = render_crew_diagram(struct)
            st.graphviz_chart(viz_code, use_container_width=True)
            
        with tab2:
            st.subheader("Refine Architecture")
            
            # --- Natural Language Refinement ---
            st.markdown("#### 🗣️ Natural Language")
            refine_prompt = st.text_area("Describe changes (e.g., 'Add a QA agent', 'Change the writer's goal')", height=100)
            if st.button("Apply Refinement", type="primary"):
                if not API_KEY:
                    st.error("API Key required.")
                else:
                    with st.spinner("Refining structure..."):
                        architect = CrewArchitect(API_KEY, model_name=model_option)
                        new_struct = architect.refine_crew(struct, refine_prompt)
                        if new_struct:
                            st.session_state['structure'] = new_struct
                            st.rerun()

            st.divider()
            
            # --- Manual Editing (Structured Interaction) ---
            st.markdown("#### 🛠️ Manual Edit")
            
            # Edit Agents
            with st.expander("Edit Agents"):
                for i, agent in enumerate(struct.agents):
                    st.markdown(f"**Agent {i+1}: {agent.role}**")
                    new_goal = st.text_area(f"Goal ({agent.role})", agent.goal, key=f"goal_{i}")
                    if new_goal != agent.goal:
                        struct.agents[i].goal = new_goal
                        st.session_state['structure'] = struct
                        st.rerun()
            
            # Edit Tasks
            with st.expander("Edit Tasks"):
                for i, task in enumerate(struct.tasks):
                    st.markdown(f"**Task {i+1}: {task.name}**")
                    new_desc = st.text_area(f"Description ({task.name})", task.description, key=f"desc_{i}")
                    if new_desc != task.description:
                        struct.tasks[i].description = new_desc
                        st.session_state['structure'] = struct
                        st.rerun()

        with tab3:
            st.subheader("Generated Python Code")
            code = generate_crew_code(struct)
            st.code(code, language="python")
            st.download_button(
                "Download crew.py",
                data=code,
                file_name=f"{struct.project_name.replace(' ', '_').lower()}_crew.py",
                mime="text/x-python"
            )
            
        with tab4:
            st.json(struct.model_dump())
    else:
        st.info("Generated results will appear here.")
