import json
import uuid
from datetime import datetime
import streamlit as st
import google.generativeai as genai

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import LayeredLayout, ManualLayout

# --- CONFIGURATION ---
API_KEY = ""
try:
    if "gemini_api_key" in st.secrets:
        API_KEY = st.secrets["gemini_api_key"]
except:
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

# --- 3. THE CONVERTERS (STRUCTURE <-> FLOW) ---

def crew_to_flow(crew: CrewStructure):
    nodes = []
    edges = []
    
    agent_id_map = {} # role -> uuid
    
    # 1. Agents
    for agent in crew.agents:
        u_id = str(uuid.uuid4())
        agent_id_map[agent.role] = u_id
        
        nodes.append(StreamlitFlowNode(
            id=u_id,
            pos=(0, 0), # Layout will handle this
            data={
                'label': f"🧑‍💻 {agent.role}", 
                'type': 'agent',
                'role': agent.role,
                'goal': agent.goal,
                'backstory': agent.backstory,
                'allow_delegation': agent.allow_delegation,
                'verbose': agent.verbose
            },
            node_type="default",
            style={'background': '#E8F0FE', 'border': '1px solid #1967D2', 'color': '#000', 'width': 200}
        ))
        
        # Tools attached to Agent
        for tool in agent.tools:
            t_id = str(uuid.uuid4())
            nodes.append(StreamlitFlowNode(
                id=t_id,
                pos=(0, 0),
                data={
                    'label': f"🔧 {tool.name}",
                    'type': 'tool',
                    'name': tool.name,
                    'description': tool.description,
                    'is_custom': tool.is_custom,
                    'python_logic': tool.python_logic
                },
                node_type="input", # Just visual distinction
                style={'background': '#FEF7E0', 'border': '1px solid #F9AB00', 'color': '#000', 'width': 150}
            ))
            edges.append(StreamlitFlowEdge(
                id=str(uuid.uuid4()),
                source=t_id,
                target=u_id,
                animated=False,
                style={'stroke': '#F9AB00', 'strokeDasharray': '5,5'}
            ))

        # Knowledge attached to Agent
        for know in agent.knowledge_sources:
            k_id = str(uuid.uuid4())
            nodes.append(StreamlitFlowNode(
                id=k_id,
                pos=(0, 0),
                data={
                    'label': f"🧠 {know.title}",
                    'type': 'knowledge',
                    'title': know.title,
                    'source_type': know.source_type,
                    'content_or_path': know.content_or_path,
                    'description': know.description
                },
                node_type="input",
                style={'background': '#E6F4EA', 'border': '1px solid #137333', 'color': '#000', 'width': 150}
            ))
            edges.append(StreamlitFlowEdge(
                id=str(uuid.uuid4()),
                source=k_id,
                target=u_id,
                animated=False,
                style={'stroke': '#137333', 'strokeDasharray': '5,5'}
            ))

    # 2. Tasks
    task_name_map = {} # name -> uuid
    for task in crew.tasks:
        t_id = str(uuid.uuid4())
        task_name_map[task.name] = t_id
        
        nodes.append(StreamlitFlowNode(
            id=t_id,
            pos=(0, 0),
            data={
                'label': f"📋 {task.name}",
                'type': 'task',
                'name': task.name,
                'description': task.description,
                'expected_output': task.expected_output,
                'async_execution': task.async_execution,
                'agent_role': task.agent_role
            },
            node_type="default",
            style={'background': '#FFFFFF', 'border': '1px solid #5f6368', 'color': '#000', 'width': 220}
        ))
        
        # Link Agent to Task (Execution)
        if task.agent_role in agent_id_map:
            edges.append(StreamlitFlowEdge(
                id=str(uuid.uuid4()),
                source=agent_id_map[task.agent_role],
                target=t_id,
                animated=True,
                label="executes",
                style={'stroke': '#1967D2', 'strokeWidth': 2}
            ))

    # 3. Task Dependencies (Context)
    for task in crew.tasks:
        if task.context_from:
            current_id = task_name_map.get(task.name)
            for ctx_name in task.context_from:
                prev_id = task_name_map.get(ctx_name)
                if current_id and prev_id:
                     edges.append(StreamlitFlowEdge(
                        id=str(uuid.uuid4()),
                        source=prev_id,
                        target=current_id,
                        animated=True,
                        label="context",
                        style={'stroke': '#5f6368'}
                    ))

    return nodes, edges

def flow_to_crew(nodes: List[StreamlitFlowNode], edges: List[StreamlitFlowEdge], original_structure: CrewStructure) -> CrewStructure:
    agents: List[AgentConfig] = []
    tasks: List[TaskConfig] = []
    
    # Maps for reconstruction
    node_map = {node.id: node for node in nodes}
    agent_role_map = {} # node_id -> role
    
    # 1. Collect Agents
    for node in nodes:
        if node.data.get('type') == 'agent':
            role = node.data.get('role', 'New Agent')
            agent = AgentConfig(
                role=role,
                goal=node.data.get('goal', 'Goal...'),
                backstory=node.data.get('backstory', 'Backstory...'),
                allow_delegation=node.data.get('allow_delegation', False),
                verbose=node.data.get('verbose', True),
                tools=[],
                knowledge_sources=[]
            )
            agents.append(agent)
            agent_role_map[node.id] = role

    # 2. Collect Tools and Knowledge (connected to Agents)
    for edge in edges:
        source_node = node_map.get(edge.source)
        target_node = node_map.get(edge.target)
        
        if not source_node or not target_node:
            continue
            
        # Tool -> Agent
        if source_node.data.get('type') == 'tool' and target_node.data.get('type') == 'agent':
            # Find the agent config object
            target_role = target_node.data.get('role')
            agent_obj = next((a for a in agents if a.role == target_role), None)
            if agent_obj:
                tool = ToolConfig(
                    name=source_node.data.get('name', 'New Tool'),
                    description=source_node.data.get('description', ''),
                    is_custom=source_node.data.get('is_custom', False),
                    python_logic=source_node.data.get('python_logic', None)
                )
                agent_obj.tools.append(tool)

        # Knowledge -> Agent
        if source_node.data.get('type') == 'knowledge' and target_node.data.get('type') == 'agent':
            target_role = target_node.data.get('role')
            agent_obj = next((a for a in agents if a.role == target_role), None)
            if agent_obj:
                know = KnowledgeConfig(
                    title=source_node.data.get('title', 'New Knowledge'),
                    source_type=source_node.data.get('source_type', 'text'),
                    content_or_path=source_node.data.get('content_or_path', ''),
                    description=source_node.data.get('description', '')
                )
                agent_obj.knowledge_sources.append(know)

    # 3. Collect Tasks
    task_id_map = {} # node_id -> task_name
    
    for node in nodes:
        if node.data.get('type') == 'task':
            name = node.data.get('name', 'New Task')
            task_id_map[node.id] = name
            
            # Find execution agent from edges
            assigned_agent_role = "Unassigned"
            for edge in edges:
                if edge.target == node.id:
                    src = node_map.get(edge.source)
                    if src and src.data.get('type') == 'agent':
                        assigned_agent_role = src.data.get('role')
                        break
            
            # If not connected via edge, fallback to data property or first agent
            if assigned_agent_role == "Unassigned":
                assigned_agent_role = node.data.get('agent_role', agents[0].role if agents else "Placeholder")

            task = TaskConfig(
                name=name,
                description=node.data.get('description', ''),
                expected_output=node.data.get('expected_output', ''),
                async_execution=node.data.get('async_execution', False),
                agent_role=assigned_agent_role,
                context_from=[]
            )
            tasks.append(task)

    # 4. Task Dependencies
    for edge in edges:
        source_node = node_map.get(edge.source)
        target_node = node_map.get(edge.target)
        
        if source_node and target_node and source_node.data.get('type') == 'task' and target_node.data.get('type') == 'task':
             # Task -> Task (Context)
             target_task_name = task_id_map.get(target_node.id)
             source_task_name = task_id_map.get(source_node.id)
             
             task_obj = next((t for t in tasks if t.name == target_task_name), None)
             if task_obj and source_task_name:
                 if source_task_name not in task_obj.context_from:
                     task_obj.context_from.append(source_task_name)
    
    return CrewStructure(
        project_name=original_structure.project_name,
        agents=agents,
        tasks=tasks,
        process_type=original_structure.process_type,
        manager_agent_role=original_structure.manager_agent_role
    )


# --- 4. THE CODE GENERATOR (SAME AS BEFORE) ---
def generate_crew_code(crew: CrewStructure):
    code = """import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
"""
    code += "\n# --- CUSTOM TOOLS ---\n"
    has_custom_tools = False
    for agent in crew.agents:
        for tool in agent.tools:
            if tool.is_custom and tool.python_logic:
                code += f"{tool.python_logic}\n\n"
                has_custom_tools = True
    if not has_custom_tools:
        code += "# No custom tools defined.\n"

    code += f"""

# --- CREW DEFINITION ---
@CrewBase
class {crew.project_name.replace(" ", "")}Crew():
    '''{crew.project_name} crew'''
"""
    for i, agent in enumerate(crew.agents):
        tool_names = [f"'{t.name}'" for t in agent.tools if not t.is_custom]
        custom_tool_refs = []
        for t in agent.tools:
             if t.is_custom:
                 custom_tool_refs.append(t.name.lower().replace(" ", "_"))
        all_tools = tool_names + custom_tool_refs
        
        knowledge_setup = ""
        if agent.knowledge_sources:
            knowledge_setup = "knowledge_sources=[\n"
            for k in agent.knowledge_sources:
                if k.source_type == "text":
                    knowledge_setup += f"            StringKnowledgeSource(content='{k.content_or_path}', metadata={{'description': '{k.description}'}}),\n"
                elif k.source_type == "url":
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
    code += f"""
    @crew
    def crew(self) -> Crew:
        '''Creates the {crew.project_name} crew'''
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.{crew.process_type},
            verbose=True,
        )

# --- FLOW ENTRY POINT ---
from crewai.flow.flow import Flow, start, listen

class {crew.project_name.replace(" ", "")}Flow(Flow):
    @start()
    def start_crew(self):
        print("Starting the {crew.project_name} Flow")
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

with st.sidebar:
    st.title("🧩 TauPal")
    st.markdown("Your AI Architect for CrewAI Systems.")
    
    user_api_key = st.text_input("Gemini API Key", value=API_KEY, type="password")
    if user_api_key:
        API_KEY = user_api_key
        
    model_option = st.text_input("Gemini Model", value="gemini-flash-latest")
    st.info("Tip: Describe your workflow in detail.")

st.markdown("## 💠 TauPal: Workflow to CrewAI Architecture")

if 'structure' not in st.session_state:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("1. Describe Workflow")
        user_prompt = st.text_area(
            "Natural Language Description", 
            "I want to create a newsletter generator. \n" 
            "1. A 'Researcher' searches specifically for the latest AI news. Tools: Search.\n" 
            "2. An 'Analyst' filters the news and summarizes them.\n" 
            "3. A 'Writer' compiles the final markdown newsletter.",
            height=300
        )
        generate_btn = st.button("Generate Architecture", type="primary", use_container_width=True)

        if generate_btn:
            if not API_KEY:
                st.error("Please provide a Gemini API Key.")
            else:
                with st.spinner("🧠 Architecting your Crew..."):
                    architect = CrewArchitect(API_KEY, model_name=model_option)
                    structure = architect.design_crew(user_prompt)
                    if structure:
                        st.session_state['structure'] = structure
                        # Initial Flow Generation
                        nodes, edges = crew_to_flow(structure)
                        st.session_state['flow_nodes'] = nodes
                        st.session_state['flow_edges'] = edges
                        st.session_state['flow_key'] = str(uuid.uuid4()) # Force redraw
                        st.rerun()
                    else:
                        st.error("Failed to generate structure.")
    with col2:
        st.info("Generated result will appear here.")

else:
    # Structure Exists - Show Tabs
    struct = st.session_state['structure']
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Interactive Canvas", "💬 AI Refinement", "💾 Code", "📋 JSON"])
    
    with tab1:
        col_canvas, col_props = st.columns([3, 1])
        
        with col_canvas:
            st.subheader(f"Blueprint: {struct.project_name}")
            
            # Toolbar
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            if t_col1.button("➕ Add Agent"):
                u_id = str(uuid.uuid4())
                st.session_state['flow_nodes'].append(StreamlitFlowNode(
                    id=u_id, pos=(100, 100), 
                    data={'label': 'New Agent', 'type': 'agent', 'role': 'New Agent', 'goal': '', 'backstory': ''},
                    node_type="default", style={'background': '#E8F0FE', 'border': '1px solid #1967D2', 'color': '#000', 'width': 200}
                ))
                st.session_state['flow_key'] = str(uuid.uuid4())
                st.rerun()
            if t_col2.button("➕ Add Task"):
                u_id = str(uuid.uuid4())
                st.session_state['flow_nodes'].append(StreamlitFlowNode(
                    id=u_id, pos=(150, 150), 
                    data={'label': 'New Task', 'type': 'task', 'name': 'New Task', 'description': '', 'expected_output': ''},
                    node_type="default", style={'background': '#FFFFFF', 'border': '1px solid #5f6368', 'color': '#000', 'width': 220}
                ))
                st.session_state['flow_key'] = str(uuid.uuid4())
                st.rerun()
            if t_col3.button("➕ Add Tool"):
                u_id = str(uuid.uuid4())
                st.session_state['flow_nodes'].append(StreamlitFlowNode(
                    id=u_id, pos=(50, 200), 
                    data={'label': 'New Tool', 'type': 'tool', 'name': 'New Tool', 'description': ''},
                    node_type="input", style={'background': '#FEF7E0', 'border': '1px solid #F9AB00', 'color': '#000', 'width': 150}
                ))
                st.session_state['flow_key'] = str(uuid.uuid4())
                st.rerun()
            if t_col4.button("➕ Add Knowledge"):
                u_id = str(uuid.uuid4())
                st.session_state['flow_nodes'].append(StreamlitFlowNode(
                    id=u_id, pos=(50, 250), 
                    data={'label': 'New Knowledge', 'type': 'knowledge', 'title': 'New Know.', 'source_type': 'text', 'content_or_path': ''},
                    node_type="input", style={'background': '#E6F4EA', 'border': '1px solid #137333', 'color': '#000', 'width': 150}
                ))
                st.session_state['flow_key'] = str(uuid.uuid4())
                st.rerun()

            # Canvas
            if 'flow_nodes' not in st.session_state:
                st.session_state['flow_nodes'], st.session_state['flow_edges'] = crew_to_flow(struct)
                st.session_state['flow_timestamp'] = int(datetime.now().timestamp() * 1000)

            # Check if this is a fresh generation to apply layout
            first_run = 'layout_applied' not in st.session_state
            layout = LayeredLayout(direction='right', node_node_spacing=50) if first_run else ManualLayout()
            if first_run: st.session_state['layout_applied'] = True

            flow_state = streamlit_flow(
                key=st.session_state.get('flow_key', 'flow_canvas'),
                state=StreamlitFlowState(
                    nodes=st.session_state['flow_nodes'],
                    edges=st.session_state['flow_edges'],
                    timestamp=st.session_state.get('flow_timestamp', 0)
                ),
                layout=layout,
                fit_view=first_run,
                get_node_on_click=True,
                get_edge_on_click=True
            )
            
            # Persist the state (positions, etc.)
            st.session_state['flow_nodes'] = flow_state.nodes
            st.session_state['flow_edges'] = flow_state.edges
            st.session_state['flow_timestamp'] = flow_state.timestamp

            selected_node_id = flow_state.selected_id
            
        with col_props:
            st.subheader("Properties")
            if selected_node_id:
                # Find the node
                sel_node = next((n for n in flow_state.nodes if n.id == selected_node_id), None)
                if sel_node:
                    st.write(f"**Editing: {sel_node.data.get('label', 'Node')}**")
                    node_type = sel_node.data.get('type')
                    
                    changed = False
                    new_data = sel_node.data.copy()
                    
                    if node_type == 'agent':
                        new_role = st.text_input("Role", new_data.get('role', ''))
                        new_goal = st.text_area("Goal", new_data.get('goal', ''))
                        if new_role != new_data['role'] or new_goal != new_data['goal']:
                            new_data['role'] = new_role
                            new_data['goal'] = new_goal
                            new_data['label'] = f"🧑‍💻 {new_role}"
                            changed = True
                            
                    elif node_type == 'task':
                        new_name = st.text_input("Name", new_data.get('name', ''))
                        new_desc = st.text_area("Description", new_data.get('description', ''))
                        if new_name != new_data['name'] or new_desc != new_data['description']:
                            new_data['name'] = new_name
                            new_data['description'] = new_desc
                            new_data['label'] = f"📋 {new_name}"
                            changed = True
                    
                    elif node_type == 'tool':
                        new_name = st.text_input("Name", new_data.get('name', ''))
                        new_desc = st.text_area("Description", new_data.get('description', ''))
                        if new_name != new_data['name'] or new_desc != new_data['description']:
                            new_data['name'] = new_name
                            new_data['description'] = new_desc
                            new_data['label'] = f"🔧 {new_name}"
                            changed = True

                    elif node_type == 'knowledge':
                        new_title = st.text_input("Title", new_data.get('title', ''))
                        new_content = st.text_area("Content", new_data.get('content_or_path', ''))
                        if new_title != new_data['title'] or new_content != new_data['content_or_path']:
                            new_data['title'] = new_title
                            new_data['content_or_path'] = new_content
                            new_data['label'] = f"🧠 {new_title}"
                            changed = True
                            
                    if changed:
                        # Update the specific node in session state
                        for i, n in enumerate(st.session_state['flow_nodes']):
                            if n.id == selected_node_id:
                                st.session_state['flow_nodes'][i].data = new_data
                                # Force redraw to update label
                                st.session_state['flow_key'] = str(uuid.uuid4())
                                st.rerun()

            else:
                st.info("Select a node to edit properties.")
            
            if st.button("Sync & Save Changes"):
                # Convert Flow back to Structure
                new_struct = flow_to_crew(flow_state.nodes, flow_state.edges, struct)
                st.session_state['structure'] = new_struct
                st.session_state['flow_nodes'] = flow_state.nodes # Persist positions
                st.session_state['flow_edges'] = flow_state.edges
                st.success("Structure Updated!")

    with tab2:
        st.subheader("AI Refinement")
        refine_prompt = st.text_area("Describe changes (e.g., 'Add a QA agent')", height=100)
        if st.button("Apply Refinement", type="primary"):
            if not API_KEY:
                st.error("API Key required.")
            else:
                with st.spinner("Refining..."):
                    architect = CrewArchitect(API_KEY, model_name=model_option)
                    new_struct = architect.refine_crew(struct, refine_prompt)
                    if new_struct:
                        st.session_state['structure'] = new_struct
                        nodes, edges = crew_to_flow(new_struct)
                        st.session_state['flow_nodes'] = nodes
                        st.session_state['flow_edges'] = edges
                        st.session_state['flow_key'] = str(uuid.uuid4())
                        st.rerun()

    with tab3:
        st.subheader("Generated Python Code")
        code = generate_crew_code(struct)
        st.code(code, language="python")
        st.download_button("Download crew.py", data=code, file_name="crew.py", mime="text/x-python")

    with tab4:
        st.json(struct.model_dump())