import contextlib
from utils.model_loader import ModelLoader
from prompt_library.prompts import SUPERVISOR_PROMPT, CODER_PROMPT, RESEARCHER_PROMPT, AGENT2_PROMPT, Writer_PROMPT
from langgraph.graph import StateGraph, START, END
from langgraph_sdk.runtime import ServerRuntime
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode, tools_condition
from tools.weather_info_tool import WeatherInfoTool
from tools.place_search_tool import PlaceSearchTool
from tools.web_search_tool import WebSearchTool
from agents.state import AgentState  

class GraphBuilder:
     def __init__(self, model_provider: str = "groq"):
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()

        # Initialize tools
        self.weather_tool = WeatherInfoTool()
        self.place_search_tool = PlaceSearchTool()
        self.web_search_tool = WebSearchTool()
        self.tools = [
            *self.weather_tool.weather_tool_list,
            *self.place_search_tool.place_search_tool_list,
            *self.web_search_tool.web_search_tool_list,
        ]

        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)

        # Prompts
        self.supervisor_prompt = SUPERVISOR_PROMPT
        self.agent2_prompt = AGENT2_PROMPT
        self.researcher_prompt = RESEARCHER_PROMPT
        self.coder_prompt = CODER_PROMPT
        self.writer_prompt = Writer_PROMPT

     async def supervisor_agent(self, state: AgentState):  # ← AgentState
        messages = state["messages"]
        response = await self.llm.ainvoke([self.supervisor_prompt] + messages)
        decision = response.content.strip().lower()
        print(f"=== SUPERVISOR: {decision} ===")

        if "agent1" in decision:
            next_agent = "agent1"
        elif "agent2" in decision:
            next_agent = "agent2"
        else:
            next_agent = "end"
            print(f"=== ROUTING TO END — LLM SAID: {decision} ===")
        return {"next_agent": next_agent, "messages": messages}

     async def agent1_function(self, state: AgentState):  # ← AgentState
        user_question = state["messages"]
        response = await self.llm_with_tools.ainvoke(user_question)
        return {"messages": [response]}

     async def agent2_function(self, state: AgentState):  # ← AgentState
        messages = state["messages"]
        response = await self.llm.ainvoke([self.agent2_prompt] + messages)
        decision = response.content.strip().lower()
        print(f"=== AGENT2: {decision} ===")

        if "researcher" in decision:
            next_agent = "researcher"
        else:
            next_agent = "coder"

        return {"next_agent": next_agent, "messages": messages}

     async def researcher_function(self, state: AgentState):  # ← AgentState
        user_question = state["messages"]
        researcher_llm = self.llm.bind_tools(tools=self.web_search_tool.web_search_tool_list)
        response = await researcher_llm.ainvoke([self.researcher_prompt] + user_question)
        return {"messages": [response], "next_agent": "writer"}

     async def coder_function(self, state: AgentState):  # ← AgentState
        user_question = state["messages"]
        response = await self.llm.ainvoke([self.coder_prompt] + user_question)
        return {"messages": [response], "next_agent": "end"}

     async def writer_function(self, state: AgentState):  # ← AgentState
        messages = state["messages"]
        response = await self.llm.ainvoke([self.writer_prompt] + messages)
        return {"messages": [response], "next_agent": "end"}

     def route(self, state: AgentState) -> str:  # ← AgentState
        next_agent = state.get("next_agent", "end")
        print(f"=== ROUTING TO: {next_agent} ===")
        return next_agent

     def build_graph(self):
        """Build the graph."""
        graph = StateGraph(AgentState)
        graph.add_node("supervisor", self.supervisor_agent)  # Wrap async function
        graph.add_node("agent1", self.agent1_function)  # Wrap async function
        graph.add_node("agent2", self.agent2_function)  # Wrap async function
        graph.add_node("tools", ToolNode(tools=self.tools))
        graph.add_node("researcher", self.researcher_function)  # Wrap async function
        graph.add_node("coder", self.coder_function)  # Wrap async function
        graph.add_node("writer", self.writer_function)  # Wrap async function
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges("supervisor", self.route, {
            "agent1": "agent1",
            "agent2": "agent2",
            "end": END,
        })
        graph.add_conditional_edges("agent2", self.route, {
            "researcher": "researcher",
            "coder": "coder",
        })
        graph.add_conditional_edges("agent1",tools_condition)
        graph.add_edge("tools","agent1")
        graph.add_edge("researcher", "writer")
        graph.add_edge("coder", END)
        graph.add_edge("writer", END)
        self.graph = graph.compile()
        return self.graph

     @contextlib.asynccontextmanager
     async def make_graph(self, runtime: ServerRuntime):
        """Factory function to create the graph."""
        if runtime.execution_runtime:
            # Set up resources for execution
            yield self.build_graph()
        else:
            # Return a graph without expensive resources for introspection
            yield self.build_graph()

     def __call__(self):
        """Make the GraphBuilder callable."""
        return self.build_graph()