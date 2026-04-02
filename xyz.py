from utils.model_loader import ModelLoader
from prompt_library.prompts import SUPERVISOR_PROMPT, CODER_PROMPT, RESEARCHER_PROMPT, AGENT2_PROMPT, Writer_PROMPT
from langgraph.graph import StateGraph, MessagesState, END, START
from typing import TypedDict, Annotated, List, Literal, Dict, Any
from langgraph.prebuilt import ToolNode, tools_condition
from tools.weather_info_tool import WeatherInfoTool
from tools.place_search_tool import PlaceSearchTool
from tools.web_search_tool import WebSearchTool
from langgraph.checkpoint.memory import InMemorySaver

class GraphBuilder():
     def __init__(self,model_provider: str = "groq"):
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()

        self.tools = []
        self.weather_tool = WeatherInfoTool()
        self.place_search_tool = PlaceSearchTool()
        self.web_search_tool = WebSearchTool()

        self.tools.extend([* self.weather_tool.weather_tool_list, 
                           * self.place_search_tool.place_search_tool_list,
                           * self.web_search_tool.web_search_tool_list])
        
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)

        self.graph = None
        
        self.agent2_prompt = AGENT2_PROMPT
        self.researcher_prompt = RESEARCHER_PROMPT
        self.coder_prompt = CODER_PROMPT
        self.writer_prompt = Writer_PROMPT
        self.supervisor_prompt = SUPERVISOR_PROMPT

     async def agent1_function(self,state: MessagesState):
        """Agent 1 function"""
        user_question = state["messages"]
        response = await self.llm_with_tools.invoke_async(user_question)
        checkpointer=InMemorySaver()
        return {"messages": [response]}

     async def coder_function(self,state: MessagesState):
        """Coder agent with coding capabilities"""
        user_question = state["messages"]
        input_question = [self.coder_prompt] + user_question
        response = await self.llm.invoke_async(input_question)
        return {"messages": [response],
                "next_agent": "end"}

     async def researcher_function(self,state: MessagesState):
        """Researcher agent with web search capabilities"""
        user_question = state["messages"]
        input_question = [self.researcher_prompt] + user_question
        researcher_llm = self.llm.bind_tools(tools=self.web_search_tool)
        response = await researcher_llm.invoke_async(input_question)
        return {"messages": [response],
                "next_agent": "writer"}

     async def writer_function(self,state: MessagesState):
        """Writer agent with text generation capabilities"""
        messages = state["messages"]
        system_msg = [self.writer_prompt]
        response = await self.llm.invoke_async([system_msg] + messages)
        return {"messages": [response],
                "next_agent": "end"}

     async def supervisor_agent(self,state: MessagesState):
        """Supervisor decides which agent to route to based on user input"""

        messages = state["messages"]
        task = messages[-1].content if messages else "No task"

        prompt = [self.supervisor_prompt]
        res = await self.llm.invoke_async([prompt] + task)
        decision_text = res.content.strip().lower()

        if "agent1" in decision_text:
            next_agent = "agent1"
        elif "agent2" in decision_text:
            next_agent = "agent2"
        else:
            next_agent = "end"
        checkpointer=InMemorySaver()

        return {"next_agent": next_agent, "messages": messages}

     async def agent2_function(self,state: MessagesState):
        """Agent 2 function decides which agent to route based on user input"""

        messages = state["messages"]
        task = messages[-1].content if messages else "No task"

        prompt = [self.agent2_prompt]
        res = await self.llm.invoke_async([prompt] + task)
        decision_text = res.content.strip().lower()

        if "researcher" in decision_text:
            next_agent = "researcher"
        else:
            next_agent = "coder"
        checkpointer=InMemorySaver()

        return {"next_agent": next_agent, "messages": messages}

     def route(state: MessagesState) -> Literal["AGENT1", "AGENT2", "END"]:
         """Routes to next agent based on state"""
         
         next_agent = state.get("next_agent")

         if next_agent == "end":
            return END
         if next_agent in ["AGENT1", "AGENT2"]:
             return next_agent
         
     def route2(state: MessagesState) -> Literal["researcher", "coder"]:
         """Routes to next agent based on state"""
         
         next_agent = state.get("next_agent")

         return next_agent
     

     def build_graph(self):
        graph_builder=StateGraph(MessagesState)
        graph_builder.add_node("supervisor", self.supervisor_agent)
        graph_builder.add_node("agent1", self.agent1_function)
        graph_builder.add_node("agent2", self.agent2_function)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("researcher", self.researcher_function)
        graph_builder.add_node("coder", self.coder_function)
        graph_builder.add_node("writer", self.writer_function)
        graph_builder.add_edge(START,"supervisor")
        graph_builder.add_conditional_edges("supervisor",
                                            self.route,
                                            {
                                                "agent1": "agent1",
                                                "agent2": "agent2",
                                                "end": END
                                            }
                                            )
        graph_builder.add_conditional_edges("agent1",tools_condition)
        graph_builder.add_edge("tools","agent1")
        graph_builder.add_conditional_edges("agent2",
                                            self.route2,
                                            {
                                                "researcher": "researcher",
                                                "coder": "coder"
                                            }
                                            )
        graph_builder.add_edge("coder", "writer")
        graph_builder.add_edge("agent1",END)
        graph_builder.add_edge("researcher",END)
        graph_builder.add_edge("writer",END)
        self.graph = graph_builder.compile()
        return self.graph

     def __call__(self):
        return self.build_graph() 