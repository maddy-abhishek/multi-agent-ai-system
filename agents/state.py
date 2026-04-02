from langgraph.graph import MessagesState
from typing import Optional

class AgentState(MessagesState):
    next_agent: Optional[str] = None