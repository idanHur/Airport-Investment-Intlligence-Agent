"""The agent as an explicit LangGraph: one LLM node that may call tools, one tool node, loop until done.

    START -> agent --(tool calls?)--> tools --> agent ... --> END

State is just the message list. A checkpointer keyed by thread_id remembers the conversation,
which is what makes follow-up questions work.
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from airport_agent.agent.tools import TOOLS
from airport_agent.config import OPENAI_MODEL

SYSTEM_PROMPT = """You are an analyst assistant for a firm that invests in US airport modernization projects.

RULES
- Every number you state must come from a tool result. Never estimate, recall or invent figures. If the tools
  cannot answer, say so.
- Call find_airport first if you are unsure of an IATA code (e.g. Santa Ana is SNA, but check).
- Scores and rankings are computed by deterministic code. You explain them; you never re-rank or re-score.
- Always communicate scope and uncertainty: state the year, the definition used (e.g. what 'long haul' means),
  and any flags or notes returned by the tool (small airport, load-factor gate, sampled delay months, proxy-only
  metrics). Keep these short but do not omit them.
- 'Unmet demand' is a proxy, never a measurement. Say what it is built from, and explain WHY using the real
  values (load factor, growth vs 2019, delays), not just the proxy number.
- Values whose names end in 'percentile' (or scores 0-100) are ranks among peers, not percentages of anything.
  Report real values (load_factor, passengers_vs_2019, etc.) as real values, and ranks as ranks.
- A ratio below 1.0 vs 2019 means traffic is BELOW its pre-COVID level.
- If a tool result has a `summary` field, include its content in your answer (reword if you like, drop nothing).
- Never do arithmetic on tool numbers (no multiplying shares by counts); quote the figures the tool gives.
- If a tool returns a sensitivity table (e.g. long haul at different distance cut-offs), state the headline number
  AND how much it moves under the alternatives: a definition-dependent answer must be presented as such.
- When the user's place name maps to several airports (e.g. 'LA' could be LAX, BUR, LGB, ONT), say which one you
  used and offer the others.
- If the question is ambiguous (region, year, definition), pick the sensible default, say what you chose, and
  offer the alternative.
- For follow-ups, reuse earlier context; call tools again only when you need new numbers.
- Be concise: lead with the answer, then the reasoning, then caveats."""


def build_agent():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0).bind_tools(TOOLS)

    def call_model(state: MessagesState):
        return {"messages": [llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # -> "tools" if the LLM asked for tools, else END
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=InMemorySaver())


def ask(agent, question: str, thread_id: str = "default") -> str:
    """Send one user message; the checkpointer supplies earlier turns for this thread."""
    result = agent.invoke({"messages": [("user", question)]}, config={"configurable": {"thread_id": thread_id}})
    return result["messages"][-1].content
