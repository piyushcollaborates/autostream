# agent.py - LangGraph agent for AutoStream

import os
import re
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.rag import get_context
from src.tools import collected_lead, mock_lead_capture, reset_lead, is_lead_complete

load_dotenv()

# ─────────────────────────────────────────
# 1. STATE DEFINITION
# ─────────────────────────────────────────
class AgentState(TypedDict):
    messages: List
    intent: str
    collecting_lead: bool
    lead_step: str

# ─────────────────────────────────────────
# 2. LLM SETUP
# ─────────────────────────────────────────
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

# ─────────────────────────────────────────
# 3. INTENT DETECTION NODE
# ─────────────────────────────────────────
def detect_intent(state: AgentState) -> AgentState:
    llm = get_llm()
    last_message = state["messages"][-1].content

    prompt = f"""You are an intent classifier for AutoStream, a SaaS video editing tool.

Classify the following user message into exactly one of these intents:
- greeting        : casual hello, hi, how are you, general chat
- inquiry         : asking about product, pricing, features, plans, policies
- high_intent     : ready to sign up, wants to try, wants to buy, wants to start

User message: "{last_message}"

Reply with only one word: greeting, inquiry, or high_intent"""

    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().lower()

    if intent not in ["greeting", "inquiry", "high_intent"]:
        intent = "inquiry"

    return {**state, "intent": intent}

# ─────────────────────────────────────────
# 4. ROUTER
# ─────────────────────────────────────────
def route_intent(state: AgentState) -> str:
    if state["collecting_lead"]:
        return "collect_lead"
    if state["intent"] == "greeting":
        return "handle_greeting"
    if state["intent"] == "inquiry":
        return "handle_inquiry"
    if state["intent"] == "high_intent":
        return "handle_high_intent"
    return "handle_inquiry"

# ─────────────────────────────────────────
# 5. GREETING NODE
# ─────────────────────────────────────────
def handle_greeting(state: AgentState) -> AgentState:
    llm = get_llm()
    messages = [
        SystemMessage(content="""You are a friendly sales assistant for AutoStream,
an AI-powered video editing SaaS for content creators.
Greet the user warmly and let them know you can help with
pricing, features, and getting started."""),
    ] + state["messages"]

    response = llm.invoke(messages)
    new_messages = state["messages"] + [AIMessage(content=response.content)]
    return {**state, "messages": new_messages}

# ─────────────────────────────────────────
# 6. INQUIRY NODE (RAG)
# ─────────────────────────────────────────
def handle_inquiry(state: AgentState) -> AgentState:
    llm = get_llm()
    last_message = state["messages"][-1].content
    context = get_context(last_message)

    messages = [
        SystemMessage(content=f"""You are a helpful sales assistant for AutoStream,
an AI-powered video editing SaaS for content creators.
Answer the user's question using ONLY the context below.
Be clear, friendly and concise.

CONTEXT:
{context}"""),
    ] + state["messages"]

    response = llm.invoke(messages)
    new_messages = state["messages"] + [AIMessage(content=response.content)]
    return {**state, "messages": new_messages}

# ─────────────────────────────────────────
# 7. HIGH INTENT NODE
# ─────────────────────────────────────────
def handle_high_intent(state: AgentState) -> AgentState:
    llm = get_llm()
    context = get_context("pro plan pricing features")

    messages = [
        SystemMessage(content=f"""You are a sales assistant for AutoStream.
The user is interested in signing up. Acknowledge their interest enthusiastically.
Briefly confirm the Pro plan details from context, then let them know
you will need a few details to get them started.
Ask for their name first.

CONTEXT:
{context}"""),
    ] + state["messages"]

    response = llm.invoke(messages)
    new_messages = state["messages"] + [AIMessage(content=response.content)]
    reset_lead()
    return {
        **state,
        "messages": new_messages,
        "collecting_lead": True,
        "lead_step": "name"
    }

# ─────────────────────────────────────────
# 8. LEAD COLLECTION NODE (with extraction)
# ─────────────────────────────────────────
def extract_email(text: str) -> str:
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else text.strip()

def extract_name(text: str) -> str:
    text = text.strip()
    # Remove common filler phrases
    for phrase in ["my name is", "i am", "i'm", "this is", "name is"]:
        if phrase in text.lower():
            idx = text.lower().index(phrase) + len(phrase)
            text = text[idx:].strip()
            break
    # Take only first 4 words max
    words = text.split()
    return " ".join(words[:4]).strip(".,!?")

def extract_platform(text: str) -> str:
    platforms = ["youtube", "instagram", "tiktok", "twitter", "facebook",
                 "linkedin", "twitch", "snapchat", "pinterest"]
    text_lower = text.lower()
    for platform in platforms:
        if platform in text_lower:
            return platform.capitalize()
    # If no known platform found, just return cleaned input
    return text.strip().split()[0].capitalize()

def collect_lead(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content.strip()
    lead_step = state["lead_step"]
    response_text = ""

    if lead_step == "name":
        name = extract_name(last_message)
        collected_lead["name"] = name
        response_text = f"Nice to meet you, {name}! What's your email address?"
        next_step = "email"

    elif lead_step == "email":
        email = extract_email(last_message)
        collected_lead["email"] = email
        response_text = "Got it! Which creator platform do you primarily use? (e.g. YouTube, Instagram, TikTok)"
        next_step = "platform"

    elif lead_step == "platform":
        platform = extract_platform(last_message)
        collected_lead["platform"] = platform
        next_step = "done"

        if is_lead_complete():
            result = mock_lead_capture(
                collected_lead["name"],
                collected_lead["email"],
                collected_lead["platform"]
            )
            response_text = f"""Perfect! You're all set. 🎉

Here's a summary of what we captured:
- Name     : {collected_lead['name']}
- Email    : {collected_lead['email']}
- Platform : {collected_lead['platform']}

Our team will reach out to you shortly to get your AutoStream Pro account set up.
Welcome aboard!"""
        else:
            response_text = "Something went wrong collecting your details. Please try again."

    else:
        response_text = "You're already registered! Our team will be in touch soon."
        next_step = "done"

    new_messages = state["messages"] + [AIMessage(content=response_text)]
    collecting = next_step != "done"

    return {
        **state,
        "messages": new_messages,
        "collecting_lead": collecting,
        "lead_step": next_step
    }

# ─────────────────────────────────────────
# 9. BUILD THE LANGGRAPH
# ─────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("detect_intent", detect_intent)
    graph.add_node("handle_greeting", handle_greeting)
    graph.add_node("handle_inquiry", handle_inquiry)
    graph.add_node("handle_high_intent", handle_high_intent)
    graph.add_node("collect_lead", collect_lead)

    graph.set_entry_point("detect_intent")

    graph.add_conditional_edges(
        "detect_intent",
        route_intent,
        {
            "handle_greeting": "handle_greeting",
            "handle_inquiry": "handle_inquiry",
            "handle_high_intent": "handle_high_intent",
            "collect_lead": "collect_lead"
        }
    )

    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_inquiry", END)
    graph.add_edge("handle_high_intent", END)
    graph.add_edge("collect_lead", END)

    return graph.compile()