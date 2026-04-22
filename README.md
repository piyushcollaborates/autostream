# AutoStream AI Agent
A conversational AI agent built for AutoStream, a fictional SaaS company providing
automated video editing tools for content creators. 
---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/autostream.git
cd autostream
```

### 2. Create a virtual environment
```bash
py -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory and add your Groq API key:
GROQ_API_KEY=your_groq_api_key_here

Get a free API key at https://console.groq.com

### 5. Run the agent
```bash
py main.py
```

---

## Architecture Explanation

This project uses LangGraph to build a stateful conversational AI agent. LangGraph
was chosen over plain LangChain because it allows explicit state management through
a graph of nodes, where each node represents a distinct behavior of the agent. This
makes the agent's decision-making transparent, predictable, and easy to extend.

The agent state is a typed dictionary that persists across all conversation turns,
tracking four things: the full message history, the detected intent, whether lead
collection is in progress, and which lead field is being collected next (name, email,
or platform). This state is passed between nodes on every turn, so the agent always
knows exactly where it is in the conversation.

The graph has five nodes: 
1.detect_intent (classifies every user message), 
2.handle_greeting (responds to casual messages),
3.handle_inquiry (uses RAG to answer product questions from knowledge_base.json), 
4.handle_high_intent (detects signup interest and starts lead collection), and 
5.collect_lead (collects name, email, and platform one field at a time). 

A conditional router after detect_intent decides which node to activate. 
The mock_lead_capture() tool is only triggered after all three lead fields are confirmed, preventing premature tool execution.

The LLM used is LLaMA 3.1 8B served via Groq, chosen for its speed, accuracy, and generous free tier limits.
However, it was first tested with gemini-2.5-flash and other equivalent models, but since it was hitting rate limit everytime I 
made only few requests, and also it was a bit slow than llama 3.1 8b. So i shifted to LLaMA just for the assignment purpose to test 
this agent if its working correctly or not. So LLaMA worked here for free.
---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, we would use the WhatsApp Business API provided
by Meta along with a webhook-based architecture.

Here is how it would work:

1. **Webhook Setup**: We deploy the agent as a web server using FastAPI or Flask.
   Meta sends incoming WhatsApp messages to our webhook URL as HTTP POST requests
   containing the user's phone number and message text.

2. **Message Handling**: When a POST request arrives, we extract the user's message
   and phone number, retrieve or create their conversation state from a database
   like Redis (so memory persists across messages), and pass the message into our
   LangGraph agent.

3. **Sending Replies**: After the agent generates a response, we call the WhatsApp
   Business API's messages endpoint with the reply text and the user's phone number
   to send the response back.

4. **State Persistence**: Since WhatsApp conversations are asynchronous (users can
   message hours apart), we store each user's AgentState in Redis keyed by their
   phone number. This replaces the in-memory state we use in the CLI version.

5. **Lead Capture**: When mock_lead_capture() fires, instead of printing to console
   we would POST the lead data to a CRM like HubSpot or save it to a database.

A simplified webhook handler would look like this:

```python
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    phone = data["from"]
    message = data["text"]["body"]
    
    # Load state from Redis
    state = redis.get(phone) or initial_state()
    
    # Run through LangGraph agent
    state["messages"].append(HumanMessage(content=message))
    state = graph.invoke(state)
    
    # Send reply via WhatsApp API
    reply = state["messages"][-1].content
    whatsapp_api.send_message(phone, reply)
    
    # Save updated state back to Redis
    redis.set(phone, state)
```

---

## Project Structure
autostream/
├── src/
│   ├── init.py     # Makes src a Python module
│   ├── agent.py        # LangGraph agent + state management
│   ├── rag.py          # RAG knowledge retrieval
│   └── tools.py        # Lead capture tool
├── knowledge_base.json # AutoStream pricing & policies
├── .env                # API keys (not committed to GitHub)
├── main.py             # Entry point
├── requirements.txt    # Dependencies
└── README.md           # This file
---

## Tech Stack

- **Language**: Python 3.14.4
- **Framework**: LangGraph + LangChain
- **LLM**: LLaMA 3.1 8B via Groq
- **RAG**: Local JSON knowledge base with keyword-based retrieval
- **State Management**: LangGraph StateGraph with TypedDict