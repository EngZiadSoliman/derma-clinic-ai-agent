from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from tools.prediction_tool import predict_customer_return_by_id
from tools.customer_data import lookup_customer
from tools.clinic_analytics import clinic_analytics


tools = [
    predict_customer_return_by_id,
    lookup_customer,
    clinic_analytics
]

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)



llm_with_tools = llm.bind_tools(tools)


def chatbot(state: MessagesState):
    messages = [
        SystemMessage(content="""
You are an AI assistant for a dermatology clinic.

Use lookup_customer to retrieve a customer's profile and visit history.

Use predict_customer_return_by_id to estimate a customer's
90-day return probability and evaluate discount options.

Use clinic_analytics to answer questions about clinic visits,
revenue, services, and customer acquisition channels.

Use the appropriate tool when the user asks for clinic data
or customer predictions.

Explain predictions as model estimates, not guaranteed outcomes.
The baseline and discount-response probabilities come from
separate models and must not be presented as a causal improvement.
"""),
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")

builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

builder.add_edge("tools", "chatbot")

graph = builder.compile()




import gradio as gr


def chat(message, history):
    response = graph.invoke({
        "messages": [("user", message)]
    })

    return response["messages"][-1].content


gr.ChatInterface(chat).launch(inbrowser=True)