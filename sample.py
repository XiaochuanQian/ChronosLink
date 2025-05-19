import uuid
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from langgraph.prebuilt import create_react_agent


OPENAI_API_KEY = ""
OPENAI_API_BASE_URL = "https://api.chatanywhere.tech"

llm = init_chat_model(
    model= "gpt-4o-mini-2024-07-18",
    model_provider="openai",
    api_key = OPENAI_API_KEY,
    base_url=OPENAI_API_BASE_URL,
    temperature=0.0
)

def chat(message, history, thread_id):
    # Create a new thread id if not already present
    if not thread_id:
        thread_id = str(uuid.uuid4())

    # Set up the config using the session-specific thread id
    config = {"configurable": {"thread_id": thread_id}}

    # Append the user's message and a placeholder for the bot's response to the chat history
    history = history + [(message, "")]
    response_index = len(history) - 1  # Index of the bot's response in history

    full_response = ""
    # Stream the output from the backend in chunks
    for chunk, metadata in agent_executor.stream(
        {"messages": [HumanMessage(message)]},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):
            full_response += chunk.content
            # Update the last chat tuple with the new partial response
            history[response_index] = (message, full_response)
            # Yield the updated chat history (this will update the Gradio UI)
            yield history, thread_id

# Build the Gradio interface using a State component for the thread id
with gr.Blocks() as demo:
    # gr.State() holds the unique thread id across user interactions
    thread_state = gr.State()
    chatbot = gr.Chatbot()
    msg_input = gr.Textbox(placeholder="Type your message here...")
    send_btn = gr.Button("Send")

    # The click event calls our streaming chat function, which yields ongoing updates
    send_btn.click(
        chat,
        inputs=[msg_input, chatbot, thread_state],
        outputs=[chatbot, thread_state],
    )

    # TODO: add another interaction on input submits

    # TODO: clear input box

demo.launch()