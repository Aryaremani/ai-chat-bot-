import streamlit as st
import openai
from openai import OpenAI
import json
import time
from typing import Dict, List
import os

# Get OpenAI API key from environment or Streamlit secrets
# Then set up the OpenAI client using that key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# Simulated email sending function (logs email and stores it in session state)
def send_email(to: str, subject: str, body: str) -> str:
    print(f"Email to {to}: {subject}\n{body}")
    if 'sent_emails' not in st.session_state:
        st.session_state.sent_emails = []

    email_data = {
        'to': to,
        'subject': subject,
        'body': body,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    st.session_state.sent_emails.append(email_data)

    return f"Email sent successfully to {to} with subject: {subject}"

# Tell the assistant that it can call the send_email function when needed
email_tool = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email to a user",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body content"}
            },
            "required": ["to", "subject", "body"]
        }
    }
}

# Create a new assistant that only replies using the given training content
def create_assistant(training_content: str) -> str:
    instructions = f"""
    You are a helpful AI assistant that can ONLY answer questions based on the provided training content below. 

    STRICT RULES:
    1. You must ONLY use information from the training content provided
    2. If a question cannot be answered using the training content, respond with: "I'm sorry, I can only answer questions based on the provided training content."
    3. Never use your general knowledge - stick strictly to the training content
    4. You can send emails when requested using the send_email function

    TRAINING CONTENT:
    {training_content}

    Always be helpful and friendly while staying within these constraints.
    """

    try:
        assistant = client.beta.assistants.create(
            name="AskWithMe",
            description="CONTEXT-RESTRICTED AI ASSISTANT",
            instructions=instructions,
            model="gpt-4o-mini",
            tools=[email_tool]
        )
        return assistant.id
    except Exception as e:
        st.error(f"Error creating assistant: {str(e)}")
        return None

# Talk to the assistant: send user message, get response, and handle tool calls
def chat_with_assistant(assistant_id: str, message: str, thread_id: str = None) -> tuple:
    try:
        if thread_id is None:
            thread = client.beta.threads.create()
            thread_id = thread.id

        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message
        )

        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )

        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )

            if run_status.status == 'completed':
                break
            elif run_status.status == 'requires_action':
                # If the assistant wants to use a tool (like sending an email), do it here
                tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []

                for tool_call in tool_calls:
                    if tool_call.function.name == "send_email":
                        function_args = json.loads(tool_call.function.arguments)
                        result = send_email(**function_args)
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": result
                        })

                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                st.error(f"Run failed with status: {run_status.status}")
                return None, thread_id

            time.sleep(1)

        messages = client.beta.threads.messages.list(thread_id=thread_id)
        latest_message = messages.data[0]

        return latest_message.content[0].text.value, thread_id

    except Exception as e:
        st.error(f"Error chatting with assistant: {str(e)}")
        return None, thread_id

# Main Streamlit app logic
def main():
    st.set_page_config(page_title="AskWithMe - Context-Restricted AI Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 AskWithMe")
    st.markdown("*Powered by OpenAI Assistants API with Function Calling*")

    # Sidebar: for email input and viewing saved emails
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("OpenAI API Key is set in code for deployment.")

        st.header("📧 Email Capture")
        user_email = st.text_input("Your Email Address")
        if user_email and st.button("Save Email"):
            if 'user_emails' not in st.session_state:
                st.session_state.user_emails = []
            if user_email not in st.session_state.user_emails:
                st.session_state.user_emails.append(user_email)
                st.success("Email saved!")

        if 'user_emails' in st.session_state and st.session_state.user_emails:
            st.write("**Saved Emails:**")
            for email in st.session_state.user_emails:
                st.write(f"• {email}")

    col1, col2 = st.columns([1, 1])

    # Left side: where user inputs training data and creates assistant
    with col1:
        st.header("📚 Training Content")

        training_content = st.text_area(
            "Enter your training content:",
            height=300,
            placeholder="Enter FAQs, product details, or any content you want the AI to learn from..."
        )

        if st.button("💾 Save Context & Create Assistant", type="primary"):
            if training_content.strip():
                with st.spinner("Creating your custom assistant..."):
                    assistant_id = create_assistant(training_content)
                    if assistant_id:
                        st.session_state.assistant_id = assistant_id
                        st.session_state.training_content = training_content
                        st.session_state.thread_id = None
                        st.success("✅ Assistant created successfully!")
                        st.rerun()
            else:
                st.error("Please enter some training content first!")

    # Right side: chat UI with assistant
    with col2:
        st.header("💬 Chat with AI")

        if 'assistant_id' in st.session_state:
            st.success("🟢 Assistant is ready!")
            with st.expander("View Training Content"):
                st.text(st.session_state.training_content[:500] + "..." if len(st.session_state.training_content) > 500 else st.session_state.training_content)
        else:
            st.warning("⚠️ Please save training content first to start chatting")

        if 'assistant_id' in st.session_state:
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            chat_container = st.container()
            with chat_container:
                for i, (role, message) in enumerate(st.session_state.chat_history):
                    st.chat_message(role).write(message)

            user_message = st.chat_input("Ask me anything about the training content...")

            if user_message:
                st.session_state.chat_history.append(("user", user_message))

                with st.spinner("🤔 Thinking..."):
                    response, thread_id = chat_with_assistant(
                        st.session_state.assistant_id,
                        user_message,
                        st.session_state.get('thread_id')
                    )

                    if response:
                        st.session_state.thread_id = thread_id
                        st.session_state.chat_history.append(("assistant", response))
                        st.rerun()

    # Display sent emails below the chat interface
    st.header("📨 Email Log")
    if 'sent_emails' in st.session_state and st.session_state.sent_emails:
        for i, email in enumerate(st.session_state.sent_emails):
            with st.expander(f"Email {i+1}: {email['subject']} - {email['timestamp']}"):
                st.write(f"**To:** {email['to']}")
                st.write(f"**Subject:** {email['subject']}")
                st.write(f"**Body:** {email['body']}")
                st.write(f"**Sent:** {email['timestamp']}")
    else:
        st.info("No emails sent yet. Try asking the AI to send an email!")

    # Expandable section with usage instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        ### Steps to get started:

        1. **Add Training Content**: Enter your custom content (FAQs, product info, etc.) in the left panel  
        2. **Save Context**: Click "Save Context & Create Assistant" to create your AI  
        3. **Enter Email**: Add your email address in the sidebar  
        4. **Start Chatting**: Ask questions about your training content  
        5. **Test Email Function**: Ask the AI to send an email (e.g., "Send me a summary via email")

        ### Features:
        - ✅ Context-restricted responses (AI only uses your training data)
        - ✅ Function calling for email simulation
        - ✅ Email capture and storage
        - ✅ Real-time chat interface
        - ✅ Email logging and history

        ### Example prompts:
        - "What information do you have about [topic]?"
        - "Send an email to me with the details"
        """)

if __name__ == "__main__":
    main()
