import streamlit as st
import openai
from openai import OpenAI
import json
import time
from typing import Dict, List, Optional, Tuple
import os

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=API_KEY)

def simulate_email_sending(recipient: str, subject: str, message_body: str) -> str:
    """
    Simulates sending an email by storing it in session state.
    In a real app, this would integrate with an actual email service.
    """
    print(f"📧 Mock email sent to {recipient}: {subject}")
    
    # Initialize email storage if not exists
    if 'email_history' not in st.session_state:
        st.session_state.email_history = []

    # Store email details
    email_record = {
        'recipient': recipient,
        'subject': subject,
        'body': message_body,
        'sent_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    st.session_state.email_history.append(email_record)

    return f"Email successfully sent to {recipient}"

# OpenAI function definition for email capability
EMAIL_FUNCTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email to a specified recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string", 
                    "description": "Email address of the recipient"
                },
                "subject": {
                    "type": "string", 
                    "description": "Subject line for the email"
                },
                "body": {
                    "type": "string", 
                    "description": "Main content of the email"
                }
            },
            "required": ["to", "subject", "body"]
        }
    }
}

def build_custom_assistant(knowledge_base: str) -> Optional[str]:
    """
    Creates an OpenAI assistant with custom knowledge and email capabilities.
    Returns the assistant ID if successful, None otherwise.
    """
    system_prompt = f"""
    You are a specialized assistant that can only provide information from the knowledge base provided below.

    IMPORTANT GUIDELINES:
    - Only answer questions using the information in the knowledge base
    - If you don't have the information, politely say so
    - Don't use general knowledge outside of what's provided
    - You can send emails when users request it
    - Be conversational but stay within your knowledge constraints

    KNOWLEDGE BASE:
    {knowledge_base}

    Maintain a helpful and professional tone while respecting these limitations.
    """

    try:
        assistant = openai_client.beta.assistants.create(
            name="Custom Knowledge Assistant",
            description="Assistant with domain-specific knowledge",
            instructions=system_prompt,
            model="gpt-4o-mini",
            tools=[EMAIL_FUNCTION_SCHEMA]
        )
        return assistant.id
    except Exception as error:
        st.error(f"Failed to create assistant: {str(error)}")
        return None

def handle_conversation(assistant_id: str, user_input: str, conversation_thread: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Manages the conversation flow with the OpenAI assistant.
    Handles both regular responses and function calls (like email sending).
    """
    try:
        # Create new thread if none exists
        if conversation_thread is None:
            thread = openai_client.beta.threads.create()
            conversation_thread = thread.id

        # Add user message to the conversation
        openai_client.beta.threads.messages.create(
            thread_id=conversation_thread,
            role="user",
            content=user_input
        )

        # Start the assistant run
        run = openai_client.beta.threads.runs.create(
            thread_id=conversation_thread,
            assistant_id=assistant_id
        )

        # Monitor the run status
        while True:
            current_run = openai_client.beta.threads.runs.retrieve(
                thread_id=conversation_thread,
                run_id=run.id
            )

            if current_run.status == 'completed':
                break
            elif current_run.status == 'requires_action':
                # Handle function calls (like sending emails)
                function_calls = current_run.required_action.submit_tool_outputs.tool_calls
                function_results = []

                for call in function_calls:
                    if call.function.name == "send_email":
                        args = json.loads(call.function.arguments)
                        result = simulate_email_sending(**args)
                        function_results.append({
                            "tool_call_id": call.id,
                            "output": result
                        })

                # Submit function results back to OpenAI
                openai_client.beta.threads.runs.submit_tool_outputs(
                    thread_id=conversation_thread,
                    run_id=run.id,
                    tool_outputs=function_results
                )
            elif current_run.status in ['failed', 'cancelled', 'expired']:
                st.error(f"Assistant run failed: {current_run.status}")
                return None, conversation_thread

            time.sleep(1)  # Brief pause before checking again

        # Get the latest response
        messages = openai_client.beta.threads.messages.list(thread_id=conversation_thread)
        latest_response = messages.data[0]

        return latest_response.content[0].text.value, conversation_thread

    except Exception as error:
        st.error(f"Conversation error: {str(error)}")
        return None, conversation_thread

def render_sidebar():
    """Renders the sidebar with configuration options and email management."""
    with st.sidebar:
        st.header("🔧 Settings")
        st.info("Using OpenAI API for assistant functionality")

        st.header("📬 Email Management")
        user_email = st.text_input("Your Email Address", placeholder="user@example.com")
        
        if user_email and st.button("💾 Save Email"):
            if 'saved_emails' not in st.session_state:
                st.session_state.saved_emails = []
            if user_email not in st.session_state.saved_emails:
                st.session_state.saved_emails.append(user_email)
                st.success("Email saved!")

        # Display saved emails
        if 'saved_emails' in st.session_state and st.session_state.saved_emails:
            st.subheader("Saved Emails:")
            for email in st.session_state.saved_emails:
                st.text(f"📧 {email}")

def render_knowledge_section():
    """Renders the knowledge base input section."""
    st.header("📖 Knowledge Base")
    st.write("Provide the information you want your assistant to know about.")

    knowledge_content = st.text_area(
        "Enter your knowledge base:",
        height=350,
        placeholder="Add FAQs, product information, company policies, or any specific knowledge you want the assistant to use...",
        help="The assistant will only answer questions based on this content."
    )

    if st.button("💾 Save Context", type="primary", use_container_width=True):
        if knowledge_content.strip():
            with st.spinner("Saving your context and setting up assistant..."):
                assistant_id = build_custom_assistant(knowledge_content)
                if assistant_id:
                    st.session_state.assistant_id = assistant_id
                    st.session_state.knowledge_base = knowledge_content
                    st.session_state.conversation_thread = None
                    st.success("✅ Context saved successfully!")
                    st.rerun()
        else:
            st.warning("Please add some knowledge content first!")

def render_chat_section():
    """Renders the chat interface."""
    st.header("💬 Chat Interface")

    # Show assistant status
    if 'assistant_id' in st.session_state:
        st.success("🤖 Assistant is active and ready")
        with st.expander("📋 View Knowledge Base", expanded=False):
            preview_text = st.session_state.knowledge_base
            if len(preview_text) > 300:
                preview_text = preview_text[:300] + "..."
            st.text(preview_text)
    else:
        st.warning("⚠️ Create an assistant first using the knowledge base section")
        return

    # Initialize chat history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Display chat messages
    chat_area = st.container()
    with chat_area:
        for role, message in st.session_state.conversation_history:
            with st.chat_message(role):
                st.write(message)

    # Chat input
    if user_message := st.chat_input("Ask me anything about the knowledge base..."):
        # Add user message to history
        st.session_state.conversation_history.append(("user", user_message))

        # Get assistant response
        with st.spinner("🤔 Processing your question..."):
            response, thread_id = handle_conversation(
                st.session_state.assistant_id,
                user_message,
                st.session_state.get('conversation_thread')
            )

            if response:
                st.session_state.conversation_thread = thread_id
                st.session_state.conversation_history.append(("assistant", response))
                st.rerun()

def render_email_log():
    """Displays the email activity log."""
    st.header("📨 Email Activity")
    
    if 'email_history' in st.session_state and st.session_state.email_history:
        st.write(f"**Total emails sent:** {len(st.session_state.email_history)}")
        
        for idx, email in enumerate(st.session_state.email_history, 1):
            with st.expander(f"Email #{idx}: {email['subject']} ({email['sent_at']})"):
                st.write(f"**To:** {email['recipient']}")
                st.write(f"**Subject:** {email['subject']}")
                st.write(f"**Message:**")
                st.write(email['body'])
                st.caption(f"Sent on: {email['sent_at']}")
    else:
        st.info("No emails sent yet. Try asking your assistant to send you information via email!")

def render_help_section():
    """Renders the help and usage information."""
    with st.expander("❓ How to Use This App", expanded=False):
        st.markdown("""
        ### Getting Started:

        **1. Set Up Knowledge Base**
        - Add your custom content in the "Knowledge Base" section
        - This could be FAQs, product info, policies, etc.
        - Click "Save Context" to set up your AI

        **2. Configure Email (Optional)**
        - Add your email in the sidebar
        - This enables email functionality in conversations

        **3. Start Chatting**
        - Ask questions about your knowledge base
        - The assistant will only use the information you provided
        - Try asking for emails: "Send me a summary via email"

        ### Key Features:
        - 🎯 **Focused Responses**: Only uses your provided knowledge
        - 📧 **Email Integration**: Can send information via email
        - 💬 **Natural Conversation**: Maintains context throughout the chat
        - 📊 **Activity Tracking**: Logs all email interactions

        ### Example Questions:
        - "What information do you have about [your topic]?"
        - "Can you email me the key points?"
        - "Summarize the main features for me"
        """)

def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="AskWithMe - Custom AI Assistant", 
        page_icon="🤖", 
        layout="wide"
    )
    
    # Header
    st.title("🤖 AskWithMe")
    st.caption("Your personal AI assistant trained on your custom knowledge base")
    
    # Render sidebar
    render_sidebar()
    
    # Main content in two columns
    left_col, right_col = st.columns([1, 1], gap="large")
    
    with left_col:
        render_knowledge_section()
    
    with right_col:
        render_chat_section()
    
    # Full-width sections
    st.divider()
    render_email_log()
    
    st.divider()
    render_help_section()

if __name__ == "__main__":
    main()
