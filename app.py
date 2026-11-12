
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from llm_providers import run_llm

st.set_page_config(
    page_title="My First AI Chat Assistant",
    page_icon="🤖"
)

st.title("🤖 My First AI Chat Assistant")
st.caption("Your personal AI-powered chat assistant")

# Initialize conversation memory
if "history" not in st.session_state:
    st.session_state["history"] = [
        {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        }
    ]

# Display chat history
for msg in st.session_state["history"]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your message..."):

    # Add user message
    st.session_state["history"].append({
        "role": "user",
        "content": prompt
    })

    st.chat_message("user").markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking... 🤔"):

            try:
                reply = run_llm(st.session_state["history"])

                # Save successful response
                st.session_state["history"].append({
                    "role": "assistant",
                    "content": reply
                })

                st.markdown(reply)

            except Exception as e:

                st.error(
                    "Sorry! I couldn't generate a response. "
                    "Please try again."
                )

                print(f"ERROR: {type(e).__name__}: {e}")

# Reset conversation
if st.button("🔄 Reset Conversation"):

    st.session_state["history"] = [
        {
            "role": "assistant",
            "content": "Conversation reset. How can I assist you?"
        }
    ]

    st.rerun()
