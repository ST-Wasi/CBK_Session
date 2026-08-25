import streamlit as st
 
st.title("Chat Demo")
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Show past messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
 
# Get new input
if user_input := st.chat_input("Type something..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
 
    fake_response = f"You said: {user_input}"
    st.session_state.messages.append({"role": "assistant", "content": fake_response})
    st.chat_message("assistant").write(fake_response)