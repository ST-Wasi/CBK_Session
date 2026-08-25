import streamlit as st
st.title("Chat Demo")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------ For showing past messages ------
for msg in st.session_state.messages:
    # msg is a dictionary like {"role": "user", "content": "Hello Chatbot"}
    # st.chat_message() draws a styled bubble based on the role
    # .write() puts the actual text inside the bubble
    st.chat_message(msg["role"]).write(msg["content"])

# ---- Create user input

if user_input := st.chat_input("Type Something..."):
      # Save the users message into the history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Showing the user inputs/text
    st.chat_message("user").write(user_input)
    
    fake_response = f"You said: {user_input}"  
    
    st.session_state.messages.append({"role":"assistant", "content": fake_response})
     
    st.chat_message("assistant").write(fake_response)
