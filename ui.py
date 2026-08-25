import streamlit as st

st.title("Counter App")
st.write("Hello! This side basic Streamlit")

if "count" not in st.session_state:
    st.session_state.count  = 0
    
if st.button("Click me"):
    st.session_state.count += 1

st.write(f"Button Clicked {st.session_state.count} times")
