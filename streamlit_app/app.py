import streamlit as st

st.set_page_config(
    page_title="AMC Project Demo",
    page_icon="📡",
    layout="wide"
)

st.title("Automatic Modulation Classification Web Demo")

st.write("This is my AMC final project web application.")

st.header("Project Overview")

st.write("""
This project uses classical machine learning methods to classify wireless modulation types
from I/Q signal samples.
""")

st.success("Streamlit app is running successfully.")
