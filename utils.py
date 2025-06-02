import pandas as pd
import streamlit as st
def get_mart(path):
    try:
        if 'private' in path:
            return pd.read_csv(path)
        else:
            return pd.read_csv(path)
    except FileNotFoundError:
        error_message = f"File not found: {path}. Please check if the file exists or ask for access to the file."
        print(error_message)
        st.error(error_message)
        return None
    except Exception as e:
        return e