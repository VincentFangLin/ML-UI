import streamlit as st
import pandas as pd
import os


import streamlit as st
import pandas as pd

import streamlit as st
import pandas as pd

def select_or_upload_dataset(callback_function):
    """User can upload their own dataset.

    Args:
        callback_function (function): The function to be called after the dataset is uploaded.
    """
    st.write("Upload your own dataset:")
    data = st.file_uploader("Upload Dataset", type=["csv", "txt"])
    df = pd.DataFrame()
    if data is not None:
        data.seek(0)
        st.success("Dataset uploaded successfully")
        df = pd.read_csv(data)
        df.dropna(inplace=True)
        callback_function(df)  # Execute callback function

