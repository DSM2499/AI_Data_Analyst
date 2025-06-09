import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from orchestrator import run_analysis
from pipelines.regression import linear_regression_pipeline
from pipelines.logistic import logistic_regression_pipeline
from utils.report_helpers import create_analysis_results
from report_generator import generate_report

#Load the environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")

#Global state
st.set_page_config(page_title = "AI Data Analyst", layout = "wide")
st.title("📊 AI Data Analyst")

#File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type =["csv"])

#Task description
task = st.text_area("What would you like to do with this data? (e.g., regression analysis, classification...)")

#Mode Selector
mode = st.radio("Choose analysis mode:", ["Standard Pipeline", "Chat with Data (PandasAI)"])

#Trigger Analysis
if st.button("Run Analysis"):
    if uploaded_file is None or not task:
        st.error("Please upload a CSV file and provide a task description.")
    else:
        try:
            df = pd.read_csv(uploaded_file, encoding = "utf-8")
        except Exception as e:
            try:
                df = pd.read_csv(uploaded_file, encoding = "ISO-8859-1")
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {e}")
                st.stop()

        st.subheader("📁 Dataset Preview")
        st.dataframe(df.head())
        st.write(df.dtypes)

        if mode == "Standard Pipeline":
            st.info("Running structured pipeline analysis...")
            
            with st.spinner("Conducting analysis..."):
                result = run_analysis(df, {"task_type": task})

                try:
                    if result["model_type"] == "linear_regression":
                        with st.spinner("Displaying regression analysis..."):
                            st.write("### 📈 Regression Results")
                            st.metric("R² Score", result["metrics"]["R2"])
                            st.metric("RMSE", result["metrics"]["RMSE"])
                            st.image(result["plot_path"])
                            st.write("### 📊 Coefficients")
                            try:
                                st.dataframe(result["coefficients"].head(10))
                            except:
                                st.write("No coefficients available for this model.")

                    elif result["model_type"] == "logistic_regression":
                        with st.spinner("Displayinglogistic regression analysis..."):
                            st.write("### 📈 Logistic Regression Results")
                            for key, value in result["metrics"].items():
                                st.metric(key.capitalize(), value)
                            st.image(result["plot_path"])
                            st.write("### 📊 Coefficients")
                            try:
                                st.dataframe(result["coefficients"].head(10))
                            except:
                                st.write("No coefficients available for this model.")
                    
                    if result is not None:
                        if "analysis_report" in result:
                            analysis_report = result["analysis_report"]
                        else:
                            analysis_report = create_analysis_results(task, df, result)
                        
                        st.subheader("📊 Analysis Report")
                        report_html = generate_report(analysis_report)
                        st.components.v1.html(report_html, height = 800, scrolling = True)

                        st.session_state["result"] = result
                        st.session_state["task"] = task
                except Exception as e:
                    st.error(f"❌ Error running analysis: {e}")
        
        elif mode == "Chat with Data (PandasAI)":
            from pandasai import SmartDataframe
            from pandasai.llm import OpenAI as PandasAI_OpenAI

            try:
                llm = PandasAI_OpenAI(api_token = OPENAI_API_KEY)
                smart_df = SmartDataframe(df, config = {"llm": llm})

                with st.spinner("Interpreting prompt with PandasAI..."):
                    result = smart_df.chat(task)
                
                if result is not None:
                    st.success("✅ Response:")
                    st.write(result)
                else:
                    st.warning("PandasAI did not return a response.")
            except Exception as e:
                st.error(f"❌ Error running PandasAI: {e}")