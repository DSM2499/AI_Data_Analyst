import os
import openai
from dotenv import load_dotenv
import pandas as pd
from pipelines.regression import linear_regression_pipeline
from pipelines.logistic import logistic_regression_pipeline
#from pipelines.tree import perform_decision_tree
from utils.report_helpers import create_analysis_results
import json

load_dotenv()

openai_api_key = os.getenv("OPEN_AI_KEY")
client = None
if openai_api_key:
    client = openai.OpenAI(api_key=openai_api_key)
else:
    print("[WARN] OPEN_AI_KEY not set. GPT features will be disabled.")

def infer_task(df: pd.DataFrame, task_type) -> str:
    try:
        schema = df.dtypes.to_string()
        sample_str = df.head(20).to_string()

        prompt = f"""
        You are an expert data analyst with a strong understanding of machine learning, data analysis, statistics, and data visualization.
        You will be given a dataset and a statement. 
        Your task is to analyse the instructions given to you by the user and provide a response detailed below.
        The user wants to perform the following task: {task_type}
        The DataFrame contains the following columns:
        {schema}

        The DataFrame looks like this:
        {sample_str}

        The options for task_type are:
        - regression
        - logistic_regression

        Respond in a JSON format case-sensitive for target column:
        
        {{
            "task_type": "linear regression" | "logistic regression",
            "target": "column_name"
        }}
        
        """

        if client is None:
            raise RuntimeError("OpenAI client not configured")

        response = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Please infer the target column and model."}
            ]
        )

        task_info = response.choices[0].message.content.strip()
        task_info = json.loads(task_info)
        print(f"[INFO] GPT response: {task_info}")
    except Exception as e:
        print(f"[WARN] GPT target inference failed: {e}")
        # Fallback to sensible defaults if GPT inference fails
        default_target = df.columns[-1]
        if pd.api.types.is_numeric_dtype(df[default_target]):
            inferred_type = "linear regression"
        else:
            inferred_type = "logistic regression"
        task_info = {"task_type": inferred_type, "target": default_target}
        print(f"[INFO] Using fallback task info: {task_info}")
    return task_info

def run_analysis(df: pd.DataFrame, args: dict):
    task = args.get("task_type")

    print("[INFO] Inferring target column using GPT...")
    task_info = infer_task(df, task)
    print(f"[INFO] Inferred Approach: {task_info}")

    if task_info["task_type"] == "linear regression":
        result = linear_regression_pipeline(df, task_info["target"])
    elif task_info["task_type"] == "logistic regression":
        result = logistic_regression_pipeline(df, task_info["target"])
    else:
        raise ValueError(f"Unsupported task type: {task_info['task_type']}")
    
    result["analysis_report"] = create_analysis_results(task_info["task_type"], df, result)
    return result