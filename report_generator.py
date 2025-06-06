import openai
import os
from fpdf import FPDF
from dotenv import load_dotenv
from datetime import datetime
import html

load_dotenv()

openai_api_key = os.getenv("OPEN_AI_KEY")

client = openai.OpenAI(api_key = openai_api_key)

def generate_report(analysis_results: dict,
                    output_dir = "reports"):
    """
    Generates a report using OpenAI GPT and saves it as PDF and TXT.
    Arguments:
        task_type (str): e.g., "regression"
        analysis_results (dict): results from analysis, including metrics and interpretation
        output_dir (str): directory to save reports
    Returns:
        dict: file paths for PDF and TXT reports
    """
    os.makedirs(output_dir, exist_ok = True)

    # Extract values from analysis_results dict
    dataset_info = analysis_results.get("dataset_info", "")
    methodology = analysis_results.get("methodology", "")
    model_metrics = analysis_results.get("metrics", {})
    feature_insights = analysis_results.get("feature_insights", "")
    plain_summary = analysis_results.get("plain_english_summary", "")
    technical_notes = analysis_results.get("technical_notes", "")
    limitations = analysis_results.get("limitations", "")
    
    #System Prompt
    system_prompt = f"""
    You are an AI data analyst assistant that writes analysis reports for both technical and non-technical readers.
    The report should be in a structured format with headers and explanations.
    It should cover all the information provided in the analysis results.
    You can infer insights from the data and provide recommendations.
    Think like a data scientist and provide a detailed analysis of the data.
    Explain concepts in simple terms while also including a technical section for data scientists.   
    """

    #User Prompt
    user_prompt = f"""
        Generate a structured analysis report for a data science task.

        ## Dataset Overview:
        {dataset_info}

        ## Methodology:
        {methodology}

        ## Model Metrics:
        {model_metrics}

        ## Feature Insights:
        {feature_insights}

        ## Summary (in plain language):
        {plain_summary}

        ## Technical Summary:
        {technical_notes}

        ## Limitations and Notes:
        {limitations}

        Write the report with headers and explanations.
    """
    #Call API
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        temperature = 0.7,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    if not response.choices or not response.choices[0].message.content:
        raise Exception("Failed to generate report. Please try again.")

    report_text = response.choices[0].message.content

    # Create filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(output_dir, f"report_{timestamp}.txt")
    pdf_path = os.path.join(output_dir, f"report_{timestamp}.pdf")

    with open(txt_path, "w", encoding = "utf-8") as f:
        f.write(report_text)

    #Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto = True, margin = 15)
    pdf.set_font("Arial", size = 12)

    for line in report_text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(pdf_path)

    escaped_text = html.escape(report_text).replace("\n", "<br>")

    html_report = f"""
<html>
  <body style='font-family:Arial;padding:10px'>
        <h2>📊 Analysis Report</h2>
        <p>{escaped_text}</p>
  </body>
</html>
"""

    return html_report