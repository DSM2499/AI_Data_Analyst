# 🧠 AI Data Analyst
An intelligent web application that turns your raw data into actionable insights and analysis reports using a hybrid pipeline of traditional ML techniques and LLM-based natural language orchestration.
---

## 🚀 Project Overview
AI Data Analyst is an AI-powered Streamlit app that allows users to upload CSV datasets and request analysis tasks in natural language — such as:
- Exploratory Data Analysis (EDA)
- Regression Analysis (Linear)
- Classification Analysis (Logistic Regression)
- Predictive Modeling

The system intelligently interprets user intent using GPT, chooses the appropriate analysis pipeline, executes it, and generates:
- 📈 Visual plots
- 📊 Model metrics
- 📋 Explainable feature insights
- 📄 A full natural language report generated via GPT-3.5 Turbo, suitable for both technical and non-technical audiences

## 🎨 Key Features
✅ Natural language driven orchestration

✅ GPT-based target column inference

✅ Automated pipeline selection (Regression / Classification / Decision Tree)

✅ Robust fallback handling for CSV file encoding

✅ Chat with Data mode using PandasAI integration

✅ Support for:
- Linear Regression
- Logistic Regression
- Actual vs Predicted (Regression)
- Confusion Matrix (Classification)
- Feature Importances (Tree / Coefficients)

## 🛠️ Tools & Stack
**Layer** : **Tools Used**
UI : Streamlit
ML Models: scikit-learn
Plots: Matplotlib, seaborn
Data Handling: Pandas
LLM Orchestration: OpenAI GPT 3.5-turbo
Data Chat Layer: PandasAI
Report Generation: GPT 3.5 + FPDF

## 📊 Impact & Value
- Bridges gap between technical data science and business reporting
- Provides transparent, explainable AI reports from raw data
 - Eliminates the need for users to write code for EDA and modeling
 - Boosts productivity for data teams and analysts
 - Supports non-technical stakeholders by generating human-friendly summaries
- Enables interactive data exploration via Chat mode
- Demonstrates practical use of LLM orchestration patterns for real-world analytics

## ⚙️ Implementation Methodology
#### 1️⃣ User Input
- Upload CSV
- Provide natural language task description
- Optional chat with PandasAI

#### 2️⃣ Task Understanding
- GPT analyzes task prompt and dataset schema
- Chooses appropriate pipeline:
  - Linear Regression
  - Logistic Regressions

#### 3️⃣ Pipeline Execution
- Data preprocessed (encoding handling, dummies)
- Model trained and evaluated
- Plots generated
  - Actual vs Predicted
  - Confusion Matrix
  - Feature Importance

#### 4️⃣ Report Generation
- Analysis results passed to GPT with structured prompts.
- GPT writes a multi-section report:
  - Dataset summary
  - Methodology
  - Metrics
  - Feature Insights
  - Plain-English and Technical explanations
  - Limitations
- Report saved as a PDF and displayed in app.

## 📝 Setup & Execution Guide

#### Step 1: Clone repo
```bash
git clone https://github.com/DSM2499/AI_Data_Cleaner.git
cd AI_Data_Cleaner
```

#### Step 2: Create & activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows
```

#### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Set your OpenAI API key
- Create a .env file
  ```bash
  OPEN_AI_KEY=sk-xxxxxxxxxxxxxxxxxxxxxx
  ```

#### Step 5: Run the Streamlit app
```bash
streamlit run app.py
```

## 🤝 Contribution & Feedback
Contributions welcome — feel free to open issues, fork the repo, and suggest improvements!
