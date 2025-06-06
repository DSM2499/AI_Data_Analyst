import pandas as pd

def create_analysis_results(task_type, df, result):
    """
    Creates a structured dictionary summarizing the analysis,
    suitable for generating a GPT-based report.

    Parameters:
        task_type (str): "Linear regression" or "logistic_regression"
        df (pd.DataFrame): original dataset
        result (dict): output of model analysis function

    Returns:
        dict: formatted analysis details
    """
    dataset_info = f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns. " \
                   f"The columns include: {', '.join(df.columns[:10]) + ('...' if len(df.columns) > 10 else '')}."
    
    methodology = f"A {task_type.replace('_', ' ')} model was trained using scikit-learn. " \
                  f"Standard preprocessing steps such as encoding and splitting were applied."
    
    #Metrics may vary by task
    metrics = {}

    if task_type == "linear regression":
        metrics = {
            "RMSE": result.get("metrics")["RMSE"],
            "R2 Score": result.get("metrics")["R2"]
        }
    elif task_type == "logistic regression":
        metrics = {
            "Accuracy": result.get("metrics")["Accuracy"],
            "Precision": result.get("metrics")["Precision"],
            "Recall": result.get("metrics")["Recall"],
            "F1 Score": result.get("metrics")["F1"]
        }
    elif task_type == "decision_tree":
        metrics = result.get("metrics", {})
    
    #Feature insights
    feature_importance = result.get("coefficients")
    if isinstance(feature_importance, pd.DataFrame):
        top_feats = feature_importance[:5]
        feature_insights = "\n".join([f"{row['Feature']}: {row['Coefficient']:.3f}" for _, row in top_feats.iterrows()])
    else:
        feature_insights = "Feature importance analysis not available."

    plain_english_summary = f"This {task_type.replace('_', ' ')} model was used to predict the target variable. " \
                            f"The key contributing features were highlighted and performance metrics evaluated."
    
    technical_notes = f"The model used was trained with an 80-20 train-test split. " \
                      f"Model hyperparameters were set to default for a baseline evaluation."

    limitations = f"This analysis used default settings and did not include hyperparameter tuning or cross-validation. " \
                  f"Further analysis is recommended for production use."
    
    return {
        "dataset_info": dataset_info,
        "methodology": methodology,
        "metrics": metrics,
        "feature_insights": feature_insights,
        "plain_english_summary": plain_english_summary,
        "technical_notes": technical_notes,
        "limitations": limitations
    }