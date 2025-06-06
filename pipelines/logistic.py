import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def logistic_regression_pipeline(
        df: pd.DataFrame, target: str,
        export_dir: str = "exports"
) -> dict:
    """
    Standard pipeline for logistic regression.
    """

    #Drop missing values
    df.dropna(inplace = True, subset = [target])

    #Split features into X and y
    X = df.drop(columns = [target])
    y = df[target]

    #Encode target if needed
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = y.astype('category').cat.codes

    #Identify feature types
    numeric_features = X.select_dtypes(include = ['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include = ['object', 'category', 'bool']).columns.tolist()

    #Define transformers
    numeric_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown = 'ignore', drop = 'first'))
    ])

    #Combine transformers
    preprocessor = ColumnTransformer(transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    #Model pipeline
    model_pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter = 1000))
    ])

    #Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 24)
    
    param_grid = {
        'classifier__penalty': ['l1', 'l2'],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['liblinear'],
    }

    grid_search = GridSearchCV(
        model_pipeline, param_grid, cv = 5, scoring = 'accuracy', n_jobs = -1
    )

    #Fit the model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    #Evaluate the model
    y_pred = best_model.predict(X_test)

    #Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

    #Extract feature names
    preprocessor_fitted = best_model.named_steps['preprocessor']
    feature_names = []
    
    feature_names.extend(numeric_features)

    if categorical_features:
        encoder = preprocessor_fitted.named_transformers_['cat'].named_steps['encoder']
        cat_features = encoder.get_feature_names_out(categorical_features).tolist()
        feature_names.extend(cat_features)

    all_features_names = feature_names
    try:
        #Coefficients
        coef_df_list = []
        if best_model.named_steps['classifier'].coef_.shape[0] == 1:
            coef_df = pd.DataFrame({
                'Feature': all_features_names,
                'Coefficient': best_model.named_steps['classifier'].coef_[0]
            }).sort_values(by = 'Coefficient', key = lambda x: abs(x), ascending = False)
        else:
            for class_idx, class_coef in enumerate(best_model.named_steps['classifier'].coef_):
                df = pd.DataFrame({
                    'Feature': all_features_names,
                    'Coefficient': class_coef,
                    'Class': class_idx
                })
                coef_df_list.append(df)
            coef_df = pd.concat(coef_df_list, ignore_index = True).sort_values(by = 'Coefficient', key = lambda x: abs(x), ascending = False)

    except Exception as e:
        print(f"[ERROR] Error calculating coefficients: {e}")
        coef_df = None

    #plot confusion matrix
    os.makedirs(export_dir, exist_ok = True)
    plot_path = os.path.join(export_dir, f'logistic_regression_confusion_matrix_{target}.png')

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    disp.plot(cmap = 'Blues')
    plt.title(f'Confusion Matrix - Logistic Regression')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    #Return results
    result = {
        'model': best_model,
        'model_type': 'logistic_regression',
        'metrics': metrics,
        'coefficients': coef_df,
        'plot_path': plot_path
    }

    return result