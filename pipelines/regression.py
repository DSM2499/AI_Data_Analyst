import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

def linear_regression_pipeline(df: pd.DataFrame, 
                               target: str, export_dir: str = "exports") -> dict:
    """
    Standard pipeline for linear regression.
    """

    #Drop missing values
    df.dropna(inplace = True, subset = [target])

    #Split features into X and y
    X = df.drop(columns = [target])
    y = df[target]

    #Identify feature types
    numeric_features = X.select_dtypes(include = ['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include = ['object', 'category', 'bool']).columns.tolist()
    
    #Define Transformers
    numeric_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown = 'ignore', drop = 'first'))
    ])

    #Combine preprocessing pipelines
    preprocessor = ColumnTransformer(transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    #Model pipeline
    model_pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    #Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 24)

    param_grid = {
        'regressor__fit_intercept': [True, False],
    }

    grid_search = GridSearchCV(
        model_pipeline, param_grid, 
        cv = 5, scoring = 'neg_mean_squared_error', n_jobs = -1
    )

    #Fit the model
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    #Evaluate the model
    y_pred = best_model.predict(X_test)

    #Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = mse ** 0.5

    metrics = {
        'RMSE': rmse,
        'R2': r2
    }

    preprocessor_fitted = best_model.named_steps['preprocessor']
    feature_names = []
    
    feature_names.extend(numeric_features)

    if categorical_features:
        encoder = preprocessor_fitted.named_transformers_['cat'].named_steps['encoder']
        cat_features = encoder.get_feature_names_out(categorical_features).tolist()
        feature_names.extend(cat_features)

    all_features_names = feature_names

    #Coefficients
    regressor = best_model.named_steps['regressor']
    
    coef_df = pd.DataFrame({
        'Feature': all_features_names,
        'Coefficient': regressor.coef_
    }).sort_values(by = 'Coefficient', key = lambda x: abs(x), ascending = False)

    #Plot actual vs predicted
    os.makedirs(export_dir, exist_ok = True)

    plot_path = os.path.join(export_dir, f'linear_regression_actual_vs_predicted_{target}.png')

    plt.figure(figsize = (10, 6))
    sns.scatterplot(x = y_test, y = y_pred, alpha = 0.7)
    plt.title(f'Actual vs Predicted {target}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    #Return results
    result = {
        'model': best_model,
        'model_type': 'linear_regression',
        'metrics': metrics,
        'coefficients': coef_df,
        'plot_path': plot_path
    }

    return result



