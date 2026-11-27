# ============================================================
# TubeMentor AI - Machine Learning Module
# Logistic Regression + Naive Bayes + Decision Tree
# CSV + Data Fusion
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# 1. LOAD CSV DATA
# ============================================================

def load_csv(file_path):
    """
    Load DSA dataset from CSV file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    print("\nCSV loaded successfully!")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    return df


# ============================================================
# 2. DATA CLEANING
# ============================================================

def clean_data(df):

    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Fill numerical missing values
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_columns:
        df[col] = df[col].fillna(
            df[col].median()
        )

    # Fill text missing values
    text_columns = df.select_dtypes(
        exclude=np.number
    ).columns

    for col in text_columns:
        df[col] = df[col].fillna("Unknown")

    return df


# ============================================================
# 3. DATA FUSION
# ============================================================

def fuse_datasets(df1, df2):
    """
    Combine two CSV datasets using common columns.

    Example:
    Dataset 1 -> YouTube information
    Dataset 2 -> DSA learning information
    """

    common_columns = list(
        set(df1.columns).intersection(
            set(df2.columns)
        )
    )

    if len(common_columns) == 0:

        print(
            "No common column found."
        )

        # Vertical fusion
        fused_df = pd.concat(
            [df1, df2],
            ignore_index=True,
            sort=False
        )

    else:

        print(
            "Common columns:",
            common_columns
        )

        # Horizontal / relational fusion
        fused_df = pd.merge(
            df1,
            df2,
            on=common_columns,
            how="outer"
        )

    return fused_df


# ============================================================
# 4. PREPARE ML DATA
# ============================================================

def prepare_data(
    df,
    target_column
):

    df = df.copy()

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist in CSV."
        )

    # Separate target
    y = df[target_column]

    X = df.drop(
        columns=[target_column]
    )

    # Convert categorical columns
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns

    encoders = {}

    for column in categorical_columns:

        encoder = LabelEncoder()

        X[column] = encoder.fit_transform(
            X[column].astype(str)
        )

        encoders[column] = encoder

    # Convert remaining values to numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.fillna(0)

    # Encode target if text
    target_encoder = None

    if y.dtype == "object":

        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(
            y.astype(str)
        )

    return X, y, encoders, target_encoder


# ============================================================
# 5. TRAIN LOGISTIC REGRESSION
# ============================================================

def train_logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test
):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, scaler, predictions, accuracy


# ============================================================
# 6. TRAIN NAIVE BAYES
# ============================================================

def train_naive_bayes(
    X_train,
    X_test,
    y_train,
    y_test
):

    model = GaussianNB()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, predictions, accuracy


# ============================================================
# 7. TRAIN DECISION TREE
# ============================================================

def train_decision_tree(
    X_train,
    X_test,
    y_train,
    y_test
):

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, predictions, accuracy


# ============================================================
# 8. TRAIN ALL MODELS
# ============================================================

def train_all_models(
    df,
    target_column
):

    X, y, encoders, target_encoder = prepare_data(
        df,
        target_column
    )

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------
    # Logistic Regression
    # --------------------------------

    (
        logistic_model,
        scaler,
        logistic_predictions,
        logistic_accuracy
    ) = train_logistic_regression(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------
    # Naive Bayes
    # --------------------------------

    (
        nb_model,
        nb_predictions,
        nb_accuracy
    ) = train_naive_bayes(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------
    # Decision Tree
    # --------------------------------

    (
        tree_model,
        tree_predictions,
        tree_accuracy
    ) = train_decision_tree(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------
    # Results
    # --------------------------------

    results = {

        "Logistic Regression":
            logistic_accuracy,

        "Naive Bayes":
            nb_accuracy,

        "Decision Tree":
            tree_accuracy
    }

    models = {

        "Logistic Regression":
            logistic_model,

        "Naive Bayes":
            nb_model,

        "Decision Tree":
            tree_model
    }

    predictions = {

        "Logistic Regression":
            logistic_predictions,

        "Naive Bayes":
            nb_predictions,

        "Decision Tree":
            tree_predictions
    }

    print("\n==============================")
    print("MODEL RESULTS")
    print("==============================")

    for model_name, accuracy in results.items():

        print(
            f"{model_name}: "
            f"{accuracy * 100:.2f}%"
        )

    return {
        "models": models,
        "predictions": predictions,
        "accuracies": results,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "encoders": encoders,
        "target_encoder": target_encoder
    }


# ============================================================
# 9. PREDICT USING ALL MODELS
# ============================================================

def predict_with_models(
    trained,
    X_new
):

    results = {}

    # Logistic Regression
    X_scaled = trained[
        "scaler"
    ].transform(X_new)

    results["Logistic Regression"] = (
        trained["models"][
            "Logistic Regression"
        ].predict(X_scaled)
    )

    # Naive Bayes
    results["Naive Bayes"] = (
        trained["models"][
            "Naive Bayes"
        ].predict(X_new)
    )

    # Decision Tree
    results["Decision Tree"] = (
        trained["models"][
            "Decision Tree"
        ].predict(X_new)
    )

    return results


# ============================================================
# 10. MODEL REPORT
# ============================================================

def generate_model_report(
    trained
):

    print("\n")
    print("=" * 60)
    print("TUBEMENTOR AI - ML MODEL REPORT")
    print("=" * 60)

    for model_name in trained["models"]:

        predictions = trained[
            "predictions"
        ][model_name]

        print("\nModel:", model_name)

        print(
            classification_report(
                trained["y_test"],
                predictions,
                zero_division=0
            )
        )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    csv_file = "data/dsa_dataset.csv"

    try:

        # Load CSV
        df = load_csv(
            csv_file
        )

        # Clean data
        df = clean_data(
            df
        )

        print("\nCleaned Dataset:")
        print(df.head())

        # Change this to your actual target column
        target_column = "performance"

        trained = train_all_models(
            df,
            target_column
        )

        # Print detailed report
        generate_model_report(
            trained
        )

    except Exception as e:

        print(
            "\nERROR:",
            e
        )