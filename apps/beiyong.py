import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from .utils import select_or_upload_dataset
from apps.algo.logistic import LogisticRegression, accuracy
from apps.algo.xgboost import XGBoostClassifier
from apps.algo.decision_tree import DecisionTreeClassifier
from apps.algo.random_forest import RandomForestClassifier
from apps.algo.svm import SupportVectorMachine
from apps.algo.adaboost import AdaBoostClassifier
from sklearn import metrics

lr = LogisticRegression(lr=0.001, epochs=100)
xgb = XGBoostClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
svm = SupportVectorMachine()
adaboost = AdaBoostClassifier()

def modelling(df):

    show_custom_predictions = True  # for show custom predictions checkbox

    # Show Dataset
    if st.checkbox("Show Dataset"):
        number = st.number_input("Numbers of rows to view", 5)
        st.dataframe(df.head(number))
        

    # Select Algorithm
    algos = [
        "Logistic Regression For Binary Classification",
        "XGBoost",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
        "AdaBoost"
    ]
    selected_algo = st.selectbox("Select Algorithm", algos)

    if selected_algo:
        st.info("You Selected {} Algorithm".format(selected_algo))

    # Select the Taget Feature
    all_columns = df.columns
    target_column = st.selectbox("Select Target Column", all_columns)
    target_data = df[target_column]

    if target_column:
        if st.checkbox("Show Value Count of Target Feature"):
            st.write(df[target_column].value_counts())

    # Label Encoding
    label_encoder = preprocessing.LabelEncoder()
    encoded_target_data = label_encoder.fit_transform(target_data)
    encoded_target_data_knn = pd.DataFrame(encoded_target_data, columns=[target_column])
    st.info("You Selected {} Column".format(target_column))
    remain_column = df.drop([target_column], axis=1)

    if len(df.eval(target_column).unique()) > 25:
        st.error(
            "Please select classification dataset only, or the target column must be categorical"
        )
    
    else:

        # Select the Remaining Feature and scaling them
        X = df.drop([target_column], axis=1)
        
        # Feature selection
        k = st.selectbox("Select top K features:", list(range(1, min(21, X.shape[1]+1))))
        selector = SelectKBest(f_classif, k=k)
        X_new = selector.fit_transform(X, encoded_target_data)
        selected_features = X.columns[selector.get_support()]

        st.info("Selected top {} features:".format(k))
        st.write(selected_features.tolist())

        y = encoded_target_data
        X_names = selected_features
        scaler = StandardScaler()
        X_new = scaler.fit_transform(X_new)
        X_new = pd.DataFrame(X_new, columns=X_names)

        # Train and Test Split
        test_size = st.slider(
            "Select the size of Test Dataset (test train split)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            help="Set the ratio for splitting the dataset into Train and Test Dataset",
        )

        if st.button(
            "Start Training",
            help="Training will start for the selected algorithm on dataset",
        ):

            # Splitting dataset into train and test set

            if selected_algo == "Logistic Regression For Binary Classification":
                X_train, X_test, y_train, y_test = train_test_split(
                    X_new, y, test_size=test_size, random_state=1234
                )

                if len(set(y)) == 2:
                    lr.fit(X_train, y_train)
                    show_custom_predictions = True

                    # making predictions on the testing set
                    y_pred = lr.predict(X_test)
                    print("Accuracy: ", accuracy(y_test, y_pred))

                    # comparing actual response values (y_test) with predicted response values (y_pred)
                    st.write(
                        "Model accuracy of Logistic Regression Model:",
                        metrics.accuracy_score(y_test, y_pred) * 100,
                    )
                    output(y_test, y_pred)
                else:
                    show_custom_predictions = False
                    st.error(
                        "The Target feature has more than 2 classes, please select another dataset or a different classification algorithm."
                    )

            # Implement other algorithms similarly

def output(y_test, y_pred):
    confusion_matrix_out = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix_out)

    st.write("Classification Report:")
    precision_score_out = precision_score(y_test, y_pred)
    recall_score_out = recall_score(y_test, y_pred)
    f1_score_out = f1_score(y_test, y_pred)

    data = [
        ["Precison Score", precision_score_out],
        ["Recall Score", recall_score_out],
        ["f1 Score", f1_score_out],
    ]
    df = pd.DataFrame(data, columns=["Parameter", "Value"])
    st.table(df)


def app():
    st.title("Machine Learning Modelling")
    st.subheader("Choose your dataset ")
    select_or_upload_dataset(modelling)


if __name__ == "__main__":
    app()
