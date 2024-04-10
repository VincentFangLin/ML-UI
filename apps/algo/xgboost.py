from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st

class XGBoostClassifier:
    def __init__(self):
        self.clf = XGBClassifier()

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write('Classification Report:')
        st.write(classification_report(y_test, y_pred))
        st.write('Confusion Matrix:')
        st.write(confusion_matrix(y_test, y_pred))
