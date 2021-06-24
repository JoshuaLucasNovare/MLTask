import streamlit as st
import ml

st.title('Breast Cancer Tumor Predition: Malignant vs Benign')

userinput = st.text_input('Enter your data here (30 floating numbers separated by ","):')
userinput = userinput.split(",")

if len(userinput) == 30:
    option = st.radio('What model whould you like to use?',
    ('Logistic Regression', 'Support Vector Machine', 'Kernel Support Vector Machine',
    'Training K-Nearest Neighbours', 'Decision Trees', 'Naive Bayes', 'Random Forest'))

    if (option == "Logistic Regression"):
        ml.predictor('lr', {'penalty': 'l1', 'solver': 'saga', 'max_iter': 5000}, userinput)

    elif (option == "Support Vector Machine"):
        ml.predictor('svm', {'C': 1, 'gamma': 0.8,'kernel': 'linear', 'random_state': 0}, userinput)

    elif (option == "Kernel Support Vector Machine"):
        ml.predictor('ksvm', {'C': 1, 'gamma': 0.1, 'kernel': 'rbf', 'random_state': 0}, userinput)

    elif (option == "Training K-Nearest Neighbours"):
        ml.predictor('knn', {'n_neighbors': 5, 'n_jobs':1}, userinput)

    elif (option == "Decision Trees"):
        ml.predictor('dt', {'criterion': 'gini', 'max_features': 'auto', 'splitter': 'random' ,'random_state': 0}, userinput)

    elif (option == "Naive Bayes"):
        ml.predictor('nb', {}, userinput)

    elif (option == "Random Forest"):
        ml.predictor('rfc', {'criterion': 'entropy', 'max_features': 'auto', 'n_estimators': 250,'random_state': 0}, userinput)
