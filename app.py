import streamlit as st
import ml

st.title('Breast Cancer Tumor Predition: Malignant vs Benign')
st.text('from https://www.verywellhealth.com/what-does-malignant-and-benign-mean-514240')

st.subheader('Differences Between a Malignant and Benign Tumor')
st.write('''If you have been diagnosed with a tumor, the first step your doctor will take is to find out
            whether it is malignant or benign, as this will affect your treatment plan. In short, the
            meaning of malignant is cancerous and the meaning of benign is noncancerous.
            Learn more about how either diagnosis affects your health.''')

st.subheader('Characteristics of Benign Tumors')
st.write('''Cells tend not to spread

Most grow slowly

Do not invade nearby tissue

Do not metastasize (spread) to other parts of the body

Tend to have clear boundaries

Under a pathologist's microscope, shape, chromosomes, and DNA of cells appear normal

Do not secrete hormones or other substances (an exception: pheochromocytomas of the adrenal gland)

May not require treatment if not health-threatening

Unlikely to recur if removed or require further treatment such as radiation or chemotherapy''')

st.subheader('Characteristics of Malignant Tumors')
st.write('''Cells can spread

Usually grow fairly rapidly

Often invade basal membrane that surrounds nearby healthy tissue

Can spread via bloodstream or lymphatic system, or by sending "fingers" into nearby tissue

May recur after removal, sometimes in areas other the original site

Cells have abnormal chromosomes and DNA characterized by large, dark nuclei; may have abnormal shape

Can secrete substances that cause fatigue and weight loss (paraneoplastic syndrome)

May require aggressive treatment, including surgery, radiation, chemotherapy, and immunotherapy medications''')

#from PIL import Image

#bvm = Image.open(r"images\bvm.jpg","r")
#st.image(bvm)
#st.title('Data Analysis')

#heatmap = Image.open(r"graphs\heatmap.png","r")
#st.image(heatmap)
#st.write('The heatmap shows the multiple data have negative correlatation with one another')

userinput = []
st.sidebar.header('Enter the following data: ')
userinput.append(st.sidebar.number_input('Radius Mean'))
userinput.append(st.sidebar.number_input('Texture Mean'))
userinput.append(st.sidebar.number_input('Perimeter Mean'))
userinput.append(st.sidebar.number_input('Area Mean'))
userinput.append(st.sidebar.number_input('Smoothness Mean'))
userinput.append(st.sidebar.number_input('Compactness Mean'))
userinput.append(st.sidebar.number_input('Concavity Mean'))
userinput.append(st.sidebar.number_input('Concave points Mean'))
userinput.append(st.sidebar.number_input('Symmetry Mean'))
userinput.append(st.sidebar.number_input('Fractal Dimension Mean'))
userinput.append(st.sidebar.number_input('Radius Standard Error'))
userinput.append(st.sidebar.number_input('texture Standard Error'))
userinput.append(st.sidebar.number_input('Perimeter Standard Error'))
userinput.append(st.sidebar.number_input('Area Standard Error'))
userinput.append(st.sidebar.number_input('Smoothness Standard Error'))
userinput.append(st.sidebar.number_input('Compactness Standard Error'))
userinput.append(st.sidebar.number_input('Concavity Standard Error'))
userinput.append(st.sidebar.number_input('Concave points Standard Error'))
userinput.append(st.sidebar.number_input('Symmetry Standard Error'))
userinput.append(st.sidebar.number_input('Fractal Dimension Standard Error'))
userinput.append(st.sidebar.number_input('Radius Worst'))
userinput.append(st.sidebar.number_input('Texture Worst'))
userinput.append(st.sidebar.number_input('Perimeter Worst'))
userinput.append(st.sidebar.number_input('Area Worst'))
userinput.append(st.sidebar.number_input('Smoothness Worst'))
userinput.append(st.sidebar.number_input('Compactness Worst'))
userinput.append(st.sidebar.number_input('Concavity Worst'))
userinput.append(st.sidebar.number_input('Cncave points Worst'))
userinput.append(st.sidebar.number_input('Symmetry Worst'))
userinput.append(st.sidebar.number_input('Fractal Dimension Worst'))

if len(userinput) == 30:
    st.title('Training and Prediction of Tumor.')
    st.subheader('List of Models Available:')
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
