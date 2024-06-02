import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to load data
def load_data():
    df_tornado = pd.read_csv('data/final_df_tornado.csv')
    df_non_tornado = pd.read_csv('data/final_df_non_tornado.csv')
    combined_df = pd.concat([df_tornado, df_non_tornado], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined_df, df_tornado, df_non_tornado

# Load Data
if 'combined_df' not in st.session_state:
    st.session_state.combined_df, st.session_state.df_tornado, st.session_state.df_non_tornado = load_data()

# Page header and description
st.header("Models and Predictions")
st.write("In this app, we will be using our own pre-cleaned and pre-treated data.")

# Show the head of the tornado data
st.subheader("Tornado Data")
st.write(st.session_state.df_tornado.head())

# Show the head of the non-tornado data
st.subheader("Non-Tornado Data")
st.write(st.session_state.df_non_tornado.head())

# Show the head of the combined data
st.subheader("Combined Data")
st.write("After merging and shuffling the tornado and non-tornado data:")
st.write(st.session_state.combined_df.head())

# Drop unnecessary columns
st.subheader("Drop Unnecessary Columns")
st.write("We don't need these columns to predict tornadoes:")
st.write("`'mag', 'inj', 'fat', 'loss', 'slat', 'slon', 'elat', 'elon', 'len', 'wid'`")

if st.button('Drop Columns'):
    st.session_state.combined_df.drop(['mag', 'inj', 'fat', 'loss', 'slat', 'slon', 'elat', 'elon', 'len', 'wid'], axis=1, inplace=True)
    st.write("Columns dropped.")
    st.write(st.session_state.combined_df.head())

# Standardization
st.subheader("Data Standardization")
st.write("""
Click the button below to standardize the following columns:
`'TMAX', 'TMIN', 'PRCP', 'AWND', 'WSFI', 'WSFM', 'WSFG', 'WDFG',
 'WSF1', 'WSF2', 'WSF5', 'WDF1', 'WDF2', 'WDF5', 'WDFI', 'WDFM',
 'ACMH', 'WT03', 'WT11'`
""")

if st.button('Standardize Data'):
    scale_columns = [
        'TMAX', 'TMIN', 'PRCP', 'AWND', 'WSFI', 'WSFM', 'WSFG', 'WDFG',
        'WSF1', 'WSF2', 'WSF5', 'WDF1', 'WDF2', 'WDF5', 'WDFI', 'WDFM',
        'ACMH', 'WT03', 'WT11'
    ]
    scaler = StandardScaler()
    st.session_state.combined_df[scale_columns] = scaler.fit_transform(st.session_state.combined_df[scale_columns])
    st.write("Data has been standardized.")
    st.write(st.session_state.combined_df.head())

# Initialize session state variables
if 'svc_initial_results' not in st.session_state:
    st.session_state.svc_initial_results = None

if 'svc_best_results' not in st.session_state:
    st.session_state.svc_best_results = None

if 'grid_search_results' not in st.session_state:
    st.session_state.grid_search_results = None

if 'cv_5_fold_scores' not in st.session_state:
    st.session_state.cv_5_fold_scores = None

if 'cv_10_fold_scores' not in st.session_state:
    st.session_state.cv_10_fold_scores = None

if 'initial_svc_model' not in st.session_state:
    st.session_state.initial_svc_model = None

if 'best_svc_model' not in st.session_state:
    st.session_state.best_svc_model = None

if 'svc_initial_cm_fig' not in st.session_state:
    st.session_state.svc_initial_cm_fig = None

if 'svc_best_cm_fig' not in st.session_state:
    st.session_state.svc_best_cm_fig = None

# Page header and description
st.header("SVC Model Training")

# Assuming 'features' and 'labels' are your dataset's features and target labels
X = st.session_state.combined_df.drop('tornado_occurrence', axis=1)
y = st.session_state.combined_df['tornado_occurrence']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Initial SVC Model
if st.button('Train Initial SVC Model'):
    with st.spinner('Training Initial SVC Model...'):
        initial_svc_model = SVC(kernel='linear')
        initial_svc_model.fit(X_train, y_train)
        y_pred_initial_svc = initial_svc_model.predict(X_test)
        accuracy_initial_svc = accuracy_score(y_test, y_pred_initial_svc)
        report_initial_svc = classification_report(y_test, y_pred_initial_svc, output_dict=True)
        st.session_state.svc_initial_results = {
            'accuracy': accuracy_initial_svc,
            'report': report_initial_svc
        }
        st.session_state.initial_svc_model = initial_svc_model

# Display Initial SVC Model Results
if st.session_state.svc_initial_results is not None:
    st.write("Initial SVC Model Accuracy:", st.session_state.svc_initial_results['accuracy'])
    st.write(pd.DataFrame(st.session_state.svc_initial_results['report']).transpose())

# Plot Initial SVC Model Confusion Matrix
if st.button('Plot Initial SVC Confusion Matrix'):
    if st.session_state.initial_svc_model is not None:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(st.session_state.initial_svc_model, X_test, y_test, cmap="GnBu_r", ax=ax)
        ax.set_title('Confusion Matrix for Initial SVC Model')
        st.session_state.svc_initial_cm_fig = fig
        st.pyplot(fig)
    else:
        st.write("Train the Initial SVC model first.")

if st.session_state.svc_initial_cm_fig is not None:
    st.pyplot(st.session_state.svc_initial_cm_fig)

# Perform Grid Search
if st.button('Perform Grid Search'):
    with st.spinner('Performing Grid Search...'):
        model = SVC()
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto']
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        st.session_state.grid_search_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

# Display Grid Search Results
if st.session_state.grid_search_results is not None:
    st.write("Best parameters:", st.session_state.grid_search_results['best_params'])
    st.write("Best score:", st.session_state.grid_search_results['best_score'])

# Train Best SVC Model
if st.button('Train Best SVC Model'):
    with st.spinner('Training Best SVC Model...'):
        best_svc_model = SVC(C=10, gamma='auto', kernel='rbf')
        best_svc_model.fit(X_train, y_train)
        y_pred_best_svc = best_svc_model.predict(X_test)
        accuracy_best_svc = accuracy_score(y_test, y_pred_best_svc)
        report_best_svc = classification_report(y_test, y_pred_best_svc, output_dict=True)
        st.session_state.svc_best_results = {
            'accuracy': accuracy_best_svc,
            'report': report_best_svc
        }
        st.session_state.best_svc_model = best_svc_model

# Display Best SVC Model Results
if st.session_state.svc_best_results is not None:
    st.write("Best SVC Model Accuracy:", st.session_state.svc_best_results['accuracy'])
    st.write(pd.DataFrame(st.session_state.svc_best_results['report']).transpose())

# Plot Best SVC Model Confusion Matrix
if st.button('Plot Best SVC Confusion Matrix'):
    if st.session_state.best_svc_model is not None:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(st.session_state.best_svc_model, X_test, y_test, cmap='Purples', ax=ax)
        ax.set_title('Confusion Matrix for Best SVC Model After Grid Search')
        st.session_state.svc_best_cm_fig = fig
        st.pyplot(fig)
    else:
        st.write("Train the Best SVC model first.")

if st.session_state.svc_best_cm_fig is not None:
    st.pyplot(st.session_state.svc_best_cm_fig)

# Plot Accuracy Comparison
if st.button('Plot Accuracy Comparison'):
    if st.session_state.svc_initial_results is not None and st.session_state.svc_best_results is not None:
        x = ['Initial SVC', 'Best SVC']
        y = [st.session_state.svc_initial_results['accuracy'], st.session_state.svc_best_results['accuracy']]
        
        fig, ax = plt.subplots()
        ax.bar(x, y, color=['#9fc8c8', '#298c8c'])
        ax.set_title('Accuracy Comparison between Initial and Best SVC Models')
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        st.pyplot(fig)
    else:
        st.write("Train both the Initial and Best SVC models first.")

# Cross-validation with 5 folds
if st.button('5-Fold Cross Validation'):
    with st.spinner('Performing 5-Fold Cross Validation...'):
        cv_scores_5 = cross_val_score(st.session_state.best_svc_model, X, y, cv=5, scoring='accuracy')
        st.session_state.cv_5_fold_scores = cv_scores_5
        st.write("5-Fold Cross Validation Scores:", cv_scores_5)
        st.write("Average 5-Fold CV Score:", cv_scores_5.mean())

# Cross-validation with 10 folds
if st.button('10-Fold Cross Validation'):
    with st.spinner('Performing 10-Fold Cross Validation...'):
        cv_scores_10 = cross_val_score(st.session_state.best_svc_model, X, y, cv=10, scoring='accuracy')
        st.session_state.cv_10_fold_scores = cv_scores_10
        st.write("10-Fold Cross Validation Scores:", cv_scores_10)
        st.write("Average 10-Fold CV Score:", cv_scores_10.mean())

# Initialize session state variables for Random Forest
if 'rf_initial_results' not in st.session_state:
    st.session_state.rf_initial_results = None

if 'rf_individual_wind_results' not in st.session_state:
    st.session_state.rf_individual_wind_results = None

if 'rf_no_wind_results' not in st.session_state:
    st.session_state.rf_no_wind_results = None

if 'rf_cv_5_fold_scores' not in st.session_state:
    st.session_state.rf_cv_5_fold_scores = None

if 'rf_cv_10_fold_scores' not in st.session_state:
    st.session_state.rf_cv_10_fold_scores = None

if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None

if 'rf_model_no_wind' not in st.session_state:
    st.session_state.rf_model_no_wind = None

# Initialize session state variables for Random Forest confusion matrices
if 'rf_initial_cm_fig' not in st.session_state:
    st.session_state.rf_initial_cm_fig = None

if 'rf_no_wind_cm_fig' not in st.session_state:
    st.session_state.rf_no_wind_cm_fig = None

# Create a copy of the combined_df
combined_df_copy = st.session_state.combined_df.copy()

# Random Forest Section
st.header("Random Forest Model Training")

# Train Initial Random Forest Model
if st.button('Train Initial Random Forest Model'):
    with st.spinner('Training Initial Random Forest Model...'):
        X = combined_df_copy.drop('tornado_occurrence', axis=1)
        y = combined_df_copy['tornado_occurrence']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred)
        report_rf = classification_report(y_test, y_pred, output_dict=True)
        st.session_state.rf_initial_results = {
            'accuracy': accuracy_rf,
            'report': report_rf
        }
        st.session_state.rf_model = rf_model

# Display Initial Random Forest Model Results
if st.session_state.rf_initial_results is not None:
    st.write("Initial Random Forest Model Accuracy:", st.session_state.rf_initial_results['accuracy'])
    st.write(pd.DataFrame(st.session_state.rf_initial_results['report']).transpose())

# Plot Initial Random Forest Model Confusion Matrix
if st.button('Plot Initial Random Forest Confusion Matrix'):
    if st.session_state.rf_model is not None:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(st.session_state.rf_model, X_test, y_test, cmap="GnBu_r", ax=ax)
        ax.set_title('Confusion Matrix for Initial Random Forest Model')
        st.session_state.rf_initial_cm_fig = fig
        st.pyplot(fig)
    else:
        st.write("Train the Initial Random Forest model first.")

if st.session_state.rf_initial_cm_fig is not None:
    st.pyplot(st.session_state.rf_initial_cm_fig)

# Test Individual Wind Columns
if st.button('Test Individual Wind Columns'):
    with st.spinner('Testing Individual Wind Columns...'):
        wind_columns = ['AWND', 'WSFI', 'WSFM', 'WSFG', 'WDFG', 'WSF1', 'WSF2', 'WSF5', 'WDF1', 'WDF2', 'WDF5', 'WDFI', 'WDFM']
        results = []
        for col in wind_columns:
            drop_columns = [c for c in wind_columns if c != col]
            X = combined_df_copy.drop(columns=['tornado_occurrence'] + drop_columns)
            y = combined_df_copy['tornado_occurrence']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((col, accuracy))
        st.session_state.rf_individual_wind_results = results

# Display Individual Wind Columns Test Results
if st.session_state.rf_individual_wind_results is not None:
    for col, acc in st.session_state.rf_individual_wind_results:
        st.write(f"Keeping Column: {col}, Accuracy: {acc}")

# Train Random Forest Model without Wind Columns
if st.button('Train Random Forest Model without Wind Columns'):
    with st.spinner('Training Random Forest Model without Wind Columns...'):
        combined_df_without_wind = combined_df_copy.drop(['AWND', 'WSFI', 'WSFM', 'WSFG', 'WDFG', 'WSF1', 'WSF2', 'WSF5', 'WDF1', 'WDF2', 'WDF5', 'WDFI', 'WDFM'], axis=1)
        X = combined_df_without_wind.drop('tornado_occurrence', axis=1)
        y = combined_df_without_wind['tornado_occurrence']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_model_no_wind = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model_no_wind.fit(X_train, y_train)
        y_pred = rf_model_no_wind.predict(X_test)
        accuracy_rf_no_wind = accuracy_score(y_test, y_pred)
        report_rf_no_wind = classification_report(y_test, y_pred, output_dict=True)
        st.session_state.rf_no_wind_results = {
            'accuracy': accuracy_rf_no_wind,
            'report': report_rf_no_wind
        }
        st.session_state.rf_model_no_wind = rf_model_no_wind

# Display Random Forest Model without Wind Columns Results
if st.session_state.rf_no_wind_results is not None:
    st.write("Random Forest Model without Wind Columns Accuracy:", st.session_state.rf_no_wind_results['accuracy'])
    st.write(pd.DataFrame(st.session_state.rf_no_wind_results['report']).transpose())

# Plot Random Forest Model without Wind Columns Confusion Matrix
if st.button('Plot Random Forest Confusion Matrix without Wind Columns'):
    if st.session_state.rf_model_no_wind is not None:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(st.session_state.rf_model_no_wind, X_test, y_test, cmap='Purples', ax=ax)
        ax.set_title('Confusion Matrix for Random Forest Model without Wind Columns')
        st.session_state.rf_no_wind_cm_fig = fig
        st.pyplot(fig)
    else:
        st.write("Train the Random Forest model without Wind columns first.")

if st.session_state.rf_no_wind_cm_fig is not None:
    st.pyplot(st.session_state.rf_no_wind_cm_fig)

# Cross-validation with 5 folds
if st.button('5-Fold Cross Validation for Random Forest'):
    with st.spinner('Performing 5-Fold Cross Validation for Random Forest...'):
        combined_df_without_wind = combined_df_copy.drop(['AWND', 'WSFI', 'WSFM', 'WSFG', 'WDFG', 'WSF1', 'WSF2', 'WSF5', 'WDF1', 'WDF2', 'WDF5', 'WDFI', 'WDFM'], axis=1)
        X = combined_df_without_wind.drop('tornado_occurrence', axis=1)
        y = combined_df_without_wind['tornado_occurrence']
        rf_model_no_wind = RandomForestClassifier(n_estimators=100, random_state=42)
        scores_5 = cross_val_score(rf_model_no_wind, X, y, cv=5)
        st.session_state.rf_cv_5_fold_scores = scores_5
        st.write("5-Fold Cross Validation Scores for Random Forest:", scores_5)
        st.write("Average 5-Fold CV Score:", scores_5.mean())

# Cross-validation with 10 folds
if st.button('10-Fold Cross Validation for Random Forest'):
    with st.spinner('Performing 10-Fold Cross Validation for Random Forest...'):
        combined_df_without_wind = combined_df_copy.drop(['AWND', 'WSFI', 'WSFM', 'WSFG', 'WDFG', 'WSF1', 'WSF2', 'WSF5', 'WDF1', 'WDF2', 'WDF5', 'WDFI', 'WDFM'], axis=1)
        X = combined_df_without_wind.drop('tornado_occurrence', axis=1)
        y = combined_df_without_wind['tornado_occurrence']
        rf_model_no_wind = RandomForestClassifier(n_estimators=100, random_state=42)
        scores_10 = cross_val_score(rf_model_no_wind, X, y, cv=10)
        st.session_state.rf_cv_10_fold_scores = scores_10
        st.write("10-Fold Cross Validation Scores for Random Forest:", scores_10)
        st.write("Average 10-Fold CV Score:", scores_10.mean())

# Initialize session state variables
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

if 'nn_best_model' not in st.session_state:
    st.session_state.nn_best_model = None

if 'nn_cv_scores' not in st.session_state:
    st.session_state.nn_cv_scores = None

# Section for testing different models
st.header("Testing Different Models")
st.write("We will evaluate several models to identify the best performing one for tornado prediction.")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "AdaBoost": AdaBoostClassifier(),
    "k-NN": KNeighborsClassifier(),
    "NeuralNetwork": MLPClassifier(max_iter=1000)
}

# Function to train and evaluate models
def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[model_name] = {"accuracy": accuracy, "report": report}
    return results


if st.button('Evaluate Models'):
    st.session_state.model_results = evaluate_models(models, X_train, X_test, y_train, y_test)

    for model_name, results in st.session_state.model_results.items():
        st.write(f"**Model: {model_name}**")
        st.write(f"Accuracy: {results['accuracy']:.2f}")
        st.write(pd.DataFrame(results['report']).transpose())


# Section for Neural Network Grid Search
st.header("Neural Network Grid Search")
st.write("We observed that the Neural Network gave promising results. Let's perform a grid search to find the best parameters.")

if st.button('Perform Grid Search for Neural Network'):
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100,), (100, 100), (150, 150)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive']
    }

    grid_search = GridSearchCV(estimator=MLPClassifier(max_iter=1000), param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    st.session_state.nn_best_model = grid_search.best_estimator_

    st.write(f"Best Parameters: {grid_search.best_params_}")
    st.write("Neural Network Grid Search completed.")


# Section to train the best Neural Network model
st.header("Train Neural Network with Best Parameters")

if st.button('Train Best Neural Network Model'):
    best_params = {
        'activation': 'tanh',
        'alpha': 0.05,
        'hidden_layer_sizes': (100,),
        'learning_rate': 'constant',
        'solver': 'adam'
    }
    st.session_state.nn_best_model = MLPClassifier(**best_params, max_iter=1000, random_state=42)
    st.session_state.nn_best_model.fit(X_train, y_train)
    y_pred_best_nn = st.session_state.nn_best_model.predict(X_test)
    accuracy_best_nn = accuracy_score(y_test, y_pred_best_nn)
    report_best_nn = classification_report(y_test, y_pred_best_nn, output_dict=True)

    st.write(f"Neural Network Accuracy: {accuracy_best_nn:.2f}")
    st.write(pd.DataFrame(report_best_nn).transpose())

# Section for Cross Validation
st.header("Cross Validation for Neural Network")

if st.button('Perform 5-Fold Cross Validation') and st.session_state.nn_best_model is not None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_5 = cross_val_score(st.session_state.nn_best_model, X, y, cv=cv, scoring='accuracy')

    st.session_state.nn_cv_scores = cv_scores_5

    st.write(f"5-Fold Cross-Validation Scores: {cv_scores_5}")
    st.write(f"Mean Score: {np.mean(cv_scores_5):.2f}")
    st.write(f"Standard Deviation: {np.std(cv_scores_5):.2f}")

if st.button('Perform 10-Fold Cross Validation') and st.session_state.nn_best_model is not None:
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores_10 = cross_val_score(st.session_state.nn_best_model, X, y, cv=cv, scoring='accuracy')

    st.session_state.nn_cv_scores = cv_scores_10

    st.write(f"10-Fold Cross-Validation Scores: {cv_scores_10}")
    st.write(f"Mean Score: {np.mean(cv_scores_10):.2f}")
    st.write(f"Standard Deviation: {np.std(cv_scores_10):.2f}")



