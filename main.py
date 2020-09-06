import os
import pandas as pd
import numpy as np
import pickle
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, recall_score, precision_score, f1_score, mean_squared_error
import statsmodels.api as sm
import statsmodels as sm
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# from sklearn_pandas import CategoricalImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, _tree # Import Decision Tree Classifier
from sklearn.neighbors import KNeighborsClassifier
from io import StringIO
from IPython.display import Image
import pydotplus
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, RFE
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from info_gain import info_gain


SIGMA = 0.05


def insurance_claim_classifier():
    df = pd.read_csv('insurance_claims.csv')
    print(df.info())

    print(df['fraud_reported'].value_counts())
    df['target'] = df['fraud_reported'].apply(lambda x: 1 if x == 'Y' else 0)
    print(df['target'].value_counts())

    df = df.drop('fraud_reported', axis=1)
    df = df.drop('policy_bind_date', axis=1)
    df = df.drop('insured_zip', axis=1)
    df = df.drop('incident_date', axis=1)
    df = df.drop('insured_hobbies', axis=1)
    df = df.drop('_c39', axis=1)
    df = df.drop('policy_number', axis=1)

    df = df.replace('?', np.nan)
    print(df.isna().sum())

    cat_df = df.select_dtypes(include=['object']).copy()
    print(cat_df.columns)
    print(cat_df.head())

    cat_df['policy_csl'] = cat_df['policy_csl'].map({'100/300': 1, '250/500': 2.5, '500/1000': 5})
    cat_df['insured_education_level'] = cat_df['insured_education_level'].map({'JD': 1, 'High School': 2, 'College': 3, 'Masters': 4, 'Associate': 5, 'MD': 6, 'PhD': 7})
    cat_df['incident_severity'] = cat_df['incident_severity'].map({'Trivial Damage': 1, 'Minor Damage': 2, 'Major Damage': 3, 'Total Loss': 4})
    cat_df['insured_sex'] = cat_df['insured_sex'].map({'FEMALE': 0, 'MALE': 1})
    cat_df['property_damage'] = cat_df['property_damage'].map({'NO': 0, 'YES': 1})
    cat_df['police_report_available'] = cat_df['police_report_available'].map({'NO': 0, 'YES': 1})


    cat_df1 = cat_df[['policy_csl', 'insured_education_level', 'incident_severity', 'insured_sex', 'property_damage', 'police_report_available']]
    cat_df = cat_df.drop(['policy_csl', 'insured_education_level', 'incident_severity', 'insured_sex', 'property_damage', 'police_report_available'], axis=1)
    for column in cat_df.columns:
        cat_df[column] = label_encoder.fit_transform(cat_df[column])


def load_diabetic_dataset():
    # load the dataset
    dataset = pd.read_csv('dataset_diabetes/diabetic_data.csv')

    dataset = dataset.drop('encounter_id', axis=1)
    dataset = dataset.drop('patient_nbr', axis=1)
    dataset = dataset.drop('weight', axis=1)

    # summarize the shape of the raw data
    print(dataset.shape)
    # count the number of missing values for each column
    num_missing = (dataset == '?').sum()
    # report the results
    print(num_missing)
    # replace '0' values with 'nan'
    dataset = dataset.replace('?', np.nan)
    # count the number of nan values in each column
    print(dataset.isna().sum())
    # drop rows with missing values
    dataset.dropna(inplace=True)
    # summarize the shape of the data with missing rows removed
    print(dataset.shape)

    # split dataset in features and target variable
    feature_cols = dataset.columns
    X = dataset[feature_cols]  # Features
    y = dataset.diabetesMed  # Target variable

    return X, y, feature_cols


def load_pima_dataset():
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    # load dataset
    dataset = pd.read_csv("pima-indians-diabetes.csv", header=None, names=col_names)

    # summarize the shape of the raw data
    print(dataset.shape)
    # count the number of missing values for each column
    num_missing = (dataset[['glucose', 'bp', 'skin', 'insulin', 'bmi']] == 0).sum()
    # report the results
    print(num_missing)
    # replace '0' values with 'nan'
    dataset[['glucose', 'bp', 'skin', 'insulin', 'bmi']] = dataset[['glucose', 'bp', 'skin', 'insulin', 'bmi']].replace(0, np.nan)
    # count the number of nan values in each column
    print(dataset.isna().sum())
    # drop rows with missing values
    dataset.dropna(inplace=True)
    # summarize the shape of the data with missing rows removed
    print(dataset.shape)

    # split dataset in features and target variable
    feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    X = dataset[feature_cols]  # Features
    y = dataset.label  # Target variable

    return X, y, feature_cols


def load_cvd_dataset(dataset_name):
    # load dataset
    dataset = pd.read_csv(dataset_name)

    # summarize the shape of the raw data
    print(dataset.shape)
    dataset[dataset < 0] = np.nan
    # drop rows with missing values
    dataset.dropna(inplace=True)
    # summarize the shape of the data with missing rows removed
    print(dataset.shape)

    # split dataset in features and target variable
    y = dataset.cardio  # Target variable
    dataset = dataset.drop('cardio', axis=1)
    feature_cols = list(dataset.columns)
    X = dataset[feature_cols]  # Features

    return X, y, feature_cols


def evaluate_user():
    pass


def update_model(model='generated-models/initial_model.sav',
                 dataset_name='cardiovascular-disease-dataset/cardio_train_cleaned_feedback_patients.csv',
                 upper_threshold_probability=0.75,
                 lower_threshold_probability=0.25,
                 threshold_probability=0.5):
    # load the model from disk
    loaded_model = pickle.load(open(model, 'rb'))

    X, y, feature_cols = load_cvd_dataset(dataset_name)
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test

    y_train_proba = loaded_model.predict_proba(X_train)[:, 1]
    y_train_pred = to_labels(y_train_proba, threshold_probability)
    y_train_pred_baseline = loaded_model.predict(X_train)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Accuracy baseline:", accuracy_score(y_train, y_train_pred_baseline))
    print("Precision:", precision_score(y_train, y_train_pred))
    print("Precision baseline:", precision_score(y_train, y_train_pred_baseline))
    print("Recall:", recall_score(y_train, y_train_pred))
    print("Recall baseline:", recall_score(y_train, y_train_pred_baseline))
    print("F1:", f1_score(y_train, y_train_pred))
    print("F1 baseline:", f1_score(y_train, y_train_pred_baseline))
    print("RMSE:", mean_squared_error(y_train, y_train_pred, squared=False))
    print("RMSE baseline:", mean_squared_error(y_train, y_train_pred_baseline, squared=False))

    sample_weights = []
    feedback = []
    expected_feedback = []
    left_middle_threshold = threshold_probability - ((threshold_probability - lower_threshold_probability) / 2)
    right_middle_threshold = upper_threshold_probability - ((upper_threshold_probability - threshold_probability) / 2)
    for prob in y_train_proba:
        feedback.append(np.random.choice(np.arange(1, 6), p=[0, 0, 0, 0.25, 0.75]))
        # feedback.append(np.random.choice(np.arange(1, 6), p=[0.2, 0.2, 0.2, 0.2, 0.2]))
        if lower_threshold_probability <= prob <= upper_threshold_probability:
            if left_middle_threshold <= prob <= right_middle_threshold:
                expected_feedback.append(np.arange(1, 6))
            else:
                expected_feedback.append(np.arange(3, 6))
        else:
            expected_feedback.append(np.arange(5, 6))

    test = [1 if fb in exp_fb else 0 for fb, exp_fb in zip(feedback, expected_feedback)]
    fig, axes = plt.subplots()
    axes.hist(test)
    plt.show()

    y_train_pred_test = [p if fb == 1 else 1 - p for p, fb in zip(y_train_pred, test)]
    print()

    print("Accuracy feedback:", accuracy_score(y_train, y_train_pred_test))
    print("Precision feedback:", precision_score(y_train, y_train_pred_test))
    print("Recall feedback:", recall_score(y_train, y_train_pred_test))
    print("F1 feedback:", f1_score(y_train, y_train_pred_test))
    print("RMSE feedback:", mean_squared_error(y_train, y_train_pred_test, squared=False))


def random_forest_classification(dataset_name='cardiovascular-disease-dataset/cardio_train_cleaned.csv', model_name='generated-models/initial_model.sav'):
    X, y, feature_cols = load_cvd_dataset(dataset_name)

    # X, feature_cols = select_k_best(4, X, y, feature_cols)

    # ig = []
    # iv = []
    # igr = []
    # for feature in feature_cols:
    #     ig.append(info_gain.info_gain(y, X[feature]))
    #     iv.append(info_gain.intrinsic_value(y, X[feature]))
    #     igr.append(info_gain.info_gain_ratio(y, X[feature]))

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

    # Create Decision Tree classifer object
    # clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # uncalibrated predictions
    y_proba_uncalibrated, clf = uncalibrated_classifier(clf, X_train, X_test, y_train)
    # calibrated predictions
    y_proba_calibrated, calibrated_clf = calibrate_classifier(clf, X_train, X_test, y_train)

    # Train Decision Tree Classifer
    # clf = clf.fit(X_train, y_train)
    probabilities = {
        "uncalibrated": y_proba_uncalibrated,
        "calibrated": y_proba_calibrated
    }

    diagnose_calibration_curve(probabilities, y_test)

    upper_probability_threshold, lower_probability_threshold = plot_net_benefit(y_test, y_proba_calibrated)

    # Predict the response for test dataset
    # decision_path = clf.decision_path(X_test)
    # y_log_proba = clf.predict_log_proba(X_test)
    # y_proba = clf.predict_proba(X_test)
    y_proba_calibrated = calibrated_clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    y_pred_calibrated = calibrated_clf.predict(X_test)

    distribution_of_predictions(y_test, y_pred_calibrated, y_proba_calibrated[:, 0], y_proba_calibrated[:, 1])

    TP, FP, FN, TN = perf_measure(y_test, y_pred_calibrated)
    probability_threshold = calculate_threshold(TP, FP, FN, TN)
    print("Calibrated Threshold:", probability_threshold)
    print("Net benefit:", calculate_net_benefit(TP, FP, FN, TN, probability_threshold))

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

    print("Calibrated Accuracy:", accuracy_score(y_test, y_pred_calibrated))
    print("Calibrated Precision:", precision_score(y_test, y_pred_calibrated))
    print("Calibrated Recall:", recall_score(y_test, y_pred_calibrated))
    print("Calibrated F1:", f1_score(y_test, y_pred_calibrated))
    print("Calibrated RMSE:", mean_squared_error(y_test, y_pred_calibrated, squared=False))

    # save the model to disk
    pickle.dump(calibrated_clf, open(model_name, 'wb'))

    return upper_probability_threshold, probability_threshold, lower_probability_threshold


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


# apply threshold to positive probabilities to create labels
def to_labels_range(pos_probs, min_thresh, max_thresh):
    return (np.logical_and(min_thresh <= pos_probs, pos_probs < max_thresh)).astype('int')


def distribution_of_predictions(y_test, y_pred_calibrated, y_proba_calibrated_false, y_proba_calibrated_true):
    false_pred = []
    true_pred = []
    thresholds = np.arange(0.6, 1.01, 0.1)
    for thresh in thresholds:
        y_test_temp = y_test[(thresh - 0.1 <= y_proba_calibrated_true) & (y_proba_calibrated_true < thresh)]
        y_pred_calibrated_temp = y_pred_calibrated[(thresh - 0.1 <= y_proba_calibrated_true) & (y_proba_calibrated_true < thresh)]
        # evaluate each threshold
        TP, FP, FN, TN = perf_measure(y_test_temp, y_pred_calibrated_temp)
        false_pred.append(FP + FN)
        true_pred.append(TP + TN)

        y_test_temp = y_test[(thresh - 0.1 <= y_proba_calibrated_false) & (y_proba_calibrated_false < thresh)]
        y_pred_calibrated_temp = y_pred_calibrated[(thresh - 0.1 <= y_proba_calibrated_false) & (y_proba_calibrated_false < thresh)]
        # evaluate each threshold
        TP, FP, FN, TN = perf_measure(y_test_temp, y_pred_calibrated_temp)
        false_pred.insert(0, FP + FN)
        true_pred.insert(0, TP + TN)

    labels = ['[0.0-0.1]', '[0.1-0.2]', '[0.2-0.3]', '[0.3-0.4]', '[0.4-0.5]', '[0.5-0.6]', '[0.6-0.7]', '[0.7-0.8]', '[0.8-0.9]', '[0.9-1.0]']
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width/2, true_pred, width, label='True predictions')
    ax.bar(x + width/2, false_pred, width, label='False predictions')
    ax.set_xlabel('Probability range')
    ax.set_ylabel('Frequency')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # width = 0.6
    # fig, ax1 = plt.subplots(figsize=(7.6, 4.8))
    # ax1.bar(labels, true_pred, width, label='True predictions')
    # ax1.bar(labels, false_pred, width, bottom=true_pred, label='False predictions')
    # ax1.set_xlabel('Probability range')
    # ax1.set_ylabel('Frequency')
    # ax1.legend()

    plt.show()


def plot_net_benefit(y_test, y_proba_calibrated):
    net_benefit = []
    net_benefit_all = []
    thresholds = np.arange(0, 1, 0.001)
    upper_probability_threshold = lower_probability_threshold = 0
    for thresh in thresholds:
        # evaluate each threshold
        TP, FP, FN, TN = perf_measure(y_test, to_labels(y_proba_calibrated, thresh))
        prediction_model_net_benefit = calculate_net_benefit(TP, FP, FN, TN, thresh)
        all_patients_have_disease_net_benefit = calculate_net_benefit_all(TP, FP, FN, TN, thresh)
        net_benefit.append(prediction_model_net_benefit)
        net_benefit_all.append(all_patients_have_disease_net_benefit)

        if prediction_model_net_benefit > all_patients_have_disease_net_benefit + SIGMA and lower_probability_threshold == 0:
            lower_probability_threshold = thresh
        if prediction_model_net_benefit <= 0 + SIGMA and upper_probability_threshold == 0:
            upper_probability_threshold = thresh

    nobody_has_disease = np.array([0 for i in range(len(thresholds))])
    plt.plot(thresholds, nobody_has_disease, label='If we predict that no patients have disease')
    plt.plot(thresholds, net_benefit_all, label='If we predict that all patients have disease')
    plt.plot(thresholds, net_benefit, label='Prediction model')
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    ylim = max(max(net_benefit), max(net_benefit_all), max(nobody_has_disease)) + 0.1
    plt.ylim(top=ylim, bottom=-ylim)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    return upper_probability_threshold, lower_probability_threshold


def calculate_net_benefit(TP, FP, FN, TN, pt):
    n = TP + FP + FN + TN
    return (TP / n) - ((FP / n) * (pt / (1 - pt)))


def calculate_net_benefit_all(TP, FP, FN, TN, pt):
    n = TP + FP + FN + TN
    return ((TP + FN) / n) - (((FP + TN) / n) * (pt / (1 - pt)))


def calculate_threshold(TP, FP, FN, TN):
    return (TN - FP) / ((TP - FN) + (TN - FP))


# predict uncalibrated probabilities
def uncalibrated_classifier(model, train_X, test_X, train_y):
    # fit a model
    model.fit(train_X, train_y)
    # predict probabilities
    return model.predict_proba(test_X)[:, 1], model


# predict calibrated probabilities
def calibrate_classifier(model, train_X, test_X, train_y):
    # define and fit calibration model
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated.fit(train_X, train_y)
    # predict probabilities
    return calibrated.predict_proba(test_X)[:, 1], calibrated


def diagnose_calibration_curve(probs, test_y):
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--', label="optimal")
    # reliability diagram
    for prob_name, prob_value in probs.items():
        prob_true, prob_pred = calibration_curve(test_y, prob_value, n_bins=10)
        # plot model reliability
        plt.plot(prob_pred, prob_true, marker='.', label=prob_name)
    plt.xlabel("Prediction")
    plt.ylabel("True value")
    plt.legend(loc="upper left")
    plt.show()


def select_k_best(k, X, y, feature_cols):
    # Feature extraction
    test = SelectKBest(score_func=chi2, k=k)
    fit = test.fit(X, y)
    # Summarize scores
    print(fit.scores_)

    X1 = fit.transform(X)
    features = np.array(feature_cols)[test.get_support()]
    print(features)

    return X1, features


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    y_actual = y_actual.to_numpy()

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
           TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
           FP += 1
        if y_actual[i] == y_hat[i] == 0:
           TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
           FN += 1

    return TP, FP, FN, TN


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if feature_names['{}'] <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else: # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            target = np.argmax(tree_.value[node])
            print("{}return {} Class: {}".format(indent, tree_.value[node], target))

    recurse(0, 1)


def linear_regression_feature_importance():
    # define dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=1)
    # define the model
    model = DecisionTreeClassifier()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()


if __name__ == '__main__':
    upper_probability_threshold, probability_threshold, lower_probability_threshold = random_forest_classification(
        dataset_name='cardiovascular-disease-dataset/cardio_train_cleaned_initial.csv',
        model_name='generated-models/initial_model.sav')

    update_model(upper_threshold_probability=upper_probability_threshold,
                 lower_threshold_probability=lower_probability_threshold,
                 threshold_probability=probability_threshold)
