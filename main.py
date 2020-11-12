import pandas as pd
import numpy as np
import pickle
from collections import Counter
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, classification_report, recall_score, precision_score, f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def load_cvd_dataset(dataset_name, class_0_precentage=50, class_1_precentage=50):
    # load dataset
    dataset = pd.read_csv(dataset_name)

    np.random.seed(2019)
    # get desired class distribution of dataset
    if class_0_precentage > class_1_precentage:
        class_0 = round(Counter(dataset.cardio)[0])
        class_1 = round((Counter(dataset.cardio)[0] * class_1_precentage) / class_0_precentage)
        dataset = dataset.groupby('cardio').apply(lambda x: x.sample(class_0) if x.name == 0 else x.sample(class_1)).reset_index(drop=True)
    elif class_0_precentage < class_1_precentage:
        class_0 = round((Counter(dataset.cardio)[1] * class_0_precentage) / class_1_precentage)
        class_1 = round(Counter(dataset.cardio)[1])
        dataset = dataset.groupby('cardio').apply(lambda x: x.sample(class_0) if x.name == 0 else x.sample(class_1)).reset_index(drop=True)

    print(dataset.shape)

    # split dataset in features and target variable
    y = dataset.cardio  # Target variable
    print(Counter(y))
    dataset = dataset.drop('cardio', axis=1)
    feature_cols = list(dataset.columns)
    X = dataset[feature_cols]  # Features

    return X, y, feature_cols


def load_generic_dataset(dataset_name, class_0_precentage=50, class_1_precentage=50):
    # load dataset
    dataset = pd.read_csv(dataset_name)

    np.random.seed(2019)
    if class_0_precentage > class_1_precentage:
        class_0 = round(Counter(dataset.target)[0])
        class_1 = round((Counter(dataset.target)[0] * class_1_precentage) / class_0_precentage)
        dataset = dataset.groupby('target').apply(lambda x: x.sample(class_0) if x.name == 0 else x.sample(class_1)).reset_index(drop=True)
    elif class_0_precentage < class_1_precentage:
        class_0 = round((Counter(dataset.target)[1] * class_0_precentage) / class_1_precentage)
        class_1 = round(Counter(dataset.target)[1])
        dataset = dataset.groupby('target').apply(lambda x: x.sample(class_0) if x.name == 0 else x.sample(class_1)).reset_index(drop=True)

    print(dataset.shape)

    # split dataset in features and target variable
    y = dataset.target  # Target variable
    print(Counter(y))
    dataset = dataset.drop('target', axis=1)
    feature_cols = list(dataset.columns)
    X = dataset[feature_cols]  # Features

    return X, y, feature_cols


def get_percentage(current, total):
    return current / total * 100


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def plot_distribution_of_predictions(y_test, y_pred_calibrated, y_proba_calibrated_false, y_proba_calibrated_true):
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
    ax.bar(x - width/2, true_pred, width, label='Pravilna napoved')
    ax.bar(x + width/2, false_pred, width, label='Napačna napoved')
    ax.set_xlabel('Razpon verjetnosti')
    ax.set_ylabel('Relativna frekvenca')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()


def plot_net_benefit(y_test, y_proba_calibrated, plot=True):
    sigma = 0.002

    net_benefit = []
    net_benefit_all = []
    thresholds = np.arange(0, 1, 0.01)
    upper_threshold_probability = lower_threshold_probability = 0
    for thresh in thresholds:
        # evaluate each threshold
        TP, FP, FN, TN = perf_measure(y_test, to_labels(y_proba_calibrated, thresh))
        prediction_model_net_benefit = calculate_net_benefit(TP, FP, FN, TN, thresh)
        all_patients_have_disease_net_benefit = calculate_net_benefit_all(TP, FP, FN, TN, thresh)
        net_benefit.append(prediction_model_net_benefit)
        net_benefit_all.append(all_patients_have_disease_net_benefit)

        if prediction_model_net_benefit > all_patients_have_disease_net_benefit + sigma and lower_threshold_probability == 0:
            lower_threshold_probability = thresh
        if prediction_model_net_benefit <= 0 + sigma and upper_threshold_probability == 0:
            upper_threshold_probability = thresh

    nobody_has_disease = np.array([0 for i in range(len(thresholds))])

    if plot:
        plt.plot(thresholds, nobody_has_disease, label='Če predvidevamo, da noben bolnik nima bolezni')
        plt.plot(thresholds, net_benefit_all, label='Če predvidevamo, da imajo vsi bolniki bolezen')
        plt.plot(thresholds, net_benefit, label='Klasifikator')
        plt.xlabel("Prag verjetnosti")
        plt.ylabel("Korist")
        ylim = max(max(net_benefit), max(net_benefit_all), max(nobody_has_disease)) + 0.1
        plt.ylim(top=ylim, bottom=-ylim)
        plt.axvspan(lower_threshold_probability, upper_threshold_probability, color='red', alpha=0.2, label='Območje nizkega zaupanja')
        plt.axvspan(upper_threshold_probability, 1, color='lightskyblue', alpha=0.4, label='Območje visokega zaupanja')
        plt.axvspan(0, lower_threshold_probability, color='lightskyblue', alpha=0.4)
        plt.legend()
        plt.grid(True)
        plt.show()

    return upper_threshold_probability, lower_threshold_probability


def calculate_net_benefit(TP, FP, FN, TN, pt):
    n = TP + FP + FN + TN
    return (TP / n) - ((FP / n) * (pt / (1 - pt)))


def calculate_net_benefit_all(TP, FP, FN, TN, pt):
    n = TP + FP + FN + TN
    return ((TP + FN) / n) - (((FP + TN) / n) * (pt / (1 - pt)))


def plot_calibration_curve(probs, test_y):
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--', label="popolnoma kalibriran")
    # reliability diagram
    for prob_name, prob_value in probs.items():
        prob_true, prob_pred = calibration_curve(test_y, prob_value, n_bins=10)
        # plot model reliability
        plt.plot(prob_pred, prob_true, marker='.', label=prob_name)
    plt.xlabel("Napoved")
    plt.ylabel("Prava vrednost")
    plt.legend(loc="upper left")
    plt.show()


# Calculate TP, FP, FN, TN
def perf_measure(y_actual, y_hat):
    TP = FP = TN = FN = 0
    if type(y_actual) is not np.ndarray:
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


def calculate_optimal_threshold(y, proba):
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, proba)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)

    plt.plot(recall, precision, lw=2)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()

    return thresholds[ix], fscore[ix]


def get_user_score(positive_feedback, high_confidence_examples, same_feedback=False):
    if same_feedback:
        return ((positive_feedback / high_confidence_examples) * 100) * 0.5
    return (positive_feedback / high_confidence_examples) * 100


def does_user_give_same_feedback(feedback_occurrences_percentage):
    for feedback, percentage in feedback_occurrences_percentage.items():
        if percentage > 95:
            return True
    return False


def get_current_feedback(current_feedback_type, true_value, prediction):
    if current_feedback_type is 'positive':
        return 1
    elif current_feedback_type is 'negative':
        return 0
    elif current_feedback_type is 'random':
        return np.random.randint(0, 2)
    elif current_feedback_type is 'good':
        if true_value == prediction:
            return 1
        else:
            return 0


def random_forest_classification(dataset_name='cardiovascular-disease-dataset/cardio_cleaned_train.csv',
                                 model_name='generated-models/initial_model_class0_50_class1_50.sav', class_0_percentage=50,
                                 class_1_percentage=50):
    X, y, feature_cols = load_cvd_dataset(dataset_name, class_0_precentage=class_0_percentage, class_1_precentage=class_1_percentage)
    # X, y, feature_cols = load_generic_dataset(dataset_name, class_0_precentage=class_0_percentage, class_1_precentage=class_1_percentage)

    # Create a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100)
    proba = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, proba)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    plt.plot(recall, precision, lw=2)
    plt.plot(recall[ix], precision[ix], 'o', label="Optimalen prag verjetnosti: {:.4f}".format(thresholds[ix]))
    plt.xlabel("Priklic")
    plt.ylabel("Natančnost")
    plt.legend(loc="best")
    plt.title("Krivulja natančnost-priklic pri {}%:{}% (Nekalibrirana) ".format(class_0_percentage, class_1_percentage))
    plt.show()
    print('Uncalibrated model Precision Recall curve best Threshold={:.4f}, F-Score={:.4f}'.format(thresholds[ix], fscore[ix]))
    predict = to_labels(proba, thresholds[ix])

    # Calibrate Random Forest Classifier
    calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
    calibrated.fit(X, y)
    proba_calibrated_both = cross_val_predict(calibrated, X, y, cv=5, method='predict_proba')
    proba_calibrated = proba_calibrated_both[:, 1]
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, proba_calibrated)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    plt.plot(recall, precision, lw=2)
    plt.plot(recall[ix], precision[ix], 'o', label="Optimalen prag verjetnosti: {:.4f}".format(thresholds[ix]))
    plt.xlabel("Priklic")
    plt.ylabel("Natančnost")
    plt.legend(loc="best")
    plt.title("Krivulja natančnost-priklic pri {}%:{}% (Kalibrirana) ".format(class_0_percentage, class_1_percentage))
    plt.show()
    print('Calibrated model Precision Recall curve best Threshold={:.4f}, F-Score={:.4f}'.format(thresholds[ix], fscore[ix]))
    predict_calibrated = to_labels(proba_calibrated, thresholds[ix])
    predict_calibrated_default_threshold = to_labels(proba_calibrated, 0.5)
    calibrated_optimal_threshold_probability = thresholds[ix]

    # Plot calibration curve
    probabilities = {
        "nekalibriran": proba,
        "kalibriran": proba_calibrated
    }
    plot_calibration_curve(probabilities, y)

    # Plot net benefit and reliability sections, return upper and lower threshold of low reliability section
    upper_threshold_probability, lower_threshold_probability = plot_net_benefit(y, proba_calibrated)

    plot_distribution_of_predictions(y, predict_calibrated, proba_calibrated_both[:, 0], proba_calibrated_both[:, 1])

    print("Calibrated model:")
    print("Accuracy: {:.4f}, Precision [0, 1]: [{:.4f}, {:.4f}], Recall [0, 1]: [{:.4f}, {:.4f}], F1 [0, 1]: [{:.4f}, {:.4f}], RMSE: {:.4f}".format(
        accuracy_score(y, predict_calibrated),
        precision_score(y, predict_calibrated, average=None)[0],
        precision_score(y, predict_calibrated, average=None)[1],
        recall_score(y, predict_calibrated, average=None)[0],
        recall_score(y, predict_calibrated, average=None)[1],
        f1_score(y, predict_calibrated, average=None)[0],
        f1_score(y, predict_calibrated, average=None)[1],
        mean_squared_error(y, predict_calibrated, squared=False)))

    print("Calibrated model with default threshold:")
    print("Accuracy: {:.4f}, Precision [0, 1]: [{:.4f}, {:.4f}], Recall [0, 1]: [{:.4f}, {:.4f}], F1 [0, 1]: [{:.4f}, {:.4f}], RMSE: {:.4f}".format(
        accuracy_score(y, predict_calibrated_default_threshold),
        precision_score(y, predict_calibrated_default_threshold, average=None)[0],
        precision_score(y, predict_calibrated_default_threshold, average=None)[1],
        recall_score(y, predict_calibrated_default_threshold, average=None)[0],
        recall_score(y, predict_calibrated_default_threshold, average=None)[1],
        f1_score(y, predict_calibrated_default_threshold, average=None)[0],
        f1_score(y, predict_calibrated_default_threshold, average=None)[1],
        mean_squared_error(y, predict_calibrated_default_threshold, squared=False)))

    print("Uncalibrated model:")
    print("Accuracy: {:.4f}, Precision [0, 1]: [{:.4f}, {:.4f}], Recall [0, 1]: [{:.4f}, {:.4f}], F1 [0, 1]: [{:.4f}, {:.4f}], RMSE: {:.4f}".format(
        accuracy_score(y, predict),
        precision_score(y, predict, average=None)[0],
        precision_score(y, predict, average=None)[1],
        recall_score(y, predict, average=None)[0],
        recall_score(y, predict, average=None)[1],
        f1_score(y, predict, average=None)[0],
        f1_score(y, predict, average=None)[1],
        mean_squared_error(y, predict, squared=False)))

    print()

    # save the model to disk
    pickle.dump(calibrated, open(model_name, 'wb'))

    return upper_threshold_probability, calibrated_optimal_threshold_probability, lower_threshold_probability


def update_model(model='generated-models/initial_model_class0_50_class1_50.sav',
                 dataset_name='cardiovascular-disease-dataset/cardio_cleaned_test.csv',
                 feedback_type='good', scenario=1, class_0_percentage=50, class_1_percentage=50):

    print('Scenario: {}, User feedback type: {}'.format(scenario, feedback_type))

    # load the model from disk
    loaded_model = pickle.load(open(model, 'rb'))

    # Training samples
    X_initial, y_initial, feature_cols_initial = load_cvd_dataset('cardiovascular-disease-dataset/cardio_cleaned_train.csv', class_0_precentage=class_0_percentage, class_1_precentage=class_1_percentage)
    # X_initial, y_initial, feature_cols_initial = load_generic_dataset('avila/avila-tr.csv', class_0_precentage=class_0_percentage, class_1_precentage=class_1_percentage)

    # Test samples
    X, y, feature_cols = load_cvd_dataset(dataset_name, class_0_precentage=class_0_percentage, class_1_precentage=class_1_percentage)
    # X, y, feature_cols = load_generic_dataset(dataset_name)

    # Validation samples
    X_validation, y_validation, feature_cols_validation = load_cvd_dataset('cardiovascular-disease-dataset/cardio_cleaned_validation.csv', class_0_precentage=class_0_percentage, class_1_precentage=class_1_percentage)
    # X_validation, y_validation, feature_cols_validation = load_generic_dataset('avila/avila-vl.csv')

    X_initial.reset_index(drop=True, inplace=True)
    y_initial.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    y_initial = pd.Series.to_numpy(y_initial)
    y_test = pd.Series.to_numpy(y)
    y_comb = np.append(y_initial, y_test)
    X_combined = pd.concat([X_initial, X])
    X_combined.reset_index(drop=True, inplace=True)

    y_probability = loaded_model.predict_proba(X)[:, 1]
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, y_probability)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    print('Test set precision-recall curve Optimal threshold probability={:.4f}, F-Score={:.4f}'.format(thresholds[ix], fscore[ix]))

    y_combined_probability = loaded_model.predict_proba(X_combined)[:, 1]
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_comb, y_combined_probability)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    print('Combined initial and test set precision-recall curve Optimal threshold probability={:.4f}, F-Score={:.4f}'.format(thresholds[ix], fscore[ix]))
    y_prediction = to_labels(y_probability, thresholds[ix])
    threshold_probability = thresholds[ix]

    upper_threshold_probability, lower_threshold_probability = plot_net_benefit(y_comb, y_combined_probability, plot=False)

    print('Lower threshold probability: {:.4f}, Optimal threshold probability: {:.4f}, Upper threshold probability {:.4f}'.format(
        lower_threshold_probability, threshold_probability, upper_threshold_probability))

    # Model Accuracy, how often is the classifier correct
    print("Test set results:")
    print("Accuracy: {:.4f}, Precision [0, 1]: [{:.4f}, {:.4f}], Recall [0, 1]: [{:.4f}, {:.4f}], F1 [0, 1]: [{:.4f}, {:.4f}], RMSE: {:.4f}".format(
        accuracy_score(y, y_prediction),
        precision_score(y, y_prediction, average=None)[0],
        precision_score(y, y_prediction, average=None)[1],
        recall_score(y, y_prediction, average=None)[0],
        recall_score(y, y_prediction, average=None)[1],
        f1_score(y, y_prediction, average=None)[0],
        f1_score(y, y_prediction, average=None)[1],
        mean_squared_error(y, y_prediction, squared=False)))

    all_feedback = []
    high_confidence_predictions = 0
    high_confidence_predictions_positive_user_feedback = 0
    for pred, prob, true_val in zip(y_prediction, y_probability, y):
        current_feedback = get_current_feedback(feedback_type, true_val, pred)
        all_feedback.append(current_feedback)

        if lower_threshold_probability <= prob <= upper_threshold_probability:
            confidence = 'low'
        else:
            confidence = 'high'
            high_confidence_predictions += 1

        if current_feedback == 1 and confidence is 'high':
            high_confidence_predictions_positive_user_feedback += 1

    user_feedback_occurrences = Counter(all_feedback)
    user_feedback_occurrences_percentage = {}
    for i in range(0, 2):
        if i not in user_feedback_occurrences:
            user_feedback_occurrences[i] = 0
        user_feedback_occurrences_percentage[i] = get_percentage(user_feedback_occurrences[i], len(all_feedback))

    same_feedback = does_user_give_same_feedback(user_feedback_occurrences_percentage)
    user_score = get_user_score(high_confidence_predictions_positive_user_feedback, high_confidence_predictions, same_feedback)
    user_weight = user_score / 100
    print('User score: {}'.format(user_score))

    sample_weights = []
    feedback = []
    indices = []
    index = 0
    for pred, prob, true_val in zip(y_prediction, y_probability, y):
        if lower_threshold_probability <= prob <= upper_threshold_probability:
            confidence = 'low'
        else:
            confidence = 'high'

        current_feedback = get_current_feedback(feedback_type, true_val, pred)

        if scenario == 1:
            if user_score >= 50:
                feedback.append(current_feedback)
                sample_weights.append(user_weight)
            else:
                indices.append(index)
        elif scenario == 2:
            feedback.append(current_feedback)
            sample_weights.append(user_weight)
        elif scenario == 3:
            if confidence is 'low' and user_score >= 50:
                feedback.append(current_feedback)
                sample_weights.append(user_weight)
            else:
                indices.append(index)

        index += 1

    # Use only examples with feedback
    y_prediction = np.delete(y_prediction, indices)
    y = pd.Series.to_numpy(y)
    y = np.delete(y, indices)
    X = X.drop(indices)

    if len(feedback) != 0:
        y_prediction_correction = [p if fb == 1 else 1 - p for p, fb in zip(y_prediction, feedback)]

        X_combined = pd.concat([X_initial, X])
        X_combined.reset_index(drop=True, inplace=True)

        sample_weights_initial = [1] * len(y_initial)
        feedback_sample_weights = sample_weights_initial + sample_weights

        # Predict using feedback
        clf = RandomForestClassifier(n_estimators=100)
        calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
        calibrated.fit(X_combined, np.append(y_initial, y_prediction_correction), sample_weight=feedback_sample_weights)
        proba_calibrated_both = calibrated.predict_proba(X_validation)
        proba_calibrated = proba_calibrated_both[:, 1]
        predict_calibrated = to_labels(proba_calibrated, threshold_probability)

        # Predict using truth values for comparison
        clf_without_feedback = RandomForestClassifier(n_estimators=100)
        calibrated_without_feedback = CalibratedClassifierCV(clf_without_feedback, method='sigmoid', cv=5)
        calibrated_without_feedback.fit(X_combined, np.append(y_initial, y))
        proba_calibrated_both_without_feedback = calibrated_without_feedback.predict_proba(X_validation)
        proba_calibrated_without_feedback = proba_calibrated_both_without_feedback[:, 1]
        predict_calibrated_without_feedback = to_labels(proba_calibrated_without_feedback, threshold_probability)

        print('Initial data size: {}, Number of given feedback: {}, Validation size: {}'.format(len(y_initial), len(feedback), len(y_validation)))

        print("Updated with feedback:")
        print("Accuracy: {:.4f}, Precision [0, 1]: [{:.4f}, {:.4f}], Recall [0, 1]: [{:.4f}, {:.4f}], F1 [0, 1]: [{:.4f}, {:.4f}], RMSE: {:.4f}".format(
                accuracy_score(y_validation, predict_calibrated),
                precision_score(y_validation, predict_calibrated, average=None)[0],
                precision_score(y_validation, predict_calibrated, average=None)[1],
                recall_score(y_validation, predict_calibrated, average=None)[0],
                recall_score(y_validation, predict_calibrated, average=None)[1],
                f1_score(y_validation, predict_calibrated, average=None)[0],
                f1_score(y_validation, predict_calibrated, average=None)[1],
                mean_squared_error(y_validation, predict_calibrated, squared=False)))

        print("Updated without feedback:")
        print("Accuracy: {:.4f}, Precision [0, 1]: [{:.4f}, {:.4f}], Recall [0, 1]: [{:.4f}, {:.4f}], F1 [0, 1]: [{:.4f}, {:.4f}], RMSE: {:.4f}".format(
            accuracy_score(y_validation, predict_calibrated_without_feedback),
            precision_score(y_validation, predict_calibrated_without_feedback, average=None)[0],
            precision_score(y_validation, predict_calibrated_without_feedback, average=None)[1],
            recall_score(y_validation, predict_calibrated_without_feedback, average=None)[0],
            recall_score(y_validation, predict_calibrated_without_feedback, average=None)[1],
            f1_score(y_validation, predict_calibrated_without_feedback, average=None)[0],
            f1_score(y_validation, predict_calibrated_without_feedback, average=None)[1],
            mean_squared_error(y_validation, predict_calibrated_without_feedback, squared=False)))
        print()
    else:
        print('No feedback received.')
        print()


if __name__ == '__main__':
    scenarios = [1, 2, 3]
    feedback_types = ['negative', 'positive', 'random', 'good']
    class_0_perc = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    class_1_perc = [90, 80, 70, 60, 50, 40, 30, 20, 10]
    for c_0, c_1 in zip(class_0_perc, class_1_perc):
        print()
        print()
        print('Calculating for class imbalance - class 0: {}%, class 1: {}%'.format(c_0, c_1))
        model_name = 'generated-models/initial_model_class0_' + str(c_0) + '_class1_' + str(c_1) + '.sav'
        upper_threshold_probability, threshold_probability, lower_threshold_probability = random_forest_classification(
            model_name=model_name, class_0_percentage=c_0, class_1_percentage=c_1)
        for scenario in scenarios:
            for feedback_type in feedback_types:
                update_model(model=model_name, feedback_type=feedback_type, scenario=scenario, class_0_percentage=c_0, class_1_percentage=c_1)
