# %%
import numpy as np
import json
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from utils.data_utils import get_split

def load_ircamplify_results(folders):
    true_class = []
    files = []
    is_ai = []
    confidence = []
    for folder in folders:
        folder_path = f'/data/ircamplify_results/{folder}'
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                with open(os.path.join(folder_path, filename), 'r') as f:
                    data = json.load(f)
                    job_infos = data.get('job_infos', {})
                    file_paths = job_infos.get('file_paths', {})
                    report_info = job_infos.get('report_info', {})
                    report = report_info.get('report', {})
                    result_list = report.get('resultList', [])
                    
                    for i, result in enumerate(result_list):
                        true_class.append(folder)
                        file = file_paths[i].split('/')[-1]
                        files.append(file)
                        is_ai.append(result.get('isAi'))
                        confidence.append(result.get('confidence'))
    # make it into a dataframe
    data = {
        'true_class': true_class,
        'file': files,
        'is_ai': is_ai,
        'confidence': confidence
    }
    data = pd.DataFrame(data)
    return data

def get_classifiers_results(models, X_sample_scaled, sample_files):
    true_class = []
    files = []
    svm_pred_parent = []
    svm_pred_child = []
    rf_pred_parent = []
    rf_pred_child = []
    knn_pred_parent = []
    knn_pred_child = []

    for i, file in enumerate(sample_files):
        true_class.append(file.split('/')[-5])
        files.append(file.split('/')[-1].replace('npy','mp3'))
    for name, model in models.items():
        y_pred = model.predict(X_sample_scaled)
        for i, file in enumerate(sample_files):
            if name == 'svc':
                svm_pred_parent.append(y_pred[i, 0])
                svm_pred_child.append(y_pred[i, 1])
            elif name == 'rf':
                rf_pred_parent.append(y_pred[i, 0])
                rf_pred_child.append(y_pred[i, 1])
            elif name == 'knn':
                knn_pred_parent.append(y_pred[i, 0])
                knn_pred_child.append(y_pred[i, 1])


    data = {
        'true_class': true_class,
        'file': files,
        'svm_pred_parent': svm_pred_parent,
        'svm_pred_child': svm_pred_child,
        'rf_pred_parent': rf_pred_parent,
        'rf_pred_child': rf_pred_child,
        'knn_pred_parent': knn_pred_parent,
        'knn_pred_child': knn_pred_child
    }
    data = pd.DataFrame(data)
    return data

def get_results_all(folders = ['suno', 'udio', 'lastfm']):   
    # Load trained models and scaler
    with open('models_and_scaler.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    models = saved_data['models']
    scaler = saved_data['scaler']

    # Load sample data
    if 'boomy' in folders:
        without_boomy = [folder for folder in folders if folder != 'boomy']
        X_sample, y_sample, sample_files = get_split('sample', 'clap-laion-music', without_boomy)
        X_boomy, y_boomy, sample_files_boomy = get_split('sample', 'clap-laion-music', ['boomy'])
        X_sample = np.concatenate((X_sample, X_boomy))
        sample_files = sample_files + sample_files_boomy
    else:
        X_sample, y_sample, sample_files = get_split('sample', 'clap-laion-music', folders)
    
    X_sample_scaled = scaler.transform(X_sample)

    # classifier results
    classifiers_results = get_classifiers_results(models, X_sample_scaled, sample_files)

    #  how many rows have 'suno', 'udio', 'lastfm' as the true class
    # print(classifiers_results['true_class'].value_counts())
    # if there are ircamplify results, compare
    if os.path.exists('/data/ircamplify_results/'):
        # ircamplify results
        ircamplify_results = load_ircamplify_results(folders)
        # remove rows that have repeated files in ircamplify results
        ircamplify_results = ircamplify_results.drop_duplicates(subset='file', keep='first')
        merged_data = pd.merge(classifiers_results, ircamplify_results, on=['true_class', 'file'], how='left')
        print('length of merged data:', len(merged_data))
        return merged_data
    else:
        return classifiers_results

# Function to print classification report in LaTeX format
def print_classification_report_latex(data):
    y_true = data['true_class']
    y_pred_svm_parent = data['svm_pred_parent']
    y_pred_rf_parent = data['rf_pred_parent']
    y_pred_knn_parent = data['knn_pred_parent']
    if 'is_ai' in data:
        y_pred_ai = data['is_ai']

    # Convert true_class to AI vs non-AI binary classification
    y_true_ai = np.array([False if label == 'lastfm' else True for label in y_true])
    
    y_pred_svm_ai = np.array([True if label == 'AI' else False for label in y_pred_svm_parent])
    y_pred_rf_ai = np.array([True if label == 'AI' else False for label in y_pred_rf_parent])
    y_pred_knn_ai = np.array([True if label == 'AI' else False for label in y_pred_knn_parent])

    # Confusion matrices
    if 'is_ai' in data:
        classifiers = [
            ("SVM Classifier", y_pred_svm_ai),
            ("RF Classifier", y_pred_rf_ai),
            ("KNN Classifier", y_pred_knn_ai),
            ("Ircam Amplify Classifier", y_pred_ai)
        ]
    else:
        classifiers = [
            ("SVM Classifier", y_pred_svm_ai),
            ("RF Classifier", y_pred_rf_ai),
            ("KNN Classifier", y_pred_knn_ai),
        ]
    for (title, y_pred) in classifiers:
        cm = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
        cm = cm.iloc[:-1, :-1]
        cm = cm.div(cm.sum(axis=1), axis=0)
    
        # print the confusion matrix as latex table
        print(f"{title}:\n{cm.to_latex()}\n")

    # Create classification report for each classifier
    svm_report = classification_report(y_true_ai, y_pred_svm_ai, output_dict=True)
    rf_report = classification_report(y_true_ai, y_pred_rf_ai, output_dict=True)
    knn_report = classification_report(y_true_ai, y_pred_knn_ai, output_dict=True)
    if 'is_ai' in data:
        ai_report = classification_report(y_true_ai, y_pred_ai, output_dict=True)

    # Combine reports into a LaTeX table
    table = r"\begin{table}[ht]\centering\begin{tabular}{lcccc}\hline"
    table += "\n"
    table += r"Classifier & Precision & Recall & F1-Score & Accuracy \\ \hline"
    table += "\n"
    if 'is_ai' in data:
        classifiers = [("SVM", svm_report), ("RF", rf_report), ("KNN", knn_report), ("Ircam Amp.", ai_report)]
    else:
        classifiers = [("SVM", svm_report), ("RF", rf_report), ("KNN", knn_report)]

    for classifier, report in classifiers:
        precision = report['True']['precision']
        recall = report['True']['recall']
        f1_score = report['True']['f1-score']
        accuracy = report['accuracy']
        table += f"{classifier} & {precision:.3f} & {recall:.3f} & {f1_score:.3f} & {accuracy:.3f} \\\\ \hline \n"

    table += r"\end{tabular}\caption{Parent-level classification results (AI vs. non-AI) on the validation set}\end{table}"

    print(table)

    # do the same on the child level (suno udio lastfm)
    y_pred_svm_child = data['svm_pred_child']
    y_pred_rf_child = data['rf_pred_child']
    y_pred_knn_child = data['knn_pred_child']

    # Create classification report for each classifier
    svm_report = classification_report(y_true, y_pred_svm_child, output_dict=True)
    rf_report = classification_report(y_true, y_pred_rf_child, output_dict=True)
    knn_report = classification_report(y_true, y_pred_knn_child, output_dict=True)

    # Combine reports into a LaTeX table
    table = r"\begin{table}[ht]\centering\begin{tabular}{lcccc}\hline"
    table += "\n"
    table += r"Classifier & Precision & Recall & F1-Score & Accuracy \\ \hline"
    table += "\n"
    classifiers = [("SVM", svm_report), ("RF", rf_report), ("KNN", knn_report)]

    for classifier, report in classifiers:
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1_score = report['macro avg']['f1-score']
        accuracy = report['accuracy']
        table += f"{classifier} & {precision:.3f} & {recall:.3f} & {f1_score:.3f} & {accuracy:.3f} \\\\ \hline \n"

    table += r"\end{tabular}\caption{Child-level classification results (LastFM, Suno, Udio) on the validation set}\end{table}"

    print(table)

    # Generate detailed child-level classification report for each classifier
    classifiers = [("SVM", svm_report), ("RF", rf_report), ("KNN", knn_report)]

    # Create the LaTeX table
    detailed_table = r"\begin{table}[ht]\centering\begin{tabular}{llccc}\hline"
    detailed_table += "\n"
    detailed_table += r"Classifier & Category & Precision & Recall & F1-score \\ \hline"
    detailed_table += "\n"

    # Loop through each classifier and its corresponding classification report
    for classifier, report in classifiers:
        detailed_table += f"\\multirow{{3}}{{*}}{{{classifier}}} "

        # Loop through each category (LastFM, Suno, Udio)
        for idx, category in enumerate(['lastfm', 'suno', 'udio']):
            if idx > 0:  # Add a new row for categories other than the first
                detailed_table += " & "
            precision = report[category]['precision']
            recall = report[category]['recall']
            f1_score = report[category]['f1-score']
            detailed_table += f"{category.capitalize()} & {precision:.3f} & {recall:.3f} & {f1_score:.3f} \\\\ \n"
        detailed_table += r"\hline \n"

    detailed_table += r"\end{tabular}\caption{Detailed child-level classification results (LastFM, Suno, Udio) on the validation set}\end{table}"

    # Print the detailed table
    print(detailed_table)

# Example usage
if __name__ == "__main__":
    with_boomy = False

    if with_boomy:
        folders = ['suno', 'udio', 'lastfm', 'boomy']
    else:
        folders = ['suno', 'udio', 'lastfm']

    data = get_results_all(folders)

    print_classification_report_latex(data)
