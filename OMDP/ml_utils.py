import pandas as pd
import numpy as np
import csv
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, accuracy_score


FEATS_ORIGINAL = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','SDSD','RMSSD','pNN20','pNN50','TINN','LF','HF','ULF','VLF','LFHF',
         'total_power','lfp','hfp','SD1','SD2','pA','pQ','ApEn','shanEn','D2',
         'subject','label']

FEATS = FEATS_ORIGINAL

def get_classifiers():
    """Returns a dictionary of configured classifiers."""
    return {
        'DT': tree.DecisionTreeClassifier(random_state=0),
        'RF': RandomForestClassifier(max_depth=4, random_state=0),
        'AB': AdaBoostClassifier(random_state=0),
        'KN': KNeighborsClassifier(n_neighbors=9),
        'LDA': LinearDiscriminantAnalysis(),
        'SVM': svm.SVC(),
        'GB': GradientBoostingClassifier(random_state=0)
    }

def read_csv_data(path, feats, testset_num):
    df = pd.read_csv(path, index_col=0)
    
    available_feats = [f for f in feats if f in df.columns]
    if len(available_feats) != len(feats):
        missing = set(feats) - set(available_feats)
        if missing:
             print(f"Warning: Missing features in CSV: {missing}")
    
    df = df[available_feats]

    train_df = df.loc[df['subject'] != testset_num]
    test_df = df.loc[df['subject'] == testset_num]

    X_train = train_df.drop(['label', 'subject'], axis=1).values
    y_train = train_df['label'].values
    X_test = test_df.drop(['label', 'subject'], axis=1).values
    y_test = test_df['label'].values

    return df, X_train, y_train, X_test, y_test

def evaluate_model(model, X_train, y_train, X_test, y_test, n_classes):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    if n_classes == 2:
         f1 = f1_score(y_test, y_pred)
    else:
        f1 = f1_score(y_test, y_pred, average='macro')

    if n_classes == 2:
        try:
            roc_auc = roc_auc_score(y_test, y_pred)
        except ValueError:
            roc_auc = 0.0 
    else:
        y_pred_onehot = np.eye(n_classes)[y_pred]
        y_test_onehot = np.eye(n_classes)[y_test]
        
        fpr, tpr, roc_auc_dict = dict(), dict(), dict()
        for i in range(n_classes):
            if np.sum(y_test_onehot[:, i]) == 0:
                 pass
            
            try:
                fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], y_pred_onehot[:, i])
                roc_auc_dict[i] = auc(fpr[i], tpr[i])
            except:
                roc_auc_dict[i] = 0.0

        roc_auc = np.array(list(roc_auc_dict.values())).mean()

    return roc_auc, f1, accuracy

def save_and_print_results(task_name, noise_label, result_path, metrics_storage, classifiers_order):
    with open(result_path, 'w', newline='') as file:
        writer = csv.writer(file)
        
        header = ['subject','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S13','S14','S15','S16','S17','total']
        writer.writerow(header)
        
        for metric in ['AUC', 'F1', 'ACC']:
            for clf in classifiers_order:
                data = metrics_storage[clf][metric]
                mean_val = np.mean(data)
                row_label = f"{clf}_{metric}"
                writer.writerow([row_label] + data + [mean_val])
                
    print(f"\n--- {task_name} Results ({noise_label}) ---")
    print(f"{'Classifier':<10} {'Accuracy':<10} {'F1 Score':<10}")
    print("-" * 35)
    
    summary_data = [] 
    
    for clf in classifiers_order:
        acc = np.mean(metrics_storage[clf]['ACC'])
        f1 = np.mean(metrics_storage[clf]['F1'])
        print(f"{clf:<10} {acc:.4f}     {f1:.4f}")
        
        summary_data.append({
            'Task': task_name,
            'Noise': noise_label,
            'Classifier': clf,
            'Accuracy': acc,
            'F1_Score': f1
        })
        
    print("-" * 35)
    print(f"Detailed results saved to: {result_path}\n")

    summary_file = 'result/baseline_results.csv'
        
    file_exists = os.path.isfile(summary_file)
    
    with open(summary_file, 'a', newline='') as f:
        fieldnames = ['Task', 'Noise', 'Classifier', 'Accuracy', 'F1_Score']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(summary_data)

