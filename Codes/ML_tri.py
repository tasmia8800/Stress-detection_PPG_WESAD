import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import ml_utils

WINDOW_SIZE = '120'
NOISE = ['bp_time_ens']
SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
if os.environ.get('WESAD_FAST') == 'True':
    SUBJECTS = [2, 3, 4]
N_CLASSES = 3
TASK_NAME = "Tri-class Classification"

def main():
    classifiers = ml_utils.get_classifiers()
    classifiers_order = list(classifiers.keys())
    
    for n in NOISE:
        path = '27_features_ppg_test_3/LMM/data_merged_' + n + WINDOW_SIZE + '.csv'
        result_path_all = 'result_3/LMM/all_features_' + n + WINDOW_SIZE + '.csv'
        
        metrics_storage = {clf: {'AUC': [], 'F1': [], 'ACC': []} for clf in classifiers}

        for sub in SUBJECTS:
            df, X_train, y_train, X_test, y_test = ml_utils.read_csv_data(
                path, ml_utils.FEATS, sub
            )
            df.fillna(0, inplace=True)
            
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            for clf_name, clf_model in classifiers.items():
                auc, f1, acc = ml_utils.evaluate_model(
                    clf_model, X_train, y_train, X_test, y_test, N_CLASSES
                )
                metrics_storage[clf_name]['AUC'].append(auc)
                metrics_storage[clf_name]['F1'].append(f1)
                metrics_storage[clf_name]['ACC'].append(acc)

        ml_utils.save_and_print_results(
            TASK_NAME, n, result_path_all, metrics_storage, classifiers_order
        )

if __name__ == "__main__":
    main()
