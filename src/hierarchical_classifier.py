import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from hiclass import LocalClassifierPerNode
import pickle
from utils.data_utils import get_split

RANDOM_STATE = 42

def classify_datasets():
    folders = ['suno', 'udio', 'lastfm']

    X_train, y_train_orig = get_split('train', 'clap-laion-music', folders)
    X_val, y_val_orig = get_split('val', 'clap-laion-music', folders)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # add parent to classes (e.g. 'suno' -> ['AI', 'suno'], 'udio' -> ['AI', 'udio'], 'lastfm' -> ['nonAI', 'lastfm'])
    class_hierarchy = {
        'AI': ['suno', 'udio'],
        'nonAI': ['lastfm']
    }

    y_train = np.array([['AI', folder] for folder in y_train_orig if folder in class_hierarchy['AI']] + [['nonAI', folder] for folder in y_train_orig if folder in class_hierarchy['nonAI']])
    y_val = np.array([['AI', folder] for folder in y_val_orig if folder in class_hierarchy['AI']] + [['nonAI', folder] for folder in y_val_orig if folder in class_hierarchy['nonAI']])

    base_estimators = {
        'svc': SVC(probability=True, random_state=RANDOM_STATE),
        'rf': RandomForestClassifier(random_state=RANDOM_STATE),
        'knn': KNeighborsClassifier()
    }

    results = {}
    models = {}

    for name, base_estimator in base_estimators.items():
        clf = LocalClassifierPerNode(
            local_classifier=base_estimator,
            binary_policy="inclusive"  # use inclusive policy for binary classifiers
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        results_parent = classification_report(y_val[:, 0], y_pred[:, 0], output_dict=True)
        results_children = classification_report(y_val[:, 1], y_pred[:, 1], output_dict=True)
        results[name] = {'parent': results_parent, 'children': results_children}
        models[name] = clf

    # Save results
    with open('artifacts/classification_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save models and scaler
    with open('artifacts/models_and_scaler.pkl', 'wb') as f:
        pickle.dump({'models': models, 'scaler': scaler}, f)

if __name__ == '__main__':
    classify_datasets()
