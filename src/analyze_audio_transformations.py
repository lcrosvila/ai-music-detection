import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from hiclass import LocalClassifierPerNode
from utils.data_utils import get_split

def evaluate_classifier(clf, X, y, class_hierarchy):
    y_pred = clf.predict(X)
    # Evaluate parent level
    y_parent = y[:, 0]
    y_pred_parent = y_pred[:, 0]
    
    parent_metrics = {
        'accuracy': accuracy_score(y_parent, y_pred_parent),
        'precision': precision_score(y_parent, y_pred_parent, average='weighted'),
        'recall': recall_score(y_parent, y_pred_parent, average='weighted'),
        'f1': f1_score(y_parent, y_pred_parent, average='weighted')
    }
    
    # Evaluate child level
    y_child = y[:, 1]
    y_pred_child = y_pred[:, 1]

    child_metrics = {
        'accuracy': accuracy_score(y_child, y_pred_child),
        'precision': precision_score(y_child, y_pred_child, average='weighted'),
        'recall': recall_score(y_child, y_pred_child, average='weighted'),
        'f1': f1_score(y_child, y_pred_child, average='weighted')
    }

    # evaluate parent level for class (suno, udio, lastfm)
    fine_metrics_parent = {
        'suno': {'accuracy': accuracy_score(y_parent[y_child == 'suno'], y_pred_parent[y_child == 'suno']),
                 'precision': precision_score(y_parent[y_child == 'suno'], y_pred_parent[y_child == 'suno'], average='weighted'),
                 'recall': recall_score(y_parent[y_child == 'suno'], y_pred_parent[y_child == 'suno'], average='weighted'),
                 'f1': f1_score(y_parent[y_child == 'suno'], y_pred_parent[y_child == 'suno'], average='weighted')},
        'udio': {'accuracy': accuracy_score(y_parent[y_child == 'udio'], y_pred_parent[y_child == 'udio']),
                 'precision': precision_score(y_parent[y_child == 'udio'], y_pred_parent[y_child == 'udio'], average='weighted'),
                 'recall': recall_score(y_parent[y_child == 'udio'], y_pred_parent[y_child == 'udio'], average='weighted'),
                 'f1': f1_score(y_parent[y_child == 'udio'], y_pred_parent[y_child == 'udio'], average='weighted')},
        'lastfm': {'accuracy': accuracy_score(y_parent[y_child == 'lastfm'], y_pred_parent[y_child == 'lastfm']),
                   'precision': precision_score(y_parent[y_child == 'lastfm'], y_pred_parent[y_child == 'lastfm'], average='weighted'),
                   'recall': recall_score(y_parent[y_child == 'lastfm'], y_pred_parent[y_child == 'lastfm'], average='weighted'),
                   'f1': f1_score(y_parent[y_child == 'lastfm'], y_pred_parent[y_child == 'lastfm'], average='weighted')}
    }

    # evaluate child level for class (suno, udio, lastfm)
    fine_metrics_child = {
        'suno': {'accuracy': accuracy_score(y_child[y_child == 'suno'], y_pred_child[y_child == 'suno']),
                 'precision': precision_score(y_child[y_child == 'suno'], y_pred_child[y_child == 'suno'], average='weighted'),
                 'recall': recall_score(y_child[y_child == 'suno'], y_pred_child[y_child == 'suno'], average='weighted'),
                 'f1': f1_score(y_child[y_child == 'suno'], y_pred_child[y_child == 'suno'], average='weighted')},
        'udio': {'accuracy': accuracy_score(y_child[y_child == 'udio'], y_pred_child[y_child == 'udio']),
                 'precision': precision_score(y_child[y_child == 'udio'], y_pred_child[y_child == 'udio'], average='weighted'),
                 'recall': recall_score(y_child[y_child == 'udio'], y_pred_child[y_child == 'udio'], average='weighted'),
                 'f1': f1_score(y_child[y_child == 'udio'], y_pred_child[y_child == 'udio'], average='weighted')},
        'lastfm': {'accuracy': accuracy_score(y_child[y_child == 'lastfm'], y_pred_child[y_child == 'lastfm']),
                   'precision': precision_score(y_child[y_child == 'lastfm'], y_pred_child[y_child == 'lastfm'], average='weighted'),
                   'recall': recall_score(y_child[y_child == 'lastfm'], y_pred_child[y_child == 'lastfm'], average='weighted'),
                   'f1': f1_score(y_child[y_child == 'lastfm'], y_pred_child[y_child == 'lastfm'], average='weighted')}
    }
    
    return parent_metrics, child_metrics, fine_metrics_parent, fine_metrics_child

def get_transformed(transformation, param, split, folders):
    files = []
    y = []
    for folder in folders:
        with open(f'/data/{folder}/{split}.txt', 'r') as f:
            folder_files = f.read().splitlines()
        files.extend([f'/data/{folder}/audio/transformed/{transformation}_{param}/{file}.npy' for file in folder_files])
        y.extend([folder] * len(folder_files))
    
    X = load_embeddings(files)
    y = np.array(y)
    print(f'Loaded {len(X), len(y)} samples.')
    return X, y

def main():
    split = 'test'
    with open('models_and_scaler.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    
    models = saved_data['models']
    scaler = saved_data['scaler']
    
    folders = ['suno', 'udio', 'lastfm']
    class_hierarchy = {
        'AI': ['suno', 'udio'],
        'nonAI': ['lastfm']
    }
    cutoffs = [100, 500, 1000, 3000, 5000, 8000, 10000, 12000, 16000, 20000]
    transformations = {
        'original': [split],
        'low_pass': cutoffs,
        'high_pass': cutoffs,
        'decrease_sr': [8000, 16000, 22050, 24000, 44100]
    }
    
    results = {model_name: {trans: {param: {'parent': {}, 'child': {}} for param in params} 
                            for trans, params in transformations.items()} 
               for model_name in models.keys()}
    
    for trans, params in transformations.items():
        for param in params:
            print(f"\nEvaluating on {trans} {param}:")
            if trans == 'original':
                X, y = get_split(split, 'clap-laion-music', folders)
            else:
                X, y = get_transformed(trans, param, split, folders)
            
            X_scaled = scaler.transform(X)
            y =  np.array([['AI', folder] for folder in y if folder in class_hierarchy['AI']] + 
                          [['nonAI', folder] for folder in y if folder in class_hierarchy['nonAI']])

            for model_name, model in models.items():
                print(f"\n{model_name.upper()} Classifier:")
                parent_metrics, child_metrics, fine_metrics_parent, fine_metrics_child = evaluate_classifier(model, X_scaled, y, class_hierarchy)
                results[model_name][trans][param]['parent'] = parent_metrics
                results[model_name][trans][param]['child'] = child_metrics
                results[model_name][trans][param]['fine_parent'] = fine_metrics_parent
                results[model_name][trans][param]['fine_child'] = fine_metrics_child
                
                print("Parent level metrics:")
                for metric, value in parent_metrics.items():
                    print(f"{metric}: {value:.4f}")
                
                print("\nChild level metrics:")
                for metric, value in child_metrics.items():
                    print(f"{metric}: {value:.4f}")
                
                print("\nFine level metrics:")
                for class_, metrics in fine_metrics_parent.items():
                    print(f"Class: {class_}")
                    for metric, value in metrics.items():
                        print(f"{metric}: {value:.4f}")

                
    
    with open('evaluation_results_hierarchical_test.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nEvaluation results have been saved to 'evaluation_results_hierarchical_test.pkl'")

if __name__ == "__main__":
    main()
