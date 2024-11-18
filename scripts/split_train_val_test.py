import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from functools import partial

def process_file(file, folder_path, split_criteria, min_duration):
    """
    Process a single JSON file and return the extracted data.
    
    Args:
        file (str): Filename to process
        folder_path (str): Path to the folder containing JSON files
        split_criteria (str): Either 'prompt' or 'artist'
        min_duration (int): Minimum song duration in seconds
    
    Returns:
        dict: Extracted data or None if invalid
    """
    try:
        with open(os.path.join(folder_path, file), 'r') as f:
            data = json.load(f)
            
            # Validate duration
            if not isinstance(data.get('duration'), (float, int)):
                return None
                
            if data['duration'] < min_duration:
                return None
            
            # Get the required field based on split_criteria
            required_field = data.get('prompt' if split_criteria == 'prompt' else 'artist')
            
            # Ensure required field exists and is non-empty
            if not required_field or not isinstance(required_field, str):
                return None
                
            return {
                'filename': file[:-5],
                split_criteria: required_field,
                'duration': data['duration']
            }
    except (json.JSONDecodeError, KeyError):
        return None

def load_music_data_parallel(folder_path, split_criteria, min_duration=60, max_songs=10000, random_state=42):
    """
    Load music data from JSON files in parallel.
    
    Args:
        folder_path (str): Path to the folder containing JSON files
        split_criteria (str): Either 'prompt' or 'artist' to specify dataset type
        min_duration (int): Minimum song duration in seconds
        max_songs (int): Maximum number of songs to include
        random_state (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: DataFrame containing filtered song data
    """
    assert split_criteria in ['prompt', 'artist'], "split_criteria must be either 'prompt' or 'artist'"
    
    files = os.listdir(folder_path)
    print('total files:', len(files))
    
    # Use 75% of available CPU cores to avoid system overload
    n_cores = max(1, int(cpu_count() * 0.75))
    print(f"Using {n_cores} CPU cores for parallel processing")
    
    # Create a partial function with fixed arguments
    process_func = partial(process_file, 
                         folder_path=folder_path,
                         split_criteria=split_criteria,
                         min_duration=min_duration)
    
    # Process files in parallel
    with Pool(n_cores) as pool:
        results = pool.map(process_func, files)
    
    # Filter out None results and create DataFrame
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    
    # Sample if we have more than max_songs
    if len(df) > max_songs:
        df = df.sample(n=max_songs, random_state=random_state)
    
    return df

def create_dataset_splits(df, split_criteria, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        split_criteria (str): Either 'prompt' or 'artist' to specify split method
        train_size (float): Proportion for training set
        val_size (float): Proportion for validation set
        test_size (float): Proportion for test set
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    assert split_criteria in ['prompt', 'artist'], "split_criteria must be either 'prompt' or 'artist'"
    assert np.isclose(train_size + val_size + test_size, 1.0), "Split proportions must sum to 1"
    
    if split_criteria == 'artist':
        # artist-based splitting
        unique_items = df[split_criteria].unique()
        print('unique artist:',len(unique_items))
        
        # Split artists
        train_items, temp_items = train_test_split(
            unique_items,
            train_size=train_size,
            random_state=random_state,
            shuffle=True
        )
        
        relative_val_size = val_size / (val_size + test_size)
        val_items, test_items = train_test_split(
            temp_items,
            train_size=relative_val_size,
            random_state=random_state,
            shuffle=True
        )
        
        # Create splits based on artists
        train_df = df[df[split_criteria].isin(train_items)]
        val_df = df[df[split_criteria].isin(val_items)]
        test_df = df[df[split_criteria].isin(test_items)]

        # Print detailed statistics
        print(f"\nDataset splits (by {split_criteria}):")
        print(f"Training set: {len(train_df)} songs ({len(train_df)/len(df):.1%})")
        print(f"Validation set: {len(val_df)} songs ({len(val_df)/len(df):.1%})")
        print(f"Test set: {len(test_df)} songs ({len(test_df)/len(df):.1%})")
        
    else:
        # Random splitting for prompt-based dataset
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            random_state=random_state,
            shuffle=True
        )
        
        relative_val_size = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=relative_val_size,
            random_state=random_state,
            shuffle=True
        )
        
        print(f"Dataset splits (by {split_criteria}):")
        print(f"Training set: {len(train_df)} songs ({len(train_df)/len(df):.1%})")
        print(f"Validation set: {len(val_df)} songs ({len(val_df)/len(df):.1%})")
        print(f"Test set: {len(test_df)} songs ({len(test_df)/len(df):.1%})")
    
    return train_df, val_df, test_df

def save_splits_to_txt(train_df, val_df, test_df, output_dir):
    """
    Save the filenames from each split to separate txt files.
    
    Args:
        train_df (pd.DataFrame): Training set DataFrame
        val_df (pd.DataFrame): Validation set DataFrame
        test_df (pd.DataFrame): Test set DataFrame
        output_dir (str): Directory to save the txt files
    """
    # Save train set
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_df['filename'].values))
    
    # Save validation set
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_df['filename'].values))
    
    # Save test set
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_df['filename'].values))
    
    print(f"Split files saved to {output_dir}")

def process_dataset(name, folder, output_dir, split_criteria):
    """
    Process a single dataset completely.
    
    Args:
        name (str): Name of the dataset for logging
        folder (str): Input folder path
        output_dir (str): Output directory path
        split_criteria (str): Either 'prompt' or 'artist'
    """
    print(f"\nProcessing {name} dataset...")
    df = load_music_data_parallel(folder, split_criteria=split_criteria, max_songs=10000)
    train_df, val_df, test_df = create_dataset_splits(df, split_criteria=split_criteria)
    save_splits_to_txt(train_df, val_df, test_df, output_dir)

if __name__ == '__main__':
    # Define dataset configurations
    datasets = [
        {
            'name': 'Suno',
            'folder': '/data/suno/metadata',
            'output': '/data/suno',
            'split_criteria': 'prompt'
        },
        {
            'name': 'Udio',
            'folder': '/data/udio/metadata',
            'output': '/data/udio',
            'split_criteria': 'prompt'
        },
        {
            'name': 'Lastfm',
            'folder': '/data/lastfm/metadata',
            'output': '/data/lastfm',
            'split_criteria': 'artist'
        }
    ]
    
    # Process each dataset
    for dataset in datasets:
        process_dataset(
            dataset['name'],
            dataset['folder'],
            dataset['output'],
            dataset['split_criteria']
        )
