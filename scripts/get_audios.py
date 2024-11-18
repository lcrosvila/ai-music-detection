import json
import os
import pandas as pd
import argparse
import concurrent.futures

def get_unique_songs(metadata_path):
    metadata_path = os.path.join(metadata_path, 'refs')
    metadata_files = [file for file in os.listdir(metadata_path) if file.endswith('.json')]

    metadata = {}
    mp3s = {}
    for file in metadata_files:
        file_path = os.path.join(metadata_path, file)
        print(f"Processing file: {file_path}")
        try:
            with open(file_path) as f:
                content = f.read()
                # Check for the "Too many requests." message
                if "Too many requests." in content:
                    print(f"File {file_path} contains 'Too many requests.' message. Deleting file.")
                    os.remove(file_path)
                    continue

                # Attempt to parse the content as JSON
                metadata[file] = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Skipping file due to JSON error: {file_path} - {str(e)}")
            continue

        # Process the loaded JSON data (remaining part of your code)
        if 'playlist_clips' in metadata[file]:
            for item in metadata[file]['playlist_clips']:
                item = item['clip']
                mp3s[item['audio_url']] = item
                # Unroll item['metadata']
                for key, value in item['metadata'].items():
                    mp3s[item['audio_url']][key] = value
                # Remove 'metadata' key
                mp3s[item['audio_url']].pop('metadata')
        elif 'clips' in metadata[file]:
            for item in metadata[file]['clips']:
                mp3s[item['audio_url']] = item
        elif 'data' in metadata[file]:
            for item in metadata[file]['data']:
                mp3s[item['song_path']] = item
        elif 'songs' in metadata[file]:
            for item in metadata[file]['songs']:
                mp3s[item['song_path']] = item
        else:
            print(f"Unknown structure in file: {file}")
            continue

    # Create a DataFrame with metadata of unique songs
    df_unique_songs = pd.DataFrame(mp3s.values())
    if 'song_path' in df_unique_songs.columns:
        df_unique_songs.rename(columns={'song_path': 'audio_url'}, inplace=True)
    df_unique_songs = df_unique_songs.drop_duplicates(subset='audio_url')

    return df_unique_songs.drop_duplicates(subset='audio_url')

def download_audio_file(audio_url, audio_file, metadata_file, metadata):
    if not os.path.exists(audio_file):
        os.system(f'wget {audio_url} -O {audio_file}')
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=4)
        # check if the file was downloaded correctly
        if os.path.getsize(audio_file) < 1000:
            print(f"Error downloading {audio_url}. Removing mp3 file and metadata file.")
            os.remove(audio_file)
            os.remove(metadata_file)

def download_audio_files(df, directory):
    audio_dir = os.path.join(directory, 'audio')
    metadata_dir = os.path.join(directory, 'metadata')

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for index, row in df.iterrows():
            # if the id is nan, print the row
            if pd.isna(row['id']):
                print('keys:', row.keys())
                if 'clip' in row.keys():
                    row = row['clip']

            audio_url = row['audio_url']
            audio_file = os.path.join(audio_dir, f"{row['id']}.mp3")
            metadata_file = os.path.join(metadata_dir, f"{row['id']}.json")

            futures.append(executor.submit(download_audio_file, audio_url, audio_file, metadata_file, row))

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

def main():
    parser = argparse.ArgumentParser(description='Process metadata and download audio files.')
    parser.add_argument('directory', metavar='DIR', type=str, help='directory containing metadata files')
    args = parser.parse_args()

    df = get_unique_songs(args.directory)
    print(len(df))
    download_audio_files(df, args.directory)

if __name__ == "__main__":
    main()
