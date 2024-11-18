# %%
# file gotten from: https://github.com/renesemela/lastfm-dataset-2020/blob/master/datasets/lastfm_dataset_2020/lastfm_dataset_2020.db
import sys
import os
import sqlite3
import json
import glob

# if db file does not exist, download it
if not os.path.exists('lastfm_dataset_2020.db'):
    os.system('wget https://github.com/renesemela/lastfm-dataset-2020/raw/master/datasets/lastfm_dataset_2020/lastfm_dataset_2020.db')
          
# %%
db_file_path = 'lastfm_dataset_2020.db'

if not os.path.exists('/data/lastfm/audio'):
    os.makedirs('/data/lastfm/audio')
if not os.path.exists('/data/lastfm/metadata'):
    os.makedirs('/data/lastfm/metadata')

# Connect to the database
conn = sqlite3.connect(db_file_path)

sql_query = f"""
        SELECT * FROM metadata
    """

# Execute the query
rows = conn.execute(sql_query).fetchall()

# get the names of the columns
columns = [description[0] for description in conn.execute(sql_query).description]

# %%
import requests
import re
import subprocess

def download_full_song(row, out_dir='/data/lastfm/audio'):
    # Check if the mp3 file already exists
    if os.path.join(out_dir, f'{row[0]}.mp3') in glob.glob(os.path.join(out_dir, '*.mp3')):
        return None
    
    lastfm_url = row[3]
    response = requests.get(lastfm_url)

    # Extract the YouTube URL from the Last.fm page
    if 'https://www.youtube.com/watch?v=' not in response.text:
        return None
    
    youtube_id = re.search(r'href="https://www.youtube.com/watch\?v=(.*?)"', response.text).group(1)
    youtube_url = f'https://www.youtube.com/watch?v={youtube_id}'
    print(youtube_url)

    # Use yt-dlp to download the audio
    try:
        # Download the audio file in mp3 format
        command = [
            'yt-dlp',
            '--extract-audio',         # Download audio only
            '--audio-format', 'mp3',   # Convert to mp3
            '--output', f'{out_dir}/{row[0]}.%(ext)s',  # Output filename
            youtube_url
        ]

        # Run the command
        subprocess.run(command, check=True)

        return youtube_url
     
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {youtube_url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# %%
# # shuffle the rows and select the first 100000
import random
random.seed(0)
random.shuffle(rows)
rows = rows[:100000]

for row in rows:
    # download full song
    youtube_url = download_full_song(row)
    if youtube_url is None:
        continue

    # add youtube url to metadata
    row = list(row)
    row.append(youtube_url)
    columns.append('youtube_url')
    # save the row as json file in 'metadata'
    with open(f'/data/lastfm/metadata/{row[0]}.json', 'w') as f:
        json.dump(dict(zip(columns, row)), f)
