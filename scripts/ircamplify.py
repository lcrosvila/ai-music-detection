import argparse
import requests
import time
import json
from pathlib import Path
import os

# ---------------------------------------------
# Add your API credentials HERE
# ---------------------------------------------
#load credentials
with open('credentials/credentials.json') as f:
    credentials = json.load(f)

client_id = credentials["client_id"]
client_secret = credentials["client_secret"]

def get_auth_token(client_id, client_secret):
    auth_url = "https://api.ircamamplify.io/oauth/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    response = requests.post(auth_url, json=payload)
    return response.json()["id_token"]

def create_ias_object(headers):
    manager_url = "https://storage.ircamamplify.io/manager/"
    response = requests.post(manager_url, headers=headers)
    return response.json().get('id')

def upload_file(headers, ias_id, local_file):
    storage_url = "https://storage.ircamamplify.io"
    filename = Path(local_file).name
    put_url = f"{storage_url}/{ias_id}/{filename}"
    response = requests.put(put_url, data=open(local_file, 'rb'), headers=headers)
    return response

def get_ias_url(headers, ias_id):
    manager_url = "https://storage.ircamamplify.io/manager/"
    response = requests.get(manager_url + ias_id, headers=headers)
    return response.json().get('ias')

def process_audio(headers, ias_url):
    module_url = "https://api.ircamamplify.io/aidetector/"
    payload = {'audioUrlList': [ias_url]}
    response = requests.post(module_url, headers=headers, json=payload)
    return response.json()["id"]

def wait_for_processing(headers, job_id):
    module_url = "https://api.ircamamplify.io/aidetector/"
    process_status = None
    while process_status not in ["success", "error"]:
        print("Processing...")
        time.sleep(5)
        response = requests.get(module_url + "/" + job_id, headers=headers)
        job_infos = response.json().get('job_infos')
        process_status = job_infos["job_status"]
    return process_status

def get_results(headers, job_id):
    module_url = "https://api.ircamamplify.io/aidetector/"
    response = requests.get(module_url + "/" + job_id, headers=headers)
    job_infos = response.json().get('job_infos')
    return job_infos['report_info']['report']

def upload_all_files(headers, audio_files):
    ias_url_list = []
    file_path_map = {}
    for local_file in audio_files:
        print(f"Processing file: {local_file}")
        
        # Create IAS object
        ias_id = create_ias_object(headers)
        print(f"Created IAS object with ID: {ias_id}")

        # Upload file
        response = upload_file(headers, ias_id, local_file)
        if response.status_code != 200:
            print(f"Failed to upload: {Path(local_file).name}. Status code: {response.status_code}")
            continue

        print(f"Successfully uploaded: {Path(local_file).name}")

        # Get IAS URL
        ias_url = get_ias_url(headers, ias_id)
        print(f"IAS URL: {ias_url}")
        ias_url_list.append(ias_url)
        file_path_map[ias_url] = local_file

    return ias_url_list, file_path_map

def process_audio_batch(headers, ias_url_list):
    module_url = "https://api.ircamamplify.io/aidetector/"
    payload = {'audioUrlList': ias_url_list}
    response = requests.post(module_url, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"Error: API returned status code {response.status_code}")
        print("Response content:")
        print(response.text)
        return None
    
    try:
        job_id = response.json()["id"]


        process_status = None
        while process_status not in ["success", "error"]:
            print("Processing...")
            time.sleep(5)
            response = requests.get(module_url + "/" + job_id, headers=headers)
            job_infos = response.json().get('job_infos')
            if job_infos is None:
                print("Error: Unable to get job infos")
                print("Response content:")
                print(response.text)
            else:
                process_status = job_infos["job_status"]

        response = requests.get(module_url + "/" + job_id, headers=headers)
        
        return response.json()
    except json.JSONDecodeError:
        print("Error: Unable to parse API response as JSON")
        print("Response content:")
        print(response.text)
        return None

def main():
    for dataset in ['udio', 'lastfm', 'suno']:
    # for dataset in ['boomy']:
    # for dataset in ['lastfm']:
        # Get auth token
        id_token = get_auth_token(client_id, client_secret)
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {id_token}'
        }

        # Read audio files
        with open(f'/data/{dataset}/to_do_files.txt', 'r') as f:
            audio_files = f.read().splitlines()
            audio_files = [f'/data/{dataset}/audio/{audio_file}.mp3' for audio_file in audio_files]        
        
        # batch it to do it every 20 files
        audio_file_batches = [audio_files[i:i + 20] for i in range(0, len(audio_files), 20)]

        for batch, audio_files in enumerate(audio_file_batches):
            if os.path.exists(f'/data/ircamplify_results/{dataset}/ircamplify_{batch}.json'):
                print(f"Batch {batch} already processed. Skipping.")
                continue

            # Upload all files and get IAS URLs
            ias_url_list, file_path_map = upload_all_files(headers, audio_files)
            
            if not ias_url_list:
                print("No files were successfully uploaded. Exiting.")
                return

            # Process all audio files in a single batch
            results = process_audio_batch(headers, ias_url_list)
            if results is None or 'job_infos' not in results:
                print("Unable to process audio files. Exiting.")
                return
        
            if not os.path.exists(f'/data/ircamplify_results/{dataset}'):
                os.makedirs(f'/data/ircamplify_results/{dataset}')
            
            # Save results to file
            with open(f'/data/ircamplify_results/{dataset}/ircamplify_{batch}.json', 'w') as f:
                json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
