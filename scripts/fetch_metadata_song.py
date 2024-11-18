import requests
import json
import os
import argparse

def fetch_song_data(service, song_id, save=False, save_dir=None):
    data = None

    if service == 'suno':
        url = f'https://suno.com/song/{song_id}'
        headers = {
            'accept': '*/*',
            'accept-language': 'en,es-ES;q=0.9,es;q=0.8,ca;q=0.7,sv;q=0.6',
            'next-url': f'/song/{song_id}',
            'priority': 'u=1, i',
            'referer': 'https://suno.com/create',
            'rsc': '1',
        }

        response = requests.get(url, headers=headers)
        print(f'Status code: {response.status_code}')

        plain_text = response.text
        plain_text = plain_text[plain_text.find('"clip"'):]
        plain_text = plain_text[7:].split(',"coveredClip"')[0]
        data = json.loads(plain_text)

    elif service == 'udio':
        url = f'https://www.udio.com/songs/{song_id}'
        headers = {
            'accept': '*/*',
            'accept-language': 'en,es-ES;q=0.9,es;q=0.8,ca;q=0.7,sv;q=0.6',
            'priority': 'u=1, i',
            'rsc': '1',
        }

        response = requests.get(url, headers=headers)
        print(f'Status code: {response.status_code}')

        plain_text = response.text
        plain_text = plain_text[plain_text.find('"track"'):]
        plain_text = plain_text[8:].split('}]}]}]')[0]
        data = json.loads(plain_text)
    else:
        print(f"Service '{service}' not recognized.")
        return

    # Print the retrieved data
    print(data)

    # Save the data if the --save flag is used
    if save and data:
        save_data(service, data['id'], data, save_dir)

def save_data(service, song_id, data, save_dir=None):
    # Determine the base directory: provided save_dir or current working directory
    base_dir = save_dir if save_dir else ''
    
    # Create the directory structure {base_dir}/{service}/metadata/
    directory = os.path.join(base_dir, service, 'metadata')
    os.makedirs(directory, exist_ok=True)

    # Save the data to {base_dir}/{service}/metadata/{song_id}.json
    file_path = os.path.join(directory, f'{song_id}.json')
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f'Data saved to {file_path}')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fetch song data from a service. E.g. python fetch_metadata_song.py suno 18cc37ed-391d-4160-923d-a6e652533df5')
    parser.add_argument('service', type=str, help="Service to use ('suno' or 'udio')")
    parser.add_argument('song_id', type=str, help='ID of the song to fetch')
    parser.add_argument('--save', action='store_true', help='Save the fetched data to a file')
    parser.add_argument('--dir', type=str, default=None, help='Directory where data will be saved')

    # Parse the arguments
    args = parser.parse_args()

    # Fetch the song data
    fetch_song_data(args.service, args.song_id, args.save, args.dir)

if __name__ == "__main__":
    main()
