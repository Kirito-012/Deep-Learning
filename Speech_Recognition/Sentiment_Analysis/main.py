import os
import requests
from yt_extractor import *
from api import *

def save_video_sentiments(url):
    video_info = get_video_info(url)
    audio_url = get_audio_url(video_info)
    title = video_info['title']
    title = title.strip().replace(' ', '_')
    title = "data/" + title
    
    # Ensure the 'data' directory exists for output files
    os.makedirs("data", exist_ok=True)
    
    # Fetch the audio stream and upload to AssemblyAI
    response = requests.get(audio_url, stream=True)
    if response.status_code == 200:
        upload_response = requests.post(
            'https://api.assemblyai.com/v2/upload',
            headers={'authorization': API_KEY},
            data=response.content  # Send the raw bytes directly
        )
        stable_audio_url = upload_response.json()['upload_url']
        print("Uploaded to AssemblyAI:", stable_audio_url)
        save_transcript(stable_audio_url, title, sentiment_analysis=True)
    else:
        print("Failed to fetch audio:", response.status_code)

save_video_sentiments('https://www.youtube.com/watch?v=e-kSGNzu0hM')