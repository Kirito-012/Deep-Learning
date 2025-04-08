import requests
import time
API_KEY = "f52a5273a3fb421fa1a316aefaed1b4c"

upload_endpoint = 'https://api.assemblyai.com/v2/upload'
trans_endpoint = 'https://api.assemblyai.com/v2/transcript'
headers = {'authorization': API_KEY}

# Uploading the file
def upload(filename):
        def read_file(filename, chunk_size = 5242800):
                with open(filename, 'rb') as _file:
                        while True:
                                data = _file.read(chunk_size)
                                if not data:
                                        break
                                yield data

        upload_response = requests.post(upload_endpoint, 
                                 headers = headers,
                                 data = read_file(filename))
        
        audio_url = upload_response.json()['upload_url']
        return audio_url

# Transcribing
def transcribe(audio_url):
        trans_request = {'audio_url': audio_url}
        trans_response = requests.post(trans_endpoint, json=trans_request, headers=headers)
        job_id = trans_response.json()['id']
        return job_id

# Polling
def poll(transcript_id):
        polling_endpoint = trans_endpoint + '/' + transcript_id
        polling_response = requests.get(polling_endpoint, headers=headers)
        return polling_response.json()

def get_transcription_result_url(audio_url):
        transcript_id = transcribe(audio_url)
        while True: 
                data = poll(transcript_id)
                if data['status'] == 'completed':
                        return data, None
                elif data['status'] == 'error':
                        return data, data['error']
                print(" Waiting 10 seconds...")
                time.sleep(10)

def save(audio_url):
        data, error = get_transcription_result_url(audio_url)
        if data:
                text_filename = 'transcripted_audio.txt'  
                with open(text_filename, 'w') as f:
                        f.write(data['text'])
                print("Transcription Saved")
        elif error:
                print("Error", error)
                
