import json
import requests
import time
import os  
API_KEY = "f52a5273a3fb421fa1a316aefaed1b4c"

upload_endpoint = 'https://api.assemblyai.com/v2/upload'
trans_endpoint = 'https://api.assemblyai.com/v2/transcript'
headers = {'authorization': API_KEY}

# Uploading the file (not used here, but kept for reference)
def upload(filename):
    def read_file(filename, chunk_size=5242800):
        with open(filename, 'rb') as _file:
            while True: 
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data
    upload_response = requests.post(upload_endpoint, headers=headers, data=read_file(filename))
    audio_url = upload_response.json()['upload_url']
    return audio_url

# Transcribing
def transcribe(audio_url, sentiment_analysis):
    trans_request = {
        'audio_url': audio_url,
        'sentiment_analysis': sentiment_analysis
    }
    trans_response = requests.post(trans_endpoint, json=trans_request, headers=headers)
    job_id = trans_response.json()['id']
    return job_id

# Polling
def poll(transcript_id):
    polling_endpoint = trans_endpoint + '/' + transcript_id
    polling_response = requests.get(polling_endpoint, headers=headers)
    return polling_response.json()

def get_transcription_result_url(audio_url, sentiment_analysis):
    transcript_id = transcribe(audio_url, sentiment_analysis)
    while True: 
        data = poll(transcript_id)
        if data['status'] == 'completed':
            print("Transcription completed. Response:", data)
            return data, None
        elif data['status'] == 'error':
            print("Transcription failed. Response:", data)
            return data, data['error']
        print("Waiting 10 seconds...")
        time.sleep(10)

def save_transcript(audio_url, title, sentiment_analysis):
    data, error = get_transcription_result_url(audio_url, sentiment_analysis)
    if error:
        print("Error from API:", error)
        return
    # Save transcription text if available
    if 'text' in data and data['text'] is not None:
        text_filename = 'transcripted.txt'
        text_abs_path = os.path.abspath(text_filename)  # Get absolute path
        print(f"Text to write: {data['text']}")  # Verify the text before writing
        try:
            with open(text_filename, 'w', encoding='utf-8') as f:  # Specify UTF-8 encoding
                f.write(data['text'])
                f.flush()  # Ensure the write is committed
            print(f"\n\n\\\nText saved to {text_abs_path}")
            # with open(text_filename, 'r', encoding='utf-8') as f:
            #     written_content = f.read()
            # print(f"Debug - Content read back from file: {written_content}")
        except Exception as e:
            print(f"Failed to save text file: {e}")
    else:
        print("No text available to save. 'text' field:", data.get('text'))

    # Save sentiment analysis if available
    if sentiment_analysis and 'sentiment_analysis_results' in data and data['sentiment_analysis_results'] is not None:
        json_filename = "sentiments.json"
        json_abs_path = os.path.abspath(json_filename)  # Get absolute path
        try:
            with open(json_filename, 'w', encoding='utf-8') as f:  # Specify UTF-8 encoding
                json.dump(data['sentiment_analysis_results'], f, indent=4)
            print(f"Sentiment analysis saved to {json_abs_path}")
        except Exception as e:
            print(f"Failed to save JSON file: {e}")
    else:
        print("No sentiment analysis results to save. 'sentiment_analysis_results' field:", data.get('sentiment_analysis_results'))
    
    if 'text' in data and data['text'] is not None:
        print("Transcription process completed successfully")
    else:
        print("Transcription process completed, but no files saved due to missing data")