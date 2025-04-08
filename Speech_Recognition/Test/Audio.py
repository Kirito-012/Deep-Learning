import time
import requests
import json
      
API_KEY = "f52a5273a3fb421fa1a316aefaed1b4c"
upload_endpoint = 'https://api.assemblyai.com/v2/upload'
transcription_endpoint = 'https://api.assemblyai.com/v2/transcript'
headers = {'authorization': API_KEY}
filename = '2_Apr.m4a'


def upload(filename):
      def read_file(filename, chunk_size = 5242800):
            with open(filename, 'rb') as f:
                  while True:
                        data = f.read(chunk_size)
                        if not data:
                              break
                        yield data
      upload_response = requests.post(upload_endpoint, data = read_file(filename), headers = headers)
      audio_url = upload_response.json()['upload_url']
      return audio_url

def transcribe(audio_url, sentiment_analysis):
      transcription_request = {
            'audio_url': audio_url,
            'sentiment_analysis': sentiment_analysis
                               }
      transcription_response = requests.post(transcription_endpoint, json = transcription_request, headers = headers)
      job_id = transcription_response.json()['id']
      return job_id

def poll(transcript_id):
      polling_endpoint = transcription_endpoint + '/' + transcript_id
      polling_response = requests.get(polling_endpoint, headers = headers)
      return polling_response.json()

def get_transcription_result_url(audio_url, sentiment_analysis):
      transcript_id = transcribe(audio_url, sentiment_analysis)
      while True:
            data = poll(transcript_id)
            if data['status'] == 'completed':
                  return data, None
            elif data['status'] == 'error':
                  print("Transcription failed")
                  return data, 'error'
            print('Waiting...')
            time.sleep(10)

def save_transcript(audio_url, sentiment_analysis=False):
      data, error = get_transcription_result_url(audio_url, sentiment_analysis)
      if error:
            print("Error from API:", error)
            return
      if 'text' in data and data['text'] is not None:
            text_filename = 'audio.txt'
            print(f"Text to write: {data['text']}")
            with open(text_filename, 'w',  encoding='utf-8') as f:
                  f.write(data['text'])
                  f.flush()
            print("Transcription Saved!")
            print("Checking for Sentiment Analysis...")
      else: 
            print("No text available to save")

      if sentiment_analysis:
            print("Sentiment_Analysis")
            filename = "_sentiments.json"
            with open(filename, "w", encoding='utf-8') as f:
                  json.dump(data['sentiment_analysis_results'], f, indent=4)
      else:
            print("No sentiment analysis results to save.")

url = upload(filename)
save_transcript(url, sentiment_analysis=True)