import pprint
import time
import requests
import json
      
API_KEY = "f52a5273a3fb421fa1a316aefaed1b4c"
API_KEY_LISTENNOTES = "71c22fa9bf7f4481a92b0d594dcc048b"

transcription_endpoint = 'https://api.assemblyai.com/v2/transcript'
assemblyai_headers = {'authorization': API_KEY}

listennotes_ep_endpoint = "https://listen-api.listennotes.com/api/v2/episodes"
listennotes_headers = {'X-ListenAPI-Key': API_KEY_LISTENNOTES}

def get_episode_url(episode_id):
      url = listennotes_ep_endpoint + '/' + episode_id
      response = requests.request('GET', url, headers=listennotes_headers)
      data = response.json()
      # pprint.pprint(data)
      
      audio_url = data['audio']
      eps_thumbnail = data['thumbnail']
      podcast_title = data['podcast']['title']
      eps_title = data['title']
      return audio_url, eps_thumbnail, eps_title, podcast_title

def transcribe(audio_url, auto_chapters):
      transcription_request = {
            'audio_url': audio_url,
            'auto_chapters': auto_chapters
                               }
      transcription_response = requests.post(transcription_endpoint, json = transcription_request, headers = assemblyai_headers)
      job_id = transcription_response.json()['id']
      return job_id

def poll(transcript_id):
      polling_endpoint = transcription_endpoint + '/' + transcript_id
      polling_response = requests.get(polling_endpoint, headers = assemblyai_headers)
      return polling_response.json()

def duration(tick):
      tick = tick + 1
      print("Waiting... ", tick, "sec")
      time.sleep(1)
      return tick

def get_transcription_result_url(audio_url, auto_chapters):
      transcript_id = transcribe(audio_url, auto_chapters)
      tick = 0
      while True:
            data = poll(transcript_id)
            if data['status'] == 'completed':
                  return data, None
            elif data['status'] == 'error':
                  print("Transcription failed")
                  return data, 'error'
            tick = duration(tick)            

def save_transcript(episode_id):
      audio_url, eps_thumbnail, eps_title, podcast_title =get_episode_url(episode_id)
      
      data, error = get_transcription_result_url(audio_url, auto_chapters=True)
      pprint.pprint(data)

      if error:
            print("Error:", error)
            return False

      if data:
            text_filename = 'Podcast.txt'
            with open(text_filename, 'w',  encoding='utf-8') as f:
                  f.write(data['text'])

            chapters_filename = "Chapters.json"
            with open(chapters_filename, 'w',  encoding='utf-8') as f:
                  chapters = data['chapters']
                  epsiode_data = {'chapters': chapters}
                  epsiode_data['eps_thumbnail'] = eps_thumbnail
                  epsiode_data['eps_title'] = eps_title
                  epsiode_data['podcast_title'] = podcast_title

                  json.dump(epsiode_data, f)
            print("Transcript Saved!!")
            return True


      # #             f.write(data['text'])
      # #             f.flush()
      # #       print("Transcription Saved!")
      # #       print("Checking for Sentiment Analysis...")
      # # else: 
      # #       print("No text available to save")

      #       if auto_chapters:
      #             print("Sentiment_Analysis")
      #             filename = "_sentiments.json"
      #             with open(filename, "w", encoding='utf-8') as f:
      #                   json.dump(data['sentiment_analysis_results'], f, indent=4)
      #       else:
      #             print("No sentiment analysis results to save.") 