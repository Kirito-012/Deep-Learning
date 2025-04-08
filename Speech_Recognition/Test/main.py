from api import *
import json
from yt_extractor import *

def save_video_sentiments(url):
      videos_info = get_video_url(url)
      audio_url = get_audio_url(videos_info)
      title = videos_info['title']
      title = title.strip().replace(' ', '_')

      response = requests.get(audio_url, stream = True)

      # doing the Poll thing here
      if response.status_code == 200:
            upload_response = requests.post(upload_endpoint, headers = headers, data=response.content)
            stable_audio_url = upload_response.json()['upload_url']
            print("Saved Audio to Assembly AI...")
            save_transcript(stable_audio_url, title, sentiment_analysis = True)
      else:
            print("Failed to fetch audio:", response.status_code)

# save_video_sentiments('https://www.youtube.com/watch?v=e-kSGNzu0hM')
with open('_sentiments.json', 'r') as f:
      data = json.load(f)

positives = []
negatives = []
neutrals = []

for result in data:
      text = result['text']
      if result['sentiment'] == 'POSITIVE':  
            positives.append(text)
      elif result['sentiment'] == 'NEGATIVE':
            negatives.append(text)
      else:
            neutrals.append(text)
n_pos = len(positives)
n_neg = len(negatives)
n_neu = len(neutrals)

print("Num Positives: ", n_pos)
print("Num Negatives: ", n_neg)
print("Num Neutrals: ", n_neu)

r = n_pos / (n_pos + n_neg)
print(f"Positive Ratio: {r:.2f}")