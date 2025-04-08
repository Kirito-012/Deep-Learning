import yt_dlp

ydl = yt_dlp.YoutubeDL({'format': 'bestaudio[ext=m4a]'})

def get_video_url(url):
      with ydl:
            result = ydl.extract_info(url, download = False)
      return result

def get_audio_url(video_info):
      for f in video_info['formats']:
            if f['ext'] == 'm4a' and 'url' in f:
                  return f['url']
      raise ValueError("No suitable m4a audio URL found")