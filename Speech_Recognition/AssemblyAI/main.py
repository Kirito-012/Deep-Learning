import sys
from api_communication import *

# Upload
# Transcribe
# Poll
# Save Transcript

filename = sys.argv[1]


audio_url = upload(filename)
save(audio_url)