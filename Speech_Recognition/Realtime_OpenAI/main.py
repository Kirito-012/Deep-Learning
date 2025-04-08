from multiprocessing.connection import answer_challenge
import pyaudio
import websockets
import asyncio
import base64
import json
from api import *

p = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAMES_PER_BUFFER = 3200

stream = p.open(
      format = FORMAT,
      channels = CHANNELS,
      rate = RATE,
      input = True,
      frames_per_buffer = FRAMES_PER_BUFFER
)

URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"

async def send_recieve():
      async with websockets.connect(
            URL, 
            ping_timeout = 20,
            ping_interval = 5,
            extra_headers = {"Authorization": API_KEY}
      ) as _ws:
            await asyncio.sleep(0.1)
            session_begins = await _ws.recv()
            print(session_begins)
            print("Session Begins!!")

            async def send():
                  while True:
                        try:
                              data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow = False)
                              data = base64.b64encode(data).decode('utf-8')
                              json_data = json.dumps({"audio_data" : data})
                              await _ws.send(json_data)
                        except Exception as e:
                              print(f"Error in send: {e}")
                        await asyncio.sleep(0.1)  

            async def recieve():
                  while True:
                        try:
                              result_str = await _ws.recv()
                              result = json.load(result_str)
                              prompt = result['text']
                              if prompt and result['message_type'] == 'FinalTranscript':
                                    print("Me: ", prompt)
                                    print("Bot: ", "This is my answer")

                        except Exception as e:
                              print(f"Error in send: {e}")
                        await asyncio.sleep(0.1)  

            
            send_result, recieve_result = await asyncio.gather(send(), recieve())

asyncio.run(send_recieve())