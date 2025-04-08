import wave
from pyaudio import PyAudio
import pyaudio

FRAMES_PER_BUFFER = 3200
FRAME_RATE = 16000

p = PyAudio()
stream = p.open(
        rate = FRAME_RATE,
        channels = 1,
        format = pyaudio.paInt16,
        input = True,
        frames_per_buffer = FRAMES_PER_BUFFER
) 

print("Start recording...")
seconds = 5
frames = []
for i in range(0, int(FRAME_RATE/FRAMES_PER_BUFFER*seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

aud = wave.open('./Speech_Recognition/Audio_Processing/audio.wav', 'wb')
aud.setnchannels(1)
aud.setsampwidth(p.get_sample_size(pyaudio.paInt16))
aud.setframerate(FRAME_RATE)
aud.writeframes(b''.join(frames))
aud.close()