import pyaudio
import wave

# channel    --> mono audio or dual audio
# frame_rate --> total frames in one second
# frames_per_buffer --> number of frames read in each chunk

# in the example below -->
# 1 second will be divided into 16000/3200 = 5 chunks
# that is, 0.2, 0.4, 0.6, 0.8, 1
# and each chunks will have 3200 samples

FORMAT = pyaudio.paInt16
CHANNEL = 1
FRAME_RATE = 16000
FRAMES_PER_BUFFER = 3200

p = pyaudio.PyAudio()
stream = p.open(
        format = FORMAT, 
        channels = CHANNEL, 
        rate = FRAME_RATE,
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

obj = wave.open('.\Speech_Recognition\Audio_Processing\output_record.wav', 'wb')
obj.setnchannels(1)
obj.setsampwidth(p.get_sample_size(FORMAT))
obj.setframerate(FRAME_RATE)
obj.writeframes(b''.join(frames))
obj.close()

print(FRAME_RATE/FRAMES_PER_BUFFER*seconds)