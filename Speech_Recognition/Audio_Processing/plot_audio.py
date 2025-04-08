import matplotlib.pyplot as plt
import numpy as np
import wave

obj = wave.open('.\Speech_Recognition\Audio_Processing\ocean_waves.wav', 'rb')
n_channel = obj.getnchannels()
sample_freq = obj.getframerate()
n_samples = obj.getnframes()
signal_wave = obj.readframes(-1)
obj.close()

t_audio = n_samples/sample_freq
print(t_audio)

signal_array = np.frombuffer(signal_wave, dtype=np.int16)
times = np.linspace(0, t_audio, n_samples)
if n_channel == 2:
    signal_array = signal_array[0::2]  # Take left channel (every 2nd sample starting at 0)
elif n_channel > 2:
    raise ValueError("This code handles mono or stereo only.")
plt.figure(figsize=(20,10))
plt.plot(times, signal_array)
plt.ylabel('Signal Wave')
plt.xlabel('Time in sec')
plt.xlim(0, t_audio)
plt.show()