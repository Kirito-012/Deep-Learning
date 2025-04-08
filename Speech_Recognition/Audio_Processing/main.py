import wave

obj = wave.open('.\Speech_Recognition\Audio_Processing\ocean_waves.wav', 'rb')

# framerate --> is the number of sample per second
# if we want to get the time, we do --> total number of frames/framerate

print("Number of Channels --> ", obj.getnchannels()) # this means how many channels each samples have
print("Sample Width --> ", obj.getsampwidth()) #this means 2 bytes per sample
print("Frame Rate --> ", obj.getframerate())
print("Number of frames --> ", obj.getnframes())
# print("All Parameters --> ", obj.getparams())

print("Length of Audio--> ", obj.getnframes()/obj.getframerate())

frames = obj.readframes(-1)
print(type(frames), type(frames[0]))
print(frames)

# if we divide the number of frames by number_of_channels * sample_width
print(len(frames)/4) 
obj.close()

obj_new = wave.open('.\Speech_Recognition\Audio_Processing\ocean_waves_new.wav', 'wb')
obj_new.setnchannels(2)
obj_new.setsampwidth(2)
obj_new.setframerate(44100)
obj_new.writeframes(frames)
obj_new.close()

