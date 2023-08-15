from pydub import AudioSegment

def convert_wav_to_mp3(wav_file_path, mp3_file_path, bitrate="192k"):
    try:
        # Load the .wav file using pydub
        audio = AudioSegment.from_wav(wav_file_path)

        # Export the audio to .mp3 with the specified bitrate
        audio.export(mp3_file_path, format="mp3", bitrate=bitrate)

        print(f"Conversion successful: {mp3_file_path}")
    except Exception as e:
        print("An error occurred:", str(e))

# Set the file paths
wav_file_path = r'C:\Users\Admin\Downloads\Conference.wav'
mp3_file_path = "try.mp3"

# Call the function to perform the conversion
convert_wav_to_mp3(wav_file_path, mp3_file_path)
