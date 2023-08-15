from google.cloud import speech

def transcribe_audio(audio_file_path):
    # Instantiates a client
    client = speech.SpeechClient()

    # Reads the audio file
    with open(audio_file_path, "rb") as audio_file:
        audio_data = audio_file.read()

    audio = speech.RecognitionAudio(content=audio_data)

    # Specifies the language of the audio
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    # Detects speech in the audio file
    response = client.recognize(config=config, audio=audio)

    # Retrieves the recognized text
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    return transcript

# Provide the path to your audio file
audio_file_path = "output.wav"

# Convert audio to text
text = transcribe_audio(audio_file_path)

# Print the converted text
print("Converted Text:")
print(text)
