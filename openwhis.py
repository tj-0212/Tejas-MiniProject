import whisper
import subprocess
model = whisper.load_model("base")
# result = model.transcribe(r'C:\Users\Admin\Downloads\sample-0.mp3')
result = model.transcribe(r'C:\Users\Admin\Desktop\MINI-Project\try.mp3')
print(result)
with open("trytrans.txt","w") as f:
    f.write(result["text"])
import whisper

# def transcribe_audio(audio_file_path, model_name):
#     try:
#         # Load the Whisper model with the specified name
#         model = whisper.load_model(model_name)

#         # Transcribe the audio
#         result = model.transcribe(audio_file_path)

#         # Return the transcription text
#         return result["text"]
#     except Exception as e:
#         print("An error occurred:", str(e))
#         return None

# def save_transcription_to_file(transcription_text, output_file_path):
#     try:
#         with open(output_file_path, "w") as f:
#             f.write(transcription_text)
#         print("Transcription saved successfully.")
#     except Exception as e:
#         print("An error occurred while saving transcription:", str(e))

# if __name__ == "__main__":
#     # Set the file paths
#     audio_file_path = r'C:\Users\Admin\Desktop\MINI-Project\try.mp3'
#     output_file_path = "trytrans.txt"

#     # List of ASR models you want to try
#     asr_models = ["base", "quartznet", "jasper"]

#     for model_name in asr_models:
#         # Transcribe the audio using the specified ASR model
#         transcription_text = transcribe_audio(audio_file_path, model_name)

#         if transcription_text:
#             # Save the transcription to a file
#             save_transcription_to_file(f"{model_name}:\n{transcription_text}\n\n", output_file_path)

    
