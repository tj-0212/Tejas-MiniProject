from googletrans import Translator
import re
from speak import Speak

from googletrans import Translator
import re

def get_language(text):
    # Using googletrans, detect the language of the text
    translator = Translator()
    result = translator.detect(text)
    return result.lang

def mixed_language_to_english(text):
    translator = Translator()
    segments = re.split(r'([,.\n!?])', text)  # Split text into segments based on punctuation
    translated_text = []

    for segment in segments:
        source_language = get_language(segment)
        if source_language in ['kn', 'hi']:  # If the segment contains Kannada or Hindi, translate to English
            translated_segment = translator.translate(segment, src=source_language, dest='en').text
            translated_text.append(translated_segment)
        else:  # If the segment contains English text or any other language, keep it as is
            translated_text.append(segment)

    return ''.join(translated_text)

def main():
   # Sample mixed-language input text (Kannada, Hindi, and English)
    input_file = "multiland.txt"
    with open(input_file, "r", encoding='utf-8') as file:
        input_text = file.read()
#     input_text="""As the sun enters Capricorn, let's rejoice in the spirit of harvest and togetherness. 
# ಈ ಮಂಗಳಕರ ದಿನವು ನಿಮಗೆ ಮತ್ತು ನಿಮ್ಮ ಪ್ರೀತಿಪಾತ್ರರಿಗೆ ಸಮೃದ್ಧ ಬೆಳೆಗಳನ್ನು ಮತ್ತು ಸಂತೋಷದ ಕ್ಷಣಗಳನ್ನು ತರಲಿ.Celebrate the spirit of  with diyas.
# In the land of Mahatma Gandhi and Swami Vivekananda, let's uphold the values of peace and harmony.
#  With the tricolor flying high, आइए एकजुट रहें और अपने महान राष्ट्र के विकास में योगदान दें"""

    # Convert mixed-language text to English-only
    translated_text = mixed_language_to_english(input_text)
    print("Input Text:", input_text)
    print("Translated Text:", translated_text)
    Speak(translated_text)

if __name__ == "__main__":
    main()

