import whisper
import speech_recognition as sr
import tempfile
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv
from playsound import playsound

class VoiceAssistant:
    def __init__(self, model_size: str="turbo"):
        load_dotenv()
        self.model = whisper.load_model(model_size)
        self.recognizer = sr.Recognizer()
        self.client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
        self.context = [{"role": "system", "content": "You are an AI Assistant that helps people find information. Keep your answers really concise"}]

    def listen(self) -> str:
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio.get_wav_data())
            temp_audio_path = temp_audio.name
        
        return temp_audio_path

    def audio_to_text(self, audio_file_path: str) -> str | None:
        result = self.model.transcribe(audio_file_path)

        if result is None or result.get("text") is None:
            print("There was an error transcribing")
            return None

        return result["text"]
    
    def sent_gpt_request(self, prompt: str) -> str:
        self.context.append({"role": "user", "content": prompt})
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.context
        )
        response = completion.choices[0].message.content
        self.context.append({"role": "assistant", "content": response})
        return response

    def text_to_speech(self, text):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
        )
        response.stream_to_file("output.mp3")

    def run_assistant(self): 
        audio_file = self.listen()
        text = self.audio_to_text(audio_file)
        os.remove(audio_file)
        
        print(f"You said: {text}")
        response = self.sent_gpt_request(text)
        print(f"ChatGPT: {response}")
        self.text_to_speech(response) # audio file
        playsound("output.mp3")
        sys.exit(0)

if __name__ == "__main__":
    va = VoiceAssistant()
    va.run_assistant()
