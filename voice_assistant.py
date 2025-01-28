import whisper
import speech_recognition as sr
import tempfile

class VoiceAssistant:
    def __init__(self, model_size: str="turbo"):
        self.model = whisper.load_model(model_size)
        self.recognizer = sr.Recognizer()

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

    def run_assistant(self):
        
        audio_file = self.listen()
        text = self.audio_to_text(audio_file)
        print(text)

if __name__ == "__main__":
    jarvis = VoiceAssistant()
    jarvis.run_assistant()
