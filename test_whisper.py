import whisper

print("Loading Whisper model...")

model = whisper.load_model("tiny")

print("Whisper loaded successfully!")
print("Transcribing audio...")

result = model.transcribe(
    "test_audio/test.wav.mpeg",
    fp16=False
)

print("\n========== TRANSCRIPT ==========")
print(result["text"])
print("================================")