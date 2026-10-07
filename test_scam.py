from modules.scam_detection import detect_scam_indicators


text = """
Your bank account KVS is pending.
Please verify your account.
Tell me your OTP immediately.
Otherwise your account will be blocked.
"""


results = detect_scam_indicators(text)

print("\n==============================")
print("DEEPSHIELD SCAM DETECTION TEST")
print("==============================")

for item in results:

    print("\nCategory:", item["category"])

    print("Detected:", item["matches"])