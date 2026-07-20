import subprocess
try:
    result = subprocess.run(["poetry", "add", "spacy"], capture_output=True, text=True, cwd=r"c:\Users\Rhia\Documents\AntiGravity\Rizal-Thematic-Exploration\backend")
    print("STDOUT START")
    print(result.stdout)
    print("STDOUT END")
    print("STDERR START")
    print(result.stderr)
    print("STDERR END")
except Exception as e:
    print("Error:", e)
