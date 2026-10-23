import os

folder = "audios"
print("Files in 'audios/':")
for f in os.listdir(folder):
    print("-", f)
