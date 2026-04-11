import os, shutil

os.makedirs('backend/core', exist_ok=True)
os.makedirs('backend/api', exist_ok=True)
os.makedirs('backend/webrtc', exist_ok=True)

if os.path.exists('src'):
    for f in os.listdir('src'):
        shutil.move(os.path.join('src', f), os.path.join('backend/core', f))
    os.rmdir('src')

print("Success")
