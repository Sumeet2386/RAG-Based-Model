import os
import subprocess
files=os.listdir('video')
for file in files:
    tutorial_name=file.split('.')[0]
    subprocess.run(['ffmpeg', '-i', f'video/{file}', '-vf', 'scale=320:240', f'audio/{tutorial_name}_240p.mp3'])
