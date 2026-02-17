import whisper
import json
import os
model=whisper.load_model("large-v2")
audios=os.listdir('audios')
for audio in audios:
    if ("(" in audio):
        number=audio.split('(')[1].split(',')[0]
        title=audio.split(' - ')[0]
        print(number,title)
        result=model.transcribe(audio=f"audios/Python in 8 Minutes (in Hindi) 🐍 - CodeWithHarry (240p, h264)_240p.mp3", language="hindi", task="translate",word_timestamps=False)

        chunks= []
        for segment in result["segments"]:
            chunks.append({"number":number,"title":title,"start":segment["start"], "end":segment["end"], "text":segment["text"]})

        chunks_with_metadata = {"chunks":chunks,"text":result["text"]}

        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata, f,)