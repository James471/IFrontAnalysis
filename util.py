import os
import subprocess
from PIL import Image

def create_movie(files, outdir, outname, framerate=5):
    outpath = os.path.join(outdir, f'{outname}.mp4')
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # overwrite
        '-f', 'image2pipe',
        '-vcodec', 'png',
        '-framerate', str(framerate),
        '-i', '-',  # input from stdin
        outpath
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    # Feed images to ffmpeg
    for file in files:
        img = Image.open(file).convert("RGB")  # convert to RGB just in case
        img.save(process.stdin, format='PNG')
    process.stdin.close()
    process.wait()
    print(f"Movie saved to {outpath}")