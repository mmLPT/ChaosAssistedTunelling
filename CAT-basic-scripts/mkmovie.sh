rm *.mp4
ffmpeg -framerate 5 -i hu-%05d.png -c:v libx264 out.mp4
