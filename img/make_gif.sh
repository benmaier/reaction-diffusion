#!/bin/bash
# https://superuser.com/questions/556029/how-do-i-convert-a-video-to-gif-using-ffmpeg-with-reasonable-quality

ffmpeg -y -t 15 -i gray_scott.mp4 -vf fps=10,scale=320:-1:flags=lanczos,palettegen palette.png
ffmpeg -t 15 -i gray_scott.mp4 -i palette.png -filter_complex "fps=10,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse" gray_scott.gif
