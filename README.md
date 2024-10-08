# Active-Speaker-Detection

Side project that I did during university.

Dataset used: https://github.com/clovaai/lookwhostalking

Face detection net used : S3FD https://github.com/cs-giung/face-detection-pytorch

ASD Model utilizes a pretrained S3D net:  https://github.com/kylemin/S3D


Run: " runThis.py --videoPath  " to process a video.

Requires python_speech_features , scenedetect[opencv] (I will add requirements.txt later).

Future changes include:

1) training with model2
2) using more data
3) try to utilize the temporal context of a face track (provide a sequence of frames -> extract audio-visual features from each frame -> utilize a self-attention or RNN to predict label for each frame).
