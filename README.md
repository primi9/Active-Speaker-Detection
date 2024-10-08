# Active-Speaker-Detection

Side project that I did during university. All code was developed on google colab. 

Dataset used: https://github.com/clovaai/lookwhostalking  
Face detection net used : S3FD https://github.com/cs-giung/face-detection-pytorch  
ASD Model utilizes pretrained S3D net:  https://github.com/kylemin/S3D  
Data augmentations used from: https://github.com/TaoRuijie/TalkNet-ASD  

Run: " python runThis.py {videoPath}  " to process a video.  

Requires python_speech_features , scenedetect[opencv] (I will add requirements.txt later).

Copy ColabNotebook.ipynb to a runtime with T4 in google colab and run the cells.  

Future changes include:

1) training with model2
2) using more data
3) try to utilize the temporal context of a face track (provide a sequence of frames -> extract audio-visual features from each frame -> use a self-attention or RNN to predict label for each frame).

Issue that I want to fix:
Cases of " speaking but not audible " are misclassified as speaking. To solve that issue, more cases like these should be included in the dataset, so that we can better model the audio-visual interaction of a face track. 
