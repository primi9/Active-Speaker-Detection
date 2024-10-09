#!/usr/bin/env python3

import subprocess, os, json, shutil, gdown, argparse, warnings, torch
from model import ASD_model

#import torchvision
#import torchaudio
#from torch import nn
#import numpy as np

def fill_json_file(json_file_path, pred_list):
  
  with open(json_file_path) as json_file:
    data = json.load(json_file)
  
  dict_keys = [key for key in data.keys() if key != 'mode']
  
  index = 0

  for key in sorted(dict_keys):
    track_list = data[key]
    
    for track_list_item in track_list:
      
      track_list_item["label"] = int(pred_list[index][0])
      index += 1

  with open(json_file_path, 'w') as json_file:
    json.dump(data, json_file, indent=2)

def get_model_predictions(model,pyaviFolder,batch_size,fd_infer):
  
  from torch.nn import functional as F
  
  loader = facetracksDataset(pyaviFolder, batch_size = batch_size,augmentations = False, load_state = False, eval = True, verbose = False, infer_all = fd_infer)

  pred_list = []

  with torch.no_grad():
    for i, (audios, visuals, _) in enumerate(loader,1):
      audios = audios.to(device)
      visuals = visuals.to(device)

      outputs = model(audios, visuals)
      pred_list.extend((F.sigmoid(outputs).cpu() >= 0.5).int().tolist())
  
  return pred_list

parser = argparse.ArgumentParser(description = "runASD")
parser.add_argument('video', type = str,  help = "Location of the video")
parser.add_argument('--install', dest = 'to_install', action = 'store_true', help = 'Set this if you want the script to install the necessary libraries')
args = parser.parse_args()

if not os.path.exists(args.video):
  import sys
  print("Could not find video from the specified name")
  sys.exit(1)

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dummy = torch.randn(2,2).to(device)

if args.to_install:
  print("Installing libraries")
  install_cmds = ['pip install python_speech_features', 'pip install --upgrade scenedetect[opencv]']
  for cmd in install_cmds:
    subprocess.run(cmd,shell = True,stdout=subprocess.DEVNULL)

import python_speech_features
from facetracksDataset import facetracksDataset

if not os.path.isabs(args.video):
  video_path = os.path.abspath(args.video)
else:
  video_path = args.video

video_name = os.path.basename(args.video)
video_ref_folder = os.path.splitext(video_name)[0]

current_directory = os.getcwd()

outputVideoDir = os.path.join(current_directory, "vis_asd")
os.makedirs(outputVideoDir, exist_ok = True)

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

if not os.path.exists(video_name):
  shutil.copy2(video_path,video_name)
  copied_video = True
else:
  copied_video = False

if not os.path.exists('model_weights.pt'):
  weights_url = "https://drive.google.com/file/d/1ycW9QUVdWrjxTC-McN83F3yoPN3xSAUm/view?usp=drive_link"#download the weights
  gdown.download(weights_url, 'model_weights.pt', quiet=True, fuzzy = True)

s3fd_weights_path = os.path.join("s3fd", "sfd_face.pth")
if not os.path.exists(s3fd_weights_path):
  s3fd_url = "https://drive.google.com/file/d/1-3n83r_6PZBXMeI1s7RuSAXiUHNHkFGe/view?usp=sharing"
  gdown.download(s3fd_url, s3fd_weights_path, quiet=True, fuzzy = True)

model = ASD_model(load_state = True)
model.to(device)

if device == torch.device('cpu'):
  print("Using cpu...")
  checkpoint = torch.load('model_weights.pt', map_location = torch.device('cpu'))
  batch_size = 32
  fd_infer = False
  mode = 0
else:
  print("Using gpu...")
  checkpoint = torch.load('model_weights.pt')
  batch_size = 120
  fd_infer = True
  mode = 1

model.load_state_dict(checkpoint['model'])
model.eval()


#-------DEFINE ALL THE COMMANDS-------------



#convert the video to 25fps and extract audio
convert_cmd = "python run_convert.py --data_dir ."
videoPyaviPath = os.path.join('pyavi', video_ref_folder) # this folder contains video.avi (25fps) and audio.wav
aviVideoPath = os.path.join(videoPyaviPath, 'video.avi')

#extract all frames from the video using ffmpeg
os.makedirs("videoFramesFolder", exist_ok = True)
extract_cmd = f"ffmpeg -loglevel error -y -i {aviVideoPath} -qscale:v 2 -threads 1 -f image2 -start_number 0 {os.path.join('videoFramesFolder','%d.jpg')}"

#create face tracks 
os.makedirs("jsons", exist_ok = True)
jsonPath = os.path.join('jsons',video_ref_folder + '.json')
cft_cmd = f"python CreateFaceTracks.py --videoPath {aviVideoPath} --storePath {jsonPath} --videoFramesFolder videoFramesFolder --mode {mode} --verbose"

#extract track data for each track
etd_cmd = f"python extract_frames.py --videoFile {aviVideoPath} --jsonFile {jsonPath} --destinationDir {videoPyaviPath} --videoFramesFolder videoFramesFolder"

#visualize the predictions
vis_cmd = f"python run_visualize.py --avi_dir {videoPyaviPath} --jsonPath {jsonPath} --framesFolder videoFramesFolder --out_dir {outputVideoDir}"



#----------NOW RUN THE COMMANDS


subprocess.run(convert_cmd , shell=True, stdout=None)
subprocess.run(extract_cmd , shell=True, stdout=None)
subprocess.run(cft_cmd , shell=True)
print("Finished creating json file")
subprocess.run(etd_cmd , shell=True, stdout=None)
print("Finished extracting track data")
pred_list = get_model_predictions(model,pyaviFolder = "pyavi", batch_size = batch_size, fd_infer = fd_infer)
print("Finished predictions")
fill_json_file(jsonPath, pred_list)
subprocess.run(vis_cmd, shell = True, stdout = None)

#--------CLEAN UP--------------


shutil.rmtree("pyavi")
shutil.rmtree("videoFramesFolder")
os.remove(os.path.join("jsons", video_ref_folder + ".json"))

if copied_video:
  os.remove(video_name)


#-----FINISH---------

print("Video can be found in: ", outputVideoDir)
os.chdir(current_directory)