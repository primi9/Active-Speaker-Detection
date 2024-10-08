#!/usr/bin/env python3

import os, json ,cv2, argparse, sys

def extract_video_frames(args):
    
  import subprocess

  os.makedirs(args.videoFramesFolder,exist_ok=True)

  command = ("ffmpeg -loglevel error -y -i %s -qscale:v 2 -threads 1 -f image2 -start_number 0 %s" % (args.videoFile,os.path.join(args.videoFramesFolder,'%d.jpg'))) 
  output = subprocess.call(command, shell=True, stdout=None)

  if output != 0:
    print("Assertion fails so fuck it")
    
  return

def get_video_details(args):
  
  video = cv2.VideoCapture(args.videoFile)#open video with opencv
  
  width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))#get width of frame
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))#get height of frame
  fps = int(video.get(cv2.CAP_PROP_FPS))#get fps of current video

  video_details = open(os.path.join(args.destinationDir, 'details.txt'), 'w')#file that contains details of current video

  video_details.write("Width: %lf\n" %width)
  video_details.write("Height: %lf\n" %height)
  video_details.write("FPS: %lf\n" %fps)

  video_details.close()
  video.release()

  return [width,height,fps]

def extract_track_data_25fps(args,track_item,video_details):

  track_name = track_item[0]
  track_vals = track_item[1]

  width,height,fps = video_details

  track_folder_path = os.path.join(args.destinationDir, track_name)

  if not os.path.exists(track_folder_path):
    os.mkdir(track_folder_path)

  first_point = track_vals[0]
  last_point = track_vals[-1]

  with open(os.path.join(track_folder_path, 'audio_tms.txt'), 'w') as f:
    f.write(str(first_point['frame'] * 40) + '\n' + str(last_point['frame'] * 40))
  
  track_labels = open(os.path.join(track_folder_path, 'labels.txt'), 'w')
  
  for frame_idx,i in enumerate(track_vals):

    face_bound = i['bbox']

    frame = cv2.imread(os.path.join(args.videoFramesFolder,str(i['frame']) + '.jpg'))

    frame = frame[int(face_bound[1]):int(face_bound[3]), int(face_bound[0]):int(face_bound[2])]
    cv2.imwrite(os.path.join(track_folder_path, 'frame' + str(frame_idx) + '.jpg'), frame)
    track_labels.write("0\n")

  track_labels.close()
  
  return

def extract_track_data_5fps(args,track_item,video_details):

  track_name = track_item[0]
  track_vals = track_item[1]
  
  width,height,fps = video_details

  track_folder_path = os.path.join(args.destinationDir, track_name)
  
  if not os.path.exists(track_folder_path):
    os.mkdir(track_folder_path)

  first_point = track_vals[0]
  last_point = track_vals[-1]

  with open(os.path.join(track_folder_path, 'audio_tms.txt'), 'w') as f:
    f.write(str(first_point['time'] * 1000) + '\n' + str(last_point['time'] * 1000))

  track_labels = open(os.path.join(track_folder_path, 'labels.txt'), 'w')#file that contains labels for the current track

  start_frame = int(first_point['time'] * fps)
  
  #first point
  face_bound = first_point['bbox']
  for counter in range(0,3):
    
    frame = cv2.imread(os.path.join(args.videoFramesFolder, str(start_frame + counter) + ".jpg"))
    frame = frame[int(face_bound[1]):int(face_bound[3]), int(face_bound[0]):int(face_bound[2])]
    cv2.imwrite(os.path.join(track_folder_path, 'frame' + str(counter) + '.jpg'), frame)

  track_labels.write("0\n")
  
  offset = 3
  for i in track_vals[1:-1]:

    face_bound = i['bbox']
    current_frame = start_frame + offset
    for counter in range(0,5):

      frame = cv2.imread(os.path.join(args.videoFramesFolder, str(current_frame + counter) + ".jpg"))
      frame = frame[int(face_bound[1]):int(face_bound[3]), int(face_bound[0]):int(face_bound[2])]
      cv2.imwrite(os.path.join(track_folder_path, 'frame' + str(counter + offset) + '.jpg'), frame)

    track_labels.write("0\n")
    offset += 5

  #last point
  face_bound = last_point['bbox']
  for counter in range(0,3):

    frame = cv2.imread(os.path.join(args.videoFramesFolder, str(start_frame + offset + counter) + ".jpg"))
    frame = frame[int(face_bound[1]):int(face_bound[3]), int(face_bound[0]):int(face_bound[2])]
    cv2.imwrite(os.path.join(track_folder_path, 'frame' + str(offset + counter) + '.jpg'), frame)

  track_labels.write("0\n")
  track_labels.close()

def process_video(args):
  
  details = get_video_details(args)

  with open(args.jsonFile, 'r') as json_file:
    data = json.load(json_file)
  
  if int(data['mode']) == 0:
      extract_func = extract_track_data_5fps
  else:
      extract_func = extract_track_data_25fps
 
  del data['mode']
  
  for current_item in data.items():
      extract_func(args,current_item,details)

#---parser---

parser = argparse.ArgumentParser(description="Extract and store frames/info from videos.")
parser.add_argument('--videoFile', type=str, required = True,help = 'Provide the filename of the video to be processed.')
parser.add_argument('--jsonFile', type=str, required = True,help = 'Provide the filename of the json to be processed.')
parser.add_argument('--destinationDir', default = '.',type=str, help = 'Folder to store the extracted tracks.')
parser.add_argument('--videoFramesFolder', default="", type=str)

args = parser.parse_args()

if not os.path.exists(args.videoFile):
    print("Given video does not exist...")
    sys.exit()

if not os.path.exists(args.jsonFile):
    print("Given json file does not exist...")
    sys.exit()

if args.videoFramesFolder == "":
    args.videoFramesFolder = "framesDir"
    extract_video_frames(args)
    delete_folder = True
else:
    delete_folder = False

process_video(args)

if delete_folder:
    import shutil
    shutil.rmtree(args.videoFramesFolder)