from __future__ import print_function

import os,sys,cv2,math,json,argparse, torch
import numpy as np
from s3fd import S3FD

BATCH_SIZE = 32
TRACK_COUNTER = 0
json_data = {}

def extract_video_frames(args):
    
  import subprocess

  os.makedirs(args.videoFramesFolder,exist_ok=True)

  command = ("ffmpeg -loglevel error -y -i %s -qscale:v 2 -threads 1 -f image2 -start_number 0 %s" % (args.videoPath,os.path.join(args.videoFramesFolder,'%d.jpg'))) 
  output = subprocess.call(command, shell=True, stdout=None)

  if output != 0:
    print("Assertion fails so fuck it")
    
  return

def get_start_end_frames(start_tms,end_tms):
    """
    if math.isclose(start_tms, 0.0, abs_tol=1e-9):
        start_tms = 0
    else:
        start_tms = ((start_tms // 0.2) + 1) * 0.2
    
    start_frame = int(start_tms * fps)
    
    end_tms = (end_tms // 0.2) * 0.2
    
    if (end_tms % 0.2) != 0:
      end_frame = int(round(end_tms * fps,8))
    else:
      end_frame = int(round(end_tms * fps,8)) - 1
    """
    
    start_frame = (int(start_tms // 0.2) + 1) * 5
    end_frame = int(end_tms // 0.2) * 5
    
    return start_frame, end_frame

def find_scenes(video_path, threshold=30.0):
    
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    return scene_list

def calc_overlapping_area_perc(coords,aof):

  x1,y1,x2,y2 = coords[0]
  x1_m,y1_m,x2_m,y2_m = coords[1]
  
  width_dist = min(x2_m, x2) - max(x1_m,x1)
  height_dist = min(y2_m, y2) - max(y1_m,y1)

  overlapping_area = 0
  if width_dist > 0 and height_dist > 0:
    overlapping_area = width_dist * height_dist

  return (overlapping_area / (aof[0] + aof[1]))

def createTracks(boxes, curr_frame, OAP, step = 5):

  face_tracks = []

  for entries in boxes:
    for entry in entries:
      matched = False
      coords = entry[:4]
      aof = (coords[2] - coords[0]) * (coords[3] - coords[1])
      for track in face_tracks:
        if track['updated'] == -1 or track['updated'] == 1:
          continue
        
        overlap_perc = calc_overlapping_area_perc([track['track_coords_list'][-1], coords], [track['AOF'], aof])
  
        if overlap_perc >= OAP:
          track['track_coords_list'].append(coords)
          track['updated'] = 1
          track['AOF'] = aof
          matched = True
          break
        
      if not matched:# if we didnt find any match
        new_entry = {'track_coords_list': [coords], 'updated': 1, 'starting_frame': curr_frame, 'AOF': aof}
        face_tracks.append(new_entry)
    
    deactivate_tracks(face_tracks)
    curr_frame += step

  return face_tracks

def deactivate_tracks(face_tracks):
  
  for track in face_tracks:
    if track['updated'] == 1:
      track['updated'] = 0
    elif track['updated'] == 0:
      track['updated'] = -1

def store_new_faces_25(face_tracks, MIN_TRACK_LENGTH):

  global TRACK_COUNTER
  global json_data

  for track_id,track in enumerate(face_tracks):
    
    track_coords_list = track['track_coords_list']
    
    if len(track_coords_list) < MIN_TRACK_LENGTH:
      continue
    
    start_frame = track['starting_frame']
    
    track_details = []
 
    for frame_no,coords in enumerate(track_coords_list, start_frame):
      for idx in range(4):
        if coords[idx] < 0:
          coords[idx] = 0
      
      track_details.append({"frame": frame_no, "bbox": coords.tolist(), 'label': 0})
    
    json_data["track_" + str(TRACK_COUNTER + track_id).zfill(6)] = track_details
  
  TRACK_COUNTER += len(face_tracks)

def store_new_faces_5(face_tracks, MIN_TRACK_LENGTH):

  global TRACK_COUNTER
  global json_data

  for track_id,track in enumerate(face_tracks):
    
    track_coords_list = track['track_coords_list']
    
    if len(track_coords_list) < MIN_TRACK_LENGTH:
      continue
    
    start_frame = track['starting_frame']

    track_start_tms = start_frame / 25
    
    track_details = []

    for time_idx,coords in enumerate(track_coords_list):
      for idx in range(4):
        if coords[idx] < 0:
          coords[idx] = 0

      track_details.append({"time": round(track_start_tms + 0.2 * time_idx,2), "bbox": coords.tolist(), 'label': 0})
    
    json_data["track_" + str(TRACK_COUNTER + track_id).zfill(6)] = track_details
    
  TRACK_COUNTER += len(face_tracks)

def process_scene_25(args,start_tms,end_tms):
    
    global json_data
    json_data['mode'] = 1
    
    #this will probably change
    start_frame = math.ceil(start_tms * 25) + 1
    end_frame = math.floor(end_tms * 25)#not including end_frame
    
    n_frames = end_frame - start_frame
    
    if n_frames < args.MIN_TRACK_LENGTH:
      if args.to_print:
        print("Skipping this scene...Too small")
      return
    
    frames = []
    boxes = []

    for frame_idx in range(start_frame, end_frame):
      frames.append(cv2.imread(os.path.join(args.videoFramesFolder,str(frame_idx) + ".jpg")))
    
    c_frame = 0
    while True:
      
      if (n_frames - c_frame) > BATCH_SIZE:
        boxes.extend(net.detect_faces_gpu(frames[c_frame:c_frame + BATCH_SIZE], s = args.SCALE_FACTOR, conf_th = args.CONF_THRESH))
        c_frame += BATCH_SIZE
      else:
        boxes.extend(net.detect_faces_gpu(frames[c_frame: n_frames]))
        break
        
    face_tracks = createTracks(boxes, start_frame, args.OAP, step = 1)
    store_new_faces_25(face_tracks,args.MIN_TRACK_LENGTH)
    return

def process_scene_5(args,start_tms,end_tms):
    
    global json_data
    json_data['mode'] = 0
    start_frame,end_frame = get_start_end_frames(start_tms, end_tms)#including end_frame
    
    if (end_frame - start_frame) / 5 + 1 < args.MIN_TRACK_LENGTH:
      if args.to_print:
        print("Skipping this scene...Too small")
      return
    
    boxes = []

    curr_frame = start_frame
    while True:
      
      frame = cv2.imread(os.path.join(args.videoFramesFolder,str(curr_frame) + ".jpg"))
      boxes.append(net.detect_faces(frame,conf_th = args.CONF_THRESH, scales = [args.SCALE_FACTOR]))
      
      curr_frame += 5
      if curr_frame > end_frame:
        face_tracks = createTracks(boxes, start_frame, args.OAP, step = 5)
        store_new_faces_5(face_tracks, args.MIN_TRACK_LENGTH)
        return

parser = argparse.ArgumentParser(description='Face Track Creation with S3FD')
parser.add_argument('--videoPath',required = True, type=str)
parser.add_argument('--storePath', default='.', type=str)
parser.add_argument('--videoFramesFolder', default="", type=str)
parser.add_argument('--MIN_TRACK_LENGTH', default=0, type = int)
parser.add_argument('--OAP', default=0.3, type = float)
parser.add_argument('--CONF_THRESH', default=0.9, type = float)
parser.add_argument('--mode', default=2, type=int, help = "Mode == 0 specifies that face data will be extracted every 0.2s (5fps). Mode == 1 specifies that face data will be extracted for every frame (25fps). If not specified, mode will be determined by the availability of gpu.")
parser.add_argument('--verbose',dest='to_print', action='store_true', help='Set this flag if you want the script to print details.')

args = parser.parse_args()

if not os.path.exists(args.videoPath):
    print("Video does not exist (videoPath argument error)")
    sys.exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#extract all the video frames if it they are not already extracted
if args.videoFramesFolder == "":
    args.videoFramesFolder = "framesDir"
    extract_video_frames(args)
    delete_folder = True
else:
    delete_folder = False

if args.mode == 1:
    process_scene = process_scene_25
elif args.mode == 0:
    process_scene = process_scene_5
else:
    if device == torch.device("cpu"):
        process_scene = process_scene_5
    else:
        process_scene = process_scene_25

net = S3FD(device = device)

video = cv2.VideoCapture(args.videoPath)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
assert fps == 25

video.release()

if args.to_print:
  print("Width:", width)
  print("Height:", height)

if height < 500:
  SCALE_FACTOR = 1.0
elif height < 1100:
  SCALE_FACTOR = 0.25
else:
  SCALE_FACTOR = 0.1

setattr(args,"SCALE_FACTOR", SCALE_FACTOR)

scene_list = find_scenes(args.videoPath)
scene_tms = []

last_tms = round(num_frames / 25.0,2)

if len(scene_list) == 0:
  
  scene_tms.append(0.0)
  scene_tms.append(last_tms)
else:
  for scene in scene_list:
    scene_tms.append(scene[0].get_seconds())
  
  tmp = scene_list[-1][1].get_seconds()
  scene_tms.append(tmp)
  if last_tms > tmp:
    scene_tms.append(last_tms)
  
if args.to_print:
  print("Scene timestamps:")
  print(scene_tms)

for i in range(len(scene_tms[:-1])):
  if args.to_print:
    print("Processing scene: %.2f - %.2f" %(scene_tms[i], scene_tms[i+1]))
  process_scene(args,scene_tms[i], scene_tms[i+1])

if delete_folder:
  import shutil
  shutil.rmtree(args.videoFramesFolder)

with open(args.storePath, 'w') as json_file:
  json.dump(json_data, json_file, indent=2)