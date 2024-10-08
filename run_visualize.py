#!!!!!!!!!!!! viz_labels_5 CODE FROM (with some modifications): https://github.com/clovaai/lookwhostalking
"""
LICENSE:

Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import torch, glob, pdb, json, os, subprocess, cv2, shutil, argparse, cv2, sys

def extract_video_frames(args):

  os.makedirs(args.videoFramesFolder,exist_ok=True)

  command = ("ffmpeg -loglevel error -y -i %s -qscale:v 2 -threads 1 -f image2 -start_number 0 %s" % (args.videoPath,os.path.join(args.framesFolder,'%d.jpg'))) 
  output = subprocess.call(command, shell=True, stdout=None)

  if output != 0:
    print("Assertion fails but fuck it")
    
  return

def get_ff_list(data,frames_no):

  ff_list = [[] for _ in range(frames_no)]

  for track_data in data.values():
    for val in track_data:
      ff_list[val['frame']].append(val)

  return ff_list

def vis_labels_25(args):
  
  json_file = args.jsonPath
  
  video_base_ref = os.path.splitext(os.path.basename(json_file))[0]
  
  original_video_path = args.videoFile#os.path.join(args.avi_dir, "video.avi")
  original_audio_path = args.audioFile#os.path.join(args.avi_dir, "audio.wav")
  temp_output_avi = os.path.join(args.out_dir, video_base_ref + "_temp.avi")
  output_video = os.path.join(args.out_dir, video_base_ref + ".mp4")

  video = cv2.VideoCapture(original_video_path)

  width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))#get width of frame
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))#get height of frame
  fps = int(video.get(cv2.CAP_PROP_FPS))#get fps of current video
  
  video.release()

  with open(json_file) as jf:
    data = json.load(jf)
  
  del data['mode']
  
  frames_no = len(os.listdir(args.framesFolder))
  frame_faces = get_ff_list(data, frames_no)

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  video_out = cv2.VideoWriter(temp_output_avi, fourcc, fps, (width, height))

  for frame_idx in range(frames_no):
    frame = cv2.imread(os.path.join(args.framesFolder, str(frame_idx) + ".jpg"))
    faces_data = frame_faces[frame_idx]
    
    for face_data in faces_data:
      
      face_bound = face_data['bbox']

      color = int(face_data['label']) * 255
      cv2.rectangle(frame,(int(face_bound[0]),int(face_bound[1])),(int(face_bound[2]),int(face_bound[3])),(0,color,255-color),2)

    video_out.write(frame)

  video_out.release()
  
  print("Finished writing to temp file.")
  
  #combine output video with audio
  command = ("ffmpeg -loglevel error -i %s -i %s -c:v copy -c:a aac -strict experimental %s" % (temp_output_avi,original_audio_path,output_video))
  #command = ("ffmpeg -loglevel error -y -i %s -i %s -c:v libx264 -c:a aac -strict -2 %s" % (temp_output_avi,original_audio_path,output_video) #-async 1 
  output = subprocess.call(command, shell=True, stdout=None)
  
  assert output == 0
  
  os.remove(temp_output_avi)
  return

## assumes label at 5fps and video at 25fps
def vis_labels_5(args):
    
    jsonFile = args.jsonPath
    
    with open(jsonFile) as f:
        data = json.load(f)
    
    del data['mode']
    
    ref = os.path.splitext(os.path.basename(jsonFile))[0] # YouTube reference
    
    frm_dir = args.framesFolder
    vid = args.videoFileos#.path.join(args.avi_dir,'video.avi') # saved video AVI
    aud = args.audioFile#os.path.join(args.avi_dir,'audio.wav') # saved audio PCM
    tmp_avi = os.path.join(args.out_dir,'{}.avi'.format(ref)) # temporary video file
    out_mp4 = os.path.join(args.out_dir,'{}.mp4'.format(ref)) # mp4 file to save

    # get the list of frames
    flist_len = len(glob.glob(os.path.join(frm_dir,'*.jpg')))
    
    #sort the list
    flist = []
    for item in range(flist_len):
        flist.append(os.path.join(frm_dir, str(item) + '.jpg'))

    # make an empty list of face instances
    faces = [[] for i in range(1000000)]

    for datum in data:

        for fidx, frame in enumerate(data[datum]) :

            # current frame number at 25fps
            cfr = int(frame['time']*25)

            info = {'track': datum, 'bbox': frame['bbox'] ,'label': frame['label']}#, 'eval':frame['eval']}

            for ii in range(-2,3):
                if cfr+ii >= 0:
                    faces[cfr+ii].append(info)

    # get height and width of image
    first_image = cv2.imread(flist[0])
    fw = first_image.shape[1]
    fh = first_image.shape[0]

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vOut = cv2.VideoWriter(tmp_avi, fourcc, 25, (fw,fh))

    # for every frame
    for fidx, fname in enumerate(flist):

        # read image
        image = cv2.imread(fname)

        for face in faces[fidx]:

            # get bbox coordinates
            x1 = int(face['bbox'][0])
            x2 = int(face['bbox'][2])
            y1 = int(face['bbox'][1])
            y2 = int(face['bbox'][3])

            # color of bbox
            clr = float(face['label'])*255

            # print bbox on image
            cv2.rectangle(image,(x1,y1,x2-x1,y2-y1),(0,clr,255-clr),2)

            # double box if eval is positive
            #if face['eval'] == 1:
            #    cv2.rectangle(image,(x1-6,y1-6,x2-x1+12,y2-y1+12),(0,clr,255-clr),2)

            # write track number
            cv2.putText(image,face['track'],(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

        # write to video
        vOut.write(image)

    vOut.release()

    # combine saved video with original audio
    command = ("ffmpeg -loglevel error -i %s -i %s -c:v copy -c:a aac -strict experimental %s" % (tmp_avi,aud,out_mp4))
    #command = ("ffmpeg -loglevel error -y -i %s -i %s -c:v libx264 -c:a aac -strict -2 %s" % (tmp_avi,aud,out_mp4)) #-async 1 
    output = subprocess.call(command, shell=True, stdout=None)

    assert output == 0

    os.remove(tmp_avi)

parser = argparse.ArgumentParser(description = "VisFaceTracks");

parser.add_argument('--avi_dir', type=str, required = True, help='Path where video.avi/audio.wav exist.')
parser.add_argument('--jsonPath', required = True, help="Filepath to the corresponding json file", type=str)
parser.add_argument('--framesFolder', help="Filepath to the folder containing the extracted frames of the video", type=str, default="")
parser.add_argument('--out_dir', type=str, default = 'vis_asd', help = "File path of the output directory")

args = parser.parse_args();

if not os.path.isdir(args.avi_dir):
  print("Could not find avi_dir. Exiting...")
  sys.exit()

if not os.path.exists(args.jsonPath):
  print("Could not find json file(jsonPath argument). Exiting...")
  sys.exit()

videoPath = os.path.join(args.avi_dir,"video.avi")
audioPath = os.path.join(args.avi_dir,"audio.wav")

if not os.path.exists(videoPath):
  print("Could not find video.avi from the specified avi_dir. video.avi and audio.wav need to be files of the specified avi_dir")
  sys.exit()

if not os.path.exists(audioPath):
  print("Could not find audio.wav from the specified avi_dir. video.avi and audio.wav need to be files of the specified avi_dir")
  sys.exit()

setattr(args,"videoFile", videoPath)
setattr(args,"audioFile", audioPath)

if args.framesFolder == "":
  args.framesFolder = "framesDir"
  extract_video_frames(args)
  delete_folder = True
else:
  delete_folder = False
  
os.makedirs(args.out_dir,exist_ok=True)

with open(args.jsonPath) as f:
  if int(json.load(f)['mode']) == 0:
    vis_labels = vis_labels_5
  else:
    vis_labels = vis_labels_25

vis_labels(args)

if delete_folder:
    shutil.rmtree(args.framesFolder)