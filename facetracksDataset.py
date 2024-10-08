import torch, os, glob, random, cv2
import numpy as np
import python_speech_features
from python_speech_features import mfcc,delta
from scipy.io import wavfile
from PIL import Image
from torch.utils.data import Dataset

class facetracksDataset(Dataset):

  __load_num_frames = 25 # 1 second context
  __left_boundaries = __load_num_frames // 2
  __right_boundaries = (__load_num_frames + 1) // 2
  __sr = 16000
  __resize_dims = 116
  __noise_level = 0.05
  __audio_padding_len = __sr // 2

  def __init__(self, videos_folder, batch_size = 32, augmentations = True, load_state = False, eval = False,verbose = True, infer_all = False):

      self.__videos_data_folder = videos_folder
      self.__augment = augmentations
      self.__eval_mode = eval
      self.__load_state = load_state
      self.__to_print = verbose
      self.__frames_list = []
      self.__track_details = {}
      self.__audio_dict = {}
      
      if infer_all:
         self.__frame_step = 1
         self.__audio_step = 0.04
      else:
          self.__frame_step = 5
          self.__audio_step = 0.2
      
      for i in sorted(os.listdir(videos_folder)):
          self.__load_new_video(i)
      
      self.__track_folder_len = len(next(iter(self.__track_details)))# all track folder names have the same length
      
      if load_state:

        if not os.path.exists("drive/MyDrive/ASD/loader_status.txt"):
          print("Load file does not exist. Exiting.")
          return

        with open("drive/MyDrive/ASD/loader_status.txt", 'r') as load_file:

          lines = load_file.readlines()
          self.__frames_list = [line.strip() for line in lines[:-2]]
          self.__batch_index = int(lines[-2].strip())
          self.__batch_size = int(lines[-1].strip())
      else:
        self.__batch_index = 0
        self.__batch_size = batch_size

      self.__dataset_length = len(self.__frames_list) // self.__batch_size
      
      remainder = len(self.__frames_list) % self.__batch_size
      if remainder != 0:
        self.__dataset_length += 1
        self.__last_batch_length = remainder
      else:
        self.__last_batch_length = self.__batch_size

      if verbose:
        print("Dataset length(number of batches): ", self.__dataset_length)
        print("Dataset length(number of inputs): ", len(self.__frames_list))
        print("Batch size: ", self.__batch_size)
        print("Batch index: ", self.__batch_index)
        print("Last batch length (remainder): ", self.__last_batch_length)
        print("Number of tracks loaded: %d" %len(self.__track_details))

      return

  def save_loader(self):

    with open("drive/MyDrive/ASD/loader_status.txt", 'w') as save_file:
      for line in self.__frames_list:
          save_file.write(line + '\n')
      save_file.write(str(self.__batch_index) + '\n')
      save_file.write(str(self.__batch_size))
  
  def get_track_details(self, feature_extraction = True):
    
    import copy
    
    if not feature_extraction:
      return copy.deepcopy(self.__track_details)
    
    track_details_copy = copy.deepcopy(self.__track_details)
    
    track_names = list(track_details_copy.keys())
    track_names.sort()

    track_details_copy[track_names[0]][0] = 0 #[starting_index, num_frames, num_labels]
    previous_ending_index = track_details_copy[track_names[0]][2]

    for track_name in track_names[1:]:
      track_details_copy[track_name][0] = previous_ending_index# change tms with starting index of track in frames_list
      previous_ending_index += track_details_copy[track_name][2]  
      
    return track_details_copy
  
  def get_remaining_iter(self):
    return (self.__dataset_length - self.__batch_index)

  def get_index_list(self):
    
    import copy
    
    return copy.deepcopy(self.__frames_list)
    
  def getbatch(self,index,length):
    
    batch_items = self.__frames_list[index:index + length]

    visuals = []
    audios = []

    for item in batch_items:
      
      track_name = item[1:self.__track_folder_len + 1]
      frame_num = int(item[self.__track_folder_len + 1:])
      
      num_frames = self.__track_details[track_name][1]
      
      visuals.append(self.__get_visuals(track_name, frame_num, num_frames))
      audios.append(self.__get_audio(track_name, frame_num))

    return torch.stack(audios, dim = 0), torch.stack(visuals, dim = 0)

  def get_zero_input(self):

    visual_input = torch.full((3,25,self.__resize_dims, self.__resize_dims), -1).float()
      
    mfcc_features = mfcc(np.zeros((1,self.__sr)), self.__sr, numcep = 13, winlen = 0.025, winstep = 0.01)
    
    delta_features = delta(mfcc_features, N=2)
    delta2_features = delta(delta_features, N=2)
    
    audio_input = np.stack((mfcc_features, delta_features, delta2_features), axis=0)
    audio_input = torch.from_numpy(audio_input).float()

    return audio_input, visual_input
  
  def __len__(self):
    return self.__dataset_length

  def __iter__(self):
    
    if not self.__load_state:
      self.__batch_index = 0
      if not self.__eval_mode:
        random.shuffle(self.__frames_list)
    else:
      self.__load_state = False
    
    if self.__to_print:
      print("First element: ", self.__frames_list[0])
    
    return self

  def __next__(self):

    if self.__batch_index >= self.__dataset_length:
      raise StopIteration

    batch = self[self.__batch_index]
    self.__batch_index += 1

    return batch
  
  def __getitem__(self, index):

    if index >= self.__dataset_length or index < 0:
      raise IndexError("Index out of bounds")

    if index == self.__dataset_length - 1:
      current_batch = self.__last_batch_length
    else:
      current_batch = self.__batch_size

    visuals = []
    audios = []
    labels = []

    temp_batch_index = index * self.__batch_size

    for item in range(current_batch):

      temp_str = self.__frames_list[temp_batch_index + item]
      label = int(temp_str[0])
      track_name = temp_str[1:self.__track_folder_len + 1]
      frame_num = int(temp_str[self.__track_folder_len + 1:])

      num_frames = self.__track_details[track_name][1]

      visuals.append(self.__get_visuals(track_name, frame_num, num_frames))
      audios.append(self.__get_audio(track_name, frame_num))
      labels.append(label)

    return torch.stack(audios, dim = 0), torch.stack(visuals, dim = 0) , torch.tensor(np.array(labels).reshape(current_batch, 1), dtype = torch.float32)

  def __get_visuals(self, track_name, frame_num, num_frames):

    if self.__augment:
      #source: https://github.com/TaoRuijie/TalkNet-ASD/blob/main/dataLoader.py
      augType = random.choice(['orig', 'flip', 'crop', 'rotate'])
      new_size = int(self.__resize_dims * random.uniform(0.7, 1))
      x_new, y_new = np.random.randint(0, self.__resize_dims - new_size), np.random.randint(0, self.__resize_dims - new_size)
      M = cv2.getRotationMatrix2D((self.__resize_dims/2,self.__resize_dims/2), random.uniform(-15, 15), 1)
    else:
      augType = 'orig'

    current_frame = frame_num * self.__frame_step

    faces = []
    
    left_b = max(current_frame - self.__left_boundaries , 0)
    right_b = min(num_frames, current_frame + self.__right_boundaries)
    
    for _ in range(current_frame, self.__left_boundaries):
      faces.append(np.zeros((self.__resize_dims, self.__resize_dims,3)))

    for frame_idx in range(left_b, right_b):

      face = cv2.imread(os.path.join(track_name, "frame" + str(frame_idx) + ".jpg"))[...,::-1]
      face = cv2.resize(face, (self.__resize_dims,self.__resize_dims))

      if augType == 'orig':
        faces.append(face)
      elif augType == 'flip':
        faces.append(cv2.flip(face, 1))
      elif augType == 'crop':
        faces.append(cv2.resize(face[y_new : y_new + new_size, x_new : x_new + new_size] , (self.__resize_dims,self.__resize_dims)))
      elif augType == 'rotate':
        faces.append(cv2.warpAffine(face, M, (self.__resize_dims,self.__resize_dims)))

    for _ in range(num_frames, current_frame + self.__right_boundaries):
      faces.append(np.zeros((self.__resize_dims, self.__resize_dims,3)))

    visual_input = torch.from_numpy(np.array(faces)).permute(3, 0, 1, 2).contiguous().float()
    visual_input = visual_input.mul_(2.).sub_(255).div(255)

    return visual_input

  def __get_audio(self, track_name, current_frame):

    start_tms = self.__track_details[track_name][0]

    current_tms = (start_tms + self.__audio_step * current_frame) * self.__sr

    audio_key = track_name[:track_name.rfind('/')] #na vro kati allo gia to /

    audio = self.__audio_dict[audio_key][self.__audio_padding_len + int(current_tms - self.__sr // 2) : self.__audio_padding_len + int(current_tms + self.__sr // 2)] # 1 second context...

    if self.__augment:
      noise = np.random.normal(0, audio.std(), len(audio))
      audio = audio + self.__noise_level * noise
    
    mfcc_features = mfcc(audio, self.__sr, numcep = 13, winlen = 0.025, winstep = 0.01)
    
    # Apply CMVN per feature
    mfcc_features = (mfcc_features - np.mean(mfcc_features, axis = 0, keepdims = True)) / (np.std(mfcc_features, axis = 0, keepdims = True) + (1e-10))

    delta_features = delta(mfcc_features, N=2)
    delta2_features = delta(delta_features, N=2)

    audio_input = np.stack((mfcc_features, delta_features, delta2_features), axis=0)
    audio_input = torch.from_numpy(audio_input).float()

    return audio_input

  def __load_track_details(self, track_folder):

    frame_list = []
    with open(os.path.join(track_folder,'labels.txt')) as lbs:
      for i,line in enumerate(lbs):
        frame_list.append(line.rstrip() + track_folder + str(i))  

    self.__frames_list.extend(frame_list)

    with open(os.path.join(track_folder,'audio_tms.txt')) as a_tms:
      tms = float(a_tms.readline().strip('\n')) / 1000

    num_frames = len(glob.glob(os.path.join(track_folder, "*.jpg")))
    num_labels = len(frame_list)
 
    self.__track_details[track_folder] = [tms, num_frames, num_labels]

    return

  def __load_new_video(self, video_url):

    video_folder_path = os.path.join(self.__videos_data_folder, video_url)

    video_tracks = glob.glob(os.path.join(video_folder_path, "track*"))
    if self.__eval_mode:
      video_tracks.sort()

    #load the audio for every video, since we need the audio of 1 video for multiple face tracks, sr (sampling rate) is always 16000
    sr, audio = wavfile.read(os.path.join(video_folder_path, 'audio.wav'))

    self.__audio_dict[video_folder_path] = np.pad(audio, (self.__audio_padding_len, self.__audio_padding_len), 'constant')

    for track_folder in video_tracks:
      self.__load_track_details(track_folder)
    
    return
