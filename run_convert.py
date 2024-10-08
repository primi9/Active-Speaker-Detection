#!!!!!!!!!!!! CODE FROM: https://github.com/clovaai/lookwhostalking

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

#!/usr/bin/env python3

import sys, time, os, pdb, argparse, subprocess, glob, cv2, itertools

def get_length(filename):
    result = subprocess.run(["ffprobe", "-loglevel","quiet" ,"-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


def convert_video(args, videofile, ref):

    vid_out = os.path.join(args.avi_dir,ref,'video.avi')
    aud_out = os.path.join(args.avi_dir,ref,'audio.wav')

    # Check if the conversion is already done
    if os.path.exists(aud_out) and not args.overwrite:
        print('Already done %s'%ref)
        return

    # Make new directory to save the files
    os.makedirs(os.path.join(args.avi_dir,ref),exist_ok=True)

    # Convert audio and video
    if get_length(videofile) >= args.max_seconds:
      command = ("ffmpeg -loglevel quiet -y -i %s -async 1 -qscale:v 2 -r %d -t %.1f %s" % (videofile, args.frame_rate, args.max_seconds, vid_out))
    else:
      command = ("ffmpeg -loglevel quiet -y -i %s -async 1 -qscale:v 2 -r %d %s" % (videofile, args.frame_rate, vid_out)) #-async 1  -deinterlace
    output = subprocess.call(command, shell=True, stdout=None)

    assert output == 0

    command = ("ffmpeg -loglevel quiet -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (vid_out, aud_out)) 
    output = subprocess.call(command, shell=True, stdout=None)

    assert output == 0



# ========== ========== ========== ==========
# # PARSE ARGS AND RUN SCRIPTS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "ConvertVideo");
parser.add_argument('--data_dir',       type=str, default='exps', help='Output direcotry');
parser.add_argument('--frame_rate',     type=int, default=25,   help='Frame rate');
parser.add_argument('--max_seconds',    type=float, default=1200, help='Maximum length of input video in seconds, if longer, we cut to this length');
parser.add_argument('--overwrite',      dest='overwrite', action='store_true', help='Re-run pipeline even if already run')
args = parser.parse_args();

setattr(args,'avi_dir',os.path.join(args.data_dir,'pyavi'))
setattr(args,'original_dir',args.data_dir)

exts = ['.mkv', '.mp4', '.webm']

files = [glob.glob('{}/*{}'.format(args.original_dir,ext)) for ext in exts]
files = sum(files,[])

for file in files:
    ref = os.path.splitext(os.path.basename(file))[0]
    convert_video(args,file,ref)
