"""
team amHacks 2021
GPU script
"""

from models import *
from utils import *
from sort import *
from IPython.display import HTML
from base64 import b64encode
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import skvideo.io  
from pytube import YouTube
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from IPython.display import clear_output
import pandas as pd
from scipy.signal import lfilter
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")


class_df=pd.read_csv("/content/config/coco.names", header=None)
class_list=class_df[0].to_list()

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

def download_from_yt(video_link):
    videopath=YouTube(video_link).streams.first().download("yt_downloads/")
    return(videopath)
    
def detector(input_video_path, num_frames, save_video, save_video_path, df):
  cmap = plt.get_cmap('tab20b')
  colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
  size = 720*16//9, 720
  duration = 2
  fps = 25

  # initialize Sort object and video capture
  vid = cv2.VideoCapture(input_video_path)

  if num_frames==-1:
    videodata = skvideo.io.vread(input_video_path)
    num_frames=videodata.shape[0]

  mot_tracker = Sort() 
  frame_arr=[]
  #while(True):
  
  for ii in tqdm(range(num_frames)):
      ret, frame = vid.read()
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      pilimg = Image.fromarray(frame)
      detections = detect_image(pilimg)
      #print(detections)##
      img = np.array(pilimg)
      pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
      pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
      unpad_h = img_size - pad_y
      unpad_w = img_size - pad_x
      unique_obj_ids=[]
      unique_obj_labels=[]
      if detections is not None:
          tracked_objects = mot_tracker.update(detections.cpu())
          #rint(tracked_objects)
          unique_labels = detections[:, -1].cpu().unique()
          #print(unique_labels)##
          n_cls_preds = len(unique_labels)
          for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
              box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
              box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
              y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
              x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

              color = colors[int(obj_id) % len(colors)]
              color = [i * 255 for i in color]
              cls = classes[int(cls_pred)]
              cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
              cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
              cv2.putText(frame, cls  + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

              unique_obj_ids.append(obj_id)
              unique_obj_labels.append(cls_pred)

      counts_list=[] #refresh for each frame
      for j in range(0,len(df.columns)):
          counts_list.append(unique_obj_labels.count(j))

      frame_arr.append(frame)
      df.loc[len(df)] = counts_list

  if save_video:
    skvideo.io.vwrite(save_video_path, frame_arr)
    print("output video saved at: "+save_video_path)

  return(save_video_path)


def count_total_unique_instances(df, class_list):
    tot_counts=[]
    ret_dict={}

    for i in class_list: 
        ret_dict["total unique "+i+"s"]=None

    for c in class_list:
        all_instances=list(insight_df[c])
        count=0
        for i in range(1,len(all_instances)):
            if all_instances[i]>all_instances[i-1]:
                count+=1
        tot_counts.append(count)

    for i in range(len(tot_counts)):
        ret_dict[list(ret_dict.keys())[i]]=tot_counts[i]

    return(ret_dict)
          
def visualize_classes(df, class_list, denoise, save_image, show, image_name): 
    n = 20 # the larger n is, the smoother curve will be in case denoise = True
    b = [1.0 / n] * n
    a = 1

    plt.rcParams['figure.figsize'] =10, 5
    plt.xlabel("frame number", fontsize = 15)
    plt.ylabel("count", fontsize = 15)

    all_arr=[]
    for c in class_list:
        arr=list(df[c])
        all_arr.append(arr)
        
    for i in range(len(all_arr)):
        if denoise:
            plt.plot(lfilter(b,a,all_arr[i]), label="count - "+class_list[i])
        else:
            plt.plot(all_arr[i], label="count - "+class_list[i])

    plt.grid()
    plt.legend(loc="upper left", numpoints=1)
    #plt.legend(loc="best")
    if show:
        plt.show()
    if save_image:
        plt.savefig(image_name)
        plt.clf()
        print("frame vs count plot saved at: ", image_name)

def plot_total_instances(df, class_list, save_image, image_name):
    dict_tots=count_total_unique_instances(insight_df, class_list)

    plt.rcParams['figure.figsize'] =10, 5
    width = 0.50
    labels=list(dict_tots.keys())
    ind = np.arange(len(list(dict_tots.keys())))
    plt.bar(ind, list(dict_tots.values()), width, label="Total unique instances", color='green', alpha=0.5)
    plt.xticks(ind, labels, rotation='vertical', fontsize=10)
    plt.ylabel('Number of unique instances', fontsize=15)
    plt.xlabel("Classes", fontsize=15)
    plt.grid(True)
    plt.legend(loc='best')
    if save_image:
        plt.savefig(image_name)
        print("Barplot of total unique instances saved at: ", image_name)
        
    else:
        plt.show()

config_path='/content/config/yolov3.cfg'
weights_path='/content/yolov3.weights'
class_path='/content/config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.to(device)
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

video_link = input('Enter the youtube video link: ')
local_path = download_from_yt(video_link)

## https://www.youtube.com/watch?v=MNn9qKG2UFI&list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB&index=2&ab_channel=KarolMajek
# https://www.youtube.com/watch?v=GJNjaRJWVP8&ab_channel=CCTVPeople
# https://www.youtube.com/watch?v=jjlBnrzSGjc&ab_channel=Panasonicsecurity

insight_df = pd.DataFrame([], columns=class_list[0:9])
len(insight_df.columns)

output_vid_path = detector(local_path,
                         num_frames=500,  # set as -1 for all frames
                         save_video=True, 
                         save_video_path="output_1.mp4", 
                         df=insight_df)

visualize_classes(insight_df,
                  ["car", "truck", "person", "motorbike"], 
                  denoise=True, 
                  save_image=True,
                  show=True,
                  image_name="count_vs_frame_plot.png")

plot_total_instances(insight_df,
                     ["car", "truck", "person", "motorbike"],
                     save_image=True,
                     image_name="bar_plot.png")

insight_df.to_csv("insight.csv")
print("insights dataframe saved as: insight.csv")
