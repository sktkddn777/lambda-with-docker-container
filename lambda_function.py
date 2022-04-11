import torch
import numpy as np
import cv2
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator

import boto3
import urllib.parse
import uuid

s3 = boto3.client('s3')

def apply_yolo(img_path, upload_path):
  # Model Setting
  MODEL_PATH = '/var/task/weights/e50b32.pt'

  img_size = 416
  conf_thres = 0.5  # confidence threshold
  iou_thres = 0.45  # NMS IOU threshold
  max_det = 1000  # maximum detections per image
  classes = None  # filter by class
  agnostic_nms = False  # class-agnostic NMS

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  ckpt = torch.load(MODEL_PATH, map_location=device)
  model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
  class_names = ['fire'] # model.names
  stride = int(model.stride.max())
  colors = ((0, 255, 0)) # (gray, red, green)

  img = cv2.imread(img_path, cv2.IMREAD_COLOR)

  img_input = letterbox(img, img_size, stride=stride)[0]
  img_input = img_input.transpose((2, 0, 1))[::-1]
  img_input = np.ascontiguousarray(img_input)
  img_input = torch.from_numpy(img_input).to(device)
  img_input = img_input.float()
  img_input /= 255.
  img_input = img_input.unsqueeze(0)

  pred = model(img_input, augment=False, visualize=False)[0]

  # postprocess
  pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
  pred = pred.cpu().numpy()

  pred[:, :4] = scale_coords(img_input.shape[2:], pred[:, :4], img.shape).round()
  annotator = Annotator(img.copy(), line_width=3, example=str(class_names), font='data/malgun.ttf')

  for p in pred:
          class_name = class_names[int(p[5])]

          x1, y1, x2, y2 = p[:4]
          print(p)
          annotator.box_label([x1, y1, x2, y2], '%s %d' % (class_name, float(p[4]) * 100), color=colors[int(p[5])])

  result_img = annotator.result()
  cv2.imwrite(upload_path, result_img)
  return

def lambda_handler(event, context):
  bucket = event['Records'][0]['s3']['bucket']['name']
  key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
  tmpkey=key.replace('/','')
  download_path='/var/task/tmp/{}{}'.format(uuid.uuid4(), tmpkey)
  upload_path='/var/task/tmp/model-{}'.format(tmpkey)

  s3.download_file(bucket, key, download_path)
  apply_yolo(download_path, upload_path)
  s3.upload_file(upload_path,'{}-model'.format(bucket), key)
