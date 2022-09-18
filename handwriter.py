import csv
import copy
import datetime
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from threading import Timer
import os
import qrcode
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import cv2 as cv
import mediapipe as mp

from model import KeyPointClassifier
from utils import CvFpsCalc

from app import get_args, \
                calc_landmark_list, calc_bounding_rect, \
                pre_process_landmark, \
                draw_bounding_rect, draw_landmarks

write_color = {
  'Right':(255, 0, 0, 255), # (B, G, R, A)
  'Left':(0, 255, 0, 255)
}

def main():
  # 引数解析
  args = get_args()

  cap_device = args.device
  cap_width = args.width
  cap_height = args.height

  use_static_image_mode = args.use_static_image_mode
  min_detection_confidence = args.min_detection_confidence
  min_tracking_confidence = args.min_tracking_confidence

  assert args.interpolation_num > 0

  use_brect = True

  gauth = GoogleAuth()
  gauth.LocalWebserverAuth()

  drive = GoogleDrive(gauth)

  # 保存場所: My Drive/img_folder
  img_folder = drive.ListFile({'q': 'title = "img_folder"'}).GetList()
  if len(img_folder) == 0:
    metadata = {'title': 'img_folder', 'mimetype': 'application/vnd.google-apps.folder'}
    f = drive.CreateFile(metadata)
    f.Upload()
    img_folder = drive.ListFile({'q': 'title = "img_folder"'}).GetList()

  parent_id = img_folder[0]['id'] # img_folderのid

  # カメラ準備
  cap = cv.VideoCapture(cap_device)
  cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
  cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

  cap_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
  cap_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
  print("(height, width): ", (cap_height, cap_width))

  # モデルロード
  mp_hands = mp.solutions.hands
  hands = mp_hands.Hands(
      static_image_mode=use_static_image_mode,
      max_num_hands=1,
      min_detection_confidence=min_detection_confidence,
      min_tracking_confidence=min_tracking_confidence,
  )

  keypoint_classifier = KeyPointClassifier()

  # ラベル読み込み
  with open('model/keypoint_classifier/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

  # FPS計測モジュール
  cvFpsCalc = CvFpsCalc(buffer_len=10)

  # 人差し指の軌跡を記録する
  writer = np.zeros((cap_height, cap_width, 4), dtype=np.float32)
  prev_x, prev_y = None, None
  which_hand = None

  while True:
    fps = cvFpsCalc.get()

    # キー処理(ESC : フォトモード)
    key = cv.waitKey(1)
    if key == 27:  # ESC
      if not photo_mode(cap, writer, drive, parent_id, args.delay):
        break

    ret, image = cap.read()
    if not ret:
      break

    image = cv.flip(image, 1)  # ミラー表示
    debug_image = copy.deepcopy(image)

    # 検出実施
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks is not None:
      for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                            results.multi_handedness):
        # 外接矩形の計算
        brect = calc_bounding_rect(debug_image, hand_landmarks)
        # ランドマークの計算
        landmark_list = calc_landmark_list(debug_image, hand_landmarks)

        # 相対座標・正規化座標への変換
        pre_processed_landmark_list = pre_process_landmark(
            landmark_list)

        # ハンドサイン分類
        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

        # 描画
        debug_image = draw_bounding_rect(use_brect, debug_image, brect)
        debug_image = draw_landmarks(debug_image, landmark_list)
        which_hand_prev = which_hand
        which_hand = handedness.classification[0].label[0:]
        writer, prev_x, prev_y = draw_writer(which_hand, which_hand_prev, hand_sign_id, landmark_list, cap_height, cap_width, writer, prev_x, prev_y, args.interpolation_num)

        debug_image = draw_info_text(
            debug_image,
            brect,
            handedness,
            keypoint_classifier_labels[hand_sign_id],
        )

    # debug_image = mask(debug_image, writer)
    debug_image = write(debug_image, writer)

    debug_image = draw_info(debug_image, fps)

    # 画面反映
    cv.imshow('Hand Writer', debug_image)

  cap.release()
  cv.destroyAllWindows()
  cv.waitKey(1)

def photo_mode(cap, writer, drive:GoogleDrive, parent_id, delay=180):
  now = datetime.datetime.now()
  path = now.strftime('%Y-%m-%d %H:%M:%S') + '.png'
  while True:
    ret, image = cap.read()
    if not ret:
      break

    image = cv.flip(image, 1)  # ミラー表示
    image = write(image, writer)

    debug_image = copy.deepcopy(image)

    cv.putText(debug_image, 'Photo Mode', (10, 30), cv.FONT_HERSHEY_SIMPLEX
                ,1.0, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow('Hand Writer', debug_image)

    key = cv.waitKey(1) # キー処理(Enter: 画像保存, ESC: 終了)
    if key == 13: # Enter
      cv.imwrite(path, image)
      # Now Uploading
      now_uploading = copy.deepcopy(debug_image)
      cv.putText(now_uploading, 'Now Uploading...', (100, 200), cv.FONT_HERSHEY_TRIPLEX
                  ,4.0, (0, 0, 0), 10, cv.LINE_AA)
      cv.imshow('Hand Writer', now_uploading)
      cv.waitKey(1)

      # GDrive保存
      metadata = {'title': path, 'mimetype': 'image/png', 'parents': [{'id': parent_id}]}
      f = drive.CreateFile(metadata)
      f.SetContentFile(path)
      f.Upload()
      f.InsertPermission({'type': 'anyone',
                          'value': 'anyone',
                          'role': 'reader'})
      Timer(delay,f.Delete).start()

      cv.putText(debug_image, 'Uploading Finished!', (80, 200), cv.FONT_HERSHEY_TRIPLEX
                  ,3.0, (0, 0, 0), 8, cv.LINE_AA)
      cv.imshow('Hand Writer', debug_image)
      cv.waitKey(1)

      # QRコード
      url = 'https://drive.google.com/file/d/{}/view'.format(f.get('id'))
      qr = qrcode.make(url)
      qr = qr.convert('L')
      link = Image.new('L', (600, 550), 255) # 背景
      link.paste(qr, (75, 50))
      draw = ImageDraw.Draw(link)
      font = ImageFont.truetype('./hiraginokakugoW4.ttc', 20) # フォントを指定
      text = 'Here is the link to GDrive. / Google Driveのリンクはこちら'
      draw.text(xy=(300, 25), text=text, fill=0, font=font, anchor='mm')
      text = 'Press key to close / キーを押して閉じる'
      draw.text(xy=(300, 50), text=text, fill=0, font=font, anchor='mm')
      link = np.array(link, dtype=np.uint8)
      cv.imshow('Link', link)
      cv.waitKey(0)
      cv.destroyWindow('Link')
      cv.waitKey(1)
      os.remove(path)
      return True
    elif key == 27: # ESC
      return True
    elif key == 113: # 'q'
      return False

def draw_writer(which_hand, which_hand_prev, hand_sign_id, landmark_list, cap_height, cap_width, writer, prev_x, prev_y, interpolation_num):
  if hand_sign_id == 2: # 指差し
    index_finger = landmark_list[8]
    if prev_x is not None and which_hand == which_hand_prev:
      interpolation_x = list(map(int, np.linspace(prev_x, index_finger[1], interpolation_num)[1:])) # 前の点と検出した点で10個の線形補間
      interpolation_y = list(map(int, np.linspace(prev_y, index_finger[0], interpolation_num)[1:])) #
      for x, y in zip(interpolation_x, interpolation_y):
        writer = cv.circle(
          writer,
          (y, x),
          20,
          write_color[which_hand],
          -1,
        )
    else:
      writer = cv.circle(
          writer,
          index_finger,
          20,
          write_color[which_hand],
          -1,
      )
    prev_x, prev_y = index_finger[1], index_finger[0]
  else: # パーかグー
    prev_x, prev_y = None, None # 点の履歴を削除
    if hand_sign_id == 0: # パー
      # 人差し指の軌跡を初期化
      writer = np.zeros((cap_height, cap_width, 4), dtype=np.float32)
  return writer, prev_x, prev_y

def write(image, writer):
  image[:, :] = image[:, :] * (1 - writer[:, :, 3:] / 255) + \
                      writer[:, :, :3] * (writer[:, :, 3:] / 255)
  return image

def draw_info_text(image, brect, handedness, hand_sign_text):
  cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
              (0, 0, 0), -1)

  info_text = handedness.classification[0].label[0:]
  if hand_sign_text != "":
      info_text = info_text + ':' + hand_sign_text
  cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

  return image

def draw_info(image, fps):
  cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
              1.0, (0, 0, 0), 4, cv.LINE_AA)
  cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
              1.0, (255, 255, 255), 2, cv.LINE_AA)

  return image

if __name__ == '__main__':
  main()
