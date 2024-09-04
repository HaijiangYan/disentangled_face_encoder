from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def extract_face(img_dir, out_dir, backend_model=0, align=True, face_show=False):
  #face detection and alignment
  backends = [
    'opencv', 
    'ssd', 
    'dlib', 
    'mtcnn', 
    'fastmtcnn',
    'retinaface', 
    'mediapipe',
    'yolov8',
    'yunet',
    'centerface',
  ]

  face_img = DeepFace.extract_faces(
    img_path = img_dir, 
    detector_backend = backends[backend_model], # ssd
    align = align,
  )
  # show the face
  # Convert the numpy array to a PIL Image
  img = Image.fromarray((face_img[0]['face']*255).astype(np.uint8))
  # Save the image
  img.save(out_dir)

  if face_show:
    img.show()




