import sys
import cv2
sys.path.append("/mnt/e/WorkSpace/CPlusPlus/2d_pose_estimation/bind/build/")
import inference

root_path = "/mnt/e/WorkSpace/CPlusPlus/2d_pose_estimation/bind/"
input_img = cv2.imread("family.jpg")
inference.detectImg(input_img, root_path)
cv2.imwrite("family_out.jpg", input_img)








