# Object-Detection-and-Tracking-Using-YoloV2-and-OpenCV

In this project I wanted to count LPG Cylinders going in and out of a gate. For this purpose, I had resorted to the following:
1. Annotation tool to mark the LPG cylinders in images
2. Training Yolo V2 network on these Images
3. Implementing Centroid tracking on the objects identified, using opencv source code obtained from Pyimageserach
4. Finally counting the number of cylinders going up and down of a gate.

---------------------------------------------------------------------------
# USAGE
# To read and write back out to video:
# python Cylinder_Counter_Final.py --input videos/example_123.mp4 --output output/Cylinder_top_count_15_9000.avi --yolo yolo-cylinder
# python Cylinder_Counter_Final.py --input videos/cylinder_images/Truckfull1.jpg --output output/Cylinder_count.jpg --yolo yolo-cylinder
# python Cylinder_counter.py --input http://192.168.1.200:3333/test.mjpeg --output output/cylinder_testnidad30.avi --yolo yolo-cylinder
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#   --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#   --output output/webcam_output.avi
# Add Area movement constraint only significant movement in the area to be considered as movement- Detect when there is no movement--- Zero frame is the frame--- 
# Multiple detection is an issue Think about it

----------------------------------------------------------------------------
Resources Used:
https://www.pyimagesearch.com/ - Centroid Tracking
https://manivannan-ai.medium.com/how-to-train-yolov2-to-detect-custom-objects-9010df784f36 - How to train Yolo V2 on custom Images

