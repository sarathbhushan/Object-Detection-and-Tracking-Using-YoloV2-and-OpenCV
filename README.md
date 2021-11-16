# Object-Detection-and-Tracking-Using-YoloV2-and-OpenCV

Problem Statement:
Stock Keeping is crucial to run a successful business. We have many ERP solutions to address the same. But in the LPG cylinder business many modern solutions could not be applied given the constraints to use technology around the premises of storage. As, no elctronic or electro magnetic deivces are permitted within the storage godown. No modern stock keeping practices could be deployed and stock keeping was still a manual affait. And the prevailing black market doesn't help the case either, with LPG cylinders being sold in black by delivery guys/agents illegitimately. A nexus of stock keeper, agents and delivery guys can harm the business badly. 
  I had witnessed such problems faced by my father's LPG distributorship a couple of times, when he was heavily penalised (7-10 lakhs) for the imbalance in the outgoing and incoming stocks. So this has been one of my goals, to use object detection on CCTV video to solve this problem. 

Approach:
In this project I wanted to count LPG Cylinders going in and out of a gate. For this purpose, I had resorted to the following:
1. Data Gathering from the site and web images.
2. Annotation tool to mark the LPG cylinders in images
3. Training Yolo V2 network on these Images
4. Use dlib tracker to track the detected objects.
5. Implementing Centroid tracking on the objects identified, using opencv source code obtained from Pyimageserach to understand the direction of movement.
6. Finally counting the number of cylinders going up and down of a gate.

---------------------------------------------------------------------------
USAGE
To read from CCTV and write back out to disk the count of cylinders going in and out of godown.

----------------------------------------------------------------------------
Resources Used:
https://www.pyimagesearch.com/ - Centroid Tracking
https://manivannan-ai.medium.com/how-to-train-yolov2-to-detect-custom-objects-9010df784f36 - How to train Yolo V2 on custom Images
