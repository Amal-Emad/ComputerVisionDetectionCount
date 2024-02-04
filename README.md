# People Counting Project with YOLOv8 📊

## Project Overview:

**People Counting using YOLOv8:** This project employs the YOLOv8 model for real-time object detection, specifically focusing on people within a video stream. The primary objective is to accurately count the number of individuals entering and exiting predefined areas.

## 1. Abstract

In the realm of computer vision, this project utilizes YOLOv8, a state-of-the-art object detection model, along with additional tools such as OpenCV (cv2), pandas, numpy, ultralytics, and a custom object tracking module. The YOLOv8 model provides precise bounding box predictions, and the tracker ensures the continuity of object identities across frames.

## 2. Libraries and Tools:

### - OpenCV (cv2):
OpenCV is a powerful computer vision library used for image and video processing. It provides essential functions for image manipulation, object detection, and video analysis.

[Learn more about OpenCV](https://opencv.org/)

### - Pandas:
Pandas is a data manipulation library that plays a crucial role in this project for handling and processing data efficiently. It is used here to manage bounding box data obtained from YOLOv8 predictions.

[Explore Pandas documentation](https://pandas.pydata.org/)

### - NumPy:
NumPy is a fundamental package for scientific computing in Python. In this project, it is utilized for numerical operations, particularly in converting YOLOv8 output to a Pandas DataFrame.

[NumPy documentation](https://numpy.org/)

### - Ultralytics:
Ultralytics is a computer vision library that facilitates easy integration with YOLOv8 for training and inference. It streamlines the process of working with YOLO models.

[Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics)

### - Custom Tracker:
The custom tracking module plays a crucial role in maintaining object identities across frames, ensuring a smooth and consistent counting process.

## 3. Region of Interest (ROI) in Computer Vision:

In computer vision, a Region of Interest (ROI) is a specific area within an image or video frame that is selected for further analysis. In this project, two distinct ROIs, namely Area 1 and Area 2, are defined within the video stream. These ROIs are crucial for monitoring individuals entering and exiting the market. The project utilizes polygonal shapes to define these regions and track movements within them.

## 4. Project Structure:

The project involves defining two distinct areas, namely Area 1 and Area 2, within the video stream. Individuals crossing from Area 1 to Area 2 are considered as entering the market, while those crossing from Area 2 to Area 1 are considered as exiting the market.

## 5. Demo Overview:

- **People Counting using YOLOv8 - Test Video**

  
  [![People Counting Demo Gif](vid/TEST1-gif.gif)](vid/TEST1.mp4)

## 6. Future Development:

This project lays the foundation for advanced people counting applications. Future iterations can explore implementing the same concept in different videos or transitioning to real-time detection using cameras.

Your feedback and contributions are welcome as we continue to refine and extend the capabilities of this People Counting Project.

🚀 **Ready to Join the Counting?** [Let's Track People with YOLOv8](main.py)
