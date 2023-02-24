# Final project in the Methods for detecting cyber attacks course
## Topic: Machine learning to detect malware in video files

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Description

The subject of my final project in the Cyber Attack Detection Methods course is a learning machine that knows how to detect malware in video files.

With the advancement of technology, new malware technologies and ways to carry out  attacks are emerging.
MP4 is the most common video file format due to the fact that it provides high-quality video in a relatively small size compared to other video formats, and provides lossless compression after editing and re-saving the files. Video files are not generally considered malicious file types, but it is possible for malware to be embedded in a video file or disguised as a video file. Therefore, MP4 is considered a good candidate for embedding video malware. 

#### MP4
The data in an MP4 file is divided into two parts:
(1) the media-related data: includes streams of video, audio, and subtitles; 
(2) metadata: specifies information about the original data which assists in identifying the nature and features of that data and making it easier to use, search, and manage the video such as flags, timestamps, bit rate and so forth.

#### Where can malicious content be found?
Damage in the video may be in 3 possible places: header, video data and end of file, therefore these will be part of our tests.

### Dataset
warning! Some of the videos contain content that may harm your computer, download them in a virtual machine. Link to dataset: https://1drv.ms/u/s!ApgPVNBNwhrXgyPPPamaCsgf8H2u?e=9BrWyf

The dataset in this paper will be 333 videos in MP4 format. Since there is a shortage of finding malicious videos to download, we had to create a few, and the ones we could find we included in the dataset. The data set includes: videos that were previously malicious, normal videos that were downloaded to train the learning machine, and new malicious videos that we created through the following process: We downloaded several videos, took each video and split it into FRAMES using Python code that we wrote. Next, we selected one of the FRAMES and inserted a message into it by inserting malicious information in the LSB bit. An MP4 is a common “container format” for video files that allows you to store a lot of video and audio information in a smaller file size. The minimum file size in the dataset is 17069 bits and the maximum 14637345 bits. Some videos contain malware and some don't, we would like to develop a learning machine that will know how to recognize this. 



To run the app: `streamlit run app.py`
### view my Streamlit app in your browser press on the Local URL:
![run](https://github.com/RazElbaz/Task-1-Anomaly-detection/blob/main/images/run.png)

### To run with docker:
1) Building a Docker image:
`docker build -t streamlitapp:latest .`
2) Creating a container:
`docker run -p 8501:8501 streamlitapp:latest`

### Docker build:

![dockerbuild](https://github.com/RazElbaz/Final-Project-Machine-Learning-Malicious-Video/blob/main/pictures/docker_build.png)

### Docker run:

![dockerrun](https://github.com/RazElbaz/Final-Project-Machine-Learning-Malicious-Video/blob/main/pictures/docker_run.png)

### The app:

![app](https://github.com/RazElbaz/Final-Project-Machine-Learning-Malicious-Video/blob/main/pictures/sreamlit_app.png)

## Malicious video

![mal_video](https://github.com/RazElbaz/Final-Project-Machine-Learning-Malicious-Video/blob/main/pictures/mal_video.png)

## Not a malicious video

![not_mal_video](https://github.com/RazElbaz/Final-Project-Machine-Learning-Malicious-Video/blob/main/pictures/not_mal_video.png)

## Data Visualization

![chart1](https://github.com/RazElbaz/Final-Project-Machine-Learning-Malicious-Video/blob/main/pictures/data_chart.png)

![chart2](https://github.com/RazElbaz/Final-Project-Machine-Learning-Malicious-Video/blob/main/pictures/data_chart_2.png)


