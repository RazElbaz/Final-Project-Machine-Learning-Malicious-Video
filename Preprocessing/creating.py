# I used the following website: https://medium.com/@vedanshvijay/steganography-5d9d8a557587
# Installing and Importing required modules:
from os.path import join
import moviepy.editor
from PIL import Image
import cv2
import os
import shutil

original_image_file = ""  # to make the file name global variable
count = 52

def frames_to_video():
    if not os.path.exists("./malVideo"):
        os.makedirs("malVideo")
    global count
    if not os.path.exists("./temp"):
        os.makedirs("temp")
    path = "./temp"

    frames = []
    # ignore .DS_Store
    files = [f for f in os.listdir(path) if not f.startswith('.')]
    files.sort(key=lambda x: int(x[0:-4]))
    print(files)

    for f in files:
        filename = join(path, f)

        # reading each file
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        height, width, channels = img.shape
        size = (width, height)

        # append img to frames array
        frames.append(img)

    fps = 25
    output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    # writing image to video
    for img in frames:
        output.write(img)
    output.release()

    videoClip = moviepy.editor.VideoFileClip("output.mp4")
    audioClip = moviepy.editor.AudioFileClip("output.mp3")
    finalClip = videoClip.set_audio(audioClip)
    finalClip.write_videofile("malware{}.mp4".format(count), fps)
    count+=1

def video_to_frames(input_video):
    # change to video folder
    if not os.path.exists("./temp"):
        os.makedirs("temp")

    # get file path for video
    # input_video = input("Enter Video Name To Encryption: ")
    cap = cv2.VideoCapture(input_video)  # default fps: 30
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps = " + str(fps))

    # change bak to parent directory

    path_to_save = './temp'

    current_frame = 0
    while (True):

        # capture each frame
        ret, frame = cap.read()
        # stop loop when video ends
        if not ret:
            break

        # Save frame as a png file
        name = str(current_frame) + '.png'

        print('Creating: ' + name)
        cv2.imwrite(os.path.join(path_to_save, name), frame)

        # keep track of how many images you end up with
        current_frame += 1

    # release capture
    cap.release()
    cv2.destroyAllWindows()
    print('Frames saved!')
    # Extract audio from video
    video = moviepy.editor.VideoFileClip(input_video)
    audio = video.audio

    audio.write_audiofile('output.mp3')


class LSB():
    # encoding part :
    def encode_image(self, img, msg):
        length = len(msg)
        if length > 255:
            print("text too long! (don't exeed 255 characters)")
            return False
        encoded = img.copy()
        width, height = img.size

        index = 0
        for row in range(height):
            for col in range(width):
                r, g, b = img.getpixel((col, row))
                # first value is length of msg
                if row == 0 and col == 0 and index < length:
                    asc = length
                elif index <= length:
                    c = msg[index - 1]
                    asc = ord(c)
                else:
                    asc = b
                encoded.putpixel((col, row), (r, g, asc))
                index += 1
        return encoded

    # decoding part :
    def decode_image(self, img):
        width, height = img.size
        msg = ""
        index = 0
        for row in range(height):
            for col in range(width):
                r, g, b = img.getpixel((col, row))
                # first pixel r value is length of message
                if row == 0 and col == 0:
                    length = b
                elif index <= length:
                    msg += chr(b)
                index += 1
        lsb_decoded_image_file = "lsb_" + original_image_file
        # img.save(lsb_decoded_image_file)
        ##print("Decoded image was saved!")
        return msg


# This function would delete the temp directory
def clean_temp(path="./temp"):
    if os.path.exists(path):
        shutil.rmtree(path)
        print("[INFO] temp files are cleaned up")


# This function would extraxt audio from the video so as to stitch them back later.
def input_main(f_name, input_string):
    # get frames from video
    video_to_frames(f_name)
    # change dir
    os.chdir("./temp/")

    # choose frame to encode
    img_name = "10.png"
    if not os.path.isfile(img_name):
        raise Exception("Image not found")
    print("Chose 10.png to encode")
    input_string="malware"
    img = LSB().encode_image(Image.open(img_name), input_string)
    img.save(img_name)
    os.chdir("..")
    # build encoded video
    frames_to_video()
    clean_temp()


# Inputting the video📹
# First we would decide the inputs. That is first the user needs to specify whether they want to encode or decode a video. Then they would input the video file with extension which we would read using OpenCV.
if __name__ == "__main__":
    folder = r'/home/raz/PycharmProjects/video/LSBsteganography-WithoutExpandingTheVideoSize/malVideo/'


    for file_name in os.listdir(folder):

        source = folder + file_name
        # print(file_name)
        input_main(source, "malware")


