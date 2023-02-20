import numpy as np
import streamlit as st
from PIL import Image
import joblib
from collections import Counter

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import pandas as pd
# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import os, time
import ffmpeg
import sys
from pprint import pprint  # for printing Python dictionaries in a human-readable way
from pathlib import Path
import ffmpeg
import sys
from pprint import pprint # for printing Python dictionaries in a human-readable way
from pathlib import Path

from sklearn.preprocessing import LabelEncoder

gender_nv_model = open("model.pkl", "rb")
gender_clf = joblib.load(gender_nv_model)
import glob
import pathlib
import os
import cloudmersive_virus_api_client
from cloudmersive_virus_api_client.rest import ApiException
from pprint import pprint

df = pd.read_csv("out.csv")
def vir(path):
    configuration = cloudmersive_virus_api_client.Configuration()
    configuration.api_key['Apikey'] = "42d07908-4cec-4f99-9324-71424b190a71"
    api = cloudmersive_virus_api_client.ScanApi(cloudmersive_virus_api_client.ApiClient(configuration))
    virus = []

    (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(path)
    if size <= 3000000:  # the limit for the free tier (3 MB)
        ans = api.scan_file(path)
        pprint(ans)
        curr = str(ans)
        if "False" in curr:
            return 1
        else:
            return 0
    else:
        return 0


def findfile(name, path):
    for dirpath, dirname, filename in os.walk(path):
        if name in filename:
            return os.path.join(dirpath, name)


# filepath = findfile("file2.txt", "/")
# print(filepath)

df = pd.read_csv("out.csv")


def predict(vid):
    df = pd.read_csv("out.csv")

    print("----------------------")
    # files = glob.glob('*.mp4')
    # for file in files:
    #     print(file)
    p = findfile(vid.name, "/")
    print(p)

    path = findfile(vid.name, "/")
    (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(path)
    new_rows = (
    vid.name, time.ctime(os.path.getmtime(path)), time.ctime(os.path.getctime(path)), mode, ino, dev, nlink, uid, gid,
    size, atime, mtime, ctime)
    new_rows = list(new_rows)
    print(new_rows)
    dict = {}
    dict[0]=ffmpeg.probe(path)["streams"]
    for key, value in dict.items():
        new_rows.append(value)
    j = 0

    for key, value in dict.items():
        for i in value:
            # print(len(list(i.items())))
            for k, v in list(i.items()):
                new_rows.append(v)
                # print(k,v)
                # print(k)
                j = j + 1
                # print(j)
        # print("----------------------")
        # print(len(value))
        # print("----------------------")
    virus = vir(path)
    new_rows.append(virus)
    # print(new_rows)



    new_rows = [x for x in new_rows if type(x) == int or type(x) == float ]
    # print(new_rows)
    list_f=['mode', 'ino', 'dev', 'nlink', 'uid', 'gid','size', 'atime', 'mtime', 'ctime','index','width','height','coded_width','coded_height','has_b_frames','level','refs','start_pts','duration_ts','tag_virus']
    df1 = pd.read_csv("out.csv",usecols =list_f)
    print(df1.columns)
    real=pd.read_csv("out.csv",names=["mal"])
    print(real["mal"].shape)

    X = df1[df1.columns.to_list()].to_numpy()
    print(len(df1.columns.to_list()))


    def categorize(row):
        if row['mal'] == "benign":
            return 0
        else:
            return 1

    real["mal"] = real.apply(lambda row: categorize(row), axis=1)
    print(np.stack(real["mal"]))
    y=(real["mal"][0:-1])

    print(y.shape)
    print(X.shape)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y,test_size=0.1765, random_state=42 )
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, Y_train)
    # define input
    new_input = [new_rows]
    # get prediction for new input
    new_output = classifier.predict(new_input)
    # summarize input and output
    print(new_input, new_output)
    # Fit the model on training set
    # model = LogisticRegression()
    # model.fit(X_train, Y_train)
    # isolation_forest = IsolationForest(contamination=0.05, n_estimators=100)
    # isolation_forest.fit(df.values)
    # df["mal"] = pd.Series(isolation_forest.predict(df.values))
    # df["mal"] = df["mal"].map({1: 0, -1: 1})
    # # print(df["anomaly"].value_counts())
    # if df['mal'].iloc[-1] == 1:
    #     print("Anomalous point")
    #     return 1
    # elif df['mal'].iloc[-1] == 0:
    #     return 0
    #     print("Not an anomalous point")

def predict_gender(video):
    result = predict(video)
    # return result


def load_images(file_name):
    img = Image.open(file_name)
    return st.image(img, width=300)


def main():
    """Gender Classifier App
    With Streamlit

  """

    st.title("Final Project ML Video")
    html_temp = """
  <div style="background-color:blue;padding:10px">
  <h2 style="color:grey;text-align:center;">Streamlit App </h2>
  </div>

  """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("MP4 Files")
    video_file = st.file_uploader("Upload File", type=['mp4'])
    video_bytes = None
    if st.button("Process"):
        if video_file is not None:
            file_details = {"Filename": video_file.name, "FileType": video_file.type, "FileSize": video_file.size}
            st.write(file_details)
            # Check File Type
            if video_file.type == "mp4":
                # raw_text = video_file.read() # read as bytes
                # st.write(raw_text)
                # st.text(raw_text) # fails
                st.video(video_file, format="video/mp4", start_time=0)
                st.text(str(video_file.read(), "utf-8"))  # empty
                video_bytes = video_file.read()
                # st.text(raw_text) # Works
                st.write(video_bytes)  # works

    else:
        st.subheader("About")
        st.info("Built with Streamlit")
        st.info("Jesus Saves @JCharisTech")
        st.text("Jesse E.Agbe(JCharis)")
    if st.button("Predict"):
        video_bytes = video_file.read()
        result = predict_gender(video_file)
        # print(result)
        # if result == -1:
        #   prediction = "Not an anomalous point"
        #   img = 'good.png'
        # else:
        #   result == 1
        #   prediction = "Anomalous point"
        #   img = 'bad.png'

        # st.success('mal: {} '.format(video_file.title(),prediction))
        # load_images(img)


main()
