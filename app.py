import numpy as np
import streamlit as st
from PIL import Image
import joblib
from collections import Counter
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

gender_nv_model = open("model.pkl", "rb")
gender_clf = joblib.load(gender_nv_model)
import glob
import pathlib
import os
import cloudmersive_virus_api_client
from cloudmersive_virus_api_client.rest import ApiException
from pprint import pprint


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
    dict = {}
    dict[0] = ffmpeg.probe(path)["streams"]
    j = 0

    for key, value in dict.items():
        for i in value:
            print(len(list(i.items())))
            for k, v in list(i.items()):
                new_rows.append(v)
                # print(k)
                j = j + 1
                print(j)
        # print("----------------------")
        # print(len(value))
        # print("----------------------")
    virus = vir(path)
    new_rows.append(virus)
    print("----------------------")
    print(len(new_rows))
    print("----------------------")
    # append row to the dataframe
    # df = pd.read_csv("out.csv")
    # df = df.append(new_rows, ignore_index=True)
    # features_list = ['video', 'Last modified', 'Created', 'mode', 'ino', 'dev', 'nlink', 'uid', 'gid', 'size', 'atime',
    #                  'mtime', 'ctime', 'metadata', 'index', 'codec_name', 'codec_long_name', 'profile', 'codec_type',
    #                  'codec_time_base', 'codec_tag_string', 'codec_tag', 'width', 'height', 'coded_width',
    #                  'coded_height', 'has_b_frames', 'pix_fmt', 'level', 'chroma_location', 'refs', 'is_avc',
    #                  'nal_length_size', 'r_frame_rate', 'avg_frame_rate', 'time_base', 'start_pts', 'start_time',
    #                  'duration_ts', 'duration', 'bit_rate', 'bits_per_raw_sample', 'nb_frames', 'disposition', 'tags',
    #                  'sample_aspect_ratio', 'display_aspect_ratio', 'sample_fmt', 'sample_rate', 'channels',
    #                  'channel_layout', 'bits_per_sample', 'max_bit_rate', 'color_range', 'color_space',
    #                  'color_transfer', 'color_primaries', 'quarter_sample', 'divx_packed', 'tag_codec_name',
    #                  'tag_profile', 'tag_codec_type', 'tag_codec_time_base', 'tag_codec_tag_string', 'tag_level',
    #                  'tag_nal_length_size', 'tag_r_frame_rate', 'tag_start_pts', 'tag_start_time', 'tag_frame_rate',
    #                  'tag_duration', 'tag_bit_rate', 'tag_display_aspect_ratio', 'tag_virus', 'tag_Offset']
    # features_list = features_list[-16:-1]
    # input_attributes = df[features_list]
    # isolation_forest = IsolationForest(contamination=0.05, n_estimators=100)
    # isolation_forest.fit(input_attributes.values)
    # df["mal"] = pd.Series(isolation_forest.predict(input_attributes.values))
    # df["mal"] = df["ma;"].map({1: 0, -1: 1})
    # # print(df["anomaly"].value_counts())
    # if df['mal'].iloc[-1] == 1:
    #     print("Anomalous point")
    #     return 1
    # elif df['mal'].iloc[-1] == 0:
    #     return 0
    #     print("Not an anomalous point")

    # -----------------------------------------------------------------

    # df = pd.read_csv("out.csv")
    # filename = findfile(vid.name,"/")
    #
    # import pandas
    # from sklearn import model_selection
    # from sklearn.linear_model import LogisticRegression
    # import joblib
    # features_list = ['video', 'Last modified', 'Created', 'mode', 'ino', 'dev', 'nlink', 'uid', 'gid', 'size', 'atime', 'mtime', 'ctime', 'metadata', 'index', 'codec_name', 'codec_long_name', 'profile', 'codec_type', 'codec_time_base', 'codec_tag_string', 'codec_tag', 'width', 'height', 'coded_width', 'coded_height', 'has_b_frames', 'pix_fmt', 'level', 'chroma_location', 'refs', 'is_avc', 'nal_length_size', 'r_frame_rate', 'avg_frame_rate', 'time_base', 'start_pts', 'start_time', 'duration_ts', 'duration', 'bit_rate', 'bits_per_raw_sample', 'nb_frames', 'disposition', 'tags', 'sample_aspect_ratio', 'display_aspect_ratio', 'sample_fmt', 'sample_rate', 'channels', 'channel_layout', 'bits_per_sample', 'max_bit_rate', 'color_range', 'color_space', 'color_transfer', 'color_primaries', 'quarter_sample', 'divx_packed', 'tag_codec_name', 'tag_profile', 'tag_codec_type', 'tag_codec_time_base', 'tag_codec_tag_string', 'tag_level', 'tag_nal_length_size', 'tag_r_frame_rate', 'tag_start_pts', 'tag_start_time', 'tag_frame_rate', 'tag_duration', 'tag_bit_rate', 'tag_display_aspect_ratio', 'tag_virus', 'tag_Offset']
    # features_list=features_list[-16:-1]
    # print(features_list)
    # print(df.columns)
    # X = df[features_list].to_numpy()
    #
    # # This column is the desired prediction we will train our model on
    # y = np.stack(df["mal"])
    # # dataframe = pd.read_csv("out.csv", names=features_list[-16:-1], header=None)
    #
    # X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y,test_size=0.1765, random_state=42 )
    # # Fit the model on training set
    # model = LogisticRegression()
    # model.fit(X_train, Y_train)
    # # save the model to disk
    #
    # joblib.dump(model, filename)
    #
    # # some time later...
    #
    # # load the model from disk
    # loaded_model = joblib.load(filename)
    # result = loaded_model.score(X_test, Y_test)
    # # ans=j.predict(filename)
    # print(filename)
    #
    # print(result)
    # -----------------------------------------------------------------
    # return ans
    # return
    # result = joblib.dump(gender_clf, vid.name)
    # print(result)

    import pandas
    from sklearn import model_selection
    # from sklearn.linear_model import LogisticRegression
    # import joblib
    # features_list = ['video', 'Last modified', 'Created', 'mode', 'ino', 'dev', 'nlink', 'uid', 'gid', 'size', 'atime', 'mtime', 'ctime', 'metadata', 'index', 'codec_name', 'codec_long_name', 'profile', 'codec_type', 'codec_time_base', 'codec_tag_string', 'codec_tag', 'width', 'height', 'coded_width', 'coded_height', 'has_b_frames', 'pix_fmt', 'level', 'chroma_location', 'refs', 'is_avc', 'nal_length_size', 'r_frame_rate', 'avg_frame_rate', 'time_base', 'start_pts', 'start_time', 'duration_ts', 'duration', 'bit_rate', 'bits_per_raw_sample', 'nb_frames', 'disposition', 'tags', 'sample_aspect_ratio', 'display_aspect_ratio', 'sample_fmt', 'sample_rate', 'channels', 'channel_layout', 'bits_per_sample', 'max_bit_rate', 'color_range', 'color_space', 'color_transfer', 'color_primaries', 'quarter_sample', 'divx_packed', 'tag_codec_name', 'tag_profile', 'tag_codec_type', 'tag_codec_time_base', 'tag_codec_tag_string', 'tag_level', 'tag_nal_length_size', 'tag_r_frame_rate', 'tag_start_pts', 'tag_start_time', 'tag_frame_rate', 'tag_duration', 'tag_bit_rate', 'tag_display_aspect_ratio', 'tag_virus', 'tag_Offset']
    #
    # final_df = df.reindex(features_list,axis=1)
    # print(features_list)
    # for column in df.columns[df.isna().any()].tolist():
    #     # df.drop(column, axis=1, inplace=True)
    #     df[column] = df[column].fillna(0)
    # # # We convert the feature list to a numpy array, this is required for the model fitting
    # print(df[final_df].to_numpy())
    # X = df[final_df].to_numpy()
    # # dataframe = pandas.read_csv(url, names=features_list[-16:-1])
    # array = df.values
    # # X = df[features_list[-16:-1]].to_numpy()
    # # Recheck all datatype before training to see we don't have any objects in our features
    # # In this example our model must get features containing only numbers so we recheck to see if we missed anything during preprocessing
    # y = np.stack(df["mal"])
    # test_size = 0.33
    # seed = 7
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1765, random_state=42)    # Fit the model on training set
    # model = LogisticRegression()
    # model.fit(X_train, y_train)
    # # save the model to disk
    # filename = vid.name
    # joblib.dump(model, filename)
    #
    # # some time later...
    #
    # # load the model from disk
    # loaded_model = joblib.load(filename)
    # result = loaded_model.score(X_test, y_test)
    # print(result)

    # gender_nv_model = open("model.pkl", "rb")
    # gender_nv_model = open("model.pkl", "rb")
    # gender_clf = joblib.load(gender_nv_model)
    # # model = joblib.load(gender_nv_model)
    # result = gender_clf.predict(filename)
    # print(result)

    # features_list = ['video', 'Last modified', 'Created', 'mode', 'ino', 'dev', 'nlink', 'uid', 'gid', 'size', 'atime', 'mtime', 'ctime', 'metadata', 'index', 'codec_name', 'codec_long_name', 'profile', 'codec_type', 'codec_time_base', 'codec_tag_string', 'codec_tag', 'width', 'height', 'coded_width', 'coded_height', 'has_b_frames', 'pix_fmt', 'level', 'chroma_location', 'refs', 'is_avc', 'nal_length_size', 'r_frame_rate', 'avg_frame_rate', 'time_base', 'start_pts', 'start_time', 'duration_ts', 'duration', 'bit_rate', 'bits_per_raw_sample', 'nb_frames', 'disposition', 'tags', 'sample_aspect_ratio', 'display_aspect_ratio', 'sample_fmt', 'sample_rate', 'channels', 'channel_layout', 'bits_per_sample', 'max_bit_rate', 'color_range', 'color_space', 'color_transfer', 'color_primaries', 'quarter_sample', 'divx_packed', 'tag_codec_name', 'tag_profile', 'tag_codec_type', 'tag_codec_time_base', 'tag_codec_tag_string', 'tag_level', 'tag_nal_length_size', 'tag_r_frame_rate', 'tag_start_pts', 'tag_start_time', 'tag_frame_rate', 'tag_duration', 'tag_bit_rate', 'tag_display_aspect_ratio', 'tag_virus', 'tag_Offset']
    # df = pd.read_csv("out.csv", names=["mal"], header=None)
    # print(features_list)
    # features_list = features_list[-16:-1]
    # final_df = df.reindex(features_list,axis=1)
    # print(features_list)
    # # # We convert the feature list to a numpy array, this is required for the model fitting
    # X = df[final_df].to_numpy()
    # # Recheck all datatype before training to see we don't have any objects in our features
    # # In this example our model must get features containing only numbers so we recheck to see if we missed anything during preprocessing
    # y = np.stack(df["mal"])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1765, random_state=42)
    # print("----------------------")
    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)
    # counter = Counter(y)
    # print(counter)
    # print("----------------------")
    # # Fit the model on training set
    # gender_nv_model.fit(X_train, y_train)
    # # save the model to disk
    # joblib.dump(gender_nv_model, filename)
    # # load the model from disk
    # loaded_model = joblib.load(filename)
    # result = loaded_model.score(X_test, y_test)
    # print(result)

    # append row to the dataframe
    # df = df.append(new_rows, ignore_index=True)
    # input_attributes = df[["mal"]]
    # isolation_forest = IsolationForest(contamination=0.05, n_estimators=100)
    # isolation_forest.fit(input_attributes.values)
    # df["mal"] = pd.Series(isolation_forest.predict(input_attributes.values))
    # df["mal"] = df["mal"].map({1: 0, -1: 1})
    # # print(df["anomaly"].value_counts())
    # if df['mal'].iloc[-1]== 1:
    #     print("Anomalous point")
    #     return 1
    # elif df['mal'].iloc[-1] == 0:
    #     return 0
    #     print("Not an anomalous point")

    # file.close()


# naiveBayesModel = open("model.pkl", "wb")
# joblib.dump(df,naiveBayesModel)
# naiveBayesModel.close()
#
#
#
#
# gender_nv_model = open("model.pkl","rb")
# gender_clf = joblib.load(gender_nv_model)

def predict_gender(video):
    result = predict(video)
    return result


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
        print(result)
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
