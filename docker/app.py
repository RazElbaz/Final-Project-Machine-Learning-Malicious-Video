import numpy as np
import streamlit as st
from PIL import Image
import joblib
import pandas as pd
from sklearn import model_selection
import ffmpeg
import os
import cloudmersive_virus_api_client
from pprint import pprint
import plotly.express as px
import plotly.graph_objects as go
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

def predict_video(video):
    result = prediction(video)
    return result

def findfile(name, path):
    for dirpath, dirname, filename in os.walk(path):
        if name in filename:
            return os.path.join(dirpath, name)

df = pd.read_csv("out.csv")


def prediction(vid):
    path = findfile(vid.name, "/")
    print(path)
    new_rows = {}
    dict = {}
    dict[0] = ffmpeg.probe(path)["streams"]
    for key, value in dict.items():
        new_rows["metadata"] = value
    j = 0
    diction = new_rows["metadata"][0]
    # print(dict.items())
    for key, value in dict.items():
        print(key)
        for i in value:
            for k, v in list(i.items()):
                new_rows[k] = v

                j = j + 1

    def categorize(row, list1):
        print(row)
        print(list1)
        for i in range(len(list1)):
            if row == list1[i]:
                return i

        return -1

    list1 = list(['h264', 'mpeg4', 'rawvideo', 'aac'])
    new_rows['tag_codec_name'] = categorize(diction.get('codec_name'), list1)
    print(new_rows['tag_codec_name'])
    list1 = list(["High", "Main", "Constrained Baseline", "Simple Profile", None, "LC"])
    new_rows['tag_profile'] = categorize(diction.get('profile'), list1)
    new_rows['tag_codec_type'] = 0
    list1 = list(['1/50', '1001/60000', '1/60', '2069999/120060000', '3/173', '125089/6635520',
                  '1/58', '500/16757', '3/100' '1/48', '125/7118', '1/20', '1713103/99360000',
                  '1349999/78300000', '19853/1190000', '17753/888000', '269/17880',
                  '27817/2949120', '50/2997', '2343103/135900000', '5539/332500', '3081/185000',
                  '134651/2734080', '625/18694', '8081/312320', '18925/908842',
                  '912161/43740000', '1/10', '1/48000', '139259/4177920', '500/29901',
                  '2519999/146160000', '1147/132720', '175507/7875000', '3013931/181440000'])
    new_rows['tag_codec_time_base'] = categorize(diction.get('codec_time_base'), list1)
    print(new_rows['tag_codec_time_base'])
    list1 = list(['avc1', 'xvid', '[0][0][0][0]', 'mp4a'])
    new_rows['tag_codec_tag_string'] = categorize(diction.get('codec_tag_string'), list1)
    print(new_rows['tag_codec_tag_string'])
    list1 = list([50, 30, 21, 40, 31, 22, 12, 32, 42, 3, -99, None, 41])
    new_rows['tag_level'] = categorize(diction.get('level'), list1)
    print(new_rows['tag_level'])
    list1 = list(['4', None])
    new_rows['tag_nal_length_size'] = categorize(diction.get('nal_length_size'), list1)
    print(new_rows['tag_nal_length_size'])
    list1 = list(['25/1', '30000/1001', '30/1', '29/1', '173/6', '16757/1000', '50/3', '24/1'
                     , '3559/125', '10/1', '60/1', '60000/1001', '2997/100', '18694/625', '1199/50'
                     , '24000/1001', '0/0', '15/1', '29901/1000', '59/1', '269/12'])
    new_rows['tag_r_frame_rate'] = categorize(diction.get('r_frame_rate'), list1)
    print(new_rows['tag_r_frame_rate'])
    list1 = list([0, 707, 3690, 3407])
    new_rows['tag_start_pts'] = categorize(diction.get('start_pts'), list1)
    print(new_rows['tag_start_pts'])
    list1 = list(['0.000000'])
    new_rows['tag_start_time'] = categorize(diction.get('start_time'), list1)
    print(new_rows['tag_start_time'])

    def categorize2(row):
        print(row)
        rate = str(row)
        rate = rate.split('/')
        if rate[-1] == "1":
            return 0
        else:
            return 1

    r = list(diction.keys()).index('avg_frame_rate')
    new_rows['tag_frame_rate'] = categorize2(list(diction.values())[r])
    print(new_rows['tag_frame_rate'])

    def categorize3(row):
        if float(row) < 5.0:
            return 0
        elif float(row) < 10.0:
            return 1
        elif float(row) < 20.0:
            return 2
        else:
            return 3

    new_rows['tag_duration'] = categorize3(diction.get('duration'))
    print(new_rows['tag_duration'])

    sort2 = [11703, 15303, 17051, 72746, 77032, 84244, 84244, 106709, 107904, 116050, 128947, 133900, 143721, 144118,
             144152, 153430, 155114, 156249, 161637, 164299, 164532, 168558, 171138, 173586, 175252, 183083, 184961,
             185355, 188065, 191752, 191900, 195844, 196242, 200039, 203404, 213834, 217696, 217735, 219791, 223336,
             224403, 229214, 231504, 231616, 234446, 239507, 243541, 245199, 245340, 245387, 248756, 248964, 249532,
             249886, 250387, 252812, 258349, 259801, 264464, 265261, 265261, 265719, 270092, 271556, 273930, 274409,
             275854, 277041, 279394, 281725, 284545, 287444, 289508, 295913, 298402, 298597, 298869, 300866, 305461,
             312526, 315222, 318636, 328165, 333502, 343117, 344195, 344195, 345583, 346210, 346970, 353512, 366131,
             367159, 371753, 374470, 381461, 386163, 395414, 399018, 402536, 405481, 406910, 410954, 421927, 429463,
             433674, 436195, 449049, 450549, 462835, 466370, 484305, 516725, 537564, 544500, 608730, 616160, 665745,
             787996, 788592, 807380, 816463, 855525, 891358, 991598, 1113560, 1142210, 1304607, 1323945, 1589049,
             1727608, 2053620, 2069124, 2119170, 2652000, 3519905, 3967846, 3971019, 4304012, 4565234, 4613550, 4672498,
             4769466, 4784966, 4807029, 4807379, 4828012, 4832729, 4842131, 4842131, 4858085, 4863678, 4876157, 4890493,
             4895228, 4903426, 4905519, 4905519, 4909544, 4945958, 4992009, 4999034, 5017672, 5064522, 5085053, 5106089,
             5117610, 5130952, 5163839, 5172580, 5210459, 5211119, 5219735, 5219735, 5232150, 5265162, 5266382, 5266382,
             5266748, 5313080, 5319110, 5331588, 5331588, 5381217, 5389497, 5389497, 5398161, 5398161, 5406347, 5410350,
             5485577, 5511671, 5519798, 5570475, 5581448, 5581448, 5611998, 5614521, 5623119, 5623119, 5629254, 5658925,
             5658925, 5658960, 5688393, 5709660, 5709660, 5729124, 5761314, 5764492, 5765676, 5765676, 5838676, 5858478,
             5865971, 5865971, 5922762, 5937099, 5938377, 5969951, 5975133, 5991570, 6003257, 6004351, 6019332, 6048111,
             6096019, 6109250, 6119065, 6121203, 6134894, 6155816, 6165950, 6167591, 6175002, 6193355, 6195958, 6209699,
             6215114, 6218758, 6225503, 6227808, 6246757, 6254271, 6255696, 6257245, 6259720, 6275084, 6282675, 6284785,
             6294655, 6295005, 6296532, 6297484, 6318534, 6332494, 6332948, 6332958, 6340104, 6342269, 6346845, 6347171,
             6362296, 6367953, 6396121, 6408950, 6415214, 6435953, 6453113, 6497623, 6504563, 6508603, 6511020, 6526731,
             6530645, 6533882, 6536241, 6550336, 6554221, 6569603, 6586865, 6597261, 6598288, 6600664, 6604397, 6608445,
             6639240, 6649961, 6651042, 6666334, 6666879, 6669531, 6682286, 6684146, 6772143, 6788195, 6796061, 6803216,
             6810212, 6843478, 6851573, 6865680, 6896137, 6900798, 6910503, 6929521, 6934487, 6944780, 7033349, 7038471,
             7081449, 7101145, 7101510, 7164853, 7165966, 7212681, 7320944, 7334906, 7368861, 7592616, 7927637, 8054592,
             11397433, 11397433, 11397433, 11397433, 11397433, 11397433, 11397433, 11397433, 11397433, 11397433,
             11397433]

    def Average(lst):
        return sum(lst) / len(lst)

    def categorize4(row):
        if float(row) < avg:
            return 1
        else:
            return 0

    avg = Average(sort2)
    new_rows['tag_bit_rate'] = categorize4(diction.get('bit_rate'))
    print(new_rows['tag_bit_rate'])
    list1 = list([None, '16:9', '1:1', '4:3', '25:18', '2:3' ',1048317:699424', '127:90', '27:20' , '3:2', '239:180', '259:144', '269:180', '160:77', '361:270', '237:109','427:240', '241:180', '23:16' '124:89', '33:40', '8:5', '9:16'])
    new_rows['tag_display_aspect_ratio'] = categorize(diction.get('display_aspect_ratio'), list1)
    print(new_rows['tag_display_aspect_ratio'])
    print(new_rows.keys())
    virus = vir(path)
    new_rows["tag_virus"] = virus

    df = pd.read_csv("out.csv")
    print(df.columns.to_list)
    real=pd.read_csv("out.csv",names=["mal"])
    print(real["mal"].shape)
    features_list = df.columns.to_list()
    features_list.remove('mal')
    features_list = features_list[-16:-1]
    print(features_list)
    X = df[features_list].to_numpy()
    y = (df["mal"])
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y,test_size=0.1765, random_state=42 )



    # define input
    new_input = list(new_rows.values())[-15:]
    print(new_input)


    from sklearn import neighbors

    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,Y_train)
    new_output = knn.predict([new_input])
    print(new_input, new_output)
    return new_output[0]



def load_images(file_name):
    img = Image.open(file_name)
    return st.image(img, width=150)


def main():
    image = Image.open('./pictures/creating-rolling-out-effective-cyber-security-strategy.jpeg')
    image2 = Image.open('./pictures/index.jpeg')
    col1, mid, col2 = st.columns([1, 1, 100])
    with col1:
        st.image(image, width=150)
    with mid:
        st.image(image2, width=150)
    # st.image(image,image2, width=200)
    st.title("Final Project ML Video")


    st.subheader("MP4 Files")
    video_file = st.file_uploader("Upload File", type=['mp4'])

    # checkbox
    # check if the checkbox is checked
    # title of the checkbox is 'Show/Hide'
    if st.checkbox("Video Info"):
        # display the text if the checkbox returns True value
        if video_file is not None:
            file_details = {"Filename": video_file.name, "FileType": video_file.type, "FileSize": video_file.size}
            st.write(file_details)
            # Check File Type
            if video_file.type == "mp4":
                st.video(video_file, format="video/mp4", start_time=0)
                st.text(str(video_file.read(), "utf-8"))  # empty
                video_bytes = video_file.read()
                st.write(video_bytes)  # works

    if st.button("Predict"):
        result = predict_video(video_file)
        print(result)
        if result == "benign":
          img = 'good.png'
          st.success('The video: {}  is not a malicious video'.format(video_file.name))
        else:

          img = 'bad.png'
          st.warning('The video: {}  is a malicious video'.format(video_file.name))


        load_images(img)



    # https://www.geeksforgeeks.org/python-plotly-tutorial/#Bar
    st.subheader("Data Visualization")
    df = pd.read_csv("out.csv")
    charts = st.selectbox("charts", [None,'Bubble charts : mal','Bubble charts : virus','Bar Chart : r_frame_rate','Pie Charts : tag_Offset','Pie Charts : tag_virus','Pie Charts : mal','Scatter Plot : bit_rate'])
    if charts=='Bubble charts : mal':
        fig = px.scatter(df, x="mal",color="mal")
        st.plotly_chart(fig)
    elif charts == 'Bubble charts : virus':
        fig = px.scatter(df, y="tag_virus", color="tag_virus")
        st.plotly_chart(fig)
    elif charts=='Bar Chart : r_frame_rate':
        fig = px.bar(df, x="r_frame_rate")
        st.plotly_chart(fig)
    elif charts=='Pie Charts : tag_Offset':
        fig=px.pie(df, names="tag_Offset")
        st.plotly_chart(fig)
    elif charts=='Pie Charts : tag_virus':
        fig=px.pie(df, names="tag_virus")
        st.plotly_chart(fig)
    elif charts=='Pie Charts : mal':
        fig=px.pie(df, names="mal")
        st.plotly_chart(fig)
    elif charts == 'Scatter Plot : bit_rate':
        fig = px.scatter(df, y="bit_rate")
        st.plotly_chart(fig)

    # print(df.columns)



    st.subheader("About")
    st.info("Raz Elbaz Final Project")
    st.info("Link to the project:  "+ "https://github.com/RazElbaz/Final-Project-Machine-Learning-Malicious-Video")

main()
