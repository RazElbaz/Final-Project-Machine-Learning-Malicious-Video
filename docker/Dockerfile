

FROM python:3.9.13




WORKDIR /app
RUN apt-get update && apt-get install -y apache2 && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install joblib
RUN pip install sklearn
RUN pip install streamlit
RUN pip install Image
RUN pip install -U scikit-learn
RUN pip install ffmpeg-python
RUN pip install git+https://github.com/Cloudmersive/Cloudmersive.APIClient.Python.Virus.git
RUN pip install pandas
RUN pip install plotly
RUN pip install Pillow
RUN pip install pprintpp
ADD . /app

COPY . /app
EXPOSE 8501
CMD streamlit run app.py