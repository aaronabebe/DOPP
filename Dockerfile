FROM python:3.7.2-slim
EXPOSE 8501

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip
#RUN pip install streamlit
RUN pip install -r requirements.txt

ADD transformed.csv /app
ADD continents.csv /app
ADD unesco_poverty_dataset.csv /app
COPY data_prep.py /app
COPY plots.py /app
ENTRYPOINT [ "streamlit", "run"]
CMD ["data_prep.py"]