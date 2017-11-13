FROM python:2.7

EXPOSE 8888
EXPOSE 6006

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY jupyter_notebook_config.py /root/.jupyter/

COPY ./src /app/src
COPY ./images /app/images
#COPY ./checkpoints /app/checkpoints
#COPY ./frozen /app/frozen
#COPY ./data/train/labels.csv /app/data/train/
#COPY ./data/*.tfrecords /app/data/
#COPY ./data/breeds.csv /app/data/
COPY ./*.ipynb /app/
#COPY ./summary /app/summary
COPY ./setup /app/setup

RUN mkdir /data
VOLUME /data

WORKDIR /app

#CMD [ "jupyter", "notebook", "--allow-root"]
CMD ["/bin/bash"]
