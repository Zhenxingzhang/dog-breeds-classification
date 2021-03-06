FROM gcr.io/tensorflow/tensorflow:1.5.0-gpu


COPY run_tools.sh /
RUN chmod +x /run_tools.sh

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.14.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "-vv", "-g", "--"]


RUN apt-get update && apt-get install -y --no-install-recommends \
	tmux \
	python-tk \
	vim \
	git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

#COPY jupyter_notebook_config.py /root/.jupyter/

COPY ./src /app/src
COPY ./images /app/images
COPY ./data/breeds.csv /app/data/breeds.csv
COPY ./notebooks/*.ipynb /app/notebooks/
COPY ./setup /app/setup
COPY ./config /app/config

#CMD [ "jupyter", "notebook", "--allow-root"]
CMD ["/run_tools.sh", "--allow-root"]