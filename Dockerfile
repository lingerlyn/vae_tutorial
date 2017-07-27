FROM jupyter/scipy-notebook

USER root

WORKDIR /srv/jupyter

# add dir contents
ADD . /srv/jupyter

RUN pip install -r requirements.txt

WORKDIR /srv/jupyter/notebooks

# sign notebooks
RUN jupyter trust *.ipynb

EXPOSE 8000

CMD jupyter notebook --no-browser --port=8000 --allow-root