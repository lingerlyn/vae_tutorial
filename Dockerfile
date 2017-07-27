FROM jupyter/scipy-notebook

WORKDIR /srv/jupyter

# add dir contents
ADD . /srv/jupyter

RUN pip install -r requirements.txt

EXPOSE 8000

CMD jupyter notebook --no-browser --port=8000