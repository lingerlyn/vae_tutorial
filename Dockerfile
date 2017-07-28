FROM jupyter/scipy-notebook

USER root

WORKDIR $HOME

# add dir contents
ADD . $HOME

RUN pip install -r requirements.txt

WORKDIR notebooks

# sign notebooks
RUN jupyter trust *.ipynb

EXPOSE 8888

USER $NB_USER
#CMD jupyter notebook --no-browser --port=8888 --allow-root