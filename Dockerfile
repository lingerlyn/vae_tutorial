FROM jupyter/scipy-notebook

USER $NB_USER

WORKDIR $HOME

# add dir contents
ADD . $HOME

USER root

RUN pip install -r requirements.txt

WORKDIR notebooks

# give notebook user permission
RUN chown -R $NB_USER .

USER $NB_USER

# sign notebooks
RUN jupyter trust *.ipynb

EXPOSE 8888

USER $NB_USER
#CMD jupyter notebook --no-browser --port=8888 --allow-root