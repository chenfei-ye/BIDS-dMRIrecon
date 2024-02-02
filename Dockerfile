FROM mindsgo-sz-docker.pkg.coding.net/neuroimage_analysis/base/msg_baseimage_cuda11:deepFS
MAINTAINER Chenfei <chenfei.ye@foxmail.com>

# install prerequisite for python-spams 
RUN apt update && apt-get install -y \
	liblapack-dev \ 
	libatlas-base-dev \
	g++ \
	libxrender1
	
# install python packages 
# Cython is prerequisite for python-spams 	
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ scikit-image \
	h5py \
	Cython \
	dmri-amico \
	dppd \
	cvxpy \
	vtk \
	dipy \
	python-spams

# TractSeg install
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ TractSeg==2.8 

# RUN download_all_pretrained_weights
RUN	mkdir /root/.tractseg && \
	wget -O /root/.tractseg/tractseg.zip http://api.open.brainlabel.org/data/ycf/tractseg.zip && \
	unzip -d /root/.tractseg /root/.tractseg/tractseg.zip && \
	rm /root/.tractseg/tractseg.zip
	

ADD ./ /
COPY /atlases /pipeline/atlases
# COPY ./TractSeg_scripts/img_utils.py /opt/conda/lib/python3.8/site-packages/tractseg/libs/   
# COPY ./TractSeg_scripts/tractseg_prob_tracking.py /opt/conda/lib/python3.8/site-packages/tractseg/libs/
RUN chmod +x /Roi3D/VtpRoi3DGenerator
CMD ["python3", "/run.py"]
