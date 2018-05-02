# Machine Learning Engineer Nanodegree 

## Steganalysis of LSB Matching in Greyscale Images - Capstone Project

### Software Requirements

This project uses both Python 2 and Python 3. Python 2 is used in the 'Image Preprocessing.ipynb' notebook. All other python 
files use Python 3. I have used the Anaconda distribution of python for development. It includes libraries numpy, scipy, 
pandas, sklearn and matplotlib used in this project. The additional libraries used are seaborn, pillow and pywt. All of the
mentioned libraries can be installed using *'pip install library_name'* or *'conda install library_name'*.

### Dataset Details

I have used the BOSSbase dataset as my base dataset. It is available for download at:

http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip

I have included samples for both the original images in the BOSSbase dataset as well as the images after performing LSB 
matching. The steg images have been generated using the tool available at https://github.com/daniellerch/aletheia. The command 
used to generate the steg images is:

$ python aletheia.py lsbm-sim bossbase 0.40 bossbase_lsb
