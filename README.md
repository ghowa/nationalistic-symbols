# Contextualising Bandera: Deep Watching Demo Scripts

This repository combines demo scripts and pretrained models to recognise various nationalist(ic) symbols (using Detectron2) and politicians (using Facenet) from Eastern Europe.

## Setup

First you have to manually install Detectron2 and Facenet, see instructions [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) and [here](https://github.com/sepastian/facenet/blob/master/HOWTO.md). After installing Detectron2 and Facenet, you can use pip install -r requirements.txt to install the other needed packages. **Please note that Facenet needs Tensorflow, which as of yet is not running under Python 3.9. You can create a virtual environment using Python 3.8 with `python3.8 -m venv .`** 

## Usage

Symbol recognition in a video:

`python scripts/demo.py --config configs/nationalist_symbols.yaml --video-input video.mp4 --output output/ --frame-skip 10 --opts MODEL.DEVICE cpu`

Create 'output' folder first; annotated video and a json holding all recognized symbols will be stored here. Use --frame-skip to speed up the process (only each 
*n*th frame is used for recognition) and use --opts MODEL.DEVICE cpu if you don't have a Nvidia GPU.

Face recognition in images (stored in the 'images' folder):

`python src/classifier.py CLASSIFY images/ ../models/20180402-114759/20180402-114759.pb ../models/politicians.pkl --batch_size 1000`

Note: You have to download the generic pretrained model first (see install instructions).

## Acknowledgements

This project was funded by the [DI4DH initiative](https://www.uibk.ac.at/digital-humanities/ausschreibung-di4dh.html) at the University of Innsbruck, Austria.
