# Contextualising Bandera: Deep Watching Demo Scripts

This repository combines demo scripts and pretrained models to recognise various nationalist(ic) symbols (using Detectron2) and politicians (using Facenet) from Eastern Europe.

## Setup

First you have to manually install Detectron2 and Facenet, see instructions [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) and [here](https://github.com/sepastian/facenet/blob/master/HOWTO.md). After installing Detectron2 and Facenet, you can use pip install -r requirements.txt to install the other needed packages.

## Usage

Symbol recognition in a video:

`python scripts/demo.py --config configs/nationalist_symbols.yaml --video-input video.mp4 --output output/ --frame-skip 10 --opts MODEL.DEVICE cpu`

Create 'output' folder first; annotated video and a json holding all recognized symbols will be stored here. Use --frame-skip to speed up the process (only each 
*n*th frame is used for recognition) and use --opts MODEL.DEVICE cpu if you don't have a Nvidia GPU.

Face recognition in images (stored in the 'images' folder):

`python src/classifier.py CLASSIFY images/ ../models/20180402-114759/20180402-114759.pb ../models/politicians.pkl --batch_size 1000`

Note: You have to download the generic pretrained model first (see install instructions). In order to pull our own models from the repository, you need Git LFS.

## Model Performance

Evaluation results for segm: 
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|--------|--------|--------|-------|--------|--------|
| 58.611 | 75.794 | 64.684 |  nan  | 10.000 | 62.292 |

Per-category segm AP: 
| category              | AP     | category             | AP     | category             | AP     |
|-----------------------|--------|----------------------|--------|----------------------|--------
| bandera 1             | 52.962 | bandera 2            | 75.438 | bandera 3            | 83.650 |
| crest_bg              | 83.072 | crest_by             | 84.087 | crest_by_opp         | 71.723 |
| crest_me              | 72.738 | crest_pl             | 88.285 | crest_rs             | 76.234 |
| crest_ua              | 41.392 | cross                | 35.618 | cross_orthodox       | 38.685 |
| cross_serbian         | 70.247 | eu                   | 65.126 | falanga              | 49.673 |
| flag_bg_hanging       | 58.628 | flag_bg_waving       | 68.266 | flag_by_hanging      | 75.524 |
| flag_by_waving        | 80.001 | flag_by_opp_hanging  | 30.986 | flag_by_opp_waving   | 37.250 |
| flag_me_hanging       | 74.230 | flag_me_waving       | 94.675 | flag_rs_hanging      | 69.894 |
| flag_rs_waving        | 69.869 | flag_ru_hanging      | 40.152 | flag_ru_waving       | 64.079 |
| flag_soc_hanging      | 73.437 | flag_soc_waving      | 84.485 | flag_ua_hanging      | 43.399 |
| flag_ua_waving        | 45.136 | flag_upa_hanging     | 11.081 | flag_upa_waving      | 35.334 |
| george_ribbon_hanging | 38.060 | george_ribbon_waving | 56.608 | george_ribbon_folded | 40.598 |
| hammer_sickle         | 31.685 | nato                 | 85.251 | oun                  | 56.942 |
| ss_rune               | 27.394 | swastika             | 33.306 | swoboda              | 60.679 |
| three_fingers         | 45.443 | wolfsangel           | 57.564 |                      |    

## Acknowledgements

Initial training annotations were funded by the German [BMBF](https://www.bmbf.de) at the University of Passau, Germany. Further annotations were funded by the [DI4DH initiative](https://www.uibk.ac.at/digital-humanities/ausschreibung-di4dh.html) at the University of Innsbruck, Austria. Training was conducted using the infrastructure of the [Research Center High Performance Computing](https://www.uibk.ac.at/fz-hpc/) in Innsbruck.
