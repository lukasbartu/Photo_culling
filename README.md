# Photo_culling

## Description
Program automatically creates a summary of image sequence. It considers technical and aesthetic quality of each image as well as content and structural similarity between images to avoid selecting duplicate or very similar images. 

## Usage
Software is usable in its raw console version (console_main.py) and we also developed graphical user interface for more intuitive usage of our software (gui_main.py). Graphical version also has more features that are harder to implement in console version and thus its use is the recommended 
This project is fully described in my Bachelor's thesis (will add link when it is finished) which also includes manual.

## Authors
Authored by Lukáš Bartůněk under supervision of Professor Jan Kybic

## Acknowledgment
SIFT similarity is edited code taken from https://github.com/adumrewal/SIFTImageSimilarity/blob/master/SIFTSimilarityInteractive.ipynb.

NIMA image quality is edited code taken from https://github.com/yunxiaoshi/Neural-IMage-Assessment.

Technical quality is implemented using library brisque (https://pypi.org/project/brisque/)

Graphical user interface is created using demos and examples from library PySimpleGUI (https://www.pysimplegui.org/en/latest/)


