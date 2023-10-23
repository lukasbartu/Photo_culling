# Photo_culling

## Description
Program automatically creates a summary of image sequence. It considers technical and aesthetic quality of each image as well as content and structural similarity between images to avoid selecting duplicate or very similar images. 

## Usage
Software is usable in its raw console version (main.py) and we also developed graphical user interface for more intuitive usage of our software (gui_main.py).
At the moment software takes in parameters specified by user to create a tailored summary. It has two possible modes of selecting summary where one is based on final summary size where similar images are less likely to be included in the final output. The other selects summary of all images above certain image quality while avoiding any similar images.

## Roadmap
In future we will be working on creating software where user doesn't have to specify parameters of selection, but the parameters are pretrained with possibility of user influencing them by software usage. This approach should be easier to use for single-use users but more customizable for long-time users.

## Possible future features
Adding the ability to read and write image metadata for easier integration with standard tools.

## Authors and acknowledgment
Authored by Lukáš Bartůněk under supervision of Professor Jan Kybic
