# Photo_culling

DONE - Familiarize yourself with existing tools and approaches in this domain
           and suggest a suitable architecture for the photo selection tool.
DONE - Test existing approaches for evaluating image quality, image similarity and  image content.
DONE - Evaluate several methods for combining extracted image information for photo selection.
DONE - Implement an easy-to-use software tool for photo  selection, allowing the user to set basic parameters of the process.
TODO - Experimentally evaluate the performance of the developed software by comparing it with other existing tools,
          by comparing it with human performance, or by evaluating user satisfaction.
TODO - [Optional] Add the ability to read and write image metadata for easier integration with standard tools.
TODO - [Optional] Create a graphical user interface allowing to interactively examine and alter the selection process.
TODO - [Optional] Improve the selection process by learning from a (possibly large) dataset of manually selected photographs.




# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

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
