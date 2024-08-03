## Mask R-CNN for Object Detection and Segmentation on custom dataset

This is an implementation of Mask RCNN on our custom dataset which is the teeth of the excavator's sprocket.
The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

## Contents:

[custom.ipynb](https://github.com/SaraBallkoci/Mask-RCNN/custom.ipynb): Jupyter notebook with the custom implementation for the loading of the dataset, creation of bounding boxes and masks.

[custom.py](https://github.com/SaraBallkoci/Mask-RCNN/custom.py): Python script version of the custom implementation.

[dataset.zip](https://github.com/SaraBallkoci/Mask-RCNN/dataset.zip): Contains the dataset used for training and validation (290 annotated images with their json files fro traning, 74 annotated images with their json files for validation).

[mask_rcnn_object_0037.zip](https://github.com/SaraBallkoci/Mask-RCNN/mask_rcnn_object_0037.zip): Contains the weight file created by the training process, used for testing purposes.

[requirements.txt](https://github.com/SaraBallkoci/Mask-RCNN/requirements.txt): List of dependencies required to run the project.

[test_images.zip](https://github.com/SaraBallkoci/Mask-RCNN/test_images.zip): Contains test images used for validation.

[test_model.ipynb](https://github.com/SaraBallkoci/Mask-RCNN/test_model.ipynb): Jupyter notebook for testing and validating the model.

## Prerequisites:

Python 3.8.18

TensorFlow 2.4

## Installation:

Clone the repository:

-- git clone https://github.com/SaraBallkoci/Mask-RCNN.git

Install the required libraries:

-- pip install -r requirements.txt

Download the Mask R-CNN model and weights:

Download the mrcnn folder and pre-trained COCO weights (mask_rcnn_coco.h5) from Matterport's Mask R-CNN https://github.com/matterport/Mask_RCNN repository and place them in the project directory. 

Unzip the dataset.zip, test_images.zip and mask_rcnn_object_0037.zip and put them in your project's directory.

## How to use

### If you want to train the model again:
Open custom.ipynb in Jupyter Notebook and run the cells to perform tooth segmentation and train the model. The trining is done starting by the pre-trained COCO weights (mask_rcnn_coco.h5) from [release](https://github.com/matterport/Mask_RCNN/releases) page.

### If you want to see the results of the training:
Open test_model.ipynb in Jupyter Notebook and follow the instructions to test and validate the model using the provided test images. The model is tested with the trained file from [mask_rcnn_object_0037.zip](https://github.com/SaraBallkoci/Mask-RCNN/mask_rcnn_object_0037.zip). 


