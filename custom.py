import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import imgaug
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = "/home/saraballkoci/Sara"
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + tooth

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    LEARNING_RATE=0.001

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):
       
   
    def load_custom(self, dataset_dir, subset):
        # Add the class
        self.add_class("object", 1, "tooth")

        # Make sure the subset is either 'train' or 'val'
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Iterate over all JSON files in the directory
        for filename in os.listdir(dataset_dir):
            if filename.endswith(".json"):
                json_file = os.path.join(dataset_dir, filename)
                # print(f"Processing {json_file}...")  # Print the current JSON file being processed

                annotations_json = json.load(open(json_file))
                shapes = annotations_json.get('shapes', [])

                print("shapes:", len(shapes))

                # If there are no annotations in the file, skip it
                if not shapes:
                    continue

                # Extract image filename and load the image
                image_filename = annotations_json['imagePath'].split('\\')[-1]
                image_path = os.path.join(dataset_dir, image_filename)
                # print(f"Loading image {image_path}...")  # Print the image file being loaded

                num_ids = [1] * len(shapes)
                
                print("numids",num_ids)
                
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                # Add the image to the dataset
                self.add_image(
                    "object",
                    image_id=image_filename,
                    path=image_path,
                    width=width, height=height,
                    polygons=[{
                        'name': 'polygon',
                        'all_points_x': [point[0] for point in shape['points']],
                        'all_points_y': [point[1] for point in shape['points']]
                    } for shape in shapes],
                    num_ids=num_ids
                )
  
    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not an object dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Initialize the mask array
        mask = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(image_info["polygons"]):
            # Clipping the coordinates to ensure they are within the image dimensions
            all_points_x = np.clip(p['all_points_x'], 0, image_info["width"] - 1)
            all_points_y = np.clip(p['all_points_y'], 0, image_info["height"] - 1)

            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom('/home/saraballkoci/Sara/dataset', 'train')
    dataset_train.prepare()
    
    dataset_val = CustomDataset()
    dataset_val.load_custom('/home/saraballkoci/Sara/dataset', 'val')
    dataset_val.prepare()
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
                # learning_rate=config.LEARNING_RATE,
                # epochs=250,
                # layers='heads')
                
    model.train(dataset_train,dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads', #layers='all', 
                augmentation = imgaug.augmenters.Sequential([ 
                imgaug.augmenters.Fliplr(1), 
                imgaug.augmenters.Flipud(1), 
                imgaug.augmenters.Affine(rotate=(-45, 45)), 
                imgaug.augmenters.Affine(rotate=(-90, 90)), 
                imgaug.augmenters.Affine(scale=(0.5, 1.5)),
                imgaug.augmenters.Crop(px=(0, 10)),
                imgaug.augmenters.Grayscale(alpha=(0.0, 1.0)),
                imgaug.augmenters.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                imgaug.augmenters.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                imgaug.augmenters.Invert(0.05, per_channel=True), # invert color channels
                imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                
                ]
                
                )) 
    #this augmentation is applied consecutively to each image. In other words, for each image, the #augmentation apply flip LR, and then followed by flip UD, then followed by rotation of -45 and 45, then followed by another rotation of -90 and 90, and lastly followed by scaling with factor 0.5 and 1.5. '''

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
				
				
config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH
        # Download weights file
if not os.path.exists(weights_path):
  utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

train(model)			