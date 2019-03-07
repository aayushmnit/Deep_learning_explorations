# USAGE
"""
python find_similar_image.py \
--input_path "../data/caltech101/" \
--img_url "https://cdn.pixabay.com/photo/2016/06/08/00/03/pizza-1442946_1280.jpg" \
--output_path "./output/output.png" \
--n_items 6 \
--show_image 
"""

import click
import requests
from io import BytesIO
from pathlib import Path
import pickle
from PIL import Image as pil_img
import numpy as np
from fastai.vision.data import ImageDataBunch
from fastai.vision.transform import get_transforms
from fastai.vision.learner import create_cnn
from fastai.vision import models
from fastai.vision.image import pil2tensor, Image
import matplotlib
import matplotlib.pyplot as plt
import cv2
from imutils import resize

@click.command()
@click.option(
    "--input_path",
    "-ip",
    default="./",
    required=True,
    help="Path for images, models and LSH table",
)
@click.option(
    "--img_url",
    "-url",
    default="http://something.com",
    required=True,
    help="URL to get the image",
)
@click.option(
    "--output_path", "-op", default="./", help="Path to the output similar images"
)
@click.option(
    "--show_image", "-s", is_flag=True, help="Display image in a openCV window"
)
@click.option(
    "--n_items",
    "-n",
    type=click.IntRange(min=1, max=20),
    default=5,
    help="Number of similar images b/w 1 to 20",
)
def find_similar_images(input_path, img_url, output_path, show_image, n_items):
    '''
    Function to run everything together
    '''
    resp, url_img = download_img_from_url(img_url)
    print("Image Download successful:{0}".format(resp))
    if resp:
        print("Load databunch")
        data_bunch = load_image_databunch(input_path, classes)

        print("Create a model")
        learner = load_model(data_bunch, models.resnet34, "stg2-rn34")

        print("Add a Hook")
        sf = SaveFeatures(learner.model[1][5])

        print("Load LSH table")
        lsh = pickle.load(open(Path(input_path) / "lsh.p", "rb"))

        print("Return similar items")
        get_similar_images(
            url_img, learner, sf, lsh, show_image, output_path, n_items=n_items
        )

    else:
        print(
            "Image cannot be downloaded from URL please check the url link and try again."
        )


def load_image_databunch(input_path, classes):
    """
    Code to define a databunch compatible with model
    """
    tfms = get_transforms(
        do_flip=False,
        flip_vert=False,
        max_rotate=0,
        max_lighting=0,
        max_zoom=1,
        max_warp=0,
    )

    data_bunch = ImageDataBunch.single_from_classes(
        Path(input_path), classes, ds_tfms=tfms, size=224
    )

    return data_bunch


def load_model(data_bunch, model_type, model_name):
    """
    Function to create and load pretrained weights of convolutional learner
    """
    learn = create_cnn(data_bunch, model_type, pretrained=False)
    learn.load(model_name)
    return learn


def download_img_from_url(url):
    '''
    Function to download image given a valid url
    '''
    try:
        response = requests.get(url)
        img = pil_img.open(BytesIO(response.content))
        resp = True
    except:
        resp = False
        img = np.nan
    return resp, img


class SaveFeatures:
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))

    def remove(self):
        self.hook.remove()


def image_to_vec(url_img, hook, learner):
    '''
    Function to convert image to vector
    '''
    print("Convert image to vec")
    _ = learner.predict(Image(pil2tensor(url_img, np.float32).div_(255)))
    vect = hook.features[-1]
    return vect


def get_similar_images(
    url_img, conv_learn, hook, lsh, show_image, output_path, n_items=5
):
    ## Converting Image to vector
    vect = image_to_vec(url_img, hook, conv_learn)

    ## Finding approximate nearest neighbours using LSH
    response = lsh.query(vect, num_results=n_items + 1, distance_func="hamming")

    ## Dimension calculation for plotting
    columns = 3
    rows = int(np.ceil(n_items + 1 / columns)) + 1

    ## Plotting function
    fig = plt.figure(figsize=(2 * rows, 3 * rows))
    for i in range(1, columns * rows + 2):
        ## Plotting the url_img in center of first row
        if i == 1:
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(url_img)
            plt.axis("off")
            plt.title("Input Image")
        ## Plotting similar images row 2 onwards
        elif i < n_items + 2:
            ret_img = pil_img.open(response[i - 1][0][1])
            fig.add_subplot(rows, columns, i + 2)
            plt.imshow(ret_img)
            plt.axis("off")
            plt.title(str(i - 1))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)

    ## Display if show_image is mentioned in argument
    if show_image:
        img = cv2.imread(output_path, 1)
        img = resize(img, width=600)
        cv2.imshow("Similar images output", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


classes = [
    "BACKGROUND_Google",
    "Faces",
    "Faces_easy",
    "Leopards",
    "Motorbikes",
    "accordion",
    "airplanes",
    "anchor",
    "ant",
    "barrel",
    "bass",
    "beaver",
    "binocular",
    "bonsai",
    "brain",
    "brontosaurus",
    "buddha",
    "butterfly",
    "camera",
    "cannon",
    "car_side",
    "ceiling_fan",
    "cellphone",
    "chair",
    "chandelier",
    "cougar_body",
    "cougar_face",
    "crab",
    "crayfish",
    "crocodile",
    "crocodile_head",
    "cup",
    "dalmatian",
    "dollar_bill",
    "dolphin",
    "dragonfly",
    "electric_guitar",
    "elephant",
    "emu",
    "euphonium",
    "ewer",
    "ferry",
    "flamingo",
    "flamingo_head",
    "garfield",
    "gerenuk",
    "gramophone",
    "grand_piano",
    "hawksbill",
    "headphone",
    "hedgehog",
    "helicopter",
    "ibis",
    "inline_skate",
    "joshua_tree",
    "kangaroo",
    "ketch",
    "lamp",
    "laptop",
    "llama",
    "lobster",
    "lotus",
    "mandolin",
    "mayfly",
    "menorah",
    "metronome",
    "minaret",
    "nautilus",
    "octopus",
    "okapi",
    "pagoda",
    "panda",
    "pigeon",
    "pizza",
    "platypus",
    "pyramid",
    "revolver",
    "rhino",
    "rooster",
    "saxophone",
    "schooner",
    "scissors",
    "scorpion",
    "sea_horse",
    "snoopy",
    "soccer_ball",
    "stapler",
    "starfish",
    "stegosaurus",
    "stop_sign",
    "strawberry",
    "sunflower",
    "tick",
    "trilobite",
    "umbrella",
    "watch",
    "water_lilly",
    "wheelchair",
    "wild_cat",
    "windsor_chair",
    "wrench",
    "yin_yang",
]

if __name__ == "__main__":
    find_similar_images()

