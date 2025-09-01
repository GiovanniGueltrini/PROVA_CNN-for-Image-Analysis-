import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import random
from torchvision import transforms
from .custom_model_Resnet import CustomResNet
from .segmentation import Dice_Sorensen
from .dummy_images_function import make_the_object
from .dummy_images_function import create_dummy_images
import matplotlib.pyplot as plt


def Dice_Sorensen(ImageT,ImageF):  
    intersection= np.sum(np.multiply(ImageT, ImageF))
    union=np.sum(ImageT)+np.sum(ImageF)
    return((2*intersection)/union)
    
def segmentation_prova(weights_path,image):
    """
    Performs a small segmentation experiment based on Grad-CAM. Given the model
    weights and a pair (input image, ground truth), it computes the Grad-CAM
    attention map, normalises it, and binarises it with a fixed threshold. It then
    compares the resulting mask with the ground truth by computing the
    Dice–Sørensen coefficient (DSC).
    
    Visual output
    -------------
    Displays a Matplotlib figure with two side-by-side images:
      • Left: binarised Grad-CAM mask (prediction).
      • Right: ground-truth mask.
    The figure title shows the DSC value.
    
    Parameters
    ----------
    weights_path : str
        Path to the `.pth` file containing the model's state_dict.
    image : tuple | list
        (input_image, ground_truth), both 2D NumPy arrays (H×W) of the same shape;
        `ground_truth` is a binary mask (0/1).
    
    Returns
    -------
    float
        The Dice–Sørensen coefficient (DSC).
    """
  
    ground_truth=image[1]
    result=run_gradcam_with(weights_path,image[0])
    
    interest_cam_image=result[0]
    #normalize pixel values 
    norm_interest_cam_image= (interest_cam_image - np.min(interest_cam_image)) / (np.max(interest_cam_image) - np.min(interest_cam_image))
    #define and apply the threshold
    threshold = 0.8
    matrice_binaria = np.where(norm_interest_cam_image>= threshold, 1,0)
    
    #compute the DSC between ground truth and the heat map 
    DSC=Dice_Sorensen(ground_truth, matrice_binaria)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    #Heat map
    axes[0].imshow(matrice_binaria, cmap='gray')
    axes[0].set_title('Immagine GradCam')
    axes[0].axis('off')
    
    #Ground Truth
    axes[1].imshow(ground_truth, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    fig.suptitle(f"Valore DSC: {DSC:.4f}", fontsize=12, y=0.1)
    plt.subplots_adjust(top=1, bottom=0.4)
    plt.tight_layout()
    plt.show()
    return (DSC)
