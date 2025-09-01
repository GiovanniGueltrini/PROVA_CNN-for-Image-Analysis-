import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from torchvision import transforms
from .custom_model_Resnet import CustomResNet  # Importa il tuo modello personalizzato
from .dummy_images_function import make_the_object
from .dummy_images_function import create_dummy_images



def run_gradcam_with(weights_path):
    """
    Apply Grad-CAM to a  model loaded from weights file and
    display the heatmap overlaid on a synthetic test image.
    
    Parameters
    - weights_path (str): path to the `.pth` file containing the `state_dict`.
        - image (np.ndarray): synthetic 2D image (H×W) to be used as model input.
        
    Returns
    - result: raw Grad-CAM map (array with shape [1, H, W] as per `GradCAM` output).
   
    """
  
    #define model and load_weight
    model = CustomResNet(n_classes=2,dropout_percentage=0)
    model.load_state_dict(torch.load(weights_path))
    
    #select the target layer to be evaluated
    target_layers = [model.conv7_2_1]  # Assicurati che questa sia la layer corretta
    
    # create a test image
    image = create_dummy_images(1024, 1024, 50, 1)  
    input_image_1 = torch.from_numpy(image)
    input_image = input_image_1.unsqueeze(0).unsqueeze(0)  
    input_tensor = input_image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).float()
    
    # Initialize Grad-CAM function 
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # select the label 
    target_category = 1  
    #Create a target for the class 
    target = ClassifierOutputTarget(target_category)
    
    # apply the model
    output = model(input_tensor)
    
    # Perform back-propagation
    model.zero_grad()  
    output[:, target_category].backward(retain_graph=True) 
    
    
    
    #apply Grad cam
    result = cam(input_tensor=input_tensor, targets=[target])
    image_3d = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1) 
    # Visualize the result
    cam_image = show_cam_on_image(np.array(image_3d) , result[0], use_rgb=False)
    
    import matplotlib.pyplot as plt
    plt.imshow(cam_image)
    plt.axis('off')
    plt.show()
