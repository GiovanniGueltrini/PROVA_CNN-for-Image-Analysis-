import numpy as np
import random
import matplotlib.pyplot as plt
import torch

def create_dummy_images(Length_Image, High_Image,Ray_circle, object_presence):
    """  
      Generate two images: one with random pixel values and one that is entirely black.      
       
      If the value of "object_presence" is 1, it is use the function "make_the_object" to create a with circle
      a with cyrcle inside.
      

    
    """
    # Create an image with random noise
    Image = np.random.uniform(0.5, 1, size=(Length_Image, High_Image))
    Clean_Image = np.zeros((Length_Image, High_Image))

    New_image=Image
    #Generate a random number to make the size of the object vary
    random_number = random.uniform(0.6, 1)
    Ray_circle=Ray_circle*random_number
    X_coordinate = np.random.randint(Ray_circle,Length_Image-Ray_circle ) 
    Y_coordinate = np.random.randint(Ray_circle,Length_Image-Ray_circle )
    Image_with_object=make_the_object(X_coordinate, Y_coordinate, Image, Ray_circle,1)
    Clean_Image_with_object=make_the_object(X_coordinate, Y_coordinate, Clean_Image, Ray_circle,0)
    New_image=Image_with_object        
    return New_image, Clean_Image_with_object


def make_the_object(X_coordinate, Y_coordinate, Image, ray_circle,noise):
    """ This function create a with circle in a image, the dimensione and the position of the circle is
       define by the input parameter 
       parameter: 
           Image: the initial image
           X_coordinat,y_coordinate: position of the object
           ray_circle: ray of the object
           noise: if noise is 1, the circle will be noisy,
                  if noise is 0, the circle will be noisy enterily witj  """
    New_image = Image
    (X_image, Y_image)= Image.shape
    if noise==0:
        for i in range(X_image):
            for j in range( Y_image):
                if np.square(i-X_coordinate)+np.square(j-Y_coordinate)<np.square(ray_circle):
                    New_image[i,j]=1
    else:
        for i in range(X_image):
            for j in range( Y_image):
                if np.square(i-X_coordinate)+np.square(j-Y_coordinate)<np.square(ray_circle):
                    #each pixel of the object area will become white with a certain probability
                    probability = random.uniform(0, 1)
                    if probability>0.30:
                       New_image[i,j]=1            
    return New_image
def generate_dataset(Dim_dataset=10,Length_Image=1024, High_Image=1024, Ray_circle=50, save_path="dataset_git.pth"):
    """
     Generates a dataset with dummy images and saves it to a .pth file.

    Args:
        Dim_dataset (int): total number of images to generate
        Ray_circle (int): radius of the circle to be drawn
        save_path (str): save path for the .pth file
    Returns:
        dict: dataset containing train/val/test split
    """

    images = []
    objects = []

    # Generazione immagini e label
    for i in range(Dim_dataset):
        if i < Dim_dataset / 2:
            im=create_dummy_images(Length_Image, High_Image, Ray_circle, 1)
            images.append(im[0])
            objects.append(1)
        else:
            im= create_dummy_images(Length_Image, High_Image, Ray_circle, 0)
            images.append(im[0])
            objects.append(0)

    # Conversione in numpy
    images = np.array(images)
    objects = np.array(objects)

    # Shuffle del dataset
    indices = np.random.permutation(len(images))
    shuffle_images = images[indices]
    shuffle_objects = objects[indices]

    # Split in train/val/test
    N = len(images)
    split_idx_1 = N * 7 // 10
    split_idx_2 = split_idx_1 + N * 15 // 100

    data = {
        "train_data": shuffle_images[:split_idx_1],
        "train_labels": shuffle_objects[:split_idx_1],
        "validation_data": shuffle_images[split_idx_1:split_idx_2],
        "validation_labels": shuffle_objects[split_idx_1:split_idx_2],
        "test_data": shuffle_images[split_idx_2:],
        "test_labels": shuffle_objects[split_idx_2:]
    }

    # Salvataggio dataset
    torch.save(data, save_path)
    
    return data
