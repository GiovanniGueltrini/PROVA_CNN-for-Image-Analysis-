import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
def make_the_object(X_coordinate, Y_coordinate, Image, ray_circle):
    """ This function create a with circle in a image, the dimensione and the position of the circle is
       define by the input parameter """
    New_image = Image
    (X_image, Y_image)= Image.shape
    for i in range(X_image):
        for j in range( Y_image):
            if np.square(i-X_coordinate)+np.square(j-Y_coordinate)<np.square(ray_circle):
                New_image[i,j]=1
    return New_image


def create_dummy_images(Length_Image, High_Image,Ray_circle, object_presence):
  """  
  create an image with pixel values randomly distributed.
  
  If the value of "object_presence" is 1, it is use the function "make_the_object" to create a with circle
  a with cyrcle inside.
  
  """
    Image=np.random.rand(Length_Image, High_Image)
    New_image=Image
    
    if object_presence==1:
        X_coordinate = np.random.randint(Ray_circle,Length_Image-Ray_circle ) 
        Y_coordinate = np.random.randint(Ray_circle,Length_Image-Ray_circle )
        Image_with_object=make_the_object(X_coordinate, Y_coordinate, Image, Ray_circle)
        New_image=Image_with_object
        
    return New_image



images = []
objects = []
#define the dimensione of the dataset and generation of the image with the label
Dim_dataset=301
for i in range(150):
    if i<Dim_dataset/2:      
        images.append(create_dummy_images(256,256, 40,1))
        objects.append(1)
    else:
        images.append(create_dummy_images(256,256, 40,0))
        objects.append(0)
#costruction of the dataset with the image and the labels
data_set=list(zip(images,objects))


# define the propotrion between training and test set
Dim_training=int(Dim_dataset*0.7)
Dim_test=int(Dim_dataset*0.3)
images = np.array(images)  
objects = np.array(objects)
indices = np.random.permutation(len(images))

# reorder the dataset randomly
shuffle_images = images[indices]
shuffle_objects = objects[indices]
N = len(images)
split_idx = N * 8 // 10  # Divisione intera per ottenere un intero

# create the final dataset 
data = {
    "train_data": shuffle_images[:split_idx],  # Immagini di training
    "train_labels": shuffle_objects[:split_idx],  # Etichette di training
    "test_data": shuffle_images[split_idx:],  # Immagini di test
    "test_labels": shuffle_objects[split_idx:]  # Etichette di test
}

# Salvare il dataset su file
torch.save(data, "dataset_cnn.pth")

print("Dataset salvato correttamente!")
