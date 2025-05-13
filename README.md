# Semantic Segmentation on IDD Lite Dataset

This project focuses on semantic segmentation of road scenes in unstructured environments using the U-Net architecture. Unlike conventional datasets designed for structured, rule-abiding roads, this project leverages a dataset that captures the complexity and diversity of Indian roads.

## Dataset

While many recent datasets for autonomous navigation focus on structured environments (e.g., well-marked lanes, low category variation, adherence to traffic rules), this dataset breaks away from those assumptions.

- **Original Dataset:**  
  - 10,000 images from 182 drive sequences  
  - Captured on roads in Hyderabad, Bangalore, and surrounding areas  
  - High-resolution images (mostly 1080p, with some 720p)  
  - Finely annotated with **34 semantic classes**, including many not found in standard datasets like Cityscapes

- **IDD-Lite Subset (Used in this Project):**  
  - A smaller, more manageable version of the original dataset  
  - ~50MB in size, suitable for local machines  
  - Contains **7 core classes**  


## Model Architecture

This project uses the U-Net model, a convolutional neural network architecture originally designed for biomedical image segmentation. U-Net is well-suited for pixel-wise classification tasks and performs efficiently even on relatively small datasets like IDD-Lite.

### Architecture Details

U-Net follows an **encoder–bottleneck–decoder** architecture:

- **Encoder (Downsampling Path):**  
  Sequential Conv2D layers with increasing filters:  
  64 → 128 → 256 → 512 → 1024  
  Each block consists of:
  - Conv2D  
  - Batch Normalization  
  - ReLU activation  
  - MaxPooling  

- **Bottleneck:**  
  The deepest part of the network that captures the most abstract and compressed feature representation of the input image.

- **Decoder (Upsampling Path):**  
  Each stage includes:
  - Transposed convolution (Conv2DTranspose) for upsampling  
  - Skip connections: Concatenation with the corresponding encoder feature map  
  - Conv2D + BatchNorm + ReLU to refine features  

- **Output Layer:**  
  A Conv2D(num_classes, kernel_size=1, activation='softmax') layer outputs per-pixel class predictions for semantic segmentation.


## Preprocessing

Preprocessing includes the following steps:

- **Resizing:**  
  All input images and segmentation masks are resized to a fixed shape (typically a power of 2) to maintain consistent spatial dimensions during the U-Net’s downsampling and upsampling operations.

- **Normalization:**  
  Pixel values of input images are scaled to the range [0, 1] for faster and more stable training.

- **Encoding Labels:**  
  Ground truth segmentation masks are stored as class indices ranging from 0 to num_classes - 1. These are used directly with loss functions like SparseCategoricalCrossentropy.

- **Augmentation:**  
  - Random horizontal flipping  

These augmentations are applied dynamically in the data pipeline to help the model generalize better. However, over-augmentation or inappropriate transformations may degrade performance, so they are used cautiously.


## Performance Metrics

The model was evaluated using standard semantic segmentation metrics:

- **Mean Intersection over Union (mIoU):**  
  Measures the overlap between predicted and ground truth segmentation masks, averaged across all classes.  
  > **mIoU achieved:** *0.467*

- **Accuracy:**  
  Measures the proportion of correctly classified pixels over the entire dataset.  
  > **Accuracy achieved:** *83.64593*

## Citation

- **Dataset:**

  [https://idd.insaan.iiit.ac.in](https://idd.insaan.iiit.ac.in)

