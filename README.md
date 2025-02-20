# ArtFusion: Neural Style Synthesis with TensorFlow 2


## Overview

ArtFusion is a neural style transfer project built using TensorFlow 2 and Keras. This project implements a modular pipeline for transferring the artistic style from one image onto the content of another. The pipeline supports both VGG19 and InceptionV3-based feature extractors, allowing you to experiment with multiple architectures and adjust various parameters like content and style loss weighting. The optimization is performed using the L-BFGS algorithm from SciPy, and the code is organized into reusable functions for easy experimentation and integration.

## Features

 * Modular design with separate functions for preprocessing, loss calculation, gradient computation, and image deprocessing.

 * Switchable backbone networks for style extraction (VGG19 and InceptionV3).

 * Utilizes TensorFlow 2’s eager execution and Keras backend for clarity and ease of modification.

 * Optimization via the SciPy L-BFGS algorithm for high-quality style synthesis.

 * Simple examples provided for applying style transfer on multiple image pairs.


## Usage

The sorting of images and the execution of style transfer is handled using function calls in the file (e.g., run_style_transfer). The pipeline consists of these steps:

1. Preprocess Image
 - Preprocess content and style images using the appropriate routines for VGG19 or InceptionV3.
 - A function (preprocess_image_instantiator) loads an image, resizes it maintaining its aspect ratio, converts it to a NumPy array, expands dimensions, and applies the model-specific preprocessing.

2. Build the Model
 - Build a composite input tensor containing the content image, the style reference image, and a placeholder (or variable) for the combination image.
 - Instantiate the VGG19 (or InceptionV3) network using pretrained ImageNet weights (pre-downloaded if running locally or available in your environment).

3. Define Loss Functions
 - The content loss is computed as the mean squared error between the feature maps of the content layer (e.g., “block5_conv2”) for the content image and the combination image.
 - The style loss is computed based on the Gram matrix computed from activations of multiple style layers (e.g., “block1_conv1”, “block2_conv1”, etc.) comparing style image and combination image.

4. Optimize
 - Use the SciPy L-BFGS algorithm to minimize the total loss (weighted sum of content and style losses) with respect to the pixels of the combination image.
 - The Evaluator class wraps loss evaluation and gradient computation to interface with the optimizer.

5. Deprocess and Save
 - Convert the optimized image back into a displayable format by undoing the preprocessing steps (e.g., adding back mean values and converting from BGR to RGB for VGG19).
 - Save and/or display the final generated image.
