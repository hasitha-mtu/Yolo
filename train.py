import numpy as np
import pandas as pd
from ultralytics import YOLO
from roboflow import Roboflow
import os
import matplotlib.pyplot as plt
import cv2

def train():
    rf = Roboflow(api_key="oZuzlKNoohLU4r69rVA0")
    project = rf.workspace("research-fkjgx").project("water-segmenting-v1")
    version = project.version(1)
    dataset = version.download("yolov11")
    print(f'dataset : {dataset}')
    model = YOLO("yolo11n-seg.pt")
    train_results = model.train(
        data="/kaggle/working/Water-Segmenting-v1-1/data.yaml",
        epochs=10,
        imgsz=640,
        device=0,
    )
    print(f'train_results : {train_results}')
    model.save("best_water_segmentation_model.pt")
    model = YOLO("best_water_segmentation_model.pt")
    test_folder = "/kaggle/working/Water-Segmenting-v1-1/test"  # replace with your test folder path

    # List the contents of the test folder
    files = os.listdir(test_folder)
    print(f"Files in the test folder: {files}")

    # Perform inference on all images in the 'images' subfolder
    results = model.predict(source="/kaggle/working/Water-Segmenting-v1-1/test/images",  # direct path to images folder
                            save=True,  # save the output images with segmentation masks
                            conf=0.25,  # confidence threshold
                            device=0)  # specify device (0 for GPU, -1 for CPU)
    # Directory where the segmented images are saved
    result_dir = "runs/segment/predict"

    # List all image files in the directory
    result_images = os.listdir(result_dir)

    # Filter only valid image files (e.g., .jpg, .png)
    image_extensions = ['.jpg', '.jpeg', '.png']
    result_images = [img for img in result_images if os.path.splitext(img)[1].lower() in image_extensions]

    # Define number of images and grid size
    num_images = len(result_images)
    cols = 3  # Number of columns for the grid
    rows = (num_images // cols) + (num_images % cols > 0)  # Calculate rows needed

    # Display all images
    plt.figure(figsize=(15, rows * 5))  # Increase height based on number of rows
    for i, img_file in enumerate(result_images):
        img_path = os.path.join(result_dir, img_file)

        # Load and process the image
        segmentation_result = cv2.imread(img_path)
        segmentation_result = cv2.cvtColor(segmentation_result,
                                           cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color display

        # Plot each image in a grid
        plt.subplot(rows, cols, i + 1)  # Adjust grid based on rows and columns
        plt.imshow(segmentation_result)
        plt.axis('off')  # Turn off axis

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    train()