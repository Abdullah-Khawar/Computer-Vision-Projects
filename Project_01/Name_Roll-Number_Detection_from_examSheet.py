import os
import numpy as np
import csv
import easyocr
import cv2

# Initialize EasyOCR Reader
reader = easyocr.Reader(["en"], gpu=True)

# Define the path for the dataset and output directories
dataset_dir = "../top_halves"
output_dir = "results"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the CSV file for writing
with open("boxes.csv", mode="w", newline="") as file:
    writer = csv.writer(file, delimiter=" ")

    for i in range(1, 102):
        img_path = os.path.join(dataset_dir, f"image{i}.jpg")
        img = cv2.imread(img_path)
        text_ = reader.readtext(img)

        # Initialize the coordinates for both boxes (set to 0 if not detected)
        roll_number_coords = [0, 0, 0, 0]
        name_coords = [0, 0, 0, 0]

        for t_ in text_:
            bbox, text, score = t_
            if "2023" in text:
                # Detected First Name box
                top_left = np.array(bbox[0])
                top_right = np.array(bbox[1])
                bottom_right = np.array(bbox[2])
                bottom_left = np.array(bbox[3])

                new_top_left_2 = np.array([top_right[0] + 15, top_right[1] + 109])
                new_bottom_left_2 = np.array(
                    [bottom_right[0] + 37, bottom_right[1] + 137]
                )

                new_top_left_2_shifted = new_top_left_2 + np.array([270, 0])
                new_bottom_left_2_shifted = new_bottom_left_2 + np.array([270, 0])

                new_top_right_2 = new_top_left_2_shifted + np.array(
                    [new_bottom_left_2_shifted[0] - new_top_left_2_shifted[0], 0]
                )
                new_bottom_right_2 = new_bottom_left_2_shifted + np.array(
                    [new_top_right_2[0] - new_top_left_2_shifted[0], 0]
                )

                pt1_new = (int(new_top_left_2[0]), int(new_top_left_2_shifted[1]))
                pt2_new = (int(new_bottom_right_2[0]), int(new_bottom_right_2[1]))
                cv2.rectangle(img, pt1_new, pt2_new, (0, 255, 0), 2)

                name_coords = [
                    int(new_top_left_2_shifted[1]),  # top-left row coordinate
                    int(new_top_left_2[0]),  # top-left column coordinate
                    int(new_bottom_right_2[1]),  # bottom-right row coordinate
                    int(new_bottom_right_2[0]),  # bottom-right column coordinate
                ]

                # Detected Second Roll no box
                new_top_left_2 = np.array([top_right[0] - 388, top_right[1] + 115])
                new_bottom_left_2 = np.array(
                    [bottom_right[0] - 368, bottom_right[1] + 152]
                )

                new_top_left_2_shifted = new_top_left_2 + np.array([295, 0])
                new_bottom_left_2_shifted = new_bottom_left_2 + np.array([270, 0])

                new_top_right_2 = new_top_left_2_shifted + np.array(
                    [new_bottom_left_2_shifted[0] - new_top_left_2_shifted[0], 0]
                )
                new_bottom_right_2 = new_bottom_left_2_shifted + np.array(
                    [new_top_right_2[0] - new_top_left_2_shifted[0], 0]
                )

                pt1_new = (int(new_top_left_2[0]), int(new_top_left_2_shifted[1]))
                pt2_new = (int(new_bottom_right_2[0]), int(new_bottom_right_2[1]))

                roll_number_coords = [
                    int(new_top_left_2_shifted[1]),  # top-left row coordinate
                    int(new_top_left_2[0]),  # top-left column coordinate
                    int(new_bottom_right_2[1]),  # bottom-right row coordinate
                    int(new_bottom_right_2[0]),  # bottom-right column coordinate
                ]

                cv2.rectangle(img, pt1_new, pt2_new, (0, 255, 0), 2)

        # Save the image in the output folder
        output_img_path = os.path.join(output_dir, f"image{i}.jpg")
        cv2.imwrite(output_img_path, img)

        # Save all 8 coordinates (roll_number_coords and name_coords) to CSV
        writer.writerow([f"image{i}.jpg"] + roll_number_coords + name_coords)
