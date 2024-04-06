import csv
import random
import os
import cv2

def create_csv(csv_filename, field_names):
    """
    Creates CSV file with its header fields

    Args:
        filename: csv file name
        field_names: column headers
    """

    with open(csv_filename, mode="w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()


def vid_processing(folder_path, csv_filename, field_names):
    """
    Opens and processes each video file in "folder_path" directory

    Args:
        folder_path: folder where the videos are located
        csv_filename: CSV file where outputs from detection modules will be stored
    """

    #Listing all the files in the folder_path directory
    video_files = os.listdir(folder_path)

    #Opening the CSV file
    with open(csv_filename, mode="a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)

        #Process each video file
        for video_id,video_file in enumerate(video_files, start=1):
            video_path = os.path.join(folder_path, video_file)

            #Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open {video_file}")
                continue
            
            #Processing each frame
            frame_num = 0    
            while True:
                ret, frame = cap.read()

                if not ret:
                    break
                
                frame_num += 1

                #Insert module outputs here
                crowd_density = random.randint(0, 100)
                crowd_count = random.randint(0, 100)
                loitering = random.randint(0, 100)
                low_conc = random.randint(0, 100)
                med_conc = random.randint(0, 100)
                high_conc = random.randint(0, 100)
                # concealment = [random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)]
                rbp = random.randint(0, 1)

                #Appending data to the CSV file
                writer.writerow({
                    "video_id": video_file,
                    "frame_num": frame_num,
                    "crowd_density": crowd_density,
                    "crowd_count": crowd_count,
                    "loitering": loitering,
                    "low_concealment": low_conc,
                    "med_concealment": med_conc,
                    "high_concealment": high_conc,
                    "rbp": rbp
                })

            cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    #Declaring CSV filename and header fields
    csv_filename = "data.csv"
    field_names = ["video_id","frame_num","crowd_density","crowd_count","loitering","low_concealment","med_concealment","high_concealment","rbp"]

    #Declaring folder path of the videos to be processed
    folder_path = r"C:\Users\janrh\OneDrive - University of the Philippines\Acads\4TH YEAR (23-24)\2ND SEM\EE 199\DATASETS\anomaly_trimmed"

    create_csv(csv_filename, field_names)
    vid_processing(folder_path, csv_filename, field_names)


