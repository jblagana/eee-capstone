#pip install moviepy

import os
from moviepy.video.io.VideoFileClip import VideoFileClip

def trim_video(input_path, output_path, duration):
    clip = VideoFileClip(input_path)
    trimmed_vid = clip.subclip(0, duration)
    trimmed_vid.write_videofile(output_path)
    clip.close()
    trimmed_vid.close()
    print("--------------------")

if __name__ == "__main__":
    input_folder = "trimming_video/data"           #Input folder of video files
    output_folder = "trimming_video/output"        #Folder where trimmed videos will be saved

    duration = 31                                  #Set duration of video to be clipped

    # Loop through vides files in the input folder
    for video_file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)
        trim_video(input_path, output_path, duration)