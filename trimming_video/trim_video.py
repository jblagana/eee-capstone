#pip install moviepy

import os
from moviepy.video.io.VideoFileClip import VideoFileClip

def trim_video(input_path, output_path, duration):
    vid = VideoFileClip(input_path)
    vid_duration = vid.duration
    
    if vid_duration <= duration:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(input_path, 'rb') as fsrc, open(output_path, 'wb') as fdst:
            fdst.write(fsrc.read())
        print("Vid duration ({}s) <= {}s. Copied file successfully.".format(vid_duration, duration))
    else: 
        trimmed_vid = vid.subclip(0, duration)
        trimmed_vid.write_videofile(output_path)
        vid.close()
        trimmed_vid.close()

    vid.close()
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