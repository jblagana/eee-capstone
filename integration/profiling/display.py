# Displaying fps
import matplotlib.pyplot as plt
import csv

parsed_data = []
csv_file = "integration/profiling/fps_log.csv"
with open(csv_file, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        filename, frame_num, fps = row
        frame_num = int(frame_num)
        fps = float(fps)
        parsed_data.append((filename, frame_num, fps))

# Plot FPS data
plt.figure()
for filename, frame_num, fps in parsed_data:
    plt.scatter(frame_num, fps, label=filename, marker='.')

plt.xlabel('Frame Number')
plt.ylabel('FPS')
plt.title('FPS of Video Files')
# plt.legend()
plt.show()


# Displaying block statistics
import os
import subprocess

def visualize_profiles(profiling_folder):
    profiles = os.listdir(profiling_folder)
    for profile in profiles:
        if profile.endswith(".prof"):
            subprocess.Popen(["snakeviz", os.path.join(profiling_folder, profile)])

if __name__ == "__main__":
    profiling_folder = "integration/profiling"
    visualize_profiles(profiling_folder)



