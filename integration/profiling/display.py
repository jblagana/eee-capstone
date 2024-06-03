# Displaying fps
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8
})

import os
import subprocess

import csv
import pandas as pd

import os
import psutil

def display_fps():
    # Iterate through files in the directory
    for file_name in os.listdir(profiling_folder):
        if "fps_log" in file_name and file_name.endswith('.csv'):
            title = os.path.splitext(file_name)[0]

            # Parse and plot data
            parsed_data = []
            csv_file = os.path.join(profiling_folder, file_name)
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
                plt.scatter(frame_num, fps, label=filename, marker='.', s=7)

            plt.xlabel('Frame Number')
            plt.ylabel('FPS')
            plt.title(title)  # Set title with file name
            plt.savefig(f'integration/profiling/{title}.png')
            # plt.show()
            plt.close()


def display_statistics():
    # Displaying block statistics
    try:
        subprocess.Popen(["snakeviz", os.path.join(profiling_folder, "profiling_total.prof")])
    except Exception:
        pass


def display_resource_jetson():
    # Display resource consumption
    try:
        time = []
        cpu_usage = []
        gpu_usage = []
        used_RAM = []
        tot_RAM = []

        # Open the CSV file
        csv_file_path = os.path.join(profiling_folder, "resource_log-Jetson.csv")
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Skip the header row
            for i, row in enumerate(csv_reader):
                time.append(i) # index is in unit seconds
                cpu_usage.append(sum(map(float, row[2:6])) / 4)
                gpu_usage.append(float(row[6]))
                used_RAM.append(float(row[7]))
                tot_RAM.append(float(row[8]))
                

        # Create a 1x3 subplot
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))

        # Plot CPU usage
        axs[0].plot(time, cpu_usage, label="CPU")
        axs[0].set_title("CPU Usage")
        axs[0].set_ylabel("Usage (%)")
        axs[0].legend()

        # Plot GPU usage
        axs[1].plot(time, gpu_usage, label="GPU")
        axs[1].set_title("GPU Usage")
        axs[1].set_ylabel("Usage (%)")
        axs[1].legend()

        # Plot memory usage
        axs[2].plot(time, used_RAM, label="Memory")
        axs[2].set_title(f"Memory Usage\n{tot_RAM[0]}")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Usage (kB)")
        axs[2].legend()

        # Plot power usage
        # axs[1, 1].plot(time, power_usage, label="Power")
        # axs[1, 1].set_title("Power Usage")
        # axs[1, 1].set_xlabel("Time (s)")
        # axs[1, 1].set_ylabel("Usage (Watts)")
        # axs[1, 1].legend()

        # Adjust layout
        plt.tight_layout()
        plt.savefig('trt_integration/profiling/resource_jetson.png')
        # plt.show()

    except:
        pass

def display_resource():
    # Display resource consumption
    try:
        df = pd.read_csv(os.path.join(profiling_folder, "resource_nonJetson.csv"))

        # Extract relevant columns
        time = df.index
        cpu_usage = df[["CPU"]]
        cpu_usage = cpu_usage.mean(axis=1) # Average cpu usage of all cpu cores
        gpu_usage = df["GPU"]
        memory_usage = df["RAM"]

        # Create a 1x3 subplot
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))

        # Plot CPU usage
        axs[0].plot(time, cpu_usage, label="CPU")
        axs[0].set_title("CPU Utilization")
        axs[0].set_ylabel("Usage (%)")
        axs[0].legend()

        # Plot GPU usage
        axs[1].plot(time, gpu_usage, label="GPU")
        axs[1].set_title("GPU Utilization")
        axs[1].set_ylabel("Usage (%)")
        axs[1].legend()

        # Plot memory usage
        axs[2].plot(time, memory_usage, label="Memory")
        axs[2].set_title(f"Memory Usage\n(Total = {psutil.virtual_memory().total:.2e} Bytes)")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Usage (Bytes)")
        axs[2].legend()

        # Adjust layout
        plt.tight_layout()
        # plt.savefig('integration/profiling/resource_nonJetson.png')
        plt.show()

    except:
        pass

if __name__ == "__main__":
    profiling_folder = "integration/profiling"

    display_fps()
    display_resource_jetson()
    display_resource()
    display_statistics()

