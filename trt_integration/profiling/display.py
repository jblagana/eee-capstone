# Displaying fps
import matplotlib.pyplot as plt
import csv

import os
import subprocess

import pandas as pd
import numpy as np

def display_fps():
    parsed_data = []
    csv_file = os.path.join(profiling_folder, "fps_log.csv")
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


def display_statistics():
# Displaying block statistics
    profiles = os.listdir(profiling_folder)
    for profile in profiles:
        if profile.endswith(".prof"):
            subprocess.Popen(["snakeviz", os.path.join(profiling_folder, profile)])


def display_resource():
# Display resource consumption
    df = pd.read_csv(os.path.join(profiling_folder, "resource_log.csv"))

    # Extract relevant columns
    time = df.index
    cpu_usage = df[["CPU1", "CPU2", "CPU3", "CPU4"]]
    gpu_usage = df["GPU"]
    memory_usage = df["RAM"]
    power_usage = df["Power TOT"]

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))

    # Plot CPU usage
    for cpu in cpu_usage.columns:
        axs[0, 0].plot(time, cpu_usage[cpu], label=cpu)
    axs[0, 0].set_title("CPU Usage")
    axs[0, 0].set_ylabel("Usage (%)")
    axs[0, 0].legend()

    # Plot GPU usage
    axs[0, 1].plot(time, gpu_usage, label="GPU (%)")
    axs[0, 1].set_title("GPU Usage")
    axs[0, 1].set_ylabel("Usage")
    axs[0, 1].legend()

    # Plot memory usage
    axs[1, 0].plot(time, memory_usage, label="Memory (%)")
    axs[1, 0].set_title("Memory Usage")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Usage")
    axs[1, 0].legend()

    # Plot power usage
    axs[1, 1].plot(time, power_usage, label="Power (Watts)")
    axs[1, 1].set_title("Power Usage")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Usage")
    axs[1, 1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    profiling_folder = "integration/profiling"

    display_fps()
    display_resource()
    display_statistics()

