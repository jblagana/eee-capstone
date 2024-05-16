import os
import subprocess

def visualize_profiles(profiling_folder):
    profiles = os.listdir(profiling_folder)
    for profile in profiles:
        if profile.endswith(".prof"):
            print(f"Visualizing {profile} with SnakeViz...")
            subprocess.Popen(["snakeviz", os.path.join(profiling_folder, profile)])

if __name__ == "__main__":
    profiling_folder = "integration/profiling_folder"
    visualize_profiles(profiling_folder)
