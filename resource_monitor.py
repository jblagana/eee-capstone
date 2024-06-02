import os
import psutil
import time
import csv
import subprocess

profiling_folder = "integration/profiling"

# CSV file to log resource usage
csv_file = os.path.join(profiling_folder, 'resource_nonJetson.csv')

try:
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Index', 'CPU', 'GPU', 'RAM'])

        index = 0
        print("Resource monitoring started...")
        
        while True:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().used

            # Get GPU usage using nvidia-smi
            try:
                gpu_usage_output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
                gpu_usage = float(gpu_usage_output.decode('utf-8').strip())
            except subprocess.CalledProcessError:
                gpu_usage = 0.0  # If nvidia-smi fails, set GPU usage to 0

            csv_writer.writerow([index, cpu_usage, gpu_usage, memory_usage])

            index += 1

            # Sleep for a second
            time.sleep(1)

except KeyboardInterrupt:
        print("Closed.")






# animated version
# import psutil
# import time
# import datetime
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import csv

# # Lists to store the data
# timestamps = []
# cpu_usages = []
# memory_usages = []

# # Create the figure for plotting
# fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 5))

# # Get the current datetime
# now = datetime.datetime.now()

# # Format the datetime string to be used in the filename
# datetime_str = now.strftime("%Y%m%d_%H%M%S")

# # Function to update the data and the plots
# def update(i):
#     # Get the current time
#     current_time = time.time() - start_time

#     # Get CPU usage
#     cpu_usage = psutil.cpu_percent()
#     cpu_usages.append(cpu_usage)

#     # Get Memory usage
#     memory_usage = psutil.virtual_memory().used
#     memory_usages.append(memory_usage)

#     timestamps.append(current_time)

#     # Log the data to a CSV file
#     with open(f'log_{datetime_str}.csv', 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([i, datetime.datetime.now(), cpu_usage, memory_usage])

#     # Update the plots
#     ax1.clear()
#     ax1.plot(timestamps, cpu_usages)
#     ax1.set(xlabel='Time (seconds)', ylabel='CPU Usage (%)',
#             title='CPU Usage over Time')

#     ax2.clear()
#     ax2.plot(timestamps, memory_usages)
#     ax2.set(xlabel='Time (seconds)', ylabel='Memory Usage (bytes)',
#             title='Memory Usage over Time')

# # Start the animation
# start_time = time.time()
# ani = animation.FuncAnimation(fig, update, interval=1000)

# plt.tight_layout()
# plt.show()
