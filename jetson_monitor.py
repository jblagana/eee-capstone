# This file is part of the jetson_stats package.
# Copyright (c) 2019-2023 Raffaello Bonghi.

from jtop import jtop, JtopException
import csv
import argparse
import time

def jetson_stats_mod():
    stats = {'time': time.datetime.now(), 'uptime': jtop.uptime}
    # CPU
    for idx, cpu in enumerate(jtop.cpu['cpu']):
        stats["CPU{idx}".format(idx=idx + 1)] = 100 - int(cpu['idle']) if cpu['online'] else 'OFF'

    # GPU
    for idx, gpu in enumerate(jtop.gpu.values()):
        gpu_name = 'GPU' if idx == 0 else 'GPU{idx}'.format(idx=idx)
        stats[gpu_name] = gpu['status']['load']

    # MEMORY
    stats['used_RAM'] = jtop.memory['RAM']['used'],,
    stats['tot_RAM'] = jtop.memory['RAM']['tot']
    # stats['RAM'] = jtop.memory['RAM']['used'] / tot_ram if tot_ram > 0 else 0

    
    return stats
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jtop logger")
    parser.add_argument("--file", action="store", dest="file", default="trt_integration/profiling/resoure_log-jetson.csv")
    args = parser.parse_args()

    print("Simple jtop logger")
    print(f"Saving log to {args.file}")

    try:
        with jtop() as jetson:
            # Create CSV file and set up CSV writer
            with open(args.file, "w") as csvfile:
                
                stats = jetson_stats_mod()

                writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
                writer.writeheader()
                writer.writerow(stats)

                # Start loop
                while jetson.ok():
                    stats = jetson_stats_mod()
                    writer.writerow(stats)
                    # print(f"Logged at {stats['time']}")

                    time.sleep(1)

    except JtopException as e:
        print(e)
    except KeyboardInterrupt:
        print("Closed.")
    except IOError:
        print("I/O error")
