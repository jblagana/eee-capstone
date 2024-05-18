# This file is part of the jetson_stats package.
# Copyright (c) 2019-2023 Raffaello Bonghi.

from jtop import jtop, JtopException
import csv
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jtop logger")
    parser.add_argument("--file", action="store", dest="file", default="resoure_log-jetson.csv")
    args = parser.parse_args()

    print("Simple jtop logger")
    print(f"Saving log to {args.file}")

    try:
        with jtop() as jetson:
            # Create CSV file and set up CSV writer
            with open(args.file, "w") as csvfile:
                stats = jetson.stats
                writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
                writer.writeheader()
                writer.writerow(stats)

                # Start loop
                while jetson.ok():
                    stats = jetson.stats
                    writer.writerow(stats)
                    # print(f"Logged at {stats['time']}")

                    time.sleep(1)

    except JtopException as e:
        print(e)
    except KeyboardInterrupt:
        print("Closed.")
    except IOError:
        print("I/O error")
