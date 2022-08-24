#!/usr/bin/env python3

import os
import shlex
import time
from datetime import timedelta
import subprocess
from tqdm.auto import tqdm

train_objects = ["butter", "cookies", "cheese", "mac_cheese", "raisins"]
test_objects = ["pudding", "granola", "popcorn", "spaghetti"]
all_objects = train_objects + test_objects


max_objs_unordered = {
    "butter" : 50,
    "cookies": 20,
    "cheese": 50,
    "mac_cheese" : 30,
    "raisins" : 40,
    "pudding" : 40,
    "granola" : 20,
    "popcorn" : 30,
    "spaghetti": 20
}

max_objs_ordered = {
    "butter" : 100,
    "cookies": 50,
    "cheese": 100,
    "mac_cheese": 70,
    "raisins" : 100,
    "pudding" : 100,
    "granola" : 50,
    "popcorn" : 70,
    "spaghetti": 50
}

num_scenes = 2
num_poses_per_scene = 5
scene = "scene2"

logs_dir = "./output_logs/"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# random drops
total = 0
for obj in all_objects:
    total += max_objs_unordered[obj]

count = 0
with tqdm(total=total) as pbar, open(os.path.join(logs_dir, "error.log"), "w") as errorfile:
    for obj in all_objects:
        with open(os.path.join(logs_dir, "{}_random.log".format(obj)), "w") as logfile:
            num_objs = 0
            while num_objs < max_objs_unordered[obj]:

                batch_size = num_objs if num_objs < 10 else 10

                params = "object={} dataset.num_scenes={} num_objs={} scene={} dataset.num_poses_per_scene={} batch_size={}".format(
                    obj, num_scenes, num_objs, scene, num_poses_per_scene, batch_size
                )

                cmd = "blenderproc run ./src/gen_data.py pack_type=random " + params
                pbar.display(cmd, pos=1)
                pbar.set_description("[{}][{}/{}]".format(obj, num_objs, max_objs_unordered[obj]))

                start = time.time()
                subprocess.run(shlex.split(cmd), stdout=logfile, stderr=errorfile, universal_newlines="\n")
                end = time.time()

                pbar.write("[{:0>8}] {}".format(str(timedelta(seconds=end-start)), cmd))

                if num_objs < 10:
                    num_objs += 1
                    count += 1
                elif 10 < num_objs < 40:
                    num_objs += 5
                    count += 5
                else:
                    num_objs += 10
                    count += 10

                pbar.update(count - pbar.n)



# orderely packed
total = 0
for obj in all_objects:
    total += max_objs_ordered[obj]

count = 0
with tqdm(total=total) as pbar, open(os.path.join(logs_dir, "error.log"), "w") as errorfile:
    for obj in all_objects:
        with open(os.path.join(logs_dir, "{}_ordered.log".format(obj)), "w") as logfile:
            num_objs = 0
            while num_objs < max_objs_ordered[obj]:
                params = "object={} dataset.num_scenes={} num_objs={} scene={} dataset.num_poses_per_scene={}".format(
                    obj, num_scenes, num_objs, scene, num_poses_per_scene    
                )

                cmd = "blenderproc run ./src/gen_data.py pack_type=ordered " + params
                pbar.display(cmd, pos=1)
                pbar.set_description("[{}][{}/{}]".format(obj, num_objs, max_objs_ordered[obj]))

                start = time.time()
                subprocess.run(shlex.split(cmd), stdout=logfile, stderr=errorfile, universal_newlines="\n")
                time.sleep(.5)
                end = time.time()

                pbar.write("[{:0>8}] {}".format(str(timedelta(seconds=end-start)), cmd))

                if num_objs < 10:
                    num_objs += 1
                    count += 1
                elif 10 < num_objs < 40:
                    num_objs += 5
                    count += 5
                else:
                    num_objs += 10
                    count += 10

                pbar.update(count - pbar.n)

