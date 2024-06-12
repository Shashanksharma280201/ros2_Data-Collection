import os
import pickle
from PIL import Image
import argparse
import tqdm # type: ignore
import yaml
from rclpy import init, shutdown
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image as ImageMsg
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import cv_bridge
import rosbag2_py
from rosidl_runtime_py import deserialize_message
from vint_train.process_data.process_data_utils import *


def main(args: argparse.Namespace):
    init()

    # load the config file
    with open("vint_train/process_data/process_bags_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # create output dir if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # iterate recursively through all the folders and get the path of files with .bag extension in the args.input_dir
    bag_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".bag") and "diff" in file:
                bag_files.append(os.path.join(root, file))
    if args.num_trajs >= 0:
        bag_files = bag_files[: args.num_trajs]

    executor = SingleThreadedExecutor()

    # processing loop
    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptions(uri=bag_path)
            reader.open(storage_options)
        except Exception as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")
            continue

        # name is that folders separated by _ and then the last part of the path
        traj_name = "_".join(bag_path.split("/")[-2:])[:-4]
        traj_folder = os.path.join(args.output_dir, traj_name)
        if not os.path.exists(traj_folder):
            os.makedirs(traj_folder)

        bag_img_data = {"obs_images": [], "diff_images": []}
        bag_traj_data = {"odom": []}

        for topic, msg, t in reader.read_messages():
            msg_type = reader.get_message_type(topic)

            if topic == "/usb_cam_front/image_raw":
                img = deserialize_message(msg, ImageMsg)
                img_data = cv_bridge.CvBridge().imgmsg_to_cv2(img, "rgb8")
                bag_img_data["obs_images"].append(Image.fromarray(img_data))

            elif topic == "/chosen_subgoal":
                img = deserialize_message(msg, ImageMsg)
                img_data = cv_bridge.CvBridge().imgmsg_to_cv2(img, "rgb8")
                bag_img_data["diff_images"].append(Image.fromarray(img_data))

            elif topic == "/odom":
                odom = deserialize_message(msg, Odometry)
                bag_traj_data["odom"].append(odom)

        for i, (obs_image, diff_image) in enumerate(zip(bag_img_data["obs_images"], bag_img_data["diff_images"])):
            obs_image.save(os.path.join(traj_folder, f"{i}.jpg"))
            diff_image.save(os.path.join(traj_folder, f"diff_{i}.jpg"))

        with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
            pickle.dump(bag_traj_data['odom'], f)

        reader.close()

    shutdown()
    executor.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the datasets with rosbags",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="../datasets/tartan_drive/",
        type=str,
        help="path for processed dataset (default: ../datasets/tartan_drive/)",
    )
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="number of bags to process (default: -1, all)",
    )
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="sampling rate (default: 4.0 hz)",
    )

    args = parser.parse_args()
    print(f"STARTING PROCESSING DIFF DATASET")
    main(args)
    print(f"FINISHED PROCESSING DIFF DATASET")
