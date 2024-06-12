import os
import argparse
import yaml
import tqdm # type: ignore
import pickle
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from PIL import Image

def get_rosbag2_bag_files(input_dir):
    bag_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".db3"):
                bag_files.append(os.path.join(root, file))
    return bag_files

def main(args: argparse.Namespace):
    # load the config file
    with open("vint_train/process_data/process_bags_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create output dir if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # iterate recursively through all the folders and get the path of files with .db3 extension in the args.input_dir
    bag_files = get_rosbag2_bag_files(args.input_dir)
    if args.num_trajs >= 0:
        bag_files = bag_files[:args.num_trajs]

    # processing loop
    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
            converter_options = ConverterOptions('', '')
            reader = SequentialReader()
            reader.open(storage_options, converter_options)
        except Exception as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")
            continue

        # name is that folders separated by _ and then the last part of the path
        traj_name = "_".join(bag_path.split("/")[-2:])[:-4]

        # load the hdf5 file
        bag_img_data, bag_traj_data = get_images_and_odom(
            reader,
            config[args.dataset_name]["imtopics"],
            config[args.dataset_name]["odomtopics"],
            eval(config[args.dataset_name]["img_process_func"]),
            eval(config[args.dataset_name]["odom_process_func"]),
            rate=args.sample_rate,
            ang_offset=config[args.dataset_name]["ang_offset"],
        )

        if bag_img_data is None or bag_traj_data is None:
            print(
                f"{bag_path} did not have the topics we were looking for. Skipping..."
            )
            continue
        # remove backwards movement
        cut_trajs = filter_backwards(bag_img_data, bag_traj_data)

        for i, (img_data_i, traj_data_i) in enumerate(cut_trajs):
            traj_name_i = traj_name + f"_{i}"
            traj_folder_i = os.path.join(args.output_dir, traj_name_i)
            # make a folder for the traj
            if not os.path.exists(traj_folder_i):
                os.makedirs(traj_folder_i)
            with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
                pickle.dump(traj_data_i, f)
            # save the image data to disk
            for j, img in enumerate(img_data_i):
                img.save(os.path.join(traj_folder_i, f"{j}.jpg"))

def get_images_and_odom(reader, imtopics, odomtopics, img_process_func, odom_process_func, rate, ang_offset):
    # Implement your logic to extract images and odom data from the ROS 2 bag
    # For example purposes, placeholder return values are provided.
    bag_img_data = []
    bag_traj_data = []
    # Placeholder for data extraction logic
    return bag_img_data, bag_traj_data

def filter_backwards(bag_img_data, bag_traj_data):
    # Implement your logic to filter backwards movement
    # For example purposes, a placeholder return value is provided.
    cut_trajs = [(bag_img_data, bag_traj_data)]
    return cut_trajs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # get arguments for the recon input dir and the output dir
    # add dataset name
    parser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        help="name of the dataset (must be in process_config.yaml)",
        default="tartan_drive",
        required=True,
    )
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
    # number of trajs to process
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="number of bags to process (default: -1, all)",
    )
    # sampling rate
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="sampling rate (default: 4.0 hz)",
    )

    args = parser.parse_args()
    # all caps for the dataset name
    print(f"STARTING PROCESSING {args.dataset_name.upper()} DATASET")
    main(args)
    print(f"FINISHED PROCESSING {args.dataset_name.upper()} DATASET")
