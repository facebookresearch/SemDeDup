# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import torch
import time
import numpy as np
import logging
import os
import pickle
import argparse
import yaml
import pprint
import submitit
import pathlib
from typing import Union, Optional
from utils import get_logger


def faiss_index_to_gpu(cpu_index):
    """
    Convert a Faiss CPU index to a GPU index.
    """
    # Configure GPU cloner options
    cloner_options = faiss.GpuClonerOptions()
    cloner_options.useFloat16 = False
    cloner_options.usePrecomputed = False
    cloner_options.indicesOptions = faiss.INDICES_CPU

    # Configure Faiss GPU resources
    gpu_resources = faiss.StandardGpuResources()

    # Convert CPU index to GPU index
    gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index, cloner_options)

    return gpu_index


def compute_centroids(
    data: Union[np.memmap, np.ndarray],
    ncentroids: int = 1000,
    niter: int = 100,
    seed: int = 1234,
    Kmeans_with_cos_dist: bool = False,
    save_folder: str = "",
    logger: logging.Logger = None,
    verbose: bool = True,
):

    """
    Runs K-means clustering on the input data using "faiss" and saves the following output files:

          1)faiss k-means index object (pickle file).
          2)k-means centroids (numpy array).
          3)Distance to centroid for data points in <data> (numpy array).
          4)Nearest centroid for data points in <data> (numpy array).
    args:
        data: A float32 numpy memmap array or numpy array of shape [dataset_size x d], where d is the embedding vector size..
        ncentroids: number of kmeans clusters/centroids.
        niter: The number of iterations to run the K-means algorithm for.
        seed: The random seed to use for reproducibility.
        Kmeans_with_cos_dist: (boolean) when True, run spherical kmeans.
        save_folder: path to save/load output files.
        logger: A logger instance to use for logging.

    returns:
        faiss k-means object
    """
    os.makedirs(save_folder, exist_ok=True)
    # -- Compute Kmeans centroids
    logger.info(
        f"Running Kmeans clustering using faiss on dataset of shape {data.shape} ...."
    )
    logger.info(f"Kmeans parameters: {locals()} ....")
    # pprint.pprint(locals(), width=1, indent=4)

    d = data.shape[1]
    # -- Use GPUs for clustering when available
    use_gpu = torch.cuda.is_available()

    device = "cuda" if use_gpu else "cpu"

    logger.info(f"Clustering on {device} ....")

    spherical = (
        Kmeans_with_cos_dist  # -- spherical=True when Kmeans_with_cos_dist is True
    )

    ## -- Step 1) Train faiss kmeans
    kmeans = faiss.Kmeans(
        d,
        ncentroids,
        niter=niter,
        verbose=verbose,
        seed=seed,
        spherical=spherical,
        gpu=use_gpu,
    )  ## -- faiss.Kmeans "gpu" argument: bool or int, optional. False: don't use GPU, True: use all GPUs, number: use this many GPUs.

    # -- If kmeans centroids are not saved - > create and train faiss Kmeans clustering object
    kmeans_obj_file_loc = pathlib.Path(save_folder, "kmeans_index.pickle")

    if not os.path.exists(kmeans_obj_file_loc):
        start_time = time.time()
        kmeans.train(data)
        logger.info(f"Time for clustering (mins): {(time.time()-start_time)/(60):.2f}")

        # -- Move kmeans index to cpu to save it
        kmeans_index = faiss.index_gpu_to_cpu(kmeans.index)
        logger.info(f"faiss kmeans index to store: {type(kmeans_index)}")
        ## -- Save faiss kmeans index object as pickle file
        with open(kmeans_obj_file_loc, "wb") as file:
            pickle.dump(kmeans_index, file)
        ## -- save faiss kmeans centroids as npy file
        np.save(pathlib.Path(save_folder, "kmeans_centroids.npy"), kmeans.centroids)

        logger.info(f"Saved!")

    else:
        # -- Else, load stored kmeans object
        logger.info(
            f"Loading faiss Kmeans index pickle file from {kmeans_obj_file_loc}"
        )
        with open(kmeans_obj_file_loc, "rb") as file:
            kmeans_index = pickle.load(file)
            if use_gpu:
                # -- move kmeans index to gpu
                kmeans_index = faiss_index_to_gpu(kmeans_index)
            kmeans.index = kmeans_index

    ## -- Step 2) Find the nearest centroid for each data point, l2 distance search
    ## -- nearest_cent: the nearest centroid for each example in data. dist_to_cent: contains the squared L2 distances.
    start_time = time.time()
    dist_to_cent, nearest_cent = kmeans.index.search(data, 1)
    dist_to_cent, nearest_cent = dist_to_cent.squeeze(1), nearest_cent.squeeze(1)
    logger.info(
        f"Time for finding nearest centroid for each data point (mins): {(time.time()-start_time)/(60):.2f}"
    )

    ## -- save faiss nearest_cent and dist_to_cent as .npy files
    dist_to_cent_file = pathlib.Path(save_folder, "dist_to_cent.npy")
    nearest_cent_file = pathlib.Path(save_folder, "nearest_cent.npy")
    np.save(dist_to_cent_file, dist_to_cent)
    np.save(nearest_cent_file, nearest_cent)

    return kmeans


def main(args):
    # Initialize logger
    log_file = os.path.join(args.save_folder, "compute_centroids.log")
    logger = get_logger(
        file_name=log_file,
        level=logging.INFO,
        stdout=True,
    )

    # Load configuration file for clustering
    confg_file = args.confg_file

    with open(confg_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    with open(
        pathlib.Path(params["save_folder"], "clustering_params.txt"), "w"
    ) as fout:
        pprint.pprint(params, fout)

    ## -- Load clustering parameters
    seed = params["seed"]
    emb_memory_loc = params[
        "emb_memory_loc"
    ]  ## -- numpy menmap where embeddings are stored
    dataset_size = params["dataset_size"]
    emb_size = params["emb_size"]
    niter = params["niter"]
    ncentroids = params["ncentroids"]
    save_folder = params["save_folder"]
    Kmeans_with_cos_dist = params["Kmeans_with_cos_dist"]

    ## -- Load embeddings
    data = np.memmap(
        emb_memory_loc, dtype="float32", mode="r", shape=(dataset_size, emb_size)
    )

    ## -- Compute centroids
    compute_centroids(
        data,
        ncentroids,
        niter,
        seed,
        Kmeans_with_cos_dist,
        save_folder,
        logger,
        True,
    )


if __name__ == "__main__":
    # Configure command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--confg-file",
        type=str,
        default="configs/openclip/paralellized_kmeans_dino_embs_configs.yaml",
        help=".yaml config file path",
    )
    # -- slurm parameters
    parser.add_argument(
        "--partition", type=str, default="scaling_data_pruning", help="partition"
    )
    parser.add_argument("--ngpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--cpus-per-task", type=int, default=10, help="number of cpus")
    parser.add_argument(
        "--timeout", type=int, default=1500, help="job timeout in minutes"
    )

    args = parser.parse_args()

    # Load configuration file
    with open(args.confg_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    ## -- Logging directory
    args.save_folder = params["save_folder"]

    # SLURM CONFIG
    PARTITION = args.partition
    NODES = 1
    NGPUS = args.ngpus
    CPUS_PER_TASKS = args.cpus_per_task
    TIMEOUT = args.timeout

    # Configure submitit executor
    submitit_path = f"{args.save_folder}/compute_centorids_job_%j"
    executor = submitit.AutoExecutor(folder=submitit_path, slurm_max_num_timeout=30)
    executor.update_parameters(
        slurm_partition=PARTITION,
        nodes=NODES,
        tasks_per_node=1,
        cpus_per_task=CPUS_PER_TASKS,
        gpus_per_node=NGPUS,
        slurm_mem_per_gpu="55G",
        timeout_min=TIMEOUT,
    )

    # Submit job
    job = executor.submit(main, args)
    print("Submitted job_id:", job.job_id)
