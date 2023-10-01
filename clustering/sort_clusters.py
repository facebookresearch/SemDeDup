# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import time
import numpy as np
import logging
from tqdm import tqdm
import pandas as pd
import os
import pathlib
import yaml
import math
import os
from typing import List
import random
import numpy as np
import submitit
import torch
import pprint
from tqdm import tqdm
import argparse
from typing import List, Tuple, Union
from utils import get_logger


def assign_and_sort_clusters(
    data: Union[np.memmap, np.ndarray],
    paths_list: Union[np.memmap, np.ndarray],
    sim_metric: str = "cosine",
    keep_hard: bool = True,
    kmeans_with_cos_dist: bool = False,
    save_folder: str = "",
    sorted_clusters_file_loc: str = "",
    cluster_ids=range(5000),
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Assigns data points to clusters and sorts each cluster items based on distance to its centroid.

    Args:
        data (np.memmap): A memory-mapped array containing the data points.
        paths_list (np.memmap): A memory-mapped array containing the paths of the data points.
        sim_metric (str): The similarity metric to use for clustering. Defaults to "cosine".
        keep_hard (bool): When True, we sort cluster items in descending order by the similarity to cluster centroid. Defaults to True.
        kmeans_with_cos_dist (bool): Whether to use cosine distance for K-means clustering. Defaults to False.
        save_folder (str): The location of the K-means centroids file. Defaults to "".
        sorted_clusters_file_loc (str): The location to save the sorted clusters file. Defaults to "".
        logger (logging.Logger): A logger object to log messages. Defaults to None.
        cluster_ids (list): The range of cluster IDs to sort. Defaults to range(5000).

    Returns:
        pd.DataFrame: A DataFrame containing the sorted clusters.
    """

    assert sim_metric in [
        "l2",
        "cosine",
    ], f"Unsupported similarity metric '{sim_metric}'."
    assert not (
        kmeans_with_cos_dist and sim_metric == "l2"
    ), "Cannot use cosine distance with L2 similarity metric."

    # If Kmeans_with_cos_dist is True, set spherical=True. This is the spherical parameter of faiss kmeans clustering.
    spherical = kmeans_with_cos_dist

    # Step 3: Sort each class/cluster
    logger.info("Ranking...")
    kmeans_centroids_file_loc = pathlib.Path(save_folder, "kmeans_centroids.npy")
    dist_to_cent_file_loc = pathlib.Path(save_folder, "dist_to_cent.npy")
    nearest_cent_file_loc = pathlib.Path(save_folder, "nearest_cent.npy")
    kmeans_centroids = np.load(kmeans_centroids_file_loc)
    nearest_cent = np.load(nearest_cent_file_loc)
    dist_to_cent = np.load(dist_to_cent_file_loc)

    start_time = time.time()

    dist_df = pd.DataFrame(
        {
            "paths_list": paths_list,
            "nearest_cent": nearest_cent,
            "dist_to_cent": dist_to_cent,
        }
    )

    sorted_clusters = rank_within_cluster(
        data,
        dist_df,
        kmeans_centroids,
        sim_metric,
        keep_hard,
        spherical,
        cluster_ids,
        sorted_clusters_file_loc,
    )
    logger.info(f"Time for ranking: {(time.time() - start_time) / 60:.2f} mins")
    logger.info("DONE!")

    return sorted_clusters


def rank_within_cluster(
    data: Union[np.memmap, np.ndarray],
    dist_df: pd.DataFrame,
    centroids: np.ndarray,
    sim_metric: str = "cosine",
    keep_hard: bool = True,
    spherical: bool = False,
    cluster_ids: List[int] = range(50000),
    sorted_clusters_file_loc: str = "",
) -> List[List[Tuple[str, int, float, int]]]:
    """
    Sorts each cluster items by the distance to the cluster centroid.
    Cluster is represented as list of tuples. Each tuple has 4 values:
        example_path: unique path to the example/image/text doc, for imagenet it could be something like "n04235860_14959.JPEG",
        example_id_in_dataset: int between 0 and cluster_size-1
        dist_to_cent: cosine distance to cluster centroid
        cluster_id: cluster number (from 0 to number of clusters)

    Arguments:
    data -- the data for which the clusters were created (np.ndarray or np.memmap)
    dist_df -- DataFrame with the distances between the data points and the centroids, nearest centroid for each example, and path to each example.
    centroids -- np.ndarray with the centroids for each cluster.
    sim_metric -- the similarity metric used to compute distances, should be one of ["cosine", "l2"]
    keep_hard -- a boolean ehen True, we sort cluster items in descending order by the similarity to cluster centroid. Defaults to True.
    spherical -- a boolean True means spherical was used for computing centroids (used for cosine similarity).
    cluster_ids -- a list of cluster ids to process. Each slurm job will process part of the clusters.
    sorted_clusters_file_loc -- the location to save the sorted clusters.

    Returns:
    A list of cluster representations, where each representation is a list of tuples with 4 values.
    -- exampel for a cluster (the list bellow is sorted by dist_to_cent in descending order)
        [
          [example_name, example_id_in_dataset, dist_to_cent, cluster_label],
          [example_name, example_id_in_dataset, dist_to_cent, cluster_label],
                                        .
                                        .
                                        .
                                                                    ]
    """

    assert sim_metric in [
        "cosine",
        "l2",
    ], "sim_metric should be one of ['cosine', 'l2']"
    os.makedirs(sorted_clusters_file_loc, exist_ok=True)

    sorted_clusters_list = []
    for cluster_c in tqdm(cluster_ids):
        if os.path.exists(f"{sorted_clusters_file_loc}/cluster_{cluster_c}.npy"):
            print(f"Cluster {cluster_c} exits, skipping....")
            continue

        cluster_df = dist_df.loc[dist_df["nearest_cent"] == cluster_c]

        cluster_items = list(cluster_df.index)  ## -- ids of examples in cluster c
        if sim_metric == "cosine":
            if spherical:
                cluster_dists_to_cent = list(1 - cluster_df["dist_to_cent"])
            else:
                cluster_c_centroid = torch.Tensor(centroids[cluster_c])
                sim_to_cent = torch.nn.CosineSimilarity(dim=1)(
                    torch.Tensor(data[cluster_items]), cluster_c_centroid
                )
                cluster_dists_to_cent = (1 - sim_to_cent).tolist()

        elif sim_metric == "l2":  # -- get l2 distance from "dist_to_cent" array
            cluster_dists_to_cent = list(cluster_df["dist_to_cent"])

        cluster_label = np.full((len(cluster_df)), cluster_c).tolist()
        example_paths = list(cluster_df["paths_list"])
        sort_descending = keep_hard
        cluster_sorted = sorted(
            zip(example_paths, cluster_items, cluster_dists_to_cent, cluster_label),
            key=lambda x: x[2],
            reverse=sort_descending,
        )  # -- sort_descending = True for descending sort

        sorted_clusters_list.append(
            cluster_sorted
        )  # -- Descending dists. list of of list of tuples (example, dist). The ith list of tuples corresponds to cluster i
        sorted_cluster_file_path = f"{sorted_clusters_file_loc}/cluster_{cluster_c}.npy"
        np.save(sorted_cluster_file_path, cluster_sorted)
    return sorted_clusters_list


class SLURMJob(submitit.helpers.Checkpointable):
    """
    - Each SLURMJob will calculate and save the encodings for a list of shards.
    - Parallelize shards across jobs so that preemption in the middle of an epoch isn't a problem and because we want to
    keep the shard structure anyway.
    - Process more than one shard per job because each shard takes about a minute so we want to amortize overhead.
    - Preempted jobs get resubmitted. Already computed shards get skipped internally.

    """

    def __init__(self, args, cluster_ids: List[str]):
        self.args = args
        self.cluster_ids = cluster_ids
        assert args.ncentroids == len(self.cluster_ids)

    def seed_everything(self, seed=42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _encode_shard(self, args):
        # Configure logger
        logger = get_logger(
            file_name=f"{args.save_folder}/clustering-logs.log",
            level=logging.INFO,
            stdout=True,
        )
        args.logger = logger

        data = np.memmap(
            args.emb_memory_loc,
            dtype="float32",
            mode="r",
            shape=(args.dataset_size, args.emb_size),
        )
        paths_list = np.memmap(
            args.paths_memory_loc,
            dtype=args.path_str_dtype,
            mode="r",
            shape=(args.dataset_size,),
        )

        assign_and_sort_clusters(
            data,
            paths_list,
            args.sim_metric,
            args.keep_hard,
            args.Kmeans_with_cos_dist,
            args.save_folder,
            args.sorted_clusters_file_loc,
            args.cluster_ids_for_job,
            args.logger,
        )

        return

    def __call__(self):
        self.seed_everything(self.args.seed)

        num_clusters = len(self.cluster_ids)
        print(
            f"There are {num_clusters} clusters: {self.cluster_ids[0]} to  {self.cluster_ids[-1]}"
        )

        job_env = submitit.JobEnvironment()

        print(f"There are {args.num_tasks} tasks in this job")
        print(f"This is the task #{job_env.local_rank}")

        ## devide clusters across jobs (cpus)
        num_clusters_per_job = int(math.ceil(num_clusters / args.num_tasks))
        task_rank = job_env.local_rank
        start = task_rank * num_clusters_per_job
        end = (task_rank + 1) * num_clusters_per_job
        end = min(end, num_clusters)

        cluster_ids_for_job = self.cluster_ids[start:end]
        print(
            f"This job/task will process {len(cluster_ids_for_job)} clusters: cluster {cluster_ids_for_job[0]} to cluster {cluster_ids_for_job[-1]}"
        )

        self.args.cluster_ids_for_job = cluster_ids_for_job

        self._encode_shard(self.args)


def launch_jobs(args):
    """
    Runs the clustering job using the specified configuration file and SLURM parameters.

    """
    confg_file = args.config_file
    ## -- load kmeans clustering parameters from configs file
    with open(confg_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    with open(pathlib.Path(params["save_folder"], "clustering_params.txt"), "w") as f:
        pprint.pprint(params, f)

    args.save_folder = params["save_folder"]
    args.emb_memory_loc = params["emb_memory_loc"]
    args.paths_memory_loc = params["paths_memory_loc"]
    args.dataset_size = params["dataset_size"]
    args.emb_size = params["emb_size"]
    args.path_str_dtype = params["path_str_dtype"]  # "S24" for LAION
    args.ncentroids = params["ncentroids"]
    args.seed = params["seed"]
    args.sim_metric = params["sim_metric"]
    args.keep_hard = params["keep_hard"]
    args.Kmeans_with_cos_dist = params["Kmeans_with_cos_dist"]
    args.save_folder = params["save_folder"]
    args.sorted_clusters_file_loc = params["sorted_clusters_file_loc"]
    args.cluster_ids_for_job = list(range(args.ncentroids))

    ## -- SLURM CONFIG
    PARTITION = args.partition
    SLURM_ARRAY_PARALLELISM = 1000
    NODES = 1
    TIMEOUT = args.timeout
    CPUS_PER_TASKS = args.cpus_per_task
    TASKS_PER_NODE = args.num_tasks

    ## -- SUBMIT
    submitit_path = f"{args.save_folder}/clustering-jobs/%j"
    executor = submitit.AutoExecutor(folder=submitit_path, slurm_max_num_timeout=30)
    executor.update_parameters(
        slurm_partition=PARTITION,
        slurm_array_parallelism=SLURM_ARRAY_PARALLELISM,
        nodes=NODES,
        tasks_per_node=TASKS_PER_NODE,
        cpus_per_task=CPUS_PER_TASKS,
        timeout_min=TIMEOUT,
    )

    jobs = []

    ## -- Start a job with <args.num_tasks> task. Each task will process part of the clusters
    with executor.batch():
        exp = SLURMJob(args, list(range(0, args.ncentroids)))
        job = executor.submit(exp)
        jobs.append(job)

    for job in jobs:
        print("Submitted job_id:", job.job_id)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/openclip/paralellized_kmeans_dino_embs_configs.yaml",
    )
    # -- slurm parameters
    parser.add_argument(
        "--partition", type=str, default="scaling_data_pruning", help="partition"
    )
    parser.add_argument("--num-tasks", type=int, default=10, help="number of tasks")
    parser.add_argument(
        "--cpus-per-task", type=int, default=5, help="number of cpus per task"
    )
    parser.add_argument(
        "--timeout", type=int, default=500, help="job timeout in minutes"
    )

    args = parser.parse_args()

    # Submit the job
    launch_jobs(args)
