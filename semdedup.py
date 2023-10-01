# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import numpy as np
import pandas as pd
import submitit
import torch
from tqdm import tqdm
import pickle
import random
import math
import time
import pprint
from constants import DIST_METRIC_INDEX


def init_memmap_embs(
    embs_memory_loc: str, dataset_size: int, emd_size: int = 512, dtype: str = "float32"
) -> np.memmap:
    """
    Initializes a memory-mapped NumPy array to read embeddings of examples.

    Args:
        embs_memory_loc (str): Path to the memory-mapped file.
        dataset_size (int): Size of the dataset.
        emd_size (int): Dimensionality of the embeddings.
        dtype (str): Data type of the embeddings.

    Returns:
        np.memmap: A memory-mapped NumPy array.
    """
    embs = np.memmap(
        embs_memory_loc, dtype=dtype, mode="r", shape=(dataset_size, emd_size)
    )
    return embs


class SemDeDupJob(submitit.helpers.Checkpointable):
    """
    - Each SLURMJob will run SemDeDup on number of clusters and save dataframe with which examples to keep from each cluster.
    - Parallelize job_start_cluster across jobs so that preemption in the middle of an epoch isn't a problem and because we want to
    keep the shard structure anyway.
    - Process more than one cluster per job=> run multiple taks inside each jobs.
    - Preempted jobs get resubmitted. Already precessed clusters get skipped internally.
    """

    def __init__(self, args, job_start_cluster: int):
        self.args = args
        self.job_start_cluster = job_start_cluster
        random.seed(args.seed)

    def _contains_duplicates(self, arr):
        return len(np.unique(arr)) != len(arr)

    def semdedup(self, cluster, cluster_reps, device):
        st = time.time()
        ## -- compute pairwise cos sim between cluster items, then replace to diagonal with zeros to ignore self similarity
        cluster_reps.to(device)
        pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
        del cluster_reps
        pair_w_sim_matrix.fill_diagonal_(0.0)
        assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]

        ## -- get paths to cluster i images
        image_urls = cluster[:, 0]

        ## -- make sure all the paths are unique this ensure that the duplicates are really stored many time times on memory
        assert not self._contains_duplicates(image_urls)

        ## -- We need upper tringular matrix because (1)we don't need to look at self sim (always=1) (2)we need the compinations not permutations
        triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)

        ## -- if the max sim between one example and any other example is > 1-eps, remove this example
        M = torch.max(triu_sim_mat, dim=0)[0].cpu()
        print(f"Step time: {time.time()-st}(s)")

        return M

    def _process_shard(self, start_cluster: int, end_cluster: int):
        # print("SemDeDup params: ", self.args)
        st = time.time()

        embs = init_memmap_embs(
            self.args.embs_memory_loc, self.args.dataset_size, self.args.emd_size
        )

        step_time = []

        for cluster_id in tqdm(range(start_cluster, end_cluster)):
            step_st = time.time()

            df_file_loc = os.path.join(
                self.args.save_loc, f"dataframes/cluster_{cluster_id}.pkl"
            )

            if os.path.exists(df_file_loc):  # and os.path.exists(dict_file_loc):
                print(f"{df_file_loc} exists, moving on")
                continue

            ## -- load cluster i representations
            cluster_i = np.load(
                os.path.join(
                    self.args.sorted_clusters_path, f"cluster_{cluster_id}.npy"
                )
            )
            # 1) store cluster size
            cluster_size = cluster_i.shape[0]
            print("cluster_size: ", cluster_size)

            if cluster_size == 1:
                points_to_remove_df = pd.DataFrame()
                points_to_remove_df["indices"] = [0]
                for eps in self.args.eps_list:
                    ## We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
                    points_to_remove_df[f"eps={eps}"] = [False]
                if self.args.save_loc != "":
                    ## --save df
                    with open(df_file_loc, "wb") as file:
                        pickle.dump(points_to_remove_df, file)
                print("DONE cluster_id ", cluster_id)
                continue

            ## -- By default, we keep hard examples from groups
            clutser_items_indices = list(range(cluster_size))
            ## -- OR: shuffle cluster to keep random example from each group
            if self.args.which_to_keep.lower() == "random":
                random.shuffle(clutser_items_indices)
                cluster_i = cluster_i[clutser_items_indices]
            ## -- OR: reverse cluster to keep easy examples
            if self.args.which_to_keep.lower() == "easy":
                clutser_items_indices = clutser_items_indices[::-1]
                cluster_i = cluster_i[clutser_items_indices]

            ## -- indices for cluster items in the dataset
            cluster_ids = cluster_i[:, 1].astype("int32")
            cluster_reps = embs[cluster_ids]
            cluster_reps = torch.tensor(cluster_reps)

            M = self.semdedup(cluster_i, cluster_reps, self.args.device)

            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = clutser_items_indices

            for eps in self.args.eps_list:
                ## -- 5) We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
                eps_points_to_remove = M > 1 - eps
                points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

            if self.args.save_loc != "":
                ## --save df
                with open(df_file_loc, "wb") as file:
                    pickle.dump(points_to_remove_df, file)

            step_time.append_cluster(time.time() - step_st)
            print("DONE cluster: ", cluster_id)

        print(
            f"DONE in {((time.time()-st)/60):.2f} minutes, Average Step time {(sum(step_time)/len(step_time)):.2f}(s)"
        )
        return

    def __call__(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(self.args))
        job_start_cluster = self.job_start_cluster

        print(
            f"This job will process clusters {job_start_cluster} to  {min(self.args.num_clusters, job_start_cluster+self.args.clusters_per_job)}"
        )

        job_env = submitit.JobEnvironment()

        print(f"There are {job_env.num_tasks} tasks in this job")

        print(f"I'm the task #{job_env.local_rank} on node {job_env.node}")
        print(f"I'm the task #{job_env.global_rank} in the job")

        ## divide clusters across tasks (cpus)
        num_clusters_per_task = int(
            math.ceil(self.args.clusters_per_job / job_env.num_tasks)
        )
        task_rank = job_env.local_rank
        start_cluster = job_start_cluster + task_rank * num_clusters_per_task
        end_cluster = job_start_cluster + (task_rank + 1) * num_clusters_per_task
        end_cluster = min(self.args.num_clusters, end_cluster)
        end_cluster = min(end_cluster, job_start_cluster + self.args.clusters_per_job)
        print(
            f"This task will process {num_clusters_per_task} clusters: cluster {start_cluster} to cluster {end_cluster}"
        )
        print(
            f"This task will process cluster {start_cluster} to cluster {end_cluster}"
        )

        self._process_shard(start_cluster, end_cluster)
