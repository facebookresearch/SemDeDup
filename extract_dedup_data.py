# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from tqdm import tqdm
import pickle
import numpy as np
from constants import IMAGE_NAME_INDEX


def extract_pruned_data(
    sorted_clusters_path,
    semdedup_pruning_tables_path,
    eps,
    num_clusters,
    output_txt_path,
    retreive_kept_samples=True,
):

    ## -- list of paths to the examples we want to keep/remove.
    example_paths = []

    for cluster_id in tqdm(range(0, num_clusters)):

        cluster_i = np.load(
            os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy")
        )
        with open(
            f"{semdedup_pruning_tables_path}/cluster_{cluster_id}.pkl", "rb"
        ) as file:
            semdedup_pruning_tables = pickle.load(file)

        ## -- See which examples to keep/remove from this cluster.
        ## -- Use retreive_kept_samples=True when kept dataset size <= 50%. This will return a smaller output text file,
        ## -- semdedup_pruning_tables contain True values for the examples to be removed.
        images_to_keep_or_remove = semdedup_pruning_tables[f"eps={eps}"][
            semdedup_pruning_tables[f"eps={eps}"] == (not retreive_kept_samples)
        ].index.to_numpy()
        if "indices" in semdedup_pruning_tables.columns:
            cluster_i = cluster_i[semdedup_pruning_tables["indices"]]
        ## -- retrieve only the examples we want and add to the list.
        dedup_cluster = cluster_i[images_to_keep_or_remove]
        example_paths += dedup_cluster[:, IMAGE_NAME_INDEX].astype("<U32").tolist()

    with open(output_txt_path, "w") as fp:
        fp.write("\n".join(example_paths))

    print(f"DONE saving {len(example_paths)} image paths")

    return
