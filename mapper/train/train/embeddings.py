from math import ceil
from os.path import join, isfile
import pickle
import os
import re
import numpy as np
import pyarrow.parquet as pq


class Embeddings:

    def __init__(self, path: str):
        shards = []
        for filename in os.listdir(path):
            matches = re.match(r"^embeddings_(\d+).npy$", filename)
            if matches:
                idx = int(matches.groups()[0])
                shards.append((np.load(join(path, f"embeddings_{idx}.npy"),
                                       mmap_mode="r"), idx))

        shards.sort(key=lambda x: x[1])

        assert len(shards) > 0, "No embeddings shards were found."

        for (_, shard_idx), i in zip(shards, range(len(shards))):
            assert shard_idx == i, f"Embedding shard missing: {shard_idx} {i}."

        self.shards = tuple(shard for shard, _ in shards)

        for i in range(1, len(self.shards) - 1):
            assert self.shards[i].shape[0] == self.shards[i - 1].shape[
                0], "Shards (except the last one) need to contain the same number of elements."

        if len(self.shards) > 1:
            assert self.shards[-1].shape[0] <= self.shards[0].shape[
                0], "The last shard has to contain at most the number of elements as previous shards do."

        self.n_per_shard = ceil(
            sum(shard.shape[0] for shard in self.shards) / len(self.shards))

        if isfile(join(path, "nodes.pickle")):
            with open(join(path, "nodes.pickle"), "rb") as f:
                self.qid2idx = pickle.load(f)
        else:
            self.qid2idx = {
                qid: i
                for i, qid in enumerate(
                    pq.read_table(join(path, "nodes.parquet"), columns=["qid"])
                    ["qid"].to_pylist())
            }
            with open(join(path, "nodes.pickle"), "wb") as f:
                pickle.dump(self.qid2idx, f)

    def __getitem__(self, qid):
        idx = self.qid2idx.get(qid)

        if idx is None:
            return None

        return self.shards[idx // self.n_per_shard][idx % self.n_per_shard]
