import os
import glob
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
import pandas as pd
import numpy as np
from collections import defaultdict

class MARSDataset(Dataset):
    def __init__(self, index_csv, root_dir, transform=None, sequence_len=25):
        self.df = pd.read_csv(index_csv)
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_len = sequence_len
        
        # Map person_ids to integers
        self.pids = sorted(self.df['person_id'].unique())
        self.pid2label = {pid: i for i, pid in enumerate(self.pids)}
        self.num_classes = len(self.pids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames_dir = row['frames_dir']
        pid = row['person_id']
        label = self.pid2label[pid]
        
        # Load frames
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        
        # Handle sequence length
        num_frames = len(frame_paths)
        if num_frames == 0:
            raise ValueError(f"No frames found in {frames_dir}")
            
        if num_frames < self.sequence_len:
            # Pad with last frame
            diff = self.sequence_len - num_frames
            frame_paths += [frame_paths[-1]] * diff
        elif num_frames > self.sequence_len:
            # Sample uniformly or take first N?
            # For training, random sampling might be better, but for now let's take uniform
            indices = np.linspace(0, num_frames-1, self.sequence_len).astype(int)
            frame_paths = [frame_paths[i] for i in indices]
            
        frames = []
        for p in frame_paths:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
            
        # Stack frames: [T, C, H, W]
        frames = torch.stack(frames)
        
        return frames, label

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances.
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        
        for index, row in data_source.df.iterrows():
            pid = row['person_id']
            self.index_dic[pid].append(index)
            
        self.pids = list(self.index_dic.keys())
        
        # Estimate length
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = self.index_dic[pid]
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            
            np.random.shuffle(idxs)
            batch_idxs_dict[pid] = idxs

        avai_pids = list(self.index_dic.keys())
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            
            for pid in selected_pids:
                batch_idxs_dict[pid]
                
                # Pop K instances
                for _ in range(self.num_instances):
                    final_idxs.append(batch_idxs_dict[pid].pop(0))
                    
                if len(batch_idxs_dict[pid]) < self.num_instances:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length
