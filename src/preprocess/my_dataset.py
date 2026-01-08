import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class MyDataset(Dataset):
    def __init__(self, index_csv, root_dir, transform=None, sequence_len=25, split='test'):
        """
        Custom Dataset for categorized evaluation.
        
        Args:
            index_csv (str): Path to the labeled CSV (e.g., sequences_labeled.csv)
            root_dir (str): Root directory where images are stored.
            transform (callable, optional): Transforms to apply to images.
            sequence_len (int): Fixed number of frames per sequence.
            split (str): 'train', 'val', or 'test'.
        """
        self.df = pd.read_csv(index_csv)
        
        # Standardize columns based on split_dataset.py logic
        rename_dict = {}
        if 'person_global_id' in self.df.columns:
            rename_dict['person_global_id'] = 'person_id'
        if 'rgb_dir' in self.df.columns:
            rename_dict['rgb_dir'] = 'frames_dir'
        if rename_dict:
            self.df = self.df.rename(columns=rename_dict)

        if split is not None:
            if 'split' in self.df.columns:
                self.df = self.df[self.df['split'] == split].reset_index(drop=True)
            else:
                print(f"Warning: 'split' column not found in {index_csv}. Using all data.")
        
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_len = sequence_len
        
        # Map person_ids to integers (consistent with training labels)
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
        
        # Extract metadata
        metadata = {
            'sequence_id': row.get('sequence_id', ''),
            'cam_id': row.get('cam_id', ''),
            'illum_label': row.get('illum_label', 'unknown'),
            'occlusion_label': row.get('occlusion_label', 'unknown')
        }
        
        # Load frames
        full_frames_dir = os.path.join(self.root_dir, frames_dir)
        frame_paths = sorted(glob.glob(os.path.join(full_frames_dir, "*.jpg")))
        
        # Handle sequence length (logic from MARSDataset)
        num_frames = len(frame_paths)
        if num_frames == 0:
            raise ValueError(f"No frames found in {full_frames_dir}")
            
        if num_frames < self.sequence_len:
            # Pad with last frame
            diff = self.sequence_len - num_frames
            frame_paths += [frame_paths[-1]] * diff
        elif num_frames > self.sequence_len:
            # Uniform sampling
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
        
        return frames, label, metadata
