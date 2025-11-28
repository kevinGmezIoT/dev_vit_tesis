import os
import argparse
import pandas as pd
import shutil
from tqdm import tqdm
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MARS dataset")
    parser.add_argument("--raw_root", type=str, required=True, help="Path to raw MARS dataset")
    parser.add_argument("--out_root", type=str, required=True, help="Path to processed MARS dataset")
    parser.add_argument("--sequence_len", type=int, default=25, help="Length of sequence")
    parser.add_argument("--overlap", type=int, default=10, help="Overlap between sequences")
    return parser.parse_args()

def process_tracklet(tracklet_path, out_dir, sequence_len, overlap, person_id, cam_id, tracklet_id):
    frames = sorted(glob.glob(os.path.join(tracklet_path, "*.jpg")))
    num_frames = len(frames)
    
    if num_frames < sequence_len:
        # If tracklet is shorter than sequence_len, we can either skip it or pad it.
        # For now, let's skip very short tracklets or just take what we have if we handle variable length.
        # But the requirement is fixed length. Let's replicate frames if needed or just skip.
        # Strategy: Loop/Repeat frames to reach sequence_len
        pass 

    sequences = []
    
    # Generate sequences
    step = sequence_len - overlap
    if step < 1: step = 1
    
    for i in range(0, num_frames, step):
        end = i + sequence_len
        if end > num_frames:
            if i == 0: # Special case: short tracklet, take all and pad/repeat later in dataset or here
                 seq_frames = frames
            else:
                break # Ignore tail if not enough frames
        else:
            seq_frames = frames[i:end]
        
        if len(seq_frames) < sequence_len:
             # Pad by repeating last frame? Or just ignore.
             # Let's ignore tails for now unless it's the only sequence.
             if num_frames >= sequence_len:
                 continue

        # Create sequence directory
        seq_name = f"cam{cam_id}_seq{tracklet_id}_{i:04d}"
        seq_dir = os.path.join(out_dir, seq_name, "rgb")
        os.makedirs(seq_dir, exist_ok=True)
        
        # Copy frames
        for j, frame_path in enumerate(seq_frames):
            shutil.copy(frame_path, os.path.join(seq_dir, f"frame{j:04d}.jpg"))
            
        sequences.append({
            "person_id": person_id,
            "camera_id": cam_id,
            "sequence_id": f"{tracklet_id}_{i:04d}",
            "frames_dir": seq_dir,
            "num_frames": len(seq_frames)
        })
        
    return sequences

def main():
    args = parse_args()
    
    if not os.path.exists(args.raw_root):
        print(f"Error: Raw root {args.raw_root} does not exist.")
        return

    # MARS structure usually: bbox_train/ID/tracklet/*.jpg
    # We assume a structure like: raw_root/bbox_train/0001/0001C1T0001/*.jpg
    # Adjust based on actual MARS structure. 
    # Standard MARS: bbox_train/xxxx/xxxxCxsxxxx.jpg (all in one folder per ID? or subfolders?)
    # Actually MARS 'bbox_train' has subfolders per ID. Inside ID, there are images.
    # Filename format: 0001C1T0001F001.jpg -> ID, Cam, Tracklet, Frame.
    # We need to group by Tracklet.
    
    # Let's assume the user puts 'bbox_train' and 'bbox_test' inside raw_root.
    
    splits = ['bbox_train', 'bbox_test']
    all_sequences = []
    
    for split in splits:
        split_dir = os.path.join(args.raw_root, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} not found. Skipping.")
            continue
            
        print(f"Processing {split}...")
        
        # Iterate over IDs
        person_ids = sorted(os.listdir(split_dir))
        
        for pid in tqdm(person_ids):
            pid_path = os.path.join(split_dir, pid)
            if not os.path.isdir(pid_path): continue
            
            # In MARS, images are usually directly in the ID folder.
            # We need to group them by tracklet.
            # Filename: {ID}C{Cam}T{Tracklet}F{Frame}.jpg
            # Example: 0001C1T0001F001.jpg
            
            images = sorted(glob.glob(os.path.join(pid_path, "*.jpg")))
            tracklets = {}
            
            for img_path in images:
                fname = os.path.basename(img_path)
                # Parse filename
                # 0001 C1 T0001 F001 .jpg
                # ID: 0-4
                # Cam: 5-6 (C1)
                # Tracklet: 7-11 (T0001)
                # Frame: 12-15 (F001)
                
                try:
                    cam_str = fname[4:6] # C1
                    track_str = fname[6:11] # T0001
                    track_key = f"{cam_str}_{track_str}"
                    
                    if track_key not in tracklets:
                        tracklets[track_key] = []
                    tracklets[track_key].append(img_path)
                except:
                    print(f"Skipping malformed file: {fname}")
                    continue

            # Process each tracklet
            split_name = 'train' if 'train' in split else 'test'
            # Create ID dir in processed
            id_out_dir = os.path.join(args.out_root, split_name, pid)
            os.makedirs(id_out_dir, exist_ok=True)
            
            for t_key, t_frames in tracklets.items():
                # Sort frames just in case
                t_frames.sort()
                
                # Create a temp dir for the tracklet to reuse existing logic or just modify logic
                # Let's just pass the list of frames to a modified process function or handle here.
                
                # Logic to split tracklet into sequences
                num_frames = len(t_frames)
                step = args.sequence_len - args.overlap
                if step < 1: step = 1
                
                # If tracklet is too short, we might skip or keep it as one sequence (padding handled in dataset)
                if num_frames < args.sequence_len:
                    # Keep as one sequence
                    indices = [0]
                else:
                    indices = range(0, num_frames - args.sequence_len + 1, step)
                
                for i in indices:
                    if num_frames < args.sequence_len:
                        seq_frames = t_frames # Take all
                        suffix = "full"
                    else:
                        seq_frames = t_frames[i : i + args.sequence_len]
                        suffix = f"{i:04d}"
                    
                    seq_name = f"{t_key}_{suffix}"
                    seq_dir = os.path.join(id_out_dir, seq_name, "rgb")
                    os.makedirs(seq_dir, exist_ok=True)
                    
                    for j, fpath in enumerate(seq_frames):
                        shutil.copy(fpath, os.path.join(seq_dir, f"frame{j:04d}.jpg"))
                        
                    all_sequences.append({
                        "split": split_name,
                        "person_id": pid,
                        "camera_id": t_key.split('_')[0], # C1
                        "sequence_id": seq_name,
                        "frames_dir": seq_dir,
                        "num_frames": len(seq_frames)
                    })

    # Save index
    df = pd.DataFrame(all_sequences)
    df.to_csv(os.path.join(args.out_root, "index.csv"), index=False)
    print(f"Saved index to {os.path.join(args.out_root, 'index.csv')}")

if __name__ == "__main__":
    main()
