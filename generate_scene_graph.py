import os
import numpy as np
import shutil

def run_sgg(src_path, dst_path):
    device = 7
    image_path = src_path
    output_path = dst_path
    
    os.system(f"bash run_eval.sh {device} {image_path} {output_path}")

def get_sequence(src_dir, seq_len=10):
    frame_names = os.listdir(src_dir)
    start_idx = 0
    end_idx = len(frame_names) - 1

    sampling_sequence = list(np.linspace(start_idx, end_idx, seq_len, dtype=np.int)) # [start_idx, end_idx]
    return sampling_sequence

def copy_frames(src_dir, dst_dir, sequence):
    for idx in sequence:
        shutil.copy(f"{src_dir}/frame_{idx:05d}.jpg", f"{dst_dir}/frame_{idx:05d}.jpg")

def main():
    # Source path of extracted frames.
    base_src_path = "/home/aimaster/lab_storage/skim/scene_graphs/zero-shot-action-recognition/data_engineering/frames"
    # Destination path for scene_graphs.
    base_dst_path = "/home/aimaster/lab_storage/skim/scene_graphs/zero-shot-action-recognition/data_engineering/scene_graph/"
    # Name of dataset.
    dataset_name = "OlympicSports"
    
    src_dataset_path = os.path.join(base_src_path, dataset_name)
    dst_dataset_path = os.path.join(base_dst_path, dataset_name)
    
    # Create directory at the path. ex) ..../scene_graph/OlympicSports
    if not os.path.exists(dst_dataset_path):
        os.mkdir(dst_dataset_path)

    # Action names like basketball_layup, bowling, clean-and-jerk...
    action_names = os.listdir(src_dataset_path)

    for action_name in action_names:
        src_video_path = os.path.join(src_dataset_path, action_name)
        dst_video_path = os.path.join(dst_dataset_path, action_name)

        # Create directory at the path. ex) ..../scene_graph/OlympicSports/basketball_layup
        if not os.path.exists(dst_video_path):
            os.mkdir(dst_video_path)

        video_names = os.listdir(src_video_path)

        for video_name in video_names:
            src_frame_path = os.path.join(src_video_path, video_name)
            dst_frame_path = os.path.join(dst_video_path, video_name)

            # Create directory at the path. ex) ..../scene_graph/OlympicSports/basketball_layup/8eGmno0lZKM_00001_00169
            if not os.path.exists(dst_frame_path):
                os.mkdir(dst_frame_path)

            # Get sequence per a length of video and a length of sequence.
            sequence = get_sequence(src_dir=src_frame_path, seq_len=10)

            # Move per sampled frames to the destination directory.
            copy_frames(src_dir=src_frame_path, dst_dir=dst_frame_path, sequence=sequence)
            
            # Run Scene Graph Generator.
            run_sgg(src_path=dst_frame_path, dst_path=dst_frame_path)

if __name__ == "__main__":
    main()