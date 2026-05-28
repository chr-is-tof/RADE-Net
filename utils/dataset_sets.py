import random
import os


def split_test_file_into_parts(parts=4):
    input_path = "/home/wsluser/workspace/LRawFusion/K-Radar-main-repo/lraw_decoupeled/experiments/data_split"
    output_path = "/home/wsluser/workspace/LRawFusion/K-Radar-main-repo/lraw_decoupeled/experiments/data_split"

    files = ['test_cam_file_paths.txt',
             'test_gt_file_paths.txt',
             'test_ldr_file_paths.txt',
             'test_tess_file_paths.txt']

    for file in files:
        with open(f"{input_path}/{file}", "r") as infile:
            lines = infile.readlines()

        for i in range(0, len(lines), len(lines) // parts):
            part_lines = lines[i:i + (len(lines) // parts)]
            part_number = i // (len(lines) // parts) + 1
            tmp_path = f"{output_path}/{file.split('.')[0]}_subpart_{part_number}.txt"
            with open(tmp_path, "w") as outfile:
                outfile.writelines(part_lines)
            print(f"Wrote {len(part_lines)} lines to {tmp_path}")


def sample_from_each_subset(samples=10):
    input_path = "/mnt/z/projects/LRawFusion/K-Radar-main-repo/lraw_decoupeled/experiments/UNet/25_09_12_1351/data_split"
    output_path = "/mnt/z/projects/LRawFusion/K-Radar-main-repo/lraw_decoupeled/experiments/UNet/25_09_12_1351/data_split"

    files = ['test_cam_file_paths.txt',
             'test_gt_file_paths.txt',
             'test_ldr_file_paths.txt',
             'test_tess_file_paths.txt']

    # Read all files into lists
    file_lines = {}
    for file in files:
        with open(f"{input_path}/{file}", "r") as infile:
            file_lines[file] = infile.readlines()

    # Group indices by subset
    subset_indices = {}
    for idx, line in enumerate(file_lines[files[0]]):
        subset = line.split("/")[-3]
        subset_indices.setdefault(subset, []).append(idx + 1) # + 1 because subset is 1-indexed

    # For each subset, sample indices and write the same lines for all files
    cam_file = []
    gt_file = []
    ldr_file = []
    tess_file = []
    for subset, indices in subset_indices.items():
        sampled_indices = random.sample(indices, min(samples, len(indices)))

        cam_file.extend([file_lines['test_cam_file_paths.txt'][i] for i in sampled_indices])
        gt_file.extend([file_lines['test_gt_file_paths.txt'][i] for i in sampled_indices])
        ldr_file.extend([file_lines['test_ldr_file_paths.txt'][i] for i in sampled_indices])
        tess_file.extend([file_lines['test_tess_file_paths.txt'][i] for i in sampled_indices])

    lists = [cam_file, gt_file, ldr_file, tess_file]
    for i, file in enumerate(files):
        out_path = f"{output_path}/{file.split('.')[0]}_sampled.txt"
        with open(out_path, "w") as outfile:
            outfile.writelines(lists[i])
        print(f"Wrote {len(lists[i])} lines to {out_path}")


def split_off_val_set(cfg, val_percent_split=0.2):
    input_path = "/home/wsluser/workspace/RADE-Net_private/experiments/train.txt"
    val_output_path = "/home/wsluser/workspace/RADE-Net_private/experiments/val.txt"
    train_output_path = "/home/wsluser/workspace/RADE-Net_private/experiments/train_reduced.txt"

    with open(input_path, "r") as in_file:
        train_split = [line.strip() for line in in_file]

    if cfg is not None:
        path_labels = cfg.PATH_TO_LABELS
    else:
        path_labels = "/home/wsluser/workspace/RADE-Net_private/labels/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility"

    # Count objects per class in each file and in total
    file_class_counts = []
    total_class_counts = {}
    for line in train_split:
        subset = line.split(",")[0]
        gt = line.split(",")[1]
        gt_file_path = f"{path_labels}/{subset}/{gt}"
        
        if not os.path.exists(gt_file_path):
            print(f"File does not exist: {gt_file_path}")
        
        class_counts = {}
        with open(gt_file_path, "r") as gt_file:
            lines = gt_file.readlines()

        for l in lines[1:]:  # Skip header line
            list_vals = l.rstrip('\n').split(', ')
            avail = list_vals[1]
            cls_name = (list_vals[3])

            # Only counts radar ground truths
            if avail != "R":
                continue

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            total_class_counts[cls_name] = total_class_counts.get(cls_name, 0) + 1
        file_class_counts.append((line, class_counts))
 
    # Compute the specified percentage for validation targets
    val_targets = {cls: int(val_percent_split * count) for cls, count in total_class_counts.items()}
    val_counts = {cls: 0 for cls in total_class_counts}
    val_lines = []
    train_lines = []

    # Randomly shuffle files to to achieve a proper distribution
    random.shuffle(file_class_counts)

    # Greedily match files to reach the target counts
    for line, class_counts in file_class_counts:
        # Check if adding this file would help reach the target for any class
        add_to_val = False
        for cls, count in class_counts.items():
            if val_counts[cls] < val_targets[cls]:
                add_to_val = True
                break
        if add_to_val:
            val_lines.append(line)
            for cls, count in class_counts.items():
                val_counts[cls] += count
        else:
            train_lines.append(line)

    # Computes metrics
    cls_ratios = {cls: round(val_counts[cls] / total_class_counts[cls], 4) if total_class_counts[cls] > 0 else 0 for cls in total_class_counts}
    cls_ratios_absolute_difference = {cls: abs(val_counts[cls] - val_targets[cls]) for cls in total_class_counts}

    # Sort lines to maintain original order
    val_lines = sorted(
        val_lines,
        key=lambda x: (
            int(x.split(",")[0]),
            int(x.split(",")[1].split("_")[0])
        )
    )
    train_lines = sorted(
        train_lines,
        key=lambda x: (
            int(x.split(",")[0]),
            int(x.split(",")[1].split("_")[0])
        )
    )

    # Saves splits
    with open(val_output_path, "w") as val_file:
        val_file.writelines([l + "\n" for l in val_lines])

    with open(train_output_path, "w") as train_file:
        train_file.writelines([l + "\n" for l in train_lines])

    print("--- Split Summary ---\n")
    print(f"Total class counts in the train set:\n{total_class_counts}\n")
    print(f"Target count based on the validation split of {val_percent_split * 100}%:\n{val_targets}\n")
    print(f"Greedily matched count:\n{val_counts}\n")
    print(f"Absolute difference between target and achieved count:\n{cls_ratios_absolute_difference}\n")
    print(f"Achieved ratio per class:\n{cls_ratios}\n")
    print(f"Val set: {len(val_lines)} files, Train set: {len(train_lines)} files")


if __name__ == "__main__":
    split_off_val_set(None, val_percent_split=0.2)