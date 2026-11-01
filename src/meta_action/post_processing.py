# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Sequence, TypedDict

import tqdm

import meta_action.utils.io as io_utils
from meta_action.data_structures.motion import MotionTags
from meta_action.utils.constant import AGENT_NAME_LEN, DELTA_TIMESTAMP, SCENE_LEN, START_TS


class MotionTagDict(TypedDict, total=False):
    """Typed representation of one serialized motion tag."""

    agents: List[str]
    interval: List[int]
    tag: str
    type: str


def get_overlap(interval1: Sequence[int], interval2: Sequence[int]) -> Optional[List[int]]:
    """Return overlap between two inclusive intervals.

    Args:
        interval1: `[start, end]` for first interval.
        interval2: `[start, end]` for second interval.

    Returns:
        Overlap as `[start, end]` if intervals overlap, otherwise `None`.
    """
    start1, end1 = interval1
    start2, end2 = interval2

    # Check for overlap
    if start1 <= end2 and end1 >= start2:
        # Calculate overlapping section
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        return [overlap_start, overlap_end]
    return None  # No overlap


def filter_scene_tags(
    scene_id_tags: Sequence[str], scene_interval: Sequence[int], trajdata_ts: float
) -> MotionTags:
    """Convert raw tag strings into structured tag objects for a scene interval.

    Args:
        scene_id_tags: Raw tag strings loaded from per-clip json.
        scene_interval: `[start_ts, end_ts]` in trajdata index space.
        trajdata_ts: Trajdata timestep in seconds.

    Returns:
        `MotionTags` containing filtered/shifted tags for the target scene interval.
    """
    if trajdata_ts <= 0:
        raise ValueError(f"trajdata_ts must be positive, got {trajdata_ts}")
    if trajdata_ts > DELTA_TIMESTAMP:
        raise ValueError(
            f"trajdata_ts ({trajdata_ts}) must be <= DELTA_TIMESTAMP ({DELTA_TIMESTAMP})"
        )

    filtered_tags: List[MotionTagDict] = []
    ts_ratio = int(DELTA_TIMESTAMP / trajdata_ts)

    for tag_str in scene_id_tags:
        tag_type = tag_str.split("(")[0].split("Temporal")[0]
        tag_s, tag_e = tag_str.split("at ")[-1].split(")")[0].split("-")

        tag_s, tag_e = int(tag_s) * ts_ratio, int(tag_e) * ts_ratio

        overlap = get_overlap((tag_s, tag_e), scene_interval)
        if overlap is None:
            continue

        is_binary = "," in tag_str

        if is_binary:
            agents = tag_str.split("(")[-1].split(" at")[0].split(", ")
        else:
            agents = [tag_str.split("(")[-1].split(" at")[0]]

        arg_type = "binary" if is_binary else "unary"

        overlap[0] -= scene_interval[0]
        overlap[1] -= scene_interval[0]

        tag: MotionTagDict = {
            "agents": agents,
            "interval": overlap,
            "tag": tag_type,
            "type": arg_type,
        }

        filtered_tags.append(tag)

    return MotionTags([filtered_tags])


def get_scene_motion_tag(
    scene_id_tags: Sequence[str], scene_dt: float, scene_start_ts: int, scene_len: int
) -> MotionTags:
    """Build scene-level motion tags from full-clip tags."""
    scene_end_ts = scene_start_ts + scene_len
    scene_interval = (scene_start_ts, scene_end_ts)
    return filter_scene_tags(scene_id_tags, scene_interval, scene_dt)


def format_to_yaml_sorted(motion_tags: Sequence[Mapping[str, Any]]) -> str:
    """Format sorted motion tags into the legacy text output format.

    Args:
        motion_tags: Sequence of tag dictionaries.

    Returns:
        Multiline string where each line is one formatted meta-action.
    """
    # Sort the tags first by starting time, then by tag name
    sorted_tags = sorted(motion_tags, key=lambda x: (x["interval"][0], x["tag"]))

    # Initialize a dictionary to hold the formatted data
    formatted_data: Dict[str, List[Dict[str, Any]]] = {}

    # Process each tag in the sorted list of motion tags
    for tag in sorted_tags:
        action = tag["tag"]
        agent = tag["agents"][0]
        start_time, end_time = tag["interval"]

        # If the action is not in the dictionary, add it
        if action not in formatted_data:
            formatted_data[action] = []

        # Append the agent and times to the action's list
        formatted_data[action].append(
            {
                "Agent": agent[:AGENT_NAME_LEN],
                "Start_time": start_time,
                "End_time": end_time,
            }
        )

        if len(tag["agents"]) > 1:
            formatted_data[action][-1]["Agent2"] = tag["agents"][1][:AGENT_NAME_LEN]

    yaml_string = ""

    for action, details in formatted_data.items():
        for detail in details:
            yaml_string += f"  {action} - "
            yaml_string += f"Agent:<{detail['Agent']}>, "
            yaml_string += f"Start:{detail['Start_time']}, "
            yaml_string += f"End:{detail['End_time']}\n"

    return yaml_string


def process_clip(clip_id: str, save_root: str, clip_id_to_file: Mapping[str, str]) -> None:
    """Post-process one clip and save final text labels."""
    file_path = clip_id_to_file[clip_id]
    with open(file_path, encoding="utf-8") as file:
        scene_id_tags = json.load(file)

    # Use clip-specific timeline length inferred from raw tags when available.
    inferred_scene_len = SCENE_LEN
    try:
        max_end = -1
        for tag_str in scene_id_tags:
            tag_e = int(tag_str.split("at ")[-1].split(")")[0].split("-")[1])
            max_end = max(max_end, tag_e)
        if max_end >= 0:
            inferred_scene_len = max(max_end + 1, START_TS + 1)
    except Exception:
        inferred_scene_len = SCENE_LEN

    scene_tags = get_scene_motion_tag(scene_id_tags, DELTA_TIMESTAMP, START_TS, inferred_scene_len)
    if len(scene_tags) == 0:
        return

    # Directly write tags without additional cleaning/merging/filling.
    formatted_output = format_to_yaml_sorted(scene_tags[0])
    save_file = os.path.join(save_root, f"{clip_id}.txt")

    with open(save_file, "w", encoding="utf-8") as file:
        file.writelines(formatted_output)


def process_batch(meta_action_dir: str, save_dir: str, num_workers: int) -> None:
    """Post-process all raw json label files under one directory.

    Args:
        meta_action_dir: Directory containing raw per-clip `.json` files.
        save_dir: Output directory for final per-clip `.txt` files.
        num_workers: Number of worker threads.
    """
    # Ensure output directory exists.
    os.makedirs(save_dir, exist_ok=True)

    # Build clip-id -> raw json path mapping.
    clip_files, _ = io_utils.load_list_from_folder(meta_action_dir)
    clip_id_to_file: Dict[str, str] = {}
    for clip_file in clip_files:
        clip_name = clip_file.replace("\\", "/").split("/")[-1].split(".")[0]
        clip_id_to_file[clip_name] = clip_file
    clip_ids = list(clip_id_to_file.keys())
    random.shuffle(clip_ids)

    # “freeze” save_dir and clip_id_to_file into the function
    worker = partial(process_clip, save_root=save_dir, clip_id_to_file=clip_id_to_file)

    # Run clip-level post-processing in parallel.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm.tqdm(executor.map(worker, clip_ids), total=len(clip_ids)))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Post-process raw meta-action JSON outputs into per-clip text files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of worker threads used to process clips in parallel.",
    )
    argparser.add_argument(
        "--raw_meta_action_dir",
        type=str,
        required=True,
        help="Directory containing raw per-clip meta-action JSON files.",
    )
    argparser.add_argument(
        "--meta_action_res_dir",
        type=str,
        required=True,
        help="Output directory for post-processed per-clip TXT files.",
    )
    args = argparser.parse_args()

    process_batch(args.raw_meta_action_dir, args.meta_action_res_dir, args.num_workers)
