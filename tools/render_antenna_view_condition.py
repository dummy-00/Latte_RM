import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_ids(ids):
    parsed = []
    for value in ids:
        for part in value.split(","):
            part = part.strip()
            if part:
                parsed.append(part)
    return parsed


def natural_json_ids(root):
    ids = []
    for path in Path(root).glob("*.json"):
        ids.append(path.stem)
    return sorted(ids, key=lambda item: int(item) if item.isdigit() else item)


def natural_key(text):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def select_antennas(antennas, antenna_index, use_all_antennas):
    if use_all_antennas:
        return antennas
    if not antennas:
        return []
    if antenna_index < 0 or antenna_index >= len(antennas):
        raise IndexError(f"antenna-index {antenna_index} is out of range for {len(antennas)} antennas")
    return [antennas[antenna_index]]


def antenna_jobs(antennas, antenna_index, use_all_antennas, per_antenna):
    if per_antenna:
        return [(index, [point]) for index, point in enumerate(antennas)]
    if use_all_antennas:
        return [("all", antennas)]
    selected = select_antennas(antennas, antenna_index, use_all_antennas)
    return [(antenna_index, selected)]


def rasterize_buildings(buildings, image_size, height_count):
    occupancy = np.zeros((height_count, image_size, image_size), dtype=bool)
    for z in range(1, height_count + 1):
        img = Image.new("L", (image_size, image_size), 0)
        draw = ImageDraw.Draw(img)
        for polygon, height_values in buildings:
            if not height_values:
                continue
            if float(height_values[0]) < z:
                continue
            draw.polygon([tuple(point) for point in polygon], fill=255)
        occupancy[z - 1] = np.asarray(img, dtype=np.uint8) > 0
    return occupancy


def mark_antenna_slice(mask, antennas, antenna_radius):
    draw = ImageDraw.Draw(mask)
    radius = max(0, antenna_radius)
    for point in antennas:
        if len(point) < 2:
            continue
        x, y = float(point[0]), float(point[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)


def boundary_targets(width, height, height_count):
    targets = []
    for z in range(1, height_count + 1):
        for x in range(width):
            targets.append((x, 0, z))
            targets.append((x, height - 1, z))
        for y in range(1, height - 1):
            targets.append((0, y, z))
            targets.append((width - 1, y, z))
    return targets


def trace_visible_ray(occupancy, visible, start, end):
    x0, y0, z0 = start
    x1, y1, z1 = end
    steps = int(max(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)))
    if steps <= 0:
        return

    height_count, height, width = occupancy.shape
    last = None
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        z = int(round(z0 + (z1 - z0) * t))
        point = (x, y, z)
        if point == last:
            continue
        last = point
        if x < 0 or x >= width or y < 0 or y >= height or z < 1 or z > height_count:
            continue
        if occupancy[z - 1, y, x]:
            return
        visible[z - 1, y, x] = True


def render_visibility_slices(occupancy, antennas, antenna_height, antenna_radius):
    height_count, image_h, image_w = occupancy.shape
    visible = np.zeros_like(occupancy, dtype=bool)
    antenna_z = int(round(antenna_height))
    antenna_z = min(max(antenna_z, 1), height_count)
    starts = []
    for point in antennas:
        if len(point) < 2:
            continue
        starts.append((int(round(point[0])), int(round(point[1])), antenna_z))

    targets = boundary_targets(image_w, image_h, height_count)
    for start in starts:
        if start[0] < 0 or start[0] >= image_w or start[1] < 0 or start[1] >= image_h:
            continue
        for target in targets:
            trace_visible_ray(occupancy, visible, start, target)

    if 1 <= antenna_z <= height_count:
        img = Image.fromarray((visible[antenna_z - 1].astype(np.uint8) * 255), mode="L")
        mark_antenna_slice(img, antennas, antenna_radius)
        visible[antenna_z - 1] = np.asarray(img, dtype=np.uint8) > 0

    return visible


def render_occupancy_slices(occupancy, antennas, antenna_height, antenna_radius):
    result = occupancy.copy()
    antenna_z = int(round(antenna_height))
    if 1 <= antenna_z <= occupancy.shape[0]:
        img = Image.fromarray((result[antenna_z - 1].astype(np.uint8) * 255), mode="L")
        mark_antenna_slice(img, antennas, antenna_radius)
        result[antenna_z - 1] = np.asarray(img, dtype=np.uint8) > 0
    return result


def save_slices(slices, output_dir, invert):
    os.makedirs(output_dir, exist_ok=True)
    for index, mask in enumerate(slices, start=1):
        image = mask.astype(np.uint8) * 255
        if invert:
            image = 255 - image
        Image.fromarray(image, mode="L").save(os.path.join(output_dir, f"h{index}.png"))


def slices_complete(output_dir, height_count):
    return all(os.path.isfile(os.path.join(output_dir, f"h{index}.png")) for index in range(1, height_count + 1))


def render_slices_for_antennas(args, occupancy, antennas):
    if args.mode == "occupancy":
        return render_occupancy_slices(
            occupancy,
            antennas,
            args.antenna_height,
            args.antenna_radius,
        )
    return render_visibility_slices(
        occupancy,
        antennas,
        args.antenna_height,
        args.antenna_radius,
    )


def render_radiomap_sample(args, occupancy_cache, sample_name, condition_id, antenna_point):
    output_dir = os.path.join(args.output_root, sample_name)
    if args.skip_existing and slices_complete(output_dir, args.height_count):
        return output_dir, "skipped"

    if condition_id not in occupancy_cache:
        building_path = os.path.join(args.buildings_root, f"{condition_id}.json")
        if not os.path.isfile(building_path):
            raise FileNotFoundError(f"Missing building json: {building_path}")
        buildings = load_json(building_path)
        occupancy_cache[condition_id] = rasterize_buildings(
            buildings,
            args.image_size,
            args.height_count,
        )
    slices = render_slices_for_antennas(args, occupancy_cache[condition_id], [antenna_point])
    save_slices(slices, output_dir, args.invert)
    return output_dir, "saved"


def worker_render_radiomap(payload):
    args_dict, sample = payload
    args = argparse.Namespace(**args_dict)
    sample_name, condition_id, antenna_point = sample
    output_dir, status = render_radiomap_sample(args, {}, sample_name, condition_id, antenna_point)
    return sample_name, output_dir, status


def parse_radiomap_name(name):
    match = re.match(r"^(?P<condition_id>[^_]+)_X(?P<x>-?\d+)_Y(?P<y>-?\d+)$", name)
    if match is None:
        return None
    return match.group("condition_id"), [int(match.group("x")), int(match.group("y"))]


def radiomap_samples(radiomap_root):
    samples = []
    for path in Path(radiomap_root).iterdir():
        if not path.is_dir():
            continue
        parsed = parse_radiomap_name(path.name)
        if parsed is None:
            continue
        condition_id, antenna_point = parsed
        samples.append((path.name, condition_id, antenna_point))
    return sorted(samples, key=lambda item: natural_key(item[0]))


def render_radiomap_aligned(args):
    if args.radiomap_root is None:
        return False

    samples = radiomap_samples(args.radiomap_root)
    if args.limit is not None:
        samples = samples[: args.limit]

    if args.num_workers <= 1:
        occupancy_cache = {}
        for index, (sample_name, condition_id, antenna_point) in enumerate(samples, start=1):
            output_dir, status = render_radiomap_sample(
                args,
                occupancy_cache,
                sample_name,
                condition_id,
                antenna_point,
            )
            if index == 1 or index % args.log_every == 0 or index == len(samples):
                print(f"{status} {index}/{len(samples)} {sample_name}: {output_dir}", flush=True)
        return True

    args_dict = vars(args).copy()
    payloads = [(args_dict, sample) for sample in samples]
    with mp.Pool(args.num_workers) as pool:
        for index, (sample_name, output_dir, status) in enumerate(
            pool.imap_unordered(worker_render_radiomap, payloads),
            start=1,
        ):
            if index == 1 or index % args.log_every == 0 or index == len(samples):
                print(f"{status} {index}/{len(samples)} {sample_name}: {output_dir}", flush=True)
    return True


def render_one(args, condition_id):
    building_path = os.path.join(args.buildings_root, f"{condition_id}.json")
    antenna_path = os.path.join(args.antenna_root, f"{condition_id}.json")
    if not os.path.isfile(building_path):
        raise FileNotFoundError(f"Missing building json: {building_path}")
    if not os.path.isfile(antenna_path):
        raise FileNotFoundError(f"Missing antenna json: {antenna_path}")

    buildings = load_json(building_path)
    antennas = load_json(antenna_path)
    occupancy = rasterize_buildings(buildings, args.image_size, args.height_count)

    output_dirs = []
    for antenna_label, selected_antennas in antenna_jobs(
        antennas,
        args.antenna_index,
        args.use_all_antennas,
        args.per_antenna,
    ):
        slices = render_slices_for_antennas(args, occupancy, selected_antennas)
        if args.per_antenna:
            output_dir = os.path.join(args.output_root, condition_id, f"antenna_{antenna_label:03d}")
        elif antenna_label == "all":
            output_dir = os.path.join(args.output_root, condition_id, "all_antennas")
        else:
            output_dir = os.path.join(args.output_root, condition_id)
        save_slices(slices, output_dir, args.invert)
        output_dirs.append(output_dir)
    return output_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Render per-height 2D antenna view slices from building geometry and antenna points."
    )
    parser.add_argument("--buildings-root", default="/home/plj/buildings")
    parser.add_argument("--antenna-root", default="/home/plj/antenna")
    parser.add_argument("--output-root", default="/home/plj/antennaView")
    parser.add_argument("--radiomap-root", default=None, help="Render folders aligned to radiomap sample names, e.g. /path/train with id_Xx_Yy folders.")
    parser.add_argument("--ids", nargs="*", default=None, help="Ids to render, e.g. 0 1 2 or 0,1,2.")
    parser.add_argument("--limit", type=int, default=None, help="Render only the first N ids when --ids is omitted.")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--skip-existing", action="store_true", help="Skip output folders that already contain h1..hN.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--height-count", type=int, default=20)
    parser.add_argument("--antenna-height", type=float, default=1.0)
    parser.add_argument("--antenna-index", type=int, default=0, help="Antenna point index to render when not using all antennas.")
    parser.add_argument("--use-all-antennas", action="store_true", help="Use the visibility union from every antenna point.")
    parser.add_argument("--per-antenna", action="store_true", help="Render one h1..hN slice folder for every antenna point.")
    parser.add_argument("--antenna-radius", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=("visibility", "occupancy"),
        default="visibility",
        help="visibility renders line-of-sight free space; occupancy renders building/antenna slices.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert black/white output.")
    args = parser.parse_args()

    if render_radiomap_aligned(args):
        return

    ids = parse_ids(args.ids) if args.ids else natural_json_ids(args.buildings_root)
    if args.limit is not None:
        ids = ids[: args.limit]

    for condition_id in ids:
        output_dirs = render_one(args, condition_id)
        print(f"saved {condition_id}: {len(output_dirs)} antenna view folder(s)")


if __name__ == "__main__":
    main()
