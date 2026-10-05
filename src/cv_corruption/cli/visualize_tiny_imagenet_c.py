"""Audit Tiny ImageNet-C and produce paired galleries and a Chinese report."""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


CV_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = CV_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from cv_corruption.config.loader import parse_config, save_config
from cv_corruption.visualization.galleries import validate_groups
GROUPS = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}


def save_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def font(args, size):
    return ImageFont.truetype(str(args.font_path) if args.font_path else "DejaVuSans.ttf", size)


def draw_text(draw, xy, text, args, size=None, max_width=None):
    size = size or args.font_size
    face = font(args, size)
    while max_width and draw.textbbox((0, 0), text, font=face)[2] > max_width and size > 8:
        size -= 1
        face = font(args, size)
    draw.text(xy, text, fill=args.text_color, font=face)


def paste_image(canvas, path, xy, args):
    size = args.tile_size
    with Image.open(path) as image:
        canvas.paste(image.convert("RGB").resize((size, size), getattr(Image.Resampling, args.interpolation.upper())), xy)
    x, y = xy
    if args.border_width:
        ImageDraw.Draw(canvas).rectangle((x, y, x + size - 1, y + size - 1),
                                         outline=args.border_color, width=args.border_width)


def save_gallery(canvas, directory, stem, args):
    """Save each display figure in both screen and publication formats."""
    if args.save_png:
        path = args.output / "images" / "png" / directory.name / f"{stem}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, dpi=(args.dpi, args.dpi))
    if args.save_pdf:
        figure = plt.figure(figsize=(canvas.width / args.dpi, canvas.height / args.dpi),
                            dpi=args.dpi, frameon=False)
        axis = figure.add_axes((0, 0, 1, 1))
        axis.imshow(canvas)
        axis.set_axis_off()
        path = args.output / "images" / "pdf" / directory.name / f"{stem}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=args.dpi, transparent=args.transparent,
                       bbox_inches=None, pad_inches=0)
        plt.close(figure)


def new_canvas(size, args):
    """Create an RGB canvas when opaque output is requested, otherwise RGBA."""
    if args.transparent:
        return Image.new("RGBA", size, (0, 0, 0, 0))
    return Image.new("RGB", size, args.background)


def scan(data, clean, args):
    words = dict(line.rstrip().split("\t", 1) for line in (clean / "words.txt").read_text().splitlines())
    expected_labels = set((clean / "wnids.txt").read_text().splitlines())
    slices, class_rows, labels, identity, examples = [], [], None, None, {}
    byte_total = block_total = checked = 0
    sizes, modes = Counter(), Counter()
    corruptions = [name for group in args.groups.values() for name in group]
    for corruption in corruptions:
        severity_dirs = [str(level) for level in args.severities]
        for severity in severity_dirs:
            directory = data / corruption / severity
            current_labels = {path.name for path in directory.iterdir() if path.is_dir()}
            assert current_labels == expected_labels, (corruption, severity, "label mismatch")
            labels = current_labels
            current_identity = {}
            count = 0
            for label in sorted(labels):
                entries = sorted(os.scandir(directory / label), key=lambda entry: entry.name)
                images = [entry for entry in entries if entry.is_file() and entry.name.lower().endswith(".jpeg")]
                assert len(images) == 50, (corruption, severity, label, len(images))
                class_rows.append({"corruption": corruption, "severity": int(severity), "label": label, "count": len(images)})
                for entry in images:
                    assert entry.name not in current_identity, entry.name
                    current_identity[entry.name] = label
                    metadata = entry.stat()
                    byte_total += metadata.st_size
                    block_total += metadata.st_blocks * 512
                if args.sample_index >= len(images):
                    raise ValueError("sample_index exceeds the per-class image count")
                sample = Path(images[args.sample_index].path)
                with Image.open(sample) as image:
                    image.load()
                    assert image.size == (64, 64), sample
                    sizes[str(image.size)] += 1
                    modes[image.mode] += 1
                checked += 1
                examples.setdefault(label, images[args.sample_index].name)
                count += len(images)
            assert count == 10000, (corruption, severity, count)
            if identity is None:
                identity = current_identity
            else:
                assert current_identity == identity, (corruption, severity, "sample/label mismatch")
            slices.append({"corruption": corruption, "severity": int(severity), "count": count})
        print(f"Audited {corruption}: {len(args.severities)} severities, {10000 * len(args.severities):,} images", flush=True)

    clean_files = {path.name for path in (clean / "test/images").glob("*.JPEG")}
    assert set(identity) == clean_files, "Clean test filenames do not match corruptions"
    clean_counts = {split: sum(1 for _ in (clean / split).rglob("*.JPEG")) for split in ("train", "val", "test")}
    clean_bytes = clean_blocks = 0
    for path in clean.rglob("*"):
        if path.is_file():
            metadata = path.stat()
            clean_bytes += metadata.st_size
            clean_blocks += metadata.st_blocks * 512
    label_rows = [{"label": label, "name": words[label], "per_condition": 50,
                   "total_corrupted": 50 * len(corruptions) * len(args.severities)} for label in sorted(labels)]
    summary = {
        "corruption_types": corruptions, "severity_levels": args.severities,
        "label_count": len(labels), "total_images": sum(row["count"] for row in slices),
        "unique_base_images": len(identity), "images_per_condition": 10000,
        "images_per_label_per_condition": 50, "jpeg_bytes": byte_total,
        "jpeg_allocated_bytes": block_total, "clean_counts": clean_counts,
        "clean_file_bytes": clean_bytes, "clean_allocated_bytes": clean_blocks,
        "decoded_samples": checked, "decoded_dimensions": dict(sizes), "decoded_modes": dict(modes),
        "clean_filename_set_match": True, "same_filename_is_same_image": False,
        "labels_consistent_across_conditions": True,
        "tiny_c_archive_bytes": 1794775040, "tiny_c_archive_md5": "f9c9a9dbdc11469f0b850190f7ad8be1",
        "clean_archive_bytes": 248100043, "clean_archive_md5_local": "90528d7ca1a48142e341f4ef8d21d0de",
    }
    return summary, slices, class_rows, label_rows, examples, identity


def thumbnail(path, size):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32).ravel()


def galleries(data, clean, output, label_rows, examples, args):
    names = {row["label"]: row["name"] for row in label_rows}
    label = args.selected_label
    if label not in examples:
        raise ValueError(f"Unknown selected_label: {label}")
    labels = sorted(examples)
    if args.sample_selection == "every_10th":
        selected = labels[::args.class_stride][:args.class_count]
    elif args.sample_selection == "random":
        selected = sorted(np.random.default_rng(args.seed).choice(
            labels, size=min(args.class_count, len(labels)), replace=False).tolist())
    elif args.sample_selection == "all":
        selected = labels
    else:
        selected = labels[:args.class_count]
    query_labels = sorted(set((selected if args.generate_classes else []) + [label]))
    queries = [thumbnail(data / args.match_corruption / str(args.match_severity) / item / examples[item],
                         args.match_size) for item in query_labels]
    candidates = sorted((clean / "test/images").glob("*.JPEG"))
    if len(candidates) < 2:
        raise ValueError("Need at least two clean images for content matching")
    pixels = np.stack([thumbnail(path, args.match_size) for path in candidates])
    matches, match_rows = {}, []
    for item, query in zip(query_labels, queries):
        errors = np.mean((pixels - query) ** 2, axis=1)
        ranked = np.argsort(errors)
        matches[item] = candidates[ranked[0]]
        suffix = f"{args.match_size}x{args.match_size}"
        match_rows.append({"label": item, "corrupted_filename": examples[item],
                           "clean_filename": matches[item].name,
                           f"best_mse_{suffix}": float(errors[ranked[0]]),
                           f"second_best_mse_{suffix}": float(errors[ranked[1]])})
    if args.write_metadata:
        save_csv(output / "metadata/clean_matches.csv", list(match_rows[0]), match_rows)

    all_corruptions = [name for group in args.groups.values() for name in group]
    image_name, clean_path = examples[label], matches[label]
    header = args.title_font_size + args.font_size * 2 + 44 if args.show_title else args.font_size + 30
    left = args.row_label_width
    step = args.tile_size + args.tile_gap
    row_step = args.tile_size + args.row_gap
    width = left + (1 + len(args.severities)) * step + 20
    tasks = []
    if args.generate_overview:
        tasks.append(("overview", "all", all_corruptions,
                      f"{len(all_corruptions)} corruptions"))
    if args.generate_groups:
        tasks.extend(("groups", group, values, group) for group, values in args.groups.items())
    for directory, stem, corruptions, title in tasks:
        canvas = new_canvas((width, header + row_step * len(corruptions) + 20), args)
        draw = ImageDraw.Draw(canvas)
        if args.show_title:
            draw_text(draw, (args.title_x, args.title_y),
                      f"Tiny ImageNet-C | {title} | {label} | {names[label].split(',')[0]}",
                      args, args.title_font_size, max_width=width - 40)
            draw_text(draw, (args.title_x, args.title_y + args.title_font_size),
                      f"Clean {clean_path.name} / corrupted {image_name}", args, max_width=width - 40)
        for column, text in enumerate(["Clean"] + [f"Severity {s}" for s in args.severities]):
            draw_text(draw, (left + column * step, header - args.font_size - 12), text,
                      args, max_width=args.tile_size)
        for row, corruption in enumerate(corruptions):
            y = header + row * row_step
            for line, word in enumerate(corruption.split("_")):
                draw_text(draw, (18, y + 20 + line * (args.font_size + 4)), word,
                          args, max_width=left - 30)
            paste_image(canvas, clean_path, (left, y), args)
            for column, severity in enumerate(args.severities, 1):
                paste_image(canvas, data / corruption / str(severity) / label / image_name,
                            (left + column * step, y), args)
        save_gallery(canvas, output / directory, stem, args)

    if args.generate_classes:
        columns = min(args.class_columns, len(selected))
        rows = (len(selected) + columns - 1) // columns
        class_step = args.tile_size + max(args.tile_gap, 30)
        class_row_step = args.tile_size + args.font_size * 4 + 28
        class_header = args.title_font_size + 35 if args.show_title else 20
        canvas = new_canvas((20 + columns * class_step, class_header + rows * class_row_step), args)
        draw = ImageDraw.Draw(canvas)
        if args.show_title:
            draw_text(draw, (args.title_x, args.title_y),
                      f"{len(selected)} of {len(labels)} labels | matched clean test images",
                      args, args.title_font_size, max_width=canvas.width - 40)
        for index, item in enumerate(selected):
            x, y = 20 + (index % columns) * class_step, class_header + (index // columns) * class_row_step
            paste_image(canvas, matches[item], (x, y), args)
            draw_text(draw, (x, y + args.tile_size + 6), item, args, max_width=args.tile_size)
            title = names[item].split(",")[0]
            lines = [""]
            for word in title.split():
                trial = (lines[-1] + " " + word).strip()
                if draw.textbbox((0, 0), trial, font=font(args, args.font_size))[2] > args.tile_size and lines[-1]:
                    lines.append(word)
                else:
                    lines[-1] = trial
            for line_number, line in enumerate(lines[:2]):
                draw_text(draw, (x, y + args.tile_size + 12 + args.font_size * (line_number + 1)),
                          line, args, max_width=args.tile_size)
        save_gallery(canvas, output / "classes", "labels", args)
    if args.write_manifest:
        manifest = [{"corruption": name, "severity": severity, "label": label, "filename": image_name,
                     "clean": str(clean_path.resolve()),
                     "corrupted": str((data / name / str(severity) / label / image_name).resolve())}
                    for name in all_corruptions for severity in args.severities]
        (output / "metadata/sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def report(summary, slices, args):
    s = summary
    lines = [
        "# Tiny ImageNet-C 数据集展示", "",
        "本报告仅统计配置中选定的 corruption 与 severity，来自本次本地文件扫描。", "",
        "| 项目 | 实测信息 |", "| --- | --- |",
        f"| 类别数 | {s['label_count']} |",
        f"| 损坏种类数 | {len(s['corruption_types'])} |",
        f"| 强度等级 | {s['severity_levels']} |",
        f"| JPEG 总数 | {s['total_images']:,} |",
        f"| 唯一基础图像 | {s['unique_base_images']:,} |",
        f"| 每个损坏与强度条件 | {s['images_per_condition']:,} 张 |",
        f"| 解码抽检数 | {s['decoded_samples']:,} 张 |",
        f"| 图像尺寸 | {s['decoded_dimensions']} |",
        f"| JPEG 文件内容 | {s['jpeg_bytes']:,} 字节 |",
        f"| JPEG 已分配磁盘块 | {s['jpeg_allocated_bytes']:,} 字节 |", "",
        "每个条件每类解码一张图像，并非对所有图像逐像素解码。类别标签来自目录的 WordNet ID。", "",
        "## 损坏数量", "", "| 损坏 | 选定强度合计 |", "| --- | ---: |",
    ]
    for name in s["corruption_types"]:
        lines.append(f"| {name} | {sum(row['count'] for row in slices if row['corruption'] == name):,} |")
    lines += ["", "## clean 配对", "",
              "clean test 同名文件不保证与损坏图像内容相同。展示图通过内容检索配对；它不是官方 clean test 标签，也不是性能评测配对。",
              f"匹配使用 {args.match_corruption} / severity {args.match_severity} 的 {args.match_size} x {args.match_size} RGB 均方误差，需人工复核。", ""]
    paths = []
    if args.generate_overview:
        paths.append(("损坏总览", "overview/all"))
    if args.generate_groups:
        paths.extend((group, f"groups/{group}") for group in args.groups)
    if args.generate_classes:
        paths.append(("类别样例", "classes/labels"))
    for title, stem in paths:
        for extension, enabled in (("png", args.save_png), ("pdf", args.save_pdf)):
            if enabled:
                relative = Path(os.path.relpath(args.output / "images" / extension / f"{stem}.{extension}", args.report_path.parent)).as_posix()
                lines.extend([f"![{title}]({relative})" if extension == "png" else f"[{title} PDF]({relative})", ""])
    lines += ["## 复现", "", "运行参数保存在输出目录的 resolved_config.json。",
              "统计 CSV 与展示样本清单受 write_metadata / write_manifest 配置控制。",
              "本脚本没有下载、删除归档或重新验证归档 MD5。", "",
              "来源：[官方项目](https://github.com/hendrycks/robustness)、[Tiny ImageNet-C](https://zenodo.org/records/2536630)。", ""]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CV_ROOT / "data/raw/tiny_imagenet_c/Tiny-ImageNet-C")
    parser.add_argument("--clean", type=Path, default=CV_ROOT / "data/raw/tiny_imagenet_c/tiny-imagenet-200")
    parser.add_argument("--output", type=Path, default=CV_ROOT / "runs/visualizations/tiny_imagenet_c")
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--selected-label", default="n01443537")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-selection", choices=["every_10th", "first", "random", "all"], type=str, default="every_10th")
    parser.add_argument("--class-stride", type=int, default=10)
    parser.add_argument("--class-count", type=int, default=20)
    parser.add_argument("--class-columns", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--tile-gap", type=int, default=8)
    parser.add_argument("--row-gap", type=int, default=36)
    parser.add_argument("--row-label-width", type=int, default=200)
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--title-font-size", type=int, default=22)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--text-color", default="#24282d")
    parser.add_argument("--background", default="white")
    parser.add_argument("--transparent", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a transparent canvas for PNG and PDF output")
    parser.add_argument("--title-x", type=int, default=20)
    parser.add_argument("--title-y", type=int, default=10)
    parser.add_argument("--border-color", default="#3A3D42")
    parser.add_argument("--border-width", type=int, default=3)
    parser.add_argument("--interpolation", type=str, choices=["nearest", "bilinear", "bicubic", "lanczos"], default="nearest")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--match-corruption", default="jpeg_compression", type=str,
                        choices=[name for group in GROUPS.values() for name in group])
    parser.add_argument("--match-severity", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--match-size", type=int, default=16)
    for name in ("save-png", "save-pdf", "show-title", "write-metadata", "write-manifest", "write-report",
                 "generate-overview", "generate-groups", "generate-classes"):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(groups=GROUPS)
    args = parse_config(parser, argv, default_config=CV_ROOT / "configs/visualization/tiny_imagenet_c.yaml",
                        path_fields=("data", "clean", "output", "font_path", "report_path"))
    names = validate_groups(args.groups, args.severities)
    if any(name not in {name for group in GROUPS.values() for name in group} for name in names):
        parser.error("groups contains an unsupported Tiny ImageNet-C corruption")
    positive = (args.class_stride, args.class_count, args.class_columns, args.tile_size,
                args.row_label_width, args.font_size, args.title_font_size, args.dpi, args.match_size)
    if any(value <= 0 for value in positive) or any(value < 0 for value in (args.sample_index, args.tile_gap, args.row_gap, args.border_width)):
        parser.error("Sizes, font sizes, DPI and class sampling settings must be positive; indices/gaps non-negative")
    if args.border_width > args.tile_size // 2:
        parser.error("border_width must be smaller than half the tile size")
    if not (args.save_png or args.save_pdf):
        parser.error("Enable save_png or save_pdf")
    if args.seed < 0:
        parser.error("seed must be non-negative")
    args.report_path = args.report_path or args.output / "reports/report.md"
    return args


def main():
    args = parse_args()
    output = args.output
    directories = {name: output / name for name in ("overview", "groups", "classes", "statistics", "metadata")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    summary, slices, class_rows, labels, examples, identity = scan(args.data, args.clean, args)
    if args.write_metadata:
        (directories["metadata"] / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        save_csv(directories["metadata"] / "condition_counts.csv", ["corruption", "severity", "count"], slices)
        save_csv(directories["metadata"] / "class_counts.csv", ["corruption", "severity", "label", "count"], class_rows)
        save_csv(directories["metadata"] / "labels.csv", ["label", "name", "per_condition", "total_corrupted"], labels)
    totals = [{"corruption": name, "count": sum(row["count"] for row in slices if row["corruption"] == name)} for name in summary["corruption_types"]]
    if args.write_metadata:
        save_csv(directories["statistics"] / "corruption_totals.csv", ["corruption", "count"], totals)
    if args.generate_overview or args.generate_groups or args.generate_classes or args.write_manifest:
        galleries(args.data, args.clean, output, labels, examples, args)
    if args.write_report:
        report(summary, slices, args)
    save_config(output / "resolved_config.json", args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
