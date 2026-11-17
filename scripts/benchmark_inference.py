import argparse
from pathlib import Path
import time
from PIL import Image
import statistics
import numpy as np

from app.api.inference import PyTorchInference


def ranged_type(data_type, min_val, max_val):
    """Provides range value checks for passed datatypes."""
    def range_check(arg):
        try:
            val = data_type(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"must be a valid {data_type}")

        if val < min_val or val > max_val:
            raise argparse.ArgumentTypeError(f"must be in range ({min_val}, {max_val})")
        return val

    return range_check


def parse_args():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Whatdog Benchmark Inference")

    parser.add_argument(
        "--image-path",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "Images"
        / "n02110958-pug"
        / "n02110958_326.jpg",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--inference", choices=["pytorch", "onnx"], default="pytorch")
    parser.add_argument(
        "--class_names_path",
        type=Path,
        default=PROJECT_ROOT / "app" / "training" / "artifacts" / "class_names.json",
    )
    parser.add_argument("--warmup-runs", type=ranged_type(int, 0, 1000), default=20)
    parser.add_argument("--measured-runs", type=ranged_type(int, 1, 10000), default=200)

    return parser.parse_args()


def main():
    print("===== INFERENCE BENCHMARK =====")
    args = parse_args()

    # Switch loaded inference class based on args
    if args.inference == "pytorch":
        inference = PyTorchInference(
            checkpoint_path=args.model_path, class_names_path=args.class_names_path
        )
    else:
        print(
            "ONNX inference not yet implemented. Currently only pytorch inference is available."
        )
        return  # a temporary catch since ONNX inference has not yet been implemented

    print("=== warm CPU predict() latency ===")
    measured_times = []

    with Image.open(args.image_path) as source:
        print("Loading image...")
        image = source.convert("RGB")

    print(f"Starting {args.warmup_runs} warmup runs...")
    for _ in range(args.warmup_runs):
        inference.predict(image)

    print(f"Performing {args.measured_runs} measured runs...")
    for _ in range(args.measured_runs):
        start_time = time.perf_counter()
        inference.predict(image)
        end_time = time.perf_counter()
        measured_times.append(end_time - start_time)

    print(f"Inference sample count: {len(measured_times)}")
    print("=== INFERENCE TIMES (milliseconds) ===")
    print(f"Mean    | {statistics.mean(measured_times) * 1000:.3f}")
    print(f"Median  | {statistics.median(measured_times) * 1000:.3f}")
    print(f"p95     | {np.percentile(measured_times, 95) * 1000:.3f}")
    print(f"Maximum | {max(measured_times) * 1000:.3f}")


if __name__ == "__main__":
    main()
