import os
import json
from torchvision.ops import box_convert
import logging
import luigi
from PIL import Image
import subprocess
import torch
import torchvision.transforms as T
import numpy as np
import cv2
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

logger = logging.getLogger('luigi-interface')

_sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")

GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize once
_grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

def sam21_cut_person(image_path: str) -> Image.Image:
    logger.info("sam21_cut_person")
    logger.info(image_path)
    """
    Uses Grounded SAM2 to extract the 'person' from the image and return a cropped PIL image.
    """
    image_source, image_tensor = load_image(image_path)  # numpy, tensor
    _sam2_predictor.set_image(image_source)

    # Grounded DINO to get person boxes
    boxes, confidences, labels = predict(
        model=_grounding_model,
        image=image_tensor,
        caption="person",
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    if len(boxes) == 0:
        return Image.open(image_path).convert("RGB")  # fallback

    # Convert boxes to xyxy for SAM2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        masks, scores, _ = _sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    # Use first mask (assume top-scoring detection)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    mask = masks[0] > 0.35  # convert to boolean mask

# Apply mask
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    result = image_rgb.copy()
    result[~mask] = 255

    # Crop to bounding box
    y_indices, x_indices = np.where(mask)
    if not len(y_indices):
        return Image.fromarray(result)
    y0, y1 = y_indices.min(), y_indices.max()
    x0, x1 = x_indices.min(), x_indices.max()
    cropped = result[y0:y1, x0:x1]

    return Image.fromarray(cropped)

class ExtractSharpFramesFromVideo(luigi.Task):
    """
    Extract sharpest frames from each interval in a video using FFmpeg + OpenCV.
    Outputs a JSON manifest listing all extracted frame paths.
    """
    video_path = luigi.Parameter()
    output_dir = luigi.Parameter(default="temp/extracted_frames")
    src_dir = luigi.Parameter()
    interval_sec = luigi.IntParameter(default=2)
    fps = luigi.IntParameter(default=5)
    max_workers = luigi.IntParameter(default=6)

    def output(self):
        src_dir_name = os.path.basename(self.src_dir.rstrip("/"))
        file_name = os.path.basename(self.video_path.rstrip("/"))
        file_parent_dir_name = os.path.basename(os.path.dirname(self.video_path))
        manifest_dir = os.path.join(self.output_dir, src_dir_name, file_parent_dir_name)
        os.makedirs(manifest_dir, exist_ok=True)
        manifest_path = os.path.join(manifest_dir, f"frames_{file_name}.json")
        return luigi.LocalTarget(manifest_path)

    def run(self):
        import tempfile
        import shutil
        from concurrent.futures import ThreadPoolExecutor

        def variance_of_laplacian(image):
            return cv2.Laplacian(image, cv2.CV_64F).var()

        def extract_window(start_time):
            tmpdir = os.path.join(tempfile.gettempdir(), f"sharp_extract_{os.path.basename(self.video_path)}_{start_time}")
            os.makedirs(tmpdir, exist_ok=True)
            pattern = os.path.join(tmpdir, "frame_%03d.jpg")

            result = subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-hwaccel", "cuda",
                "-ss", str(start_time), "-t", str(self.interval_sec),
                "-i", self.video_path,
                "-vf", f"fps={self.fps}",
                pattern
            ], capture_output=True, text=True)

            sharpest_score = -1
            sharpest_frame = None

            for fname in os.listdir(tmpdir):
                fpath = os.path.join(tmpdir, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                score = variance_of_laplacian(gray)
                if score > sharpest_score:
                    sharpest_score = score
                    sharpest_frame = img

            out_path = None
            if sharpest_frame is not None:
                frame_name = f"f_{video_name}_{start_time:04d}.jpg"
                out_path = os.path.join(video_output_dir, frame_name)
                cv2.imwrite(out_path, sharpest_frame)

            shutil.rmtree(tmpdir, ignore_errors=True)
            return out_path

        # Prepare output dir
        src_dir_name = os.path.basename(self.src_dir.rstrip("/"))
        video_name = os.path.basename(self.video_path.rstrip("/"))
        video_parent_dir_name = os.path.basename(os.path.dirname(self.video_path))
        video_output_dir = os.path.join(
            self.output_dir,
            src_dir_name,
            video_parent_dir_name,
        )
        os.makedirs(video_output_dir, exist_ok=True)

        # Get duration of video
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", self.video_path],
            capture_output=True, text=True
        )
        duration = float(result.stdout.strip())
        jobs = list(range(0, int(duration), self.interval_sec))

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(extract_window, jobs))

        # Save manifest
        valid_results = [p for p in results if p is not None]
        with self.output().open("w") as f:
            json.dump(valid_results, f, indent=2)

class ListImages(luigi.Task):
    src_dir = luigi.Parameter()

    def requires(self):
        # Return a dict of {name: ExtractSharpFramesFromVideo}
        requirements = {}

        for root, _, files in os.walk(self.src_dir):
            for file in files:
                if file.lower().endswith((".mp4", ".mov", ".webm", ".mkv")):
                    video_path = os.path.abspath(os.path.join(root, file))
                    print("video_path")
                    print(video_path)
                    name = os.path.splitext(os.path.relpath(video_path, self.src_dir))[0].replace("/", "_")
                    requirements[name] = ExtractSharpFramesFromVideo(video_path=video_path, src_dir=self.src_dir)

        return requirements

    def output(self):
        src_dir_name = os.path.basename(self.src_dir.rstrip("/"))
        return luigi.LocalTarget(f"output/{src_dir_name}/image_list.txt")

    def run(self):
        src_dir_name = os.path.basename(os.path.normpath(self.src_dir))
        out_dir = f"output/{src_dir_name}"
        os.makedirs(out_dir, exist_ok=True)

        all_image_paths = []

        # 1. Find direct image files
        for root, _, files in os.walk(self.src_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    abs_path = os.path.abspath(os.path.join(root, file))
                    all_image_paths.append(abs_path)

        # 2. Find extracted frames from ExtractSharpFramesFromVideo
        for task in self.input().values():  # from requires()
            done_file = task.path
            frames_dir = os.path.dirname(done_file)
            print("done_file")
            print(done_file)
            with open(done_file) as jf:
                for img_path in json.load(jf):
                    all_image_paths.append(os.path.abspath(img_path))

        # 3. Save all
        with self.output().open("w") as f:
            for path in sorted(all_image_paths):
                f.write(path + "\n")

class CropAndSaveImage(luigi.Task):
    src_dir = luigi.Parameter()
    index = luigi.IntParameter()

    def requires(self):
        return ListImages(src_dir=self.src_dir)

    def output(self):
        with self.input().open("r") as f:
            rel_path = f.read().splitlines()[self.index]
        src_dir_name = os.path.basename(os.path.normpath(self.src_dir))
        rel_name = os.path.basename(os.path.normpath(rel_path))
        rel_parent_dir_name = os.path.basename(os.path.dirname(rel_path))

        output_path = os.path.join("output", src_dir_name, rel_parent_dir_name, rel_name)
        output_path = os.path.splitext(output_path)[0] + "_cropped.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return luigi.LocalTarget(output_path)

    def run(self):
        with self.input().open("r") as f:
            image_path = f.read().splitlines()[self.index]

        cut_image = sam21_cut_person(image_path)

        width, height = cut_image.size
        if height > 1024:
            scale_factor = 1024 / height
            new_width = int(width * scale_factor)
            new_height = 1024
            cut_image = cut_image.resize((new_width, new_height), Image.LANCZOS)

        cut_image.save(self.output().path)

class GenerateCaption(luigi.Task):
    src_dir = luigi.Parameter()
    index = luigi.IntParameter()

    def requires(self):
        return CropAndSaveImage(src_dir=self.src_dir, index=self.index)

    def output(self):
        cropped_path = self.input().path
        txt_path = os.path.splitext(cropped_path)[0] + ".txt"
        return luigi.LocalTarget(txt_path)

    def run(self):
        img_path = self.input().path
        absolute_path = os.path.abspath(img_path)
        result = subprocess.run(
            ["python", "luigi-caption/caption.py", "--image-path", absolute_path],
            capture_output=True,
            text=True
        )

        caption = result.stdout.strip()
        caption = caption.strip()

        if caption != "":
            with self.output().open("w") as f:
                f.write(caption + "\n")
        else:
            logger.warn("got empty caption for " + absolute_path)


class GenerateAllCaptions(luigi.Task):
    src_dir = luigi.Parameter()

    def requires(self):
        return ListImages(src_dir=self.src_dir)

    def output(self):
        src_dir_name = os.path.basename(os.path.normpath(self.src_dir))
        return luigi.LocalTarget(f"output/{src_dir_name}/all_captions.done")

    def run(self):
        logger.info(self.src_dir)
        src_dir_name = os.path.basename(os.path.normpath(self.src_dir))
        logger.info(src_dir_name)
        with open(f"output/{src_dir_name}/image_list.txt", "r") as f:
            image_paths = f.read().splitlines()

        total = len(image_paths)
        remaining = 0
        for idx in range(total):
            remaining += 1
            yield GenerateCaption(src_dir=self.src_dir, index=idx)

            logger.info(f"[GenerateAllCaptions] {remaining} of {total} captions remaining.")

        with self.output().open("w") as f:
            f.write("done\n")

if __name__ == "__main__":
    luigi.run()
