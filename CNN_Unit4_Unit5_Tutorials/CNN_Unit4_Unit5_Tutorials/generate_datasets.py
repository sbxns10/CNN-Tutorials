"""
Dataset Generator for Unit 4 and Unit 5 Deep Learning Tutorials
---------------------------------------------------------------
This script uses Python's Pillow library to generate synthetic RGB PNG datasets
for all six Deep Learning tutorials:
1. datasets/general_images/ (Apple, Banana, Orange)
2. datasets/traffic_signs/  (No_Entry, Speed_Limit, Stop)
3. datasets/cats_dogs/      (Cats, Dogs)

Each class folder receives 36 valid RGB PNG images (128x128 pixels) with varied
positions, sizes, colors, and backgrounds so they can be used immediately for
training and testing Convolutional Neural Networks (CNNs).
"""

import math
import random
from pathlib import Path
from PIL import Image, ImageDraw


# Set a fixed random seed for reproducibility
random.seed(42)

# Define base project directory using pathlib so it works from any working directory
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"

IMAGE_SIZE = (128, 128)
IMAGES_PER_CLASS = 36


def random_bg_color(base_rgb, variance=20):
    """Return a slightly varied background RGB color."""
    return tuple(
        max(0, min(255, c + random.randint(-variance, variance)))
        for c in base_rgb
    )


# ==============================================================================
# DATASET A: General Images (Apple, Banana, Orange)
# ==============================================================================
def draw_apple(index: int) -> Image.Image:
    """Draw a recognizable red apple with a stem and green leaf."""
    bg = random_bg_color((240, 245, 250))
    img = Image.new("RGB", IMAGE_SIZE, bg)
    draw = ImageDraw.Draw(img)

    cx = 64 + random.randint(-12, 12)
    cy = 68 + random.randint(-10, 10)
    radius = random.randint(28, 38)

    # Apple body (red variations)
    red_shade = (
        random.randint(190, 235),
        random.randint(20, 50),
        random.randint(25, 55),
    )
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=red_shade,
        outline=(140, 15, 20),
        width=2,
    )

    # Brown stem at the top
    draw.line(
        [(cx, cy - radius + 4), (cx + random.randint(-3, 3), cy - radius - 12)],
        fill=(101, 67, 33),
        width=4,
    )

    # Green leaf next to the stem
    leaf_box = [cx + 2, cy - radius - 10, cx + 18, cy - radius + 2]
    draw.ellipse(leaf_box, fill=(50, 168, 82), outline=(30, 120, 50))

    return img


def draw_banana(index: int) -> Image.Image:
    """Draw a recognizable yellow curved banana."""
    bg = random_bg_color((245, 242, 235))
    img = Image.new("RGB", IMAGE_SIZE, bg)
    draw = ImageDraw.Draw(img)

    cx = 64 + random.randint(-10, 10)
    cy = 64 + random.randint(-10, 10)
    w = random.randint(34, 44)
    h = random.randint(18, 26)

    yellow_shade = (
        random.randint(235, 255),
        random.randint(205, 230),
        random.randint(20, 60),
    )

    # Draw banana body as a thick curved polygon/arc
    points = []
    for angle_deg in range(-150, -25, 10):
        rad = math.radians(angle_deg)
        px = cx + int(w * math.cos(rad))
        py = cy + int(h * math.sin(rad)) + 15
        points.append((px, py))
    for angle_deg in range(-25, -150, -10):
        rad = math.radians(angle_deg)
        px = cx + int((w - 12) * math.cos(rad))
        py = cy + int((h - 10) * math.sin(rad)) + 8
        points.append((px, py))

    draw.polygon(points, fill=yellow_shade, outline=(180, 150, 20))
    # Banana tips
    if points:
        draw.ellipse(
            [points[0][0] - 3, points[0][1] - 3, points[0][0] + 3, points[0][1] + 3],
            fill=(110, 80, 30),
        )

    return img


def draw_orange(index: int) -> Image.Image:
    """Draw a recognizable textured orange citrus fruit."""
    bg = random_bg_color((238, 248, 242))
    img = Image.new("RGB", IMAGE_SIZE, bg)
    draw = ImageDraw.Draw(img)

    cx = 64 + random.randint(-12, 12)
    cy = 66 + random.randint(-10, 10)
    radius = random.randint(28, 38)

    orange_shade = (
        random.randint(235, 255),
        random.randint(130, 165),
        random.randint(10, 35),
    )
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=orange_shade,
        outline=(200, 100, 10),
        width=2,
    )

    # Small green calyx button on top
    draw.ellipse([cx - 4, cy - radius + 2, cx + 4, cy - radius + 8], fill=(60, 140, 50))

    # Add citrus pore dots
    for _ in range(15):
        dx = random.randint(-radius // 2, radius // 2)
        dy = random.randint(-radius // 2, radius // 2)
        draw.point((cx + dx, cy + dy), fill=(210, 105, 5))

    return img


# ==============================================================================
# DATASET B: Traffic Signs (No_Entry, Speed_Limit, Stop)
# ==============================================================================
def draw_no_entry(index: int) -> Image.Image:
    """Draw a red circular No-Entry sign with a horizontal white bar."""
    bg = random_bg_color((220, 230, 240))
    img = Image.new("RGB", IMAGE_SIZE, bg)
    draw = ImageDraw.Draw(img)

    cx = 64 + random.randint(-8, 8)
    cy = 64 + random.randint(-8, 8)
    r = random.randint(34, 42)

    # Outer white rim and red circle
    draw.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3], fill=(255, 255, 255))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(215, 25, 32))

    # Horizontal white bar in the center
    bar_w = int(r * 0.7)
    bar_h = max(6, int(r * 0.22))
    draw.rectangle(
        [cx - bar_w, cy - bar_h, cx + bar_w, cy + bar_h],
        fill=(255, 255, 255),
    )

    return img


def draw_speed_limit(index: int) -> Image.Image:
    """Draw a circular white Speed-Limit sign with a thick red border and dark center mark."""
    bg = random_bg_color((225, 235, 225))
    img = Image.new("RGB", IMAGE_SIZE, bg)
    draw = ImageDraw.Draw(img)

    cx = 64 + random.randint(-8, 8)
    cy = 64 + random.randint(-8, 8)
    r = random.randint(34, 42)
    border = max(7, int(r * 0.24))

    # Red border circle and white interior
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 30, 35))
    draw.ellipse(
        [cx - r + border, cy - r + border, cx + r - border, cy + r - border],
        fill=(250, 250, 250),
    )

    # Draw stylized '50' digits in black in the center
    # Digit '5'
    draw.rectangle([cx - 14, cy - 10, cx - 3, cy - 7], fill=(20, 20, 20))
    draw.rectangle([cx - 14, cy - 10, cx - 11, cy], fill=(20, 20, 20))
    draw.rectangle([cx - 14, cy - 2, cx - 3, cy + 1], fill=(20, 20, 20))
    draw.rectangle([cx - 6, cy - 2, cx - 3, cy + 10], fill=(20, 20, 20))
    draw.rectangle([cx - 14, cy + 7, cx - 3, cy + 10], fill=(20, 20, 20))

    # Digit '0'
    draw.ellipse([cx + 2, cy - 10, cx + 15, cy + 10], outline=(20, 20, 20), width=3)

    return img


def draw_stop_sign(index: int) -> Image.Image:
    """Draw a red octagonal STOP sign."""
    bg = random_bg_color((235, 235, 245))
    img = Image.new("RGB", IMAGE_SIZE, bg)
    draw = ImageDraw.Draw(img)

    cx = 64 + random.randint(-8, 8)
    cy = 64 + random.randint(-8, 8)
    r = random.randint(36, 44)

    # Regular 8-sided octagon vertices
    octagon = []
    for i in range(8):
        angle = math.radians(22.5 + i * 45)
        px = cx + int(r * math.cos(angle))
        py = cy + int(r * math.sin(angle))
        octagon.append((px, py))

    draw.polygon(octagon, fill=(200, 20, 25), outline=(255, 255, 255))

    # Inner white cross/letters indicator to distinguish texture from No Entry
    for dx in [-12, -4, 4, 12]:
        draw.rectangle([cx + dx - 2, cy - 6, cx + dx + 2, cy + 6], fill=(255, 255, 255))

    return img


# ==============================================================================
# DATASET C: Cats and Dogs (Cats, Dogs)
# ==============================================================================
def draw_cat(index: int) -> Image.Image:
    """Draw a simple cat face with pointed triangular ears and whiskers."""
    bg = random_bg_color((245, 240, 250))
    img = Image.new("RGB", IMAGE_SIZE, bg)
    draw = ImageDraw.Draw(img)

    cx = 64 + random.randint(-8, 8)
    cy = 68 + random.randint(-8, 8)
    r = random.randint(26, 32)

    fur_color = (
        random.randint(110, 150),
        random.randint(110, 150),
        random.randint(120, 165),
    )

    # Pointed triangular ears
    left_ear = [(cx - r + 2, cy - 10), (cx - r - 4, cy - r - 18), (cx - 6, cy - r + 4)]
    right_ear = [(cx + r - 2, cy - 10), (cx + r + 4, cy - r - 18), (cx + 6, cy - r + 4)]
    draw.polygon(left_ear, fill=fur_color, outline=(70, 70, 80))
    draw.polygon(right_ear, fill=fur_color, outline=(70, 70, 80))

    # Round head
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fur_color, outline=(70, 70, 80), width=2)

    # Green cat eyes
    draw.ellipse([cx - 14, cy - 8, cx - 6, cy], fill=(50, 200, 70))
    draw.ellipse([cx + 6, cy - 8, cx + 14, cy], fill=(50, 200, 70))

    # Pink triangle nose
    draw.polygon([(cx - 4, cy + 4), (cx + 4, cy + 4), (cx, cy + 9)], fill=(240, 130, 150))

    # Long horizontal whiskers
    for wy in [-2, 2, 6]:
        draw.line([(cx - 8, cy + 6), (cx - r - 10, cy + 4 + wy)], fill=(40, 40, 40), width=1)
        draw.line([(cx + 8, cy + 6), (cx + r + 10, cy + 4 + wy)], fill=(40, 40, 40), width=1)

    return img


def draw_dog(index: int) -> Image.Image:
    """Draw a simple dog face with floppy brown ears, snout patch, and red tongue."""
    bg = random_bg_color((240, 250, 240))
    img = Image.new("RGB", IMAGE_SIZE, bg)
    draw = ImageDraw.Draw(img)

    cx = 64 + random.randint(-8, 8)
    cy = 62 + random.randint(-8, 8)
    r = random.randint(26, 32)

    fur_color = (
        random.randint(175, 210),
        random.randint(125, 160),
        random.randint(70, 100),
    )
    ear_color = (110, 70, 35)

    # Floppy hanging oval ears on the sides
    draw.ellipse([cx - r - 12, cy - 10, cx - r + 8, cy + 28], fill=ear_color)
    draw.ellipse([cx + r - 8, cy - 10, cx + r + 12, cy + 28], fill=ear_color)

    # Head
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fur_color, outline=ear_color, width=2)

    # Lighter snout area
    draw.ellipse([cx - 16, cy + 2, cx + 16, cy + 24], fill=(235, 210, 175))

    # Dark eyes
    draw.ellipse([cx - 14, cy - 10, cx - 6, cy - 2], fill=(30, 25, 20))
    draw.ellipse([cx + 6, cy - 10, cx + 14, cy - 2], fill=(30, 25, 20))

    # Round black nose and pink tongue
    draw.ellipse([cx - 6, cy + 4, cx + 6, cy + 13], fill=(25, 25, 25))
    draw.ellipse([cx - 4, cy + 16, cx + 4, cy + 27], fill=(235, 85, 100))

    return img


def generate_all_datasets():
    """Generate all sample images across the three datasets."""
    dataset_specs = {
        DATASETS_DIR / "general_images" / "Apple": ("apple", draw_apple),
        DATASETS_DIR / "general_images" / "Banana": ("banana", draw_banana),
        DATASETS_DIR / "general_images" / "Orange": ("orange", draw_orange),
        DATASETS_DIR / "traffic_signs" / "No_Entry": ("no_entry", draw_no_entry),
        DATASETS_DIR / "traffic_signs" / "Speed_Limit": ("speed_limit", draw_speed_limit),
        DATASETS_DIR / "traffic_signs" / "Stop": ("stop", draw_stop_sign),
        DATASETS_DIR / "cats_dogs" / "Cats": ("cat", draw_cat),
        DATASETS_DIR / "cats_dogs" / "Dogs": ("dog", draw_dog),
    }

    total_created = 0
    for folder_path, (prefix, draw_fn) in dataset_specs.items():
        folder_path.mkdir(parents=True, exist_ok=True)
        for i in range(1, IMAGES_PER_CLASS + 1):
            img = draw_fn(i)
            file_path = folder_path / f"{prefix}_{i:03d}.png"
            img.save(file_path, format="PNG")
            total_created += 1
        print(f"Generated {IMAGES_PER_CLASS} PNG images in: {folder_path.relative_to(BASE_DIR)}")

    print(f"\nSuccessfully generated {total_created} total RGB PNG images across 8 class folders.")


if __name__ == "__main__":
    generate_all_datasets()
