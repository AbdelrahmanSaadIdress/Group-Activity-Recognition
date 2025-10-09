import os
from AnnotationsExtraction import BoxInfo, Extractor, AnnotationPreparer

# ==============================================================
#                     PATH CONFIGURATION
# ==============================================================

# Use Linux-style paths for WSL
DATASET_ROOT = "/mnt/d/Desktop/04 Project/volleyball-datasets"

ANNOTATIONS_DIR = os.path.join(
    DATASET_ROOT, "volleyball_tracking_annotation", "volleyball_tracking_annotation"
)
IMAGES_DIR = os.path.join(DATASET_ROOT, "volleyball_", "videos")
SAVE_DIR = "./processed_data"

# Example annotation file path
EXAMPLE_FILE = os.path.join(ANNOTATIONS_DIR, "1", "9570", "9570.txt")

print("\n🚀 Starting Annotation Extraction Pipeline...\n")

# ==============================================================
#                      PATH VALIDATION
# ==============================================================

if not os.path.exists(EXAMPLE_FILE):
    raise FileNotFoundError(f"❌ Annotation file not found:\n{EXAMPLE_FILE}")

print(f"📁 Annotation file found: {EXAMPLE_FILE}")
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"📂 Output directory ensured at: {SAVE_DIR}")

# ==============================================================
#                       TESTING BoxInfo
# ==============================================================

print("\n==================== Testing BoxInfo ====================\n")

with open(EXAMPLE_FILE, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

if not lines:
    raise ValueError("❌ The annotation file is empty!")

# Display first 3 parsed boxes
for i, line in enumerate(lines[:3], start=1):
    try:
        box = BoxInfo(line)
        print(f"[Line {i}] Parsed successfully:")
        print(f"   → Box: {box.box}")
        print(f"   → Category: {box.category}")
        print(f"   → Frame ID: {box.frame_id}")
        print(f"   → Valid: {box.is_valid}")
        print("-" * 70)
    except Exception as e:
        print(f"⚠️  Error parsing line {i}: {e}")

# ==============================================================
#                       TESTING Extractor
# ==============================================================

print("\n==================== Testing AnnotationExtractor ====================\n")

TARGET_FRAME = 9575

try:
    frame_boxes = Extractor.extract_crops_annot(EXAMPLE_FILE, TARGET_FRAME)
    print(f"✅ Frames near {TARGET_FRAME}: {list(frame_boxes.keys())[:5]}")
except Exception as e:
    print(f"❌ Error extracting crops annotations:\n   {e}")

try:
    frame_annots = Extractor.extract_frame_annot(EXAMPLE_FILE)
    print(f"✅ First 5 frame annotations: {list(frame_annots.items())[:5]}")
except Exception as e:
    print(f"❌ Error extracting frame annotations:\n   {e}")

# ==============================================================
#                   TESTING AnnotationPreparer
# ==============================================================

print("\n==================== Testing AnnotationPreparer ====================\n")

CROPS_ANNOTS_PATH = os.path.join(DATASET_ROOT, "volleyball_tracking_annotation", "volleyball_tracking_annotation")
FRAMES_ANNOTS_PATH = os.path.join(DATASET_ROOT, "volleyball_", "videos")

if not os.path.exists(CROPS_ANNOTS_PATH):
    raise FileNotFoundError(f"❌ Crops annotations path not found:\n{CROPS_ANNOTS_PATH}")

if not os.path.exists(FRAMES_ANNOTS_PATH):
    raise FileNotFoundError(f"❌ Frames annotations path not found:\n{FRAMES_ANNOTS_PATH}")

try:
    print("🧩 Preparing annotations (this may take a while)...\n")
    matches_data = AnnotationPreparer.prepare_annotations(
        crops_annots_path=CROPS_ANNOTS_PATH,
        frames_annots_path=FRAMES_ANNOTS_PATH,
        save_path=SAVE_DIR
    )
    print(f"✅ Processed {len(matches_data)} matches successfully.")

    print("\n💾 Loading saved annotations...")
    annotations = AnnotationPreparer.load_annotations(SAVE_DIR)
    print(f"✅ Loaded {len(annotations)} matches from saved data.")
except Exception as e:
    print(f"❌ Error during annotation preparation:\n   {e}")

# ==============================================================
#                       COMPLETION
# ==============================================================

print("\n🎉 Annotation extraction pipeline completed successfully!\n")

print("Annotation keys:", list(annotations.keys())[:10])
print("Type of keys:", [type(k) for k in list(annotations.keys())[:10]])

print(annotations["48"])


from helpers.logger import setup_logging

logger = setup_logging("./experiments/run_01")
logger.info("Starting annotation extraction...")
logger.warning("Missing frames for match 35.")
logger.error("Failed to process match 12.")
