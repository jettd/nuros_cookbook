import os, tarfile, time, math, pathlib
# TODO: Evaluate if this is absolutely necessary on Nuros
# limit thread creation; slurm does not seem to like too many threads
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras import applications, layers, models, optimizers, callbacks, utils

# --------- Hardware & precision ----------
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
num_gpus = len(gpus)
print("Visible GPUs:", num_gpus)

mixed_precision.set_global_policy("mixed_float16")  # L40 loves this

# --------- Paths ----------
HOME = pathlib.Path(os.path.expanduser("~"))
DATA_DIR = HOME / "data" / "food-101"
RAW_TAR = DATA_DIR / "food-101.tar.gz"
IMAGES_DIR = DATA_DIR / "images"
META_DIR = DATA_DIR / "meta"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------- Download Food-101 once ----------
if not RAW_TAR.exists() and not IMAGES_DIR.exists():
    url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    print("Downloading Food-101 (~5GB)...")
    fname = utils.get_file(RAW_TAR.name, origin=url, cache_dir=str(DATA_DIR.parent), cache_subdir=DATA_DIR.name)
    # move from Keras cache to our path if needed
    if fname != str(RAW_TAR):
        os.replace(fname, RAW_TAR)

if RAW_TAR.exists() and not IMAGES_DIR.exists():
    print("Extracting Food-101...")
    with tarfile.open(RAW_TAR, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    # extraction creates DATA_DIR/food-101/* -> flatten
    inner = DATA_DIR / "food-101"
    for p in ("images", "meta"):
        src = inner / p
        dst = DATA_DIR / p
        if src.exists() and not dst.exists():
            os.replace(src, dst)

assert IMAGES_DIR.exists() and META_DIR.exists(), "Food-101 not found after download/extract."

# --------- Build tf.data pipeline ----------
IMG_SIZE = 380  # use 380 for EfficientNetB4; 600 for B7 (needs more VRAM)
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 32
EPOCHS = 30

# Food-101 provides train/test file lists
def read_split_list(txt_path):
    with open(txt_path, "r") as f:
        items = [line.strip() for line in f]
    # lines are like 'apple_pie/1005649'; map to full path
    paths = [IMAGES_DIR / (x + ".jpg") for x in items]
    labels = [x.split("/")[0] for x in items]
    classes = sorted({c for c in labels})
    class_to_id = {c:i for i,c in enumerate(classes)}
    y = [class_to_id[c] for c in labels]
    return paths, y, classes

train_list = META_DIR / "train.txt"
test_list  = META_DIR / "test.txt"
train_paths, train_y, classes = read_split_list(train_list)
test_paths,  test_y,  _       = read_split_list(test_list)
NUM_CLASSES = len(classes)
print("Classes:", NUM_CLASSES, "Train imgs:", len(train_paths), "Test imgs:", len(test_paths))

def decode_load_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bicubic")
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Data augmentation! (to improve generalization)
def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.stateless_random_brightness(img, max_delta=0.1, seed=(1,2))
    img = tf.image.stateless_random_contrast(img, lower=0.9, upper=1.1, seed=(3,4))
    return img, label

def make_ds(paths, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((list(map(str, paths)), labels))
    if training:
        ds = ds.shuffle(8192, reshuffle_each_iteration=True)
    ds = ds.map(decode_load_resize, num_parallel_calls=4)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds

# Split training data for validation
val_split = int(0.02 * len(train_paths))
val_paths, val_y = train_paths[:val_split], train_y[:val_split]
train_paths, train_y = train_paths[val_split:], train_y[val_split:]

train_ds = make_ds(train_paths, train_y, training=True)
val_ds = make_ds(val_paths, val_y, training=False)

print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

# Calculate steps per epoch for training metrics
train_steps = len(train_paths) // BATCH_SIZE

# --------- Model (EfficientNetB4) ----------
BACKBONE = "B4"  # Stick with B4 for reliability, (B7 might be buggy)
input_shape = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_rgb")

# Build EfficientNetB4 from scratch (weights=None means no pretrained weights)
# This avoids transfer learning complications but still uses the proven architecture
base = applications.EfficientNetB4(include_top=False, weights=None, input_tensor=input_shape)
base.trainable = True  # Train all layers from scratch

# Add classification head
x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
model = models.Model(input_shape, outputs)

# Compile model
opt = optimizers.Adam(1e-4)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Model built successfully!")
model.summary()

# --------- Training setup ----------
ckpt_dir = HOME / "runs" / "food101_efficientnet" / BACKBONE
ckpt_dir.mkdir(parents=True, exist_ok=True)

callbacks_list = [
    callbacks.ModelCheckpoint(str(ckpt_dir / "ckpt-{epoch:02d}.keras"),
                              save_weights_only=False, save_best_only=False),
    callbacks.CSVLogger(str(ckpt_dir / "log.csv")),
]

# --------- Training ----------
print(f"Starting training for {EPOCHS} epochs...")
t0 = time.time()
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks_list,
    verbose=2
)
dt = time.time() - t0

# Calculate throughput
total_images = BATCH_SIZE * train_steps * EPOCHS
throughput = total_images / dt
print(f"Training completed in {dt:.1f} seconds")
print(f"Throughput: {throughput:.1f} images/sec (batch={BATCH_SIZE}, epochs={EPOCHS}, steps/epoch={train_steps})")

# --------- Final evaluation ----------
print("Evaluating on test set...")
test_ds = make_ds(test_paths, test_y, training=False)
test_results = model.evaluate(test_ds, verbose=0)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")

print(f"Results saved to: {ckpt_dir}")