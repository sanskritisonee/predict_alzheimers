try:
    import tensorflow as tf
except ModuleNotFoundError:
    print("TensorFlow is not installed in this environment.")
    print("Install Python 3.10 or 3.11 and run: python -m pip install tensorflow keras")
    raise SystemExit(0)

print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs detected: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu}")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth for available GPUs.")
    except Exception as exc:
        print(f"Could not set memory growth: {exc}")
else:
    print("No GPU available for TensorFlow. CPU will be used.")
