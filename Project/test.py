import tensorflow as tf
import os
print(os.environ["CUDA_VISIBLE_DEVICES"])

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def a() -> int:
    print()


def b() -> int:
    print()
