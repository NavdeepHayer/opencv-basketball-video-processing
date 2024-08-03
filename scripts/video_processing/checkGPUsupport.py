import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
for gpu in gpus:
    print("Device name:", gpu.name)

from tensorflow.python.platform import build_info as tf_build_info

print("CUDA version TensorFlow was built against:", tf_build_info.cuda_version)
print("cuDNN version TensorFlow was built against:", tf_build_info.cudnn_version)

