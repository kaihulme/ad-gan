# medical-gan

Kai Hulme's final year Computer Science thesis at University of Bristol.

MRI pre-processing pipelines and classification of ailments in the brain, with generation of MRI slices using generative adversarial networks (GANs).

## Requirements

- Docker
- Nvidia-Docker(2)

Two development containers are provided depending on the task:

- `.devcontainer_tf_gpu/` _(TensorFlow models)_
- `.devcontainer_neuro/` _(MRI pre-processing)_

Rename target container to `.devcontainer/` to use in VSCode with the [Remote Containers](https://code.visualstudio.com/docs/remote/containers) extension.

### tf_gpu

- Tensorflow GPU environment.
- Main development container for interfacing with TensorFlow models.
- Based on `tensorflow/tensorflow:latest-gpu` image.

### neuro

- NiPype neuroimaging pipeline environment.
- Development container for MRI preprocessing pipelines.
- Based on `nipype/nipype:latest` image.


