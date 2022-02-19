# tpuserve: Experimenting with Cloud TPU VMs as Model Servers

Google Cloud TPUs are designed for throughput - they work well for situations that benefit from massive parallelization and when precision isn't necessarily a concern, such as when training large neural networks. They are not designed to perform well in situations where low-latency at small batch sizes is important. Despite this, I wanted to see how Cloud TPU VMs would perform as platforms for model serving.

I am working on a free hosted version for those without access to Cloud TPU VMs who want to mess around with serving their models.

## Building and Running

You can run tpuserve with Elixir and CMake installed on a TPU VM. If you want to build a release binary with Burrito, you must also have zig, 7z, and gzip installed and in your path. With dependencies installed:

```
# Must run as sudo!
sudo mix run --no-halt
```

Or, for a release:

```
MIX_ENV=prod mix release
```

Or if you just have the binary:

```
sudo TPUSERVE_INSTALL_DIR=. tpuserve_linux
```

## Exporting Models

tpuserve serves models from a *model repository*. A repository is just a directory where each subdirectory represents a model endpoint. Each endpoint requires a `model.hlo.txt` and a `config.json` file. See the [](configuration guide) for information on constructing model configurations. An example model repository looks something like:

```
- models
    - resnet50
        - config.json
        - model.hlo.txt
    - bert
        - config.json
        - model.hlo.txt
```

The model repository above would serve endpoints for `resnet50` and `bert`. The `model.hlo.txt` is an exported HLO module from an XLA JIT-compiled function. Follow the steps below with the framework of your choice to export your model to a `model.hlo.txt`.

### Elixir: Nx and EXLA

You can export HLO text with `EXLA.export/3`:

```elixir
fun = fn x -> Nx.sum(x) end
model_hlo = EXLA.export(fun, [Nx.tensor([1, 2, 3])])
File.write!("model.hlo.txt", model_hlo)
```

### Python: JAX

You can export HLO text by constructing XLA Computations and then calling `as_hlo_text`:

```python
import jax
import jax.numpy as jnp

INPUT_SHAPE = (1, 1000)
fun = lambda x: jnp.sum(x)
comp = jax.xla_computation(fun)(jnp.ones(INPUT_SHAPE))

with open('model.hlo.txt', 'w') as outfile:
  outfile.write(comp.as_hlo_text())
```

### Python: TensorFlow

TensorFlow will use XLA when constructing a `tf.function` with `jit_compile=True`. You can extract the graph with `experimental_get_compiler_ir`:

```python
import tensorflow as tf

fun = lambda x: tf.math.reduce_sum(x)
comp = tf.function(fun, jit_compile=True)

with open('model.hlo.txt', 'w') as outfile:
  outfile.write(comp.experimental_get_compiler_ir(tf.ones(INPUT_SHAPE))())
``

If you have a Keras model:

```python
model = tf.keras.applications.ResNet101()
# Extract to tf.function, there is also model.predict_function
# but it doesn't seem to like exporting to IR
comp = tf.function(lambda x: model(x), jit_compile=True)
    
with open('model.hlo.txt', 'w') as outfile:
  outfile.write(comp.experimental_get_compiler_ir(tf.ones((1, 224, 224, 3,)))())
```

## Requests

You can send data for inference requests using Base64 encoded JSON or Msgpack. tpuserve will respond with either Base64 encoded JSON or Msgpack based on the content-type header. Input names must match the names in `config.json`.

Data should be a binary representation of your input data in a row-major memory layout. tpuserve will respond with a row-major result.

## Endpoints

The following endpoints are currently available:

| Endpoint               | Type   | Description                                                     |
|------------------------|--------|-----------------------------------------------------------------|
| `/v1/status`           | `GET`  | Sends 200 response and Up if the server is currently running    |
| `/v1/list_models`      | `GET`  | Responds with list of active endpoints and their configurations |
| `/v1/inference/:model` | `POST` | Send Inference request to given model name                      |