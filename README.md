# tpuserve: Experimenting with Cloud TPU VMs as Model Servers

**NOTE: THIS IS JUST A FUN LEARNING AND EXPERIMENTATION PROJECT. DO NOT USE THIS.**

## Building

```
git clone https://github.com/seanmor5/tpuserve.git
cd tpuserve && make
```

## Running

Export models as HLO (see [https://github.com/google/jax/blob/main/jax/tools/jax_to_ir.py](jax_to_ir.py)) and then run:

```
tpuserve /path/to/model/directory
``` 