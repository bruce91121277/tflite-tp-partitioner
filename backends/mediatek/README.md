# Mediatek NPU verification backend

This optional backend adapts TFLite contracts and input captures, stages
user-provided artifacts, invokes externally installed NCC/Neuron Runtime
commands over SSH, compares outputs, and emits numerical reports.

No vendor SDK, compiler, DLA artifact, Neuron Runtime library, key, or device
capture is distributed in this repository. The user must install and authorize
the required external tools and device access.

Inspect the supported commands with:

    tflite-tp-verify --help

A caller may select a normal OpenSSH configuration file through the
TFLITE_TP_SSH_CONFIG environment variable. The runner passes that configuration
to OpenSSH/SCP; it does not read or store private key material. Use host aliases
and credentials managed by your own OpenSSH setup.

NCC compilation, host Golden execution, and NPU exactness are reported as
separate statuses. Never infer NPU exactness from a successful compile or
partial output coverage. Before running against hardware, inspect the command
help and backend source and provide your own model, inputs, compiler, and
device configuration.