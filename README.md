[title]: - "Working with Tensorflow, GPUs, and containers"
[TOC]

## Overview

In this tutorial, we explore GPUs and containers on OSG, using the popular Tensorflow
sofware package. Tensorflow is a good example here as the software is too complex to 
bundle up and ship with your job. Containers solve this problem by defining a full
OS image, containing not only the complex software package, but dependencies and
environment configuration as well.

[https://www.tensorflow.org/](https://www.tensorflow.org/) desribes TensorFlow as:

> TensorFlow is an open source software library for numerical
> computation using data flow graphs. Nodes in the graph represent
> mathematical operations, while the graph edges represent the
> multidimensional data arrays (tensors) communicated between them. The
> flexible architecture allows you to deploy computation to one or more
> CPUs or GPUs in a desktop, server, or mobile device with a single
> API. TensorFlow was originally developed by researchers and engineers
> working on the Google Brain Team within Google's Machine Intelligence
> research organization for the purposes of conducting machine learning
> and deep neural networks research, but the system is general enough to
> be applicable in a wide variety of other domains as well.


## Defining container images

Defining containers is fully described in the [Docker and Singularity Containers](https://support.opensciencegrid.org/support/solutions/articles/12000024676)
section. Here we will just provide an overview of how you could take something
like an existing Tensorflow image provided by OSG staff, and extend it by
adding your own modules to it. Let's assume you like Tensorflow version
2.3. The definition of this image can be found in Github: [Dockerfile](https://github.com/opensciencegrid/osgvo-tensorflow/blob/2.3/Dockerfile). You don't really need to 
understand how an image was built in order to use it. As described in
the containers documentation, make sure the HTCondor submit file has:


    Requirements = HAS_SINGULARITY == TRUE
    +SingularityImage = "/cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow:2.3"


If you want to extend an existing image, you can just inherit from the 
parent image available on DockerHub [here](https://hub.docker.com/r/opensciencegrid/tensorflow).
For example, if you just need some additional Python packages, your
new Dockerfile could look like:


    FROM opensciencegrid/tensorflow:2.3
    
    RUN python3 -m pip install some_package_name


You can then `docker build` and `docker push` it so that your new
image is available on DockerHub. Note that OSG does _not_ provide
any infrastructure for these steps. You will have to complete
them on your own computer or using the DockerHub build
infrastructure.


## Adding a container to the OSG CVMFS distribution mechanism

How to add a container image to the OSG CVMFS distribution mechanism is also
described in [Docker and Singularity Containers](https://support.opensciencegrid.org/support/solutions/articles/12000024676),
but a quick scan of the [cvmfs-singularity-sync](https://github.com/opensciencegrid/cvmfs-singularity-sync) and specifically the `docker_images.txt` file show us that the tensorflow
images are listed as:

    opensciencegrid/tensorflow:*
    opensciencegrid/tensorflow-gpu:*

Those two lines means that all tags from those two DockerHub repositories should
be mapped to `/cvmfs/singularity.opensciencegrid.org/`. On the login node, try
running:

    ls /cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow:2.3/

This is the image in its expanded form - something we can execute with Singularity!


## Testing the container on the submit host

First, download the files contained in this tutorial to the login node using the `tutorial` command and `cd` into the tutorial directory that is created:

    tutorial tensorflow-containers
    cd tutorial-tensorflow-containers

Before submitting jobs to the OSG, it is always a good idea to test your code
so that you understand runtime requirements. The containers can be tested
on the OSGConnect submit hosts with `singularity shell`, which will drop you
into a container and let you exlore it interactively. To explore the 
Tensorflow 2.3 image, run:

    singularity shell /cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow:2.3/

Note how the command line prompt changes, providing you an indicator that 
you are inside the image. You can exit any time by running `exit`. Another
important thing to note is that your `$HOME` directory is automatically 
mounted inside the interactive container - allowing you to access your
codes and test it out. First, start with a simple python3 import test to
make sure tensorflow is available:

    $ python3
    Python 3.6.9 (default, Jul 17 2020, 12:50:27) 
    [GCC 8.4.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import tensorflow
    2021-01-15 17:32:33.901607: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
    2021-01-15 17:32:33.901735: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    >>>

Tensorflow will warn you that no GPUs where found. This is expected as we
do not have GPUs attached to our login nodes, and it is fine as Tensorflow
works fine with regular CPUs (slower of course).

Exit out of Python3 with `CTRL+D` and then we can run a Tensorflow testcode
which can be found in this tutorial:

    $ python3 test.py 
    2021-01-15 17:37:43.152892: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
    2021-01-15 17:37:43.153021: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    2021-01-15 17:37:44.899967: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2021-01-15 17:37:44.900063: W tensorflow/stream_executor/cuda/cuda_driver.cc:312] failed call to cuInit: UNKNOWN ERROR (303)
    2021-01-15 17:37:44.900130: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (login05.osgconnect.net): /proc/driver/nvidia/version does not exist
    2021-01-15 17:37:44.900821: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2021-01-15 17:37:44.912483: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2700000000 Hz
    2021-01-15 17:37:44.915548: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x4fa0bf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2021-01-15 17:37:44.915645: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2021-01-15 17:37:44.921895: I tensorflow/core/common_runtime/eager/execute.cc:611] Executing op MatMul in device /job:localhost/replica:0/task:0/device:CPU:0
    tf.Tensor(
    [[22. 28.]
     [49. 64.]], shape=(2, 2), dtype=float32)

We will again see a bunch of warnings regarding GPUs not being available, but as
we can see by the `/job:localhost/replica:0/task:0/device:CPU:0` line, the code ran
on one of the CPUs. When testing your own code like this, take note of how much
memory, disk and runtime is required - it is needed in the next step.

Once you are done with testing, use `CTRL+D` or run `exit` to exit out of 
the container. Note that you can _not_ submit jobs from within the container.


## Running a CPU job

If Tensorflow can run on GPUs, you might be wondering why we might want to run
it on slower CPUs? One reason is that CPUs are plentiful while GPUs are still
somewhat scarce. If you have a lot of shorter Tensorflow jobs, they might 
complete faster on available CPUs, rather than wait in the queue for the
faster, less available, GPUs. The good news is that Tensorflow code should
work in both enviroments automatically, so if your code runs too slow on CPUs,
moving to GPUs should be easy.

To submit our job, we need a submit file and a job wrapper script. The
submit file is a basic OSGConnect flavored HTCondor file, specifying that 
we want the job to run in a container. `cpu-job.submit` contains:

    universe = vanilla
    
    # Job requirements - ensure we are running on a Singularity enabled
    # node and have enough resources to execute our code
    # Tensorflow also requires AVX instruction set and a newer host kernel
    Requirements = HAS_SINGULARITY == True && HAS_AVX2 == True && OSG_HOST_KERNEL_VERSION >= 31000
    request_cpus = 1
    request_gpus = 0
    request_memory = 1 GB
    request_disk = 1 GB
    
    # Container image to run the job in
    +SingularityImage = "/cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow:2.3"
    
    # Executable is the program your job will run It's often useful
    # to create a shell script to "wrap" your actual work.
    Executable = job-wrapper.sh
    Arguments = 
    
    # Inputs/outputs - in this case we just need our python code.
    # If you leave out transfer_output_files, all generated files comes back
    transfer_input_files = test.py
    #transfer_output_files = 
    
    # Error and output are the error and output channels from your job
    # that HTCondor returns from the remote host.
    Error = $(Cluster).$(Process).error
    Output = $(Cluster).$(Process).output
    
    # The LOG file is where HTCondor places information about your
    # job's status, success, and resource consumption.
    Log = $(Cluster).log
    
    # Send the job to Held state on failure. 
    #on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
    
    # Periodically retry the jobs every 1 hour, up to a maximum of 5 retries.
    #periodic_release =  (NumJobStarts < 5) && ((CurrentTime - EnteredCurrentStatus) > 60*60)
    
    # queue is the "start button" - it launches any jobs that have been
    # specified thus far.
    queue 1


And job-wrapper.sh:


    #!/bin/bash
    
    set -e
    
    # set TMPDIR variable
    export TMPDIR=$_CONDOR_SCRATCH_DIR
    
    echo
    echo "I'm running on" $(hostname -f)
    echo "OSG site: $OSG_SITE_NAME"
    echo
    
    python3 test.py 2>&1


The job can now be submitted with `condor_submit cpu-job.submit`. Once the job
is done, check the files named after the job id for the outputs.


## Running a GPU job

When moving the job to be run on a GPU, all we have to do is update two lines
in the submit file: set `request_gpus` to `1` and specify a GPU enabled 
container image for `+SingularityImage`. The updated submit file can be found
in `gpu-job.submit` with the contents:


    universe = vanilla
    
    # Job requirements - ensure we are running on a Singularity enabled
    # node and have enough resources to execute our code
    # Tensorflow also requires AVX instruction set and a newer host kernel
    Requirements = HAS_SINGULARITY == True && HAS_AVX2 == True && OSG_HOST_KERNEL_VERSION >= 31000
    request_cpus = 1
    request_gpus = 1
    request_memory = 1 GB
    request_disk = 1 GB
    
    # Container image to run the job in
    +SingularityImage = "/cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow-gpu:2.3"
    
    # Executable is the program your job will run It's often useful
    # to create a shell script to "wrap" your actual work.
    Executable = job-wrapper.sh
    Arguments = 
    
    # Inputs/outputs - in this case we just need our python code.
    # If you leave out transfer_output_files, all generated files comes back
    transfer_input_files = test.py
    #transfer_output_files = 
    
    # Error and output are the error and output channels from your job
    # that HTCondor returns from the remote host.
    Error = $(Cluster).$(Process).error
    Output = $(Cluster).$(Process).output
    
    # The LOG file is where HTCondor places information about your
    # job's status, success, and resource consumption.
    Log = $(Cluster).log
    
    # Send the job to Held state on failure. 
    #on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
    
    # Periodically retry the jobs every 1 hour, up to a maximum of 5 retries.
    #periodic_release =  (NumJobStarts < 5) && ((CurrentTime - EnteredCurrentStatus) > 60*60)
    
    # queue is the "start button" - it launches any jobs that have been
    # specified thus far.
    queue 1


Submit a job with `condor_submit gpu-job.submit`. Once the job is complete, check
the `.out` file for a line stating the code was run under a GPU. Something similar
to:


    2021-02-02 23:25:19.022467: I tensorflow/core/common_runtime/eager/execute.cc:611] Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0    


The `GPU:0` parts shows that a GPU was found and used for the computation.

