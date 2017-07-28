### Descriptions

[NNPACK](https://github.com/Maratyszcza/NNPACK) is an acceleration package for neural network computations, which can run on x86-64, ARMv7, or ARM64 architecture cpus. it's very useful for us using NNPACK to speed up running speed when deploy the trained model on mobile device.

MXNet has integrated NNPACK for forward propagation(only inference) in convolution/max-pooling/fully-connected, so you may consider using NNPACK now.


### Conditions
The underlying implementation of NNPACK utilize some other acceleration methods, such as [fft](https://arxiv.org/abs/1312.5851), [winograd](https://arxiv.org/abs/1509.09308), but these algorithms work better on some special `batch size`, `kernel size`, `stride` etc., so not all convolution/max-pooling/fully-connected can be powered by NNPACK. If some conditions are not met, it will change to the default implementation with MXNet automatically.  

nnpack only support Linux or OS X host system, that is to say, Windows is not supported at present.
The following table will tell you which satisfaction will NNPACK work.

| operation      | conditions |
|:---------      |:---------- |
|convolution     |2d convolution `and` no-bias=False `and` dilate=(1,1) `and` num_group=1 `and` batch-size = 1 or batch-size > 1 && stride = (1,1);|
|pooling         | max-pooling `and` kernel=(2,2) `and` stride=(2,2) `and` pooling_convention=full    |
|fully-connected| without any restrictions |

### Build/Install NNPACK with MXNet

Now, if the trained model meets some conditions of using NNPACK, you can build MXNet with NNPACK support. here is the steps for you:  
* install NNPACK based on this [tutorials](https://github.com/Maratyszcza/NNPACK#building), that's to say you need ninja to build NNPACK. make sure add `--enable-shared` when running configure.py(i.e. `python configure.py --enable-shared`), because MXNet will link NNPACK dynamically.  
* set lib path of NNPACK as the environment variable, such as `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$YOUR_NNPACK_INSTALL_PATH/lib`
* add the include file of NNPACK and its third-party to  `ADD_CFLAGS` in config.mk, such as `ADD_CFLAGS = -I$(YOUR_NNPACK_INSTALL_PATH)/include -I$(YOUR_NNPACK_INSTALL_PATH)/pthreadpool/include`
* set `USE_NNPACK = 1` in config.mk.
* [build MXNet](http://mxnet.io/get_started/setup.html#overview).

### NNPACK Performance

Though not all conv/pool/fc layer can make full use of NNPACK, it indeed can speed up some popular deep learning models such as Alexnet, VGG, Inception-bn.

here we use `example/image-classification/benchmark_score.py`(changed with  more range of batch-size) to benchmark it, cpu is e5-2670, MXNET_CPU_NNPACK_NTHREADS=4.

build MXNet without NNPACK, the log is:
```
INFO:root:network: alexnet
INFO:root:device: cpu(0)
INFO:root:batch size  1, image/sec: 6.389429
INFO:root:batch size  2, image/sec: 7.961457
INFO:root:batch size  4, image/sec: 8.950112
INFO:root:batch size  8, image/sec: 9.578176
INFO:root:batch size 16, image/sec: 9.701248
INFO:root:batch size 32, image/sec: 9.839940
INFO:root:batch size 64, image/sec: 10.075369
INFO:root:batch size 128, image/sec: 10.053556
INFO:root:batch size 256, image/sec: 9.972228
INFO:root:network: vgg
INFO:root:device: cpu(0)
INFO:root:batch size  1, image/sec: 1.223822
INFO:root:batch size  2, image/sec: 1.322814
INFO:root:batch size  4, image/sec: 1.383586
INFO:root:batch size  8, image/sec: 1.402376
INFO:root:batch size 16, image/sec: 1.415972
INFO:root:batch size 32, image/sec: 1.428377
INFO:root:batch size 64, image/sec: 1.443987
INFO:root:batch size 128, image/sec: 1.427531
INFO:root:batch size 256, image/sec: 1.435279
```

build MXNet with NNPACK, log is:

```
INFO:root:network: alexnet
INFO:root:device: cpu(0)
INFO:root:batch size  1, image/sec: 19.027215
INFO:root:batch size  2, image/sec: 12.879975
INFO:root:batch size  4, image/sec: 17.424076
INFO:root:batch size  8, image/sec: 21.283966
INFO:root:batch size 16, image/sec: 24.469325
INFO:root:batch size 32, image/sec: 25.910348
INFO:root:batch size 64, image/sec: 27.441672
INFO:root:batch size 128, image/sec: 28.009156
INFO:root:batch size 256, image/sec: 28.918950
INFO:root:network: vgg
INFO:root:device: cpu(0)
INFO:root:batch size  1, image/sec: 3.980907
INFO:root:batch size  2, image/sec: 2.392069
INFO:root:batch size  4, image/sec: 3.610553
INFO:root:batch size  8, image/sec: 4.994450
INFO:root:batch size 16, image/sec: 6.396612
INFO:root:batch size 32, image/sec: 7.614288
INFO:root:batch size 64, image/sec: 8.826084
INFO:root:batch size 128, image/sec: 9.193653
INFO:root:batch size 256, image/sec: 9.991472
```

It shows that NNPACK will speed up about 2X~7X against the original MXNet cpu.

### Tips

NNPACK aims to provide high-performance implementations of some layers for multi-core CPUs, so you can easily set the thread number by change environment value of `MXNET_CPU_NNPACK_NTHREADS`. but we found that the performance is not proportional to the number of threads, suggest use 4~8 threads when using NNPACK.
