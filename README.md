# CPPN 
Compositional Pattern Producing Network Implemented in Python3 with PyTorch

This should work **out of the box** with just a couple packages:

* PyTorch 1.0+, past versions may work but are untested 
* Scipy
* Numpy
* Imageio
* Tifffile

This really has to be played with to get the full extent of the possibilities here. 

If you're here at all then I assume you know what you want. 

Knobs to tune:
* Dimensionality of uniform sampling distribution ('--z') 
* Scaling factor on sample magnitude (--scale)
* Generator Depth
* Generator layer width (--net)
* Output resolution (---x_dim, --y_dim)
* Output channels (1 for bw, 3 for RGB) (--c_dim)
* Activation functions: `tanh`, `elu`, `sin`, `cos`, etc. 

To make all the images for a looped video, with 10 images and 50 interpolation frames between each image"

`python3 cppn.py --walk --y_dim 512 --x_dim 512 --scale 10 --net 32 --c_dim 1 --n 10 --interpolation 50`

Use this ffmpeg command to make an mp4 from generated frames

`ffmpeg -framerate 7 -i <fn>_%d.jpg -c:v libx264 -crf 23 output.mp4`

To generate a single 1080x1080 grayscale image:

`python3 cppn.py --sample --n 1 --y_dim 1080 --x_dim 1080 --scale 10 --net 32 --c_dim 1 --exp test`

### Metadata Retrieval and Reproducing

Each image is saved in pairs, with both a lightweight png and a tiff file for each generated image. The tiff file has metadata corresponding to the random seed (torch and numpy) and the noise sample (z) used to generate the image. These are useful for reproduction. In this way we can generate hundreds of small images quickly, and choose which ones we want to regenerate in higher resolution. 

Say that we generated a single image with: 

`python3 cppn.py --sample --n 1 --y_dim 256 --x_dim 256 --scale 10 --net 32 --c_dim 3 --exp test --name_style simple`

We can reproduce this image, increasing the resolution to 1024:

`python3 reproduce_images.py --file --name trials/test/image_0.tif --exp test_1024 --x_dim 1024 --y_dim 1024`

We can do this in a batched mode, by upscaling every image in a given directory. 

`python3 reproduce_images.py --dir --name trials/test --exp test_1024 --x_dim 1024 --y_dim 1024`

### Results 

![results](results/normal_z_3_2.png) 

![results](results/sin3_1.png)

![results](results/sin_2.png)

![results](results//test_4.png)

![results](results/sin_mix_3.png)

![results](results/sin_mix3_2.png)

Some of this code, specifically the coordinates function I borrowed from [hardmaru](https://github.com/hardmaru/cppn-tensorflow). Its a good implementation, but it uses TF (which I personally find hard to parse)
