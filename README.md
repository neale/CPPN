# CPPN 
Compositional Pattern Producing Network Implemented in Python3 with PyTorch 0.4.0

This should work **out of the box** with just a couple packages:

* PyTorch 0.4.0+ (1.0+ tested)
* Scipy
* Numpy
* Imageio
* Tifffile

This really has to be played with to get the full extent of the possibilities here. 

If you're here at all then I assume you know what you want. 

Knobs to tune:
* latent **z** distribution (uniform/normal/multimodal)
* scaling factor on **z**
* number of FC layers for complexity
* number of units per FC layer for complexity
* output resolution
* channels in output (1 for bw, 3 for rgb)
* functions on the linear output. `tanh`, `elu`, `sin`, `cos`, etc. 

To make all the images for a looped video

`python3 cppn.py --walk True --y_dim 512 --x_dim 512 --scale 10 --net 32 --c_dim 1`

I've been using this ffmpeg command to make an mp4

`ffmpeg -framerate 7 -i <fn>_%d.jpg -c:v libx264 -crf 23 output.mp4`

To generate a single 1080x1080 grayscale image:

`python3 cppn.py --sample True --n 1 --y_dim 1080 --x_dim 1080 --scale 10 --net 32 --c_dim 1 --exp test`

### Metadata Retrieval

Each image is saved in pairs, with both a lightweight png and a tiff file for each generated image. The tiff file has metadata corresponding to the random seed (torch and numpy) and the noise sample (z) used to generate the image. These are useful for reproduction. In this way we can generate hundreds of small images quickly, and choose which ones we want to regenerate in higher resolution. 


### Results 

![results](results/normal_z_3_2.png) 

![results](results/sin3_1.png)

![results](results/sin_2.png)

![results](results//test_4.png)

![results](results/sin_mix_3.png)

![results](results/sin_mix3_2.png)

Some of this code, specifically the coordinates function I borrowed from [hardmaru](https://github.com/hardmaru/cppn-tensorflow). Its a good implementation, but it uses TF (which I personally find hard to parse)
