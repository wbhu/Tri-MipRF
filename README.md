# Tri-MipRF

Official PyTorch implementation (coming soon) for the paper:

> **Tri-MipRF: Tri-Mip Representation for Efficient Anti-Aliasing Neural Radiance Fields**
>
> ***ICCV 2023***
>
> Wenbo Hu, Yuling Wang, Lin Ma, Bangbang Yang, Lin Gao, Xiao Liu, Yuewen Ma
>
> <a href='https://arxiv.org/abs/2307.11335'><img src='https://img.shields.io/badge/arXiv-2201.12576-red'></a> <a href='https://wbhu.github.io/projects/Tri-MipRF'><img src='https://img.shields.io/badge/Project-Video-Green'></a>

<p align="center">
<video src="assets/lego.mp4" width="98%"/>
</p>

> <b>Instant-ngp (left)</b> suffers from aliasing in distant or low-resolution views and blurriness in
> close-up shots, while <b>Tri-MipRF (right)</b> renders both fine-grained details in close-ups
> and high-fidelity zoomed-out images.

<p align="center">
<img src="assets/overview.jpg" width="97%"/>
</p>

> To render a pixel, we emit a <b>cone</b> from the camera’s projection center to the pixel on the
> image plane, and then we cast a set of spheres inside the cone. Next, the spheres are
> orthogonally projected
> on the three planes and featurized by our <b>Tri-Mip encoding</b>. After that the feature
> vector is fed into the tiny MLP to non-linearly map to
> density and color. Finally, the density and
> color of the spheres are integrated using volume rendering to produce final color for the pixel.


<p align="center">
<img src="assets/teaser.jpg" width="50%"/>
</p>

> Our Tri-MipRF achieves state-of-the-art rendering quality while can be reconstructed efficiently,
> compared with cutting-edge radiance fields methods, <i>e.g.,</i> NeRF, MipNeRF, Plenoxels,
> TensoRF, and Instant-ngp. Equipping Instant-ngp with super-sampling (named Instant-ngp<sup>↑5×</sup>)
> improves the rendering quality to a certain extent but significantly slows down the reconstruction.

## **TODO**

- [ ] Release source code.

## **Citation**

If you find the code useful for your work, please star this repo and consider citing:

```
@inproceedings{hu2023Tri-MipRF,
        author      = {Hu, Wenbo and Wang, Yuling and Ma, Lin and Yang, Bangbang and Gao, Lin and Liu, Xiao and Ma, Yuewen},
        title       = {Tri-MipRF: Tri-Mip Representation for Efficient Anti-Aliasing Neural Radiance Fields},
        booktitle   = {ICCV},
        year        = {2023}
}
```


## **Related Work**

- [Mip-NeRF (ICCV 2021)](https://jonbarron.info/mipnerf/)
- [Instant-ngp (SIGGRAPH 2022)](https://nvlabs.github.io/instant-ngp/)
- [Zip-NeRF (ICCV 2023)](https://jonbarron.info/zipnerf/)