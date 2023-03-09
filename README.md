# D2Net
This is the Pytorch implementation of the paper [Deep Denoising Network in Frequency Domain for Hyperspectral Image](https://ieeexplore.ieee.org/abstract/document/9910374).

## Introduction
Since the existing hyperspectral image denoising methods suffer from excessive or incomplete denoising, leading to information distortion and loss, this letter proposes a deep denoising network in the frequency domain, termed D2Net. Our motivation stems from the observation that images from different HSI bands share the same structural and contextual features while the reflectance variations in the spectra are mainly fallen on the details and textures. We design the D2Net in three steps: (1) spatial decomposition, (2) spatial-spectral denoising, and (3) refined reconstruction. It achieves multi-scale feature learning without information loss by adopting the rigorous
symmetric discrete wavelet transform (DWT) and inverse discrete wavelet transform (IDWT). In particular, the specific design for different frequency components ensures complete noise removal and preservation of fine details. Experiment results demonstrate that our D2Net can attain a promising denoising performance.

## Citation
If you find our work useful in your research or publication, please cite:

```latex
@article{pan2022sqad,
  @ARTICLE{9910374,
  author={Pan, Erting and Ma, Yong and Mei, Xiaoguang and Huang, Jun and Fan, Fan and Ma, Jiayi},
  journal={IEEE/CAA Journal of Automatica Sinica}, 
  title={D2Net: Deep Denoising Network in Frequency Domain for Hyperspectral Image}, 
  year={2023},
  volume={10},
  number={3},
  pages={813-815},
  doi={10.1109/JAS.2022.106019}}
```
