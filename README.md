# PE3R: Perception-Efficient 3D Reconstruction
<div align="center">
  <img src="./imgs/pe3r.gif" width="100%" ></img>
  <br>
  <em>
      (PE3R reconstructs 3D scenes using only 2D images and enables semantic understanding through language.) 
  </em>
</div>
<br>

> **PE3R: Perception-Efficient 3D Reconstruction**    
> [Jie Hu](https://hujiecpp.github.io), [Shizun Wang](https://littlepure2333.github.io/home/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)    
> [xML Lab, National University of Singapore](https://sites.google.com/view/xml-nus/home?authuser=0)    
> 📔 [[paper]](https://arxiv.org/abs/2503.07507) 🎥 [[video]](https://youtu.be/iFRijE4GQv4) 🤗 [[demo]](https://huggingface.co/spaces/hujiecpp/PE3R)

### Why PE3R
* 🚀 Input efficiency: Operate solely with 2D images.
* 🚀 Time efficiency: Accelerated 3D semantic reconstruction.
* 🚀 Generalizability: Zero-shot generalization across scenes and objects.

### Quick Start
#### Install
```bash
conda create --name pe3r
conda activate pe3r
git clone 
pip install requirements.txt
```
#### Usage
```bash
python pe3r_demo.py
```

### Acknowledgements
- [DUSt3R](https://github.com/naver/dust3r) / [MASt3R](https://github.com/naver/mast3r)
- [SAM](https://github.com/facebookresearch/segment-anything) / [SAM2](https://github.com/facebookresearch/sam2) / [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [SigLIP](https://github.com/google-research/big_vision)

### BibTeX
```BibTeX
@article{hu2025pe3r,
  title={PE3R: Perception-Efficient 3D Reconstruction},
  author={Hu, Jie and Wang, Shizun and Wang, Xinchao},
  journal={arXiv preprint arXiv:2503.07507},
  year={2025}
}
```
