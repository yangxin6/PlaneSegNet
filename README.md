# PlaneSegNet

## Network
![Fig6.jpg](imgs%2FFig6.jpg)

## Environment

```bash
sudo apt-get install libsparsehash-dev

conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# We use CUDA 11.8 and PyTorch 2.1.0 for our development of PTv3
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

cd libs/pointgroup_ops
python setup.py install
cd ../..


# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu118  # choose version match your local cuda version

# Open3D (visualization, optional)
pip install open3d
 conda install -c conda-forge gcc


# MinkowskiEngine
sudo apt install libopenblas-dev
conda install -c conda-forge blas openblas
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd lib/MinkowskiEngine
python setup.py install --blas_include_dirs=/opt/anaconda3/envs/pointcept2/include --blas=openblas
```


## Dataset

We conducted tests on a total of 20 datasets obtained from different types of sensors. 

datasets [link](https://www.kaggle.com/datasets/yangxin6/maize-population-for-semantic-segmentaiton)


Additionally, we express our gratitude to several scholars who shared their data with us. We processed and annotated these data for testing purposes. The original links to these data include:
- [(q)](https://www.mdpi.com/2077-0472/12/9/1450)
- [(r,s,t)](http://arxiv.org/abs/2107.10950)


## Train

```bash
python tools/train.py --config-file configs/corn3d_group_semantic/full/semseg-spvunet-v1m2-base.py
```

## Test

```bash
python tools/test.py --config-file configs/corn3d_group_semantic/full/semseg-spvunet-v1m2-base.py  --options save_path="{weight_path}"  weight="{weight_path}/model_best.pth"
```
We provide our best model weights here: [model_pth](https://www.kaggle.com/datasets/yangxin6/planesegnet-model-pth)


## Reference
- [Pointcept](https://github.com/Pointcept/Pointcept)

## Citation

If you find this project useful in your research, please consider cite:
