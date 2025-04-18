# 📋 Specular Highlight Removal Benchmark

[Dual-Hybrid Attention Network for Specular Highlight Removal](https://arxiv.org/abs/2407.12255)

[Xiaojiao Guo](https://orcid.org/0009-0002-9177-8266) 👨‍💻‍ , [Xuhang Chen](https://cxh.netlify.app/) 👨‍💻‍ , [Shenghong Luo](https://shenghongluo.github.io/), [Shuqiang Wang](https://people.ucas.edu.cn/~wangshuqiang?language=en) 📮 and [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/) 📮 ( 👨‍💻‍ Equal contributions, 📮 Corresponding authors)

**University of Macau, SIAT CAS, Huizhou University, Baoshan University**

2024 ACM International Conference on Multimedia (ACM MM 2024)

## [Code](https://github.com/CXH-Research/DHAN-SHR) | [Benchmark dataset (Kaggle)](https://www.kaggle.com/datasets/xuhangc/acm-mm-2024-dehighlight-dataset)

# 🔮 Dataset

The comprehensive benchmark dataset is available at Kaggle.

# ⚙️ Usage

## Training

You may download the dataset first, and then specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in `config.yml`.

For single GPU training:

```bash
python train.py
```

For multiple GPUs training:

```bash
accelerate config
accelerate launch train.py
```

If you have difficulties with the usage of `accelerate`, please refer to [Accelerate](https://github.com/huggingface/accelerate).

## Inference

Please download our pre-trained models [here](https://github.com/CXH-Research/DHAN-SHR/releases/tag/Weight) and specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in `config.yml`, then execute:

```bash
python test.py
```

# 💗 Acknowledgement

This work was supported in part by the Science and Technology Development Fund, Macau SAR, under Grant 0141/2023/RIA2 and 0193/2023/RIA3, in part by the National Key Research and Development Program of China (No. 2023YFC2506902), the National Natural Science Foundations of China under Grant 62172403, the Distinguished Young Scholars Fund of Guangdong under Grant 2021B1515020019.

# 🛎 Citation

If you find our work helpful for your research, please cite:

```bib
@inproceedings{guo2024dualhybrid,
  title={Dual-Hybrid Attention Network for Specular Highlight Removal},
  author={Xiaojiao Guo and Xuhang Chen and Shenghong Luo and Shuqiang Wang and Chi-Man Pun},
  booktitle={ACM Multimedia 2024},
  year={2024},
  url={https://openreview.net/forum?id=9D8waGSbCZ}
}
```

