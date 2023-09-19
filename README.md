# Tensorflow Implementation of Image2Stylegan++
This repository is Tensorflow Implementation of Image2StyleGAN++ proposed by [Image2StyleGAN++: How to Edit the Embedded Images?](https://arxiv.org/pdf/1911.11544.pdf). The StyleGAN structure is mainly based on [taki0112 StyleGAN-Tensorflow](https://github.com/taki0112/StyleGAN-Tensorflow) architecture. In this repository, we only implement reconstruction, crossover, inpainting and style transfer(Image2StyleGAN).

## Installation
* Clone this repository

    ```shell script
    git clone https://github.com/Rayhchs/Image2Styleganv2-Implmentation.git
    ```

* Setup environment

	* python3.8

    ```shell script
    pip install -r requirements.txt
    ```

* Download stylegan checkpoint from [here](https://drive.google.com/file/d/1R4TbOTalsleaK-s-f2VV9b5n8GGYh0t8/view?usp=drive_link).

Or you want to convert your pretrained checkpoint, you can follow:

### Checkpoint Conversion
This repository converts [NVlabs](https://github.com/NVlabs/stylegan) pretrained model into [taki0112](https://github.com/taki0112/StyleGAN-Tensorflow) architecture using [stylegan convert architecture](https://github.com/aydao/stylegan-convert-architecture). Or you can download the converted pretrained model from [here](https://drive.google.com/file/d/1R4TbOTalsleaK-s-f2VV9b5n8GGYh0t8/view?usp=drive_link).

## Usage

```shell script
sh run.sh
```

That's it. Image results will be saved to `./result`.

### Important Arguments
|    Argument    |                                                                                                       Explanation                                                                                                       |
|:--------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      phase      | Which process to be implement including reconstruction, crossover, inpainting, style_transfer |
|    img_a   | Filename of master image |
|    img_b   | Filename of slave image |
|    test_folder   | Folder of img_a and img_b |
|    mask   | Mask for inpainting and crossover |
|    epoch_w   | Iteration for w+ space optimization |
|    epoch_n   | Iteration for noise space optimization |

## Advanced Usage

This repository uses totally different lambda value from papers. If you are interested, you can modify args.lambda_mse, args.lambda_p, args.lambda_mse1, args.lambda_mse2 in main.py for your implementation.

## Results

| Phase | Master image | Slave image | Transformed image |
|:-------------------------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|
| Reconstruction | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/obama.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/obama.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/results/reconstruction_obama.jpg" width="256">|
| Reconstruction | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/Ryan.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/Ryan.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/results/reconstruction_Ryan.jpg" width="256">|
| Crossover, mask=[350, 650, 650, 850] | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/obama.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/Ryan.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/results/crossover_obama.jpg" width="256">|
| Crossover, mask=[350, 650, 650, 850] | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/Ryan.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/obama.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/results/crossover_Ryan.jpg" width="256">|
| Inpainting, mask=[350, 650, 650, 850] | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/obama.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/obama.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/results/inpainting_obama.jpg" width="256">|
| Inpainting, mask=[350, 650, 650, 850] | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/Ryan.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/Ryan.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/results/inpainting_Ryan.jpg" width="256">|
| Style Transfer | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/obama.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/sketch.png" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/results/sketch.png_obama.jpg" width="256">|
| Style Transfer | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/Ryan.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/sketch.png" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/results/sketch.png_Ryan.jpg" width="256">|
| Style Transfer | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/obama.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/cartoon.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/results/style_transfer_obama_cartoon.jpg" width="256">|
| Style Transfer | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/Ryan.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/cartoon.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/results/style_transfer_Ryan.jpg" width="256">|
| Style Transfer | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/obama.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/starry_night.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/results/starry_night_obama.jpg" width="256">|
| Style Transfer | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/Ryan.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/dataset/starry_night.jpg" width="256"> | <img src="https://github.com/Rayhchs/Image2Styleganv2-Implmentation/blob/main/results/starry_night_obama.jpg" width="256">|

## Acknowledge
* Tensorflow StyleGAN structure is borrowed from [taki0112 StyleGAN-Tensorflow](https://github.com/taki0112/StyleGAN-Tensorflow).

* Pretrained model is borrowed from [NVlabs](https://github.com/NVlabs/stylegan).

* Model conversion is based on [stylegan convert architecture](https://github.com/aydao/stylegan-convert-architecture)

Thanks for amazing works!

## References
* StyleGAN
```bibtex
@article{DBLP:journals/corr/abs-1812-04948,
  author       = {Tero Karras and
                  Samuli Laine and
                  Timo Aila},
  title        = {A Style-Based Generator Architecture for Generative Adversarial Networks},
  journal      = {CoRR},
  volume       = {abs/1812.04948},
  year         = {2018},
  url          = {http://arxiv.org/abs/1812.04948},
  eprinttype    = {arXiv},
  eprint       = {1812.04948},
  timestamp    = {Tue, 01 Jan 2019 15:01:25 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1812-04948.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
* Image2StyleGAN
```bibtex
@article{DBLP:journals/corr/abs-1904-03189,
  author       = {Rameen Abdal and
                  Yipeng Qin and
                  Peter Wonka},
  title        = {Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?},
  journal      = {CoRR},
  volume       = {abs/1904.03189},
  year         = {2019},
  url          = {http://arxiv.org/abs/1904.03189},
  eprinttype    = {arXiv},
  eprint       = {1904.03189},
  timestamp    = {Sat, 23 Jan 2021 01:19:25 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1904-03189.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
* Image2StyleGAN++
```bibtex
@article{DBLP:journals/corr/abs-1911-11544,
  author       = {Rameen Abdal and
                  Yipeng Qin and
                  Peter Wonka},
  title        = {Image2StyleGAN++: How to Edit the Embedded Images?},
  journal      = {CoRR},
  volume       = {abs/1911.11544},
  year         = {2019},
  url          = {http://arxiv.org/abs/1911.11544},
  eprinttype    = {arXiv},
  eprint       = {1911.11544},
  timestamp    = {Mon, 26 Jun 2023 20:49:40 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1911-11544.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```