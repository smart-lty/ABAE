# ABAE: Utilize Attention to Boost Graph Auto-Encoder

This is the code for the paper "**ABAE: Utilize Attention to Boost Graph Auto-Encoder, PRICAI 2021**". Our architecture is based on the encoder-decoder model. The components of this model are shown as follows.

![](https://i.loli.net/2021/06/10/exbZRm5M1iP2Fk8.png)

### Environment
1. Create an environment `conda create -n ABAE python=3.7`  `conda activate ABAE`
2. Install relative packages. Please see requirements.txt for a whole view.

### Reproduce
Readers can reproduce the results reported in the paper. There is two steps:
1. Change the hyperparameters by changing the `config.py` file.
2. Run `python main.py` to start training.

### Results

Our architecture has achieved state-of-the-art in *Link Prediction*. Total comparance is given.

![image-20210610103103064](https://i.loli.net/2021/06/10/6VUGZFHD2Jbxhmo.png)

### Citation
Cite this paper as:

Liu T., Li Y., Sun Y., Cui L., Bai L. (2021) ABAE: Utilize Attention to Boost Graph Auto-Encoder. In: Pham D.N., Theeramunkong T., Governatori G., Liu F. (eds) PRICAI 2021: Trends in Artificial Intelligence. PRICAI 2021. Lecture Notes in Computer Science, vol 13032. Springer, Cham. 
