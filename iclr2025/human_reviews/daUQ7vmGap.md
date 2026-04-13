## Human Reviewer 1

### Summary
The following paper explores the effect of conducting Dynamic Sparse Training (DST) within a neural network towards its robustness to adversarial samples. First, the paper introduces the Dynamic Sparsity Corruption Hypothesis, which postulates that performing DST on artificial neural networks at a low sparsity regime improves its robustness against image and video corruption tasks compared to dense training. Next, The paper proceeds to validate this hypothesis by conducting experiments on various DST algorithms under different deep learning architectures, data modalities, and data corruption scenarios and their severity and comparing their robustness and accuracy with dense training. Then, the authors analyze spectral and spatial domains to show that DST works as an implicit regularizer for neural networks by reducing its attention to high-frequency components or less meaningful features.

### Strengths
1. Writing is easy to follow, and the content organization is clear, making the paper easy to read and comprehend.
2. A comprehensive set of evaluations is conducted to show Dynamic Sparse Training (DST) achieves better performance in terms of robustness accuracy than its dense counterparts.
3. While this paper does not propose a new method that improves state-of-the-art performance at a specific benchmark, it does provide new insights into the unexpected benefit of performing DST on image and video corruption tasks in favor of dense training on varying neural network architectures.

### Weaknesses
I hope the authors can provide revision for the paper based on the following points:

1. There is no detailed explanation for several experiments conducted in the main paper. Such details are as follows: 
- The experiment was conducted on the ImageNet-3DCC dataset in Section 4.3. One needs to find Table 3 at the end of page 10 to see the model's sparsity ratio employed for the experiment. Even then, there is no detailed explanation of how to train the model for experiments at both the ImageNet-3DCC dataset and the UCF101 dataset for video representation.
- There are lack of information regarding the details of the experiment conducted in Section 6.1. It only briefly mentions that it uses an MLP model without explicitly mentioning its detailed architecture. In addition, I fail to find details/recipes on how you train the model to achieve results in Table 2.

2. I also don't get why the sparsity ratio differs for experiments in different Sections (namely Section 4.1 and the rest of the subsections in Section 4). Namely, in section 4.1, the authors are trying to evaluate the robustness accuracy in various sparsity ratios ranging from 0.3 to 0.7, whereas the rest of the experiments are conducted at a low sparsity regime of 0.1. I also want to ask the same question regarding the motivation in setting the initial density of GraNet to be 0.8 and the motivation for setting the soft memory bound for MEST to 10% of the target density. I hope the author can address the reasoning for doing so in their rebuttal.

3. The main claims that this paper is trying to address are a bit problematic for me. For example, the DSCR hypothesis mentions a "low sparsity regime", but the authors fail to quantify which range of values would be suitable to be defined as a "low" sparsity regime and one that is not in principle. This issue, I think, connects with issue 2 pointed out above, which makes the sparsity constraint $S$ for different experiments about different datasets different.

4. (Minor) These are things that might need to be revised in the paper and the appendix:
- "Y-axis in Figure 3" should be "x-axis in Figure 3". [Section 4.2]
- **ImageNet-C** -> "generated from Tiny-ImageNet" should be "generated from ImageNet" [Appendix A.2]
- **CIFAR-100** -> ReNet-34 should be ResNet-34 [Appendix A.3.2]
- "(c)" should be "(a)" [Figure 7 Right-Top] (I think it's better to put (a), (b), and (c) below the bottom figure and then explain the right figure without mentioning the top and bottom part so that (d), (e), and (f) don't get wasted there).

### Questions
1. In Figure 3 within the main paper, what do these five bars represent in each corruption type? Please indicate this in a legend within one of the graphs for clarity.
2. Regarding analyses conducted in Section 5.1 and Section 5.2, what are the motivations for using the random regrowth procedure for both MEST and GraNet instead of their gradient-based regrowth variants for comparison in Figure 5 and Figure 7? I am asking because the results displayed on the paper (Figure 3, Figure 4, Table 1, and Table 3) use the latter instead of the former.
3. What causes each DST method to achieve worse performance in a particular type of corruption (i.e., single_freq (single frequency greyscale) in ImageNet-$\bar{\mathbf{C}}$)?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper explores whether Dynamic Sparse Training (DST) results in more OOD Robust Models than Dense Training.
To this extent, the paper reports its findings using various DST methods over multiple architectures and different datasets.
The findings are consistent across experiments that DST methods lead to more OOD robust models than Dense Training.

### Strengths
The structure of the paper is quite different from usual submissions and is a welcome change, the style of having research questions for section headings and related experiments, findings, and discussion in those sections is interesting and keeps the reader engaged.

### Weaknesses
**W1- Lack of Proper Literature Review.**
The biggest weakness of this work is that it only analyzes its findings from the narrow lens of "Dynamic Sparse Training" because almost all of the questions asked in this work have been asked before from a larger context of Deep Neural Networks and Sparse Deep Neural Networks and have been successfully answered. 
The related work section of this paper is very poorly written and researched.
Had significant effort been invested into finding true related works done by the community this paper would have been significantly different. 
For example, in section 5.1 the paper attempts to find a correlation between inherent properties of dynamic sparse training and increased OOD robustness, however in their work [1] already showed that sparse models can be very OOD Robust, and while this proposed work on an average looks only up to 50% sparsity, [1] showed that 90% sparse models can be more robust than their dense counterparts. While [1] does not specifically look into DST, the overall observations from [1] and this work are not very dissimilar, and thus maybe there is more at play here than "inherent properties of DST", that is these observations are not unique to DST and have been explored before and shown that models can be trained to have **"reduced connectivity to specific features that are determined to be less crucial"**.

Another example is in section 5.2 the paper does an analysis using a high-pass filter and a low-pass filter. Here the conclusion in the paper is that low-frequency information is more important, high-frequency information is less important and DST methods learn some kind of inherent regularisation to focus on "more information i.e. low frequencies" and thus are more robust. 
However, the claim that "low-frequency information" is "more important" information and "high-frequency information" is "less important" has not been supported by studies from this paper or from any other paper cited here.
This is essential because another exists to explain the observations that have nothing to do with "less important" and "more important" but finds its basis in signal theory and is therefore more plausible. This explanation is provided by [2] and [3] that show using only "low-frequency information" one can avoid aliasing and avoiding aliasing leads to a reduction in spectral artifacts which leads to increased OOD robustness. So, maybe DST is inherently leading to less aliasing due to the sparsity and is therefore increasing OOD Robustness. 
Therefore, despite the experiments in the paper, the conclusions presented in the paper do not hold without doubt as other plausible explanations exist.

W2 - **Poorly Written.**
While the structure of the paper is quite good, the content is very poorly written and has many mistakes. Just to prove the point I will point out a few, so that similar mistakes can be avoided in the future. 

1. Figure 7 subcaptions are wrong, (a) has been written as (c). 
2. Math variables are not defined, example before eqn 1, it is written M $\in \mathcal{M}$, however $\mathcal{M}$ has not been defined. And in Preliminaries 3.1, $p_{tr}$ and $p_{te}$ are decomposed, but no reason has been given for it, a reason should be provided for doing an operation such as decomposition, moreover there exists $i$=1 in subscripts but $i$ is never defined.


W3- **Missing Important citations**
While I have covered some of the missing important related works in W1, it is paramount to note the missing citation for [4]. The frequency properties of common corruptions are referenced multiple times in this paper however these properties are not general knowledge but findings from [4] and they must be cited for their work.

W4 - **Outdated Data Augmentation techniques used in Appendix A.6**
MixUp is an old data augmentation technique, significantly newer data augmentation techniques exist now. Some of the most common ones while not very new are AugMix and DeepAugment [5], it might be prudent to rather (or additionally) ablate with these in in Appendix A.6.


**References** 

[1] Hoffmann, Jasper, et al. "Towards improving robustness of compressed cnns." ICML Workshop on Uncertainty and Robustness in Deep Learning (UDL). Vol. 2. 2021.

[2] Grabinski, Julia, et al. "Frequencylowcut pooling-plug and play against catastrophic overfitting." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

[3] Grabinski, Julia, Janis Keuper, and Margret Keuper. "Fix your downsampling ASAP! Be natively more robust via Aliasing and Spectral Artifact free Pooling." arXiv preprint arXiv:2307.09804 (2023).

[4] Yin, Dong, et al. "A Fourier perspective on model robustness in computer vision." Advances in Neural Information Processing Systems 32 (2019).

[5] Hendrycks, Dan, et al. "The many faces of robustness: A critical analysis of out-of-distribution generalization." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

### Questions
**Q1- Section 3.1**

I fail to understand the significance of $p_{tr}(y|x)=p_{te}(y|x)$, is it supposed to mean that training and test samples are iid? If yes, then why is mentioned explicitly? What is the significance of explicitly mentioning this detail?

**Q2 - Appendix A.3.2**

To the best of my current understanding, the FLOPs calculated are mere estimations, right? These are not real gains in FLOPs that are materialized by the hardware infrastructure of today? If so, then please mention that in Appendix A.3.2, otherwise, it misleads the reader.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper raises an intuitive research question through a thorough review of dense training, static sparse training, dynamic sparse training, and their counterpart model robustness, "Is somehow DST implicitly improving model robustness". Then, propose the Dynamic Sparsity Corruption Robustness (DSCR) Hypothesis "Dynamic Sparse Training at low sparsity levels helps to improve model robustness against image corruption in comparison with Dense Training." Finally, it validates the DSCR Hypothesis on two types of data, images, and videos, using several traditional and modern deep learning architectures for computer vision and three widely studied Dynamic Sparse Training algorithms. The findings reveal a new yet-unknown benefit of Dynamic Sparse Training and open new possibilities in improving deep learning robustness beyond the current state of the art. I think this is a good paper and worth accepting.

### Strengths
1. Problem statement. This paper clearly narrates the robustness from dense training, static sparse training to dynamic sparse training.
2. Necessary and sufficient experiments. From convolution-based network to transformer-based network, this paper makes a thoroughly empirical experiment to justify the robustness of dynamic sparse training.
3. Besides the above, section 5 explores the inner reasons for the robustness of dynamic sparse training models. Analyzing how DST and Dense training models utilize their parameters in the spatial domain and process the input in the spectral domain.

### Weaknesses
The research direction of robustness only focus on the classification task (with dataset cifar10/100-c, ImageNet-c), why not extend to new tasks/datasets like segmentation, pose estimation, and image matching et al?

### Questions
same as the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper challenges the conventional use of dense training for model robustness in deep learning. Through extensive experiments across multiple datasets, architectures, and Dynamic Sparse Training (DST) methods, the authors demonstrate that DST can consistently outperform dense training in image corruption robustness. Analysis reveals that DST's superior performance stems from its implicit regularization mechanism, which naturally focuses on low-frequency features while reducing sensitivity to high-frequency components. The findings are validated across both image and video tasks, suggesting new directions for improving neural network robustness.

### Strengths
The experimental evaluation is thorough and well-structured, covering both image and video domains, multiple architectures (CNNs and Transformers), and various corruption types and severity levels.

### Weaknesses
1.	The core contribution appears to be an empirical observation, while the technical contributions are limited.
2.	While the results show improved robustness against corruption, the paper doesn't extensively analyze whether this comes at the cost of other aspects of model performance (e.g., clean accuracy).
3.	There is a clear error in describing Figure 3's axis. The text states "the Y-axis in Figure 3 represents different types of corruption" when it appears to be the X-axis that shows corruption types.
4.	The considered datasets to evaluate the proposed method are limited. There are a lot of robustness benchmarks, including ImageNet-A/R/P. It is unclear whether the proposed method is able to consistently improve robustness across diverse benchmarks.
5.	It would be interesting to compare with SOTA robust training methods, rather than only comparing with the standard dense training.
6.	The comparisons against SOTA robust training methods and robust models should be included, including RVT and FAN.
7.	The comparison of training cost should be reported.

### Questions
More experiments and comparisons with SOTA training methods or robust models would be beneficial.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4