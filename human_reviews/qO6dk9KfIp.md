# Laplace Sample Information:  Data Informativeness Through a Bayesian Lens

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Accurately estimating the informativeness of individual samples in a dataset is an important objective in deep learning, as it can guide sample selection, which can improve model efficiency and accuracy by removing redundant or potentially harmful samples. 
We propose $\text{\textit{Laplace Sample Information}}$ ($\mathsf{LSI}$) measure of sample informativeness grounded in information theory widely applicable across model architectures and learning settings.
$\mathsf{LSI}$ leverages a Bayesian approximation to the weight posterior and the KL divergence to measure the change in the parameter distribution induced by a sample of interest from the dataset.
We experimentally show that $\mathsf{LSI}$ is effective in ordering the data with respect to typicality, detecting mislabeled samples, measuring class-wise informativeness, and assessing dataset difficulty.
We demonstrate these capabilities of $\mathsf{LSI}$ on image and text data in supervised and unsupervised settings.
Moreover, we show that $\mathsf{LSI}$ can be computed efficiently through probes and transfers well to the training of large models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper considered an information theoretical framework in understanding the data sample importance. Specifically, the intuition is further formulated as the difference of parameter distributions. Experiments on various and extensive experiments demonstrate the effectiveness of the proposed methods.

### Strengths
- I think this paper raised an important problem in explaining the important data sample. 
- The proposed method is theoretically principled with Bayes principle. 
- Extensive experimental results are presented with CV and NLP datasets (since this method is model agnostic). 
- The paper is clearly written and accessible.

### Weaknesses
There are two major weak points within the paper. Therefore I currently provide a borderline score. 

1. About the novelty. How this method is different from the influence function based method. Both methods require some calculation on the higher-order derivative. Based on the given context, I find it a bit hard to differentiate the differences. 
2. About the calculation LSI in line 212, I could not understand how exactly \hat_{theta}^{-i} and \Sigma^{-i} are estimated. Here the technical detail is very limited, therefore I could not understand why estimating these can be done after training. Please give clear details how LSI is calculated. 
3. (Minor) The paper is a bit large by itself, 32Mb ! I could not properly read the results due to its large file.

### Questions
See the questions in weakness sections.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper describes a novel approach to measuring the informativeness of individual samples in machine learning datasets, named the Laplace Sample Information (LSI). LSI is designed to quantify how much information each sample contributes to the model's parameters using a Bayesian approximation. The method applies the Laplace approximation to generate a distribution over model parameters and subsequently employs Kullback-Leibler divergence to measure each sample's unique contribution to this distribution. 

Experiments on CIFAR-10, CIFAR-100, ImageNet subsets (Imagewoof and Imagenette), a pediatric pneumonia dataset, and the IMDb text sentiment dataset. LSI revealed that only a small subset of samples contributes significant information to the model, while most samples have low informativeness. On CIFAR-10 with 10% label noise, mislabeled samples generally had higher LSI values than correctly labeled samples, showing that LSI can effectively highlight mislabeled data. Other experiments collectively validated LSI’s utility across various applications: ranking samples by informativeness, detecting mislabeled data, assessing dataset and class difficulty, and improving model generalization.

### Strengths
- LSI offers a new perspective on sample informativeness, focusing directly on sample information flow rather than related metrics like sample difficulty. This approach addresses a gap in the current literature, where sample informativeness has often been approximated indirectly.

- LSI is adaptable across different model architectures, learning tasks (supervised and unsupervised), and data modalities (e.g., images, text). LSI has valuable applications in identifying mislabeled samples, estimating dataset difficulty, and detecting out-of-distribution data.

- By using small probe models, the authors make LSI computationally feasible for larger models and datasets. This approach yields a substantial efficiency gain without significantly compromising the integrity of the informativeness rankings.

- The authors evaluate LSI on various datasets, including CIFAR-10, CIFAR-100, ImageNet subsets, and text datasets, with extensive experiments validating LSI’s effectiveness in distinguishing sample informativeness. They also investigate LSI’s behavior under label noise.

### Weaknesses
- The experiments mainly evaluate the proposed LSI measures. How does the proposed measure compare to other existing related measures, such as gradient similarity, influence functions, and TRAK (Park et al., ICML'23)? 

- Although LSI uses a probe model for efficiency, computing LSI still requires repeated training sessions, especially in large datasets. It would be better to describe its computation complexity and compare it with existing measures. 

- The reliance on the Laplace approximation assumes that the model parameters follow a multivariate Gaussian distribution. While the authors justify this through the central limit theorem, it would be better to provide empirical justifications, especially for highly complex or non-convex loss landscapes. 

- The method relies on approximations of the Hessian, particularly when using diagonal approximations for large models. It would be better to analyze how this approximation affects LSI in different neural architectures.

### Questions
Please see weaknessnes

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Laplace Sample Information (LSI), a metric to quantify how much individual data points contribute to a neural network’s learning process. By approximating a Bayesian model through the Laplace method, LSI uses KL divergence to capture each sample’s impact on the model’s parameters, which can be particularly beneficial for detecting mislabeled data, understanding dataset difficulty and informativeness. Empirically, the authors validate LSI across a range of datasets and demonstrate that it transfers well from smaller “probe” models to larger architectures.

### Strengths
* **Originality and Quality**
The approach is novel because it uses a post-hoc Bayesian approximation to evaluate sample informativeness in standard neural networks. The authors include experiment results on multiple datasets, from CIFAR-10 to IMDb, showing that LSI can detect mislabeled samples and assess informativeness across classes. 

* **Clarity and Significance**
The paper is well-organized and easy to follow. As claimed in the paper, LSI is beneficial for identifying potentially mislabeled data, as well as evaluating dataset difficulty and informativeness. The proposed method could have a substantial impact, particularly for data-centric AI and applications where dataset quality is key.

### Weaknesses
**1. Computational Cost**

Calculating LSI requires Hessians, which may be expensive on large datasets. While the probe model helps, a more detailed breakdown of computation time, evaluation quality, and scalability would be helpful for readers to understand the effectiveness of LSI.

**2. Limitations of Probe Model** 

While using a smaller probe model makes computation faster, it could introduce discrepancies in sample rankings when applied to larger models. Could the authors provide more insight into how well LSI rankings from the probe model align with those from larger models?

**3. Understanding the Effectiveness of LSI**

During the investigation of low/high LSI samples, is it possible to have statistical measurements about the comparisons/differences between high and low LSI samples. It is not clear if the observation of the difference among a few samples can generalized to the larger scale. Besides, there are no baseline performances reported in this paper. (i.e., for the detection of mislabeled samples, I believe there are certain methods already working on it.)

**4. Experiments on Real-world Mislabeled Samples**

In the section ```Focusing on Mislabeled Samples```, it would be more beneficial if authors could test the effectiveness of LSI on real-world noise (CIFAR-10N [R1], or CIFAR-10H [R2]), instead of synthetic ones (random label flips). 

**References:**

R1: Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations. ICLR 2022.

R2: Human uncertainty makes classification more robust. ICCV 2019.

### Questions
Pleaser refer to the weakness section.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a measure named Laplace Sample Information (LSI) to estimate the informativeness of individual samples in a dataset for deep learning methods. Specifically, LSI leverages a Bayesian approximation to the weight posterior and the KL divergence to measure the change in the parameter distribution induced by a leave-one-out sample from the dataset. Experimental results show that LSI is effective in ordering the data w.r.t. typicality, detecting mislabeled samples, measuring class-wise informativeness, and assessing dataset difficulty.

### Strengths
1. Overall, this paper is well-written and easy to follow.
2. This paper considers the important task of estimating the sample information.
3. Experimental results illustrate the effectiveness of the proposed measure LSI.

### Weaknesses
1. The novelty of the proposed method might be limited as the Laplacian approximation is widely used in the Bayesian machine learning community.
2. Theoretically, formal results are lacking to support the effectiveness of the proposed method.
3. Large experiments involving large datasets and models are lacking, which results in that this work might have little influence in practice.

Minor issues:
1. Typos exist in Lines 194-195: shownAppendix.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
2
