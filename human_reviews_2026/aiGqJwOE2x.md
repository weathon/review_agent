# InfoScan: Information-Efficient Visual Scanning via Resource-Adaptive Walks

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
High-resolution visual representation learning remains challenging due to the quadratic complexity of Vision Transformers and the limitations of existing efficient approaches, where fixed scanning patterns in recent Mamba-based models hinder content-adaptive perception.
To address these limitations, a novel Information-aware Scanning mechanism (InfoScan) tailored for state-space visual backbones is proposed, which dynamically allocates computational resources to the most salient regions of an image. 
Specifically, InfoScan rigorously assesses the informativeness of image patches by integrating entropy with local structural analyses, formulates a joint optimization objective balancing fine-grained detail preservation and broader contextual coherence, and learns an adaptive scanning policy via reinforcement learning. 
Built upon the innovative Visual Information State Space (VISS) block, InfoScan establishes a new family of models that achieve superior efficiency-accuracy trade-offs across diverse tasks. 
Extensive empirical evaluation in different downstream vision tasks demonstrates that our information-driven dynamic scanning paradigm offers a robust and principled alternative to fixed or global-first traversal methods. 
Collectively, our work positions adaptive, content-aware processing as a promising and effective new paradigm for efficient high-resolution visual representation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes InfoScan, an SSM-based visual foundation model developed to improve the scanning strategy from deterministic, which has been adopted in most existing related studies, to adaptive / content-aware. The authors propose an approach to estimate the importance of a certain image patch, and more inspiringly, a reinforcement learning-based formulation of the patch selection algorithm. Experimental results demonstrate the clear performance gain brought by the InfoScan method and the contribution of each newly proposed modules.

### Strengths
Follow the instruction, the strengths of this manuscript are summarized in the following four aspects:

1. **Originality**: The motivation of this study is intuitive and reasonable. Although it may be straightforward or even easy to think of the idea, this work is one of the first studies that works it out (to the best of the reviewers' knowledge).

2. **Quality**: This manuscript is generally of good quality, including the contant organization of language usage.

3. **Clarity**: The clarity of the manuscript is satisfactory. The central idea can be received despite the appendix is frequently mentioned to help make clear of some technical details.

4. **Significance**: This work may be inspiring to researchers in fields like (Visual) SSMs and representation learning.

### Weaknesses
Although this work is somehow inspiring to me, there is much room for further improvement and more information is required to develop a more comprehensive evaluation.

**1. The Computation Overhead:** The content-agnostic scanning patterns are there for a reason. It is crucial to quantify the extra burden imposed by the new content-aware modules. Figure 1 reports FPS and GPU memory versus Vim-Ti and DeiT-Ti, yet omits VMamba, which is the principal baseline throughout the paper. To be concrete, a detailed comparison on the computational complexity in terms of throughput (train and test), GPU memory, FLOPs are needed to gain a comprehensive evaluation of the efficiency of the proposed method. Without these metrics, it is difficult to judge whether InfoScan’s accuracy gains justify its added complexity.

**2. Qualitative Analysis**: InfoScan is motivated by its ability to adapt the scan path to image content, but no qualitative evidence is shown. Visualizations of the learned scanning routes overlaid on sample images (or attention heatmaps) would help readers verify that the model behaves as intended. Quantitative scores alone cannot reveal whether improvements stem from genuine content awareness or simply from a larger, more intricate architecture.

**3. Insufficient Benchmark Methods:** In the main quantitative experiments, the authors mainly compare InfoScan to the benchmarking CNN and ViT models, as well as VMamba and Vim, which are too restricted. More CNN/ViT/SSM-based models recently proposed in top venues should be included for comparison. For instance, Deformable Mamba, another Visual Mamba-related work with deformable scanning route, appearing in CVPR 2025, should at least be included for comparison.

**4. Heavy Manual Design:** InfoScan relies on numerous hand-crafted mechanisms, hyper-parameters, and loss terms, raising questions about its generalizability and robustness. Please discuss the sensitivity of performance to these design choices and, if possible, include ablation studies or automatic tuning results to demonstrate stability across datasets and tasks.

### Questions
Please refer to the 'Weaknesses' section for the details of my concerns. I am willing to improve my ratings if the first three concerns are addressed by concrete experimental results and illustrations.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces InfoScan, an information-aware dynamic scanning mechanism for high-resolution visual representation learning. Unlike fixed or global scanning methods in prior Mamba-based models, InfoScan allocates computation adaptively to the most informative image regions by combining entropy-based and structural analysis. It optimizes for both fine detail and contextual coherence through a reinforcement-learned adaptive scanning policy. Built on the Visual Information State Space (VISS) block, InfoScan achieves a superior efficiency–accuracy trade-off across various vision tasks, highlighting content-aware processing as a promising direction for efficient high-resolution vision models.

### Strengths
1. InfoScan demonstrates strong performance and efficiency, significantly outperforming VMamba.

2. The ablation studies on reward components, patch selection/path planning, and scanning methods are comprehensive, validating the effectiveness of the proposed approach.

3. The design logic of InfoScan is sound—it allocates more computation to complex regions, effectively balancing performance and computational cost.

### Weaknesses
1. More visualizations of the learned policy model for path planning could be provided to illustrate how simple and complex regions correspond to different scanning patterns.

### Questions
1. What is the overall objective of InfoScan? How the policy model for path planning and downstream tasks are jointly optimized?

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
4

### Summary
This paper introduces InfoScan，a visual scanning method for a mamba-based vision backbone. The core innovation is an information-aware scanning mechanism that dynamically allocates computational resources to the most salient image patches. This is achieved through three key components: an Information Scoring Module that quantifies patch informativeness using Shannon entropy and local variance, a Patch Selection Model that optimizes patch size for efficiency and fidelity, and a Path Planning Module that uses Reinforcement Learning to learn an adaptive scanning policy. Extensive experiments on image classification, object detection, and semantic segmentation demonstrate state-of-the-art or highly competitive performance.

### Strengths
1. The motivation and method are novel.  The idea of replacing fixed, content-agnostic scanning patterns with a dynamic, information-driven policy is reasonable. The motivation is clearly grounded.

2. The Methodology is good.  It rigorously formalizes the problem with a joint optimization objective and a reinforcement learning setup. The combination of information theory, variance, and RL is good, which may impact other researchers.

3.  The paper provides extensive evaluations across multiple core vision tasks and domains. The results are compelling, consistently showing performance improvements over strong baselines (CNNs, ViTs, SSMs). The gains in zero-shot cross-dataset generalization are good.

### Weaknesses
1. While the overall model is parameter-efficient, the paper does not provide a detailed analysis of the latency overhead introduced by the Information Scoring Module and the iterative, sequential decision-making of the Path Planning Module during inference. 

2. The hyperparameters of the model architecture are not presented. More technical details should be well presented.

3. The information importance evaluation method should be explained more solidly or provide more ablation studies.

4. More previous Mamba-based vision backbones should be included.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work aims to solve a challenging task of high-resolution visual representation learning which is constrained by the quadratic complexity of popular Transformer-based networks and fixed scanning patterns of Mamba. The authors improve the Mamba scanning mechanism by designing a reinforcement learning based methods learning the adaptive scanning policy. Instead of treating each image patch equally, this work combines local structural information and entropy to evaluate the information values of image patches, allocating more computational resources to the salient ones. Extensive experiments on various computer vision tasks demonstrate the favorable accuracy-efficiency balance of this work.

### Strengths
a.	This work is well organized and easy to follow. The authors effectively identify the weakness of existing Mamba-based methods which overlook the content-adaptive perception, especially from the aspect of scanning strategy.

b.	This work builds a series of vision backbones that achieve superior performance than existing methods with similar numbers of parameters.

c.	The authors design a dynamic scanning strategy by building this task as a Markov decision process and solving it by the reinforcement learning algorithm.

### Weaknesses
i.	The authors only evaluate their methods on pure vision based tasks. Recently, the vision transformers also show high performance on the vision-language large models (VLLMs), such as LLava [1]. Could the proposed vision backbones be applied on these applications?

ii.	Is the training process of the scanning policy stable? Please compare the computational costs in the training phase between your method and existing Mamba-based methods.

iii.	Figure 1 is not mentioned and introduced in this manuscript.

iv.	Does the scanning policy bring any randomness in the testing phase? Is it sensitive to the image resolution? Please analyze the behaviors of the scanning policy if the training images and testing images have different resolutions.

v.	Does the policy network have to repeatedly run on each network block for every image?

vi.	The proposed method has too many new hyperparameters, such k2, k3, beta, than original Mamba.

[1] Liu H, Li C, Wu Q, et al. Visual instruction tuning[J]. Advances in neural information processing systems, 2023, 36: 34892-34916.

### Questions
See weakness.

### Soundness
4

### Presentation
4

### Contribution
3
