# Enabling True Global Perception in State Space Models for Visual Tasks

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Despite the importance of global contextual modeling in visual tasks, a rigorous mathematical definition remains absent, and the concept is still largely described in heuristic or empirical terms. Existing methods either rely on computationally expensive attention mechanisms or are constrained by the recursive modeling nature of State Space Models (SSMs), making it challenging to achieve both efficiency and true global perception. To address this, we first propose a mathematical definition of global modeling for visual images, providing a theoretical foundation for designing globally-aware and interpretable models. Based on in-depth analysis of SSMs and frequency-domain modeling principles, we construct a complete theoretical framework that overcomes the limitations imposed by SSMs' recursive modeling mechanism from a frequency perspective, thereby adapting SSMs for global perception in image modeling. Guided by this framework, we design the Global-aware SSM (GSSM) module and formally prove that it satisfies definitional requirements of global image modeling. GSSM leverages a Discrete Fourier Transform (DFT)-based modulation mechanism, providing precise front-end control over the SSM's modeling behavior, and enabling efficient global image modeling with linear-logarithmic complexity. Building upon GSSM, we develop GMamba, a plug-and-play module that can be seamlessly integrated at any stage of Convolutional Neural Networks (CNNs). Extensive experiments across multiple tasks, including object detection, semantic segmentation, and instance segmentation, across diverse model architectures, demonstrate that GMamba consistently outperforms existing global modeling modules, validating both the effectiveness of our theoretical framework and the rigor of proposed definition. Code is available at \url{https://github.com/Xinmu-Tantai/GMamba-GSSM}

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a novel module, Global-aware SSM (GSSM), which is the core of a plug-and-play block called GMamba. The key idea is to use a Discrete Fourier Transform (DFT)-based modulation mechanism to "inject" global information into the features before they are processed by the SSM. This frequency-domain pre-modulation guides the SSM's state updates with a true global perspective, enabling efficient global modeling with linear-logarithmic complexity

### Strengths
The proposed GMamba block shows consistent and significant performance improvements when added to various backbones (e.g., ResNet, Swin, ConvNeXt) . It outperforms existing global modeling methods, including Transformer and other Mamba variants, on several dense-prediction tasks (semantic segmentation, object detection, and instance segmentation) across multiple datasets.

### Weaknesses
1. The paper's theoretical claims rest heavily on its new definition of global perception—"global gradient dependency". This definition requires that the Frobenius norm of the gradient of the output with respect to any input pixel is bounded by a non-zero constant $\tau$. This is a very low bar. A gradient that is infinitesimally small but non-zero would satisfy this definition, but this may not align with an intuitive or practical understanding of "global influence." The claim of "true" global perception is therefore only as strong as the acceptance of this new, and debatable, definition.


2. Although the authors claim that the proposed GSSM enables efficient global image modeling, there is no related experimental data on practical inference latency or GPU memory cost. Relying only on FLOPs may not accurately reflect the module's true efficiency, as operations like 2D-DFT can have different hardware utilization profiles than standard convolutions.

3. The experimental validation is extensive on dense prediction tasks (segmentation and detection). However, a standard benchmark for vision backbones is ImageNet classification. The absence of this benchmark makes it slightly more difficult to assess the module's generalizability as a fundamental building block for all vision tasks.

4. Some illustrations of previous work in this paper appear to be incorrect and may mislead readers. The authors show the ViM scanning routes in Figure 1(c), but this depiction seems to be wrong (as noted in other work, e.g., [1]). This potential misrepresentation of a key baseline method is a concern.

[1]. Visual mamba: A survey and new outlooks, 2024.

### Questions
Please check the weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the limited global perception of State Space Models (SSMs) in vision tasks by proposing a frequency-domain modulation approach. The proposed GSSM block applies DFT-based frequency decomposition to extract global semantic information and uses adaptive modulation coefficients to guide SSM's state updates before sequential processing, achieving linear-logarithmic complexity O(nlogn). The authors show that the proposed GMamba block is plug-and-play across different CNN architectures. Experiments on semantic segmentation (Vaihingen, Potsdam, LoveDA, UAVid), object detection, and instance segmentation (MS-COCO) demonstrate consistent improvements of 2-3% mIoU across ResNet, Swin Transformer, and ConvNeXt backbones. While the theoretical framework looks good, the practical gains are modest given the added complexity, and the core idea of frequency-domain processing for vision is well-established in prior work.

### Strengths
- The paper provides an interesting mathematical definition of global image modeling and establishes a complete theoretical framework showing how DFT-based modulation can enrich the SSMs with global perception capabilities. 

- The motivation is simple yet effective: SSMs rely on sequential state updates, which limit them to local rather than global modeling, and existing methods using different scanning strategies have not enhanced SSM's global modeling capability.

- The authors did extensive experiments on multiple tasks (semantic segmentation, object detection, instance segmentation) across many datasets with different backbone architectures, demonstrating the generalization capability of GMamba models.

### Weaknesses
- While the theoretical framework is novel, the core idea of using frequency-domain information for computer vision is well-established. (i.e, ICCV'23 works like SPANet [1]) and recent methods FAD [2]. The paper doesn't sufficiently differentiate its approach from these existing frequency-based methods.

- Existing work, such as Vim and VMamb, has demonstrated that SSMs can achieve global receptive fields through bidirectional processing and multi-directional scanning. The paper's claim that current SSMs lack global perception seems overstated.

- The related work and the comparison are missing recent state-space models (i.e, GroupMamba[3] (CVPR'25), MambaVision[4] (CVPR'25). Also, the provided implementation of GMamba_Block.py in the supplementary material appears to be largely derived from the MambaVisionMixer module introduced in MambaVision [4]. The authors have added their frequency components on top of that code. However, they neither cite nor compare their results with MambaVision paper [4].

- The reported improvements are relatively modest (around 2–3% mIoU) while introducing a significant parameter increase of 20–35%. It remains unclear whether these gains come from the added parameters or from the model’s inherent effectiveness. For example, in Table 2, the baseline UNet-ConvNeXt(S) has 58.42M parameters, whereas incorporating GMamba raises this to 71.06M, yielding 2.89% improvement in mIoU. A fair comparison would require scaling the baseline (e.g., by increasing the number of channels or blocks) to match 71.06M parameters, to determine if the gains are truly architectural rather than parameter-driven.

[1] https://openaccess.thecvf.com/content/ICCV2023/papers/Yun_SPANet_Frequency-balancing_Token_Mixer_using_Spectral_Pooling_Aggregation_Modulation_ICCV_2023_paper.pdf

[2] https://arxiv.org/pdf/2505.08349
 
[3] https://openaccess.thecvf.com/content/CVPR2025/papers/Shaker_GroupMamba_Efficient_Group-Based_Visual_State_Space_Model_CVPR_2025_paper.pdf

[4] https://openaccess.thecvf.com/content/CVPR2025/papers/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.pdf

### Questions
- It is unclear whether the GMamba models used for object detection and instance segmentation are initialized with backbones pre-trained on ImageNet or not. If yes, what is the top-1 accuracy of GMamba on ImageNet?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles the efficiency-global perception trade-off in visual tasks by proposing a rigorous mathematical definition of global image modeling and a frequency-domain modulated framework. Leveraging 2D-DFT’s global properties, the authors design the GSSM module and plug-and-play GMamba block, which seamlessly integrate into CNNs/Transformers. Extensive experiments across semantic segmentation, object detection, and instance segmentation show GMamba outperforms existing modules with linear-logarithmic complexity.

### Strengths
- First rigorous mathematical definition of global image modeling with formal proofs.
- DFT-based modulation overcomes SSMs’ local limitation without complex scanning.
- Strong performance across tasks, backbones, and datasets with efficiency advantages.
- Thorough ablations for validating key design choices (GSSM components, frequency contributions).

### Weaknesses
- Insufficient analysis of scaling to ultra-high-resolution images (e.g., 4K+) and inference speed (FPS).
- Lack of explicit comparison with recent frequency-domain SSM variants (e.g., FreqMamba [1]).
- No analysis of failure cases.
- Lack experiments on the Imagenet benchmark.

[1] Freqmamba: Viewing mamba from a frequency perspective for image deraining

### Questions
please refer to the weakness part.

### Soundness
3

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
This paper introduces the Global-aware SSM (GSSM) module, which integrates frequency-domain modulation using Discrete Fourier Transform (DFT) to enable true global perception in State Space Models (SSMs). ​ The authors propose a mathematical definition for global image modeling and prove that GSSM satisfies this definition. ​ They design GMamba, a plug-and-play module that enhances global contextual understanding in Convolutional Neural Networks (CNNs) with linear-logarithmic complexity. Extensive experiments on semantic segmentation, object detection, and instance segmentation demonstrate GMamba’s superior performance and efficiency compared to existing methods. ​ The study highlights the importance of frequency-domain information and adaptive modulation for balancing global semantics and local precision.

### Strengths
1. True Global Perception: GMamba, powered by the GSSM module, achieves true global perception by leveraging the Discrete Fourier Transform (DFT) for frequency-domain modulation. This enables the model to capture global semantic information efficiently. 

2. Efficiency: GMamba exhibits linear-logarithmic computational complexity, making it significantly more efficient than traditional self-attention mechanisms with quadratic complexity.

3. Plug-and-Play Design: GMamba can be seamlessly integrated into various stages of CNNs and other backbone architectures, enhancing their global modeling capabilities without requiring major architectural changes. 

4. Scalability: GMamba demonstrates scalable performance gains when integrated with more powerful backbone architectures, such as ConvNeXt-Small and Swin Transformer-Tiny.

5. Effective Frequency Decomposition: The use of both high-frequency and low-frequency components enhances global semantic modeling and local detail preservation, with adaptive modulation further optimizing their integration.

### Weaknesses
1. Limitation: The reliance on DFT-based frequency-domain modulation may introduce challenges in scenarios where frequency-domain information is less effective, such as highly noisy or irregular data.

2. Impact: While the frequency-domain approach enhances global perception, it may struggle in cases where spatial features dominate or where frequency information is less relevant.

3. Although GMamba is more efficient than self-attention mechanisms, it still introduces additional computational overhead compared to simpler SSM-based methods like Vim or TinyViM. 

4. While GMamba is described as "plug-and-play," its integration requires careful tuning of parameters such as modulation coefficients and frequency-domain weights. This could increase the complexity of implementation and training.

5. The robustness of GMamba in such challenging scenarios remains unclear, which could affect its reliability in real-world applications.

### Questions
1. The paper claims to provide the first rigorous mathematical definition of global image modeling, which is a significant contribution. However, it does not compare this definition with existing heuristic approaches in detail, leaving room for further exploration of how it improves interpretability and theoretical support.

2. How does the frequency-domain transfer function derived for SSMs compare to other global modeling techniques, such as attention mechanisms?

3. How does GMamba's linear-logarithmic complexity compare to the linear complexity of other SSM-based methods like Vim, VMamba or spatial mamba?

### Soundness
3

### Presentation
3

### Contribution
3
