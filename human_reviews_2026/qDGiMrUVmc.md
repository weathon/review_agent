# SONIC: Spectral Oriented Neural Invariant Convolutions

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 2

## Abstract
Convolutional Neural Networks (CNNs) rely on fixed-size kernels scanning local patches, which limits their ability to capture global context or long-range dependencies without very deep architectures. Vision Transformers (ViTs), in turn, provide global connectivity but lack spatial inductive bias, depend on explicit positional encodings, and remain tied to the initial patch size. Bridging these limitations requires a representation that is both structured and global.  We introduce SONIC (Spectral Oriented Neural Invariant Convolutions), a continuous spectral parameterisation that models convolutional operators using a small set of shared, orientation-selective components. These components define smooth responses across the full frequency domain, yielding global receptive fields and filters that adapt naturally across resolutions. Across synthetic benchmarks, large-scale image classification, and 3D medical datasets, SONIC shows improved robustness to geometric transformations, noise, and resolution shifts, and matches or exceeds convolutional, attention-based, and prior spectral architectures with an order of magnitude fewer parameters. These results demonstrate that continuous, orientation-aware spectral parameterisations provide a principled and scalable alternative to conventional spatial and spectral operators.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes SONIC (Spectral Oriented Neural Invariant Convolutions), a spectral-domain alternative to spatial convolutional networks (CNNs) and attention mechanisms. SONIC aims to combine the global receptive field propertly of transformers and spectral models, while enjoying strong spatial inductive bias and parameter efficiency that CNNs have. 

SONIC learns a set of spectral filters in the Fourier domain, each parametrized by six parameters modeling the amplitude, orientation, and oscillatory nature of the spectral transfer function. Learned matrices B and C factorize these filters and give a representation that is in theory resolution invariant, interpretable and also parameter efficient (parameters scale linearly with channels).

### Strengths
* The filters are highly parameter efficient and defined as continuous spectral functions, therefore have a number of desireable properties like resolution invariance and interpretability. 
* The papers contains a concise background section and  clear text presenting the method.

### Weaknesses
W1) Limited Experimental Validation: The experimental validation of the proposed method is limited. SynthShape appears to serve mainly as a proof-of-concept for demonstrating desired properties, but lacks realism. On PROMIS/Prostate158, the comparisons are only against basic baselines, and the observed gains over a simple UNet are relatively small. 

Moreover, there are no results on standard computer vision benchmarks (e.g., CIFAR, ImageNet, ADE20K). I do not expect state-of-the-art result in such generic cases, it is however important to demonstrate the broader applicability of SONIC to more generic tasks such as image classification or semantic segmentation.

W2) Lack of Empirical Support for Resolution Invariance: A central claim of the paper is that the method is resolution-invariant. While this may hold in theory, there is no clear experimental evidence to support it. A direct empirical evaluation across varying resolutions would greatly strengthen this claim.

W3) Undiscussed Compute and Memory Overhead: The method requires FFT/IFFT operations at every layer, which likely introduces non-negligible memory and computational costs. These costs are not discussed beyond theoretical costs.

W4) No Hyperparameter ablations: The impact of key hyperparameters (eg K or M) is not explored. An ablation study would help clarify how these parameters affect model performance and stability.

W5) Missing Discussion of Related Work on Hybrid Architectures
There is extensive prior work on combining global receptive fields with efficient CNN-style architectures in hybrid models. Classic examples include Non-Local Networks (Wang et al., CVPR 2018) and A²-Nets (Chen et al., NeurIPS 2018).
Given the motivation outlined in the introduction ("With the proposed method, which enables global receptive fields using significantly fewer parameters, we aim to narrow this conceptual gap"), a discussion of these and similar works is important to situate SONIC in the broader landscape and clarify its novelty.

### Questions
Q1) could you devise an experiment to directly test resolution invariance on real data?

Q2) It is hard for me to undestand the visualization in Fig 1. What are the axes? Please summarize the core observations from the Figure. Zooming in (so that white space is minimized) would imporve reeadability.

Q3) Same with Fig2. What are the main observations  from this figure?

Q4) Can you add some form of timings in Table 2? how does Sonic compare to the spatial baselines wrt that? ie what is the actual overhead of all the FFT/iFFTs?

Q5) How do different values of modes and channels affect performance?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces SONIC (Spectral Oriented Neural Invariant Convolutions), a novel spectral framework that replaces spatial convolutions with a compact set of orientation-aware transfer functions learned directly in the Fourier domain. By modeling filters as continuous frequency functions, SONIC achieves global receptive fields, resolution invariance, and linear parameter scaling with channels. Experiments on synthetic geometric segmentation and 3D prostate cancer MRI detection show that SONIC delivers stronger robustness and comparable or superior accuracy to CNN and Transformer baselines while using far fewer parameters.

### Strengths
1. The paper presents a spectral framework for multidimensional signals that offers global receptive fields, complete convolutional capability, and built-in resolution invariance. It provides a lightweight and flexible foundation for building scalable, adaptable vision models.
2. The paper presents comprehensive empirical validation across both synthetic and real-world settings.

### Weaknesses
1. The innovations mentioned in the abstract and contributions are mainly about unifying existing convolution kernels, spectral filtering, and state-space kernels under one spectral framework. However, the idea of parameterizing operators in the frequency or linear domain already exists in models such as S4ND, GFNet, FNO, and Mamba. The so-called directional modes only add a few interpretable parameters (e.g., direction, scale, damping) to the frequency response function, but in essence, it is still a functional representation of a frequency-domain convolution kernel.
2. The SynthShape dataset seems to be self-generated to verify the effectiveness of the proposed method (it is not a public dataset, so the persuasiveness of the results is questionable). In addition, there are no ablation experiments—for example, there is no comparison between using spectral factorization and directly learnable spectrum, nor any analysis of the effects of hyperparameters such as parameter count or directional constraints.
3. The experiments are limited to a toy dataset (SynthShape) and a single medical imaging application. There are no results on standard 2D benchmarks (such as CIFAR-10/100, ImageNet, or ADE20K) to test general visual performance and scalability.
4. The readability of the paper is not very good, and many formulas are not clearly explained.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces Spectral Oriented Neural Invariant Convolutions (SONIC), a novel neural building block which implements learnable spectral filters directly in the Fourier domain. The method is motivated by the shortcomings of current fully-connected, convolutional and attention based feature extraction blocks and derives a novel frequency domain feature motivated by State Space models (like MAMBA).  This allows the implementation of low-rank, orientation-aware global operators in the frequency domain which is suitable for modelling long-range relationships in the data (like large spatial kernels),  while remaining highly parameter-efficient and modestly computational expensive (compared to large spatial kernels). 

The Experimental evaluation on a provided toy dataset SynthShape demonstrates the theoretically derived properties regarding robustness against distortions, noise and geometric transformations in comparison to CNNs and Transformers. A second experiments shows a prove of concept on real medical (MRI) data.

### Strengths
The paper is well written, very well motivated and easy to follow. The presented idea is novel and presents an interesting concept, bringing core elements of state-space system into the the Frequency domain. 

The fact that the learnable filters are formulated in a continuous parameterization is particularly noteworthy. This actually opens the door to solve many sampling related problems in current network designs (aliasing causing low robustness, limitation to fixed input sizes, low robustness against geometric transformations ...).

The second main contribution of the paper is the low-rank formulation of frequency domain filters. Prior approaches often suffer from large amounts of necessary learnable parameters of crude forced reduction of the same. Transferring the concepts of state-space systems into the frequency domain presents a novel approach to engage these problems.   

The authors provide a very detailed and honest limitation section - this is very much appreciated!

### Weaknesses
There are several aspects in which the paper could be improved:

1) the paper mentions several previous approaches of Fourier-domain feature extraction (page 3 bottom), but does not compere to these methods in the experiments or in terms of computational complexity

2) the authors missed to discuss and to compare to [1] - another Fourier-domain approach of efficient large kernel implementations. 

3) the experiments comparing to CNNs do not show the used kernel size (also not in the appendix). Here it would be good to compare not only to 3x3 klernels, but also to some spatial large kernel implementations like [2] 

4) the proposed SONIC block is purely linear - this makes it necessary to conduct two expensive Fourier-Transformation between blocks, in order to be able to apply localized non-linear transformations. This massive practical drawback is inherent to all existing frequency space methods and thus remains unsolved

5) the authors discuss the possibility to combine their method with other (spatial) kernels. This indeed appears to be a very promising approach which would not be very difficult to implement - why did the authors not pursue this?  


[1] Grabinski, Julia, Janis Keuper, and Margret Keuper. "As large as it gets-studying infinitely large convolutions via neural implicit frequency filters." Transactions on Machine Learning Research 2024 (2024): 1-42.

[2] Ding, Xiaohan, et al. "Scaling up your kernels to 31x31: Revisiting large kernel design in cnns." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

### Questions
Q1: many of the existing Fourier-Domain methods are quite sensitive to the normalization of the coefficients. Did the authors investigate this for their method?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SONIC (Spectral Oriented Neural Invariant Convolutions), a method that learns compact, directional filters directly in the Fourier domain, being different from CNNs and ViTs. SONIC factorizes multi-channel frequency responses using a small set of shared, oriented components, which are mixed across channels via learned matrices. This design provides a global receptive field, is inherently resolution-invariant, and maintains high parameter efficiency. Empirical validation on synthetic and medical imaging benchmarks demonstrates that SONIC matches or exceeds the accuracy of established models like nnU-Net while using fewer parameters.

### Strengths
- The paper overall is easy to follow.

### Weaknesses
- My main concern is the unclear comparisons with the existing solutions. In Section 2 (line 149), the authors mentioned some limitations of previous solutions like GFNet and FNO. However, there is no further study to show how the proposed method overcomes these limitations and why these limitations are important. Besides, there is also no direct comparison with these methods in the experiments. For example, there are two limitations of GFNet mentioned: "the FFT grid is tied to the input resolution, and the number of learnable parameters scales with the discretisation of the frequency spectrum". But as shown in Fig. 6 of GFNet, the model can be directly adapted to other resolutions with finetuning and works well on detection tasks with images of varying resolutions. The parameters involved in frequency-domain operations also account for only a small proportion of the model’s total parameters. Therefore, I am still concerned about the significance of the contribution in the paper.

- The experimental study presented in the paper is a bit weak for publishing at a top-tier conference like ICLR. To show the contribution of the proposed solution, it would be better to either show significant improvements over previous methods with solid experiments (e.g., on ImageNet and directly comparing with GFNet, FNO, etc.) or present the unique property of the new solution that may have a large potential impact. The current experimental study in the paper is not very convincing to me.

### Questions
As mentioned in the "weaknesses" section, the paper can be further improved in several aspects. My main concerns are the comparisons with the existing solutions in the frequency domain and the relatively weak experimental study.

### Soundness
2

### Presentation
2

### Contribution
2
