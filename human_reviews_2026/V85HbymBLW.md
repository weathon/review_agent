# LogART: Pushing the Limit of Efficient Logarithmic Post-Training Quantization

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Efficient deployment of deep neural networks increasingly relies on Post-Training Quantization (PTQ). Logarithmic PTQ, in particular, promises multiplier-free hardware efficiency, but its performance is often limited by the nonlinear and symmetric quantization grid and standard rounding-to-nearest (RTN) approach. While learnable rounding has significantly advanced linear PTQ, its application to the non-linear and often discrete nature of logarithmic domain remains unexplored. This paper introduces learnable Logarithmic Adaptive Rounding Techniques (LogART) that pioneer task-aware learnable rounding specifically for the logarithmic domain. LogART further extends the learnable rounding strategy to flexibly support outlier-aware, asymmetric, and hardware-friendly dynamic logarithmic bases, determined in a distribution-aware manner using an efficient search strategy. Extensive experiments demonstrate that LogART achieves state-of-the-art accuracy while maintaining efficiency in quantizing models across various architectures and ultra-low bitwidths, outperforming existing logarithmic PTQ methods and paving the way for more effective hardware deployment. The code is available at https://github.com/logart-lab/logart.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces LogART, a novel post-training quantization (PTQ) method designed for logarithmic quantization of neural network weights. Unlike linear quantization, logarithmic quantization aligns better with weight distributions often found in large models, and enables hardware efficiencies through base-2 arithmetic. LogART innovates by combining learnable rounding in the logarithmic domain with an adaptive quantizer that supports outlier resilience, asymmetry, and multi-base logarithmic representations. The authors also propose a multi-level hyperparameter search and a hardware approximation function (HAF) to make their scheme efficient and hardware-friendly. They evaluate LogART on a variety of architectures, LLMs, CNNs, and ViTs, showing higher accuracy than prior log-based PTQ, and meaningful hardware-area and power benefits.

### Strengths
- LogART supports multi-base quantization, e.g., mixing base-2 and base-$\sqrt{2}$.


- The HAF (Hardware Approximation Function) approximates $\sqrt{2}$ with shift-add operations, trading off minimal accuracy loss for hardware efficiency. They simulate area and power in a 28 nm process, showing substantial savings compared to conventional designs.

- Experiments across LLMs, CNNs, and ViTs strengthen the claim of generality.

- The ablation studies clearly show the impact of each component (LLR, OHS, dynamic base, etc.).

### Weaknesses
- The evaluations focus primarily on 3-bit quantization. It would be useful to show how LogART performs at even lower (2-bit) weights and to compare it with other PTQ methods like QuIP[1].

- The paper mentions compatibility with activation quantization but does not deeply explore joint weight-activation quantization. Joint methods are often more relevant for real deployments.

[1] QuIP: 2-Bit Quantization of Large Language Models With Guarantees

### Questions
- Error Bound Analysis: do you have theoretical bounds on the error introduced by the HAF approximation of $\sqrt{2}$? 

- Lower Bitwidth: have you experimented with 2-bit weight quantization? If so, how does LogART perform?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel adaptive weight rounding technique called LogART for improving logarithmic post-training quantization (logarithmic PTQ) performance on large language models (LLMs). The method introduces key components including dynamic base selection, scaling factor selection, adaptive base selection, and learnable weight rounding, significantly enhancing the performance of 3-bit channel-wise weight quantization models while maintaining hardware efficiency.

### Strengths
(1) First proposes a learnable weight rounding technique specifically for logarithmic PTQ, solving the problem that existing methods cannot apply learnable weight rounding in non-linear quantization.
(2) Designs multiple combinable components (DBS, SFS, ABS, LLR), with experiments showing that these components produce synergistic effects when used together, significantly reducing perplexity (PPL).
(3) Extensively validates LogART's effectiveness on LLMs such as OPT-125M and LLaMA2-7B, demonstrating significant performance improvements under 3-bit quantization.
(4) Analyzes the impact of calibration datasets on performance, proving LogART's robustness to calibration data sources.

### Weaknesses
(1) The paper lacks theoretical analysis of the proposed method, particularly in-depth explanation of interactions between components.
(2) Experiments are primarily focused on LLMs, with insufficient validation on other model types such as CNNs or ViTs, limiting the method's generalizability.
(3) The paper does not discuss the additional computational overhead of LogART in practical deployment, especially the extra training steps required for learning rounding parameters, which may contradict the core goal of quantization techniques to simplify models and improve hardware efficiency.

### Questions
(1) The paper mentions that LogART's components can be used in combination, but does not detail how these components work together, nor whether there exists an optimal combination.
(2) The paper uses WikiText-2 and C4 as calibration datasets in experiments, but does not deeply explore the impact mechanism of different calibration datasets on performance, nor how to choose the best calibration dataset.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents several methods that combined together improve the current SoTA on logarithm PTQ:
- Learnable Logarithmc Rounding (LLR): train the best rounding on calibration data set;
- Dynamic Base Quantizer (DBS): combine sqrt(2) and 2 bases;
- Asymmetric Quantizer (ABS): induce an asymmetry by clamping close to 0 values;
- Scaling Factor (SFS): train a scaling factor for clamping on a calibration data set, to better handle outliers.

### Strengths
- The paper advances the SotA on logarithmic quantization, mostly by combining/transfering known methods;
- Adequate systematic ablation study is made (available in the supplementary material, albeit only for Transformers);
- The paper evaluates its methods on both CNN and Transformers models;
- Although secondary in the main paper, a study of hardware implementation of the dual sqrt(2) and 2 base log is performed.

### Weaknesses
- I did not understood the asymmetric quantizer scheme: it seems to me that your are only clamping small values after quantization, therefore actually reducing the effective number of bits! I don't understand how this can improve things and indeed, from your ablation studies, it seems to degrade the PPL when not used in combination with LLR.
- The paper is very incremental and essentially combines/transfers known methods from the SoTA.

### Questions
If you could explain how and why ABS work.
Also, you only made full ablation study on Transformers. I would be very interested to see a similar full ablation study on CNN.

### Soundness
3

### Presentation
3

### Contribution
2
