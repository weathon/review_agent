# LEGACY: A Lightweight Dynamic Gradient Compression Strategy for Distributed Deep Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Distributed learning has achieved remarkable success in training deep neural networks (DNNs) on large datasets, but the communication bottleneck limits its scalability. Various compression techniques have been proposed to alleviate this limitation; however, they either use fixed parameters throughout training or rely on complex and computationally intensive methods to adapt compression parameters. Instead of the hard-to-tune hyperparameters required by adaptive compressors, this paper investigates the impact of two fundamental factors in DNN training—the layer size of the networks and their training phases—to design a simple yet efficient dynamic scheduler for any compressor, guiding the selection of compression parameters. We present a **L**ightweight **E**fficient **G**r**A**dient **C**ompression strategy**Y** or LEGACY, which, in theory, can work with any compression technique to produce a simple dynamic counterpart. We benchmark LEGACY on distributed and federated training, involving seven different DNN architectures, ranging from ResNet, Transformer-XL, to GPT-2, across large and challenging datasets, including ImageNet, WikiText-103, and OpenWebText. On ImageNet-1K, with an equivalent average data volume, LEGACY's dynamic compression strategies improve the Top-1 accuracy of ResNet-50 by 7-11% compared to uniform Top-0.1% compression, while on WikiText-103, the layer-based dynamic strategy reduces the perplexity of Transformer-XL by ~26% relative to the same baseline. In addition, we evaluate LEGACY under constrained and federated settings, and demonstrate that it scales effectively to a 100-worker configuration while maintaining strong accuracy under aggressive compression. We publish anonymized code at: https://github.com/LEGACY-compression/LEGACY.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript introduces a practically-driven compression strategy for federated learning, supported by theoretical analysis and extensive benchmarking. A central finding is that compression should be adapted to individual layer sizes and the training phase. This insight offers valuable guidance for designing more efficient communication in federated learning. The work effectively balances theory and experimentation, providing useful design principles for adaptive compression methods.

### Strengths
1、The paper is supported by rigorous simulations that empirically validate the proposed method. The experiments are well-constructed, employing relevant baselines and metrics to clearly demonstrate the approach's effectiveness across diverse conditions.
2、Complementing the empirical work, the authors include a theoretical analysis that grounds the core principles of their compression strategy. Although not the paper's central thrust, this analysis adds valuable depth and strengthens the overall argument.

### Weaknesses
1、While the paper explores a broad parameter space, the sensitivity of the method to specific parameter choices and its potential failure conditions remain unclear. A more systematic investigation, or at least a discussion of possible failure modes, would help to better assess the approach's reliability and practical applicability.
2、Regarding the theoretical contribution: as noted in the manuscript, the analysis of compression operators and their convergence rates appears to align with existing theoretical frameworks. It would be helpful to clarify the novel theoretical insights offered by this work compared to prior analyses.
3、The models used in the experiments are relatively small and may not reflect modern large-scale architectures. Since their communication overhead is inherently limited, the results may not fully demonstrate the method's effectiveness in high-communication settings. Experiments on larger generative models, such as GPT-family architectures, would strengthen the evaluation.

### Questions
1、To better assess the practical applicability of the method, could the authors delineate the scenarios or conditions under which the proposed compression framework might exhibit performance degradation? A discussion of such limitations would be valuable for understanding the robustness of the approach.
2、The theoretical analysis presented, while sound, appears to align with established convergence rates for standard compressors in the literature. Could the authors clarify the novel theoretical contribution of their work in this context?
3、The experimental validation primarily uses smaller-scale architectures. To demonstrate the scalability and general applicability of the method, could the authors provide results on larger, contemporary models, such as generative transformers (e.g., GPT variants)?

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
3

### Summary
The paper introduces LEGACY, a simple scheduler to make gradient compression dynamic in distributed deep learning. Instead of using a fixed compression ratio, LEGACY adapts compression by training phase (light compression early, heavy later) and by layer size (compress large layers more, small layers less). 
The scheduler works with any existing compressor without additional compute overhead or per-iteration tuning. Theoretical analysis shows the intuition behind the proposed method and guarantees the convergence of compressed stochastic gradient descent. Comprehensive experiments under different settings demonstrate the effectiveness of LEGACY compared to other compression methods.

### Strengths
1. The paper explores a dynamic compression strategy that adapts to training phase and layer size, which can potentially improve the efficiency of distributed deep learning.
2. It provides sufficient experimental results under various settings that support the effectiveness of the proposed method.
3. The proposed compression method is lightweight and can be integrated with other compression techniques.

### Weaknesses
Compared to existing adaptive compression methods such as Accordion and L-GreCo (Section 4.3.1), LEGACY does not clearly demonstrate advantages in either accuracy or data volume. Therefore, the novelty and practical impact of the method appear somewhat limited.

### Questions
1. In practice, is the chosen compression schedule consistent with the theoretical assumptions and requirements?  There appears to be a gap between the theory and the experimental settings that is not clearly explained or discussed.
2. The paper introduces two dynamic compression strategies: by training phase and by layer size. And it also states that they can mix the two strategies up. Is there any theoretical analysis for the mixed strategy?

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
3

### Summary
This paper presents a communication-efficient federated learning method. The proposed method is scheduler of gradient compression, which is agnostic to compression (quantization) method. The proposed method is based on the observation that the number of parameters for individual layers could be very different. Furthermore, at different stages of model training, different compression strength is necessary to achieve good balance between performance and communication cost reduction. To demonstrate the above ideas, this paper provides theoretical analysis on the compression bounds and convergence. Although theoretical analysis is provided, the experiments are still empirical. Testing on several datasets and model architectures, including ResNet and Transformer, this paper shows better results with different quantization methods, compared with other adaptive compression techniques like Accordion. 

From my perspective, the complexity of layers and the training stages are essentially related to gradient norm (later epochs produce smaller norms) and direction. It can be analyzed from the compression errors in norm and cosine similarity after applying the quantization. Therefore, a more controlled scheduler might be easier to deploy than an empirical method studied in this work, by checking the compression errors caused by layer size and training epoch numbers. 

I vote this paper as "6: marginally above the acceptance threshold. But would not mind if paper is rejected", but could change my decision in my final voting.

### Strengths
1. A very simple and quantization-agnostic approach is proposed for communication-efficient federated learning. Based on this approach, several models and datasets are applied with better performance than previous work.

2. Theoretical analysis is provided, which enhances the motivation of this work.

### Weaknesses
1. The selection of compression strength is still empirical. It increases the challenge of deployment of this model in the real world. For example, the data from individual clients might be very different, that some clients may provide training data very different to major distribution. For this client, the gradient norm will be very different to others. Therefore, different clients may also need different compression strength. 

2. It will be helpful to show the performance curves over different training epochs and different communication costs.

### Questions
In the research history, it's known that early layers are learned at the beginning of training. In particular, when we finetune from a pretrained model, the early layers do not change drastically, compared with later layers. It is interesting to know how to balance the compression strength for lower / higher layers at the beginning / end of training.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work itroduce  a lightweight, open-source framework, referred to as LACACY, which dynamically compresses gradients in distributed DNN training. Its scheduler, guided only by layer size and training phase, offers a simple  alternative to compute-intensive adaptive compressors. Benchmarking with Top-k, Random-k, QSGD, and PowerSGD shows that LEGACY achieves consistent performance gains over uniform on Cifar10, Cifar100 and other datasets.

### Strengths
The paper presents a unified framework that enables various communication-compression algorithms to dynamically adjust their compression levels. Experimental results demonstrate that this proposed scheme is both simple and effective.

### Weaknesses
1. **Theoretical Motivation:** The motivations for the dynamic compression strategies are not fully convincing.
   * The **training phase-based** strategy is justified by Lemma 1, which relies on convergence properties of Gradient Descent for convex objectives. However, this theoretical foundation may not directly translate to the non-convex and stochastic optimization of Deep Neural Networks, limiting its applicability.
   * The **layer size-based** strategy relies on the unverified assumption that larger layers are more over-parameterized and can tolerate higher compression. This claim lacks theoretical or empirical support within the paper.

2. **Insufficient Experimental Evidence:** While the paper claims LEGACY is a "lightweight yet effective dynamic scheduler," the experiments do not fully substantiate this claim.
   * There is no evidence demonstrating that LEGACY's lightweight design translates to significant wall-clock time reduction compared to compute-intensive adaptive methods like L-GreCo and Accordion. The time saved by its simple scheduler might be negligible compared to overall compression computation.
   * Performance comparisons against key counterparts are limited, primarily appearing in the ResNet-18 on CIFAR-100 experiment. This narrow scope limits generalizability.
   * The reported accuracy for uniform PowerSGD and L-GreCo is notably lower than results in the original L-GreCo paper [1] under similar settings. Additionally, the compression ratio used for Accordion is lower than for other methods, potentially creating an unfair comparison. These issues undermine the reliability of the experimental findings.
In summary, the comparative experiments lack breadth, and their results are not entirely convincing.


[1] Alimohammadi, Mohammadreza, et al. "L-GreCo: Layerwise-Adaptive Gradient Compression for Efficient and Accurate Deep Learning." arXiv:2210.17357, 2022.

### Questions
1. The architectures (e.g., ResNet-9, ResNet-18) and datasets (e.g., CIFAR-10/100, WikiText) used for evaluation are somewhat outdated. Could the authors demonstrate LEGACY's effectiveness on more modern benchmarks (e.g., Vision Transformers or contemporary datasets)?

2. In Table 5, the reported baseline accuracy for training ResNet-50 on ImageNet-1K is 59.43%, significantly lower than standard benchmarks. Could the authors explain this discrepancy in their training setup?

### Soundness
3

### Presentation
2

### Contribution
3
