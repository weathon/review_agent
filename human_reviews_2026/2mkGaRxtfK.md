# EvoIR: Towards All-in-One Image Restoration via Evolutionary Frequency Modulation

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
All-in-One Image Restoration (AiOIR) tasks often involve diverse degradation that require robust and versatile strategies. However, most existing approaches typically lack explicit frequency modeling and rely on fixed or heuristic optimization schedules, which limit the generalization across heterogeneous degradation. To address these limitations, we propose EvoIR, an AiOIR‑specific framework that introduces evolutionary frequency modulation for dynamic and adaptive image restoration. Specifically, EvoIR employs the Frequency-Modulated Module (FMM) that decomposes features into high- and low-frequency branches in an explicit manner and adaptively modulates them to enhance both structural fidelity and fine-grained details. Central to EvoIR, an Evolutionary Optimization Strategy (EOS) iteratively adjusts frequency-aware objectives through a population-based evolutionary process, dynamically balancing structural accuracy and perceptual fidelity. Its evolutionary guidance further mitigates gradient conflicts across degradation and accelerates convergence. By synergizing FMM and EOS, EvoIR yields greater improvements than using either component alone, underscoring their complementary roles. Extensive experiments on multiple benchmarks demonstrate that EvoIR outperforms state-of-the-art AiOIR methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces EvoIR, a new framework for All-in-One Image Restoration (AiOIR). The proposed method consists of two main components: a Frequency-Modulated Module (FMM) that processes low- and high-frequency features in separate branches, and an Evolutionary Optimization Strategy (EOS) designed to dynamically balance a fidelity loss and a perceptual loss during training. The authors conduct experiments on several AiOIR benchmarks, reporting state-of-the-art performance in both three and five degradation settings.

### Strengths
* The authors have conducted a comprehensive set of experiments, including multiple AiOIR benchmarks and single-task evaluations. The effort to validate the method across this breadth of tasks is commendable.
* The combination of frequency-domain feature modulation (FMM) and evolutionary loss optimization (EOS) is interesting in the context of All-in-One image restoration.

### Weaknesses
* The Frequency-Modulated Module (FMM), while effective, builds heavily on existing frequency-domain methods (e.g., Kong et al., 2023; Cui et al., 2025, as cited in the paper). The paper does not sufficiently clarify how FMM differs from or improves upon prior frequency-based approaches for AiOIR.

* The Evolutionary Optimization Strategy (EOS) is presented as a key contribution, but it may be perceived as a training-time heuristic rather than a core architectural innovation. The ablation shows only marginal gains when EOS is added alone (+0.08 dB PSNR), raising questions about its necessity.

* The computational cost and training time overhead introduced by EOS are not discussed. Given that EOS involves multiple generations of population-based search on a validation set, its efficiency and scalability remain unclear.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes EvoIR, an all-in-one image restoration framework that combines a Frequency-Modulated Module (FMM) to explicitly decompose and enhance features in high- and low-frequency domains, together with an Evolutionary Optimization Strategy (EOS) that dynamically adjusts loss weights during training through a population-based evolutionary search. The goal is to improve the generalization of image restoration models across heterogeneous degradations.

### Strengths
1. The paper recognizes a key limitation in existing all-in-one restoration models that mostly rely on spatial-domain operations. The introduction of frequency-domain decomposition is intuitive and aligns well with the physical properties of image degradations.
2. The proposed EOS provides a novel angle for loss balancing by introducing evolutionary algorithms. This avoids the need for manual tuning of loss weights, which is a practical challenge in multi-degradation learning.
3. EvoIR demonstrates robustness under increased degradation types, and the improvements remain stable even in more challenging settings with five degradations.

### Weaknesses
1. While the combination of frequency decomposition and evolutionary optimization is well-engineered, both components are based on existing techniques. The FMM is closely related to frequency gating and hybrid CNN-transformer architectures, while the EOS is a standard evolutionary search mechanism. The overall framework appears to be a modular integration of known methods rather than introducing a fundamentally new paradigm for image restoration.
2. The two main contributions claimed in the introduction are largely combinations of existing techniques rather than genuinely novel conceptual advancements. Compared with recent works such as frequency-domain Transformers and adaptive loss optimization strategies, the paper does not sufficiently articulate the unique necessity or distinct innovation of its proposed approach at a mechanistic level.
3. The paper does not provide theoretical analysis on why the frequency splitting leads to better degradation generalization or how evolution-based loss scheduling mitigates gradient conflicts in a principled way. The improvements are mostly empirical.
4. The model involves FFT/IFFT operations, spectral gating, dynamic masking, and population search, which increases system complexity. The paper does not clearly report FLOPs or training cost associated with EOS. Without this information, it is difficult to assess whether EvoIR provides a favorable accuracy-efficiency trade-off.
5. The claimed contributions in the introduction are largely architectural and engineering in nature. The paper does not convincingly articulate what new insight is gained about degradation modeling beyond designing a stronger backbone.

### Questions
See the above parts.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes EvoIR, a new framework for All-in-One Image Restoration (AiOIR). The goal is to create a single model capable of handling multiple, diverse image degradations. The claimed core contribution is twofold: 1) a Frequency-Modulated Module (FMM), which explicitly decomposes features into high- and low-frequency branches and modulates them separately to balance structural and textural recovery , and 2) an Evolutionary Optimization Strategy (EOS), a population-based algorithm that dynamically searches for the optimal weights between a fidelity loss ($\mathcal{L}_{fid}$) and a perceptual loss ($\mathcal{L}_{perc}$) during training. The authors conduct experiments on 3-task (N+H+R) and 5-task (N+H+R+B+L) synthetic benchmarks, reporting state-of-the-art (SOTA) results that outperform recent methods.

### Strengths
1.	The paper addresses the All-in-One Image Restoration (AiOIR) problem, which is a highly significant and practical challenge in computational photography and computer vision.
2.	The primary strength of the paper lies in its novel combination of ideas. While frequency-domain processing and dynamic loss weighting are not new individually, integrating a population-based evolutionary search (EOS) to directly optimize the loss weights for a frequency-aware architecture (FMM) is an original approach for this problem.

### Weaknesses
1.	The claim of novelty for frequency-aware processing should be tempered. Operating in the frequency domain is a common and well-established strategy in image restoration. Recent works like AdaIR (which uses frequency-domain prompts) and other Transformer-based models (e.g., SFHformer) have already successfully leveraged frequency-domain features for image restoration. The FMM module appears to be a well-engineered variant of this existing paradigm rather than a completely new one.
2.	The paper's claims of superiority are weakened by a questionable trade-off between model size and performance. The proposed EvoIR has 36.68M parameters. This is nearly 50% larger than a strong recent baseline like MOCE-IR (25.35M). However, on the complex 5-task benchmark, this significant increase in model size yields a marginal PSNR improvement of only +0.28 dB (30.83 vs 30.58) and, more concerningly, a decrease in SSIM (0.918 vs 0.919). This suggests a poor efficiency-to-performance ratio. A critical missing comparison is a baseline (e.g., MOCE-IR) scaled up to a similar parameter count (~36M). It is plausible that a larger baseline model could match or exceed EvoIR's performance, which would call into question the architectural benefits of FMM
3.	 The EOS component, while interesting, introduces two major concerns that are not addressed:
- The EOS algorithm requires freezing the main network and running a multi-generation evolutionary search every 500 training iterations. This introduces a non-trivial computational overhead. The authors fail to quantify this cost. How much longer does it take to train EvoIR compared to the baseline (Model 'a' in Table 4)? This is a critical detail for reproducibility and practical adoption.

- The evolutionary search is constrained to a 1D simplex between only two losses: $\mathcal{L}_{fid}$ and $\mathcal{L}_{perc}$11. This choice feels arbitrary. Why only these two? Many modern restoration models benefit from a combination of losses (e.g., $\mathcal{L}_1$, SSIM, LPIPS, GAN loss). The impact of limiting the search to this specific combination is not justified.

4.	This is the most significant weakness of the paper. The experiments are conducted only on synthetic datasets where the training and testing distributions are nearly identical. There is no validation on unseen degradation types or severities. There is no cross-domain validation (e.g., training on synthetic data and testing on real-world data). The complete absence of experiments on real-world benchmarks is a major omission. Without this analysis, it is impossible to know if the proposed FMM and EOS components are truly learning to robustly restore images or if they are simply overfitting to the specific synthetic degradations used in the training set.

### Questions
• The authors should quantify the computational overhead of the EOS strategy. Specifically, what is the percentage increase in total wall-clock training time when enabling EOS, compared to the baseline model for the same number of epochs?

• The authors should provide a deeper justification for limiting the EOS search space. Have the authors investigated the impact of incorporating more diverse objectives into the search, such as gradient-based losses or learned perceptual metrics (e.g., CLIP-IQA), and how would different objectives types affect the optimization?

•  EvoIR (~37M params) is nearly 50% larger than MOCE-IR (~25M), yet this yields only a marginal PSNR gain (+0.28 dB) and a lower SSIM on the 5-task benchmark. This raises a question about the fairness of the comparison. How can the authors be sure the improvements stem from the proposed FMM/EOS modules rather than just an increase in model scale? Would a scaled-up MOCE-IR, matched for parameter count, perform just as well or better?

• The paper's validation is limited to synthetic benchmarks. How well does the proposed method generalize to real-world scenarios or, at unseen degradations not included in the training mix? To assess robustness and mitigate concerns of overfitting, the authors should provide additional experiments, such as zero-shot evaluations and real-world benchmarks.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors propose a novel AiOIR framework with adaptive frequency modulation and evolutionary loss optimization. The method is well designed with detailed experiments. Experimental results reveal the effectiveness and efficiency of the proposed EvoIR.

### Strengths
1. The combination of frequency modulation with evolutionary strategy is suitable. Both of them ensure the structural information and fine-grained details.
2. Extensive experiments are conducted in details. The performance under 3D and 5D settings are all good enough with compared params. Its average results are superior than others and especially on deraining and dehazing.
3. Component ablations indicate synergy between FMM and EOS and show faster convergence with EOS.

### Weaknesses
1. More detailed parameter settings of EOS can be discussed with some additional ablation study, like the training iterations T.

2. Visualization of T-SNE can help readers more to view the effectiveness of EvoIR.

### Questions
1. Would other perceptual strategies can enhance the performance besides EOS with SSIM?
2. To better show the performance of the EvoIR, the authors can use T-SNE to visualize features of different degradation.
3. Can this method deal with some mixed degradation types, like rain+haze?
4. There are some parameters in EOS setting. Can the authors provide some ablation study to validate the effectiveness of it?
5. Typos like RSE-FFTB in caption of Fig. 2 should be fixed.
6. Altough the methodology and figures are illustrated well, writing need improvement. Section 3.2 includes some redundant paragraphs.

### Soundness
3

### Presentation
3

### Contribution
3
