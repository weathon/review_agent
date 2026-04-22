# NIC-RobustBench: A Comprehensive Open-Source Toolkit for Neural Image Compression and Robustness Analysis

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Adversarial robustness of neural networks is a rapidly developing and important research area, ensuring the reliable use of neural network models in areas such as computer vision, natural language processing, and others. With the emergence of neural image compression (NIC) methods and the introduction of the first standard, JPEG AI, the question of robustness has also become critical in image compression. Unstable NIC models are more prone to producing distortions and can be exploited to compromise downstream vision tasks that rely on compressed data. Most existing benchmarks for NIC focus primarily on compression efficiency and reconstruction quality, while prior robustness studies have been limited to a small number of codecs and attacks. To address this gap, we introduce \textbf{NIC-RobustBench}, the first open-source framework for evaluating the adversarial robustness of NIC methods. It integrates 6 adversarial attacks and 7 defense strategies in addition to traditional Rate-Distortion (RD) evaluation. The framework includes the largest number of codec types among all known NIC libraries and is easily scalable. The paper provides a comprehensive overview of NIC-RobustBench alongside an extensive robustness evaluation of modern NIC methods. Our code is publicly available at \textit{link hidden for blind review}. We believe our framework will become an essential tool for developing robust neural image compression techniques.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an open source library, NIC-RobustBench, for evaluating the adversarial robustness of state of the art neural image compression (NIC) models. The framework includes a variety of NIC models, white box adversarial attacks, and robustness metrics. Additionally, the paper performs an evaluation of existing models and identifies some interesting trends about what types of models are most vulnerable to attacks. Finally, the paper proposes several defense strategies and analyzes the effects of them in their evaluation.

### Strengths
1. The paper includes an open-source package for experimenting with many state-of-the-art NIC codecs.
2. The paper finds interesting trends about types of models and robustness (e.g., larger models are less secure, higher compression rates are the most robust).

### Weaknesses
1. The paper has limited novelty and contribution. Specifically, all of the NIC models and attacks are from prior work. Most of the metrics are also borrowed from prior work, with the exception of $\delta$, which the paper does not motivate well. Finally, there is very limited discussions or theoretical insight into the empirical findings.
2. The evaluation datasets are limited and not ideal for image compression tasks. The Cityscapes and NIP 2017 datasets have very low resolution images and Kodak, while having slightly higher resolution, only has 24 images. I would recommend the authors evaluate on CLIC and/or ImageNet.
3. The paper is hard to read in its current state. Specifically, many of the captions are not complete (e.g., what dataset/attack is used in 4a? what dataset in 4b? Where did the images in figure 1 come from?). Many acronyms are presented, but not explained (e.g., what is IQA?, what does VMAF stand for?).  Additionally, the equations include some unclear/inconsistent terminology (e.g., equation 1 is missing an objective (min), equation 2 uses $L(x, x', C(x), C(x'))$, but L is later defined as $L: X \times X \to \mathbb{R}$, and equations 2 and 3 have inconsistent uses of $x'$ and $x+\delta$).

### Questions
1. What is the motivation for $\delta$? When should we use $\delta$ vs. $\Delta$?
2. Are there any cases where $\Delta VMAF$ is negative in figure 4b?
2. Do you have any insight into why generative codecs are the most vulnerable? Have you tried comparing generative and discriminative codecs of the same size?

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
4

### Summary
This paper introduces NIC-RobustBench, a new open-source library and its associated benchmark for evaluating the adversarial robustness of neural image compression (NIC) methods. The framework supports various adversarial attacks and defense strategies for image compression tasks, enabling a systematic assessment of NIC models' performance in complex adversarial settings. The paper also conducts experiments on several NIC models, examining both attacks and defenses, providing valuable insights into the robustness of NIC models.

### Strengths
1. This paper provides a comprehensive compilation of adversarial attack methods, defense strategies, and evaluation metrics for NIC methods, offering a valuable resource for future research in this domain.
2. The experimental section is extensive, covering 10 mainstream NIC models, 6 attack methods, 7 defense strategies, and multiple datasets, offering numerous case studies.
3. The modular design in the framework diagram reflects excellent engineering, facilitating reproducibility and ensuring extensibility for future work.

### Weaknesses
1. The paper focuses more on the practical tool aspect and lacks theoretical discussions, particularly regarding the relationship between adversarial robustness and compression mechanisms.
2. The attacks and defense strategies presented in the paper are derived from existing research; the paper mainly integrates these methods rather than proposing novel algorithms, lacking a degree of innovation.

### Questions
1. While the paper provides a robust experimental framework, it lacks theoretical analysis. Could the authors further explain why certain NIC architectures, such as generative models, are more susceptible to attacks?
2. Most of the attacks and defense strategies in NIC-RobustBench are based on existing research. Did the authors propose any specific improvements or novel modifications to these methods within the framework?
3. The experimental section is comprehensive, but the analysis of the results is rather superficial. Could the authors provide deeper explanations of experimental phenomena, such as whether an increase in model scale influences the emphasis on image reconstruction, thereby affecting robustness?
4. Did the authors perform experiments to evaluate the impact of adversarial defense methods on RD (Rate-Distortion) performance?
5. We acknowledge the contributions of your work to the field, but it may not fully align with the criteria for ICLR 2026 Main Track. We thus wonder if you would consider submitting it to a more suitable track？

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
This paper presents NIC-RobustBench, an open-source, modular benchmark/framework for evaluating the adversarial robustness of neural image compression (NIC) models. It unifies a large set of NIC codecs, six white-box attacks with multiple objectives, and seven defenses and provides a standard pipeline to measure both rate–distortion and robustness, plus impact on downstream CV tasks. The authors then run a broad study over 10+ NIC models and show generative/large codecs are more vulnerable, attack objective matters a lot, and some simple reversible defenses stabilize NICs.

### Strengths
a) Compared with existing NIC libraries that mostly focus on RD (CompressAI, TFC, NeuralCompression), this is the first one that systematically bakes in both attacks and defenses, and it clearly has the widest model set so far.
﻿
b) Many “benchmark” papers stop at attacks; this one also implements reversible/geometric defenses and diffusion-style purification (DiffPure), so it supports end-to-end studies.

### Weaknesses
a) Novelty is mainly engineering/integration. The paper stands on putting things together, not on a new attack/defense or a new robustness metric. That’s OK for a benchmark, but then it must be very explicit in positioning vs. CompressAI (Bégain et al. 2020), TFC (Ballé et al. 2024), NeuralCompression (Muckley et al. 2021).
﻿
b) Only white-box attacks. The authors argue black-box is costly, but for a benchmark that wants to be comprehensive, skipping black-box / transfer attacks is a real limitation.
﻿
c) Evaluation datasets are small/narrow. KODAK (24 imgs), Cityscapes slice, and NIPS 2017 (1000 imgs) are standard but small; given that NIC is now used in high-res and domain-specific settings, this somewhat weakens the “comprehensive” claim.
﻿
d) Defense set is mostly classical transforms. Apart from DiffPure, the seven defenses are flip/roll/rotate/color/ensembles. This is fine as a baseline, but there is a long line of adversarial purification/denoising/consistency defenses that could be referenced.

e) The image is not good (e.g., Fig 2 and Fig 3). Firstly, no vector graphics were used, and secondly, some fonts are too small

### Questions
a) How do you envision adding a “slow but realistic” black-box track (e.g., per-epoch evaluation, cached queries) without breaking the CI-friendly design?
﻿
b) How to evaluate “cost vs. robustness” profile in the benchmark so users can compare defenses at a fixed latency/memory budget?
﻿
c) Some of your findings (larger ≈ less robust; generative ≈ less robust) look quite strong. Do you think these will still hold on higher-res, more diverse datasets, or are they partly an artifact of the current three datasets?

### Soundness
3

### Presentation
2

### Contribution
3
