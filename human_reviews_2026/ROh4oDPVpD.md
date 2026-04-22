# TriC-Motion: Tri-Domain Causal Modeling Grounded Text-to-Motion Generation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Text-to-motion generation, a rapidly evolving field in computer vision, aims to produce realistic and text-aligned motion sequences. Current methods primarily focus on spatial-temporal modeling or independent frequency domain analysis, lacking a unified framework for joint optimization across spatial, temporal, and frequency domains. This limitation hinders the model's ability to leverage information from all domains simultaneously, leading to suboptimal generation quality. Additionally, in motion generation frameworks, motion-irrelevant cues caused by noise are often entangled with features that contribute positively to generation, thereby leading to motion distortion. To address these issues, we propose Tri-Domain Causal Text-to-Motion Generation (TriC-Motion), a novel diffusion-based framework integrating spatial-temporal-frequency-domain modeling with causal intervention. TriC-Motion includes three core modeling modules for domain-specific modeling, namely Temporal Motion Encoding, Spatial Topology Modeling, and Hybrid Frequency Analysis. After comprehensive modeling, a Score-guided Tri-domain Fusion module integrates valuable information from the triple domains, simultaneously ensuring temporal consistency, spatial topology, motion trends, and dynamics. Moreover, the Causality-based Counterfactual Motion Disentangler is meticulously designed to expose motion-irrelevant cues to eliminate noise, disentangling the real modeling contributions of each domain for superior generation. Extensive experimental results validate that TriC-Motion achieves superior performance compared to state-of-the-art methods, attaining an outstanding R@1 of 0.612 on the HumanML3D dataset. These results demonstrate its capability to generate high-fidelity, coherent, diverse, and text-aligned motion sequences. Code is available at: \url{https://caoyiyang1105.github.io/TriC-Motion/}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces TriC-Motion, a new diffusion-based framework for text-to-motion generation that unifies spatial, temporal, and frequency-domain modeling under a causal learning paradigm. Authors address two limitations of existing approaches: 1) the lack of an integrated multi-domain representation that captures temporal dynamics, spatial joint topology, and motion frequency characteristics simultaneously, and 2) the entanglement of motion-relevant and irrelevant cues that degrade generation quality. To overcome these issues, TriC-Motion introduces three key components: Tri-Domain Modeling Modules, Score-guided Tri-domain Fusion, and Causality-based Counterfactual Motion Disentangler. Experimental results on HumanML3D and SnapMoGen demonstrate good performance gains over state-of-the-art baselines.

### Strengths
+ The performance of the proposed method on both HumanML3D and SnapMoGen dataset surpasses most baseline methods, proving the effectiveness of modeling all three domains.
+ The paper writing is clear and easy to follow.
+ The exploration of causal intervention in motion generation domain is interesting and inspiring.

### Weaknesses
- The main idea of modeling all three domains at the same time is less convincing and requires more intuitive explanation and theoretically grounded analysis.
- The architecture is heavy in computing and complex in the architecture. Therefore, it is important to also compare the run time, computation cost and model size with baseline methods.
- The training stability is also worrying due to the complex architecture. The loss weighting needs more deep analysis and sensitivity test.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This study proposes TriC-Motion, a diffusion-based framework for text-to-motion generation that integrates spatial, temporal, and frequency domain modeling with causal intervention to ensure temporal consistency, spatial topology, and dynamic coherence.
Experimental results show that TriC-Motion achieves an R1-Precision of 0.612 on the HumanML3D dataset, significantly outperforming existing methods and generating motions with superior realism, consistency, diversity, and text alignment.

### Strengths
1.The proposed method demonstrates strong text–motion consistency.

2.The introduction of causal learning reduces the impact of irrelevant information on motion generation.

### Weaknesses
1.Please provide t-SNE or other visualization analyses that disentangle motion-irrelevant and motion-relevant information to demonstrate the effectiveness of the proposed method.

2.The paper’s joint temporal–frequency–spatial strategy improves text–motion alignment (R@1, R@2, R@3), but FID did not improve; therefore you cannot claim that generation quality has improved, and the conclusions stated in the abstract are not supported.

3.What advantages does using DistilBERT for word-level and sentence-level feature encoding have compared to CLIP?

4.The proposed TME, STM, HFA, and S-Fus are commonly used extraction and fusion strategies in the temporal–spatial–frequency domain and lack novelty.

5.The methods compared in Figure 5 are all general approaches; it’s unclear whether the proposed method outperforms contemporary spatio-temporal modeling methods. If possible, provide qualitative comparisons (or quantitative comparisons if allowed) against methods from papers accepted to CVPR 2025, ICCV 2025, and NeurIPS 2025.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes TriC-Motion, a diffusion-based framework integrating spatial-temporal-frequency-domain modeling with causal intervention. It includes Temporal Motion Encoding, Spatial Topology Modeling, and Hybrid Frequency Analysis, with a Score-guided Tri-domain Fusion and a Causality-based Counterfactual Motion Disentangler to expose motion-irrelevant cues and fuse valuable information across domains.

### Strengths
1. The first work that simultaneously integrates spatial, temporal, and frequency domains into a unified motion generation framework

2. Introduces a causality-based counterfactual motion disentangler to expose motion-irrelevant cues and disentangle the real modeling contributions of each domain.

3. Provides ablation studies indicating the effectiveness of each domain branch and the causal-intervention design.

### Weaknesses
1. The paper uses a perceptual loss defined in the same motion–text embedding space used by the HumanML3D evaluator (the author could clear this point if I'm wrong). Using the same feature extractor for training and inference would inflate the performancve. The author could do an ablation study that removes this loss term to show that the R-precision gain is not from this loss term.

2. No visualization results. Quantitative metrics in text-to-motion is proven to be fragile and sometimes misaligned with human judgment. For motion, demo videos are necessary. I don’t see any supplementary videos, which makes it hard to judge the actual quality.

3. The metrics themselves aren’t convincing. Even if R-precision is SOTA, an R-precision much higher than the ground truth is not meaningful and doesn’t reflect visual quality. Even worse, FID is significantly poorer compared to current methods.

4. Missing strong baseline results. Compare against recent, stronger works (MARDM [1], MotionStreamer [2], MotionLCM v2 [3], SALAD [4]) with both qualitative and quantitative results. Also consider including a human study.

5. Minor typo. In the introduction you say R@1 is 0.573, which doesn’t match Table 1.

Reference

[1] Rethinking Diffusion for Text-Driven Human Motion Generation

[2] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space

[3] MotionLCM-V2: Improved Compression Rate for Multi-Latent-Token Diffusion

[4] SALAD: Skeleton-aware Latent Diffusion for Text-driven Motion Generation and Editing

### Questions
Refer to Weaknesses.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a 'tri-domain + causal' framework for text-to-motion generation. Built on MDM, it models motion in parallel with TME (temporal encoding), STM (skeleton-topology GCN), and HFA (hybrid frequency analysis via DWT+FFT). A dual-score S-Fus module fuses motion/semantic signals, and TIJ injects text via cross-attention. Training uses CCMD counterfactual decomposition (factual + counterfactual branches) to suppress spurious cues.

### Strengths
1. The proposed method achieves remarkable improvement on the R Precision metric.

2. The paper is well-written, ensuring that its content is easily understandable for readers.

3. It is the first time for casual learning to be used in text-to-motion generation, making significant contributions to the research community.

### Weaknesses
My primary concern is the choice of baselines. Under the HumanML3D evaluation protocol, the evaluator is too weak: many recent methods already surpass the 'ground truth', making R-Precision on HumanML3D unreliable. Meanwhile, the FID gap to stronger methods is large (0.285 vs 0.033), so the proposed method shows no advantage on HumanML3D. Porting the approach to a MoMask baseline should not be difficult; the authors should adopt a more appropriate baseline; otherwise, it may look like trading motion quality for text consistency, which does not substantiate effectiveness.

In addition, the ablation should include more combinatorial settings to better demonstrate effectiveness.

Finally, the paper lacks a demo video as supplementary material and a user study to subjectively assess text-motion alignment and motion quality.

### Questions
Please kindly refer to the weaknesses mentioned above.

### Soundness
2

### Presentation
3

### Contribution
2
