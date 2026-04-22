# SPRINT: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 2, 6

## Abstract
Diffusion Transformers (DiTs) deliver state-of-the-art generative performance but their quadratic training cost with sequence length makes large-scale pretraining prohibitively expensive. Token dropping can reduce training cost, yet naïve strategies degrade representations, and existing methods are either parameter-heavy or fail at high drop ratios. We present SPRINT (Sparse--Dense Residual Fusion for Efficient Diffusion Transformers), a simple method that enables aggressive token dropping (up to 75%) while preserving quality. SPRINT leverages the complementary roles of shallow and deep layers: early layers process all tokens to capture local detail, deeper layers operate on a sparse subset to cut computation, and their outputs are fused through residual connections. Training follows a two-stage schedule: long masked pre-training for efficiency followed by short full-token fine-tuning to close the train--inference gap. On ImageNet-1K 256^2, SPRINT achieves 9.8x training savings with comparable FID/FDD, and at inference, its Path-Drop Guidance (PDG) nearly halves FLOPs while improving quality. These results establish SPRINT as a simple, effective, and general solution for efficient DiT training.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper addresses the prohibitive quadratic training cost of Diffusion Transformers (DiTs). The authors propose SPRINT (Sparse-Dense Residual Fusion), a novel training and architecture strategy that enables aggressive token dropping (up to 75%) while preserving quality. SPRINT divides the DiT into three parts: a dense shallow encoder (fθ) that processes all tokens to capture local detail, a sparse deep middle block (gθ) that processes a subset of tokens to model global semantics, and a decoder (hθ) that fuses features from both paths via a residual connection. The method uses a two-stage training schedule (long sparse pre-training followed by short full-token fine-tuning) and a structured group-wise token sampling strategy to ensure local coverage. Additionally, the paper introduces Path-Drop Guidance (PDG), an efficient substitute for Classifier-Free Guidance (CFG) that nearly halves inference FLOPs by bypassing the expensive middle blocks during the unconditional pass. Experiments on ImageNet show SPRINT achieves significant training speedups (e.g., 9.8x fewer TFLOPs to reach comparable quality to SiT-XL) and improved inference efficiency with PDG.

### Strengths
1. Significant Training Efficiency: The primary strength is the massive reduction in training computation. The paper demonstrates that SPRINT can reach a quality level comparable to the 1400-epoch SiT-XL baseline in only 200 epochs, translating to a 9.8x reduction in TFLOPs. This is a highly practical and valuable contribution. 
 2. Novel and Effective Inference Acceleration (PDG): The proposed Path-Drop Guidance (PDG) is an excellent finding. It cleverly repurposes the SPRINT architecture to create an efficient alternative to CFG, using the shallow path as a ""weaker network"". This method nearly halves inference FLOPs while also improving generation quality, as shown in Tables 1, 2, 3, and 10, and Figure 7. 
 3. Simplicity and Generality: The SPRINT framework is conceptually simple and not tied to a specific architecture. The authors demonstrate its broad applicability by successfully integrating it with SiT, U-ViT, and the alignment-based REPA, showing consistent and significant improvements in all cases. 
 4. Strong Empirical Validation: The paper is supported by extensive experiments and insightful ablations. The validation of the sparse-dense design (Table 4), the structured sampling strategy (Table 5), and the optimal layer allocations (Tables 6, 8) builds strong confidence in the method's design. The analysis of feature specialization (Fig. 4) provides good intuition for why the method works.

### Weaknesses
1. Limited Ablation on Fine-tuning Stage: The paper adopts a fixed two-stage schedule: long sparse pre-training followed by a 100K-iteration full-token fine-tuning stage. While effective, the sensitivity to the duration of this fine-tuning is not analyzed. It is unclear how much fine-tuning is necessary to close the train-inference gap or if more fine-tuning would yield further gains. An ablation on this hyperparameter would strengthen the paper. 
 2. Conceptual Justification for PDG: The paper empirically shows that PDG works well by using the shallow path fθ as the unconditional estimate, inspired by Auto Guidance. This is effective, but a deeper conceptual analysis of why fθ is a suitable unconditional predictor would be beneficial. Does the SPRINT training force fθ to learn an ""average"" or ""blurry"" representation that mimics a true unconditional pass? A brief discussion would be insightful.

### Questions
1.The performance of PDG is excellent. Does the SPRINT training enable PDG? In other words, could PDG (i.e., using only the first few layers fθ for the unconditional pass) be applied as an inference-time optimization to a standard, densely trained SiT model, or is the sparse-dense training process necessary for fθ to become an effective unconditional predictor?
2. Could you provide experimental results on text-to-image models (such as PixArt, Flux), similar to the fine-tuning experiments on FLUX, to more convincingly demonstrate the effectiveness of your proposed method?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces SPRINT, a novel framework for efficient training of Diffusion Transformers (DiTs) by leveraging sparse-dense residual fusion. It enables aggressive token dropping (up to 75%) while preserving representation quality, significantly reducing training costs (up to 9.8×) and inference FLOPs. SPRINT trains DiTs in two stages: sparse pre-training and short full-token fine-tuning to bridge the train-inference gap. It also introduces Path-Drop Guidance (PDG), a more efficient alternative to classifier-free guidance, further improving generation quality and efficiency. The method is simple, architecture-agnostic, and applicable across various resolutions and models.

### Strengths
+ Good performance
+ The proposed Dense shallow path  and sparse deep path can effectively accelerate the training speed.

### Weaknesses
1. More discussion on Path-Drop Guidance should be included in the Introduction. Currently, the manuscript treats it as merely a supplementary design.

2. The font size in the tables should be consistent.

### Questions
see weakness

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
This paper proposes SPRINT, a training method for Diffusion Transformers (DiTs) that aims to reduce training costs through aggressive token dropping (up to 75%). The core idea is to partition the DiT into encoder-middle-decoder components, where the encoder processes all tokens, the middle blocks operate on sparse tokens, and outputs are fused through residual connections. The authors claim 9.8x training savings with comparable quality on ImageNet-1K $256^2$.

### Strengths
1. Practical Problem: The paper addresses the important issue of quadratic training costs in DiTs, which is highly relevant for the community.
2. Strong Empirical Results: The reported 9.8x training speedup with maintained quality is impressive if valid.
3. Architecture Agnostic: The method appears to work across different architectures (SiT, UViT) and can be combined with other techniques like REPA.
4. Comprehensive Experiments: The paper includes extensive ablations and analysis across multiple settings.

### Weaknesses
Major Concerns

1. Limited Technical Novelty.
The core contribution appears to be a modification of MDTv2, essentially replacing the side-interpolator with simple residual connections. The encoder-middle-decoder architecture is questonable, and the paper fails to provide compelling theoretical or empirical justification for why this specific design should outperform existing methods like MDTv2.

2. Insufficient Comparison with Prior Work.
The paper does not adequately explain why SPRINT should be superior to MDTv2. The fundamental question remains unanswered: what specific advantages does replacing MDTv2's side-interpolator with residual connections provide? The paper lacks rigorous analysis of this key design choice.

3. Questionable Performance Claims.
From Table 3, SPRINT appears to underperform compared to MDTv2 (in terms of FID). This contradicts the paper's claims of superiority and raises questions about the experimental setup and evaluation fairness.

4. Lack of Theoretical Foundation.
The paper provides insufficient theoretical justification for why the proposed encoder-middle-decoder architecture can support such high drop rates (75%). The explanation about shallow vs. deep layer specialization is intuitive but lacks rigorous analysis or proof.

5. Missing Critical Analysis.
The paper doesn't adequately address:
- Why simple residual fusion should be better than more sophisticated fusion mechanisms
- Why SPRINT can tolerate 75% drop rate
- How the method compares to MDTv2 in controlled settings with identical experimental conditions

Minor Issues
- Writing Quality: Some sections lack clarity, particularly the technical description of the fusion mechanism.
- Experimental Setup: More details needed on fair comparison protocols with baseline methods.
- Ablation Studies: While extensive, the ablations don't address the core question of why (if really so) this approach works better than MDTv2.

### Questions
1. Can you provide a direct, controlled comparison with MDTv2 using identical experimental settings?
2. What theoretical or empirical evidence supports the advantages of the residual fusion over MDTv2's approach?
3. Why are the FLOPs for MDTv2 missing in Table 3? It has open-sourced official code, you should check it and measure the costs.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes SPRINT, a method to accelerate Diffusion Transformer (DiT) training through sparse-dense residual fusion. SPRINT processes all tokens in shallow layers for local details, drops 75% of tokens in deep layers for global semantics, and fuses outputs via residual connections. Training uses sparse pre-training followed by brief full-token fine-tuning. On ImageNet-1K 256×256, SPRINT achieves 9.8× training speedup with comparable quality.

### Strengths
(1)SPRINT achieves 9.8× training speedup on ImageNet-1K while maintaining comparable quality. The method adds only 0.3% parameters and preserves standard DiT blocks, making it easy to integrate. Strong generalization across architectures (SiT, U-ViT, REPA) demonstrates practical value.
(2)Path-Drop Guidance (PDG) halves inference FLOPs while improving quality. Comprehensive experiments reveal complementary roles of sparse-deep and dense-shallow features, providing valuable insights into DiT representation mechanisms and explaining why the sparse-dense fusion design is effective.

### Weaknesses
(1)The paper claims that two-stage training can "close the train-inference gap," but does not quantify how large this gap actually is.

### Questions
(1)Why is there no comparative validation using features from different layers for unconditional guidance?
(2)After pre-training with 75% drop ratio, if full-token inference is performed directly (without fine-tuning), how much would the performance degrade?

### Soundness
3

### Presentation
3

### Contribution
3
