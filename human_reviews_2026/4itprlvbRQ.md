# LayerSync: Self-aligning Intermediate Layers

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
We propose LayerSync, a domain-agnostic approach for improving the generation quality and the training efficiency of diffusion models. Prior studies have highlighted the connection between the quality of generation and the representations learned by diffusion models, showing that external guidance on model intermediate representations accelerates training. We reconceptualize this paradigm by regularizing diffusion models with their own intermediate representations. Building on the observation that representation quality varies across diffusion model layers, we show that the most semantically rich representations can act as an intrinsic guidance for weaker ones, reducing the need for external supervision. Our approach, LayerSync, is a self-sufficient, plug-and-play regularizer term with no overhead on diffusion model training and generalizes beyond the visual domain to other modalities. LayerSync requires no pretrained models nor additional data. We extensively evaluate the method on image generation and demonstrate its applicability to other domains such as audio, video, and motion generation. We show that it consistently improves the generation quality and the training efficiency. For example, we speed up the training of flow-based transformer by over 8.75$\times$ on ImageNet dataset and improve the generation quality by 23.6\%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents LayerSync, a self-contained, plug-and-play regularization method for diffusion models leveraging their own intermediate representation. Based on the analysis on internal representations, LayerSync uses the model’s semantically rich layers as intrinsic guidance requiring no additional parameters or external components. Experimental results show consistent improvements in generation quality and training efficiency across multiple domains including, image, audio, video, and human motion.

### Strengths
- The overall flow of the paper is clear. To overcome the limitation of relying on external alignment, the authors first analyze the internal representations of diffusion models using discriminative tasks. They then propose a simple self-contained alignment between earlier and later layers, where the latter hold richer semantic information. Finally, they validate the idea through cross-domain experiments and detailed analysis of internal representations.

- The main strengths of the paper are broad generalization and thorough experiments. The proposed method is lightweight and domain-agnostic, showing consistent improvements across different modalities without the need for extra losses or external modules.

### Weaknesses
- The paper gives a simple heuristic for choosing which layers to align (Section 3.4), but more experiments on this would help. In Figure 3, layer 20 seems to work best for two tasks, yet the paper aligns layer 8 → 16 instead. Also, both SiT-L and SiT-XL use the same 8th layer (from Table 8). A layer-pair ablation could show which combinations are most effective and why.
- The authors claim that “LayerSync establishes a virtuous cycle that progressively refines the entire feature hierarchy,” but this isn’t clearly supported. It would be more convincing to show how the representations evolve during training or how this behavior changes with different layer selections.
- It seems the alignment is applied without considering timestep. Is there any analysis on how aligning at different timestep affects the results?

### Questions
- How does LayerSync perform with CFG in Table 2?
- The overall writing and presentation could be improved for better readability. Some sections, such as Section 4.3 (Results), would need detailed explanations. There are also a few minor typos and formatting issues (e.g., line 265 “table → Table”, 266 “SiTXL → SiT-XL”, 327 “10.000 → 10,000”, 355 missing space after “Results.”, 359 missing comma before “we”, Eq (6) notation issue).

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents LayerSync as a self-sufficient, plug-and-play regularization for diffusion models across modalities, but the claims are undermined by a lack of solid theoretical grounding and insufficiently rigorous empirical validation. The experimental comparisons appear incomplete, and I identified errors in data analysis that could misrepresent the method’s effectiveness (e.g., the reported speedups and generalization results may be overstated). Without stronger theoretical justification, more comprehensive baselines, and corrected analyses, the work risks misleading reviewers and does not convincingly support its central contributions.

### Strengths
- The core idea behind LayerSync is notably simple and clear, which contributes to the overall readability and accessibility of the paper.
- The authors provide comprehensive empirical validation across multiple domains, including image, video, and audio generation, demonstrating the versatility of their approach.
- Compared to the selected baselines, the proposed method exhibits a tangible improvement in convergence speed, as evidenced by the experimental results.

### Weaknesses
- The choice of SiT as the sole baseline is not sufficiently convincing. Although Table 1 demonstrates some improvements when LayerSync is added to SiT under various model sizes, the baseline itself performs relatively poorly. Achieving gains over a weak baseline does not provide strong evidence of the proposed method's effectiveness.
- In the experiments presented in Table 2, LayerSync does not outperform SiT-XL/x+REPA or SiT-XL+REED in terms of FID or training epochs. However, the authors highlight their own results in bold, which may mislead reviewers. Furthermore, the experimental comparison remains limited to relatively weak baselines and does not include more recent and competitive approaches such as DJEPA or VAR. As a result, the evaluation is not comprehensive, and it is difficult to assess the true effectiveness of the proposed architecture.。

### Questions
- The paper lacks solid theoretical analysis of the LayerSync mechanism itself. In the original SiT architecture, skip connections between blocks already help maintain feature consistency across layers. The additional loss introduced by LayerSync may simply increase the overall gradient magnitude, which could accelerate convergence in a manner similar to using a larger learning rate. While this is a hypothesis, it raises questions about the underlying mechanism driving the observed improvements.
- I strongly encourage the authors to provide more theoretical insights and empirical evidence regarding LayerSync in their rebuttal. For example, is the gradient norm significantly larger compared to the baseline? Would directly amplifying the feature gradients at corresponding locations in the baseline yield similar training benefits? Addressing these points would greatly clarify the contribution and effectiveness of LayerSync.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to accelerate the training of diffusion transformers without external guidance to avoid dependency on an external model that introduces additional training cost and limited domains. To this end, the authors propose LayerSync, a simple yet effective model that uses features from a later layer of the diffusion model itself as an alignment target. They show that LayerSync accelerates training of diffusion models in a domain-agnostic manner.

### Strengths
1. This paper proposes a simple yet effective framework that accelerates the training of the diffusion transformer.

2. This paper shows that LayerSync is modality-agnostic (image, audio, human motion), demonstrating the effectiveness of the proposed method in various domains that may have challenges in using pre-trained representations.

3. This paper introduces the heuristic guidance to find a layer to align, which is highly practical for using diffusion models in different domains.

### Weaknesses
1. A comparison to a more alignment method is needed. For instance, despite SRA [1] using the EMA model (which can make training of DiT slower), I think comparison with it in terms of training time (e.g., wall-time clock) is important to claim that we indeed need an alignment strategy that does not depend on an additional module.

2. The authors claim that REPA has limitations in that a pre-trained representation model is not available for various domains. However, several studies have proposed methods to obtain a pre-trained representation model in a domain-agnostic manner [2-4]. If we use such models, we may accelerate the training of the diffusion model even faster than LayerSync (in terms of training time), and we can use pre-computed features to reduce the memory cost.

[1] Jiang et al., No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves, Arxiv 2025 \
[2] Tamkin et al., DABS: A Domain-Agnostic Benchmark for Self-Supervised Learning, NeurIPS 2021 \
[3] Baevski et al., Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language, Arxiv 2022 \
[4] Jang et al., Modality-Agnostic Self-Supervised Learning with Meta-Learned Masked Auto-Encoder, NeurIPS 2023

### Questions
Please answer the weaknesses.

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
5

### Summary
This paper introduces LayerSync, a parameter-free, self-contained regularization strategy for diffusion models. Unlike previous work that aligns diffusion model representations with large external models, LayerSync aligns a model’s own weaker intermediate layers to its stronger, semantically richer layers, thus providing internal guidance. The method is simple (involving an intra-model alignment loss over patchwise cosine similarities), incurs negligible overhead, and is broadly applicable across data modalities—including images, audio, motion, and video. Empirical results on ImageNet, MTG-Jamendo (audio), HumanML3D (motion), and CLEVRER (video), as well as comprehensive ablations and internal feature analysis, demonstrate that LayerSync consistently outperforms prior self-contained baselines and narrows the performance gap with externally guided methods.

### Strengths
- LayerSync is evaluated across four distinct modalities (image, audio, motion, and video), each with strong quantitative results and improvements in training efficiency, generation quality, and representation quality (Tables 1–4, 7; Figure 1). These cross-domain demonstrations suggest the approach’s generality.
- Unlike methods requiring external pretrained models (e.g., DINOv2, VLMs), LayerSync requires no additional data or parameters, making it attractive for non-visual domains and resource-limited scenarios.
- For image generation on ImageNet, LayerSync accelerates convergence by over 8.75x (Table 1/Figure 1b) compared to baseline diffusion transformers, and outpaces self-contained alternatives like Dispersive.
- Qualitative and quantitative comparisons (e.g., Figure 2, Figure 3, Figure 15-16) support claims about quality and internal changes, and supplementary figures show robust improvements (e.g., noise robustness, block dropping effects).

### Weaknesses
- The strategy for picking “strong” and “weak” layers is based on architectural distance and exclusion of “final 20%” blocks. While Table 5 suggests robustness to randomization, the method remains ad-hoc, and a more systematic (possibly data-driven or information-theoretic) approach could yield better guarantees. There’s no analysis as to how the reference layer selection interacts with model depth or varying data/architecture.
- The paper is primarily empirical; while this is fine, and the practical results are compelling, the absence of theoretical grounding for the expected improvements (e.g., analysis on generalization, capacity, information propagation) or limitations of the approach means its scope is less clear. For instance, are there pathological settings (e.g., excessive over-alignment leading to collapse) or fundamental limits with this approach where external representations would still be needed?
- Although the paper notes that training with LayerSync increases resilience to block removal, it is not clear whether stronger self-alignment could lead to diminished feature diversity, especially in representations, and whether downstream performance is robust to such collapse.
- For audio, motion, and video, the paper mostly reports aggregate scores (e.g., FAD, FVD, FID) with limited visualization or error analysis of failure modes. Given the modality-general claim, richer qualitative evidence would reinforce the universality claim.

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
