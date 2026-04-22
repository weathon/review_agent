# DualToken: Towards Unifying Visual Understanding and Generation with Dual Visual Vocabularies

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
The differing representation spaces required for visual understanding and generation pose a challenge in unifying them within the autoregressive paradigm of large language models. A vision tokenizer trained for reconstruction excels at capturing low-level visual appearance, making it well-suited for visual generation but lacking high-level semantic representations for understanding tasks. Conversely, a vision encoder trained via contrastive learning aligns well with language but struggles to decode back into the pixel space for generation tasks. To bridge this gap, we propose DualToken, a method that unifies representations for both understanding and generation within a single tokenizer. However, directly integrating reconstruction and semantic objectives creates conflicts, leading to degraded performance in both reconstruction fidelity and semantic accuracy. Instead of forcing a single codebook to capture both visual appearance and semantics, DualToken disentangles them by introducing separate codebooks for high-level semantics and low-level visual details. As a result, DualToken achieves 0.25 rFID and 82.0% zero-shot accuracy on ImageNet, and demonstrates strong effectiveness in downstream MLLM tasks for both understanding and generation. Specifically, our method surpasses VILA-U by 5.8 points on average across ten visual understanding benchmarks and delivers a 13% improvement on GenAI-Bench. Notably, incorporating dual visual tokens outperforms using a single token type on both understanding and generation tasks. We hope our research offers a new perspective on leveraging dual visual vocabularies for building unified vision–language models. Project page is available [here](https://songweii.github.io/dualtoken-project-page/).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles the challenge of unifying visual understanding and generation within the autoregressive framework of large language models. The authors propose DualToken, which disentangles high-level semantic and low-level visual representations through separate codebooks to reconcile their conflicting objectives. This design enables effective integration of reconstruction and semantic learning, achieving strong results on both understanding and generation tasks. Overall, the work presents a well-motivated direction toward unified multimodal representation learning.

### Strengths
1. The paper presents a clear and well-articulated motivation, with a logical flow from problem identification to solution design.

2. The evaluation covers a broad range of downstream tasks across both visual understanding and generation, demonstrating comprehensive empirical validation.

3. The proposed disentangled dual-codebook design is conceptually elegant and effectively resolves the conflict between reconstruction and semantic alignment.

### Weaknesses
1. Although the proposed disentangled dual-codebook design is effective, it lacks substantial architectural novelty. Moreover, the insights presented in this work have already appeared in recent vision tokenizer studies, such as UniTok.

2. The claim that “dual tokens promote each other” seems somewhat overstated. As shown in Table 4 (rows (c) and (d)), the pixel tokens do not consistently enhance performance on understanding tasks.

3. The reported downstream results are primarily based on experiments with a 3B model. Additional studies across different LLM scales are needed to validate the method’s generality and scalability.

4. The paper lacks a detailed ablation study on how the boundary between low-level and high-level layers in the vision encoder is determined, which would help clarify the effectiveness of the proposed disentanglement.

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes DualToken, a unified tokenization pipeline that decomposes visual inputs into both content tokens and discrepancy tokens, aiming to improve visual-language and vision-only tasks under the same framework. The method is evaluated on several benchmarks to show improved representation consistency and task performance.

### Strengths
1. The motivation to unify visual tokenization across tasks is reasonable.

2. The writing is generally clear and the architecture is presented in an organized manner.

3. Experiments cover multiple downstream tasks.

### Weaknesses
1. The core idea of decomposing visual features into semantic and pixel parts has been explored extensively in prior tokenizer and unified representation works (e.g., TokenFlow, UniTok, UniToken, Bagel, Show-o2, etc). The paper does not clearly articulate what is conceptually new here.

2. The experimental comparisons omit recent strong tokenizer-based or unification SOTA models. Without these, the claimed performance improvement is not convincing.

3. The paper frames itself as “unifying tokenizers,” but the design appears largely heuristic, and no strong evidence is provided that this formulation generalizes broadly or scales to high-performing unification models.

4. There is limited explanation of how or why the proposed discrepancy tokens help. The model may simply benefit from an increased token budget rather than a meaningful representational factorization.

5. Some baselines are outdated or under-optimized. The improvement margins are small, and significance is unclear.

6. The conclusion reads brief and generic. It does not help reinforce the contributions or position the work in the broader literature.

### Questions
1. Why were stronger modern tokenizers not included in comparisons? Without these, the results are incomplete.

2. Can the authors provide concrete justification that the content/discrepancy decomposition corresponds to meaningful semantic factors, rather than just acting as an ad-hoc feature split?

3. What is the computational overhead introduced by the two-branch tokenization versus standard single-stream tokenizers?

4. How does the method perform when scaling to higher-resolution or longer input sequences?

5. Could the authors release qualitative examples to demonstrate that discrepancy tokens capture non-trivial representation variation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes DualToken, a unified visual tokenizer that decouples the tokenization process by applying quantization to shallow and deep layers of a single SigLIP backbone for reconstruction and semantic learning, respectively. The resulting tokenizer demonstrates strong reconstruction performance and achieves consistent improvements over VILA-U across both visual understanding and generation tasks.

### Strengths
- The proposed method demonstrates solid results across both understanding and generation tasks.
- The paper is well written and easy to follow.

### Weaknesses
1. Insufficient Experiments: The paper would benefit from more comprehensive ablations on model design, such as analyzing how the choice of reconstruction layer affects the final performance.
2. Limited Baseline Comparison: The single-encoder vs. dual-encoder experiment lacks comparison with stronger dual-branch architectures, which undermines the validity of the method’s claimed effectiveness.

### Questions
1. Table 1 would benefit from a clearer organization, for example by grouping models according to factors such as the number of model parameters, downsampling ratio or number of tokens, codebook size.
2. The experiment of comparing single-encoder versus dual-encoder designs would be more convincing if compared against recent dual-branch architectures such as TokenFlow[1] or MuseVL[2], which also aim to reconcile semantic and reconstruction requirements rather than merely stacking two unrelated encoders.

[1] Qu, Liao, et al. "Tokenflow: Unified image tokenizer for multimodal understanding and generation." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[2] Xie, Rongchang, et al. "Muse-vl: Modeling unified vlm through semantic discrete encoding." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

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
4

### Summary
This paper proposes DualToken, a novel approach aimed at unifying visual understanding and generation within autoregressive large language models (LLMs). The core idea is to introduce dual visual vocabularies—one for low-level visual details (used in generation) and another for high-level semantics (used in understanding). By decoupling these two objectives, the authors resolve the conflict between reconstruction and semantic tasks, leading to improved performance in both areas. The approach also shows strong effectiveness in multimodal understanding benchmarks, making it a promising solution for unified vision-language models. The method is efficient, with minimal additional computational overhead, and the results highlight the advantages of unifying visual understanding and generation in a single framework.

### Strengths
The writing in this paper is clear and well-structured, effectively explaining the novel concept of dual visual vocabularies and its benefits in unifying visual understanding and generation. The experimental design is robust, with thorough evaluations across multiple benchmarks, demonstrating the superiority of DualToken over existing methods in both reconstruction and semantic tasks. The authors also provide comprehensive comparisons and detailed ablation studies, which strengthen the validity of their claims and offer valuable insights into the practical impact of their approach.

### Weaknesses
1.Heuristic layer selection limits generalization and reusability. The method applies reconstruction supervision to early layers and semantic supervision to a fixed deep layer, yet this design choice lacks thorough ablation studies and relies heavily on heuristics, raising concerns about its generalizability across architectures and tasks.

2.Overly simplified semantic supervision may undermine representation quality. The use of L2 regression to pretrained SigLIP’s final-layer features, without contrastive objectives or cosine similarity constraints, raises questions about whether this feature-level alignment is sufficient to preserve rich semantic capacity.

### Questions
Please refer to the above weakness.

### Soundness
3

### Presentation
3

### Contribution
3
