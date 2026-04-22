# HiddenGuard: Detecting and Interpreting NSFW Prompts in Text-to-Image Models through Uncovering Harmful Semantics

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 4

## Abstract
As text-to-image (T2I) models advance and gain widespread adoption, their associated safety concerns are becoming increasingly critical. Malicious users exploit these models to generate Not-Safe-for-Work (NSFW) images using harmful or adversarial prompts, underscoring the need for effective safeguards to ensure the integrity and compliance of model outputs. However, existing detection methods often exhibit low accuracy and inefficiency.
In this paper, we propose HiddenGuard, an interpretable defense framework leveraging the hidden states of T2I models to detect NSFW prompts. HiddenGuard extracts NSFW features from the hidden states of the model's text encoder, utilizing the separable nature of these features to detect NSFW prompts. The detection process is efficient, requiring minimal inference time. HiddenGuard also offers real-time interpretation of results and supports optimization through data augmentation techniques. Our extensive experiments show that HiddenGuard significantly outperforms both commercial and open-source moderation tools, achieving over 95\% accuracy across all datasets and improving efficiency by at least tenfold.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces HiddenGuard, an interpretable defense framework for Text-to-Image (T2I) models that detects Not-Safe-for-Work (NSFW) prompts by analyzing the intermediate hidden states of the model's text encoder. The key insight is that harmful semantics form linearly separable clusters within specific attention heads' hidden states, allowing HiddenGuard to extract NSFW features and calculate an NSFW score for prompt detection. Through extensive experiments, HiddenGuard demonstrates superior performance with over $95\%$ accuracy across all datasets and significantly greater computational efficiency compared to commercial and state-of-the-art moderation tools, while also offering text- and image-based interpretation of results.

### Strengths
1. This paper is easy to follow.
2. This detection method should be time-efficient.
3. This method outperforms compared baselines.

### Weaknesses
1. The performance of this detection method on benign images are not well evaluated.
2. The evaluation only compares safety filter baselines.
3. Some figures are blurry and different from other figures.

### Questions
1. How is time cost of HiddenGuard compared with other baselines?

2. As shown in Figure 2, it seems benign and adv/nsfw examples are still mixed up?

3. What about other alignment-based methods for preventing the generation of harmful images, such as SafeGen, Mace, SafeText? They are not included in the evaluation.

4. Are benign prompts/images remaining the same after the detection method?

5. Figure 6 looks blurry.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces HiddenGuard, a novel defense framework for detecting NSFW prompts in text-to-image models. The core problem is that malicious users can craft prompts to generate harmful content, and existing detection methods based on raw text or final embeddings are often inefficient or inaccurate. The key insight of HiddenGuard is that NSFW semantics, while entangled in the final embedding space, form linearly separable clusters within the intermediate hidden states of the model's text encoder. The authors conduct extensive experiments showing that HiddenGuard outperforms several commercial and open-source baselines in both effectiveness and computational efficiency.

### Strengths
1. The experimental setup is thorough. The authors compare HiddenGuard against a strong set of eight baselines, including commercial APIs and state-of-the-art models, on a diverse collection of datasets that include both regular and adversarial prompts.
2. The dual-modal interpretation framework is interesting. By providing explanations for both why a prompt is flagged and how the harmful content manifests in the output.

### Weaknesses
1. The paper presents the hidden state's difference in different attention head as a key design intuition motivating the entire framework. However, the concept of semantic specialization in attention heads is a well-established phenomenon in the broader transformer literature for both LLMs and VLMs. The paper fails to properly contextualize this observation within prior work on model interpretability, thereby overstating the novelty of its foundational premise.
2. The motivation for using pre-MLP hidden states hinges on the claim that "some information may be obscured or discarded during this process" by the MLP layer. This is a critical assertion that is never substantiated with evidence. The paper would be much stronger if it demonstrated empirically that representations from $ATT^l(Z^{l-1})$ are indeed superior for this task compared to $Z^l$. Without this evidence, the design choice feels more like a heuristic than a principled decision.
3. The theoretical analysis of adversarial robustness in Appendix A.7 is based on a local Lipschitz continuity assumption. This assumption connects the perturbation in the final embedding space to changes in the hidden-state-based score function. However, the paper provides no empirical validation for this assumption. It is unclear how large the Lipschitz constant $K$ is in practice or how stable it is across different models and prompts. Without such validation, the derived certified robustness radius remains purely theoretical and its practical relevance is questionable.
4. The paper suffers from minor but noticeable presentation issues that detract from its quality. For instance, line 226 misses a period.
5. The paper lacks a fine-grained analysis of performance across different NSFW categories. While Table 7 shows accuracy for a multi-category classifier, a full breakdown of TPR and FPR for each category (sexual, violence, hate, etc.) for both HiddenGuard and the baselines is missing. Overall metrics can obscure critical failures where a model performs well on one type of harmful content but fails on another, which is essential information for a safety-focused tool.
6. The paper highlights TPR@1%FPR as a key metric. While a standard benchmark, a 1% False Positive Rate is arguably too high for real-world deployment. In a large-scale system, this would incorrectly block a massive number of benign prompts, leading to a poor user experience. A more compelling evaluation would assess performance at much lower FPRs (0.1% or 0.01%) to better demonstrate the model's practical viability.

### Questions
1. Can you provide an ablation study comparing the performance of HiddenGuard when using pre-MLP hidden states versus post-MLP hidden states?
2. Can you provide a more detailed performance breakdown, including both TPR and FPR, for each distinct NSFW category (sexual, violence, hate, and so on)? How does HiddenGuard's performance on these specific categories compare to the baselines?
3. Given that a 1% FPR can be prohibitive for real-world deployment, could you provide evaluation results at lower FPR thresholds, such as TPR@0.1%FPR or TPR@0.01%FPR? This would provide a more realistic picture of HiddenGuard's practical utility.

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
3

### Summary
This paper proposes HiddenGuard, an interpretable defense framework for detecting NSFW prompts in text-to-image models by analyzing hidden states of the text encoder. It identifies linearly separable NSFW semantic features in attention heads, enabling accurate and efficient detection. The method supports real-time interpretation across text and image modalities and resists adversarial attacks. Experiments show HiddenGuard outperforms SOTA and open-source tools and minimal computational overhead.

### Strengths
- **Significance**: This work is to investigate the separability of NSFW semantics within the hidden states of text encoders in text-to-image (T2I) models. It introduces HiddenGuard, a unified defense framework that is efficient, interpretable, and optimizable.  
- **Innovation**: The paper proposes a new NSFW detection paradigm based on intermediate hidden states rather than raw input text or final embeddings. By integrating multi-head attention feature aggregation with a dual-modality explanation mechanism, the approach achieves strong interpretability and generalization capability.  
- **Performance**: The method consistently outperforms state-of-the-art approaches across nine benchmark datasets and demonstrates robustness against unseen adversarial attacks.

### Weaknesses
1. The proposed method relies on a pre-constructed NSFW prompts dataset. Although this dataset is compiled from existing open-source datasets, the effectiveness of these prompts across different text-to-image (T2I) models has not been rigorously validated. As T2I models evolve, some prompts may become ineffective or obsolete, potentially introducing significant noise into the dataset and undermining the reliability of the method.
2. Although Figure 1 illustrates the feature distributions across different layers, highlighting the differences between NSFW-related features and benign features. It does not address whether distinct categories of NSFW features are entangled or coupled with each other. Since Table 7 reports detection performance for different NSFW categories, an effective method, in principle, should be capable of clearly distinguishing between these categories at the feature level. If the proposed approach truly disentangles NSFW semantics as claimed, it should inherently support fine-grained discrimination among different types of NSFW prompts. The authors should investigate and discuss potential feature coupling across NSFW categories to better validate this capability.
3. Text-to-image (T2I) models are inherently coupled architectures that jointly model textual and visual semantics. However, the proposed method only examines textual features and completely ignores visual signals. I am uncertain whether this text-only approach will remain effective as T2I models continue to evolve—especially since newer models may encode or obscure NSFW content in ways that are not fully reflected in the text encoder’s hidden states alone.  
4. The method appears tightly constrained by the scope of the curated NSFW prompts dataset. It is unclear whether the approach can generalize to out-of-distribution or unseen types of harmful prompts that fall outside the categories covered in the training data. If a malicious prompt uses novel phrasing, metaphors, or domain-specific language not present in the dataset, the detector may fail to recognize it, raising concerns about real-world robustness and coverage.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes HiddenGuard, an interpretable and efficient defense framework that leverages the hidden states of text-to-image (T2I) models to detect NSFW prompts. By extracting separable NSFW features from the model’s text encoder, HiddenGuard enables accurate and real-time detection with minimal inference cost. Extensive experiments demonstrate that HiddenGuard outperforms both commercial and open-source moderation tools, achieving over 95% accuracy across multiple datasets while significantly improving computational efficiency.

### Strengths
1. The ROC curves consistently lie above the baselines and remain stable across different datasets.

2. The proposed method demonstrates insensitivity to training data size — even with a small number of labeled samples, it achieves clear separation between “NSFW” and “benign” content.

3.  The method shows robustness against adaptive attacks, as it does not rely on keyword lists but rather on directional semantics, making it resilient to targeted evasion attempts.

4. Instead of classifying final text embeddings, the authors extract linearly discriminative directional vectors from intermediate attention outputs (layer-by-layer and head-by-head) using LDA, and compute scores via projection along these directions. This effectively avoids the problem of “semantic entanglement” in final embeddings.

5. The workflow is well designed: the front-end detector decides whether to allow or block content; on the text side, it assigns token-level importance; on the image side, it performs progressive denoising along the “NSFW direction” to generate comparison images. Samples that manage to bypass detection are reintroduced for data augmentation, enabling iterative feature reinforcement.

### Weaknesses
1. The construction of the benign dataset might be overly optimistic — using MSCOCO filtered by sensitive keywords may not adequately test against hidden or paraphrased expressions, potentially leading to overestimation of performance.

2. Evaluation on the generation side is conducted only on Stable Diffusion v1.4, which is widely known to be one of the most vulnerable models. Therefore, the observed decline in ASR (Attack Success Rate) under white-box conditions may be overestimated.

3. The paper assumes that in CLIP, causal masking in self-attention leads to semantic aggregation only at the EOS token. However, it is unclear whether this assumption holds for bidirectional (non-causal) encoders — this point remains conceptually confusing.

4. The paper claims “significantly lower average query time,” but it does not specify the computation environment (GPU/CPU, batch size, or threading setup), making it difficult to interpret the reported efficiency gains.

5. From a theoretical perspective, Section A7.1 introduces a local Lipschitz constant K and defines a “certified robustness radius” in A7.2. In principle, K must have an upper bound: if estimated globally, the large K value (due to deep stacked layers and piecewise-linear + softmax structure in attention) would cause the radius to shrink to zero, rendering the certification meaningless. Thus, K should be locally or sample-dependently estimated. Although the theory section acknowledges this, the implementation does not specify the norm type, estimation procedure, or concrete values for K, leaving the process somewhat opaque and non-reproducible.

### Questions
Refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
