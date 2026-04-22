# Model Unmerging: Making Your Models Unmergeable for Secure Model Sharing

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Model merging leverages multiple finetuned expert models to construct a multi-task model with low cost, and is gaining increasing attention. However, as a growing number of finetuned models become publicly available, concerns about the safety of model merging have emerged. Unauthorized merging may infringe on developers' rights and risk leaking sensitive personal information. Most existing methods focus on detecting whether a merged model originates from a specific source model, but fail to effectively prevent illegal merging. In this paper, we propose MergeLock, an active protection mechanism that disrupts model parameters to render them unmergeable, thereby directly preventing unauthorized model merging. Specifically, leveraging the inherent symmetry of the attention mechanism in Transformer-based models, we randomly sample two pairs of invertible matrices and apply them to the Query-Key (QK) and Value-Output (VO) branches. This transformation keeps the model's output unchanged while pushing it away from the shared parameter space of other finetuned models. Extensive experiments across both vision and language tasks demonstrate that MergeLock can degrade the performance of merged models by over 95\% when a protected model is involved in most cases, demonstrating its effectiveness. Moreover, we further demonstrate that merged models protected by MergeLock cannot be effectively recovered using low-cost restoration methods, further enhancing robustness against unauthorized merging.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes "MergeLock," a method to prevent the unauthorized merging of fine-tuned models, thereby protecting intellectual property and mitigating potential security risks. The core idea is to apply structured, invertible transformations to the self-attention layers of Transformer-based models. These transformations, composed of random, permutation, and scaling matrices, push the model's parameters into a different loss basin while preserving its original performance. The authors argue that this makes the protected model incompatible with others for merging, as any attempt to merge them results in a model with severely degraded performance. The paper provides a theoretical analysis of symmetries in Transformer layers and presents extensive experiments across vision and language tasks to demonstrate MergeLock's effectiveness, showing that it significantly hinders model merging and is robust against a specific data-free alignment attack.

### Strengths
1. **Well-Defined and Relevant Problem:** The paper addresses a timely and important issue in the MLaaS ecosystem. As model merging becomes more common, the need for proactive defense mechanisms against unauthorized use and potential malicious merging is increasingly critical.
2. **Solid Theoretical Motivation:** The authors provide a clear explanation of the symmetry properties in both FFN and self-attention layers, building a convincing case for why self-attention layers are a more suitable and powerful target for creating unmergeable models. The use of invertible transformations to maintain functional equivalence is a sound and elegant approach.

### Weaknesses
1.  **Limited Novelty:** The core concept of using parameter-space transformations to disrupt model merging has been explored in prior work, most notably PaRaMS. While MergeLock employs a more complex transformation (random, permutation, and scaling matrices vs. just permutations and diagonal scaling), the fundamental idea is not entirely new. The contribution feels more like an incremental improvement over an existing concept rather than a novel paradigm.
2.  **Questionable Practicality:** The method requires authorized users to possess the transformation matrices (A and B) as a "secret key" to potentially reverse the process or merge models permissively. This introduces significant practical challenges that are not addressed. These matrices can be large, leading to substantial storage and distribution overhead. More importantly, it is unclear why this complex key-sharing mechanism is superior to simply encrypting the model weights with a standard password-based key, which would be more conventional and easier to manage.
3.  **Insufficient Evaluation of Alignment Attacks:** The paper's defense against alignment-based recovery is a key claim, but the threat model seems too narrow.
    *   The evaluation relies on a single, closed-form alignment strategy (the Kabsch algorithm). Given that the perturbations are more complex than simple permutations, more powerful iterative, gradient-based optimization techniques could potentially find better alignments and serve as a stronger attack baseline. The pre-trained model initialization and the often low-rank, sparse nature of fine-tuning updates could be exploited for more sophisticated attacks.
    *   The paper does not adequately discuss the instances where the alignment attack achieves partial success (e.g., performance on Cars in Table 1 improves from 3.4% to 6.2%, and on EuroSAT in Table 2 from 6.0% to 10.9%). Analyzing these cases could reveal potential vulnerabilities or failure modes of MergeLock, but they are currently overlooked.
4.  **Limited Experiments on Large Models:** While the inclusion of ViT-L/14 and Flan-T5 is a good step, the field is rapidly moving towards much larger models (e.g., Llama-3, Qwen-3, etc.). Demonstrating the method's effectiveness and scalability on models with tens or hundreds of billions of parameters would significantly strengthen the paper's impact and relevance.

### Questions
See weaknesses.

### Soundness
2

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
3

### Summary
The paper tackles unauthorized model merging and proposes MergeLock, a parameter transformation that makes a fine-tuned model functionally equivalent on its task yet unmergeable with other models. It exploits symmetry in Transformer self-attention: for each head it inserts two pairs of invertible matrices into the QK and VO branches so Q,K,V,O are re-parameterized without changing outputs. The

### Strengths
1. The problem is interesting with clear threat model & motivatio.
2. The method theoretically grounded and architecturally generic.
3. Empirical evidence shows the effectiveness of the method

### Weaknesses
1. Recent theory[1] formalizes continuous symmetries (rotation/gauge) for QK/VO in Transformers. Can this kind of methods recover the MergeLock?
2. Results are on CLIP-ViT and a mid-scale encoder-decoder LLM (Flan-T5). There’s no large decoder-only LLM (e.g., LLaMA-style) or cross-attention evaluation.

[1] Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion, ICML 2025

### Questions
Please refer to the Weakness.

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
The paper proposes the MergeLock method. Model merging is a method to take models trained on individual tasks and merges them to create a single multi-task capable model. As time goes by, more fine-tuned models get created and released to the public which are available for merging. However, this also opens up further opportunities for breaches in privacy and security since unapproved merging could violate developers’ intellectual property rights and potentially expose sensitive personal data. MergeLock leverages the symmetry property of the Feed Forward Network weight matrices and apply transformations such that the parameters reside in equivalent but separate spaces. This ensures the downstream task accuracy remains good while also changing the parameter spaces enough that merging the new transformed weights with other models corrupt the final merged model.

### Strengths
1. The problem tackled is interesting and is not yet well studied. This could be a good early contribution to a nascent field. 
2. The paper is well-written and the ideas are explained well.
3. The use of symmetry properties of FFNs is very creatively used here.
4. Good empirical results.

### Weaknesses
1. Incremental novelty compared to Wei et al (2025) (PaRaMS).
2. No significantly new observations or insights are presented.
3. Applicability and generalizability still in question.

### Questions
1. Please highlight more differences between PaRaMs and MergeLock. If it is just on how the distances differ in the space then that feels like a very lightweight contribution.
2. Some parts have strong similarities with the PaRaMs paper, for example the figures showing the basins in 2D space, the section MLP Parameter Rearrangement in PaRaMs and the Symmetry in Feedforward Networks section in MergeLock, etc. - please make them a lot more distinctive.
3. How does it perform across other types of LLMs or LLMs of different sizes which can have a high diversity of basins and basin sizes? It would be highly appreciated if the authors can make some comments or have some discussions on this.
4. How easy would it be to estimate A and B matrices if the base and other fine-tuning model weights are known? i.e. there is the statement "such protection can be reversed by simply applying the inverse permutation" made about Wei et al, but discussions about similar implications for MergeLock are missing.
5. How expensive is this process of applying A and B?

### Soundness
3

### Presentation
3

### Contribution
3
