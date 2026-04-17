# DEPTRAI: Detachable External‐memory layer for Parameter-Transformer Injection

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large language models (LLMs) quickly become outdated because the factual knowledge they encode is fixed at training time, and retraining for every new fact is prohibitively expensive. Prior ``internal'' editors apply closed-form perturbations directly to the feed-forward weights, but each new patch is applied in place to the base model, causing edits to accumulate, interfere, and preventing straightforward revocation. We present DEPTRAI—\textbf{D}etachable \textbf{E}xternal-memory layer for \textbf{P}arameter-\textbf{Tra}nsformer \textbf{I}njection—that stores each edited fact as a key–value tuple outside the model, leaving all original weights frozen. At inference, the frozen FFN produces a subject key, which is routed to the nearest stored key using a Mahalanobis metric that mirrors the inverse-covariance scaling of closed-form editors. A lightweight gate then either substitutes the edited value or preserves the base projection. This design turns factual patching into a reversible database-style update rather than a permanent modification of parameters. DEPTRAI achieves the highest average performance on sequential editing tasks, outperforming the latest dual-memory method WISE by \textbf{15–20\%},

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes DEPTRAI, an editing mechanism for large language models that can be understood as combining two prior lines of work: (1) GRACE-style external key–value memory with routing at inference time, where edits are stored outside the base model and selectively activated, and (2) MEMIT-style closed-form weight update analysis, where desired factual changes are framed as solving a regularized normal equation that trades off edited keys K₁ and preserved keys K₀ via an inverse-covariance term. DEPTRAI keeps the GRACE-like idea of storing edits as key–value pairs and retrieving them at inference, but replaces GRACE’s cosine/dot-product router with a Mahalanobis metric (reduced to dot product form) that is derived directly from the MEMIT/AlphaEdit closed-form mixing coefficients $\beta=K_1^\top C^{-1}k$, effectively interpreting $C^{-1}=K_0K_0^T+K_1K_1^T$ as a whitening transform and turning retrieval into *which stored key would MEMIT have blended into this FFN output?*

In experiments on LLaMA and Qwen models (3/8B, 3B) across sequential editing benchmarks like ZsRE and hallucination correction, DEPTRAI maintains high reliability and locality over long edit sequences (up to 1,000 edits) and reports 15–25% higher average performance than WISE at depth, while noting remaining limitations such as weaker generalization across synonyms and some residual locality interference.

### Strengths
- DEPTRAI elegantly combines the strength between two lines of knowledge editing works
- Good lifelong editing performance on ZsRE and SelfCheckGPT with $\leq$ 1k timesteps

### Weaknesses
- Missing some more recent lifelong editing baselines such as sLKE [1], LeMOE [2], and ELDER [3].
- Evaluation is not sufficient regarding
  - The finetuning baseline should adopt the fair setups as discussed in [4,5]. The FT-L, FT-M are ill-defined baselines which might mislead the community.
  - Lack layer-wise ablations as the baselines and DEPTRAI choose different layers for editing.
  - The scaling of timestep is only to 1k. More timesteps can be shown, e.g., up to 5k.
- The contribution is a bit limited as the novelty is mainly the metric for key similarity.
- Writing quality can be improved. For example, Figure 1 does not explicitly show the fundamental difference between DEPTRAI and existing approaches such as GRACE and WISE.


> [1] Cheng, YuJu, et al. "Serial lifelong editing via mixture of knowledge experts." Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2025.\
> [2] Wang, Renzhi, and Piji Li. "LEMoE: Advanced Mixture of Experts Adaptor for Lifelong Model Editing of Large Language Models." Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024.\
> [3] Li, Jiaang, et al. "ELDER: Enhancing Lifelong Model Editing with Mixture-of-LoRA." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 23. 2025.\
> [4] Gangadhar, Govind, and Karl Stratos. "Model editing by standard fine-tuning." arXiv preprint arXiv:2402.11078 (2024).\
> [5] Yang, Wanli, et al. "Fine-tuning Done Right in Model Editing." arXiv preprint arXiv:2509.22072 (2025).

### Questions
1. Can you provide experiments results with the additional baselines?
2. Can you perform ablation studies on layer selection and may be other aspects to further analyze the proposed approach?
3. Can you add a section to discuss the fundamental similarity and difference between MoE adapters/LoRA and codebook-style editing?

### Soundness
3

### Presentation
2

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
This paper proposes DEPTRAI, a novel knowledge editing framework for Large Language Models, aiming to address the issues of accumulating interference and irreversibility in existing "internal editors"  during sequential editing. The core idea of DEPTRAI is to keep the base LLM weights completely frozen, storing each new fact as a key-value pair in an external memory. Its key innovation lies in a principled router based on Mahalanobis distance, which determines during inference whether to replace the model's internal projection with the external stored value.

### Strengths
**Strength 1**: Instead of using a trainable router or simple cosine similarity, it ingeniously derives the external routing rule (Mahalanobis distance) from the mathematical principles of internal editing methods. This internal-editing-inspired external routing is a highly novel and principled design.

### Weaknesses
**Weakness 1**: The paper claims in the contribution and ablation studies that the Mahalanobis distance router is "robust to surface form variations" and outperforms cosine similarity. However, in the conclusion (Section 6), the authors list "stored keys may not generalize well to synonyms or transliterations" as a current limitation, which appears inconsistent with the earlier claim of robustness to surface variations.

**Weakness 2**: The theoretical derivation in Section 3.1 (from the $\beta$ coefficient to the Mahalanobis distance) requires $\Lambda$ to be a global covariance matrix ($C^{-1}$), in order to compare all keys in the same "whitened" space. However, the implementation part in Section 3.2 ("Memory structure") implies $\Lambda$ is local, defining the external storage as $\mathcal{E}=\{(\mu_{j},\Lambda_{j},v_{j})\}_{j=1}^{M}$, where each entry $j$ has its own $\Lambda_j$. This creates a theoretical contradiction.

More critically, the formula in Section 3.2 (Eq. 14)  used to calculate this local $\Lambda_j$ appears to be a critical typo. The formula first defines $\mu_j = k_j$, and then immediately uses the term $(k_j - \mu_j)$ to calculate $\Sigma_j$. This necessarily results in the term being a zero vector, causing $\Sigma_j$ to degenerate into $\epsilon I$. This would cause the Mahalanobis distance to degenerate into a (scaled) Euclidean distance, thereby completely undermining the paper's core argument about the superiority of Mahalanobis distance (relative to Euclidean or cosine similarity).

If $\Lambda$ is global, the authors must clarify how $K_0$ is collected and how $C^{-1}$ is efficiently updated during sequential editing (since $C$ depends on $K_1$).

### Questions
See weaknesses

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
3

### Summary
DEPTRAI (Detachable External-memory layer for Parameter-Transformer Injection) is a method for integrating new facts in model, essentially a combination of ideas from:
(a) ROME/MEMIT/AlphaEdit: W_out matrices in FFN layers are associative memories (key/value stores) that can be perturbed/updated (Delta) to account for edits (with keys/values for subject/its edit).
(b) GRACE: updates are triggered only after a threshold condition is met (deferral mechanism).

However: DEPTRAI does not inject Delta updates in model parameters (as in (a)) and does not finetune the values (as in the separately kept codebook in (b)). So the model is not touched and external memory updates do not trigger backpropagation (in that aspect DEPTRAI has strong similarity to the external scope detector idea in [1] or as originally in SERAC). Extensive benchmarking focusing on the sequential editing task (multiple methods/datasets/metrics) identifies the strengths of the proposed approach.

[1] Das, P., Chaudhury, S., Nelson, E., Melnyk, I., Swaminathan, S., Dai, S., Lozano, A., Kollias, G., Chenthamarakshan, V., Navratil, J. and Dan, S., 2024, July. Larimar: Large Language Models with Episodic Memory Control. In International Conference on Machine Learning (pp. 10109-10126). PMLR.

### Strengths
- Comprehensive empirical results: multiple methods and datasets are benchmarked; multiple metrics are reported.

- This is an interesting and simple synthesis of core ideas from model editing literature (however editing an external memory instead).

### Weaknesses
- Not touching model parameters (train-free) and still being able to adapt it to new facts is a very appealing idea, but intuitively this should have limitations. There are some hints in the Conclusion (Lines 421-423) but the reader would definitely appreciate more details on this. 

- Presentation can be improved: in particular some results could move to the Appendix, key notions could then be developed further and notation or equations could be revisited for corrections. Please see the Questions slot for details/suggestions.

- There is not a clean signal regarding the superiority of DEPTRAI: for example empirical results in Table 4 (Appendix D) are not as encouraging as results in Tables 1 and 2 (main text).

### Questions
- Is Figure 3 a plot of Rel. columns from Table 1?

- Line 234: mu_j = k_j? then Sigma_j's would only be epsilon I?

- Line 262 / Equation (22): Could you clarify the notation/symbol immediately following = sign?

- Lines 278-279: FT-L and FT-M seem to refer respectively to ROME and MEMIT? Is so why using additional alternative names?

- Lines 264-268: Is there an intuition behind the interesting separation in Figure 2 for some of the models?

- Lines 302-306: Since this part is deferred to the the Appendinx, a better use of this space would be to explain/clarify further the key scores in this work: reliability, generalization and locality (and expanding Eq (23)).


- For serializable external memory:
  - what is the cost of computing Sigma_j's, inverting them (Lambda_j's) and storing?
  - how relatively important can these additional space/time complexities be, assuming typical target factual edit cardinalities M?


- How would results from MEMIT compare? Assuming that instead of making sequential updates up to T items (i.e. one-by-one for T=1, 10, 100, 1000) as in the manuscript, we added all T items in one shot (or even updating in item batches of > 1 items), would we expect to see benefits (i.e. sequential vs batch updates)?

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
3

### Summary
The paper proposes **DEPTRAI**, a detachable external-memory layer for knowledge editing that leaves base LLM weights untouched. Each factual edit is stored as a key–value pair outside the model; at inference, the frozen FFN emits a subject key that is routed to the closest stored key via a Mahalanobis metric derived from the closed-form mixing coefficient used by ROME/MEMIT, and a lightweight gate injects the edited value or preserves the base projection. This reframes editing as reversible, database-style updates rather than cumulative in-place perturbations, mitigating interference and easing revocation. Across LLaMA-3.2-3B, Qwen-2.5-3B, and LLaMA-3.1-8B on ZsRE sequential edits and a hallucination correction setup, DEPTRAI sustains high reliability and near-perfect locality at large edit depths, outperforming recent dual-memory baselines.

### Strengths
The core idea—moving edits out of the base weights into a detachable key–value memory with a principled router—is intuitive yet original. By storing a single subject key and edited value per fact and routing with a Mahalanobis metric derived from the closed-form mixing coefficient used in ROME/MEMIT, the method turns knowledge editing into a reversible, database-style retrieval-and-injection step, avoiding cumulative interference and simplifying revocation/audit.   Methodological quality is strong: the paper motivates the Mahalanobis router from the mixing-coefficient analysis and adds an explicit gate to balance old vs. new information at inference, giving a clear mechanism for locality and reliability. Clarity is good, with an explicit contrast to in-place editors and a step-by-step description of the external layer; the framing “from in-place perturbation to detachable memory” makes the contribution easy to grasp and to implement in existing pipelines.

### Weaknesses
The evaluation is concentrated on editing-specific suites and a small “general capability” check that is largely short-form classification/MC (SST, MRPC, RTE, CoLA) plus MMLU. This leaves open whether edits preserve or disrupt performance on harder, generative reasoning and coding tasks. Concretely, the paper does not report effects on contemporary math/coding benchmarks or long-form QA after large edit batches. To strengthen external validity, please (i) add rigorous pre/post-edit results on **AIME’24/’25** and **MATH500** (ii) include **LiveCodeBench** to assess code reliability under heavy edits; (iii) use an instruction-following suite such as **IFEval** to probe instruction adherence; and (iv) include a simple open-domain QA set (e.g., **SimpleQA**) to check whether retrieval-style edits bias factual QA.

### Questions
See more details in weakness.

### Soundness
2

### Presentation
2

### Contribution
2
