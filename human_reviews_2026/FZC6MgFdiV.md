# Knowledge-Sensitive Dynamic Module Editing: Precise Knowledge Revision for Multimodal Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Multimodal Large Language Models (MLLMs) struggle with efficient knowledge updates because their internal representations distribute information across lengthy and heterogeneous visual-textual sequences. This distribution makes traditional "locate-then-edit" methods, despite being highly effective in text-only models, largely ineffective for MLLMs. The resulting challenges include inaccurate localization of knowledge, poor generalization of edits, and unintended damage to unrelated knowledge. To bridge this gap, we introduce KDKE, a novel Knowledge-sensitive Dynamic multimodal Knowledge Editing framework tailored for MLLMs. KDKE introduces an Integrated Module Contribution Score to precisely quantify the impact of different modules on specific knowledge outputs. This enables a dynamic module selection mechanism that identifies critical parameters for each edit instance adaptively. We further develop a constrained adaptive editing algorithm, which injects LoRA parameters into selected modules and optimizes them under multi-objective constraints to ensure reliable editing, robust generalization, and strict locality. Extensive experiments on multiple model architectures and benchmarks demonstrate that KDKE superior editing accuracy and consistently strong overall performance, providing an effective and reliable solution for knowledge editing in multimodal settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces the Integrated Module Contribution Score to quantify each module's contribution to the target knowledge, enabling precise identification of modules that require editing.

### Strengths
The paper defines the Integrated Module Contribution Score and demonstrates that it outperforms existing methods such as VisEdit and LTE on MMEdit across multiple architectures.

### Weaknesses
1. Lack comparison with other dynamic, layer-selective editing methods (e.g., WilKE).
2. It's easier to achieve high accuracy comparatively because of MMEdit's dataset design (generalization samples share same answers with the original question, and there is domain gap between locality samples and edited samples). It's needed to validate on more challenging datasets.
3. Further validation is required in lifelong editing scenarios.

### Questions
1. It's better to include comparisons of runtime and memory to identify the requirements of the module selection.
2. It's better to include the recent backbones, such as Qwen-VL 2.5 [1].
3. Complete and verify related-work citations (e.g., ICE, GRACE, MeLLo), and clarify whether GRACE belongs to context-based editing methods.
4. Perform a thorough check of symbols and grammar (e.g., quotation marks and notation).

[1] Qwen2.5-vl technical report.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents KDKE, a knowledge-sensitive dynamic knowledge editing framework for multimodal large language models (MLLMs). By introducing an integrated module contribution score to quantify each module’s impact on output tokens, KDKE enables instance-wise dynamic module selection. It further employs LoRA-based constrained adaptive editing and a multi-objective loss to balance reliability, generalization, and locality. Experiments on BLIP2-OPT, LLaVA-V1.5, and MiniGPT-4 show that KDKE achieves superior performance over existing MLLM and LLM knowledge editing baselines.

### Strengths
1. Clear research motivation. The paper precisely identifies the key challenges in editing MLLMs—namely, the distributed and entangled nature of multimodal knowledge and the limitations of directly transferring text-only editing paradigms.
2. Comprehensive evaluation. The proposed framework is validated across multiple popular MLLMs, demonstrating broad applicability.
3. Thorough ablation studies. Each component of the proposed method is systematically analyzed through detailed ablation experiments.

### Weaknesses
1. Potential confusion between knowledge storage and retrieval. The key innovation, integrated module contribution score, quantifies each module’s impact on output probabilities to identify those most influential for editing. However, this may conflate *where knowledge is retrieved* with *where it is stored*. As shown in [1], such methods typically locate retrieval sites (often in the final layers) rather than true storage sites (commonly in earlier layers). By selecting components based on the last token’s contribution and applying LoRA editing, the method effectively modifies retrieval modules, potentially introducing conceptual confusion.

2. Unclear methodological description. Sections 3.1 and 3.2 lack clarity and contain multiple inconsistencies. For example, discrepancies exist between Equations (6) and (9), and the notation $c_T^p(Z)$ in line 222 is inconsistent with the rest of the text. Important symbols such as $t$ and $m$ in Equation (11) are undefined, and the variable $z$ is ambiguously used—sometimes as a token representation, other times as a module (line 259). The meaning of $\tau_{ratio}$ (line 263) is also unclear. Furthermore, Equation (11) itself is under-specified: $C_{[t]}^P(z)$ should represent a distribution, but it is unclear what raising a distribution to the $m$-th power signifies.

3. Writing and formatting issues. The paper exhibits several stylistic problems, including missing punctuation after equations throughout. It is recommended to reorganize all symbolic representations and complete methodological expressions.

4. Lack of statistical significance in results. None of the reported experiments include multiple trials or 95% confidence intervals. Statistical validation is essential for knowledge-editing tasks, as shown in [1].

5. Limited experimental exploration. While the method is tested on several models, deeper investigations are missing—for example, whether continuous edits on the same model degrade performance, how efficiency compares with other methods, and how editing quality changes under continuous edits.

[1] Locating and Editing Factual Associations in GPT.

### Questions
1. Is there a clear difference between how textual and visual knowledge is stored in MLLMs? Does the paper provide any deeper insight into this distinction? Understanding the modality-specific storage mechanisms is crucial, as they directly influence the most suitable strategy for knowledge editing.

2. In Dynamic Module Selection Mechanism of Section 3.2, which hyperparameters must be predefined, and how are their specific values determined in practice? Moreover, how is the rationality of these chosen values justified?

3. How stable are the editing results—does continuous editing on the same model lead to degradation in performance? Additionally, how efficient is the proposed method compared to existing approaches?

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
4

### Summary
The paper addresses the challenge of knowledge editing in Multimodal Large Language Models (MLLMs), where traditional "locate-then-edit" methods from text-only LLMs are ineffective due to the distributed nature of multimodal knowledge. The authors propose KDKE, a Knowledge-sensitive Dynamic multimodal Knowledge Editing framework.

### Strengths
1.  The paper proposes the Integrated Module Contribution Score (IMCS), an attribution method designed for locating knowledge in MLLMs. It is based on the Transformer's residual stream linearity and is computed in a single forward pass, offering a potential alternative to more computationally intensive localization methods.
2.  The framework introduces a "Dynamic Module Selection" mechanism. This approach identifies knowledge-related modules on a per-instance basis, moving beyond static selection strategies. The experimental results suggest this dynamic approach is more effective than fixed-layer editing.

### Weaknesses
* **Ambiguity in Multi-Token Target Editing:** The paper's core metric, IMCS $C_{[t]}(z)$, is defined for a single target token $t$. This is clear for VQA tasks, but for sequence-output tasks like image captioning (E-IC), the target is a sequence. The paper fails to specify how the target token $t$ is chosen or how IMCS is aggregated in this multi-token scenario, creating a methodological ambiguity that harms reproducibility.
* **Clarity on Computational Cost:** The paper claims the IMCS calculation is "lightweight" and a "single inference step." However, it requires computing the $zW_V$ projection for all $2L$ modules, which appears significantly more expensive than a standard forward pass. A transparent analysis of this overhead (e.g., FLOPs or wall-clock time) compared to standard inference or causal tracing is missing.
* **Sensitivity of Dynamic Selection Hyperparameters:** The dynamic module selection algorithm introduces new hyperparameters ($\alpha_{gap}$, $\tau_{ratio}$) to control its thresholds. The paper provides no sensitivity analysis for these parameters, leaving it unclear how their values affect the number of selected modules and the final editing performance.
* **Under-specified Retrieval for $\mathcal{D}_{sim}$:** The locality loss relies on a set of "similar samples" $\mathcal{D}_{sim}$ from semantic retrieval. This retrieval step is a non-trivial part of the method but is not detailed. The mechanism, quality, and quantity of these retrieved samples could significantly impact locality, making this an under-specified component.

### Questions
1.  (Ref. Weakness 1) How is the IMCS computed for multi-token targets, such as in the E-IC dataset? Is it based on the first token, or an aggregation (e.g., mean, max, union) across the entire target sequence?
2.  (Ref. Weakness 2) Could you provide a concrete analysis of the computational overhead (e.g., FLOPs or wall-clock time) of the IMCS calculation, comparing it to both a standard forward pass and a full causal tracing run?
3.  (Ref. Weakness 3) What is the sensitivity of the model's performance and the number of selected modules to the dynamic selection hyperparameters ($\alpha_{gap}$, $\tau_{ratio}$)?
4.  (Ref. Weakness 4) Could you please detail the retrieval mechanism for the $\mathcal{D}_{sim}$ samples used in the locality loss (e.g., model, similarity metric, number of samples)?
5.  What was the motivation for freezing the LoRA 'A' matrix and only training 'B'? Did this random subspace projection prove sufficient for all edit types?

### Soundness
3

### Presentation
3

### Contribution
2
