# HiCoLoRA: Addressing Context-Prompt Misalignment via Hierarchical Collaborative LoRA for Zero-Shot DST

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 2

## Abstract
Zero-shot Dialog State Tracking (zs-DST) is essential for enabling Task-Oriented Dialog Systems (TODs) to generalize to new domains without costly data annotation. A central challenge lies in the semantic misalignment between dynamic dialog contexts and static prompts, leading to inflexible cross-layer coordination, domain interference, and catastrophic forgetting. To tackle this, we propose Hierarchical Collaborative Low-Rank Adaptation (HiCoLoRA), a framework that enhances zero-shot slot inference through robust prompt alignment. It features a hierarchical LoRA architecture for dynamic layer-specific processing (combining lower-layer heuristic grouping and higher-layer full interaction), integrates Spectral Joint Domain-Slot Clustering to identify transferable associations (feeding an Adaptive Linear Fusion Mechanism), and employs Semantic-Enhanced SVD Initialization (SemSVD-Init) to preserve pre-trained knowledge. Experiments on multi-domain datasets MultiWOZ and SGD show that HiCoLoRA outperforms baselines, achieving SOTA in zs-DST. Code is available at \href{https://anonymous.4open.science/r/HiCoLoRA-96EB}{Anonymous Github}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes HiCoLoRA (Hierarchical Collaborative Low-Rank Adaptation), a framework for zero-shot DST (Dialog State Tracking) in task-oriented dialog systems. The framework uses: 1) UniRep-LoRA and SemAdapt-LoRA with Adaptive Linear Fusion to balance domain-agnostic and domain-specific features; 2) Spectral Joint Domain-Slot Clustering to identify transferable associations across domains/slots to guide the fusion mechanism; 3) Semantic-Enhanced SVD Initialization to pre-trained knowledge via singular value modulation. Experimental results on MultiWOZ and SGD show that HiCoLoRA outperforms previous SOTA method DualLoRA in the cross-domain zero shot DST setting.

### Strengths
- The hierarchical collaborative architecture is well-motivated and effectively addresses the cross-domain transfer limitations of existing LoRA-based approaches.
- The authors conducted extensive comparisons of the proposed method against multiple baselines across several categories (non-pre-trained methods, LLM-prompting, LoRA variants) and demonstrated the effectiveness of the proposal.
- Detailed ablation studies help the audience understand the impact of each proposed component within the HiCoLoRA framework.

### Weaknesses
- The proposed architecture is rather complex while achieving only marginal gains (~2% compared to DualLoRA in Table 1; ~1% compared to other LoRA-based methods in Table 3). Additionally, Spectral Joint Domain-Slot Clustering require dataset-specific tuning, which limits the scalability of the proposal.
- Some domains in SGD (RideSharing, Calendar, etc.) are excluded due to limited samples or atypical slot structures. This raises questions about the model's ability to generalize to long-tail domains that typically have less available data.

### Questions
See weakness.

### Soundness
3

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
4

### Summary
This paper proposes HiCoLoRA, a hierarchical collaboration–based LoRA architecture for zero-shot dialogue state tracking (zs-DST).
The method introduces: 1. Hierarchical LoRA design to improve context-prompt alignment; 2. Spectral clustering to disentangle domain-slot representations; 3. SemSVD-Init to preserve pre-trained knowledge during fine-tuning.

Experiments on MultiWOZ and SGD benchmarks show that HiCoLoRA achieves notable performance gains ( +5.4 % and +9.4 % JGA improvement over SOTA), validating each component through detailed ablation and case studies.

### Strengths
The paper is logically structured and easy to follow. The combination of hierarchical LoRA and spectral clustering is novel within the DST domain. The experiments are comprehensive, covering multiple datasets and model scales (from T5-small to 13B), and the ablation studies effectively support the main claims.

### Weaknesses
I think the writing style and methodology design are both good, but I still have a few concerns:

* The paper introduces several design components (hierarchical structure, adaptive fusion, spectral clustering, SemSVD-Init) and verifies their effectiveness through comprehensive ablations. However, it’s unclear why these specific components were designed and combined together. Could you elaborate on the underlying motivation or rationale behind this particular combination? Right now, the framework feels somewhat like a stack of independent modules rather than a unified design.

* The proposed method seems quite novel within DST. Do you plan to extend it to more general tasks in the future, such as instruction tuning or domain adaptation?

* You claim that SemSVD-Init helps preserve pre-trained knowledge. Could you explain this mechanism in more detail? Also, what is the motivation for preserving pre-trained knowledge in this context—why is it necessary for your method?

* It would be interesting to test HiCoLoRA on more challenging or diverse setups, such as cross-dataset evaluations or cross-domain transfer. This could better demonstrate its robustness and generalization ability.

* Missing references or baselines:

(1) Towards LLM-driven dialogue state tracking. EMNLP 2023

(2) Zero-shot Cross-domain Dialogue State Tracking via Context-aware Auto-prompting and Instruction-following Contrastive Decoding. EMNLP 2024

### Questions
* How does HiCoLoRA compare to other hierarchical or multi-level LoRA approaches outside DST (e.g., instruction tuning or domain adaptation)?

* Have you analyzed the inference-time computational overhead introduced by the multi-layer LoRA branches?

* Can SemSVD-Init be applied independently of the hierarchical architecture? If so, how does it perform on its own?

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
This paper proposes HiCoLoRA (Hierarchical Collaborative Low-Rank Adaptation), a novel framework for zero-shot Dialog State Tracking (zs-DST) in task-oriented dialog systems. The authors propose an advanced architecture to solve the task based on LoRA. Lower layers use heuristic grouping to encode local semantic features while higher layers use a full collaboration to model global semantic features and intent. Moreover, they propose a domain clustering technique to identify semantic relationships between domains and slot prompts. The method performs well over existing SOTA in MWoZ and SGD dataset.

### Strengths
- SOTA performance in a constrained (no pre-trained LLM) setting in both MWoZ and SGD.

### Weaknesses
The proposed architecture (including hierarchical structure, spectral clustering, adaptive fusion, and specialized initialization) introduces substantial complexity, making the model difficult to reproduce, implement, and maintain. However, simply prompting—without any fine-tuning—a large LLM (GPT4) achieves considerably higher performance on the same benchmark dataset [1], as shown in Table 4 of the appendix. Moreover, when starting from the same base model, the performance improvement obtained through the proposed method is marginal.

For instance:

FnCTOD (2024) – LLaMA2-13B: 62.2 / 46.8 / 60.9 / 67.5 / 60.3 → avg. 59.5
HiCoLoRA (2025) – LLaMA2-13B: 62.0 / 42.0 / 61.0 / 65.0 / 69.0 → avg. 60.0

This suggests that the added architectural complexity does not translate into meaningful empirical gains over simpler prompting approaches.

[1] https://arxiv.org/pdf/2402.10466

### Questions
Why not comparing the proposed method with a pretrain LLM? even smaller size.

### Soundness
3

### Presentation
3

### Contribution
1
