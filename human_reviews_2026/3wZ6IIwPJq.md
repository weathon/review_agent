# Prompt-Robust Vision-Language Models via Meta-Finetuning

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Vision-language models (VLMs) have demonstrated remarkable generalization across diverse tasks by leveraging large-scale image-text pretraining. However, their performance is notoriously unstable under variations in natural language prompts, posing a considerable challenge for reliable real-world deployment. To address this prompt sensitivity, we propose Promise, a meta-learning framework for prompt-Robust vision-language models via meta-finetuning, which explicitly learns to generalize across diverse prompt formulations. Our method operates in a dual-loop meta-finetuning setting: the inner loop adapts token embeddings based on a set of varied prompts, while the outer loop optimizes for generalization on unseen prompt variants. To further improve robustness, we introduce an adaptive prompt weighting mechanism that dynamically emphasizes more generalizable prompts and a token-specific learning rate module that fine-tunes individual prompt tokens based on contextual importance. We further establish that Promise’s weighted and preconditioned inner update provably (i) yields a one-step decrease of the outer empirical risk together with a contraction of across-prompt sensitivity, and (ii) tightens a data-dependent generalization bound evaluated at the post-inner initialization. Across 15 benchmarks spanning base-to-novel generalization, cross-dataset transfer, and domain shift, our approach consistently reduces prompt sensitivity and improves performance stability over existing prompt learning methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a meta learning framework Promise, which explicitly improves the robustness of VLM to natural language prompt changes through inner and outer dual loop fine-tuning. This method introduces adaptive prompt weights and token level learning rates, significantly reducing prompt sensitivity and stabilizing performance on 15 benchmarks. Theoretical proof of the reliability of its inner loop and tightening of the data dependency generalization boundary.

### Strengths
1. Explore the robustness of VLM to prompt and analyze the shortcomings of existing methods, which rely on prompt templates in finetuning and inference processes.

2. Propose the meta learning framework Promise, which includes two parts: inner and outer loops, to help VLM models adapt to different prompt templates.

3. This paper verifies the reliability of the Promise framework from both experimental and theoretical perspectives.

### Weaknesses
1. The main body of the Promise framework is inner and outer loops, but the motivation behind this design is not clear. Further explanation (or explicit display) is needed on which issues correspond to the inner and outer loop design in the process of VLM adapting to different prompt templates.

2. The improvement brought by the Promise framework is limited, with most indicators improving by 1 to 2 points, and some indicators not exceeding state-of-the-art model accuracy.

3. Further research is needed on ablation of inner and outer circulation, such as (1) comparing the accuracy of the original model with the accuracy after removing adaptive prompt weighting and adaptive learning rate; (2) Simplify the outer loop to the overall accuracy of Gaussian distributed random numbers.

### Questions
See [Weaknesses].

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
4

### Summary
This paper introduces PROMISE, a meta-learning framework designed to improve prompt robustness in VLMs such as CLIP. 
PROMISE explicitly learns to generalize across diverse prompt formulations through a dual-loop meta-finetuning strategy. 
The framework integrates two novel components: adaptive prompt weighting, which emphasizes more generalizable templates; and
token-specific adaptive learning rates, providing fine-grained control over token adaptation.
The authors provide theoretical guarantees that the weighted, preconditioned inner update decreases the outer empirical risk and contracts across-prompt sensitivity. 
Experiments on 15 benchmarks show consistent improvements over state-of-the-art prompt tuning methods.

### Strengths
1. The dual-loop meta-finetuning framework is well-motivated and effectively bridges meta-learning and prompt optimization.
2. Evaluations covering base-to-novel, cross-dataset, and domain generalization tasks demonstrate consistent performance improvements and robustness.

### Weaknesses
1. Limited analysis of computational cost: while the dual-loop setup is theoretically appealing, its efficiency relative to single-loop baselines could be quantified more precisely. Appendix D mentions training time, but more systematic comparisons would help.
2. Prompt distribution diversity: The paper uses 60 pre-defined prompt templates from PromptSRC. It would be valuable to analyze whether PROMISE remains effective under more syntactically or semantically diverse prompt distributions.

### Questions
1. How sensitive is PROMISE to the choice of prompt template pool size (e.g., 30/60/80 templates)?
2. Which template is used in table 4, CLIP template or LLM template？

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
4

### Summary
This paper presents PROMISE, a meta-finetuning approach that aims to enhance the prompt robustness of vision-language models (VLMs), particularly CLIP-based ones. The work addresses the well-known issue that such models are often sensitive to small variations in natural language prompts. PROMISE introduces a dual-loop meta-learning strategy: the inner loop fine-tunes soft prompt embeddings using one set of prompt templates, while the outer loop updates meta-parameters based on a separate, unseen set. This setup is intended to simulate prompt distribution shifts and encourage cross-template consistency.

In addition, the framework includes adaptive prompt weighting, which learns to emphasize more generalizable templates, and token-specific adaptive learning rates, inspired by MetaSGD, to refine updates on individual prompt tokens. The paper provides a theoretical argument that the weighted, preconditioned inner updates can reduce empirical risk and improve consistency across prompts.

Experiments on fifteen benchmark datasets show modest but consistent gains over prior prompt-learning methods such as CoOp and CoCoOp, especially under unseen prompt templates. The results indicate improved stability rather than large absolute performance jumps. Overall, the paper is technically sound and carefully executed, though its contribution feels somewhat incremental given the maturity of CLIP-style prompt learning and the field’s current shift toward large multimodal foundation models.

### Strengths
The paper targets an identifiable and practically relevant problem — the sensitivity of CLIP-like vision-language models to prompt wording. The motivation is easy to follow, and the setup is consistent with prior work in prompt learning.

The proposed dual-loop meta-finetuning structure is coherent and well-explained. The integration of adaptive prompt weighting and token-level learning rates forms a complete framework rather than an isolated trick.

Evaluations cover 15 datasets, with results that are consistent and reproducible. The paper also explicitly tests under unseen prompt templates, which directly supports the stated goal of prompt robustness.

The writing is clear, diagrams are intuitive, and the ablation studies are reasonably thorough. It is easy to reproduce the method and interpret the findings.

Although the absolute accuracy improvement is limited, the model demonstrates stable behavior under prompt shifts, which may have practical value in applications relying on frozen CLIP backbones.

### Weaknesses
Conceptually incremental despite a solid formulation.
The meta-learning setup is reasonable, but the idea of using a dual-loop structure to optimize different objectives (inner for adaptation, outer for generalization) has already appeared in several prompt-tuning variants. Many recent works also explore meta-learning–based prompt optimization. As a result, the conceptual novelty of PROMISE is somewhat limited, even if the execution is careful.

Dependence on an outdated backbone.
All experiments are based on frozen CLIP encoders. While CLIP remains a convenient benchmark, it is no longer representative of the current generation of multimodal models. Evaluating PROMISE on stronger or trainable backbones (e.g., SigLIP, BLIP-2, or LLaVA) would make the contribution more convincing and show whether the method still holds under modern architectures.

Marginal quantitative improvement.
Although PROMISE reduces performance variance across prompts, its absolute accuracy gains are small (typically within 1–2%). Considering that the framework adds a meta-finetuning phase, the computational cost may outweigh the benefit in practical settings.

Ablation and analysis depth.
The paper does include component-wise ablations, but it remains unclear how each part (e.g., adaptive weighting, token-specific learning rate) contributes to robustness. The method could be further analyzed against simpler baselines, such as regularization-based approaches, to isolate where the gain truly comes from.

### Questions
Could the authors clarify how sensitive PROMISE is to the choice of the base prompt set used in the inner and outer loops? For example, would the performance change noticeably if both sets were constructed differently or partially overlapped?

How well does PROMISE generalize to newer or trainable backbones beyond CLIP (e.g., BLIP-2, SigLIP, or LLaVA)? Given that CLIP is becoming less central in recent multimodal research, a discussion or small-scale experiment on this point would help clarify the method’s broader applicability.

The paper claims improved prompt robustness, but do these benefits persist under other distribution shifts? In other words, is PROMISE’s robustness specific to language variation, or does it extend to multimodal shifts more generally?

Regarding the theoretical analysis, could the authors provide more intuition or empirical evidence that the “risk contraction” property holds beyond the simplified assumptions made in Section 3.3?

Have the authors considered or tested lighter-weight alternatives that might yield similar robustness with lower training cost? It would be helpful to understand the efficiency trade-off.

### Soundness
3

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
4

### Summary
This paper introduces PROMISE, a meta-learning framework designed for prompt-robust vision-language models through meta-finetuning. It achieves performance improvements via dual-loop fine-tuning and provides comprehensive experiments in both in-domain and cross-domain settings.

### Strengths
+ Decent results have been achieved in both in-domain and cross-domain experiments.

+ A dual-loop fine-tuning method has been proposed, achieving enhanced robustness and generalization.

### Weaknesses
+ The ablation experiments in this paper investigate the impact of adaptive prompt weighting but do not explore the effects of standalone Inner-Loop Finetuning or Outer-Loop Finetuning.

+ The current description in the paper gives the impression that the method merely introduces an adaptive prompt weighting mechanism based on MAML, and without this component, it seemingly reduces to the standard MAML approach.

+ In Table 6 of the paper, the best result in the "-S" category is actually 50.93% from FepMVP, not the 50.28% reported in this paper. A similar issue appears at line 450 of the paper, where the improvement on the FGVCAircraft dataset should be 2.79% for H rather than the reported 2.34%. There are numerous such issues in the paper, and it is recommended that the authors carefully review them. Additionally, all tables in the paper should highlight the second-best results in a different color, similar to the best results, and include a separate row explicitly stating the percentage improvement to avoid such errors and facilitate better observation. There are minor formatting issues with the tables; table captions should be placed above the tables rather than below.

+ While the proposed method achieves notable improvements on some datasets, the overall enhancement is limited. Moreover, the performance on certain datasets lags significantly behind prior methods, such as on EuroSAT, where it is comprehensively lower than previous approaches.

### Questions
Please refer to the weaknesses section for additional questions and queries!

### Soundness
3

### Presentation
3

### Contribution
3
