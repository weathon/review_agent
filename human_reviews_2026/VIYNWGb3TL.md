# DistillMoE: Multi-Faceted Knowledge Distillation for Cross-Tokenizer Embedding Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
Cross-Tokenizer Knowledge Distillation for Large Language Models (LLMs), embedding models present significant challenges, primarily due to tokenizer mismatches and limitations of traditional distillation frameworks in capturing the diverse semantic signals encoded by the teacher. We propose DistillMoE, a framework that addresses these challenges through a dual-level strategy. At the sequence level, DistillMoE employs a lightweight Mixture-of-Experts module to distill sentence representations, where each expert specializes in a distinct semantic perspective: pointwise, contrastive, and pairwise. A trainable router assigns inputs to experts, letting each objective be optimized separately, thus enabling seamless integration of diverse losses without heavy tuning. At the token level, we introduce DynamicCKA to align teacher–student hidden states for fine-grained knowledge transfer. This refinement yields teacher-aware sentence embeddings, enabling the MoE to assign more informative expert weightings and enhance multi-faceted distillation. Empirically, when distilling state-of-the-art text embedding models (e.g., LLM2Vec, BGE-M3, Qwen3) into a compact BERT base student, DistillMoE consistently outperforms prior CTKD baselines across multiple datasets. These results demonstrate the effectiveness of combining multi-perspective sequence-level distillation with token-level alignment to obtain compact yet high-fidelity embedding models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces DistillMoE for Cross-Tokenizer Knowledge Distillation (CTKD) specifically designed for distilling large embedding models (like LLM2Vec, BGE-M3, Qwen3) into smaller students like BERT-base. In order to address the challenge of transferring knowledge between models with different tokenizers and vocabularies, DistillMoE proposes a dual-level solution. At the sequence level, a lightweight Mixture-of-Experts (MoE) module is designed to capture distinct semantic facets of the teacher's knowledge: pointwise alignment (cosine loss), contrastive alignment (InfoNCE loss), and pairwise relation preservation (ranking loss); at the token level, a DynamicCKA module is introduced to align the teacher and student hidden states for the knowledge transfer. Extensive experiments on text classification, sentence pair classification, and semantic textual similarity (STS) tasks demonstrate that DistillMoE consistently outperforms existing baselines.

### Strengths
* The proposed dual-level design is new, as well as the idea of using MoE for multi-faceted knowledge transfer
* The empirical evaluation is extensive, and the performance gains are substantial and consistent across a wide range of tasks and teacher models
* The paper is well-structured and clearly written

### Weaknesses
* The computational overhead during training is reported in Appendix D (Table 7). It would be appreciated if the authors could provide more discussion on optimizing DynamicCKA
* Although MoE dynamically weights different losses, the overall framework still introduces new hyperparameters alpha and lambda. It would be helpful if the authors could provide more experimental results of different values

### Questions
1. In practice, how often does the gating network produce a near-one-hot allocation vs. a more balanced soft distribution?
2. As the DynamicCKA module adds significant training cost, have you explored any simpler or more efficient token-level alignment baselines?
3. Apart from different teacher models, how does DistillMoE scale when the student model is even smaller?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes DistillMoE, which combines a token-level alignment loss (called DynamicCKA) with sequence-level auxiliary losses (implemented as three different "experts"), which are weighted via a MoE-style router. The authors show this alleviates some problems with manually set weights as the weighting is learned via the router. Empirically, the authors show their method outperforms various baselines on text classification, sentence pair classification and semantic textual similarity tasks when distilling a 7B LLM2Vec model into a 110M Bert-base student.

### Strengths
S1: The paper proposes a new token alignment method for cross-tokenizer distillation (DynamicCKA) 

S2: The paper proposes to use a MoE-style setup to dynamically combine different auxiliary distillation losses without manually specifying their weights

### Weaknesses
W1: the proposed method is only evaluated using a single, relatively small bert-base model.
  - does the proposed method also work for causal language models?
  - do you have an intuition about the scaling behavior of the proposed method to larger models?
  - What effect do different degrees of tokenizer difference have?

Also see raised questions. The low score is primarily due to open questions regarding hyper-parameter tuning, missing baselines, limited evaluation and some methodology details which are not clear to me. I will raise the score if these are appropriately addressed.

### Questions
- Q1: how does the proposed method compare to the Approximate Likelihood Matching proposed by Minixhofer et al., 2025 (https://arxiv.org/abs/2503.20083)? This seems like a highly relevant baseline method and related work.
- Q2: l. 252ff: I'm not sure I understood how the $\alpha_{p,q}$ is constructed. Here's my understanding: We're normalizing the cosine sim of each student and teacher token by the other cosine similarities of that student token with all other teacher tokens. Then we pick - from **ALL teacher tokens in the entire sequence** - those that have the highest cosine similarities with that student token and construct the alignment that way. Is this correct? 
  - Do you have qualitative analysis of these alignments? A priori, there seems to be a large potential for noise in the proposed token alignment mechanism.
  - How is the projection Q trained?

- Q3: l. 133ff (Section 3.1.2) 
  - how are the projection matrices W for Expert 1 and Expert 2 trained?
  - how is the sentence embedding for the Bert base student calculated (mean / special token / something else?)

- Q4: "hyperparameter tuning" -- the $\alpha$ and $\lambda$ hyper parameters are tuned for each target dataset. The optimal values show a large variability per setting (particularly $\alpha$).
  - were hyper-parameters for the baselines tuned similarly?
  - which data splits were used for hyper-parameter tuning?

- Q5: some of the results in Table 2 are quite close. Could you show error bars over multiple runs to quantify the variance in the evaluated methods?

suggestion: the ablations in Table 12 are very important in my opinion, this and a discussion of these should be moved to the main body if possible given an extra page. Your work proposes a whole stack of new methods and it is very important to discern which parts of the proposed full method are most important.

### Soundness
2

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
This paper presents DistillMoE, a distillation strategy with two components. At the sequence level, the authors propose a a mixture of three experts, which are distilled with a cosine, contrastive, and pariwise loss to capture different aspects of teacher knowledge. For tokens, the authors propose DynamicCKA to address tokenizer mismatch.

### Strengths
1. The author proposes quite a few techniques.
2. Authors show consistent improvements over baselines.

### Weaknesses
1. The selection of tasks is a bit unusual. It would be nice to see how this method works with instruction tuning and reasoning distillation.
2. The paper attempts to address two separate problems. One being knowledge transfer loss and the other being tokenizer mismatch. As a result, I feel the authors don't study either very in depth. For example, there are existing methods that just focus on the tokenizer mismatch problem, and the paper can benefit from comparing with some baseline approaches.

### Questions
For comparisons in Table 1 (for example), does DistillMoE use extra parameters because of the MoE structure vs other competing methods?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces DistillMoE, specifically designed to solve the "cross-tokenizer" problem. This problem arises when a small "student" model needs to learn from a large "teacher" model, but the two models use different vocabularies (tokenizers), making direct knowledge transfer difficult. To handle the tokenizer mismatch, the paper proposes DynamicCKA. This method aligns the hidden states of the teacher and student models. At the sentence level, a lightweight MoE module learns the teacher’s knowledge: pointwise for semantics, contrastive for geometry, and pairwise for sentence relations.

### Strengths
S1: The novelty lies in repurposing the MoE architecture—not just for capacity or efficiency, but as a tool for knowledge transfer. By assigning different semantic objectives (pointwise, contrastive, pairwise) to separate experts, the authors create a structured framework that captures diverse aspects of the teacher model’s representations.

S2: The authors provide theoretical justification for the behavior of the gating mechanism, lending credibility to their architectural choices. The authors evaluate their method on a set of tasks (STS, text classification, sentence-pair classification), demonstrating the general applicability of their framework. They also compare against a strong and comprehensive suite of recent state-of-the-art CTKD baselines.

### Weaknesses
W1: The most significant weakness of DistillMoE is the substantial computational overhead introduced during training, which is not sufficiently justified in the context of creating an efficient student model. Moreover, the authors should discussed it in the main paper.

W2: The paper's choice of the three expert objectives (pointwise, contrastive, pairwise) is intuitively appealing but lacks a strong theoretical foundation. It feels more like a well-engineered recipe than a principled decomposition of knowledge. 

W3: The experiments, while comprehensive on standard benchmarks, are confined to an in-domain setting and fail to measure two critical aspects of a distilled model's quality: generalization to unseen domains and actual inference efficiency. All models are trained and evaluated on splits of the same datasets (e.g., trained on SciTail train set, tested on SciTail test set). The paper claims to create a powerful, general-purpose embedding model, but provides no evidence of its out-of-domain (OOD) generalization capabilities. Furthermore, no inference speed or latency metrics are reported.

### Questions
Q1: The paper provides a good description of what each expert does, but not why these three perspectives are the necessary and sufficient set for capturing the teacher's knowledge.

### Soundness
3

### Presentation
3

### Contribution
2
