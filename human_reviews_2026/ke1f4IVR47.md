# Stable Preference Optimization via Two-sided Contrastive Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Offline preference optimization has proven effective for aligning large language models (LLMs). However, existing methods often suffer from objective misalignment, which drives models toward yielding degenerate language patterns (i.e. nonsensical tokens and incoherent phrases) with moderately extended fine-tuning. In this paper, we propose Stable Preference Optimization (StaPO), a novel method designed to address this challenge. We first unify existing offline preference optimization approaches under a one-sided contrastive (OsC) learning framework, showing that OsC inherently maximizes the contrastive logit—the average or summed log-probability difference between preferred and dispreferred responses—without proper constraints. This unconstrained maximization of the contrastive logit, can gradually erode the LLM's core linguistic functionality. StaPO mitigates this via a two-sided contrastive (TsC) learning framework with dual-margin constraints. The left margin, akin to the OsC-based methods, ensures effective preference learning, while the right margin limits excessive growth of the contrastive logit, thereby preventing the collapse of the well-trained language system.  Empirical evaluations conducted on standard benchmarks, such as AlpacaEval2, Arena-Hard, and MT-Bench, highlight significant improvements achieved by StaPO compared to OsC-based methods. While StaPO consistently maintains stable win rates and entropy levels across multiple finetuning epochs, OsC-based methods show abnormally increasing or decreasing language entropy and deteriorating performance. These benefits of StaPO are consistently observed across diverse model architectures, including both base and instruction-tuned architectures like Mistral (7B) and Llama 3 (8B).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Stable Preference Optimization (StaPO), which introduces a two-sided contrastive objective to stabilize offline preference optimization. Instead of unboundedly enlarging the logit gap between preferred and dispreferred responses (as in DPO/SimPO), StaPO constrains the gap within a bounded interval using left and right margins. Experiments on UltraFeedback with Mistral-7B and Llama-3-8B show that StaPO yields more stable training dynamics and consistent improvements on AlpacaEval 2.0, Arena-Hard, and MT-Bench.

### Strengths
1. The proposed idea is simple yet effective. Adding a right-margin constraint to prevent over-optimization is intuitive and yields clear empirical benefits.

2. The writing is clear and well structured, which makes the technical idea easy to follow.

### Weaknesses
1. Based on my experience with DPO-style training, the probabilities of both chosen and rejected responses often decrease simultaneously, rather than the logit gap increasing monotonically as claimed. Including training-curve plots of chosen vs. rejected probabilities would make the argument more convincing.

2. StaPO requires tuning two margin hyperparameters (mₗ, mᵣ), adding complexity compared to SimPO or DPO.

3. In Table 3, the main results are reported at epoch 5, while DPO and SimPO typically overfit in later epochs. A fairer comparison would evaluate baselines at their best epoch (e.g., epoch 2–3). After several epochs, these methods often produce longer responses and show performance drops on knowledge-intensive QA tasks (e.g., OpenLLM leaderboard). Including such indicators would better demonstrate StaPO’s resistance to overfitting.

### Questions
Please see the “Weaknesses” section above.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper unifies existing offline preference optimization methods under a one-sided contrastive (OsC) learning framework and highlights that unconstrained maximization of the contrastive logit can gradually erode the LLM’s core linguistic capabilities. To address this, the authors propose StaPO, a two-sided contrastive (TsC) learning framework that both facilitates preference learning and constrains the excessive growth of contrastive logits.

### Strengths
- Provides an insightful unification of current offline preference optimization approaches.

- The two-sided contrastive design is conceptually elegant, and directly addresses a clear limitation of existing OsC-based methods.

- The paper is generally well-written, clearly organized, and easy to follow.

### Weaknesses
- Although StaPO claims reduced sensitivity, the margin values (m_l, m_r) still require tuning (or the gap); providing a principled criterion for selecting them would enhance reproducibility.

- The contribution could be strengthened by discussing StaPO’s behavior on out-of-domain or unseen preference distributions (i.e., its potential alignment tax).

- The lack of statistical significance analysis makes it difficult to assess the reliability of the reported improvements.

### Questions
1. What is the expression or definition of Z used in StaPO? Have you tested different Z variants listed in the table within the StaPO framework?
2. In Table 2, does this configuration generalize across all model families? Do you have any suggestions regarding the tuning of m_l?
3. How sensitive is the performance to the specific margin gap (m_r - m_l)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Existing offline preference optimization methods often cause LLMs to generate nonsensical language over extended fine-tuning. This paper argues this "objective misalignment" stems from the unconstrained maximization of the log-probability difference between preferred and dispreferred responses. It proposes Stable Preference Optimization (StaPO), a novel two-sided contrastive learning framework with dual-margin constraints. While a left margin ensures effective preference learning, a crucial right margin limits excessive logit growth, preventing the model’s linguistic collapse. On benchmarks like AlpacaEval2, StaPO demonstrates significant and stable performance improvements over existing methods, maintaining consistent win rates and avoiding performance degradation across models.

### Strengths
- Identifies a Critical Limitation: The paper explicitly defines "objective misalignment"—a core flaw in existing offline preference methods (e.g., DPO, SimPO) where extended fine-tuning leads to degenerate language (nonsensical tokens, incoherent phrases). This addresses a practical pain point often overlooked in prior work.

- Unifies Existing Methods Under a Theoretical Framework: It formalizes DPO, SimPO, CPO, and R-DPO into a one-sided contrastive (OsC) learning framework, showing that all these methods inherently maximize the "contrastive logit" (log-probability difference between preferred/dispreferred responses) without constraints. This unification simplifies understanding of why OsC methods erode LLMs’ linguistic capabilities over time.

- Innovative Two-Sided Contrastive (TsC) Design: The proposed StaPO introduces dual-margin constraints (left margin for preference alignment, right margin to limit excessive contrastive logits) to balance preference learning and linguistic coherence. The gradient analysis (Section 3.2) clearly explains how TsC stabilizes training—e.g., gradients adjust bidirectionally to keep logits within bounds, preventing deterministic collapse.

### Weaknesses
- While the paper mentions fixing the margin gap (mᵣ - mₗ = 0.2) works well across datasets, it provides no justification for why 0.2 is optimal (e.g., no ablation on gap sizes other than 0.2). The right margin’s impact on "linguistic coherence" is measured indirectly via entropy, but there is no qualitative analysis of how different margin values affect specific linguistic traits (e.g., fluency, factuality, or diversity in open-ended generation).
- The paper illustrates degenerate outputs with 2 examples (Figures 4–5: low-entropy repetitive text, high-entropy mixed-language nonsense), but it lacks a systematic categorization of degenerate patterns (e.g., token repetition, topic drift, logical inconsistency) or quantitative metrics for coherence (e.g., BLEU for fluency, F1 for factuality). It does not test whether StaPO avoids other common failures of OsC methods (e.g., "reward hacking"—overfitting to preference data at the cost of generalizability).
- The paper focuses on comparing StaPO to OsC methods (DPO, SimPO, etc.) but largely ignores non-contrastive offline methods like IPO (Azar et al., 2024) or RRHF (Yuan et al., 2023) in depth.

### Questions
see weakness

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
4

### Summary
The submission proposes a partial unification of existing preference optimization work into a one-sided contrastive framework and further propose a two-sided contrastive loss that is more fit for language models. Empirical validation is done using two model families and 3 evaluation datasets.

### Strengths
- Empirical validation does seem favorable compared to existing works
- The idea of a two-sided contrastive loss with double margins is straightforward

### Weaknesses
- The method lacks theoretical grounding. The only non-empirical evidence provided by the authors is the following quote"However, this discriminative principle conflicts with generative
language modeling, which requires maintaining balanced probability distributions across multiple
plausible outputs to ensure linguistic coherence and diversity."
However, this quote has no references, or other more in-depth justification in the given manuscript. Limiting deviation from the original model was exactly what the KL divergence term in the DPo was for, and yet the paper fails to explain why this doesn't work in practice. The given solution is a somewhat hard clip of the contrastive logit value, which seems to lack theoretical justification, albeit effective empirically

### Questions
- Does the DPO loss also degenerate like SimPO despite the KL divergence term?

### Soundness
3

### Presentation
2

### Contribution
2
