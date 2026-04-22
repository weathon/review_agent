# The Differences Between Direct Alignment Algorithms are a Blur

- Avg Score: 4.40
- Decision: Reject
- Scores: 2, 10, 2, 2, 6

## Abstract
Direct Alignment Algorithms (DAAs) simplify LLM alignment by directly optimizing policies, bypassing reward modeling and RL. While DAAs differ in their use of SFT (one-stage vs. two-stage) and the scalar score they optimize (likelihood vs. odds ratios), the key performance drivers remain underexplored. We present a systematic comparison and analyze a previously overlooked axis - the ranking objective (pairwise vs. pointwise). To isolate this factor, we propose a unified training framework across DAAs by (i) converting one-stage methods (ORPO, ASFT) into a two-stage pipeline with an explicit SFT phase and (ii) introducing a $\beta$ parameter that places all methods in the same hyperparameter space and improves the quality of odds-ratio DAAs (ORPO, ASFT). Under this setup, the ranking objective emerges as the primary determinant of alignment quality, whereas the particular scalar score (policy–reference ratio vs. odds ratio) is secondary. We corroborate this on instruction-following tasks and further confirm it on math-reasoning benchmarks across model scales. Evidence suggests that this stems from how these objectives interact with prompt-specific biases, supported both by strictly controlled experiments and by observations on real data. Our findings underscore the need for nuanced evaluations in DAA research to avoid oversimplified claims of superiority.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
- The paper proposes a **unified training protocol** for Direct Alignment Algorithms (DAAs): (i) convert one-stage odds-ratio methods (**ORPO**, **ASFT**) into a **two-stage** pipeline with an explicit SFT phase, and (ii) introduce a **β tempering parameter** so all methods share a comparable hyperparameter space. In this unified setting, the **ranking objective (pairwise vs. pointwise)**—rather than the scalar score (policy/reference ratio vs. odds ratio)—is argued to be the primary driver of alignment quality.
- Experiments span **Llama‑3.2‑3B (Reddit TL;DR)**, **Llama‑3.1‑8B (UltraChat/UltraFeedback → AlpacaEval‑2 LC & Arena‑Hard)**, and **Qwen‑2.5 (7B/14B) for math‑reasoning**. TL;DR is judged with **GPT‑4o**; UltraFeedback uses **AlpacaEval‑2 LC** and **Arena‑Hard**. 
- **β‑tuning** improves odds‑ratio methods.

### Strengths
- **Clean unification & fair comparisons.** Converting ORPO/ASFT to two‑stage and adding β places all DAAs in a shared search space, enabling apples‑to‑apples comparisons; the paper details the tempered formulation and common tuning. 
- **Transparent, controlled evaluation** across model families/scales and benchmarks. 
- **Actionable takeaway.** After unification and tuning, **pairwise vs. pointwise** explains most observed differences; scalar score choice (odds vs. reference‑ratio) is secondary.

### Weaknesses
1) **Missing / under‑engaged related work:**  
   - **AlphaPO: Reward Shape Matters for LLM Alignment** — introduces an **α‑controlled reward‑shape** family directly relevant to your tempering/unification axis.  
   - **Adaptive Reward Margin for Direct Preference Optimization** — proposes **instance‑adaptive reward margins**, closely related to temperature/margin control. 
   - **Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints** — demonstrates how **data/reward‑quality shaping** highlights the importance of reward shaping. 
   - **RainbowPO: A Unified Framework for Combining Improvements in Preference Optimization**. 
   - **Principled Foundations for Preference Optimization** — provides a **general theoretical framing**. 

2) **Scope of claims vs. analysis focus.** The paper claims to compare DAAs broadly, yet much of the **conceptual/derivational focus centers on ORPO/ASFT**—e.g., **“3.1 Generalizing ASFT and ORPO”** and **“Tempering ASFT and ORPO”**—within an odds‑based view. There is **limited treatment of explicit reward‑shaping/margin frameworks** and other variants of DAAs including the references mentioned above. Either broaden the discussion to cover these axes, or qualify the generality of the claim.

### Questions
1) How robust is the conclusion of “**ranking dominates**”? For example, if you incorporate **reward‑shape (AlphaPO)** or **adaptive margins (AlphaDPO)** or components from **RainbowPO** under your unified protocol, is the conclusion generally hold?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This work presents comparison and analysis of Direct Alignment Algorithms - a class of methods for aligning LLMs without and explicit reward model. The authors argue that current state of DAA literature is fragmented, which makes it difficult to isolate which design choices drive improvements in alignment quality. To address this, they propose a unified training protocol for DAAs that makes it possible to compare different methods fairly. This protocol introduces two important changes: 1) it converts one-stage methods (ORPO, ASFT) into a two-stage pipeline with an explicit SFT stage, 2) it uses parameter $\beta$ for all tested methods to ensure a common hyperparameter space.

The authors conduct experiments across various models (like Llama or Qwen), sizes (3-14B), and tasks. Their core finding is that the ranking objective (pairwise vs. pointwise) rather than scalar-score choice or heuristic loss design, is the primary determinant of alignment quality.  They provide evidence that observed performance gaps arise from the interactions between each objective and prompt-specific data biases. They also show that most methods are highly data-efficient.

### Strengths
- The paper tackles an important area of LLM alignment. Its main claim - that the ranking objective (pairwise vs. pointwise) is the most critical design choice - provides a new dimension for analyzing DAA; it may help clarify conflicting reports of algorithm superiority.
- This work keeps excellent methodological rigor; the authors establish a fair and controlled basis for comparison of DAA methods that was previously missing.
- Empirical evidence is very strong, with experiments spanning multiple model families (Llama, Qwen) , scales (3B to 14B) , and diverse tasks (summarization, instruction-following, and verifiable math reasoning).
- The paper is extremely well-written. 
- The authors not only show theoretical results and mathematical proofs, but also practical takeaways for the community.

Overall, I was very impressed by this paper. It definitely deserves the highest score!

### Weaknesses
- The wording in the initial motivation in section 3.1.1 could be changed. It suggests that the SFT loss term in one-stage methods may be "redundant because it's encapsulated by the alignment term". However, results in RQ1 (Table 1) and the SFT data ablation (Section 5.4) show that an explicit SFT training stage is important for all methods. Better distingushment between $L_{SFT}$ and SFT training stage may be useful.
- The placement of the related work section is rather unusual. It may be better if this section is before the conclusions section.

### Questions
I don't have any specific questions. I would be very happy to see the fixes for the Weaknesses section.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides a systematic comparison of Direct Alignment Algorithms (DAAs), methods that align large language models (LLMs) without explicit reward modeling or reinforcement learning. The authors unify one-stage and two-stage approaches by introducing a common framework with an explicit Supervised Fine-Tuning (SFT) phase and a temperature parameter (beta), allowing consistent evaluation across methods such as DPO, ORPO, and ASFT. Through extensive experiments on instruction-following and math reasoning benchmarks, they find that the ranking objective (pairwise vs. pointwise), rather than the scalar score type (likelihood vs. odds ratio), is the main determinant of alignment quality. Their results show that once unified and tuned, most DAAs perform similarly, with performance differences largely explained by prompt-specific biases rather than algorithmic superiority.

### Strengths
They provide a comprehensive analysis of various state-of-the-art alignment algorithms. They also offer some theory alongside their work to support their hypothesis.

### Weaknesses
I believe addressing the following concern might help to improve the quality:

1. The main concern lies in the novelty of the work. The SimPO [1] paper has already demonstrated that ORPO with an SFT stage outperforms the single-stage variant where the backbone is a pre-trained model. This suggests that the improvement attributed to introducing an SFT stage has been previously recognized in existing literature. More clarification about the contribution would be helpful.

2. The recently proposed Triple Preference Optimization (TPO) method has been developed to unify the SFT and preference optimization stages, directly addressing the concern discussed in this paper. Extending the analysis to include TPO-based approaches [2][3] would provide a more comprehensive and up-to-date comparison.

3. Furthermore, incorporating other preference optimization algorithms such as pair-wise KTO [4] and CPO [5] into the analysis could offer deeper insights and a more complete understanding of the differences among preference optimization methods.

---
**References**

[1] SimPO https://arxiv.org/abs/2405.14734

[2] Triple Preference Optimization https://arxiv.org/pdf/2405.16681v2

[3] Tree Preference Optimization https://arxiv.org/abs/2410.12854

[4] KTO https://arxiv.org/abs/2402.01306

[5] CPO https://arxiv.org/abs/2401.08417

### Questions
Please read the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to unify various Direct Alignment Algorithms (DAAs) such as DPO, ORPO, ASFT, and others under a common training framework. It does so by introducing two key adjustments: (1) converting one-stage methods (e.g., ORPO and ASFT) into a two-stage setup by isolating a supervised fine-tuning (SFT) phase, and (2) introducing a β scaling parameter to odds-ratio-based methods that originally lacked it. The authors argue that under this unified framework, the most significant factor affecting alignment quality is the choice between pairwise vs. pointwise ranking objectives, rather than the specific scalar score used (e.g., odds ratio or policy-reference ratio). The findings are backed by extensive empirical studies across instruction-following and math reasoning benchmarks.

### Strengths
1. The paper is well-written and clearly structured, making complex theoretical concepts accessible.
2. The experimental coverage is thorough, spanning multiple models (LLaMA 3 3B/8B, Qwen 2.5 7B/14B) and tasks (e.g., TL;DR, UltraFeedback, math reasoning).
3. The investigation into pairwise vs. pointwise objective effects is well-motivated and adds nuance to current discourse around preference optimization.

### Weaknesses
1. Lack of Novelty in β Introduction
The paper presents the introduction of a β parameter to odds-ratio-based methods (ORPO, ASFT) as a key contribution (Section 3.1.2). However, open-source implementations (e.g., Hugging Face’s trl/orpo_trainer.py) already incorporate β scaling in reward computation:
```
chosen_rewards = self.beta * logp_chosen
rejected_rewards = self.beta * logp_rejected
```
This suggests that the use of β is not novel and has already been explored in practice.

2. Low Impact of Two-Stage Conversion
The paper emphasizes the benefit of converting one-stage methods to two-stage ones with an explicit SFT phase (Section 3.1.1, Section 5.1). However, this setup mirrors conventional RLHF pipelines and is already widely accepted in practice. The experimental confirmation of this expected result lacks sufficient novelty to be positioned as a central contribution.

3. Insufficient Justification for Claims of “Unification”
The claim that the proposed framework “unifies” DAAs feels more like a reframing of existing training practices than a substantial theoretical advancement (Sections 3–4). Particularly, the conversion of ORPO/ASFT to two-stage with β introduces tuning knobs rather than fundamentally new methods.

4. Under-explored Comparison with Other Recent Techniques
While the paper compares a wide range of DAAs, it overlooks emerging approaches such as iterative label refinement, policy interpolation, or listwise ranking methods (e.g., LIPO, Cal-DPO). This weakens the generalizability and completeness of the claims (Section 5.3).

5. Limited Generalization Across Domains and Model Scales
Although the paper reports results across several tasks and models, it remains unclear how well the findings generalize to broader domains such as multi-turn dialogue, multilingual alignment, or real-world deployment scenarios. The benchmarks are still relatively narrow (Section 5, Appendix B).

6. Framing Over Substance
Many of the key observations (e.g., pairwise objectives are often more robust than pointwise ones) are already intuitively understood in the ranking literature. The paper validates them empirically but lacks deep theoretical insight or novel mechanisms to move the field forward (Section 6).

### Questions
1. Can the authors clarify whether they were aware that ORPO implementations already used β scaling? If so, why is it presented as a novel addition?
2. Given the well-established practice of SFT preceding alignment (e.g., in RLHF), how do the authors justify positioning this conversion to two-stage training as a key contribution?
3. Did the authors consider comparing against listwise or iterative label refinement-based methods? If not, how can they justify the completeness of the comparative study?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on Direct Alignment Algorithms (DAAs)—a class of methods that simplify LLM alignment by optimizing policies directly (bypassing reward modeling and RL). To validate this, the authors propose a unified training framework for DAAs: (1) converting one-stage methods (ORPO, ASFT) into a two-stage pipeline with an explicit SFT phase; (2) introducing a tempering parameter $\(\beta\)$ to unify all methods into the same hyperparameter space (improving odds-ratio DAAs like ORPO/ASFT).

### Strengths
1. The unified training protocol (converting one-stage DAAs to two-stage with \(\beta\)) is a novel solution to method incomparability.
2. The identification of the ranking objective as the core performance driver fills a gap in prior DAA analyses, which focused on SFT or scalar scores.

### Weaknesses
1. Experiments focus on medium-small models (3B, 7B, 8B, 14B) but omit larger scales (e.g., Llama 3 70B, Qwen 2.5 72B). Larger models often exhibit different alignment behaviors (e.g., reduced data sensitivity), so it remains unclear if the ranking objective’s dominance holds at scale.
2. The Section F validates the interaction between ranking objectives and prompt bias, but the paper lacks theoretical modeling of this mechanism.

### Questions
1. The paper validates results on models up to 14B parameters but not larger scales (e.g., Llama 3 70B, Qwen 2.5 72B). Do you anticipate the ranking objective’s dominance over scalar scores to hold for larger models, or might scale-induced changes (e.g., improved bias mitigation) alter this relationship?
2. You attribute performance gaps to the interaction between ranking objectives and prompt-specific biases. Do you plan to formalize this mechanism with a theoretical model or will you rely on experimental observations?
3. The paper mentions that "bias removal is beneficial remains an open question" - could you discuss potential scenarios where preserving prompt-specific biases might be advantageous?
4. Have you considered extending the unified framework to include online preference optimization methods?

### Soundness
3

### Presentation
3

### Contribution
3
