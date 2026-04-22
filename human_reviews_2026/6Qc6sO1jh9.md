# Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Retrieval-augmented generation (RAG) improves performance on knowledge-intensive tasks but can be derailed by wrong, irrelevant, or conflicting retrieved text, causing models to rely on inaccurate evidence and cascade errors. We propose Knowledgeable-R1, a reinforcement-learning framework that explicitly trains large language models to use parametric knowledge (PK) to resist contextual interference while still exploiting external context when it is reliably helpful. Knowledgeable-R1 introduces a joint sampling scheme that generates paired responses with and without retrieval, and learns both local advantages (within each decoding regime) and global advantages under the same input to quantify when to ignore misleading context versus adopt it. We employ an asymmetric advantage transformation that amplifies exploratory behaviors toward parametric knowledge. Experiments show that Knowledgeable-R1 significantly improves robustness and reasoning accuracy in knowledge conflict scenarios and general RAG scenarios, outperforming SOTA baselines by +22.89% in counterfactual scenarios, and without degradation when the retrieved context is fully accurate.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To address the tendency of models to disregard **parametric knowledge (PK)** when the retrieved context in retrieval-augmented generation is **noisy, misleading, or contradictory**, the authors propose **Knowledgeable-R1** — a **reinforcement learning (RL)** framework that explicitly trains LLMs to resist contextual interference while still leveraging useful retrieved information.

Training follows a **GRPO-style** setup with three policies:

* **PK:** query-only (parametric knowledge)
* **CK:** query + context (context-aware)
* **RPK:** query + context, but decoding along the PK trajectory to test robustness

### Strengths
**Well-motivated problem.** The work addresses a practical and underexplored limitation of RAG systems — their lack of robustness to noisy contexts.  

**Strong empirical validation.** Results hold across two model families, four model sizes, and five datasets, showing consistent robustness gains under four interference conditions while preserving accuracy on clean contexts.  

**Comprehensive ablations.** Each design choice (joint sampling, local/global advantages, adaptive β) is empirically justified, with clear performance drops when omitted.

**Novelty.** Unlike prior approaches that dealt with noisy contexts (e.g., [1], [2], [3]), this framework explicitly operationalizes the PK–CK trade-off in a reinforcement learning manner rather than using prompt engineering or fine-tuning. It closely resembles [4], although that work is quite recent and may not have been available when this paper was written.

[1] Asai, Akari, et al. "Self-rag: Self-reflective retrieval augmented generation." NeurIPS 2023 workshop on instruction tuning and instruction following. 2023.

[2] Yoran, Ori, et al. "Making retrieval-augmented language models robust to irrelevant context." arXiv preprint arXiv:2310.01558 (2023).

[3] Wang, Fei, et al. "Astute rag: Overcoming imperfect retrieval augmentation and knowledge conflicts for large language models." arXiv preprint arXiv:2410.07176 (2024).

[4] Xu, Tingqiao, et al. "DyKnow-RAG: Dynamic Knowledge Utilization Reinforcement Framework for Noisy Retrieval-Augmented Generation in E-commerce Search Relevance." arXiv preprint arXiv:2510.11122 (2025).

### Weaknesses
**Key method part is not well-explained**. It is not quite clear for me how RPK sampling works. in line 152 RPK policy is defined as $ \pi (o_t \mid p^{\prime}, o_{<t})$ - output is conditioned on previously generated tokens $o_{<t}$ and $p^{\prime} =$ query + context. But in the next paragraph, the authors say that $o_t$ is conditioned on query-only input  $p$: "we use $o$ to denote a token sequence generated when the current policy is conditioned on $p$ (query-only input)". Then the authors say that $o$ means possibly two different things - output of PK ("answer from parametric knowledge") and output of RPK ("answer consistent with PK"). Could you please elaborate what is the difference between these two and maybe use different variables for them? What is input and output when each new token in RPK is generated? I think it is very important as RPK is one of the key differences of Knowledgeable-R1 from GRPO + RAG baseline.

**Heuristic nature of β-modulation.** The adaptive penalty scaling is empirically tuned, with limited theoretical justification. A more formal grounding or ablation across β dynamics would strengthen the argument.  

**Limited exploration of partial-conflict granularity.** The paper notes this in its limitations — sensitivity analysis on varying proportions of misleading passages would be valuable.  

**Narrow task domain.** Evaluation focuses on QA-style tasks; it remains unclear whether the learned robustness generalizes to open-ended generation, dialogue, or summarization settings.

**Confusing writing:**

- Baselines are not well explained in Section 4.1.1:  
  - CK-Plug is omitted.  
  - GRPO with RAG is not clearly described — the difference from Knowledgeable-R1 is unclear.  
  - Key explanations appear only in the appendix.  
- Lines 187–209 — indices *i*, *j*, *k* are unclear. Do they denote rollout indices?  
  - The distinction between them is not specified — could they be unified under a single index?
- Font size in Tables 2 and 3 is too small to be properly readable.

**Typos**
- In Figure 1, the text “Probability distribution” overlaps with the image border.

### Questions
- Why don't you include KL-regularizer $\lambda_{KL} D_{KL}[\pi_\theta \| \pi_{ref}]$ in your total objective to avoid drifting too far from the reference policy, same as in [1], for example?
- What does the barplot in the top-right corner (the one with values 37.3, 19.7, 8.1 and 25.8) of Figure 1 mean? 

[1] Xu, Tingqiao, et al. "DyKnow-RAG: Dynamic Knowledge Utilization Reinforcement Framework for Noisy Retrieval-Augmented Generation in E-commerce Search Relevance." arXiv preprint arXiv:2510.11122 (2025).

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The study proposes Knowledgeable-R1—a reinforcement learning (RL) framework. It explicitly trains large language models (LLMs) to leverage parametric knowledge (PK) for resisting contextual interference, while still utilizing external context when it is reliably beneficial.

Experimental results show Knowledgeable-R1 significantly boosts robustness and reasoning accuracy in both knowledge conflict scenarios and general RAG scenarios. It outperforms state-of-the-art (SOTA) baselines by 23% in counterfactual scenarios, with no performance degradation when retrieved context is fully accurate.

### Strengths
1. The paper addresses a highly practical and critical issue in Retrieval-Augmented Generation (RAG) systems: "context interference" or "Context Dominance". Specifically, when Large Language Models (LLMs) encounter retrieved context that is erroneous, irrelevant, or conflicting, they tend to over-rely on this context while ignoring their internal, more accurate Parametric Knowledge (PK).

2. It designs a multi-objective Reinforcement Learning (RL) framework. The most crucial design within this framework is the introduction of three sampling strategies, with the Robust-PK (RPK) strategy being particularly notable. Given input in the form of "query + context", this strategy trains the model to generate answers consistent with PK—effectively making the model ignore the context. This creates a dynamic competition within the model between "utilizing context" and "resisting context".

3. "Knowledge Balance Modulation" (detailed in Section 3.4) is an ingenious technical element. Through an asymmetric and adaptive transformation of the advantage function (via coefficients), it effectively prevents the RPK strategy from being overshadowed by the Context-aware (CK) strategy— which is "usually more useful"—during training. This ensures the model retains the ability to fall back on PK when confronted with "occasional" erroneous context.

### Weaknesses
1. The method introduces significant training overhead. It requires maintaining and optimizing three distinct strategic objectives (PK, CK, RPK), calculating complex "local + global" advantages, and finally implementing asymmetric modulation. This is far more complex in both implementation and computation compared to standard Supervised Fine-Tuning (SFT) or GRPO with RAG.

2. As mentioned in Section 3.2 of the paper: "During inference, the model does not explicitly switch controllers; the learned token distribution... implicitly... falls back to PK." This description is overly vague. Are these three strategies (PK, CK, RPK) merely "scaffolding" used during training, with only one model produced in the end? If so, how does this single model arbitrate between these three conflicting behaviors during inference?

3. The method appears to heavily rely on the presence of samples with "bad context" (S2-S5) in the training data. Appendix C mentions that an "equal proportion" of correct and incorrect context was randomly added to ConFiQA. If 99% of the training dataset consists of "good context" (S1), can the method still effectively learn the ability to "resist" (erroneous context)? While adaptivity may alleviate this issue, the paper lacks a sensitivity analysis of the proportion of "good/bad" context in the training data.

### Questions
1. The experimental section should include additional discussions on training costs (e.g., training duration, GPU resource consumption) and training stability.

2. To better demonstrate the necessity of the complex RL framework, it is recommended to add a stronger SFT baseline. For instance, conduct SFT on the same dataset (containing both good and bad context) used for RL, and use specific instructions such as: "Please answer with reference to the context, but if the context conflicts with your internal knowledge, prioritize your internal knowledge."

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
Targets contextual interference in RAG and proposes Knowledgeable-R1 with joint sampling, local/global advantages, and asymmetric β-shaping to balance PK vs. context use. Shows strong robustness under adversarial/contradictory/irrelevant contexts; in S1 (correct context) it’s slightly below GRPO w/ RAG. Missing pieces: finetuning baselines (e.g., Self-RAG, InFO-RAG) and sensitivity ablations (fixed vs. adaptive β, local+global advantage weights, and J_PK/J_CK/J_RPK weighting) to diagnose the S1 gap.

### Strengths
The paper squarely targets the challenge of contextual interference—conflicts between context prompts and parametric knowledge—and proposes Knowledgeable-R1, which improves robustness via joint sampling, local/global advantage design, and an asymmetric advantage transformation (reward shaping).

### Weaknesses
1. Lack Finetuning baselines. Please include finetuning methods cited in Related Work (e.g., Self-RAG, InFO-RAG) as baselines for a fair comparison.
2. β scheduling. Compare fixed β settings ({0.2, 0.5, 0.8, 1.0}) versus the adaptive β scheme to demonstrate the necessity of adaptivity.
3. Advantage composition. Validate whether combining local + global advantages is necessary (e.g., ablations varying their relative weights).
4. S1 performance gap. In settings with correct context (S1), performance is generally below GRPO w/ RAG. This may stem from the weighting among J_PK, J_CK, J_RPK, or from the advantage formulation. Please discuss and, if possible, provide experiments to diagnose these factors.

### Questions
N/A

### Soundness
2

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
5

### Summary
This paper proposes Knowledgeable‑R1, a reinforcement‑learning framework for RAG that explicitly trains a single LLM to answer from parametric knowledge, PK when no context is provided or use contextual knowledge, CK when retrieved evidence is reliable, and fall back to PK when the retrieved context is misleading. The method samples paired trajectories for the same question under three “policies”: PK (query‑only), CK (query+context), and RPK (query+context but encouraged to stay on the PK answer). It then computes advantages with two normalizations: local (within the same policy) and global (CK vs. RPK under the same input), plus an asymmetric advantage transformation with a dynamic coefficient β that down‑weights negative advantages for RPK to preserve PK as a viable fallback. Optimization uses PPO‑style clipping.
The paper evaluates five scenarios—accurate, adversarial, self‑conflicting, irrelevant, and partially‑relevant contexts—on Qwen2.5 and Llama‑3.1. It shows large gains when context is wrong or mixed, while remaining competitive when context is correct. For example, on adversarial context with Qwen2.5‑7B, NC‑MR improves from 13.47% (RAG prompting) to 43.94% (Knowledgeable‑R1), and on partly‑relevant context, HotpotQA rises from 20.36% (RAG prompting) to 31.45% (Knowledgeable‑R1).

### Strengths
The paper tackles “context dominance” in RAG and designs RL signals that explicitly arbitrate between CK and PK at the same input state (global advantage), which is the right granularity for conflict resolution. The three‑policy setup is well‑motivated and neatly summarized
The combination of local vs. global advantages and an asymmetric transformation for RPK is a coherent way to reward using context when both sources agree or CK is right (timeliness) and protect PK exploration when CK is wrong. The β adaptation rule is simple and effective.
The performance compared with baselines is promising.

### Weaknesses
Retrieval details are not clear, like what retriever, indexing corpus, and query formulations were used.

The headline gains mostly compare to RAG prompting (e.g., +30.47/+29.28/+18.09 on Qwen2.5-7B NC-MR/MC/QA; Table 2). Against GRPO w/ RAG, improvements are smaller (e.g., 43.94 vs 26.94 = +17.00). How the “23% over GRPO” figure is aggregated should be clarified. 

The paper relies on exact match accuracy but does not directly quantify context dependence like answer stability when contexts are removed/perturbed. Whether the robustness comes from mechanism (ignoring misleading passages) or coincidence (answering correctly despite them) remains unclear.

The paper claims that the model implicitly decides to use RAG context or not, but the default inference template feeds query+context, lacking the comparison of query-only and query + RAG context.

The paper reports that  fixed β will degrade performance but does not include further analysis like the sensitivity of β

### Questions
Against which baseline and which aggregation is the reported 23% gains computed? Please align the abstract with Table 2 and report confidence intervals or paired significance tests. 

What retriever, indexing corpus, and query formulations were used? How are top‑k passages truncated and ordered?

Training uses three rollout types; what is the token and wall‑clock overhead vs. GRPO w/ RAG

Can you please report context-dependence metrics(corresponding to Weakness 3)

Can you add experiments that compare the actual performance under query-only and query+context settings?(corresponding to weakness 4)

Please plot β over training and provide a simple sensitivity study?

### Soundness
3

### Presentation
3

### Contribution
3
