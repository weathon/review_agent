# Do Depth-Grown Models Overcome The Curse Of Depth? An In-Depth Analysis

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Gradually growing the depth of Transformers during training cannot only reduce training cost but also lead to improved reasoning performance, as shown by MIDAS (Saunshi et al., 2024). Thus far, however, a mechanistic understanding of these gains has been missing. In this work, we establish a connection to recent work showing that layers in the second half of non-grown, pre-layernorm Transformers contribute much less to the final output distribution than those in the first half - also known as the Curse of Depth (Sun et al., 2025, Csordás et al., 2025). Using depth-wise analyses, we demonstrate that growth via gradual middle stacking yields more effective utilization of model depth, alters the residual stream structure, and facilitates the formation of permutable computational blocks. In addition, we propose a lightweight modification of MIDAS that yields further improvements in downstream reasoning benchmarks. Overall, this work highlights how the gradual growth of model depth can lead to the formation of distinct computational circuits and overcome the limited depth utilization seen in standard non-grown models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper performs an analysis of MIDAS a gradual stacking technique to grow transformers whilst training them. The paper reproduces the MIDAS experiments on SmolLM at 360M and 1.7B scales.
The paper introduces LIDAS, slightly different to MIDAS, as when intruding a new 4 layer block B in-between B’ and B’’, takes the last two layers of B’ and first two layers of B’’. Where as MIDAS would just set B=B’. I view LIDAS less as a flashy novel method and more of a sanity check for the authors analysis experiments, although the authors find LIDAS is competitive with MIDAS.

The paper performs an in depth analysis of MIDAS and LIDAS:
1. Depth grown transformers utilise their depth more efficiently 
2. Depth grown models are more robust to block level reordering but less robust to layer level reordering or removal.
3. Block grown models exhibit cyclic patterns within the layers
4. MIDAS and LIDAS weights are more symmetric.

### Strengths
- The paper independently reproduces prior results, something often overlooked in ML research
- The papers presentation is very clear.
- The analysis presented is deep and through.

### Weaknesses
- The analysis is limited to one model, SmolLM-1.7b. Although this is an expensive analysis to perform.
- The benchmarks used are a little limited, perhaps the FineWeb-Edu benchmarks could be useful here.
- Initialisation and learning rate scheme not described for baseline, this is known to impact a models ability to use deeper layers effectively (https://arxiv.org/pdf/2505.01618).

Minor: The figures are slightly out of order with the text, e.g. Figure 1 is a long way away from where it is described. Reorganising slightly would make the presentation perfect.

### Questions
1. What learning rate and initialisation scaling methods were used during training? For example, Dey et al. (https://arxiv.org/pdf/2505.01618) show better learning rate scaling can lead to more effective training.
2. Figure 1 shows that depth grown models use their later layers, but this needed? For example it now looks like earlier layers are used less, does growing move the problem or solve the problem?
3. The paper discusses weights being more symmetric when using M/LIDAS, do we necessarily want symmetric weights in our trained language models?
4. Is there a link between M/LIDAS vs regular training and how pruning of weights post training can be conducted? For example, regular training may allow practitioners to prune more efficiently?

In the rebuttal I would be most interested in hearing about the initialisation and learning rate scaling for the baseline, then answers to the above questions.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Transformers are known not to fully exploit the deeper layers, known as the Curse of Depth. Recent depth growing techniques adaptively increase the depth of a model and can obtain more gain from the depth. This study investigates the Curse of Depth through detailed analysis comparing non-growing and growing models. The detailed analysis reveals three main observations. (i) The removal of deeper layers has significantly greater impact for depth-grown models than non-grown ones, indicating the exploitation of deeper layers in depth-grown models. (ii) The depth-grown models are more robust against computational block permutations. (iii) Depth-grown models present cyclic patterns in the attention sublayer's contribution over layers.

### Strengths
- Extensive experiments are conducted on the impact of deeper layers, comparing standard depth-fixed models and recent depth-growing models. 
- Besides, LIDAS, a variant of MIDAS method, is proposed, which attains superior performance in reasoning-intensive tasks.
- The presentation is easy to follow; the hypothesis, evidence, results, and interpretation are presented clearly.

### Weaknesses
This study experimentally collects observations on depth-non-growing and depth-growing models. While I appreciate them, one of the major weaknesses of this work is that the connection between these observations is unclear, and the practical takeaway from them is limited. 

Depth-fixed models do not fully take advantage of the depth, and deeper layers can be dropped with a subtle cost in performance. This has been known already, and the experiments collect related observations from layer-wise analysis with a contrast to depth-growing models. The introduction [l.053] writes 

> However, a clear mechanistic understanding of these gains has so far been missing. 

but I don't feel this paper fully addresses these points. The experiments strengthen the known fact but do not offer how to boost the depth-growing models (if some understanding is obtained, this should be done to some extent, which validates the understanding). 

The proposed method, LIDAS, is also kind of independent; it's not designed based on the observations. While LIDAS performs better than MIDAS in some tasks, the reason is not explained clearly or theoretically. 

I'm afraid to say that this work is preliminary or a "concatenation" of several results without any conclusive remarks. Each of the experiments reports a solid observation, and I appreciate it. The main concern is the final takeaway (and its impact) built upon them in training Transformer models,

### Questions
Please address the weaknesses. Particularly, what new understanding of depth-fixed and depth-growing models is obtained from experiments, and how is it tested? Note that this is not asking about how the experiments to obtain new observations are performed (these are already well presented), but asking what new understanding/hypothesis is obtained from the observations, and how their correctness is justified.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies depth‑growth training (e.g., MIDAS) as a remedy for the “curse of depth” observed in pre‑LN Transformers and proposes LIDAS, a layer‑wise middle‑duplication variant. Using SmolLM‑v1 at 360M and 1.7B parameters, the authors show that grown models (i) increase late‑layer utility according to multiple diagnostics (depth score, early‑exit overlap, tuned‑lens accuracy), (ii) exhibit permutable computational blocks in the middle of the network, and (iii) match or exceed MIDAS on reasoning while keeping NLL stable. Training compute is reported to decrease by ≈23% (1/1.29×) under the paper’s schedule.

### Strengths
- Mechanistic depth. Converging diagnostics (tuned‑lens, early‑exit, swap/reverse/skip) make a coherent case that growth increases effective depth usage.
- Actionable variant. LIDAS is a lightweight change that preserves or improves reasoning without harming NLL (i.e., token-level negative log-likelihood/perplexity on held-out text), indicating no regression in general language modeling quality).
- Reproducibility. Setups and intervention protocols are described clearly; the narrative is easy to follow.

### Weaknesses
- Compute accounting / fairness. Main comparisons fix steps, not FLOPs. Since growth changes training compute, a FLOPs‑matched baseline (e.g., truncating baseline steps to ≈77%) is needed to support “efficiency–performance” claims. Error bars (multi‑seed) are also missing on the headline numbers.
- Cross‑method context. The paper positions growth as a remedy for pre‑LN “curse of depth,” yet omits direct comparisons to LayerNorm scaling baselines. A small 2×2 factorial (Pre‑LN vs Mix‑LN) × (no growth vs LIDAS) would clarify whether growth is orthogonal or redundant with normalization tricks.
- Scale & breadth. Evidence is limited to 360M/1.7B and emphasizes reasoning; generalization to ≥7B/13B, instruction‑tuned, code, and knowledge‑heavy QA remains uncertain. Systems issues that may arise at scale (optimizer‑state copying, pipeline‑parallel rebalancing at growth boundaries) are not explored.

### Questions
- Backbone generality. How do findings transfer to post‑LN/parallel‑residual and MoE backbones?
- Curricula interplay. Are growth and length curriculum or token‑drop schedules complementary or redundant?
- Failure modes. Any tasks where growth hurts (knowledge‑heavy QA, code), and diagnostic clues as to why?
- Large‑scale stability. At ≥13B, do you observe optimizer‑state copy overheads or the need for short local warm‑ups at growth boundaries?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a mechanistic investigation into why progressive layer growth improves downstream reasoning performance in Transformer LMs. Starting from MIDAS, the authors develop a layer-wise growth variant (LIDAS) and present a series of depth diagnostics (1) early-exit behavior, skip/swap/reverse perturbations, and block-wise contribution/similarity analyses and (2) showing that growth increases late-layer functional utilization and mitigates the Curse of Depth in pre-LN models.

## Soundness  2 (fair)
The analyses are methodologically sound, and the triangulation of probing, perturbation, and contribution metrics is compelling. 
The primary limitation is that only 2 models of SmolLM family are studied, which is not sufficient to support the argument. Also, several claims are correlational, especially the interpretation that permutation robustness *implies* deeper utilization, which is plausible but not causally demonstrated.

## Presentation 3 (good)
The paper is exceptionally well written and well structured. The progression from motivation $\rightarrow$ analyses $\rightarrow$ implications is very clear. Some plots (Fig. 3, Fig. 4) assume background context that could be made more self-contained.

## Contribution - 3 (good)
The novelty lies in *mechanistic understanding*, not architecture: the work explains **why** depth growth improves performance, beyond reporting that it does. This deepens the conceptual grounding of growth-based reasoning improvements and is well suited for poster-level acceptance.

### Strengths
- Mechanistic insight into internal computation rather than only surface-level gains.
- Multiple convergent forms of evidence (early-exit, swap/reverse, contribution metrics).
- Clear link to Curse of Depth and demonstration of how growth reactivates late layers.
- Well-scoped and well-written with a strong explanatory narrative.
- Practical relevance for growth-based strategies in reasoning-oriented LMs.

### Weaknesses
### 1. Limited architectural generality
All experiments are on SmolLM (360M / 1.7B), a single pre-LN, short-context family. Since the contribution is mechanistic in nature, it remains unclear whether the observed “resurrected depth utilization” is a *property of staged growth itself*, or a *property of this architecture family*. 

### 2. Causal link between permutation robustness and “depth utilization” remains implicit
For example, section 4.2 shows that grown models are more robust to block-level reordering, and **claims** this as evidence of deeper utilization. However, it needs more detailed explanations to bridge this casual relation.

### 3. Performance gaps are small and lack statistical framing
In Table 1, most LIDAS vs MIDAS gains are <1pp; without multi-seed variance this is difficult to interpret; if without confidence intervals or multi-seed reporting, the differences are too small to determine whether those small gains reflect a stable trend or training stochasticity. Since LIDAS is an architectural refinement, the size and stability of differences matter for how much weight the architecture (vs. the growth mechanism) contributes to the overall story. 

### 4. The 360M + math-cooldown Primitive result is an unexplained outlier
In table 1, the improvement between LIDAS and MIDAS is dramatically larger, however, there's no explanation. (The paper itself only states that the cause of the improvements between grown model and baseline is unclear.) 
This weakens the confidence that the measured gains derive cleanly from the proposed mechanism rather than an interaction or confound.

### 5. Block-size may be an unexamined confound
The periodicity observed in section 4.3 is shown only for block size b=4. If b=2 or b=8 produces a different cycle structure, part of the reported effect may be hyperparameter-induced. A small ablation would confirm whether the pattern is intrinsic or scheduler-driven.

### 6. Need more explanation on the causal relation between symmetry and depth depth utilization 
Section 4.4 mentions that LIDAS is more block-wise symmetrical than MIDAS. However, it does not explain precisely why this offers an advantage in terms of in-depth utilization. Need more detailed explanation to build a causal link. 
Also, better if there is a causal link between this symmetry and the mechanism of LIDAS.

### Questions
In Fig. 1 and Fig. 9, the difference between the two variants is subtle (In Fig.1, it seems that LIDAS is not as good as MIDAS). Is the benefit primarily symmetry/stability, or does it yield distinct behavior in downstream layer activation? Clarifying *where* LIDAS helps would sharpen the architectural takeaway.

Additionally, please address the weaknesses stated.

### Soundness
2

### Presentation
3

### Contribution
3
