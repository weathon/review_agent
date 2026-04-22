Now I have a thorough understanding of the paper. Let me write the final review.

## Summary

CEMA proposes a black-box adversarial attack framework for multi-model multi-task text systems that reduces the multi-task attack problem to single-task classification via clustering. It queries a victim model 100 times on auxiliary texts, clusters the multi-task outputs into binary labels, trains substitute models on these pseudo-labels, generates adversarial candidates using existing text attack methods (DWB, FD, TextBugger), and selects the final adversarial example via ensemble transferability scoring across multiple substitute models. Experiments show 60–80% ASR on classification and BLEU < 0.18 on translation tasks with only 100 total queries.

## Strengths

- **Novel and practical problem setting**: Attacking black-box multi-model multi-task systems with only 100 queries is a realistic and understudied threat model. Existing multi-task attacks assume white-box access and shared parameters, which limits applicability. CEMA's plug-and-play clustering reduction (Section 4.2, Algorithm 1) that converts heterogeneous multi-task outputs into unified binary pseudo-labels is a creative and conceptually interesting approach for this setting.

- **Strong query efficiency**: CEMA uses only 100 total queries (amortized to ~0.045 per text) compared to baselines requiring 7–30+ queries per text (Table 1). This is a genuine advantage of the transfer-based approach: a one-time substitute model construction cost versus per-text query costs.

- **Empirical effectiveness with commercial APIs**: Results on Baidu Translate and Ali Translate (Table 2) demonstrate CEMA works against real-world closed-source systems, adding practical relevance beyond academic model-to-model evaluations.

- **Zero-shot robustness**: Table 6 shows CEMA maintains 66.40% ASR and 0.27 BLEU when using cross-distribution auxiliary data (SST5→Emotion), demonstrating resilience to distribution shift in auxiliary data.

- **Thorough ablations**: Tables 3–5 and Figure 2 provide ablations over the number of adversarial examples, clustering methods, cluster counts, and vectorization methods, showing the framework is not brittle to design choices.

## Weaknesses

### Fatal
None.

### Major

- **No joint multi-task success rate reported, undermining the core claim**: The paper's stated goal is "to degrade the performance of **all tasks** in a multi-task model" (Section 3). Yet all results report per-task ASR and BLEU independently. For Victim Model A, CEMA achieves 73.57% ASR on dis-sst5, 62.27% on dis-emotion, and 0.14 BLEU on opus-mt—these are per-task numbers. If task successes are roughly independent, the probability of simultaneously degrading all three tasks could be far lower (~28%). The paper never reports the fraction of inputs where all tasks are successfully attacked. Without this metric, it is impossible to evaluate whether CEMA achieves its stated multi-task attack goal versus simply being three independent single-task attacks sharing a perturbation. This gap is central to the paper's core claim, not a peripheral analysis request.

- **Missing ablation against per-task substitute models with the same query budget**: CEMA uses 100 queries to build one clustered substitute model for all tasks. A natural baseline is to allocate the same 100-query budget to train per-task substitute models (e.g., 50 queries per classification task) and attack each independently. If per-task models perform comparably, CEMA's clustering mechanism provides no advantage for multi-task attack; the benefit would come purely from transfer-based attack methods rather than the proposed framework. The absence of this ablation leaves unclear whether the clustering contribution is essential or whether the underlying attack methods (DWB, FD, TextBugger) do most of the work regardless.

### Minor

- **The theoretical contribution (Equations 2–5) is a standard union bound, not a CEMA-specific result**: The derivation shows that the probability of at least one successful adversarial example increases with the number of candidates, which is a straightforward application of the union bound on independent Bernoulli events. This is not specific to CEMA's clustering mechanism—any ensemble of methods would satisfy the same bound. The paper claims (contribution ❷) to "derive a theoretical lower bound for CEMA's success rate," but the bound applies generically. The independence assumption is also acknowledged as violated (adversarial examples are correlated), which further loosens an already universal bound. This is a minor issue because the paper's main value is empirical.

- **Query comparison is misleadingly presented**: Tables 1–2 show CEMA's queries as 0.045 per text vs. baselines' 7–30 per text. These numbers are incommensurable: CEMA's figure amortizes 100 shared queries across ~2200 texts, while baselines spend queries per text. The actual comparison is 100 total queries (CEMA) vs. potentially tens of thousands (baselines across the dataset). This is actually favorable to CEMA and should be presented honestly with total budget as the primary metric rather than per-text amortized figures that obscure the comparison.

- **The "few-shot" terminology overloads an established term**: In the NLP/ML literature, "few-shot" typically refers to learning from few examples of a new task (few-shot learning). Here it means "few queries to the victim model." While the intent is clear from context, this overloading may cause confusion and slightly weakens the framing precision.

### Trivial
None.

## Nice-to-Haves

- Report the joint multi-task success rate: for each input, whether all tasks are simultaneously degraded. This directly validates the stated goal.
- Add an ablation where the same 100-query budget is used to train separate per-task substitute models, to isolate the clustering contribution.
- Report per-task ASR conditioned on cluster label change: if the substitute model's label doesn't flip but the victim's task label does, CEMA's mechanism isn't driving success.
- Test cross-victim-model transfer (substitute trained on Model A's outputs, evaluated on Model B) to validate the transferability claim more directly.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Harsh Critic's claim that baselines increasing BLEU means the comparison is not meaningful**: Some translation baselines (e.g., TransFool with BLEU 0.77) produce higher BLEU than the original, but this is a known phenomenon in adversarial translation—some methods don't successfully attack the translation while changing the text. CEMA's low BLEU (0.14–0.23) is a genuine improvement. This is not a weakness of CEMA.

- **Harsh Critic's claim that the binary cluster choice is "circular logic"**: The paper justifies k=2 by noting that binary clusters maximize discriminability (Section 4.2) and empirically validates this in Figure 2. The circularity critique is overstated—empirical validation supplements the justification.

- **Harsh Critic's claim that "few-shot" is "misleading" to the point of being a structural flaw**: The paper explicitly defines "few-shot" in context as "few queries" (RQ3: "How to craft the adversarial examples in few-shot queries?"). While the term overloads standard usage, it is defined clearly and is not structurally misleading.

- **Strength Finder's claim about "theoretical lower bound" as a core strength**: The theoretical bound (Eqs. 2–5) is a standard union bound not specific to CEMA. It does not constitute a meaningful theoretical contribution and is removed from strengths.

- **Harsh Critic's claim about missing comparison with query-based black-box attacks for translation**: The paper already acknowledges the absence of black-box multi-task baselines and justifies using single-task methods as the only available comparison. This is a scope limitation, not a methodological flaw.

- **Strength Finder's claim that Table 1 provides "orders-of-magnitude improvement" over baselines**: This conflates per-text amortized query counts with per-text query counts, which are not comparable metrics. The query advantage is real but the "orders of magnitude" framing is misleading.

## Novel Insights

The clustering-to-classification reduction is genuinely interesting as a mechanism for handling heterogeneous multi-task outputs in a black-box setting. The key insight is that by clustering multi-task outputs into binary pseudo-labels, one can leverage existing single-task text attack methods without needing explicit task-specific labels or model access. However, the empirical evidence does not conclusively show that this clustering contributes beyond what the underlying attack methods achieve on their own, leaving the mechanism's contribution ambiguous.

## Suggestions

- Add a "joint success rate" column to Tables 1 and 2 that reports the percentage of inputs where all tasks are simultaneously degraded. This directly addresses the paper's stated goal and requires no new experimental infrastructure.
- Reframe the query comparison using total budget (100 queries vs. baseline total queries across all texts and tasks) as the primary metric, with per-text amortized as supplementary.
- Add an ablation where 50 queries per classification task are used to train task-specific substitute models (no clustering), attacking each task independently, to isolate the marginal value of the clustering framework.
- Tone down the theoretical contribution (contribution ❷): describe Equations 2–5 as providing "formal motivation" rather than a "theoretical lower bound for CEMA's success rate."

## Evaluation Axes

- **Originality**: The problem setting (black-box, multi-model, multi-task, limited queries) and clustering reduction are novel contributions. The theoretical component is not original. Moderate-to-high originality overall.
- **Importance of research question**: High—a practical attack on realistic multi-task systems is important for understanding adversarial vulnerabilities.
- **Claims supported**: Partially. The per-task results are clear, but the multi-task claim ("degrade all tasks") lacks the necessary joint evaluation. The clustering mechanism's necessity is not conclusively demonstrated.
- **Soundness of experiments**: Good per-task evaluation, appropriate ablations on clustering/vectorization/ensemble size, and commercial API testing. Missing the critical joint and per-task ablation baselines.
- **Clarity**: Generally clear presentation with good algorithm descriptions. The query metric presentation could be more transparent.
- **Value to community**: Moderate. The framework is practical and could inspire further work, but the currently presented evidence does not fully validate the multi-task claim.

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Boosting Ray Search (tIBAOcAvn4) | High anchor | 7.50 | Strong theoretical grounding + strong experiments on a well-studied problem. CEMA has weaker theory and a similar practical setting but less rigorous validation of its core multi-task claim. Below this anchor. |
| Multi-task backdoor defense (dqMqAaw7Sq) | High anchor | 7.00 | Novel multi-task defense with clear evaluation of multi-task metrics. CEMA lacks the equivalent joint evaluation. Below this anchor. |
| TIARA (4GcZSTqlkr) | Medium anchor | 4.50 | Transfer attack with substitute models, similar setting. Oversold claims and inadequate evaluation details. CEMA has more comprehensive ablations but also overclaims. CEMA is comparable to or slightly above this. |
| Model extraction via counterfactuals (ORHuMEwaC8) | Medium anchor | 5.00 | Solid idea with incomplete evaluation. Similar profile to CEMA. |
| Adversarial attacks on fine-tuned LLMs (9kR4MREN9E) | Low-medium anchor | 3.50 | Missing critical baselines (query-based attacks), oversold claims. CEMA has better ablations but similarly overclaimed multi-task capability. CEMA is above this. |
| Boosting targeted adversarial attack (gWk8WQVWGr) | Low anchor | 3.50 | Oversold "novel" method, trivial theory, poor presentation. CEMA is clearly above this—it has a genuinely novel problem setting and more thorough experimentation. |

CEMA sits above the low-scoring transfer attack papers (~3.5) that oversell marginal methods and below the high-scoring papers (~7.0–7.5) that have both strong theory and thorough evaluation. It is comparable to the medium-scoring papers (~4.5–5.0) that have a good idea but incomplete validation. However, CEMA's problem formulation is more novel than most of these, and its empirical scope is genuine (commercial APIs, ablations, zero-shot). The missing joint evaluation is significant but could be addressed in a revision. I place CEMA somewhat above the medium anchors because of the novelty of the research question and the practical relevance.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>