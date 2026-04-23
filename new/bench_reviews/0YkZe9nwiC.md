Now I have a solid understanding of the paper and the calibration landscape. Let me synthesize the final review.

## Summary

The paper introduces SIGnAL (Self-Informed Generative Active Learning), a reinforcement learning-based framework for query-synthesizing active learning. SIGnAL uses an LLM to generate synthetic data instances, selects the most informative ones via a novel acquisition function that combines informativeness (KL divergence of predictive distributions) with relevance (inverse embedding distance to labeled neighbors), and then optimizes the generator via PPO using the acquisition score as reward. Experiments on three text classification datasets (SST-2, AGNEWS, QNLI) at two data scales show SIGnAL consistently outperforming five pool-based baselines.

## Strengths

- **Well-motivated problem formulation**: The paper addresses a genuine gap in the AL literature—query-synthesizing methods for text data in low-resource scenarios where the unlabeled pool itself is limited. The formal definition of the n < b case (Section 3) goes beyond standard pool-based AL, which simply halts once the pool is exhausted.

- **Clean acquisition function design**: The acquisition function (Eq. 4, Section 4.2) that divides KL divergence by embedding distance is an intuitive and principled solution to the problem of irrelevant OOD synthetic data. The connection to CAL (Margatina et al., 2021) is clearly articulated, and the extension adds a relevance constraint that directly addresses the over-optimization failure mode identified in GAAL (Section 2.1). Figure 1 provides a clear geometric illustration.

- **Seamless transformation of acquisition function into RL reward**: The acquisition score directly becomes the PPO reward signal (Algorithm 1, lines 12–13; Eq. 6), elegantly solving the challenge of translating informativeness—which has no simple loss function—into a learnable signal. The KL penalty term mitigates over-optimization.

- **Consistent empirical improvement**: Across all 6 experimental settings in Figure 3, SIGnAL outperforms all five pool-based baselines, with the largest gains in the most data-limited scenarios (SST-2 0.1%, AGNEWS 0.1%), directly supporting the claim that generative AL is most beneficial when the unlabeled pool is small.

- **Insightful analysis of failure modes**: Section 5.4 explains why SIGnAL underperforms early (generator initially produces instances similar to in-context examples), why it performs better on SST-2/AGNEWS than QNLI (LLM pretraining bias toward entailment), and the emergent adaptive behavior where the generator gradually shifts to producing more not-entailment data as entailment data become less informative. This demonstrates genuine understanding of the framework's dynamics.

## Weaknesses

### Fatal
None.

### Major

- **No ablation isolating the RL contribution, which is the paper's central claim**: The paper's primary technical contribution is optimizing the generative model via RL using the acquisition function as reward (Sections 4.3, Abstract: "Self-Informed" = the generator is self-informed by the reward). Yet no experiment tests whether this RL optimization actually improves performance. A minimal and essential ablation would be: use the same LLM to generate data, apply the same acquisition function for selection, but skip the RL fine-tuning of the generator. Without this comparison, the paper does not establish that RL-based generator optimization helps over simply generating data with an unmodified LLM. The observed improvements could be entirely due to (a) having access to synthetic data at all, (b) the acquisition function, or (c) both. This gap means the core claimed contribution is unsupported by evidence. Papers with analogous gaps (e.g., missing ablations that prevent disentangling the core contribution from other components) have been rejected at ICLR (e.g., Diffusion Active Learning, avg 6.0, rejected for inability to disentangle diffusion vs AL contribution; IDS, avg 4.25, rejected for missing ablation studies).

- **No generative active learning baseline**: The paper compares only against pool-based baselines, justifying this by stating that "existing query-synthesizing methods are designed to handle image data" (Section 5.2). While this is factually accurate, the most natural and fair comparison is a simple "generate-then-select" pipeline using the same LLM (without RL) plus the same acquisition function—this would directly measure what RL adds. This baseline is trivial to implement and would simultaneously address the ablation gap above. Its absence makes it impossible to attribute SIGnAL's gains to the RL component rather than to the mere presence of synthetic data.

### Minor

- **Pseudo-oracle trained on full dataset introduces a confound in evaluation**: The paper uses classification models fine-tuned on the respective full datasets as pseudo-oracles to label synthetic instances (Section 5.3, accuracies 91.3%, 93.75%, 90.99%). While the paper acknowledges this introduces 6-9% label noise, the more concerning issue is that the pseudo-oracle encodes information about the full data distribution that would not be available in a realistic deployment where the oracle is also resource-constrained. In standard AL, the oracle is assumed perfect; the pseudo-oracle approximates this, but its full-dataset training means it provides higher-quality labels for synthetic data than would be achievable in practice, especially early in the AL loop. This does not invalidate the results (pool-based methods also have access to perfect labels for their pool instances), but it makes the experimental setup somewhat optimistic for SIGnAL specifically, since SIGnAL is the only method that needs oracle labels for non-pool data.

- **Ambiguity about what "0.1% and 1%" refers to**: Section 5.1 states "we randomly sample 0.1% and 1% of the original size from each dataset" to simulate "a limited initial unlabeled pool," but line 187 says methods are evaluated "with varying initial labeled datasets." Figure 3 shows pool-based baselines achieving results at 100% acquired data, which would be impossible if the unlabeled pool contained only 0.1% of the data. This suggests 0.1%/1% likely refers to the initial labeled set, but the paper's wording is contradictory, creating confusion about the experimental setup.

- **RL objective may misalign with actual classification improvement**: The generator is optimized to produce instances with high acquisition scores (Algorithm 1, line 12), but high-scoring instances might not be the ones selected (line 8 selects from the full unlabeled pool U including real data). The reward optimizes a proxy that may not perfectly align with downstream classification improvement. While this is a reasonable design choice, the paper does not discuss this potential misalignment.

- **Early-stage underperformance**: Section 5.4 acknowledges that "SIGnAL tends to underperform compared to pool-based methods during the early stages of training" because the generator initially produces instances similar to in-context examples. This is a practically significant limitation—the early AL iterations are often the most budget-constrained—that is not reflected in the abstract's general claim of effectiveness.

### Trivial
None.

## Nice-to-Haves

- Ablation with a true oracle (using ground-truth labels for synthetic data) would isolate the effect of pseudo-oracle noise, though the current setup approximates the standard AL oracle assumption reasonably well.
- Quantitative analysis of generated data quality/distribution shift before and after RL optimization (e.g., diversity metrics, distribution distance to real data) would strengthen the qualitative analysis in Section 5.4.
- Representative synthetic instances before and after RL optimization, with their acquisition scores, would qualitatively demonstrate what RL changes about the generation process.

## Removed Points

*These points were flagged but are removed or weakened after verification against the paper. Treat them with caution.*

- **"Data leakage through pseudo-oracle invalidates evaluation" (Harsh Critic, Fatal)**: The harsh critic claims the pseudo-oracle trained on the full dataset gives SIGnAL an "unfair informational advantage" that "undermines every experimental result." This is overstated. In standard AL, the oracle is assumed perfect; the pseudo-oracle approximates this, and is actually weaker (6-9% error rate). The pseudo-oracle does not directly feed information to the classifier—it only labels synthetic instances, and the classifier is trained only on the labeled set L. The pseudo-oracle's full-dataset training is a practical necessity (not a design flaw) that makes it a better approximation of the ideal oracle. I've moved a nuanced version of this concern to Minor weaknesses.

- **"Numerical instability from division by near-zero embedding distance" (Harsh Critic)**: This is a minor implementation detail; standard practice would handle this with epsilon clipping. Not substantive enough for the main review.

- **"argmax decoding contradicts diversity" (Harsh Critic)**: The paper explicitly acknowledges that LLMs tend to produce repetitive outputs and describes using diverse prompts as an alternative strategy (Section 4.1). The argmax characterization in Eq. 2 is a simplification; the actual implementation uses diverse prompts. This is acknowledged in the text.

- **"n < b case never addressed in the method" (Harsh Critic)**: The paper mentions this case in the problem definition but the method implicitly handles it—SIGnAL generates synthetic data precisely when the real pool is insufficient. The KNN step works as long as there are some labeled data points (the initial labeled set). This is a minor clarity issue, not a fundamental gap.

- **"KNN from small labeled set provides noisy signal" (Harsh Critic)**: This is inherent to any AL method that uses the current model state. Pool-based methods also suffer from poor model estimates early in training. Not specific to SIGnAL.

- **"Missing related works" (implied)**: Not included per rules—cannot verify existence of uncited works.

- **"Reproducibility concerns about epsilon term or clipping" (Harsh Critic)**: Removed per rules on reproducibility nitpicks.

## Novel Insights

The paper identifies an interesting asymmetry between the pool-based and generative AL settings that deserves more attention: in pool-based AL, the oracle assumption is "free" (labels exist but are hidden), while in generative AL, the oracle must actively provide labels for instances that never existed. This creates a practical dependency loop—the quality of generated data depends on the quality of the oracle, but the oracle's quality depends on how much real data is available. SIGnAL partially sidesteps this by using an external pseudo-oracle, but this highlights that the generative AL paradigm fundamentally requires stronger oracle assumptions than pool-based AL.

## Suggestions

- **Add the RL ablation**: This is the single most important change. Generate data with the same LLM without PPO fine-tuning, apply the same acquisition function, and compare. This directly tests whether RL optimization adds value and would make or break the paper's core claim.

- **Add a generative baseline**: A "Random-Generate + Acquire" pipeline (same LLM, random prompt selection, same acquisition function, no RL) would serve as the minimal generative AL comparison. If SIGnAL outperforms this, the RL contribution is validated.

- **Clarify the experimental setup**: Explicitly state whether 0.1%/1% refers to the initial labeled set or unlabeled pool. The current text is contradictory.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| EntiGraph (Synthetic continued pretraining) | 07yvxWDSla | 8.0 | Stronger: has theoretical model, thorough ablations, strong experiments. SIGnAL is weaker due to missing core ablation. |
| DataEnvGym | 00SnKBGTsz | 7.5 | Stronger: frames data gen as RL with student feedback, multiple tasks, comprehensive evaluation. SIGnAL has similar framing but much less thorough evaluation. |
| Diffusion Active Learning for CT | 73Q9U0vcja | 6.0 | Similar weakness pattern (missing ablation to disentangle core contribution), but more focused domain. Rejected. SIGnAL has broader evaluation but same fundamental gap. |
| AutoGeTS | JL18agpSc3 | 5.0 | Similar: LLM synthetic data for text classification, missing baselines. Rejected. SIGnAL has better baseline comparison but same core ablation gap. |
| IDS for Thorax | VbkGysQ0Rl | 4.25 | Similar: missing ablation studies to isolate contributions. Rejected. |
| AutoAL | 4RRmy9iw3c | 4.5 | Similar: AL method with novelty questions and evaluation gaps. Rejected. |
| GATE | tqiAfRT1Lq | 5.5 | Similar: missing ablations for core generative approach. Rejected. |
| Self-supervised pseudodata filtering | 2LhCPowI6i | 2.33 | Much weaker: generative models for data with genuinely weak evaluation. SIGnAL is clearly better. |

SIGnAL falls in the range of papers with missing ablations for their core contribution (4.0–6.0 range). It is stronger than the weakest examples (IDS 4.25, XAL 4.0) due to its well-motivated framework, multiple datasets, and clean acquisition function design. However, it shares the fundamental weakness of AutoGeTS (5.0) and Diffusion AL (6.0): the core claimed contribution cannot be disentangled from confounds. The missing RL ablation is particularly damaging because the paper's title and framing center on the "self-informed" RL mechanism. Without evidence that RL optimization helps, the paper's main claim is unsupported, even though the overall framework and acquisition function are valuable.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>