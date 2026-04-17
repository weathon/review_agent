Let me now synthesize my final review based on all the information I've gathered. I need to carefully evaluate each of the harsh critic's points against the actual paper content.

Let me think through the key issues systematically:

**1. Circularity of reliability estimation:** The harsh critic argues that consensus uses the noisy label y_i, making it circular under DNC. Let me check this. Looking at the paper:
- Consensus is defined as c_i = max(0, s_i · y_i). This indeed uses the noisy label y_i.
- The thresholds β_u, β_c use S_TP = {i | argmax(s_i) = argmax(y_i)}, which also relies on y_i.
- However, the paper's approach is a self-training/bootstrapping approach common in noisy label learning — you use the model's own predictions (similarity scores) alongside the noisy labels. The method drops high-uncertainty pairs entirely (S_U), keeps high-consensus pairs (S_C) with original labels, and soft-labels medium pairs (S_I). This is essentially a curriculum learning + label smoothing approach.
- The critic's concern about circularity is partially valid but somewhat overstated — many successful noisy label methods use similar self-training strategies. The key question is whether the initialization and dynamics allow convergence to reasonable reliability estimates.

**2. Greedy consensus estimation:** Assumption 1 is indeed strong. Marginal contribution might not correctly identify noisy attributes in all cases. But this is used at test time when y_i is unavailable, and the paper provides some empirical validation through ablation.

**3. MLLM at test time:** This is a legitimate concern. Using Qwen2.5-VL-72B is a very heavy model. However, the TTR module is presented as a complementary contribution and the ablation shows the training-time components (DRL, DRF) also contribute. The "MLLM Enhance" row in Table 3 shows using MLLM alone gives 56.6/69.0 vs default 58.2/69.7, suggesting TTR adds marginal improvement on top of MLLM. Wait, let me re-read... Actually "MLLM Enhance" vs "w/o TTR" suggests MLLM reasoning rethinking scores alone add about 4-5 points (52.4→56.6 on H@1 for Non-name, 50% DNC). But the overall "Default" adds TTR on top of training-time components. The DRL+DRF contribution is also significant (31.6→58.2 for w/o DRL vs default). So the MLLM is not the entire contribution but is substantial.

The bigger concern is that no fair comparison exists: no baseline uses a similar MLLM, so we can't tell how much is RULE-specific vs. MLLM-powered.

**4. Noise injection methodology:** The synthetic noise may not match real-world DNC patterns. This is a valid concern but standard in the field.

**5. Theoretical pieces (Theorem 1, 2):** Theorem 1 is indeed somewhat tautological but serves to motivate the consensus principle. Theorem 2 is in the appendix.

**6. Missing noisy correspondence baselines:** The paper only compares with standard MMEA methods, not with methods specifically designed for noisy correspondence from adjacent domains. This is a valid concern.

Now, let me assess the overall quality:

**Strengths:**
- Novel and well-motivated problem formulation (DNC in MMEA)
- Comprehensive framework addressing both training-time and test-time robustness
- Strong empirical results across datasets and noise settings
- Good ablation study
- Theoretically grounded uncertainty estimation via Dirichlet/evidential framework

**Weaknesses:**
- MLLM dependency at test time is a significant practical concern, and the lack of comparison with simpler test-time strategies makes it hard to attribute gains
- Circularity in reliability estimation (consensus uses noisy labels)
- Assumption 1 is strong and under-validated
- Scalability concerns with Dirichlet parameters
- Limited hyperparameter sensitivity analysis in main paper

Compared to calibration papers:
- Norton (noisy correspondence, 8,8,8,8 oral): much stronger novelty and cleaner methodology
- OTGM (graph matching with noisy correspondence, 6,6,5,5 withdrawn/rejected): similar direction but weaker
- R-EDL (evidential learning, 6,6,8,8,8 spotlight): incremental but solid
- READ (test-time adaptation, 8,8,8,8 poster): strong novelty and well-executed
- CvxcWCDX0h (noisy pairs, 3,3,5,3): weak novelty and methodology
- A4mJuFRMN8 (Dirichlet weighting, 6,6,6,5,6 poster): incremental but reasonable

This paper is roughly in the 5-6 range. It has a good problem formulation and strong results, but the methodology has notable issues (circularity, heavy MLLM dependency, under-validated assumptions). The contribution is significant but the approach has enough concerns that prevent it from being clearly above threshold.</think>Now I have all the information I need. Let me synthesize the final review.

## Summary

This paper identifies and formalizes the Dual-level Noisy Correspondence (DNC) problem in Multi-Modal Entity Alignment (MMEA), where both intra-entity (entity-attribute) and inter-graph (entity-entity, attribute-attribute) correspondences can be noisy. The authors propose RULE, which estimates correspondence reliability via a two-fold principle (uncertainty and consensus), uses this reliability for robust attribute fusion (DRF) and inter-graph discrepancy elimination (DRL) during training, and introduces a test-time correspondence reasoning (TTR) module using a large MLLM with Chain-of-Thought to uncover latent attribute connections during inference.

## Strengths

1. **Well-motivated problem formulation.** The DNC problem is genuinely practical and underexplored in MMEA. The statistics showing >50% noise in ICEWS benchmarks and the clear performance degradation under DNC (Fig. 1b) effectively establish the significance and urgency of the problem.

2. **Comprehensive framework addressing multiple facets of noise.** RULE handles both training-time noise (via DRL and DRF) and test-time noise (via TTR), providing a relatively complete solution. The two-fold reliability estimation principle (uncertainty + consensus) is a conceptually sound extension of prior evidential deep learning, and Theorem 1 correctly identifies the limitation of using uncertainty alone.

3. **Consistently strong empirical results.** RULE outperforms all seven baselines across five benchmarks and three noise settings (Inherent, 20%, 50% DNC) in both Non-name and All-attributes protocols. The improvements are often substantial (e.g., ~8-10 H@1 points over the best baseline on ICEWS datasets under 50% DNC).

4. **Informative ablation and analysis.** Table 3 disentangles the contributions of DRL, DRF, and TTR. The reliability distribution visualization (Fig. 3b), subset separation analysis (Fig. 4), and attribute reliability heatmaps (Fig. 5) provide useful interpretability.

5. **Test-time robustness is a novel angle.** To my knowledge, this is among the first works to address test-time robustness for MMEA, going beyond the standard training-time noise handling paradigm.

## Weaknesses

### Major

- **Heavy reliance on a 72B-parameter MLLM at test time, without cost analysis or adequate isolation of its contribution.** The TTR module uses Qwen2.5-VL-72B-Instruct for test-time reasoning, which is orders of magnitude more expensive than the baseline MMEA pipeline. The paper provides no inference time, GPU memory, or scalability analysis. More critically, the ablation does not cleanly isolate the MLLM's contribution: the "MLLM Enhance" variant (Table 3) already uses the MLLM for rethinking scores, and "w/o TTR" removes the TTR combination but retains the trained model. There is no ablation that removes the MLLM entirely from the pipeline while keeping the training-time components (DRL + DRF), making it impossible to determine how much of the performance gain comes from principled noise handling versus simply deploying a vastly more powerful pre-trained model at inference. This is the single most impactful experimental gap.

- **Partial circularity in the reliability estimation mechanism.** The consensus term c_i = max(0, s_i · y_i) in Eq. (5) explicitly uses the noisy label vector y_i. The self-adaptive thresholds β_u and β_c in Eq. (8) are computed from S_TP = {i | argmax(s_i) = argmax(y_i)}, which equates agreement between model predictions and noisy labels with "true positives." Under heavy DNC, many noisy pairs may be classified as S_TP if the model learns to match them, while some genuinely correct but hard pairs may be excluded. This creates a bootstrapping circularity: the reliability estimates used to mitigate noise depend on noisy labels and model predictions trained on that same noise. While this self-training-style approach is common in noisy label literature and can work with good initialization, the paper does not empirically analyze the accuracy of the pair division (precision/recall of S_U, S_I, S_C against ground-truth noise labels) nor discuss the convergence dynamics that would alleviate this concern.

- **Assumption 1 (marginal contribution) is strong and insufficiently validated.** The greedy consensus estimation at test time (Sec. 2.2.2, Eq. 7) relies on the assumption that correctly associated attributes yield Δ ≥ 0 while irrelevant ones yield Δ < 0. In realistic multi-modal KGs, correct but weak attributes (e.g., ambiguous images, vague text descriptions) can plausibly yield negative marginal contribution, and noisy but visually similar attributes can yield positive contribution. The paper provides no empirical analysis of how often this assumption holds or how performance degrades when it is violated. Given that this underpins the entire test-time pipeline when ground-truth labels are unavailable, this is a significant concern.

### Minor

- **Synthetic noise injection may not capture real-world DNC patterns.** The paper constructs synthetic noise via random replacement, Gaussian perturbation of images, and random character replacement for text. These are standard label-noise models but do not reflect the semantically plausible misassociations described in the introduction (e.g., visually similar but incorrect images). Experiments with semantically motivated noise (e.g., CLIP-retrieved similar-but-incorrect attributes) would strengthen evaluation.

- **Limited hyperparameter sensitivity analysis in the main text.** γ is "fixed as 0.5 for simplicity," β is fixed at 0.3, λ at 1e−4, and τ at 0.07. These directly control the noise/clean partition and loss weighting. Only γ is analyzed in Appendix G.10. Given that the reliability estimation is the core mechanism, sensitivity analysis for at least β and λ should appear prominently.

- **Scalability of Dirichlet evidence computation.** The evidence vector e_i has dimension Ñ (the number of entities in the target graph), requiring O(Ñ) similarity computations per entity per iteration. For industry-scale KGs with millions of entities, this could be prohibitive. The paper does not discuss this or propose approximation strategies.

- **Incomplete justification of Theorem 2.** Theorem 2 is cited as showing "an upper bound of query probability proportional to Q_i" but is relegated to the appendix without statement or proof in the main text. This makes it difficult to assess the theoretical grounding for the regularization term in Eq. (13).

- **No comparison with noisy-correspondence methods from adjacent domains.** All seven baselines are standard MMEA methods not designed for noise. Comparing with adapted methods from noisy label/correspondence learning (e.g., co-teaching, contrastive learning with robust losses) would better establish whether the proposed mechanisms are truly necessary.

### Trivial

- The definition of attribute-attribute correspondence (y_ij^m = 1 iff h_i^m = 1 & h_j^m = 1 & y_ij = 1) effectively makes attribute-attribute noise entirely derivative of entity-entity and entity-attribute noise. This slightly inflates the DNC problem as having "dual levels" when the attribute-attribute dimension is not independently parameterized.

## Nice-to-Haves

- Report inference cost (time, FLOPs, memory) for the TTR module, and experiment with smaller MLLMs (7B, 14B variants) to assess whether TTR gains are proportional to model scale.

- Include an ablation that uses only the training-time components (DRL + DRF) without any MLLM, to cleanly isolate the contribution of the proposed robust learning mechanisms.

- Analyze the accuracy of the pair division (precision/recall of S_U, S_I, S_C vs. ground-truth noise labels) to validate the reliability estimation mechanism directly.

- Test with semantically motivated noise injection (e.g., replacing entity images with visually similar but incorrect ones via CLIP retrieval).

## Removed Points

- **"The paper claims existing methods assume faultless correspondences, but some cited works handle noisy/missing modalities."** Removed — the paper specifically distinguishes between prior works that handle a single level of noise (e.g., missing modalities) and the DNC setting which addresses noise at both intra-entity and inter-graph levels simultaneously. This is a reasonable scope distinction, not an overstatement.

- **"No standard deviations or multiple runs are reported."** Removed — single-run evaluation with fixed seeds is the norm in MMEA and entity alignment benchmarks. Demanding confidence intervals for large-scale benchmarks is a generic nicety, not a core flaw.

- **"Theoretical pieces (Theorem 1) are tautological/not operationally useful."** Partially removed — while Theorem 1 is straightforward (low uncertainty does not guarantee correct correspondence), it serves a valid motivational purpose for introducing the consensus principle. The criticism that it doesn't provide a guarantee for the consensus definition is valid and retained in Minor weaknesses.

- **"Missing related works from adjacent domains."** Removed as a hard rule violation — cannot confirm the existence of specific uncited works. However, a generalized version (no comparison with noise-robust methods adapted to MMEA) is retained as a Minor weakness.

- **"The refinement in Eq. 12 for S_I could reinforce early-stage errors."** Weakened — this is a standard concern in self-training that applies to many noisy label methods. The paper mitigates this by excluding S_U (the highest-uncertainty pairs) entirely and using soft labels for S_I, which is a reasonable curriculum-style approach. Retained as part of the circularity concern rather than as a standalone weakness.

## Novel Insights

The paper reveals a practically important structural insight: in multi-modal entity alignment, noise at the entity-attribute level and entity-entity level are coupled — an incorrect entity-attribute pairing propagates to create incorrect inter-graph attribute-attribute correspondences. The proposed solution leverages this coupling by using inter-graph reliability to detect intra-entity noise (the DRF module), which is a clever design choice. However, the coupling also creates a vulnerability: if the inter-graph reliability estimation fails (e.g., under heavy noise), the intra-entity noise detection may also collapse. The paper does not analyze this failure mode, and the inter-graph reliability mechanism itself depends on noisy labels, creating the circularity concern identified above. The TTR module's use of an MLLM for test-time correspondence reasoning is creative but raises fairness and practicality questions that the paper does not adequately address.

## Suggestions

1. **Add a clean ablation without any MLLM** to isolate the contribution of the training-time robust learning components. Report "DRL + DRF only" on all five benchmarks.

2. **Experiment with smaller MLLMs** (e.g., Qwen2.5-VL-7B) and report cost/runtime to assess whether the TTR gains are inherent to the CoT reasoning design or simply a function of model scale.

3. **Report pair division accuracy** (precision/recall of S_U, S_I, S_C against ground-truth noise identity) to directly validate the reliability estimation mechanism.

4. **Add hyperparameter sensitivity analysis** for β (the threshold controlling the noise/clean partition) and λ (the trade-off parameter) across datasets and noise levels.

## Score and Decision

**Calibration comparison:**
- **Norton (8,8,8,8 oral):** Cleaner methodology for noisy correspondence, with no heavy external model dependency. RULE has a similarly important problem but with more serious methodological concerns.
- **OTGM (6,6,5,5 rejected):** Similar topic (noisy correspondence in graph matching), weaker methodology and results. RULE is clearly stronger.
- **R-EDL (6,6,8,8,8 spotlight):** Incremental improvement to evidential deep learning with solid experiments. RULE has a more novel problem formulation but with more unresolved concerns.
- **READ (8,8,8,8 poster):** Strong test-time adaptation work with clean experiments. RULE's test-time module is less principled due to MLLM dependency.
- **CvxcWCDX0h (3,3,5,3 rejected):** Weak noisy correspondence paper with marginal novelty. RULE is significantly stronger.

RULE addresses an important problem with a comprehensive framework and strong empirical results. However, the heavy MLLM dependency at test time without proper isolation of its contribution, the partial circularity in reliability estimation, and the under-validated Assumption 1 are substantive concerns that weaken the confidence in the core claims. The paper's contribution lies more in problem identification and the comprehensive framework than in the individual technical components, several of which have known parallels (evidential uncertainty, label smoothing, self-training). The empirical gains are impressive but cannot be cleanly attributed to the proposed robust learning mechanisms versus the MLLM's scale advantage.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>