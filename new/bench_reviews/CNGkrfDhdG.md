Now I have sufficient calibration. Let me compose the final review.

## Summary

The paper proposes CoLR, a two-stage framework for link forecasting over temporal knowledge graphs. In the first stage, a Temporal Relation Structure Graph (TRSG) captures cohesion between relations, and a time-fusion search algorithm uses both temporal proximity and cohesion weights to extract logical paths; when no path connects subject and object, a Path Supplement Strategy (PSS) samples historical edges. In the second stage, a PLM (ALBERT) encodes textual descriptions of paths while a Time2Vec+GRU encoder processes timestamp sequences. The paper also contributes three new datasets: ACLED2023 (addressing PLM information leakage), ACLED-IND (inductive), and ICEWS14-FS (few-shot).

## Strengths

- **TRSG formalization is clean and novel** (Section 4, Theorem 1, Eq. 2): Building a relation-level graph with temporally-weighted cohesion edges, then using it to guide path search via P_time + P_coh (Eqs. 4-5), is a conceptually sound and original contribution that avoids exhaustive random walks of prior symbolic methods.

- **New benchmark datasets address real evaluation gaps** (Section 6.1): ACLED2023 mitigates PLM information leakage by using 2023 events post-ALBERT's training cutoff; ACLED-IND enables entity-disjoint inductive evaluation while preserving graph connectivity; ICEWS14-FS enables few-shot testing. On ACLED2023, CoLR's improvement drops to 6.37% MRR over HGLS — more plausible than ICEWS improvements.

- **Two-stage modular architecture** (Section 5): The separation of path extraction (Stage 1) from path encoding (Stage 2) is portable and could be paired with different encoders, as the paper claims.

- **Comprehensive ablation study** (Table 3): Each component contributes meaningfully — PSS (−12.49 MRR), TSE (−6.77), RP (−4.43), TRSG (−2.97) — confirming that all proposed modules are necessary.

- **Strong cross-dataset transfer** (Table 2): CoLR trained on ICEWS14 and tested on ICEWS05-15 achieves 78.69 MRR, actually higher than its transductive result on ICEWS05-15 (76.82), demonstrating relation-level logical rules generalize across datasets with different entity sets.

## Weaknesses

### Fatal
None.

### Major

- **Baselines on new datasets use unoptimized hyperparameters, creating unfair comparison.** The paper explicitly states (Section 6.1): "For the proposed datasets, we conducted experiments for each baseline using their parameter settings on ICEWS14 and reported the results." While CoLR is tuned for each dataset, baselines run with ICEWS14 hyperparameters. On ACLED2023 — a fundamentally different dataset from a different source (ACLED vs. ICEWS) — this is particularly concerning, though the improvement margin is smaller there (6.37%). On ICEWS14-FS, the 18.08% gap is likely inflated by non-optimal baseline tuning. This doesn't invalidate the results (the gaps on YAGO are only 5.36% and on ACLED2023 6.37%), but the ICEWS14/18/05-15 numbers should be interpreted cautiously since those baseline numbers are taken from prior papers with possibly different filtering/evaluation protocols (CENET had to be re-run due to mismatch, suggesting others may also differ).

- **No PLM ablation isolates architectural contributions from pre-trained knowledge.** The ablation in Table 3 removes TRSG, TSE, PSS, and entity-level path encoding, but never replaces the PLM with a non-pretrained encoder. Since ALBERT encodes entity text like "Imran Khan; Make visit; China," its geopolitical pre-training knowledge gives CoLR an information advantage over entity-ID-based baselines. ACLED2023 partially addresses this (improvement drops to 6.37% vs. 21-30% on ICEWS), which actually supports the concern — a significant portion of the ICEWS gains likely comes from PLM pre-training knowledge. Without a PLM ablation, the relative contributions of TRSG/PSS/TSE vs. ALBERT's pre-existing knowledge remain unclear.

- **Inductive evaluation (Table 2) does not test inductive reasoning as conventionally defined.** Table 2 shows cross-dataset transfer (train on ICEWS14, test on ICEWS18), not entity-disjoint reasoning within the same domain. While the paper argues this is a valid test of rule transferability and shows strong results, the standard inductive setting in TKG reasoning requires testing on unseen entities within the same knowledge domain. The paper introduces ACLED-IND for this purpose but presents only CoLR's own score (MRR 83.63) with no baseline comparisons in the main paper, deferring these to "Appendix C.4." The core claim about inductive superiority is therefore not established by the presented evidence in the main text.

### Minor

- **Cohesion may conflate frequency with logical dependency.** The TRSG cohesion matrix $\hat{\mathbf{R}}_{coh}^\omega$ is essentially a normalized co-occurrence count with temporal weighting. High-frequency relations (e.g., "Make statement," "Criticize") will show high cohesion simply because they appear often, not because they share logical dependencies. The paper acknowledges high-frequency "conjunction" relations (Section 6.4) but does not analyze whether cohesion captures information beyond raw frequency. A frequency-normalized variant or analysis would strengthen the claim.

- **PSS's dominance in the ablation raises questions about the "logical reasoning" framing.** PSS accounts for the largest performance drop (−12.49 MRR), yet it samples a single historical *disconnected* edge from the neighborhood of s or o — not a path connecting s to o. The paper should report what fraction of queries require PSS and how performance splits between PSS and connected-path cases, to clarify what proportion of the method's success comes from actual logical path reasoning vs. contextual supplementation.

- **Equal weighting of P_time and P_coh** (Section 5.1): The combination $P_{next} = P_{time} + P_{coh}$ gives equal weight to temporal proximity and cohesion without justification. A sensitivity analysis or adaptive weighting would strengthen this design choice.

### Trivial
None.

## Nice-to-Haves

- A PLM ablation (randomly initialized encoder, frozen PLM, no-PLM) would resolve the most important open question about where the performance gains originate.
- Baseline comparisons on ACLED-IND in the main paper rather than deferred to appendix.
- Standard entity-disjoint inductive splits on existing ICEWS datasets for conventional inductive evaluation.
- Analysis of PSS usage frequency and per-category performance.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Baselines on existing datasets taken from prior papers without unified protocol"** — The harsh critic flags this as unfair, but taking baseline numbers from published work is standard practice, and the paper explicitly re-ran CENET when its protocol differed. While the concern about inconsistent filtering is legitimate in principle, there is no evidence other baselines used a fundamentally different protocol — only CENET was flagged. This is a minor concern, not a major one.

- **"The improvement margins are implausibly large"** — While the 20-30% gaps on ICEWS datasets are unusually large by TKG standards, this alone does not prove unfairness. The YAGO gap is only 5.36%, and on ACLED2023 it is 6.37%. The margins vary substantially across datasets and could reflect genuine methodological advantages. The implausibility argument is inference rather than evidence.

- **"ICEWS14 and ICEWS18 share a large fraction of entities"** — This claim from the harsh critic about cross-dataset inductive evaluation is stated without evidence. ICEWS14 covers 2014 and ICEWS18 covers 2015-2018; they share the same geopolitical entity vocabulary but likely have different entity sets depending on pre-processing. The paper's choice to evaluate cross-dataset transfer is a valid (if non-standard) evaluation of rule portability.

- **Strength claim: "Large empirical improvements" as a core strength** — This is weakened by the fairness concerns above. The improvements are real numbers but their interpretation is contested. Kept as supporting evidence in context but not listed as a standalone strength.

- **Criticism about missing missing appendix C.4** — The parser strips appendices. The appendix exists in the original submission; claiming it is "unavailable" is a parser artifact.

- **Criticism about algorithm details being deferred to Appendix A.1** — Same parser issue; appendix exists in the original submission.

- **Formatting/style nitpicks** about unclear phrasing in Section 4.3, etc. — Removed as formatting nitpicks.

## Novel Insights

The paper inadvertently reveals a fundamental tension in TKG logical reasoning: methods that claim to perform "logical reasoning over paths" may derive most of their performance from non-logical contextual cues (PSS contributes −12.49 MRR, vs. TRSG at −2.97). This suggests the field should more carefully distinguish between path-based logical reasoning and context-augmented reasoning, as the two serve different purposes and have different inductive properties. Additionally, the dramatic performance drop on ACLED2023 (6.37% improvement) vs. ICEWS (21-30%) provides rare empirical evidence for the degree to which PLM pre-training knowledge inflates TKG benchmark scores on temporally older datasets — useful data for the community.

## Suggestions

- Add a PLM ablation (at minimum: frozen PLM embeddings vs. randomly initialized encoder) to isolate architectural from pre-training contributions.
- Re-tune at least the strongest baselines on the new datasets (ACLED2023, ICEWS14-FS) and report both tuned and untuned numbers.
- Include ACLED-IND baseline comparisons in the main paper to substantiate inductive claims.
- Report the fraction of queries requiring PSS and split performance by PSS vs. non-PSS cases.

## Score and Decision

**Calibration anchors:**

- **High band (>7):** RoG (7.5, KG reasoning with LLMs, planning-retrieval framework, strong empirical + interpretable results); FIT (7.0, neural-symbolic reasoning on KGs); Deep TGC (7.33, temporal graph clustering). These papers had clean methodology, fair baselines, and sound experimental protocols.

- **Medium band (4-6):** INFER (5.5, neural-symbolic TKG reasoning — direct competitor, similar domain but accepted despite limited novelty and unclear motivation); TKG-LM (4.0, LM for TKG — flagged for overclaimed superiority and missing baselines); SimTeG (4.67, PLM for graph learning — rejected for limited novelty and being "just PLM fine-tuning"); GURRWHkPtx "LMs are Graph Learners" (5.5, rejected for overclaim and unfair comparison).

- **Low band (<3):** RGMG (2.5, KG reasoning with RL — rejected for outdated baselines and limited novelty); Verbalized Graph (2.0, LLM for graph — rejected for weak evaluation and insufficient baselines).

**Comparison:** CoLR is most comparable to INFER (5.5, accepted poster) — both are TKG extrapolation methods combining symbolic/structural reasoning with neural components. CoLR has stronger empirical results and more novel architectural ideas (TRSG, PSS) than INFER, but has more serious fairness concerns (untuned baselines on new datasets, no PLM ablation, inductive claims unsupported by main paper evidence). It is more substantive than TKG-LM (4.0) and SimTeG (4.67) — it has genuine architectural novelty beyond "just use PLM." However, the fairness/baseline issues and the lack of PLM ablation are more serious than the issues that pushed papers like "LMs are Graph Learners" (5.5) and SimTeG (4.67) into reject territory. The ACLED2023 result (6.37% improvement) suggests real methodological contribution once PLM leakage is controlled, which is more than the low-band papers can show. But on the legacy ICEWS datasets, the improvements are likely inflated, and the inductive claims are not properly supported. 

I place this paper at 5.0 — above the clearly weak papers (2-3 range) but below the threshold where we can be confident the claimed improvements are fully attributable to the proposed method. It has real contributions (TRSG, PSS, new datasets, cross-dataset transfer results) but the evaluation has enough fairness gaps that the headline numbers cannot be taken at face value.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>