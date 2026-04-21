## Summary

The paper proposes CoLR, a two-stage framework for temporal knowledge graph reasoning that (1) constructs a Temporal Relation Structure Graph (TRSG) capturing cohesion-based dependencies between relations, (2) uses a time-fusion search algorithm guided by cohesion and temporal recency to extract logical paths (supplemented by a Path Supplement Strategy for disconnected quadruplets), and (3) encodes these paths jointly via a pre-trained language model and time sequence encoder. The paper also introduces three new datasets (ACLED2023, ACLED-IND, ICEWS14-FS) targeting transductive, inductive, and few-shot scenarios, reporting large improvements over baselines across seven datasets.

## Strengths

- **Path Supplement Strategy (PSS) addresses a critical failure mode and is empirically validated.** The ablation on ICEWS14 (Table 3) shows that removing PSS drops MRR from 75.72% to 63.23% — a 12.49% absolute decrease. This is the single most impactful component and validates the claim that handling path-missing quadruplets is essential for TKG reasoning.

- **TRSG visualization provides empirical support for the cohesion hypothesis.** Figure 4 shows consistent diagonal patterns (repetitive events) and similar cohesion distributions across the three ICEWS datasets despite different sizes, supporting the claim that TRSG captures stable, transferable relation structure.

- **Geographic partitioning for ACLED-IND is a thoughtful inductive split design.** Rather than random entity splits that can disrupt graph connectivity (as the paper argues in Section 6.1), ACLED-IND uses Asia for training and Europe/Americas for testing, maintaining structural integrity while ensuring entity-level separation (`E_train ∩ E_test = ∅`).

- **The two-stage framework is modular and clean.** Separating path extraction from path encoding allows each component to be improved independently, and the approach of combining PLM text encoding with time sequence encoding is reasonable.

## Weaknesses

### Fatal

None.

### Major

- **The inductive evaluation is unsupported in the main text.** The paper's central claim is that CoLR excels in inductive scenarios, yet Table 2 reports only CoLR's numbers on ACLED-IND — the only dataset with genuinely unseen test entities — with no baseline comparisons. The paper defers these comparisons to "Appendix C.4" (line 290), which is not available for verification. Without baseline comparisons on the only truly inductive dataset, the headline inductive claim is unsupported. The cross-dataset experiments in Table 2 (e.g., train on ICEWS14, test on ICEWS18) are transfer experiments, not inductive evaluations — the ICEWS datasets share entity and relation vocabularies from the same source database, so test entities are not unseen during training.

- **Baselines on new datasets use untuned ICEWS14 hyperparameters, systematically disadvantaging them.** The paper states (line 286): "For the proposed datasets, we conducted experiments for each baseline using their parameter settings on ICEWS14 and reported the results." Applying ICEWS14-tuned hyperparameters to a completely different data source (ACLED) without any tuning is not a fair comparison, especially when CoLR's own hyperparameters were presumably tuned per-dataset. This concern is amplified by the unusually large reported improvements (21–30% MRR on ICEWS datasets), which are an order of magnitude larger than typical advances in TKG reasoning (1–3%). While large improvements are not inherently impossible, they demand rigorous evaluation, which the untuned-baseline methodology does not provide.

- **The PLM confounds the inductive reasoning claim.** The model encodes entity names as text via ALBERT (Eq. 6: Γ(s), Γ(o)), which was pre-trained on vast geopolitical web text. When ACLED-IND tests on European/American entities unseen in the Asian training graph, the PLM already possesses rich semantic representations of "France," "NATO," etc. The model does not need to inductively generalize to these entities — it can leverage the PLM's pre-trained knowledge. The ACLED-IND MRR of 83.63 (higher than most transductive results) is consistent with this confound. No experiment isolates the PLM's contribution from the model's structural reasoning capability (e.g., by using random entity embeddings or a non-pretrained encoder on ACLED-IND).

### Minor

- **Cross-dataset anomaly unexplained.** In Table 2, training on ICEWS14 and testing on ICEWS18 yields MRR 74.42, which is *higher* than training and testing on ICEWS18 itself (68.74). Similarly, training on ICEWS14 and testing on ICEWS05-15 yields 78.69, exceeding in-domain training on ICEWS05-15 (76.82). These anomalies could indicate suboptimal hyperparameter tuning for specific datasets, evaluation protocol inconsistencies, or that the cross-dataset setting accidentally provides better training signal. The paper does not discuss these results.

- **Overclaimed "transferability" without evidence.** The introduction states (line 37): "The proposed CoLR framework demonstrates significant transferability to previous logical reasoning methods, effectively optimizing their learning efficiency and reasoning performance." No experiment in the paper shows applying CoLR's components (e.g., the TRSG or cohesion-based confidence function) to improve another method's performance. The cross-dataset results in Table 2 show that CoLR transfers across datasets, not that CoLR transfers to other methods. Similarly, the claim of a "novel confidence evaluation function for discrete logical reasoning methods" (line 37) is introduced but never demonstrated empirically.

- **Ablation conducted only on ICEWS14.** Given the massive performance differences across datasets (e.g., MRR ranging from 68.74 on ICEWS18 to 94.23 on YAGO), single-dataset ablation provides limited insight into whether the components contribute consistently. Multi-dataset ablation would strengthen the conclusions.

### Trivial

None.

## Nice-to-Haves

- A PLM ablation on ACLED-IND (replacing ALBERT with a randomly initialized encoder) would definitively establish whether the inductive performance comes from structural reasoning or pre-trained knowledge.
- Path-level case analysis showing concrete examples of extracted temporal paths, their textual encodings, and how they contribute to correct predictions — this would demonstrate interpretability, which is a claimed advantage of logical reasoning methods.
- Reporting what fraction of quadruplets have connected paths vs. rely on PSS, and the average path length distribution, would clarify whether the TFSG search or the PSS heuristic drives performance.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Cohesion is just normalized co-occurrence"**: The harsh critic argues the cohesion concept adds only conceptual framing, not technical substance. However, the specific formulation (temporal-windowed, entity-mediated, degree-normalized with Theorem 1 providing a closed-form) is a genuine computational contribution beyond simple co-occurrence. The linguistic analogy is framing, not the contribution itself. Moved to Removed Points as the criticism understates the technical content.

- **"P_time is recency bias, not temporal constraint enforcement"**: The search algorithm starts from o^{t_j} and proceeds toward s^{t_i} where t_i ≤ t_j (line 195), which inherently enforces temporal ordering along paths. P_time provides a preference for more recent edges within this constrained search, which is a reasonable design choice, not a flaw.

- **"Time2Vec notation inconsistency (w_t vs w_d)"**: This is a formatting/typographical issue — removed per rules about formatting artifacts.

- **"δ ∈ (0, 0.1) is ad hoc and unexplored"**: This is a minor hyperparameter concern that doesn't affect the core contribution. Moved to Removed Points as trivial.

- **"ACLED2023 doesn't truly prevent PLM information leakage"**: The critic argues ALBERT's general web training means it already knows geopolitical patterns. However, the paper's specific claim about ACLED2023 is that it prevents *temporal* leakage (events from 2023, after ALBERT's 2020 cutoff). The semantic leakage concern is valid but is a separate point (addressed above in the PLM confound weakness). The ACLED2023 design accomplishes what it claims — preventing direct memorization of 2023 events.

- **"ICEWS14-FS is not true few-shot learning"**: The paper uses "few-shot" to mean a sparser graph (10% of events per timestamp), not meta-learning over tasks. While the terminology could be more precise, the experimental setup is clearly described and tests a legitimate scenario (information-sparse environments). The critic's demand for meta-learning-style few-shot is scope creep.

- **"Max aggregation in Eq. 8 lets noisy paths dominate"**: This is speculative without evidence. The ablation shows the overall system works well. The critic does not demonstrate this is actually a problem.

- **"Missing appendix / missing proofs"**: Removed per rules — the parser strips appendices.

## Novel Insights

The most insightful observation across the reviews is that the paper's three evaluation concerns — the unsupported inductive claim (no baselines on ACLED-IND), the untuned baselines on new datasets, and the PLM confound — form a mutually reinforcing pattern: each concern individually could be explained away, but together they suggest that the paper's headline claims of inductive superiority and 21–30% improvements rest on evaluation methodology that systematically favors CoLR. The cross-dataset anomaly (where out-of-domain training outperforms in-domain training) further suggests that something unusual is happening in the experimental setup that deserves investigation. The PSS strategy, which is the paper's most impactful component (12.49% MRR drop when removed), is also its most practical contribution — it addresses a genuine, well-defined failure mode of path-based TKG methods that other work has overlooked.

## Suggestions

- **Run baselines with tuned hyperparameters on ACLED2023 and ACLED-IND.** This is the single most important revision. Even a subset of baselines (e.g., TLogic, ALRE-IR) with proper tuning on the new datasets would substantially strengthen or correct the reported gains.
- **Add a PLM ablation on ACLED-IND** using either randomly initialized embeddings or a frozen/non-pretrained encoder. If inductive performance remains strong, the claim stands; if it collapses, the paper should reframe its contribution around the complementary role of PLM semantics rather than structural inductive reasoning.
- **Discuss the cross-dataset anomaly** in Table 2 and provide an explanation (e.g., dataset size effects, hyperparameter sensitivity, or entity/relation overlap statistics).
- **Reframe the "transferability" claim** to match what the experiments actually demonstrate (cross-dataset generalization of CoLR itself), or add experiments showing CoLR's components improving other methods.

## Evaluation

**Originality**: The TRSG construction and PSS strategy are reasonable contributions, though the individual components (normalized co-occurrence, heuristic path search, PLM+GRU encoding) are incremental. The geographic partitioning for ACLED-IND is a thoughtful design choice. The "cohesion" framing from linguistics adds conceptual novelty but limited technical novelty beyond time-windowed co-occurrence counting.

**Importance of research question**: TKG reasoning and inductive generalization are important problems. The paper addresses real gaps (path-missing quadruplets, evaluation benchmark deficiencies).

**Claim support**: The headline claims (inductive superiority, 21–30% improvements) are undermined by the evaluation methodology gaps. The inductive claim lacks baseline comparisons on the only inductive dataset. The large improvements may be inflated by untuned baselines on new datasets. The PLM confound in the inductive setting is unaddressed.

**Experimental soundness**: Significant concerns — untuned baselines on new datasets, missing inductive comparisons, unexplained anomalies, and the PLM confound all reduce confidence in the reported results.

**Clarity**: The paper is generally well-structured with clear descriptions of the framework components. The two-stage design is easy to follow. Some claims in the introduction ("transferability to previous methods") overreach what the experiments show.

**Value to community**: The PSS strategy and new datasets could be valuable contributions if validated with fair comparisons. The ACLED-IND geographic partitioning approach is a useful design pattern for inductive TKG evaluation.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| RoG | /home/wg25r/review_agent/human_reviews/ZGNWW7xZ6Q.md | 7.5 | Novel KG+LLM reasoning framework with strong, well-supported results. CoLR is notably weaker: less rigorous evaluation and overclaimed results. |
| Rethinking Complex Queries | /home/wg25r/review_agent/human_reviews/1BmveEMNbG.md | 7.0 | Formal analysis with strong validation. CoLR lacks this level of rigor. |
| INFER | /home/wg25r/review_agent/human_reviews/ExHUtB2vnz.md | 5.5 | Most topically similar — also neural-symbolic TKG reasoning on ICEWS. Had limited novelty and only ICEWS, but fair evaluation. CoLR has more serious evaluation concerns. |
| wN9HBrNPSX | /home/wg25r/review_agent/human_reviews/wN9HBrNPSX.md | 5.0 | TKG incremental training, simple method, rejected. CoLR has more contributions but also more serious evaluation flaws. |
| Is Complex Query Answering Really Complex? | /home/wg25r/review_agent/human_reviews/2FMdrDp3zI.md | 4.5 | Benchmark critique with new datasets, rejected. Similar profile of useful datasets but limited experimental validation. |
| TKG-LM | /home/wg25r/review_agent/human_reviews/T0hhkuv8I0.md | 4.0 | TKG reasoning with LM, PLM confound concerns. Very similar concern pattern to CoLR. CoLR has more structural contributions but the same PLM confound issue. |
| CAB-KGC | /home/wg25r/review_agent/human_reviews/lBrLDC7qXF.md | 3.6 | BERT-based KGC with PLM knowledge confound and missing baselines. CoLR is somewhat stronger than this but shares similar weaknesses. |
| NPLL | /home/wg25r/review_agent/human_reviews/EGxgZzDODh.md | 3.0 | Suspiciously large improvements with near-identical methodology to prior work. CoLR is more novel but the evaluation concerns are similar in spirit. |
| RGMG | /home/wg25r/review_agent/human_reviews/d1zLRzhalF.md | 2.5 | Outdated baselines, limited novelty. CoLR is clearly better than this. |

CoLR sits between TKG-LM (4.0, similar PLM confound) and INFER (5.5, similar topic but fairer evaluation). CoLR has genuine contributions (PSS, TRSG, new datasets) that TKG-LM lacks, but its evaluation methodology issues (untuned baselines, missing inductive comparisons, PLM confound) are more severe than INFER's. The paper is somewhat stronger than CAB-KGC (3.6) given its more substantial contributions and thoughtful dataset design. I place it at 4.0 — below the borderline for acceptance because the core claims are not well-supported by the evaluation as presented, but with clear potential for improvement through fair baseline evaluation and PLM ablation.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>