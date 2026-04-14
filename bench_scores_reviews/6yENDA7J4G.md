## Summary

This paper proposes a foundation-model training approach for Mixed Integer Linear Programming (MILP), centered on *MILP-Evolve*, an LLM-based evolutionary framework that generates over a thousand diverse MILP problem *classes* (not just instances) by applying prompt operators—add, delete, mutate, crossover, new—starting from eight seed classes. Models trained on this corpus are evaluated on three tasks: integrality gap (IG) prediction, learning to branch, and a novel language-MILP contrastive learning task. The central empirical finding is that class diversity matters substantially more than instance count, and that pretraining on *MILP-Evolve* data yields consistent gains over seed-class-only baselines including on MIPLIB transfer.

---

## Strengths

- **Class-level generation is a genuinely novel distinction from prior work.** Prior data augmentation for MILP (VAE-based, diffusion-based) generates more instances within fixed classes. Generating new problem *classes* via LLM evolution is a meaningful conceptual shift, and the contrast with Seed+VAE in Table 1 makes the point concretely: instance augmentation within seed classes is much less effective than class diversification.

- **The diversity-vs-quantity finding is a concrete scientific contribution.** Figure 4 controls total instance count while varying the number of training classes across all three tasks, providing a clean ablation. The consistent upward trend is compelling and challenges conventional quantity-first dataset scaling intuitions. This finding would be valuable to the broader ML-for-combinatorial-optimization community regardless of the other results.

- **Multi-task evaluation with consistent wins, including external transfer.** Table 1 shows *Ours* outperforming all baselines on all three tasks (IG prediction: 0.58 vs. 0.26 Pearson correlation; branching: 70.9% vs. 64.5% solved to optimal; language alignment: 70.54% 4-way vs. 39.68%). Table 3 further demonstrates that *MILP-Evolve* pretraining improves MIPLIB transfer for both IG and language tasks over seed-only and scratch baselines. The MIPLIB convergence curves in Figure 5 reinforce that the pretraining provides a genuinely useful initialization.

- **Attention ablation is informative.** The Ours-Attn. vs. Ours comparison in Table 1 and the observation that attention specifically helps for MIPLIB transfer (noted in Section 4) is a concrete, targeted ablation that informs the architecture design. Most papers in this area do not clearly isolate this.

- **Language-MILP contrastive learning opens a new direction.** Framing MILP understanding as a cross-modal alignment task (analogous to CLIP) is creative and provides a measurable target. Outperforming direct GPT-4o interpretation (noted in Section 5.1, with details in Appendix A.3.7) suggests the MILP encoder captures structure that raw text-based LLMs cannot.

---

## Weaknesses

### Fatal
None.

### Major

- **Train-test leakage risk through evolutionary lineage.** The paper splits the *MILP-Evolve* corpus 7:1:2 at the class level (Section 5.2), but if a test class is a near-direct descendant of a training-set class—sharing a parent, a prompt-chain ancestor, or a seed class—the held-out evaluation may overstate true out-of-distribution performance. The paper offers no analysis of whether the split was performed with respect to evolutionary families or lineage trees. This is critical: if test classes are minor mutations of training-class parents, the gains in Table 1 partly reflect in-distribution generalization within the generator's distribution, not genuine cross-class transfer. A lineage-aware split or at minimum an analysis of lineage depth distribution across splits is needed to validate the generalization claim.

- **Cannot isolate LLM structural novelty from scale.** The main thesis is that LLM-evolved class diversity drives improvement, but there is no non-LLM procedural class-generation baseline (e.g., random graph rewiring, random constraint template mutation, or combinatorial class enumeration without LLMs). The Seed+Param. baseline expands parameter diversity within fixed structural templates, but not structural diversity. Without a procedural-mutation baseline that diversifies constraint/variable topology without LLM involvement, it is impossible to determine whether the gains come from (a) LLM-generated structural novelty, (b) sheer variety from having over 1,000 varied templates of any kind, or (c) the real-world vocabulary injection and domain grounding that LLMs provide. This identification is central to the paper's contribution.

- **No direct comparison with Huang et al. (2024).** The paper explicitly frames Huang et al. (2024) as the closest concurrent work training a joint model on five MILP classes, and uses this as a primary motivation. Yet no numerical comparison appears in Tables 1–3. The paper's claim to go meaningfully beyond Huang et al. is asserted but not demonstrated. Even a qualitative performance comparison with that system's reported numbers would be informative.

- **Generated data quality and filtering statistics are absent from the main paper.** For a paper whose core contribution is a data generation pipeline, the absence of even basic statistics is a significant gap: What fraction of LLM-generated class code passes syntax filtering? What fraction produces feasible, non-trivial instances? What fraction is discarded for being too easy or too hard? How many classes remain after deduplication? Without these, the scalability and reliability of *MILP-Evolve* cannot be assessed, and the pipeline cannot be reproduced or replicated at scale.

- **Learning-to-branch is absent from MIPLIB evaluation.** Table 3 omits branching, the task with the most practical solver-acceleration relevance. The paper explains this by noting only 13 MIPLIB instances fall in the medium-difficulty window (Section 5.3), which is a legitimate constraint. However, this means the "foundation model for MILP" framing has no external benchmark support for its most impactful task. The paper should present this limitation more prominently and discuss whether alternative real-world benchmarks (e.g., ML4CO competition instances) could partially substitute.

### Minor

- **"Foundation model" framing overreaches the evidence.** The paper trains *separate* models for each task (explicitly acknowledged in Section 7), and external transfer requires fine-tuning rather than zero-shot generalization. The paper title says "towards," which is honest, but the body text sometimes conflates "multi-class pretraining" with "foundation model" in ways that mislead. Distinguishing these clearly (pretraining benefit vs. zero-shot generalization vs. unified architecture) would sharpen the paper's actual claims.

- **Language-MILP task evaluation is underspecified.** The 4-way and 10-way multiple-choice accuracy metrics (Table 1b) require clear documentation of how candidate negative descriptions are sampled: are they from the same structural family? From random classes? From closest classes in embedding space? If negatives are randomly sampled from the full class pool, the task may be doing coarse class identification rather than fine-grained semantic alignment. Additionally, whether the NV-Embed-v1 text encoder is frozen or fine-tuned is not stated, which materially affects what "MILP understanding" the model has learned versus what is carried by the pretrained language backbone.

- **Dataset statistics too vague.** "More than a thousand classes" (Sections 1 and 5.1) with no per-task breakdown, filtering rate, size distribution, or feasibility rate is insufficient for a data-centric contribution. The exact numbers vary by task (noted in Section 5.1) but are not reported, preventing readers from assessing data efficiency or fairness of baselines.

- **Selection bias in IG prediction from instance censoring.** Excluding instances unsolved within 200s (Section 5.1) biases the IG evaluation toward easier/medium instances. This means the trained model is evaluated on a censored distribution, and its practical utility on hard instances—which are exactly those where IG prediction would matter most—is untested.

- **MIPLIB filtering criteria affect transfer interpretation.** Section 5.3 filters to instances with "known optimal solution" or "meaningful description" without specifying the resulting test set size or coverage. The reported transfer gains in Table 3 could reflect biases from which MIPLIB instances are included.

### Tiny

- **Figure 4 scaling experiment could be more tightly documented.** The paper states instances are controlled across class-count conditions, but does not detail how many random subsets were averaged, whether class difficulty was balanced across conditions, or how class sampling was performed. The qualitative trend is visible but the experimental protocol needs documentation.

- **The IG normalized gap metric has a near-zero denominator risk** when the LP relaxation value is near zero. The paper defers this to Appendix A.2.1 but should at minimum state in the main text whether such instances were filtered and what share they represent.

---

## Nice-to-Haves

- **Ablation with non-LLM procedural class generation** (e.g., random constraint graph rewiring, combinatorial template expansion without LLMs) would clarify what the LLM specifically contributes beyond scale and variety.
- **Lineage-aware train/test splits** that hold out entire evolutionary subtrees would provide a cleaner assessment of out-of-distribution performance.
- **Per-class-type or per-structure-type breakdown of test performance** would reveal whether the model generalizes uniformly or only to structurally proximate classes. This is important for understanding the real reach of the approach.
- **Move the GPT-4o comparison to the main paper.** Since outperforming direct LLM interpretation is presented as a key argument for the language-MILP task, this result belongs in Table 1 rather than Appendix A.3.7.
- **Report MILP-Evolve pipeline efficiency**: total LLM token usage, wall-clock time to generate the dataset, and API cost, to allow practitioners to assess feasibility of replication.
- **A single multi-task model** (e.g., task-conditioned GNN) would validate the "foundation model" framing in a way the current architecture cannot; this is the natural next step the conclusion identifies.

---

## Removed Points

*These points are flagged for removal — treat with caution.*

- **"No comparison to handcrafted branching alternatives beyond SCIP Default"**: SCIP Default subsumes well-tuned heuristics including pseudocost branching and reliability branching; comparisons against SCIP Default are the norm in the learning-to-branch literature (Gasse et al. 2019 and successors). This is not a weakness.
- **"Unlimited number of instances phrasing is misleading"**: In context, this refers to the ability to sample fresh parameterized instances from any generated class indefinitely, which is accurate and standard.
- **"Mean baseline inflates rhetorical contrast"**: Reporting a constant-predictor baseline is standard practice and does not mislead; the relevant comparisons are against Seed+VAE and Ours.
- **"No confidence intervals across tables"**: Single-run evaluation is the norm for large-scale branching benchmarks and MILP solver comparisons in this community. This is not a methodological failure for the MILP-Evolve test sets (though for small MIPLIB test sets, some quantification of uncertainty would be useful — mentioned as a tiny/nice-to-have for MIPLIB specifically).
- **"No comparison against non-learning but stronger branching alternatives"**: The learning-to-branch literature benchmarks against SCIP Default and strong branching oracle; the paper follows this standard.
- **"Network overhead transparency"**: The paper already reports time improvement both with and without network overhead in Table 1c.
- **Strength: "well-written, topic is important, experiments are extensive"**: Removed as non-specific; specific strengths are retained above.

---

## Novel Insights

The most genuinely novel insight in this paper—distinct from and beyond the mechanism of MILP-Evolve itself—is the empirical demonstration that *class diversity dominates instance quantity* in MILP learning. The controlled Figure 4 experiments show that holding total training instances fixed and increasing the number of classes monotonically improves performance across all three tasks, while holding classes fixed and increasing instances yields minimal gain. This is a non-obvious finding that cuts against the conventional wisdom of scaling laws (more data = better models) and specifically motivates a paradigm shift from instance augmentation (VAE, diffusion) to class diversification as the primary lever for MILP generalization. This insight is likely transferable to other structured combinatorial problem domains where distribution shift across problem types is the primary challenge.

---

## Suggestions

1. **Perform and report a lineage-aware evaluation**: Split the *MILP-Evolve* corpus by evolutionary family (seed + all descendants), hold out entire families for testing, and report performance under this stricter split. Even one such experiment would substantially strengthen the generalization claim.

2. **Add a non-LLM procedural baseline** for class generation (random graph rewiring, constraint template mutation without LLMs) to isolate the specific contribution of LLM-generated structural novelty in Table 1.

3. **Report filtering statistics explicitly**: For reproducibility and to support the scalability claim, add a table or paragraph in the main paper reporting: number of LLM calls, pass rate at each filtering stage (syntax, feasibility, solve-time filter), and total valid classes generated per seed class.

4. **Clarify the language-MILP evaluation protocol**: State whether negatives in 4-way/10-way accuracy are randomly sampled or hardest nearest-neighbor classes, whether text descriptions were standardized from LLM outputs or manually written, and whether NV-Embed-v1 is frozen during contrastive training.

5. **Move the GPT-4o baseline to the main results table** (Table 1b) to support the key claim that the learned contrastive model outperforms direct LLM interpretation.

6. **Report MIPLIB test set size** explicitly in the main paper. Readers currently cannot assess the statistical reliability of Table 3 gains.

7. **Consider evaluating branching on alternative real-world benchmarks** (e.g., ML4CO competition datasets or harder procedurally generated instances) to provide at least one external generalization datapoint for the most practically impactful task.