Now I have all the information I need to write a thorough, calibrated review.

## Summary

This paper proposes a foundation model training approach for Mixed Integer Linear Programming (MILP) by training GNN-based models on diverse MILP problem classes generated via MILP-Evolve, an LLM-based evolutionary data generation pipeline. MILP-Evolve applies structured prompt operators to seed MILP classes to produce over a thousand new problem classes. The paper evaluates on three tasks—integrality gap prediction, learning to branch, and a novel language-MILP contrastive alignment—demonstrating improved generalization to held-out synthetic classes and transfer to MIPLIB benchmarks.

## Strengths

- **MILP-Evolve is a creative and genuinely novel data generation pipeline.** Using LLMs with structured evolutionary operators (add, crossover, mutate, delete constraints/variables) to generate *new MILP classes* rather than just instances within a class is a significant conceptual advance over prior instance-augmentation methods (VAE, parameter search). The t-SNE visualization (Figure 1b) qualitatively demonstrates expanded coverage of the problem space, and the pipeline design (modular class representation with Data/Optimization/Parameters functions, 10 prompt operator types) is well thought out.

- **Important insight: class diversity matters more than instance quantity.** Figure 4 provides clear experimental evidence that increasing the number of training classes while holding total instances constant improves performance, whereas reducing classes while maintaining instance count hurts performance. This is a valuable finding for the field that informs future data collection strategies.

- **Consistent out-of-domain improvements across three tasks.** Table 1 shows substantial gains on the held-out MILP-Evolve test set: 5.8× correlation improvement for IG prediction (0.10→0.58), 1.42× improvement in instances solved for branching (49.59%→70.90%), and 1.92× improvement for 4-way language alignment (37.21%→70.54%). These are not marginal improvements.

- **Positive transfer learning to MIPLIB.** Tables 2–3 and Figure 5 demonstrate that pretraining on MILP-Evolve data leads to better fine-tuning performance on MIPLIB and faster convergence, including on unseen seed classes. This provides evidence that the generated data captures useful structural diversity beyond the synthetic universe.

- **Novel task introduction (Language-MILP alignment).** The contrastive learning task mapping MILP instances to natural language descriptions is a novel and well-motivated formulation that could serve as a practical stepping stone toward MILP interpretability and text-conditioned optimization.

## Weaknesses

### Major

- **The "foundation model" framing significantly overclaims what the paper delivers.** The title, abstract, and introduction consistently frame this as progress toward "foundation models for MILP," drawing analogies to CLIP/GPT/BERT. In reality, the paper trains **three separate task-specific models** for three separate tasks. A foundation model, by standard definition, is a single model that serves as a base for multiple downstream tasks via fine-tuning or prompting. The authors acknowledge this in the conclusion ("we acknowledge that this work still trains separate models for each learning task"), but this concession is at odds with the much stronger framing throughout the paper. The actual contribution—multi-class training with diverse synthetic data—is meaningful but narrower than the framing suggests. This matters because it shapes reader expectations about generality, reuse, and the nature of the advance.

- **MILP-Evolve class diversity is claimed but poorly validated.** The paper's core premise is that MILP-Evolve produces genuinely diverse and non-redundant problem classes. However, the only quantitative evidence is a t-SNE of code embeddings (Figure 1b), which is a weak proxy for mathematical structural diversity. There is no analysis of structural properties across generated classes (e.g., distributions of constraint matrix sparsity, variable-to-constraint ratios, integrality gap distributions, constraint types, or presence of specific modeling constructs like Big-M, SOS, or symmetry breaking). Without such analysis, it is possible that many of the "1,000+" classes are near-duplicates with trivial syntactic variations, and the performance gains come from modest variation around a handful of archetypes rather than from learning broadly useful representations. The lack of generation yield statistics (how many LLM attempts failed, how many classes were filtered out) also limits assessment of scalability.

- **Evaluation on real-world (MIPLIB) data is incomplete for the most practically important task.** Learning to branch—the task most directly tied to solver acceleration—is entirely omitted from MIPLIB evaluation because only 13 instances met the 20–300s solve time criterion. While the authors acknowledge this limitation, the branching task is arguably the most impactful for real-world MILP solving, and its absence from the primary real-world benchmark weakens claims of practical impact. Alternative evaluation protocols (e.g., node count reduction, evaluating on all solvable instances regardless of time, or using different solve time bounds) could have been explored rather than dropping the task entirely.

- **Baseline comparisons conflate data diversity with architecture and scale.** The headline comparisons in Tables 1–3 compare "Ours" (trained on diverse MILP-Evolve data) against Seed, Seed+Param, and Seed+VAE baselines. While these are sensible data-oriented baselines, they do not isolate the effect of *data diversity* from other factors. Specifically: (1) the total number of training instances differs across conditions (MILP-Evolve has more classes and potentially more instances); (2) the attention mechanism is only evaluated for the "Ours" condition for IG and alignment tasks but not consistently across all baselines; (3) there is no comparison where a model with the same architecture and training budget is trained on an equally large set of conventional MILP instances (e.g., a scaled-up set of classical benchmark problems). The ablation in Figure 4 (varying classes within MILP-Evolve) partially addresses this, but it only shows that diversity within the LLM-generated distribution is beneficial—it does not show that LLM-generated diversity is superior to scaled conventional diversity.

### Minor

- **Attention mechanism analysis is thin.** The paper states that incorporating attention "improves performance, especially for transfer learning to the MIPLIB dataset" but provides no deeper analysis—no attention visualization, no ablation on subsampling ratios or number of heads, and no explanation of *why* attention specifically helps transfer. The improvement from "Ours - Attn." to "Ours" in Table 1a (deviation 20.82%→20.14%) is modest, while the Table 1b language alignment improvement is negligible (70.41%→70.54% for 4-way). The stronger transfer claim for MIPLIB is not directly supported by a matched architecture comparison.

- **Language-MILP task evaluation is narrow.** The 4-way and 10-way classification accuracy metrics are relatively simple retrieval tasks that do not fully assess alignment quality. The practical utility of this task is asserted ("lowers the entry barrier for non-experts") but not demonstrated—there is no user study, no retrieval evaluation beyond small candidate pools, and no analysis of what the embeddings capture. The descriptions used for alignment appear to be synthetically generated (since MILP-Evolve generates both instances and their descriptions), which further limits claims about real-world interpretability.

### Trivial

- None significant beyond the minor points above.

## Nice-to-Haves

- A single unified model handling all three tasks (or at least a demonstration that shared representations benefit multi-task learning) would genuinely support the foundation model framing.

- More detailed structural analysis of MILP-Evolve classes (constraint types, sparsity patterns, integrality gap distributions, Big-M usage) to validate genuine diversity beyond syntactic variation.

- Performance stratified by problem category or evolutionary depth on the held-out test set, to reveal where the model succeeds versus fails and whether evolutionary depth correlates with generalization difficulty.

- Comparison with Huang et al. (2024), the concurrent multi-class MILP work mentioned in the paper, to directly benchmark against the most relevant prior approach.

- Reporting of MILP-Evolve generation yield (number of LLM calls, rejection rates at each filter stage, and approximate API cost) to help practitioners assess feasibility and scalability.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Claim that GPT-4o or other models/tools are not independently verifiable.** The paper cites GPT-4o as the LLM used in MILP-Evolve; by our review rules, we treat cited tools as existing and available.

- **Demand for comparison against task-specific SOTA baselines (e.g., Gasse et al. 2019 per-class models).** This is an unfair comparison demand that would test a different claim. The paper's contribution is about multi-class generalization, not beating per-class specialized models. A "foundation model" is expected to trade off per-class performance for breadth.

- **Demand for confidence intervals/variance estimates.** For large-scale benchmark evaluations in this community, single-run evaluation following established protocols (e.g., Gasse et al.) is the norm. Requesting confidence intervals for these experiments is a nice-to-have rather than a core flaw.

- **Request for GAT/GraphSAGE architecture comparisons.** The GCN-with-attention architecture follows established precedent in MILP learning (Gasse et al., 2019; Scavuzzo et al., 2022). Requesting alternative GNN architectures is a generic suggestion not tied to a specific weakness in the paper's claims.

- **Concern about "contamination" between MILP-Evolve and MIPLIB via LLM pretraining.** This is speculative and not supported by evidence. MILP-Evolve generates MILP classes via structured code evolution, not by retrieving specific instances. There is no mechanism established by which GPT-4o's pretraining would cause generated classes to resemble specific MIPLIB instances in a way that invalidates the transfer experiments.

- **Demand for the paper to compare against "any alternative large MILP corpus" in transfer experiments.** This asks the paper to evaluate against a benchmark that, as the paper documents, does not exist in the literature (existing datasets lack diversity and volume, which is precisely the problem MILP-Evolve addresses).

- **Concern about LLM data contamination inflating transfer results.** Without specific evidence that generated classes reproduce or closely approximate particular MIPLIB instances, this remains speculation and is not a grounded criticism.

## Novel Insights

The most genuinely novel insight in this paper is that *class-level diversity matters substantially more than instance-level quantity* for multi-class MILP learning (Figure 4). This has direct implications for how the ML-for-MILP community should prioritize data collection efforts: investing in generating diverse problem structures rather than scaling existing benchmarks with more instances per class. The LLM-based evolutionary approach to generating *new problem classes* (rather than just new instances) is also a methodological innovation that could influence data generation practices in combinatorial optimization more broadly.

## Suggestions

- **Reframe the contribution honestly.** Replace "towards foundation models" framing with language about "multi-class pretraining with diverse synthetic data" or "diversity-driven training for generalizable MILP learning." This would align the paper's claims with its actual contributions and avoid the gap between aspirational framing and delivered results.

- **Add structural diversity analysis of MILP-Evolve classes.** Report distributions of key MILP structural properties (constraint-to-variable ratios, sparsity patterns, integrality gap ranges, presence of Big-M constraints) across generated classes. This would substantiate the diversity claim far more convincingly than a t-SNE of code embeddings.

- **Explore branching evaluation on MIPLIB with alternative metrics.** Report node count reduction, primal bound improvement, or other solver-agnostic metrics even for instances with extreme solve times, rather than omitting the task entirely.

- **Report generation statistics.** Number of LLM calls, filter rejection rates at each stage, and total API cost would help practitioners assess the feasibility of adopting MILP-Evolve.

## Score and Decision

**Calibration comparison:**

- **Foundation Models for Boolean Logic (qeY25DwmKO)**: Score 6/8/3/5 (avg ~5.5), rejected. Similar conceptual framing (GNN foundation model for a logic/optimization domain, multi-task pretraining), but weaker in data diversity (only 3-SAT instances with 100 variables) and evaluation. Our paper is significantly stronger: more diverse data generation, three tasks, transfer to MIPLIB, and meaningful scaling experiments.

- **GOAL (z2z9suDRjw)**: Score 5/6/8/6 (avg ~6.25), accepted as poster. Generalist CO model with shared backbone + task-specific adapters, evaluated on multiple routing/scheduling problems. Similar novelty in framing, comparable empirical results. Our paper has a creative data generation contribution but lacks the unified model that GOAL provides.

- **Multi-Task VRP (DKfcxPxunu)**: Score 6/6/8/3 (avg ~5.75), rejected. Multi-task learning for VRPs with zero-shot generalization claims. Weaker methodological novelty than our paper and similar overclaiming issues about "unified" approaches without delivering a truly unified model.

- **Geometric GNN Pre-Training (4S2L519nIX)**: Score 6/8/6/6 (avg ~6.5), accepted as poster. Empirical study of GNN pretraining with scaling analysis and zero-shot transfer. Solid empirical contribution without methodological breakthroughs.

Our paper occupies a space between the rejected "Foundation Models for Boolean Logic" (weaker) and the accepted "GOAL" (stronger). The creative data generation pipeline (MILP-Evolve) and the diversity-over-quantity insight are genuine contributions. However, the overclaimed foundation model framing (three separate models, not one), the incomplete MIPLIB evaluation for branching, the insufficient validation of generated class diversity, and the baseline confounds are substantial weaknesses that cannot be fully mitigated by the interesting contributions. The paper is more honest about its limitations in the conclusion, but the main text oversells.

Overall assessment: **Originality is moderate-high (MILP-Evolve is creative), importance of research question is high (MILP generalization is a real bottleneck), claims are partially supported (empirical results are strong on synthetic data but the framing overclaims), experiments are sound but with significant gaps (no branching on MIPLIB, limited diversity validation), clarity is good.**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>