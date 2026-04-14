## Summary
This paper proposes MILP-Evolve, an LLM-guided evolutionary framework that generates diverse MILP problem *classes* (rather than instances), and uses the resulting data to train GNN-based models that generalize across unseen MILP classes. The approach is evaluated on three tasks — integrality gap prediction, learning to branch, and a novel language-MILP contrastive alignment task — achieving large improvements on held-out synthetic classes and meaningful transfer gains on MIPLIB. The key empirical insight is that class diversity matters substantially more than data quantity.

---

## Strengths

- **Class-level vs. instance-level generation is a genuinely novel and important distinction.** Prior data augmentation approaches (VAEs, diffusion models) operate within a fixed class and cannot introduce new combinatorial structures. MILP-Evolve generates new *classes* via LLM-guided operators (add/delete/crossover/mutate), which is a conceptually distinct and well-motivated contribution. The comparison against Seed+VAE confirms this distinction empirically — VAE augmentation often performs no better or worse than seed-only training, while class-level diversity provides large gains.

- **The "diversity > quantity" finding is backed by controlled ablations.** Figure 4 explicitly controls total instance count while varying the number of classes (and vice versa), across all three tasks. The finding that scaling classes from tens to hundreds to thousands consistently improves performance while adding more instances per class has diminishing returns is a clear, actionable insight for the MILP learning community.

- **Large performance gains on the held-out synthetic test set are striking.** IG correlation improves from 0.10 (Seed) to 0.58 (Ours), 4-way language accuracy improves from ~37% to ~70%, and the fraction of instances solved to optimality under learning to branch improves from 49.59% to 70.90%. These are not marginal improvements.

- **MIPLIB transfer provides genuine evidence of out-of-distribution generalization.** MIPLIB is explicitly withheld from pretraining, making the transfer results (IG deviation 23.30% → 21.56%, IG correlation 0.54 → 0.59, 10-way language accuracy 72.71% → 75.57%) a credible test of the pretraining value, including the faster convergence shown in Figure 5.

- **The language-MILP contrastive task is a novel contribution within the field.** Aligning MILP instances (A,b,c matrices) with natural language descriptions is a new task definition, and the reported outperformance of GPT-4o at direct interpretation (mentioned in Appendix A.3.7) suggests the contrastive alignment approach captures something non-trivial.

---

## Weaknesses

### Fatal
None.

### Major

- **No comparison with Huang et al. (2024), the most directly relevant concurrent baseline.** The paper explicitly cites this work as a concurrent approach to multi-class MILP learning (training a joint model on 5 classes) and positions itself as extending it. Yet Huang et al. (2024) does not appear as a baseline in Tables 1–3. Without this comparison, it is impossible to assess whether the gains arise from MILP-Evolve's scale and diversity, from architectural improvements, or simply from training on more classes regardless of how they are generated. This is the single most important missing comparison.

- **No class-specific oracle baseline.** All baselines in the paper are multi-class models trained on fewer or different classes. There is no comparison against a model trained specifically for each test class. Without this, the cost of generality is unquantified — readers cannot tell whether the multi-class model trades significant per-class accuracy for breadth, or whether it approaches class-specific performance. This comparison defines whether the "foundation model approach" is practically worthwhile.

- **No zero-shot evaluation on MIPLIB.** All MIPLIB results report fine-tuning (pretraining on MILP-Evolve, then fine-tuning on MIPLIB train). The paper's pretraining framing implies that the pretrained representations should carry broadly useful MILP structure; however, zero-shot or frozen-feature performance is never reported. Fine-tuning results are a weaker test of whether the representation generalizes — they show better initialization, not direct transfer. For an ICLR paper framing this as a foundation model approach, this is a meaningful gap.

- **The language description generation process is opaque, raising potential circularity.** The paper does not explain how natural language descriptions are produced for the evolved MILP classes or for MIPLIB. If the same GPT-4o used to generate MILP code also generates the paired text descriptions, the contrastive alignment task may primarily measure consistency with the generator's internal representation rather than genuine semantic understanding. The description source needs to be clearly stated, and the MIPLIB contrastive results depend on what "meaningful description" means and whether these are instance-specific or category-level labels.

- **No uncertainty quantification on MIPLIB results.** For Table 3, several improvements are modest in absolute terms (e.g., 4-way accuracy 79.92% → 82.08%, 10-way 72.71% → 75.57%). Without confidence intervals or multiple runs, it is difficult to assess whether these differences are reliable. This is especially important since MIPLIB is the only real-world benchmark in the paper.

### Minor

- **MILP-Evolve generation statistics are absent.** The paper reports that "more than a thousand" classes are generated but does not report acceptance rates, the fraction rejected at each filtering stage, solve-time distributions, or structural diversity metrics (e.g., problem size distributions, constraint density). These statistics are necessary to assess whether the pipeline produces 1000 genuinely distinct, solver-relevant problems or a large fraction of trivially solvable or near-duplicate variants. Appendix A.1 apparently contains examples, but quantitative statistics are needed.

- **The "diversity beats quantity" ablation in Figure 4a has a potential confound.** The two curves (purple: vary instances per class, fixed class count; gray: vary class count, fixed total instances) are described, but when class count is reduced while total instances are fixed, the remaining classes receive more instances each — changing both diversity *and* per-class training density simultaneously. The paper should explicitly acknowledge this confound, even if the conclusion still holds directionally.

- **Branching is entirely unevaluated on real-world instances.** The paper explains that only 13 MIPLIB instances fall in the solvable range (20s–300s) — a valid practical explanation — but branching is the most solver-critical and commercially relevant task. Even a report on these 13 instances, with caveats, would be more informative than complete omission. Alternatively, an alternative real-world benchmark with suitable instances should be considered.

- **The "foundation model" framing in the abstract is ahead of the actual delivery.** Section 7 appropriately acknowledges that separate models are trained per task. The abstract and introduction claim "a single deep learning model" across problem classes, but this conflates cross-class generalization (delivered) with multi-task unification (not delivered). This requires clarification to avoid reader confusion, though it does not undermine the core contribution.

### Tiny

- The IG normalization formula involves division by |z_LP^0(x)|; behavior when the LP relaxation value is near zero should be addressed (even briefly), as this can produce unstable targets.
- The node subsampling strategy for the attention module is not described in the main text; this is needed for reproducibility.

---

## Nice-to-Haves

- A preliminary experiment training a single shared backbone across multiple tasks (even with task-specific heads) would directly substantiate the "foundation model" framing and strengthen the paper's thesis.
- Concrete examples of generated MILP class code (e.g., an evolved auction problem alongside its ancestor) would help readers assess whether evolution produces genuinely new combinatorial structures or superficial parameterizations.
- A compute cost table (LLM calls, total solve time for data generation, model training time) would help practitioners assess the practical barrier to replicating or extending this approach.
- A per-class breakdown of generalization performance, showing which types of MILP structures transfer well vs. poorly, would inform future directions and give a clearer picture of the approach's limits.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: MIPLIB gains described as "modest" / "not dramatic."** On the synthetic held-out test set the gains are very large (5.8x correlation improvement for IG, 1.92x language accuracy improvement). The MIPLIB absolute gains are smaller, consistent with transfer learning settings. Characterizing the paper's overall gains as modest is misleading.
- **Harsh Critic: Concerns about train/test leakage through evolution families.** Table 2 directly addresses this by using six entirely *unseen seed classes* as the test distribution. The critic largely ignores this experiment.
- **Harsh Critic: "No human or solver-expert validation of class quality."** The paper uses solver execution (feasibility, solve time, problem size) as an automated quality gate. Demanding formal expert audits is a non-standard requirement for this type of systems paper.
- **Harsh Critic: "The paper should cite more modality-general pretraining literature."** This is a stylistic/framing suggestion without bearing on the scientific contribution.
- **Harsh Critic: Architectural novelty is limited.** The GNN+attention architecture is correctly described as building on prior work. The paper's primary contribution is data-centric, not architectural; this is clearly stated and does not constitute a weakness if the experiments are valid.
- **Harsh Critic: "LLM-generated optimization code can be subtly nonsensical while still executable."** The paper runs the generated classes through a solver pipeline (feasibility check, solve time verification); valid and solver-executable code is a reasonable correctness criterion for this type of contribution.
- **Harsh Critic: Demanding confidence intervals for the large synthetic test set improvements.** Gains of 5.8x in IG correlation or from 37% to 70% in language accuracy do not require statistical significance testing to be meaningful.

---

## Novel Insights

Two observations across the three reviews warrant specific highlighting. First, the "diversity > quantity" finding — that scaling the number of training MILP *classes* matters far more than scaling the number of instances per class — is not just a result about this particular system but a potentially broadly important finding for the ML-for-combinatorial-optimization community, where most prior work implicitly treats data scaling as instance-level scaling. Second, the complete failure of VAE-based instance augmentation to improve generalization to unseen classes (often performing worse than seed-only training) is a concrete empirical finding that challenges the utility of instance-level generative approaches for cross-class transfer and motivates the class-level generation paradigm.

---

## Suggestions

1. **Add Huang et al. (2024) as an explicit baseline** in Tables 1–3. Even if code is unavailable, reproduce it using published details or include an honest discussion of why exact comparison is not possible, with qualitative comparison of the two approaches on common benchmarks.

2. **Add class-specific oracle baselines** (train on 100+ instances from each test class individually). Report the gap between the oracle and the multi-class model to quantify the cost of generalization.

3. **Report zero-shot MIPLIB evaluation** (pretrained model, no fine-tuning) alongside fine-tuning results. Even poor zero-shot results are informative about what the pretraining actually captures.

4. **Clarify the text description pipeline** for both MILP-Evolve classes and MIPLIB instances: who generates the descriptions, whether they are instance-specific or class-level, and whether any human validation was performed.

5. **Report MILP-Evolve filtering statistics**: total classes generated, fraction passing each filter, distribution of solve times, and whether any manual review was conducted on a sample.

6. **Perform learning-to-branch evaluation on at least the 13 available MIPLIB instances**, with appropriate caveats about statistical power. Alternatively, identify a supplementary real-world benchmark with more compatible instance sizes.