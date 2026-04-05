=== CALIBRATION EXAMPLE 34 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is directionally accurate: the paper is about imposing an ordering on SAE latents to improve feature consistency. However, “Enforcing Orderedness in SAEs to Improve Feature Consistency” is a bit broader than the actual technical contribution, which is closer to a nested-dropout-style ordered Top- m SAE and a specific consistency story.
- The abstract clearly states the problem, method, and high-level empirical outcome.
- The main concern is that the abstract’s theoretical claim is stronger than what the paper actually establishes in a general sense. It says OSAEs “resolve permutation non-identifiability in settings of sparse dictionary learning where solutions are unique (up to natural symmetries),” but the theorem in Section 3 requires strong extra assumptions: nonnegative codes, a pre-existing ground-truth ordering by support frequency, and a noiseless model. That is a narrower statement than the abstract suggests.
- The abstract’s empirical claim is appropriately hedged (“can help improve consistency”), which is good, but it does not mention the tradeoff in reconstruction loss that the results section later makes clear.

### Introduction & Motivation
- The motivation is strong and relevant to ICLR standards: reproducibility and feature consistency in SAEs is a real issue, and the paper frames an important gap between permutation-invariant sparse recovery and ordered/canonical feature recovery.
- The prior work discussion is useful, but the novelty relative to Matryoshka SAEs and nested dropout could be made sharper. Right now the paper’s central distinction seems to be “deterministic use of every prefix/dimension” and “strict ordering,” but the introduction does not fully clarify why Matryoshka-style sampled prefixes are insufficient beyond the intuitive exchangeability point.
- The contributions are mostly stated accurately, though the introduction slightly over-claims the scope of the theory by implying a general identifiability result, when the theorem is limited to a carefully constructed setting.
- A notable issue is that the introduction implicitly positions orderedness as a canonical mechanism for interpretability, but later results show the ordering is often tied to frequency/Zipf priors rather than a general semantic notion of feature abstraction. That distinction matters for ICLR reviewers evaluating conceptual contribution.

### Method / Approach
- The method is not fully or consistently described. The core objective is spread across Sections 2.4–2.6, but the notation is messy enough that it is hard to reconstruct the exact training objective unambiguously from the text alone.
- The paper appears to define an \(\ell\)-prefix reconstruction loss and then average it over prefixes, analogously to nested dropout. That is clear at a high level. However, there are logical ambiguities:
  - In 2.4–2.6, the loss is written for “Top-m” codes and prefix masks, but it is not fully explained how the encoder is trained with a hard Top- m operator in practice, especially since the OSAE is described as “deterministically using every feature dimension.”
  - It is unclear whether the encoder output is trained jointly under all prefixes or whether there are any special schedules/curricula beyond the warmup and unit sweeping mentioned later.
- The theoretical section is the main weakness. The key theorem in Section 3 claims exact ordered recovery under spark assumptions, but the proof is incomplete and relies on a very strong ordering prior:
  - The theorem assumes the true atoms are already ordered by support frequency.
  - It assumes all codes are nonnegative to remove sign ambiguity.
  - It essentially uses a perturbation argument to show the last atom must be recovered, then appeals to a uniqueness result, but the argument is only sketched and not fully formalized.
- The lemma “Any minimiser of \(L_{ND}\) also minimises the full-prefix loss \(L_K\)” is especially suspect as written. The proof sketch in Appendix A is hard to parse and seems to rely on perturbing only the final atom while leaving earlier prefixes unchanged. This may be plausible, but the conditions under which this argument is valid are not established clearly enough.
- There is also a conceptual gap: the theorem proves recovery of the specified ordered ground-truth dictionary in a very stylized synthetic setting, but the paper’s main empirical setting is trained on LLM activations, where the “true” ordering is not known and the ordering prior is not obviously meaningful.
- Failure modes are under-discussed. In particular, what happens when the frequency ordering is not well-defined, or when multiple features have similar support? Since the ordering prior is central, these edge cases matter.

### Experiments & Results
- The experiments do test the paper’s claims, but somewhat unevenly.
  - The toy Gaussian experiment directly tests ordered recovery and consistency under a controlled generative model.
  - The Gemma-2 2B and Pythia-70M experiments test consistency and orderedness in real SAE settings.
- That said, the empirical evaluation leaves important gaps relative to the paper’s claims:
  - The paper emphasizes consistency, but the main metrics are stability and the proposed orderedness metric. There is limited evidence connecting these metrics to downstream interpretability or mechanistic usefulness.
  - There is no ablation isolating the effect of ordered-prefix training from the effects of warmup, unit sweeping, or different prefix distributions. Since these are substantial training interventions, it is hard to attribute gains specifically to orderedness.
  - The comparison set is somewhat narrow. Matryoshka baselines and vanilla/top- m SAEs are relevant, but ICLR reviewers would likely expect stronger control baselines that also target identifiability or canonicalization, especially given the paper’s strong claims about resolving permutation ambiguity.
- The toy Gaussian table is interesting, but there is a puzzling result: OSAE improves reconstruction loss and consistency simultaneously relative to baselines, despite the authors later suggesting OSAE has a restricted solution class. This deserves deeper explanation, because it raises questions about whether the objective is genuinely more effective or whether the optimization path simply happened to be easier in that synthetic setup.
- Error bars are reported in Table 1 and some figures, which is good. However, the statistical reporting is not consistently strong:
  - Some plots appear to use only a small number of seed pairs.
  - Several appendix figures are explicitly based on a single seed pair, which is not sufficient for supporting broader claims.
  - The paper does not provide significance tests or confidence intervals for all headline comparisons.
- The orderedness metric is novel, but its validity is not fully established experimentally:
  - It is designed to capture agreement in latent ordering after Hungarian matching.
  - But since orderedness depends on a prior notion of feature rank, the metric can be high even when the learned ordering is only weakly semantically meaningful.
  - The paper would benefit from an ablation showing that higher orderedness predicts better transfer, stitching outcomes, or interpretability beyond agreement with a frequency prior.
- Datasets and metrics are broadly appropriate for the stated goal, but the empirical claims would be stronger if the paper demonstrated robustness across more model layers, more widths, and more than two LLMs.

### Writing & Clarity
- The conceptual framing is clear, but the paper is difficult to follow in the method/theory sections because the notation and claims are not always aligned.
- The biggest clarity issue is the relationship among OSAE, nested dropout, Matryoshka SAEs, Top- m SAEs, and the prefix loss. A reader has to work hard to determine exactly what is new versus what is inherited from prior work.
- The theoretical section is especially hard to parse. The theorem statement, proof sketch, and supporting lemma are not presented in a way that makes the logic easy to verify, which is a serious issue for an ICLR paper making a formal recovery claim.
- The results section is also sometimes hard to interpret because some claims are embedded in narrative prose rather than clearly tied to specific plots or tables. For example, the discussion of “crossover points” in Figure 2 and later claims about changes over training would be much more persuasive if summarized more systematically.
- The figures and tables do convey the main qualitative trends, but some appendix figures are too dense to support the claims they are meant to illustrate. The parser artifacts in the provided text are not relevant, but even ignoring those, the paper seems to rely heavily on visual inspection rather than tightly quantified summaries.

### Limitations & Broader Impact
- The limitations section is present and reasonably honest. It acknowledges that the theory relies on idealized assumptions, that misspecified ordering can over-regularize, and that compute cost is higher.
- However, the paper misses several important limitations:
  - The proposed ordering prior is not obviously universal; it is closely tied to frequency/Zipf structure in the toy model and to a particular hierarchical intuition in Matryoshka-style features. That makes general applicability uncertain.
  - The method may entrench arbitrary or dataset-specific ordering conventions rather than reveal intrinsic structure.
  - The paper does not sufficiently discuss whether orderedness could reduce the diversity of recovered features or suppress rare but important features, beyond a brief mention of over-regularization.
- Broader impact is not a major issue here, but there is a potential negative impact in over-interpreting ordered latents as more “canonical” or “truthful” than they are. In mechanistic interpretability, enforcing an order can improve reproducibility while also biasing the feature basis in ways that may hide alternative valid decompositions.

### Overall Assessment
This paper has a promising and ICLR-relevant idea: making SAE features deterministic and ordered to improve consistency across runs. The empirical results suggest the idea can help, especially on ordered or frequency-skewed settings, and the connection to nested dropout is intellectually appealing. However, the paper currently overstates the generality of its theoretical guarantees, and the method/experiment story is not yet clean enough to fully isolate the contribution of orderedness from other training choices like warmup and unit sweeping. For ICLR, I would view this as an interesting but still somewhat preliminary contribution: potentially worthwhile if the authors substantially tighten the theory, clarify the objective and training procedure, and add ablations demonstrating that orderedness itself—not just extra training machinery—is responsible for the gains.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes Ordered Sparse Autoencoders (OSAE), a variant of Matryoshka SAEs that enforces a strict deterministic ordering over latent features by treating each latent dimension as its own prefix. The paper’s main claim is that this ordering reduces permutation ambiguity and improves feature consistency/reproducibility across seeds, with theoretical support in a sparse dictionary-learning setting and empirical results on toy data plus Gemma-2B and Pythia-70M activations.

### Strengths
1. **Clear and timely problem motivation: feature inconsistency in SAEs.**  
   The paper targets an important issue for mechanistic interpretability: learned SAE features vary across seeds and hyperparameters, undermining reproducibility and cross-run comparisons. This is highly aligned with current ICLR interest in reliable representation learning and interpretability.

2. **A plausible and simple intervention: deterministic prefix ordering.**  
   OSAE’s core idea is conceptually straightforward—replace sampled nested groups with deterministic prefixes over all dimensions. This is appealing because it directly addresses the exchangeability of latent units in existing nested/Matryoshka-style SAEs.

3. **Theoretical attempt to formalize ordered recovery.**  
   The paper gives a recovery theorem under a spark/uniqueness condition and a ground-truth ordering prior. Even if idealized, this is valuable because it connects the method to classical dictionary-learning identifiability results rather than being purely empirical.

4. **Evaluation goes beyond one benchmark and includes cross-dataset tests.**  
   The authors evaluate on both Gemma-2B and Pythia-70M, and they also test same-dataset and cross-dataset consistency (Pile vs. Dolma). That is stronger than a single-model demonstration and is relevant to ICLR standards for robustness.

5. **The paper introduces an orderedness metric in addition to stability.**  
   Since stability alone is permutation-invariant, adding an orderedness metric is useful for explicitly measuring whether a method recovers features in a consistent order. This is a reasonable diagnostic for the claimed contribution.

### Weaknesses
1. **The empirical evidence is not yet strong enough for the claim of improving consistency broadly.**  
   On the toy Gaussian model, OSAE improves both stability and orderedness, but the paper also reports lower reconstruction error than baselines in that setting, which is surprising given the added constraints and raises questions about tuning or comparability. On real models, the paper admits OSAE often has worse reconstruction loss, and the gains are mostly in orderedness, while stability can be mixed or lower for later features.

2. **The method may trade one form of symmetry breaking for another without clearly proving practical usefulness.**  
   Deterministic ordering can reduce permutation ambiguity, but it also imposes a strong inductive bias that may simply force arbitrary ranking rather than recover a meaningful canonical basis in real LLM activations. The paper itself notes that the imposed order may be misspecified and can over-regularize, which is a serious concern for interpretability claims.

3. **The theoretical result is heavily idealized and relies on strong assumptions.**  
   The recovery theorem assumes nonnegative codes, exact sparsity, a strict ground-truth frequency ordering, and spark-based uniqueness. These conditions are far from the noisy, approximate, and often non-identifiable regimes where SAEs are used in practice, so the theorem supports the mechanism only in a narrow setting.

4. **Experimental rigor and reproducibility are somewhat limited for an ICLR standard.**  
   The paper reports several experiments, but the setup seems under-specified in places: hyperparameter sweeps are described only at a high level, some comparisons are based on small numbers of seeds, and certain figures suggest results are averaged over limited runs. For a method claim about consistency, ICLR would expect stronger ablations and more systematic reporting.

5. **The comparison to Matryoshka baselines is not fully decisive.**  
   The paper compares to fixed and random Matryoshka SAEs, but it is not fully clear that these baselines are optimally tuned or matched in budget, especially given the added compute and schedule changes in OSAE. Because OSAE uses unit sweeping and warmup in some settings but not others, it is hard to isolate the effect of “ordering” from training heuristics.

6. **The paper’s significance is narrower than the abstract may imply.**  
   The main practical benefit shown is improved orderedness, with some stability gains in early prefixes. But the work does not yet demonstrate downstream interpretability benefits, better feature discovery quality, or improved task performance, which weakens its significance relative to ICLR’s usual expectation for a methods paper.

### Novelty & Significance
**Novelty: Moderate.** The idea of enforcing order in latent representations is not new in itself; nested dropout already established ordered representations, and Matryoshka SAEs introduced nested-prefix training in SAEs. The novelty here is the deterministic “every dimension is its own prefix” formulation applied to SAEs, along with the claim that this better resolves permutation ambiguity in feature learning.

**Significance: Moderate to limited.** The paper addresses an important interpretability problem, but the demonstrated gains are mainly in a new metric that the paper itself introduces, and the benefits on real-world models appear partial. Against ICLR standards, this feels like a promising incremental contribution rather than a clearly strong acceptance-level result unless the empirical and ablation story is substantially strengthened.

**Clarity: Mixed.** The high-level idea is understandable, and the motivation is well framed. However, the paper is hard to parse in places due to presentation issues in the extracted text, and some technical details of the objective, schedule, and evaluation are not fully crisply explained.

**Reproducibility: Moderate.** The paper gives model names, datasets, widths, and some training details, but the lack of fully specified hyperparameters, limited-seed reporting in some analyses, and dependence on training schedules like warmup and unit sweeping make exact reproduction nontrivial.

### Suggestions for Improvement
1. **Provide stronger ablation studies isolating the effect of ordering.**  
   Separate the impact of deterministic prefixes from unit sweeping, warmup schedules, prefix distributions, and any tuning differences. A clean ablation would help show that the gains come from orderedness itself rather than optimization heuristics.

2. **Evaluate on more meaningful interpretability outcomes.**  
   Beyond stability and orderedness, test whether OSAE improves feature semantics, human interpretability, stitching behavior under controlled protocols, or downstream probing/attribution tasks. This would better justify the method’s relevance.

3. **Strengthen the theoretical story with fewer idealizations or clearer scope.**  
   If possible, extend the analysis to signed coefficients, approximate sparsity, or noisy observations. If not, explicitly delineate the theorem as a narrow identifiability result and avoid implying broader generality.

4. **Report more comprehensive and statistically grounded empirical results.**  
   Use more seeds, confidence intervals, and matched compute budgets across baselines. Include sensitivity analyses over dictionary size, sparsity level, prefix distribution, and training length.

5. **Clarify the relationship to Matryoshka SAEs and nested dropout.**  
   The conceptual difference between “sampling a few group sizes” and “deterministically using every prefix” should be spelled out more formally, ideally with a direct comparison of objectives and optimization consequences.

6. **Justify the orderedness metric more carefully.**  
   Since orderedness is central to the paper’s claim, discuss when a high orderedness score corresponds to a genuinely meaningful canonical order versus an artifact of the imposed prior. Consider supplementing it with semantic or frequency-based analyses.

7. **Discuss practical trade-offs more explicitly.**  
   The method appears to increase compute and may reduce reconstruction quality in some settings. A concise cost–benefit analysis would help readers judge whether the consistency gains are worth the overhead.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add head-to-head comparisons against stronger SAE baselines that directly target consistency/identifiability, especially quasi-orthogonality/orthogonal SAE variants and recent stability-focused methods, not just Matryoshka and vanilla SAE. At ICLR, the claim that OSAE improves feature consistency is not convincing without testing against the most relevant prior art on the same models and sparsity regime.

2. Run a systematic ablation of the ordering mechanism: no ordering loss vs sampled-prefix Matryoshka vs fully deterministic prefixes vs nested dropout vs unit sweeping. Without isolating which component actually drives the gains, it is unclear whether the improvement comes from ordering itself or from the training heuristics.

3. Evaluate across multiple layers, dictionary widths, and sparsity levels on both Gemma and Pythia, not a single layer/configuration per model. The current evidence is too narrow to support a general claim about improving SAE consistency, especially under ICLR’s expectation of robustness across settings.

4. Add a reconstruction-quality comparison under matched hyperparameter budgets and matched compute, including training time and wall-clock cost. Since OSAE changes the objective and appears to use more compute, the paper needs to show the consistency gains are not simply purchased by extra optimization effort or a worse tradeoff hidden by selective checkpointing.

5. Include a controlled synthetic recovery study where the true ordering is absent or contradictory to the prefix prior. This is necessary to test whether OSAE genuinely learns canonical orderings or just overfits to a favorable generative assumption.

### Deeper Analysis Needed (top 3-5 only)
1. Prove or empirically verify that the reported “orderedness” is not just an artifact of the evaluation metric being biased toward prefix-preserving solutions. The paper needs analysis showing that the learned order has semantic meaning, not merely improved correlation under the metric definition.

2. Analyze failure cases where OSAE worsens later-feature stability or reconstruction. The results already suggest a tradeoff, but the paper does not characterize when the ordering prior becomes harmful, which is essential for judging whether the method is broadly useful.

3. Quantify sensitivity to the prefix distribution, warmup schedule, and unit sweeping. These are not minor details: the paper itself says performance is sensitive to them, so without sensitivity analysis the main empirical claim is under-supported.

4. Analyze whether consistency gains persist after controlling for matched reconstruction error. If OSAE’s improvements disappear when baselines are tuned to the same MSE/FIFR level, then the core claim becomes much weaker.

5. Provide a clearer theoretical boundary for the exact-recovery result: what assumptions are truly needed and which are merely technical? The current theorem relies on strong ordering, nonnegativity, and uniqueness conditions, so the paper needs to clarify how restrictive the guarantee is relative to realistic SAE training.

### Visualizations & Case Studies
1. Show side-by-side feature trajectories across seeds for the same semantic feature, including the top activations and nearest-neighbor matches over training. This would reveal whether OSAE truly stabilizes individual features or only improves aggregate scores.

2. Add a case study of matched features for early, middle, and late prefixes, with textual examples of top-activating contexts. The paper claims earlier features are more consistent, but does not show whether those features are actually interpretable or just more stable numerically.

3. Visualize permutation matrices and confusion maps across seeds for each method. If OSAE really enforces order, the matching structure should be visibly near-diagonal; if not, the “orderedness” gain is probably fragile.

4. Show failure examples where the imposed order conflicts with the data, including features that split, merge, or shift positions. This would expose whether the method is robust or just succeeds on easy, frequency-ordered cases.

### Obvious Next Steps
1. Test OSAE on more realistic interpretability benchmarks where feature identity matters, not just consistency metrics on language model activations. ICLR reviewers will expect evidence that the method improves downstream interpretability rather than only internal alignment scores.

2. Extend the method to other architectures and datasets beyond two language models, or justify why the effect should be model-agnostic. The current scope is too limited for a general method paper.

3. Provide a principled way to choose the ordering prior from data rather than assuming frequency/order is known. Without this, the method depends on an oracle-like ordering assumption that may not hold in practice.

4. Release an exact, reproducible training recipe with all schedules and hyperparameters fixed. Given the sensitivity to heuristics, reproducibility is part of the contribution and should be fully specified.

# Final Consolidated Review
## Summary
This paper proposes Ordered Sparse Autoencoders (OSAE), a Matryoshka-style SAE variant that enforces a strict deterministic ordering over latent dimensions by assigning each dimension its own prefix. The motivation is to reduce permutation ambiguity and improve feature consistency across seeds, with a small theoretical result in an idealized sparse dictionary-learning setting and empirical comparisons on synthetic data plus Gemma-2B and Pythia-70M activations.

## Strengths
- The paper targets a genuinely important problem for SAE-based interpretability: feature identity is unstable across seeds and training choices, which makes cross-run comparison and mechanistic claims brittle. Framing consistency as a first-class objective is timely and relevant.
- The core intervention is simple and conceptually coherent: replace sampled nested groups with deterministic prefixes over all dimensions. This is an intelligible way to tighten the symmetry class relative to Matryoshka SAEs, and the paper backs it with a recovery theorem in a sparse coding setting and a new orderedness metric.
- The experiments do show a consistent qualitative pattern: OSAE improves orderedness on the toy model and on Gemma/Pythia, and the cross-dataset setting suggests the effect is not limited to a single corpus. The paper also acknowledges tradeoffs, which makes the empirical story more credible than if it had only reported wins.

## Weaknesses
- The theoretical guarantee is much narrower than the paper’s framing suggests. The exact recovery result relies on noiseless data, nonnegative codes, a pre-existing frequency-based ordering, and spark-based uniqueness; this is a highly stylized setting and does not yet justify broad claims about resolving permutation non-identifiability in real SAE training.
- The empirical story does not cleanly isolate the effect of ordering itself. OSAE is trained with additional machinery such as warmup and, in the toy setting, unit sweeping; the paper does not provide the ablations needed to separate “ordered prefixes” from optimization heuristics or schedule choices. That makes the causal claim about ordering too weak.
- The improvement is mostly in the paper’s own orderedness metric, while stability and reconstruction trade off depending on the setting. In the real-model experiments, OSAE is often worse in reconstruction, and later-prefix features can lose stability; without stronger evidence that this translates into better interpretability or feature quality, the practical gain remains uncertain.

## Nice-to-Haves
- A matched-budget ablation against stronger consistency-focused SAE baselines, ideally including quasi-orthogonality or other recent identifiability-oriented methods, would make the comparison more convincing.
- A more systematic sensitivity analysis over prefix distributions, sparsity levels, widths, layers, and training duration would help establish whether the effect is robust or tuned to a narrow regime.
- It would be helpful to show whether higher orderedness predicts anything beyond the metric itself, such as better stitching behavior, more semantically coherent features, or improved downstream probing.

## Novel Insights
The main novelty is not simply “ordering” in the abstract, since ordered representations and nested dropout already exist, but rather the attempt to make ordering deterministic and exhaustive in the SAE setting: every latent dimension is treated as a distinct prefix, rather than sampling a handful of group sizes. That is a sensible way to break exchangeability more aggressively than Matryoshka SAEs, and the paper’s additional orderedness metric captures a dimension of reproducibility that permutation-invariant stability cannot. The limitation is that the strongest evidence comes from scenarios where a frequency/order prior is already built into the data, so the method currently looks more like a structured inductive bias for favorable regimes than a general solution to canonical feature recovery.

## Potentially Missed Related Work
- Quasi-orthogonality / orthogonal SAE variants — directly relevant because they also target feature consistency and identifiability, and would be a natural stronger baseline for this paper’s claims.
- Other recent stability-focused SAE methods — relevant for head-to-head comparison on the same models and sparsity regimes.
- None identified beyond the above as clearly central to the specific ordered-prefix contribution.

## Suggestions
- Add an ablation that cleanly separates deterministic ordering from warmup, unit sweeping, and other schedule choices.
- Compare against stronger identifiability/consistency baselines under matched compute and matched reconstruction budget.
- State the theorem’s scope more conservatively and clearly: it is an idealized recovery result under strong assumptions, not evidence of general canonical ordering in real LLM activations.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 4.0, 4.0]
Average score: 3.0
Binary outcome: Reject
