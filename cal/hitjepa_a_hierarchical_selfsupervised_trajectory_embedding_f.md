=== CALIBRATION EXAMPLE 43 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate: the paper does propose a hierarchical JEPA-style framework for trajectory embedding. However, “for similarity computation” slightly understates that the paper’s core method is a representation-learning architecture evaluated on retrieval and fine-tuning tasks.
- The abstract clearly states the problem, the hierarchical idea, and the use of a JEPA-style predictive architecture. It is less clear on what is actually novel relative to T-JEPA: the abstract emphasizes “three-layer hierarchy” but does not distinguish whether the main contribution is architectural, objective-level, or empirical.
- Some claims are stronger than the evidence presented in the paper. In particular, “richer, multi-scale representations” and implied superiority are not yet substantiated in the abstract itself; the paper later reports mixed gains over T-JEPA, especially on Porto and some fine-tuning settings.

### Introduction & Motivation
- The problem is well-motivated: trajectory similarity computation does benefit from combining local point detail and global route semantics, and the introduction identifies the gap in single-scale embeddings and contrastive/self-supervised baselines.
- The paper does state contributions explicitly, but some are overstated. The claim that HiT-JEPA is “the first architecture to explicitly unify both fine-grained and abstract trajectory patterns within a single model” is too strong unless the authors carefully delimit prior work. The related work already discusses hierarchical and multi-granularity trajectory methods, and the paper should justify why those do not count.
- The introduction somewhat over-claims generality. It suggests a broadly applicable “unified framework” for urban trajectories, but the experiments are mostly on similarity computation with one pretraining source dataset used for zero-shot transfer to check-in and vessel data. The scope is narrower than the rhetoric.

### Method / Approach
- The method is not fully clear or reproducible from the main text alone. There are enough high-level ideas to understand the system, but key implementation details and the exact data flow between layers are ambiguous.
- A major concern is the hierarchical design in Section 3: the model creates three abstractions via Conv1D + max pooling, then applies JEPA at each level, then propagates attention from higher levels to lower levels. But the exact mechanism of “top-down spotlight” is only loosely specified. Eqs. (9)–(14) are hard to interpret as written, and it is not clear whether the propagated attention weights are used as soft masks, attention biases, or multiplicative gates.
- The sampling/masking scheme in the target and context branches is also under-specified. The paper says it randomly samples targets M times, uses a set of masking ratios, and applies successive masking with probability p, but it is unclear how the masks are generated for variable-length trajectories, how overlaps are handled in practice, and how the target/context masks differ across levels.
- There is a logical tension in the architecture: the paper claims to preserve both fine details and global semantics, yet the hierarchy is created by repeated convolution and pooling before the Transformer. That means the “point-level” representation is already transformed and pooled, so it is not obvious that level 1 truly preserves raw point-level nuance.
- The spatial tokenization step is also a potential source of loss. The paper uses H3 hex cells with node2vec embeddings, but does not discuss the effect of spatial quantization error or how sensitive results are to cell resolution beyond a single fixed choice per dataset.
- The loss combines SmoothL1 JEPA loss with VICReg regularization across all levels. This is reasonable, but the paper does not clearly justify why both context and target representations should be regularized separately, nor how the weights λ, μ, ν were selected beyond a tuning appendix.
- The theory is not the issue here; rather, the issue is that the method description leaves several operational details unclear enough that reproduction would be difficult without code.

### Experiments & Results
- The experiments do test the claimed setting of trajectory similarity computation, and they include retrieval/self-similarity, robustness perturbations, zero-shot transfer, downstream fine-tuning, and some visualization/ablation analyses. That is a strong experimental breadth.
- However, the evaluation is not perfectly aligned with the paper’s strongest claims. The main claim is hierarchical multi-scale representation learning, but the most central metrics are mean rank under synthetic query/database splits and perturbations. These do test robustness, but they do not directly isolate whether the hierarchy improves multi-scale semantic understanding.
- Baselines are generally appropriate for the core comparison: TrajCL, CLEAR, and T-JEPA are relevant. TrjSR is older and weaker, but its inclusion is acceptable as a historical baseline.
- A key issue is that the self-similarity evaluation uses a query/database construction where each query trajectory is split into odd/even points. This is a fairly artificial retrieval setting and may favor methods that are good at reconstructing dense patterns from subsampled variants. It does not fully reflect standard nearest-neighbor similarity search over independent trajectories.
- The tables appear to have parser corruption, but the underlying results still raise concerns. For example, Table 1 reports extremely small differences between HiT-JEPA and T-JEPA on Porto and T-Drive in many settings, yet the text emphasizes broad superiority. On Porto, T-JEPA is often better or essentially tied, and the paper acknowledges this only partially.
- The zero-shot results are stronger, but they also deserve more careful interpretation. The paper pretrains on Porto and transfers to TKY/NYC/AIS(AU), but it does not compare against any domain-adaptation or multi-city pretraining baseline, so it is hard to know whether the gains come from hierarchy or simply from the chosen pretraining regime.
- Table 2 is more convincing for fine-tuning, but the paper’s reporting is incomplete. It gives averages and relative improvements, but no variance, no statistical significance, and no indication of variability across runs. This is especially important because some improvements are small.
- The ablation study is helpful, but it is not sufficient. The most important ablation would be to separate the contribution of the hierarchy from the contribution of JEPA itself and from the additional VICReg regularization. The current variants mainly test interaction style, single-layer depth, and grid choice, but not whether gains come from “hierarchical” design versus extra capacity.
- There are also missing comparisons against simpler multi-scale baselines, such as a plain multi-layer Transformer with intermediate pooling or a multi-resolution encoder without JEPA. Those would materially affect the claim that the proposed hierarchical interaction is necessary.
- Error bars and statistical significance are absent throughout. For ICLR standards, that weakens confidence because many reported differences are modest.

### Writing & Clarity
- The main ideas are understandable, but several parts of the method section are difficult to follow, especially the equations around attention propagation and the masking/prediction pipeline.
- The experimental section is clearer conceptually than the method section, but it would benefit from a more concise explanation of what exactly is evaluated in self-similarity versus zero-shot transfer versus downstream fine-tuning.
- Figures 1, 2, 4, and 5 appear intended to support the hierarchy and interpretability claims. Conceptually they are useful, but the text over-interprets them. In particular, the “origin anchoring,” “pattern change,” and “destination intent” interpretation in Figure 4 seems more like a qualitative story than a validated finding.
- The paper would benefit from a tighter distinction between what is empirically demonstrated and what is interpretive speculation, especially in the attention visualization discussion.

### Limitations & Broader Impact
- The paper includes a limitations section, but it is quite minimal and not aligned with the actual technical limitations of the method.
- The most important limitations are not sufficiently acknowledged: dependence on spatial discretization, reliance on pretraining in a single source city for zero-shot transfer, and possible sensitivity to sampling/masking hyperparameters. The method may also struggle with trajectories whose semantics are not well captured by local-to-global hierarchy, such as highly irregular or multimodal mobility patterns.
- The broader impact statement is modest and reasonable in noting privacy-preserving tokenization. However, the paper does not discuss the possibility that learned mobility representations can be used for surveillance, re-identification, or sensitive behavioral inference. Given the application domain, this omission is notable.
- The claim that GPS is “blurred” by hexagonal cells is not a complete privacy guarantee; location sequences can still be identifying. This should be acknowledged more carefully.

### Overall Assessment
HiT-JEPA is a plausible and interesting extension of JEPA-style representation learning to trajectories, and the experimental package is substantial. The strongest positive is that the paper attempts to address a real weakness of prior trajectory similarity models: the inability to model multiple semantic scales in one framework. That said, for ICLR’s bar, the paper currently falls short on methodological clarity and on convincingly isolating novelty. The hierarchical interaction mechanism is not specified cleanly enough, and the empirical story relies on a somewhat artificial retrieval protocol, lacks statistical reporting, and does not include the most informative ablations or multi-scale baselines. The contribution is promising and likely useful, but the paper would need sharper methodological exposition and stronger evidence that the hierarchy itself—not just added model complexity—drives the gains.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes HiT-JEPA, a three-level self-supervised trajectory representation framework that combines point-, segment-, and trajectory-level abstractions using a JEPA-style prediction objective. The authors claim that hierarchical attention propagation between adjacent levels helps learn richer multi-scale embeddings for trajectory similarity computation, with gains shown on self-similarity retrieval, zero-shot transfer, and downstream fine-tuning across several urban and vessel trajectory datasets.

### Strengths
1. **Addresses a real and relevant limitation in trajectory representation learning.**  
   The paper identifies a plausible gap in existing methods: most prior trajectory SSL models learn a single-scale embedding and may miss complementary fine-grained and global semantics. This is well aligned with ICLR’s interest in representation learning methods that improve generalization and robustness.

2. **Clear high-level architectural idea: multi-scale hierarchical JEPA.**  
   The core idea of stacking three abstraction levels and propagating information top-down via attention-weight transfer is conceptually coherent. The method is motivated by the multi-scale nature of trajectories, and the authors explicitly connect the design to both JEPA and hierarchical learning literature.

3. **Empirical evaluation spans multiple datasets and task settings.**  
   The paper evaluates on Porto, T-Drive, GeoLife, TKY, NYC, and AIS(AU), and reports both self-similarity retrieval and zero-shot generalization, which is stronger than a single-dataset comparison. The downstream fine-tuning experiment further tests whether the learned embeddings are useful beyond nearest-neighbor retrieval.

4. **Ablation study supports some design choices.**  
   The ablation table compares hierarchical interaction, single-layer training, no-attention transfer, and rectangular tokenization. The reported degradation without these components suggests the hierarchical design contributes meaningfully, at least on the Porto setting used for the ablation.

5. **Implementation details and reproducibility materials are provided.**  
   The appendix includes dataset statistics, hyperparameters, masking ratios, and training settings, plus an anonymous code link. This is a positive sign for reproducibility, which ICLR values.

### Weaknesses
1. **The methodological novelty is incremental relative to T-JEPA and existing hierarchical learning ideas.**  
   The main extension over T-JEPA appears to be adding three abstraction levels and propagating attention weights across levels. While reasonable, the paper does not clearly demonstrate a fundamentally new learning principle; it is more of a structural adaptation than a substantially new self-supervised objective or theory. For ICLR, this may be viewed as modest novelty unless the hierarchical mechanism is convincingly shown to be broadly general and essential.

2. **The technical exposition is difficult to follow and sometimes under-specified.**  
   The method section leaves several important details unclear: how the convolutional abstractions interact with the Transformer encoders, exactly how the target masks are sampled at each level, and how the top-down attention is mathematically integrated into the next layer. The description of loss terms and attention propagation is not fully precise, making the approach hard to reproduce from the paper alone.

3. **Evidence for claims of “first” and strong superiority is not sufficiently convincing.**  
   The paper repeatedly claims that HiT-JEPA is the first to unify fine-grained and abstract trajectory patterns. This is a strong claim, but the related work and comparison baselines are limited. More importantly, the empirical gains over T-JEPA are often small, and in some settings the improvements are mixed or even negative. On Porto, the paper itself notes that HiT-JEPA is second best overall and sometimes below T-JEPA on average.

4. **Experimental reporting appears inconsistent in places.**  
   Several tables and narrative statements are hard to reconcile. For example, some results are duplicated in the extracted text, but even ignoring parsing artifacts, the reporting lacks clarity about variance, number of runs, and statistical significance. The paper often emphasizes best/worst cases without showing error bars or confidence intervals, which is important for ICLR-level empirical claims.

5. **The evaluation protocol may not fully isolate the contribution of hierarchy.**  
   Many gains are evaluated on retrieval/ranking tasks using mean rank, but it is unclear whether the hierarchical design alone drives the improvement or whether it mainly acts as extra capacity and domain-specific preprocessing. The paper does not include enough comparisons to non-hierarchical but similarly parameterized Transformer baselines, nor to stronger recent trajectory models beyond the listed four baselines.

6. **Generalization claims are broader than the evidence supports.**  
   The zero-shot evaluation is interesting, but it is conducted by pretraining on Porto and transferring to TKY/NYC/AIS(AU). This is useful but limited; the claim that the method generalizes across “heterogeneous urban and maritime datasets” is somewhat overstated given the single pretraining source and a relatively narrow set of tasks.

### Novelty & Significance
**Novelty: Moderate.** The paper combines known ingredients—JEPA-style prediction, hierarchical abstraction, convolutional downsampling, and attention-based information propagation—into a trajectory SSL framework. The combination is sensible, but the paper does not yet establish a clearly new learning paradigm or a surprising insight that would make the method stand out strongly at ICLR.

**Significance: Moderate.** Trajectory representation learning is a meaningful application area, and a robust multi-scale embedding method could be practically useful. However, the gains are incremental and the evidence for broad impact is limited by the focus on trajectory similarity retrieval/fine-tuning rather than a more general representation learning benchmark suite.

**Clarity: Below ICLR’s ideal bar.** The narrative motivation is clear, but the mathematical and algorithmic presentation is under-specified and at times hard to parse. ICLR reviewers typically expect a method paper to be especially precise.

**Reproducibility: Moderate.** The appendix includes useful hyperparameters and dataset details, and a code link is provided. Still, important implementation subtleties are not fully spelled out, and the paper does not clearly document run-to-run variability or all preprocessing decisions.

### Suggestions for Improvement
1. **Tighten the algorithmic description.**  
   Provide a clean pseudocode algorithm for training and inference, with explicit tensor shapes and a step-by-step description of how each level’s abstraction is formed and how attention is propagated across levels.

2. **Strengthen the empirical analysis with stronger baselines and fair capacity matching.**  
   Add comparisons against a non-hierarchical Transformer with similar parameter count, and if possible recent stronger trajectory foundation models or general sequence SSL baselines. This would help establish that gains come from hierarchy rather than scale.

3. **Report statistical robustness.**  
   Include mean and standard deviation over multiple random seeds for the main metrics, especially because many reported improvements over T-JEPA appear small. This is important for ICLR reviewers assessing reliability.

4. **Ablate each design choice more thoroughly.**  
   Separate the effects of: convolutional abstraction, multi-level JEPA, top-down attention fusion, VICReg regularization, and H3 tokenization. A more granular ablation would clarify what actually matters.

5. **Clarify novelty claims and position the work more carefully.**  
   Soften or precisely justify statements such as “first” unless supported by a strong literature survey. Explain more clearly how HiT-JEPA differs from prior hierarchical SSL methods in vision/NLP beyond applying a similar idea to trajectories.

6. **Broaden evaluation beyond retrieval-centric metrics.**  
   Since the method is a representation learner, consider adding more downstream tasks such as clustering quality, classification, or out-of-domain transfer under different pretraining sources. This would better support the claim of general-purpose semantic representations.

7. **Improve the discussion of generalization.**  
   Pretrain on multiple source cities or domains, not only Porto, to test whether hierarchy truly improves cross-domain robustness. This would make the zero-shot claims more persuasive.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add direct comparisons to stronger and more relevant baselines, especially recent trajectory foundation/pretraining models and non-trajectory-specific hierarchical/self-supervised sequence models. At ICLR, claiming a novel hierarchy “first” and “superior generalization” is not convincing without checking whether the gains persist against the strongest available representation learners, not just TrajCL/CLEAR/T-JEPA.

2. Add an ablation that isolates the benefit of the hierarchical interaction from simply adding more parameters/depth. The current variants do not cleanly separate “three levels,” “top-down attention fusion,” and “extra Transformer/Conv layers,” so the core claim that the hierarchy itself drives performance is not established.

3. Add experiments on trajectory classification/clustering/next-location or trip-purpose transfer, not only similarity retrieval and heuristic approximation. If the model is truly learning richer multi-scale semantics, it should transfer beyond ranking-based similarity search; otherwise the contribution is narrower than claimed.

4. Add comparisons under matched compute and model size, including parameter count, training cost, and inference latency per embedding. The paper claims efficiency and practical usability, but the reported gains are not interpreted under a fair efficiency-performance tradeoff.

5. Add robustness tests against missing GPS points, irregular timestamps, and domain shift across cities when training on multiple source cities, not only Porto-to-others zero-shot. ICLR reviewers will expect evidence that the representation is robust to the real failure modes motivating the hierarchy, not just to a specific synthetic downsampling/distortion protocol.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify what each hierarchy level actually encodes using probing or mutual information-style analysis. Right now the paper asserts point-/segment-/trajectory-level semantics, but does not show that level 1, 2, and 3 are meaningfully different rather than redundant compressions.

2. Analyze whether the top-down attention propagation is stable and causal, or merely a cosmetic visualization trick. You need evidence that propagated attention changes representation quality, not just the attention maps, because the method’s main novelty is this interaction mechanism.

3. Provide statistical significance across multiple seeds and confidence intervals for the reported gains. The margins over T-JEPA are often small, and without variance estimates the core improvement claims are not reliable enough for ICLR.

4. Analyze failure cases by trajectory type, length, sparsity, and geometric complexity. The paper claims robustness on irregular and sparse data, but it does not show when the method breaks, which is essential to trust the generalization narrative.

5. Clarify the effect of the H3 discretization choice on performance and generalization. Since the model depends on spatial tokenization, the representation quality may be driven by cell resolution rather than the hierarchy; this needs explicit analysis.

### Visualizations & Case Studies
1. Show side-by-side retrieval failures for HiT-JEPA vs. T-JEPA on hard queries. The current figures only show success cases, so they do not reveal whether the hierarchy actually fixes confusable trajectories or just performs better on easy examples.

2. Visualize embeddings or attention grouped by trajectory type and difficulty, not just generic t-SNE clusters. This would expose whether the model separates semantically meaningful behaviors or simply partitions trajectories by length, density, or region.

3. Add a layer-wise attribution case study showing how level 1, 2, and 3 contribute to one concrete trajectory match. Without this, the claim that coarse-to-fine abstractions are learned remains descriptive rather than demonstrated.

4. Show ablation visualizations for the hierarchy-disabled variants. If the method is truly using hierarchical interactions, the visual differences should be obvious; otherwise the interpretability story is not substantiated.

### Obvious Next Steps
1. Run a full ablation of the hierarchy design: remove top-down fusion, replace bilinear upsampling, vary the number of levels, and test whether each component is necessary. This should have been in the paper because the main contribution is architectural, not just performance.

2. Evaluate the method on more downstream tasks that depend on semantic trajectory understanding, such as clustering, user inference, or destination prediction. That is the natural next step to validate whether the learned hierarchy is generally useful rather than similarity-specific.

3. Test cross-dataset pretraining beyond Porto-only transfer, including training on mixed cities and transferring to unseen regions. This is needed because the strongest claim is generalization across heterogeneous mobility patterns.

4. Compare against a plain multi-scale Transformer/CNN baseline with similar tokenization and parameter budget. Without this, it is unclear whether the gains come from the proposed hierarchy or from standard multiscale modeling tricks.

# Final Consolidated Review
## Summary
This paper proposes HiT-JEPA, a three-level self-supervised trajectory representation framework for similarity computation. The model combines H3 tokenization, hierarchical convolutional abstractions, and JEPA-style prediction with top-down attention propagation to learn point-, segment-, and trajectory-level embeddings. The paper evaluates on several urban and vessel/check-in datasets with retrieval, zero-shot transfer, fine-tuning, and visualization experiments.

## Strengths
- The core motivation is real and well-aligned with trajectory data: local geometry, intermediate motion patterns, and global route semantics are genuinely multi-scale, and a hierarchical representation is a sensible direction.
- The experimental scope is broad for this niche setting: the paper evaluates on multiple datasets and includes self-similarity retrieval, zero-shot transfer, downstream fine-tuning, ablations, and qualitative analyses. The zero-shot results in particular suggest the learned embeddings transfer better than prior trajectory SSL baselines in some cross-domain settings.
- The paper includes implementation details, dataset preprocessing information, and a code release, which improves reproducibility relative to many trajectory papers.

## Weaknesses
- The main contribution is incremental and not cleanly separated from existing JEPA-style trajectory learning. HiT-JEPA mostly stacks a hierarchical encoder on top of a T-JEPA-like objective plus VICReg regularization, so it is not yet convincing that the paper introduces a fundamentally new learning principle rather than a somewhat over-engineered architectural variant.
- The method description is under-specified in key places, especially the top-down attention propagation and the exact masking/sampling pipeline. The equations are difficult to parse, and it is not clear from the paper alone how the propagated attention is applied in the next layer or how masks are coordinated across levels. That makes the core algorithm hard to reproduce without the code.
- The empirical evidence does not strongly isolate the value of the hierarchy itself. The strongest ablation is limited, and the paper does not compare against a plain multi-scale Transformer/CNN with similar capacity. As a result, the gains could plausibly come from extra depth, extra preprocessing, or stronger regularization rather than the proposed hierarchical interaction mechanism.
- The reported improvements over T-JEPA are often modest, and the paper does not provide variance across seeds or statistical significance. This is a serious issue because many results appear close, especially on Porto and parts of GeoLife; without uncertainty estimates, it is hard to know how stable the gains really are.
- The self-similarity evaluation protocol is somewhat artificial: splitting trajectories into odd/even subsequences and retrieving a paired half does not fully reflect standard nearest-neighbor trajectory search over independent samples. This makes the headline retrieval results less compelling as evidence of general similarity understanding.

## Nice-to-Haves
- A cleaner pseudocode algorithm with explicit tensor shapes and a step-by-step training/inference procedure would make the method much easier to follow.
- More granular ablations would help: separate the effects of H3 tokenization, convolutional abstraction, JEPA prediction, VICReg, and top-down attention fusion.
- Reporting mean and standard deviation over several random seeds would materially strengthen confidence in the results.

## Novel Insights
The most interesting idea here is not simply “hierarchy,” but the attempt to make hierarchy operational inside a JEPA-style latent prediction framework rather than as a separate multi-scale encoder. In principle, propagating high-level attention down to lower levels could align coarse trajectory intent with local motion evidence, which is a plausible way to reduce the usual single-scale bias of sequence models. However, the paper currently presents this as a qualitative story more than a rigorously demonstrated mechanism, so the novelty lies in the design intent more than in evidence that the mechanism is essential.

## Potentially Missed Related Work
- **T-JEPA (Li et al., 2024b)** — directly relevant since HiT-JEPA is an extension of this objective to a hierarchical setting.
- **Simformer (Yang et al., 2024)** — relevant as a recent trajectory similarity model based on a simple Transformer, useful for checking whether hierarchy is actually necessary.
- **Unitraj (Zhu et al., 2024)** — relevant as a large-scale trajectory foundation pretraining direction, though the paper does not compare against it.
- **General hierarchical SSL work in vision/NLP such as HIBERT or hierarchical MAE-style methods** — relevant conceptually because the paper’s main idea closely parallels established hierarchical representation learning patterns.

## Suggestions
- Add a matched-capacity non-hierarchical baseline and a full hierarchy ablation to show that top-down interaction, not just more parameters, drives the gains.
- Rewrite the method section into explicit pseudocode with clear mask construction, attention transfer, and tensor dimensions.
- Report multi-seed mean/std or confidence intervals on the main retrieval and fine-tuning metrics.
- If possible, add downstream tasks beyond trajectory similarity, such as clustering or classification, to show the hierarchy transfers beyond ranking-based retrieval.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 2.0, 4.0]
Average score: 4.0
Binary outcome: Reject
