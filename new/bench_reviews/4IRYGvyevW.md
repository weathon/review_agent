## Summary

The paper proposes to study “rich” feature learning via representational geometry, using manifold capacity and derived geometric statistics (radius, dimension, alignments) as tools to quantify and dissect task‑relevant manifold “untangling” across lazy–rich regimes. It combines a one‑step theoretical result in a 2‑layer teacher–student model with a series of empirical case studies on synthetic 2‑layer nets, CNNs on CIFAR, and RNNs on NeuroGym tasks, and uses this framework to describe learning strategies, learning stages, structural inductive biases, and OOD generalization behavior.

## Strengths

- Clear and technically sound exposition of manifold capacity theory and its geometric decompositions (Sec. 2.1–2.2, Def. 1, Eq. (1)), building on Chung et al. and Chou et al., and packaging them into a usable toolbox (Algorithm 1 referenced, geometric measures defined in lines 110–123).
- A nontrivial theoretical contribution (Theorem 1, Sec. 3.1) that, in a well‑specified teacher–student setting and for one gradient step, proves capacity is strictly increasing in a “richness” parameter (learning rate η) and monotonically linked to accuracy via an invertible function.
- Breadth of empirical demonstrations across synthetic 2‑layer nets, VGG‑11/ResNet‑18 on CIFAR‑10/10C/100, and RNNs on NeuroGym tasks (Secs. 2.3, 3–5), showing that the capacity/geometry pipeline is implementable in realistic scenarios.
- Interesting qualitative findings: different geometric “learning strategies” in 2‑layer nets (radius vs dimension trade‑offs, Sec. 4.1), visually distinct geometry “stages” in VGG‑11 training (Sec. 4.2, Fig. 4c), “poorer–richer” vs “wealthier–lazier” RNN regimes with differing final geometries (Sec. 5.1), and non‑monotonic OOD performance aligned with geometry changes in an “ultra‑rich” regime (Sec. 5.2, Fig. 6b–c).
- The framing is well motivated for neuroscience, where weight changes are unobservable; focusing on activity‑based manifolds and capacity is a sensible and timely direction (Intro, Sec. 1.1, Sec. 5.1).

## Weaknesses

### Fatal

None.

### Major

- **Overextension of “capacity = degree of richness” beyond proven regime.**  
  Theorem 1 rigorously connects capacity, richness (via η), and accuracy only for a single gradient step in a specific 2‑layer teacher–student Gaussian model with fixed readout (lines 157–165, 170). Almost all empirical claims and the central narrative concern full training trajectories of deep CNNs and RNNs, with multi‑stage dynamics and “ultra‑rich” regimes (Secs. 3.1 empirical part, 4, 5). The paper repeatedly states that manifold capacity “quantifies the degree of richness” and “tracks the degree of feature learning in a wide range of settings” (lines 151–166, 64–66, 137–139, 174), but provides no theory beyond t=1 and one architecture family. The empirical support is mostly qualitative (single runs, no statistics). As a result, the central identification of capacity with “amount of task‑relevant feature learning across training” is under‑justified; the scope of the theoretical result and the strength of the empirical evidence do not fully support the generality of the language used.

- **Comparisons to conventional feature‑learning measures are anecdotal and insufficient to justify “better than” claims.**  
  Section 3.2 (lines 182–188) and Fig. 3a–b claim that capacity is “better” at telling apart lazy vs rich or wealthy vs poor regimes than accuracy, weight changes, NTK‑label alignment, and representation‑label alignment. However, the evidence is limited to two synthetic 2‑layer‑net experiments on Gaussian clouds (lines 176–181), with visual comparisons but no quantitative metrics (e.g., correlation with the ground‑truth knob across seeds), no error bars, no systematic exploration of how alternative metrics are configured, and no settings where capacity fails but others succeed. Given that “capacity as a representation‑based quantification of richness” is Contribution 1 (lines 64–67), this level of evaluation is too weak to substantiate superiority over well‑established baselines.

- **OOD section interprets correlational geometry as explanatory without causal tests.**  
  The OOD experiments (Sec. 5.2, Fig. 6) show a compelling non‑monotonic OOD accuracy curve versus inverse scale factor and associated changes in capacity and geometric measures (radius and center‑axis alignment, lines 246–250). The interpretation that “expansion of manifold radius and the increase of center‑axis alignment explain the failure of OOD generalization in the ultra‑rich regime” (line 246, reiterated at 250) goes beyond the evidence: there is no manipulation of geometry (e.g., regularizers controlling radius/dimension) to test causality, nor comparisons to other plausible explanations. At present, the results show correlation in a single configuration; presenting them as an explanation overstates the claim.

### Minor

- **Manifold definition and “task relevance” remain largely assumed rather than validated.**  
  The paper defines manifolds as convex hulls of class‑conditional representations (lines 90–100) and describes “task‑relevant manifolds” in intuitive terms (lines 52–56). In all empirical cases, manifolds appear to be simply “all test examples from a given class at a layer” (Sec. 2.3, Sec. 5.2). This is a reasonable working choice, but many higher‑level interpretations—e.g., that “capacity quantifies the degree of richness (or the amount of task‑relevant features)” (line 153) and that “wealthy vs poor” regimes reflect initial task‑relevant information (lines 188–189, 231–234)—implicitly assume that these convex hulls predominantly capture task‑relevant structure. The paper does not operationally verify this, for instance by showing that directions most relevant for capacity align with directions important for classification, or by probing subspaces. This makes some narrative interpretations more speculative than necessary.

- **“Learning stages” and “learning strategies” are described from single trajectories without robustness analysis.**  
  Section 4.1–4.2 and Fig. 4a–c identify distinct geometric learning strategies and four stages in VGG‑11 training (“clustering,” “structuring,” “separating,” “stabilizing,” lines 198–205). These are read off qualitatively from one synthetic 2‑layer‑net setup and one VGG‑11 run, with no statistics over seeds/architectures/datasets, no formal change‑point detection, and only indirect links to performance beyond the coarse association with capacity. They are interesting hypotheses, but current evidence does not show that these “stages” are robust, general phenomena; the paper could more clearly frame them as exploratory rather than as established structure.

- **Lack of variance/seed reporting for key geometric quantities.**  
  Across Figs. 2–6, the trajectories of capacity and geometry are shown without means/variances over seeds or an indication of stability. Because capacity estimation involves sampling over y,T and solving QPs (lines 100–108), and because training trajectories can be noisy, it would be important to know whether the reported patterns depend on a particular seed or are representative.

- **Limited practical guidance on estimator behavior and computational cost in the main text.**  
  The main text defers almost all practical details—number of manifolds, anchor samples, QP scaling, sample complexity—to the appendix (e.g., lines 100–101, 108, 145–147, 230–231). For a methods paper intended to be used in neuroscience and ML applications, a concise summary of estimator variance vs. sample size, computational overhead relative to standard evaluation, and typical hyperparameters would make the contribution more actionable.

### Trivial

- Some terminology (e.g., “wealthier–lazier,” “poorer–richer,” “ultra‑rich”) is vivid but informal; adding brief quantitative definitions (e.g., thresholds on capacity or inverse scale factor) in the main text would improve clarity.

## Nice-to-Haves

- More systematic evaluation of capacity and geometric measures against a broader suite of representation‑based baselines (e.g., linear probe accuracies at intermediate layers, inter‑/intra‑class distance statistics, margins) on both synthetic and real datasets, with quantitative summaries (correlations with a known richness knob, OOD performance) and variance across seeds.
- Empirical or theoretical analysis of scenarios where capacity might mislead—e.g., when networks learn highly nonlinear but task‑relevant structure not well captured in the last‑layer linear readout, or when capacity increases due to task‑irrelevant variations—to better delineate the method’s limits.
- Initial steps toward causal tests in the OOD and RNN settings, e.g., introducing explicit regularizers targeting radius/dimension and examining whether predicted changes in geometry improve OOD performance or alter inductive biases.

## Removed Points

These points are flagged to be removed, treat them with caution.

- Any concern about the existence or release status of manifold capacity theory, datasets (CIFAR‑10/100, CIFAR‑10C, NeuroGym), or baselines such as NTK alignment or CKA: the paper clearly cites these and uses standard setups (e.g., lines 147–148, 184–185). Such concerns would stem from reviewer knowledge gaps, not from the paper.
- Hypothetical criticisms that the paper “does not define manifolds at all” or “never specifies what point clouds are used”: Sec. 2.1 explicitly defines manifolds as convex hulls of class‑conditional representations (lines 90–100), and Sec. 2.3/5.2 indicate use of last‑layer test representations for CIFAR experiments (lines 141–147, 248–250). While one can still question the task relevance of this choice (kept as a minor weakness), “no definition” would be factually wrong.
- Complaints about missing appendices, missing proofs, or missing references: by construction, the extracted text omits appendices and reference lists (line 256), so any such criticisms are artifacts of the extraction, not the submission.
- Pure formatting or typography nitpicks (line breaks, duplicated figure captions, etc.): the parsing clearly introduces artifacts (e.g., duplicated figure legends around lines 46–48, 135–137, 206–213), which are not the authors’ fault.

## Novel Insights

None beyond the paper’s own contributions; the main points of critique concern scope and evidential strength rather than uncovering new conceptual angles.

## Suggestions

- Soften and qualify general claims tying capacity to “degree of richness in a wide range of settings” and “explaining” OOD failure. Make clear that Theorem 1 justifies this connection in a specific one‑step 2‑layer setting, and that in deep CNNs/RNNs the evidence is empirical and largely qualitative.
- For the comparisons to conventional measures in Sec. 3.2, add quantitative analyses: multiple seeds, summary statistics (e.g., Spearman correlation with the inverse scale factor or with a ground‑truth “wealth” parameter), and perhaps additional baselines such as linear probe accuracy or margin statistics.
- In the OOD section, rephrase “explain the failure” to emphasize correlation and hypothesis generation, and, where possible, add experiments that perturb geometry (e.g., regularizers influencing within‑class spread or alignment) to test whether OOD behavior changes as predicted.
- Provide at least some seed‑averaged curves or variance bands for capacity and key geometric measures in the main text to demonstrate robustness of the reported strategies and stages.
- Clarify in the main text how many anchor samples and data points are typically needed, and give a rough sense of computational cost, so practitioners (especially in neuroscience) can judge feasibility.

## Score and Decision

### Calibration anchors

High‑scoring (avg > 7) geometry/feature‑learning papers:
- `/home/wg25r/review_agent/human_reviews/AP0ndQloqR.md` (avg 7.50, Accept (Oral)): Strong geometric analysis of reinforcement‑learning representations with clear empirical validation and tight theory–experiment link. The current paper is somewhat weaker on quantitative validation and scope control than this anchor.
- `/home/wg25r/review_agent/human_reviews/aZ1gNJu8wO.md` (avg 7.33, Accept (Spotlight)): Introduces a geometric framework for memorization in generative models with thorough experiments and careful claims; again, empirically stronger and better calibrated than the paper under review.
- `/home/wg25r/review_agent/human_reviews/Njx1NjHIx4.md` (avg 7.50, Accept (Spotlight)): Formation of representations with both theory and well‑designed experiments; similar ambition but more rigorous evaluation.
- `/home/wg25r/review_agent/human_reviews/dEypApI1MZ.md` (avg 7.20, Accept (Spotlight)): On feature learning and scaling laws; offers strong empirical evidence for claims about rich regimes vs kernels.
- `/home/wg25r/review_agent/human_reviews/JWtrk7mprJ.md` (avg 7.60, Accept (Oral)): Geometry‑heavy work with solid theoretical grounding and clear empirical support.

Medium band (4–6), representation‑geometry papers:
- `/home/wg25r/review_agent/human_reviews/CtiFwPRMZX.md` (avg 5.00, Reject): Manifold compression as a feature‑learning measure, but with overclaimed scope and limited empirical validation—quite similar in pattern to this submission.
- `/home/wg25r/review_agent/human_reviews/j7yeq2sOj3.md` (avg 5.00, Reject): Another review of the same compression paper, echoing concerns about insufficient comparisons and robustness.
- `/home/wg25r/review_agent/human_reviews/TVnkjz4MqV.md` (avg 5.50, Reject): Neural manifold regularization; interesting idea but lacking comprehensive empirical support and clear wins over baselines.
- `/home/wg25r/review_agent/human_reviews/RwCxxaHvyp.md` (avg 5.00, Reject): Uses geometric/data‑information metrics; critiqued for nice ideas but under‑developed empirical story.
- `/home/wg25r/review_agent/human_reviews/yMMIWHbjWS.md` (avg 6.00, Reject): Convex decision region analysis; conceptually strong but empirically limited.

Low band (<3):
- `/home/wg25r/review_agent/human_reviews/xA25Ib7H8U.md` (avg 2.33, Reject): Geometric analysis of continuous‑depth networks judged overly theoretical with little empirical support and unclear impact.
- `/home/wg25r/review_agent/human_reviews/RIaIpdUCPb.md` (avg 3.00, Withdrawn/Reject): Representation geometry for compositional generalization; conceptually appealing but empirically weak and under‑validated.
- `/home/wg25r/review_agent/human_reviews/A9yKCUQNnc.md` (avg 3.00, Withdrawn/Reject): Low‑dimensional representation analysis; criticized for speculative conclusions from limited data.
- `/home/wg25r/review_agent/human_reviews/b0elDO9v31.md` (avg 3.00, Reject): Geometric deep learning theory with insufficient connection to practice.
- `/home/wg25r/review_agent/human_reviews/WxqWuG431g.md` (avg 2.60, Withdrawn/Reject): Geometry of concept representations; judged too anecdotal and under‑evaluated.

Relative to these anchors, the present paper is clearly above the low band: it combines a genuine new theoretical result (Theorem 1) with multiple empirical case studies and reasonably clear writing. However, its pattern of strengths and weaknesses—good conceptual framing and some theory, but overclaimed generality, limited quantitative evaluation against baselines, and largely qualitative stage/geometry claims—resembles the medium‑band rejected geometry papers around scores 5–6 more than the accepted high‑band ones. It is perhaps slightly stronger than the weakest of those (due to Theorem 1 and breadth), but still below the threshold where human reviewers accepted geometry‑focused work.

Taking these anchors into account, an appropriate calibrated score is:

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>