# SYNC: Measuring and Advancing Synthesizability in Structure-Based Drug Design

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Designing 3D ligands that bind to a given protein pocket with high affinity is a fundamental task in Structure-Based Drug Design (SBDD). However, the lack of synthesizability of 3D ligands has been hindering progress toward experimental validation; moreover, computationally evaluating synthesizability is a non-trivial task. In this paper, we first benchmark eight classical synthesizability metrics across 11 SBDD methods. The comparison reveals significant inconsistencies between these metrics, making them impractical and inaccurate criteria for guiding SBDD methods toward synthesizable drug design. Therefore, we propose a simple yet effective SE(3)-invariant \textit{\underline{SYN}thesizability \underline{C}lassifier} (SYNC) to enable better synthesizability estimation in SBDD, which demonstrates superior generalizability and speed compared to existing metrics on five curated datasets. Finally, with SYNC as a plug-and-play module, we establish a synthesizability classifier-driven SBDD paradigm through guided diffusion and Direct Preference Optimization, where highly synthesizable molecules are directly generated without compromising binding affinity. Extensive experiments also demonstrate the effectiveness of SYNC and the advantage of our paradigm in synthesizable SBDD. Code is available at \url{https://github.com/XYxiyang/SYNC}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SYNC, a 3D-aware, SE(3)-invariant classifier for estimating molecular synthesizability in Structure-Based Drug Design (SBDD).
It benchmarks eight classical synthesizability metrics across 11 SBDD methods, finding strong inconsistencies, and proposes SYNC as a unified, differentiable, and efficient alternative.
SYNC is further embedded into two SBDD paradigms—guided diffusion and Direct Preference Optimization (DPO)—to generate ligands that are both synthesizable and high-affinity.
Extensive experiments (five curated datasets, large-scale SBDD benchmarks) show SYNC achieves higher accuracy and efficiency than existing metrics and improves molecule synthesizability without hurting docking scores

### Strengths
This paper makes a timely and practical contribution to structure-based drug design by systematically benchmarking existing synthesizability metrics and introducing a fast, 3D-aware, SE(3)-invariant classifier (SYNC) that effectively bridges molecular generation and realistic synthesis feasibility. The integration of SYNC into both guided diffusion and DPO paradigms is elegant, demonstrating consistent improvements in synthesizability across multiple SBDD backbones without sacrificing binding affinity. The experiments are comprehensive, covering five datasets, eleven baselines, and detailed ablations, making the work both reproducible and impactful for real-world drug discovery.

### Weaknesses
While technically sound, the core innovation of SYNC is incremental, primarily reusing standard EGNN architectures for classification. The reliance on proxy “easy/hard-to-synthesize” labels without experimental validation limits the practical reliability of results. Certain methodological aspects, such as gradient guidance stability, parameter sensitivity, and DPO training dynamics, lack clarity. Moreover, evaluations are restricted to two SBDD backbones and omit comparisons with more recent LLM- or synthesis-pathway-based approaches, slightly constraining the generality and novelty of the proposed framework.

### Questions
1. How sensitive is SYNC’s performance to the training distribution—does it generalize to unseen chemical scaffolds or pocket types?

2. Can SYNC outputs be calibrated to provide probabilistic synthesizability scores usable in Bayesian optimization?

3. How does SYNC compare with recent LLM-based synthesis predictors (e.g., ChemGPT retrosynthetic models)?

4. Does guided diffusion introduce mode collapse toward overly simple molecules?

5. Are there interpretability tools (e.g., attention maps or feature attribution) to rationalize why SYNC deems a molecule synthesizable?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors compared classical synthesizability metrics for classical SBDD methods and observed inconsistencies in the rankings. They thus devised a framework called SYNC as a new metric that achieved more consistent reports on various datasets. Also, since they designed such a framework to be fast, differentiable, and SE-invariant, they demonstrated that such a metric can be incorporated into diffusion guidance and DPO. Experiments showed that such incorporation benefits the synthesizability without hurting much of the affinity. They performed ablation studies to justify their choice for the framework as well as guidance, such as the use of 3D information and multistep guidance.

### Strengths
1. The presentation is straightforward and clear. 
2. The designed framework and model are well-suited for this task and provide even more benefits, such as speed and robustness. 
3. The experiments are strong and support the claims well.

### Weaknesses
1. Since SYNC is trained on the same datasets shown in Table 1, it is not entirely fair to call it "surprisingly outperforms other baselines by a wide margin", as it is data-centric. This can be related to Question 1.

2. In Table 2, many color markings are off. Minuses are marked as both green and red. This could potentially lead to false interpretation. 

3. SYNC's framework can be better shown with some illustrations. So far, it was only described in plain text and mentioned only halfway down in the paper. If this is one of the main contributions, please be more explicit in the main section.

### Questions
1. Why is SYNC so low on the generative models’ results compared to the constructed datasets such as TS1 and TS2? I wonder how much of this drop is due to the limited capacity of the model and how much is caused by its lack of generalization to out-of-distribution samples.

2. Why is there a sudden spike in SYNC during the final steps of the multi-step process? Figure 4 may require additional explanations.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles a central bottleneck in structure-based drug design by focusing on synthesizability. It first assembles five datasets with easy- and hard-to-synthesize labels and benchmarks eight widely used metrics across eleven SBDD generators, showing large and sometimes contradictory judgments among existing metrics. To address this, it introduces SYNC, an SE(3)-invariant, 3D-aware classifier that predicts synthesizability from ligand conformations. SYNC is then used as a plug-and-play control signal in two paradigms, guided diffusion and direct preference optimization, to steer TargetDiff and DecompDiff toward molecules that are easier to make while largely maintaining docking affinity. Extensive experiments, ablations, and runtime measurements suggest SYNC attains strong accuracy with low inference cost and yields consistent gains in synthesizability without notable loss in binding metrics.

### Strengths
The strengths lie in the clarity and importance of the problem framing, the breadth of the benchmark, and the empirical finding that many classical synthesizability metrics disagree in practice. The proposed classifier is well motivated for SBDD because it explicitly consumes 3D information and is designed to be SE(3)-invariant and fast, which are practical requirements for generation-time control. The integration pathways are simple yet effective, and results are consistently favorable across multiple metrics and two different backbones, with helpful tables that isolate improvements and show that binding scores remain similar or sometimes improve. The ablations are thoughtful, covering 3D versus 2D or 1D inputs, edge modeling choices, guidance schedules, and DPO iteration behavior, which strengthens the case that the observed gains are not brittle. The work also pays attention to computational cost and scalability, an often overlooked but critical aspect for large-scale screening and iterative design.

Here is the itemized summary for rebuttal and discussion:

- Clearly frames the practical importance of synthesizability in structure-based drug design
- Builds a broad benchmark with multiple datasets, metrics, and generators, revealing contradictions among common metrics
- Introduces a fast SE(3)-invariant, 3D-aware classifier that matches SBDD needs
- Provides two simple integration paths, guidance and DPO, that act as plug-and-play controls
- Improves synthesizability consistently while largely preserving docking-based binding scores

### Weaknesses
The weaknesses stem from evaluation design and possible circularity. The ES and HS labels are proxies rather than wet-lab confirmations, so real-world synthesizability remains inferred rather than demonstrated. Several baselines require thresholding or stock assumptions that can bias outcomes, and FSScore’s threshold is tuned over the same datasets, which may inflate apparent fairness. Because SYNC both evaluates and guides generation, parts of the analysis risk becoming self-referential, even if cross-metric reporting partly mitigates this. The DPO pairs are constructed using SYNC’s guidance, which could further entangle training and evaluation signals. The guided diffusion approach introduces extra compute and hyperparameters, and the K-nearest-neighbor edge construction and valency heuristics during guidance, while pragmatic, could be fragile across chemistries or larger rings. Reported binding comparisons rely on docking scores, which are useful but imperfect surrogates; stronger tests such as pose validity checks or physics-based rescoring would raise confidence. Finally, although SYNC often ranks first, margins narrow on some datasets, and the method does not outperform retrosynthetic tools on their strongest cases, leaving open how best to combine fast 3D classifiers with pathway-level verification for end-to-end design.

---

1. Uses proxy labels rather than wet-lab validation, leaving real-world synthesizability unconfirmed
2. Several baselines depend on thresholds and stock assumptions that can bias comparisons, with possible tuning leakage
3. Risk of circularity since SYNC both evaluates and guides, and DPO pairs derive from SYNC-guided samples
4. Guidance adds compute and hyperparameters; KNN edges and valency heuristics may be brittle across chemistries
5. Reliance on docking scores without stronger pose checks or physics-based rescoring limits confidence
6. Advantages narrow on some datasets and do not exceed retrosynthesis tools on their best cases
7. External validity to more diverse targets and chemotypes remains to be demonstrated

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3
