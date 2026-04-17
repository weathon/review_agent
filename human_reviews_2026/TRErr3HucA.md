# From Evaluation to Design: Using Potential Energy Surface Smoothness Metrics to Guide MLIP Architectures

- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
The reliability of machine learning interatomic potentials (MLIPs) in downstream physics tasks depends not only on reproducing reference energies and forces, but also on the smoothness of the underlying potential energy surface (PES). While prior work has evaluated smoothness indirectly—most commonly by running microcanonical molecular dynamics (MD) simulations or calculating phonon modes—such tests capture only near-equilibrium smoothness and are computationally expensive. We introduce the Bond Smoothness Characterization Test (BSCT), a simple and inexpensive benchmark that directly quantifies PES smoothness both near- and far-from-equilibrium by probing controlled bond deformations. Since BSCT measures the PES itself, it can detect a wide range of instabilities, such as discontinuities, artificial minima, or spuriously large forces. To investigate how BSCT can guide the design of scalable, physically reliable MLIPs, we start from an unconstrained Swin-Transformer-inspired backbone and conduct a controlled study on the SPICE (molecules) and MPTrj (materials) datasets. Beginning with this baseline, we introduce targeted design changes—differentiable k-nearest neighbor graphs, temperature-controlled attention, and broadened radial smearing widths. At each step, we measure the energy and forces accuracy, energy conservation in microcanonical simulations, and the BSCT metric. Our results show that BSCT improvements consistently predict reductions in MD instabilities and enable early-stage filtering of problematic models. The final BSCT-guided models achieve state-of-the-art accuracy on SPICE and MPTrj while maintaining excellent smoothness, demonstrating that optimizing for physical soundness via BSCT naturally yields high performance. Our results position BSCT as a practical, general-purpose metric for guiding the design of reliable MLIPs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces the Bond Smoothness Characterization Test (BSCT) — a new benchmark and metric designed to directly measure the smoothness of the potential energy surface (PES) predicted by machine-learned interatomic potentials (MLIPs).
The authors define a quantitative indicator called Force Smoothness Deviation (FSD), which evaluates how smoothly forces vary under systematic bond stretching/compression, providing a cheap proxy for PES regularity both near and far from equilibrium.

### Strengths
BSCT/FSD provides a direct, quantitative, and low-cost way to assess the physical smoothness of potential energy surfaces (PES), addressing the long-standing gap between accuracy-based and physics-based MLIP evaluation.

The correlation analysis between FSD and kinetic temperature spikes is elegant and convincingly supports FSD as an early physical reliability predictor.

### Weaknesses
It remains unclear whether BSCT/FSD behaves consistently for modern equivariant architectures(e.g., eSCN, ViSNet)

The “temperature-controlled attention” mechanism is intuitively justified but lacks formal analysis of how τ influences Lipschitz continuity or gradient smoothness.

BSCT currently probes only bond-length variations. It does not account for angular or torsional degrees of freedom, which may dominate PES roughness in larger or periodic systems.

### Questions
Can FSD be generalized to multi-dimensional collective coordinates (angles, torsions) rather than just single bonds?

How sensitive is FSD to the bond-type distribution or sampling density in the BSCT dataset?

Does incorporating FSD as a differentiable regularizer during training further improve PES smoothness, or is it used purely post-hoc?

What are the computational overheads of Diff-kNN compared with standard radius graphs at scale (e.g., MPTrj supercells)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose the Bond Smoothness Characterization Test (BSCT) to assess smoothness in potential energy surfaces predicted by machine-learned interatomic potentials (MLIPs). This test involves a new metric (the FSD -- force smoothness deviation), which is able to characterize smoothness and correlates well with other tests in most cases. Using this metric, as well as other tests, the authors tune their proposed architecture (MinDScAIP) to achieve very good performance on the SPICE and MPTrj datasets.

### Strengths
The BSCT test is novel and original, and the same goes for the MinDScAIP architecture. The paper proposes some interesting ideas and shows results that will be valuable for the community. The quality of the investigation is high and the presentation is clear. The empirical performance of the model is strong.

### Weaknesses
The paper and its content are of good quality. However, I believe they are lacking in some areas at the moment.

**Presentation**
- [Minor] The paper proposes two distinct novelties: the BSCT test on one side, and the MinDScAIP architecture on the other. While the sections are well-organized, the abstract does not mention the name of the new architecture, the introduction does not either, and the glue text between the different sections is often vague. For example, it would be helpful, in some points, to clearly state something along the lines of "our contribution is two-fold:...", and/or "After having established the BSCT test as a reliable method for the estimation of PES smoothness, we now proceed to...".
- [Minor] In Figure 5, it seems that the simulations have not equilibrated yet by the time the MD trajectory begins (it takes about another 20 ps for the simulation to reach the target temperature).
- [Minor] Typos: citations in lines 147-148 should be enclosed in paretheses, "adapts" -> "adopts" (line 215), double period in line 392, "bond breaks" is likely a typo in line 347, Table 5 probably contains numbers in *million* time steps per day.
- [Minor] The authors should definitely acknowledge Pozdnyakov 2023 (NeurIPS), who proposes a similar attention-based and unconstrained architecture and whose mechanism for reference frame smoothness resembles the Diff-kNN the authors propose here.

**Rigor**
- At many points the authors claim that MD is expensive (e.g., line 53: "requires costly MD to evaluate", lines 112-113: "which typically requires costly microcanonical MD to evaluate"), then they provide a test and a metric that requires first-principles evaluations. These claims should definitely be softened. See also "methods" and "questions".
- In lines 134-135, the authors claim "we focus on bonds because their ground truth PES is intrinsically smooth". This is incorrect (according to the current widespread definition of PES smoothness as "infinite-order differentiability"), as changes in the electronic ground-state along the dissociation curve can make the PES non-differentiable, just like any other cut or section of any ground-state PES. The authors should clarify what they meant here.
- Line 392: "confirming non-conservative force field". The forces are conservative in this case since they are calculated by backpropagation. It is the discontinuities in the PES caused by the standard kNN search, **combined with the finite step sizes of MD**, which cause the energy drift here. In other words, a hypothetical (and impossible in practice) MD performed with infinitely small steps would conserve energy in this case.

**Methods**
- After claiming that MD is slow, the authors propose the BSCT test, which requires first-principle evaluations. Unfortunately, I do not believe the BSCT test is very practical in its current state. If the structures the authors used are made public, it can be turned into a molecular benchmark for smoothness. However, such a benchmark would not be transferable to models created for purposes other than organic molecules, while MD can be performed on arbitrary systems. Most importantly, even if published, the new benchmark would not easily transfer to other levels of quantum chemical theory. Hence, at the moment, I do not see how it could become widely adopted by model developers without asking them to perform new (and expensive) DFT evaluations on their data (even then, that's assuming they work with molecules; no alternative is provided for materials).

### Questions
I thank the authors for this important contibution to the field. I have three questions for them:
- How would an architecture developer or a model creator use their BSCT in practice if their DFT settings are not consistent with those used in this work and/or if they are interested in applications other than molecules (materials, catalysis, etc.)?
- On lines 390-391, the authors mention that "models with proper normalization never output large forces". Is this not worrying, considering that forces can be unbounded in practice? The same concern goes for the energy head of these models.
- The results in Table 4 seem to indicate a trade-off between smoothness, which is beneficial especially when predicting phonons, and accuracy on all other properties. Do the authors have an idea of the cause of this phenomenon?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces a new benchmark for evaluating potential energy surface (PES) smoothness in machine-learning interatomic potentials (MLIPs). The proposed Bond-Smoothness Characterization Test (BSCT) probes one-dimensional bond stretching and compression trajectories, and reports a Force Smoothness Deviation (FSD) score intended to capture spurious extrema or non-smooth behavior. The authors further apply BSCT as a development signal for designing a general-purpose transformer-inspired MLIP architecture with minimal built-in geometric constraints, incorporating smoothness-oriented design choices such as broadened smearing and temperature-controlled attention. The resulting model achieves competitive performance on SPICE, MPTraj, and the Matbench Discovery benchmark suite, while maintaining good smoothness metrics.

Note: An LLM (ChatGPT 5) was used to prepare this review -- it expanded notes into the full review, and suggested some changes in phrasing. It did not provide useful suggestions for the content of the review.

### Strengths
- Introduces a physically motivated smoothness benchmark and a corresponding scalar metric (FSD) that quantifies anomalies during controlled bond stretching and compression.
- Constructs a dataset tailored to probe PES smoothness in both near- and far-from-equilibrium regimes.
- Proposes a new minimally constrained transformer architecture, and systematically studies architectural and regularization decisions with respect to PES smoothness.
- Demonstrates strong empirical performance across molecular and materials benchmarks, and attempts to connect physical soundness with downstream stability.

### Weaknesses
- The narrative feels divided between presenting a benchmark and proposing a new architecture; a clearer separation of these contributions, and a stronger articulation of how the benchmark concretely guides model design, would improve clarity.
- While the results suggest that FSD correlates with MD stability, broader and more quantitative evidence would help substantiate the claim of predictiveness.
- The paper would benefit from explicit examples where BSCT led to specific architectural choices, especially in cases where intuition or standard smoothness regularization alone would not have suggested the same outcome.
- Physical priors such as rotational equivariance are discussed only briefly; a more explicit comparison with existing physics-informed approaches would strengthen the positioning.

### Questions
- Please clarify how BSCT was used to guide architecture decisions in practice. Could the same modifications have been derived by simply optimizing for smoothness via standard regularization techniques? Are there cases where BSCT led to counterintuitive or unexpected design decisions? It would also be interesting to discuss whether automated architecture search using FSD as an objective might reproduce similar choices.
- The paper asserts that FSD correlates with MD stability. Figure 5 is compelling, but a quantitative measure of correlation would make the claim stronger.
- Table 1 highlights that GemNet-T achieves low FSD but still exhibits large energy drift, and similarly the prediction head with the lowest FSD has the worst energy conservation. I assume this reflects non-conservativity rather than smoothness. Please discuss this distinction clearly, as it may otherwise lead to confusion about FSD’s scope and whether this constitutes a limitation of the metric.
- Fu et al. (ICML 2025; https://openreview.net/forum?id=R0PBjxIbgm) in the eSEN model also emphasize smoothness. Please discuss the connection to that work and, if possible, report eSEN’s performance under BSCT to contextualize your metric.
- In Equation (3), the notation could be clearer; explicitly indicating that the derivative is taken with respect to α and evaluated at α would help readability.
- How does FSD compare to measures of non-conservativity such as the Jacobian asymmetry metric introduced by Bigi et al.? A brief, potentially empirical, comparison would help clarfiy what they capture.
- How do the authors define the term "bond"? What's your operational definition?

Minor suggestions:

- The legend of Table 5 states “Number of Steps Per Day,” which likely is intended to mean “Millions of Steps per Day.”
- The proposed architecture shares similarities with the point-edge transformer (Pozdnyakov & Ceriotti, NeurIPS 2023). It would be appropriate to acknowledge this connection.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the Bond Smoothness Characterization Test (BSCT) - a low-cost test to directly assess potential-energy-surface (PES) smoothness for ML interatomic potentials by performing 1-D bond stretches/compressions. 

Then the paper incorporates three types of inductive biases aimed to promote smoothness, differentiable neighborhood construction, temperature-controlled attention, and broadened radial smearing. The paper finds that the BSCT test is capable of indicating the associated increase of smoothness, and can serve as a cheaper alternative to molecular dynamics to estimate the stability of the model.

### Strengths
The paper argues convincingly that accuracy is not the same as reliability and that PES smoothness matters for practical simulations. This point is often overlooked, and many current models are mostly focused on the errors in energies and forces that do not provide a reliable estimation of the quality of the interatomic potential. 

The proposed smoothness indicator is cheap and was found to be able to predict instabilities in much more costly MD experiments. 

The three proposed strategies were found to be able to promote smoothness in practice.

### Weaknesses
The proposed metric, Bond Smoothness Characterization Test, is based on an isolated and, I would say, artificial part of the Potential Energy Surface. The forces are evaluated for configurations that are hardly expected to originate during molecular dynamics simulations - a molecule is stretched across one of the bonds, while the two parts are undisturbed. 

The non-smoothness of PES is expected to affect the stability of MD if it is present for the part of the configurational space that is accessible and often visited during MD trajectories. The chosen part of configurational space to judge the overall smoothness or non-smoothness of PET doesn't seem to be especially important for MD, or at least more important than the other, random parts of PES. There are no ablations comparing the proposed smoothness metric to alternatives. 

Additionally, the claim that the FSD values can be used to assess the stability of MD is not fully supported. Currently, it is given by Figure 5, which shows this dependency only for one particular case. It is unclear if this is just a coincidence or a rule, and it would be nice to have some statistically significant verification.

### Questions
Why do the authors think that this particular choice of the smoothness criteria is more relevant for MD stability compared to others? 

Can the authors propose other alternatives, such as simply estimating the Hessian of PET at random off-equilibrium points or others, and show that the proposed metric is better (e.g., better predicts stability/instability of MD) compared to other metrics? 

Can the authors provide a statistically significant justification that the FSD indeed predicts well MD stability?

### Soundness
3

### Presentation
3

### Contribution
2
