# Why We Need New Benchmarks for Local Intrinsic Dimension Estimation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 2, 6, 8, 8

## Abstract
Neural Local Intrinsic Dimension (LID) estimators are typically bound to domain-specific architectures whose inductive biases can yield inconsistent estimates for the same underlying manifold. Existing evaluations either use overly simple synthetic data (with known LID) or real datasets (with unknown LID), obscuring true performance. We introduce a principled benchmarking framework that (i) maps the same manifold into multiple domain representations while preserving its structure, enabling like-for-like cross-architecture tests; (ii) designs harder variants of popular datasets that target key manifold properties; and (iii) applies controlled transformations with known LID shifts to stress-test methods even when absolute LID is unknown. Across this suite, including non-trivial synthetic datasets, we show that accuracy on simple manifolds does not transfer across domains and that state-of-the-art methods fail under targeted stressors, revealing clear failure modes and areas for improvement. Data and code are available: https://github.com/DominikFilipiak/LID-Benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes several methods for generating high-dimensional datasets with known intrinsic dimensions from various perspectives, for the purpose of evaluating local intrinsic dimension (LID) estimation methods, and presents examples of how these datasets can be used to assess the performance of different LID estimation algorithms.

### Strengths
The paper proposes several methods for generating datasets that are well-suited or particularly challenging for intrinsic dimension estimation from various perspectives, and it conducts comprehensive evaluation experiments on representative (neural) LID estimation methods.

The diverse data manifolds presented in Subsections 3.1 through 3.9 each represent scenarios where successful intrinsic dimension estimation is desirable, and as such, the proposed benchmarks provide a valuable contribution in their current form.

### Weaknesses
The paper emphasizes the importance of benchmarking LID estimation and proposes various methods for constructing datasets; however, it lacks a discussion that would convince the broad ICLR audience why (local) intrinsic dimension estimation is important and to what extent it is a significant problem in the first place.

While the procedures for creating benchmark datasets are described in sufficient detail, the discussion of LID estimation methods themselves is insufficient. The appendix provides only brief, one-paragraph summaries of individual methods, but it does not offer insights into which methods succeed or fail on each benchmark dataset, and why, which would be necessary to derive meaningful understanding from the experiments.

### Questions
As pointed out in the “Weaknesses” section, the paper should clearly articulate why (local) intrinsic dimension estimation is important in the first place.

In addition, to convey the significance of the proposed benchmark datasets, it is necessary to explain why neural network–based methods are important, and simultaneously, to discuss the limitations of traditional fractal-dimension approaches such as box-counting dimension.

Specifically, please describe what the limitations of non–neural-based methods are, and clarify why the proposed benchmark datasets are designed primarily for evaluating neural network–based methods.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper argues that current evaluations of local intrinsic dimension (LID) estimators are either too simple (synthetics with known LID) or unverifiable (real data with unknown LID). It proposes a benchmarking framework that: (i) maps the same manifold across domains (e.g., to images/audio) to test architecture/inductive-bias invariance; (ii) hardens existing datasets along specific geometric/measure aspects (non-uniform density, curvature, boundaries, nearby/thin manifolds); and (iii) introduces controlled real-data transformations (monotonic warps, ambient-dimension extension, auxiliary dimension injection) that induce known LID shifts or preserve LID, enabling relative ground-truth checks on real data. Across a suite of synthetic, “real-like,” and transformed FMNIST datasets, the authors show that several recent neural LID estimators (LIDL, FLIPD, NB) exhibit systematic failures under these stressors, whereas a classic method (ESS) is often more stable on low-dimensional cases; none is universally robust.

### Strengths
Originality.

Introduces a domain-mapping toolbox (IDR) to test the same manifold across architectures/domains—surfacing inductive-bias sensitivity that simple synthetics miss.

Designs targeted stressors (non-uniform densities, curvature, edges, nearby/thin manifolds) and controlled real-data transforms (ME/ASE/ADI) to create falsifiable checks (LID preserved or shifted by a known amount).

Quality.

Broad, well-motivated experimental suite: Gaussians/Spheres/Uniform/Spaghetti, Moon/Funnel/Spiral (nearby/thin), Arrows (real-like with known LID), plus FMNIST transforms; summarizes per-aspect outcomes in tables/figures.

Identifies concrete failure modes (e.g., NB overestimation on Gaussians/Spheres; LIDL/FLIPD instability; sample-size bias).

Clarity.

Clear “what/why” in Introduction and contributions; formalization of IDR (PCA-based embedding that preserves manifold geometry for the tests) and per-transform rationale are well explained; figures are readable and tied to claims.

Significance.

Benchmarks uncover architecture/domain sensitivity and lack of invariance that prior evaluations hide; they provide a useful bar for future LID methods and highlight where current neural estimators fail (e.g., Arrows, boundaries, size-dependence).

### Weaknesses
1. Despite the stated “any continuous domain” framing, experiments are concentrated in the image domain (FMNIST, Arrows) and four LID estimators. This limits generality of the conclusions about cross-domain invariance. Add at least one non-image domain (audio/EEG) to exercise ME/ASE/ADI beyond images.

2. For ME/ASE/ADI, the paper argues LID preservation/shift qualitatively; however, formal guarantees (and error bounds under finite samples) are not given. Tighten the theory or at least provide simulation checks where the induced ΔLID is analytically known before porting to FMNIST.

3. IDR relies on PCA directions from a single FMNIST class chosen for visual coherence; the Discussion acknowledges this choice and compute constraints. This may bias domain mappings and downstream conclusions about architecture invariance. Consider validating IDR with alternative bases (e.g., DCT) or multiple classes, and quantify any distortions with pairwise-distance/curvature diagnostics.

4. The authors note that more principled metrics per aspect are “future work.” Even simple aggregated scores (MAE vs. known LID; calibration of ΔLID; invariance indices) with confidence intervals across seeds would make comparisons more decisive.

5. Several methods are sensitive (e.g., ESS neighbors; LIDL δ; FLIPD t). The paper shows this qualitatively, but a uniform protocol (grid, selection rule, robustness bands) would avoid accusations of cherry-picking and clarify practical deployability.

### Questions
1. For ME/ASE/ADI, can you provide proof sketches or lemmas that (a) ME preserves LID; (b) ASE preserves LID; (c) ADI increases LID by the number of injected independent parameters plus synthetic verifications before FMNIST?

2. Beyond the PCA argument, can you report quantitative diagnostics (isometry error, curvature change) showing IDR preserves manifold geometry for your choices; and try one alternate basis (e.g., DCT) to check the invariance of conclusions?

3. Could you include one audio benchmark (e.g., monotonic resampling—ASE; dynamic range warp—ME) to substantiate the “any continuous domain” claim?

4. Will you add per-aspect scalar metrics with 95% CIs (e.g., MAE/ΔLID error, invariance score) to complement plots, and report seeds and tuning protocols for each method?

5. Arrows seems especially challenging; can you ablate which property (corners, occlusion, color composition) breaks each method, and whether smoothing/anti-aliasing mitigates it?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper analyzes algorithms for estimating local intrinsic dimension, focusing on ESS, NB, LIDL, and FLIPD. The experiments cover simple synthetic datasets with known dimension and realistic datasets with complex structure and unknown dimension. For the realistic case, the paper applies transformations that preserve manifold dimension or change it in a known way and then measures changes in reported dimension to evaluate robustness. The study investigates controlled factors that affect LID estimates, including manifold curvature, thickness, sample size, non uniform density, and proximity to boundaries.

### Strengths
- The experimental design explicitly probes factors known to affect LID estimates, including curvature, thickness, sample size, non uniform density, and proximity to boundaries.

- The experimental section is thorough.

- The empirical design spans simple known dimension synthetic data and complex unknown dimension real data, which provides a broad stress test for the methods.

- The paper systematically highlights shortcomings of existing LID methods and existing benchmarks through controlled experiments.

### Weaknesses
- The Gaussian (IDR) experiment does not report the number of samples used, and tail regions such as standard deviation beyond 2 standard deviations require high sample sizes to get enough samples in the region to have stable estimates, especially for model trained methods like FLIPD and LIDL. The number of training and test samples should be reported across other experiments as well.

- The Funnel (IDR) experiment is placed under nearby manifolds, but as described it is a single component and seems more relevant to thin manifolds.

- The Spiral (IDR) experiment under nearby manifolds appears to have extremely low point density on the outer spiral, which can cause any algorithm to fail.

- Section 3.5 nearby manifolds uses terms like distance to the neighbor on the same manifold versus distance to the nearest neighbor in the ambient space without a precise definition, and the Funnel and Spiral experiments do not make these distances clear.

- Figure 9 uses a log x axis and allocates a large fraction of the range to fewer than 100 samples, which is hard to interpret for FMNIST where the ambient dimension is 28*28.

- Figure 23 (d) would be clearer with the x axis limited to a lower range because the current figure is unreadable.

- The paper mentions using PyTorch interpolation for resizing in Upscaled ASE and possibly other experiments, and known aliasing artifacts in this method might affect LID [1].

- Section 2 mentions audio but there are no audio experiments, and removing the audio discussion would improve focus.

- Section 3.7 cites work showing ESS is invariant to sample size for artificial datasets, but comparable results for the other algorithms on synthetic data where ground truth is known are missing.

[1] Parmar, Gaurav, Richard Zhang, and Jun-Yan Zhu. "On aliased resizing and surprising subtleties in gan evaluation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

### Questions
- What is the number of samples used in the Gaussian IDR experiment, and is it sufficient for reliable estimation in the tails? What are the number of samples used for the other experiments?

- Can the authors define precisely distance to the neighbor on the same manifold versus distance to the nearest neighbor in the ambient space, and explain how these distances are instantiated in the Funnel and Spiral experiments?

- In line 1471, it is claimed that "Arrows (MS) has many V-shaped corners as artifact of translating and rotating". Can this be further explained?

- When two arrows overlap it is claimed that RGB values are added (Line 1471), but this effect is not visible in Figure 2. Can the authors clarify this?

- Minor typos:

  - Line 141: "Sec. 1" should be "Sec. 2."

  - Line 168: It is unclear whether the manifold is S^4 or embedded in a 6 dimensional space. Please clarify the intended statement.

  - Line 242: The set description contains typos. Please correct the notation.

  - Line 312: "algorithm error" should be "algorithm's error."

  - Line 317: "Figure 8" refers to another paper but links to the Figure 8 in the current paper.

  - Line 412: "too big our computational" should be "too big for our computational."

  - Line 445: "worse to" should be "worse than."

  - Line 455: The word "while" should be removed.

  - Figure references are inconsistent, alternating between "Fig." and "Figure" throughout the paper.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work is seeks to bridge a gap in common benchmarking approaches for LID estimators: the literature typically focuses on either (1) simple manifolds of known dimension or (2) realistic manifolds of unknown dimension. To address this gap, a toolkit is proposed for designing new LID benchmarks whose (1) geometric properties are understood but which (2) resemble realistic data. Foremost among these tools is inverse domain representation (IDR), wherein a simple, well-understood manifold $M$ is embedded isometrically into the space of a real-world dataset $X$ using the principal components of $X$. In the paper, this amounts to creating images that *look* like FMNIST but have the geometric properties of spheres, curves, solid hypercubes, etc. Other tools are also introduced, e.g. warping the manifold or ways of adding ambient dimensions.

The work constructs a sequence of datasets by applying these tools to simple known manifolds with geometric properties that are pathological (at least from the perspective of common LID estimators). Four estimators are compared on these datasets - expected simplex skewness (ESS), normal bundle (NB), LIDL, and FLIPD. These estimators, and especially the latter 3 neural net-based ones, underperform compared to results on simpler datasets in their respective papers, establishing this set of benchmarks as a potential new frontier for LID estimation.

### Strengths
- In my opinion, this work does what it sets out to do. It goes a decent way towards bridging the aforementioned gap between simple LID benchmarks and real-world datasets.
- It is also clearly written.
- Its strength is in the way it systematizes methods for creating benchmarks and applies them to create new ones. Its "toolbox" is clever and original. If it gains traction, I could see it forming a new de facto leaderboard for LID estimators.
- Its main weakness is when it starts to work with individual estimators - see weaknesses for thorough feedback on this.

Nevertheless, as I said, this work achieves what it sets out to do, and I recommend it for acceptance.

### Weaknesses
Much of this work ends up being a direct comparison of 4 popular LID estimators. I think the work's main weak spots are in the specifics of this comparison:
- The work only summarizes at a high-level the results of its benchmarks, and provides almost no analysis of *why* specific LID estimators do well or poorly on the constructed benchmarks. The community is left by themselves to figure out what exactly is going on with the individual estimators. I believe this work would be more impactful if it were to propose fewer datasets but provide more explanation of the failure modes of specific estimators on each.
- Pursuant to the previous point, this work suggests no improvements nor provides clear next steps for improving the state of the art. Having experimented with these methods on a vast suite of LID estimation problems, the authors should be in a good position to recommend solutions to the highlighted failure modes.
- In my opinion, there is some nuance missing in the way the work compares estimators:
	- Efficiency goes ignored in this analysis. My understanding is that, if we were to choose the fastest possible usable hyperparameters for each method and amortize training time, FLIPD > LIDL >> NB >> ESS in terms of runtime cost. This is approximately the opposite order to the ranking in this manuscript.
	- The discussion work mostly treats these estimators as drop-in replacements for each other when in fact they each have advantages beyond error and efficiency. In particular, NB/LIDL/FLIPD are useful (1) online, for new points and (2) when the model's own LID is of interest. Some more specific advantages:
		- NB supplies normal vectors of the manifold, which have utility beyond a simple LID number.
		- LIDL is the only of these that works for normalizing flows and density estimators in general.
		- The FLIPD estimate of a point is differentiable and can be estimated in a single forward pass of its model.
		If perfect LIDs were my only consideration, ESS would always be my default choice as long as it is tractable (but it is often not).

### Questions
- See weaknesses 1 and 2. Do you have any insights on why the individual estimators perform as they do in specific scenarios and how they can be improved?
- Can you expand on these phrases and perhaps provide a specific citation if appropriate? I had understood density constancy as a requirement more for classical LID estimators than newer neural ones like NB, LIDL, FLIPD.
	- "in LIDL this bias is a function of the laplacian of the density"
	- "existing algorithms make specific assumptions during derivation -- [...] local density constancy"

### Soundness
4

### Presentation
3

### Contribution
3
