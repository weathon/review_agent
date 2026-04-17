# Diagnosing Manifold Collapse: Intervention-Based Topology in Low-Rank RNNs

- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Understanding how neural circuits give rise to low-dimensional manifolds remains a central challenge in neuroscience and AI. While toroidal and ring-like topologies have been observed, the mechanisms linking connectivity to emergent geometry are not fully understood. We propose an intervention-based topological framework that integrates low-rank recurrent neural networks, persistent homology, and curvature-aware scoring to study the formation, degradation and recovery of neural manifolds. Our analysis compares perturbed and unperturbed trajectories under targeted lesions (random, axis-aligned, low-norm), introducing the Causal Topological Intervention Score (CTIS) to quantify Betti number shifts with curvature weighting. We also develop a Dynamic Betti Fingerprint for anomaly detection and evaluate resilience via the Area Under Recovery Curve (AURC). Using synthetic velocity-driven trajectories and spike-train recordings from the CRCNS pfc-7 dataset, we show that structured low-rank connectivity yields toroidal dynamics, and that CTIS captures non-linear collapse thresholds aligned with recovery trends. This framework offers a principled tool for linking interventions on connectivity to interpretable topological signatures of circuit fragility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Summary

The manuscript proposes a novel framework to study the formation of
neural manifolds and their degradation and recovery following
perturbations. The aim is to establish causal links between changes in
connectivity and subsequent changes in the topology of neural
manifolds.



Soundness

Most of the manuscript is sound. The authors build their metrics on
top of well-known topological features (Betti numbers). SHAP is
applied to gain explainability insights and to identify the most
influential measures contributing to variations in CTIS. Low-rank
networks are suitable for this analysis and provide a good testbed for
linking connectivity, dynamics, and topology. The authors also explain
the metrics in intuitive terms while providing detailed justifications
in the appendix.

A puzzling point is the authors' claim that they are able to capture
the curvature of the manifold. A related question arises from this
(see questions).

In addition, the authors mention in the Introduction and Results &
Discussion sections that the low-rank recurrent network is trained,
yet no details are given about the type of training or its effects on
the read-in or recurrent weights that produce the toroidal dynamics.


Presentation

The presentation of the work is good and provides good explanations
that can be followed intuitively, also by readers who are not experts
in the mathematics of topology.


Contribution

The main contribution is the proposal of metrics and their causal
relationship to manifold integrity and collapse.  This framework
allows the study of topological breakdown in a controlled model
setting using either artificial or biological input data.

### Strengths
I very much like the authors' style of presentation, providing the
main equations and clear intuitive explanations of the different
metrics.  The manuscript carefully dissects the causal effects of
perturbations on the various metrics.  The authors draw thoughtful
links to, and distinctions from, previously established measures.

### Weaknesses
PCA-based method for measuring curvature requires additional
description and clarification in the main text.  The training
procedure of the recurrent neural networks (RNNs) is not described.
If I understand correctly, Figure 3 should summarize all sessions,
including sessions 1 and 8 shown in Figure 2 above, but the detected
anomaly events do not seem to match.

### Questions
My concern on the PCA method be in able to capture the curvature of the manifold (see "soundness").

At least based on the global PCA eigenvalue spread described in Section 3.2, this does not appear to be the case.
While we do not doubt that the PCA spectrum may correlate with the fragility or stability of the manifold, PCA is a linear method that only captures the Gaussian variance structure of the data.
It effectively approximates the data by an n-dimensional hyperplane rather than a curved manifold given the first n eigenvalues are retained.
Therefore, the proposed curvature proxy in Eq. (4) seems to reflect dimensionality rather than curvature per se.

In the methodological clarifications, the authors mention a 'local embedding' rather than the global data.
We acknowledge that if PCA is performed locally and its variation across the manifold is analyzed, such an approach could indeed capture curvature-related effects.
We therefore suggest that the authors clarify precisely what they mean by 'local PCA' and briefly explain why this can be interpreted as a measure of curvature in the main text.
Otherwise, the current description could be misleading.

It would be interesting to see whether SHAP would then ascribe more importance to the curvature.

In case these points can be clarified in the rebuttal I am happy to raise my score.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors study the effects of destructive interventions to a low-rank RNN on a variety of topological features, analogizing their study to the degredations to qualitative neuronal activity that occur during degenerative disease. They first define a series of topological (and other) signatures to be measured from their RNN: a residual variance after PCA on trajectories, Beti numbers measured over sliding windows, and differences in Betti numbers before and after interventions. These interventions involve zeroing-out weights in the RNN at a given time. The signatures are then measured on the interventions on synthetic toroidal dynamics and action potential measurements from prefrontal cortex.

### Strengths
1. I think the question the authors address is interesting and timely. From what I can tell, there is not much work on detecting or measuring changes in topological structure through persistent homology after perturbations. In that sense, the contribution is conceptually novel. 

2. The authors provide several different metrics for detecting topological changes, which on the one hand is nice as it should provide a fuller picture of these changes, though as I will mention below also make the approach seem scattered. 

3. I appreciate the effort to use real neural data, and I think this shows that the proposed metrics can--at least in principle--be applied to this data. This lays the groundwork for future work to investigate topological changes in neural activity.

### Weaknesses
1. The paper feels overloaded with metrics and lacking in deeper analysis. I count around five scores (Betti_Series_t, CTIS, AURC after recovery, Betti sensitivity, Anomaly_t) that are sometimes interconnected. I appreciate the approach to be thorough, but given the weakness of the empirical results overall, I believe it would have been better to focus one or another metric to be explored in greater depth. As it stands, I find it rather hard to read and remember each score, which are only very briefly introduced. Also, some scores do not make much sense to me. The explained variance gap $\gamma_g$ is reported as "[correlating] with the fragility of topological features." This is vague, and not immediately intuitive. A long thing (i.e. anisotropic) structure with low $\gamma_g$ could also be argued to be topologically "fragile", no? One small bump and you're out of the structure, perhaps changing the topology. Without clearer explanation, it's hard to know if my intuition is right or wrong. It's further unclear why these values should be used in weighting CTIS. On this quantity, are pre/post measured on adjacent ablations (e.g. .1 vs .2 and .2 vs .3 in Fig 1a) or only between no ablations and stronger ablations (e.g. 0 vs .1 and 0 v .2 and 0 vs .3, etc.)

2. Overall, the empirical results are unfortunately not very strong. I list some areas of weakness figure by figure. 

Fig. 1a. The peak at .4 probably represents something, but it's not clear immediately what this is. Again, can you clarify what CTIS measures? Between no ablation and increasingly strong ablations or between neighboring ablation values? If the former, it's not clear why the CTIS wouldn't permanently jump to ~12. And why not go all the way to full ablation? 

Fig. 1b. This is very hard to interpret without seeing the manifold before. Can you do a side-by-side of all of the manifolds under ablation, ideally highlighting where you think the structure disappears? 

Fig. 2a. The "sparse transient fluctuations" are not easy to see, both because the effect is seemingly small and the figures are not formatted for easy viewing. If Betti 1 is the important one and the other Betti numbers have different scales, then simply show Betti 1. Further, it is hard to know what to think of these detected anomalies. If you have no ground truth or comparison between subjects/conditions, how are we to know that these fluctuations are not just a fluke of your thresholding? Ideally we'd see, say, in condition 1 you have the fluctuations but in condition 2, you don't. Could you simulate this on your synthetic toroidal data? 

Fig. 3. My previous results on anomaly detection hold here. If there is no systematic difference between sessions related to ground truth signals or labels, then it is not clear what these anomalies mean. Put another way, maybe you just always get some small fluctuations in noisy data that gets picked up as changes in Betti 1? True, there is work saying that PH is stable to small amounts of noise, but you need to argue in detail why you believe your noise is small and these anomalies are, therefore, indicative of a real topological signal. 

Fig. 4a. It is hard to know how to compare the effects of the different lesion types since they are not balanced by magnitude. For example, maybe a low_norm ablation is more destructive in some absolute sense than an axis_aligned ablation, so the changes in CTIS are superficial. Is there any way to control for this? 

Fig. 4b. As the authors say, the results are not significant. The Bayesian ANOVA shows that changes in Betti numbers vary differently with lesion type, but what is the significance of this if recovery time is not significantly related to CTIS? Also, the main text says that CTIS has value as a "geometry-aware diagnostic metric", but aren't we talking about topology? This is an important terminological distinction I believe. 

Fig. 5,6. I appreciate the attempt to interpret the results with SHAP (which should be explained more in the main text for unfamiliar readers like myself) and sensitivity measures, but my point still stands that if CTIS is not clearly measuring anything significant, then it is not worth explaining or interpreting. 


3. I find the figures hard to read. Could axis labels for smaller figures be increased?

### Questions
1. I think the paper would start to get a good footing if the authors provided one simple, intuitive example which shows how a lesion is clearly detected by CTIS (or whichever metric) where it *should* be. By *should*, I mean that it would be on an example where we can see by eye that the topology has changed in the Betti sense. So, my first question is, could the authors provide this example, ideally by visualizing the change in topology in an obvious way and showing CTIS peaks at that moment? I believe this is what the synthetic experiments intended to be, but it is not convincing for the moment. I would be willing to start bumping my scores up if I saw this clean toy example. 

2. Can you explain your reasoning about $\gamma_g$? I don't see the intuition about topological fragility and isotropy.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes an intervention-based topological framework for studying how recurrent connectivity shapes and disrupts low-dimensional neural manifolds. The authors use low-rank RNNs as model systems and apply a combination of UMAP (for embedding), persistent homology (for Betti-number extraction), and curvature-weighted metrics to quantify manifold collapse under different perturbation types. The central metric, the Causal Topological Intervention Score (CTIS), summarizes changes in Betti numbers and curvature across “lesions” in network weights. Experiments are performed on both synthetic velocity-driven trajectories (toroidal dynamics) and spike-train recordings (CRCNS pfc-7).

While the paper raises an interesting question—how structural perturbations relate to changes in representational topology—the proposed framework appears highly empirical and under-validated. The strong reliance on UMAP, lack of sensitivity and baseline comparisons, and limited statistical support make it difficult to assess either robustness or novelty in a convincing way.

The paper explores an intriguing question and presents a creative combination of geometric and topological tools. However, the empirical results are not sufficiently rigorous to support the main claims. The dependence on UMAP, absence of sensitivity and baseline analyses, lack of statistical support, and limited connection to concrete neuroscience or ML use cases collectively weaken the contribution. With stronger validation, clearer visual explanation, and a more principled discussion of the embedding choice, this line of work could become more impactful.

### Strengths
* Interesting conceptual framing: The “intervention-based” idea of connecting network perturbations to topological changes is creative and potentially useful for both computational neuroscience and model diagnostics.
* Brings together multiple perspectives: The integration of low-rank RNNs, topology (persistent homology), and causal perturbations is conceptually appealing, and could serve as a prototype for connecting geometry and causality in neural dynamics.
* Potential cross-disciplinary relevance: The questions about “manifold/circuit fragility” and recovery could be meaningful if extended to empirical neuroscience or robust ML systems.

### Weaknesses
* Overreliance on UMAP without justification: The entire analysis depends on UMAP embeddings, yet the paper neither explains why UMAP is appropriate nor evaluates whether the results are stable under different embeddings or hyperparameters. Given that UMAP is known to be sensitive to parameters such as n_neighbors and min_dist, this is a serious limitation. At the very least, the paper should test robustness empirically or justify the choice theoretically.
* Lack of statistical rigor: Several figures (e.g., Fig. 1A, Fig. 4) lack error bars or show wide overlaps, making trends ambiguous. For instance, the sudden CTIS drop between 0.4–0.5 lesion fraction in Fig. 1A is unexplained. Without uncertainty quantification or multiple runs, the evidence for the claimed non-linear “collapse threshold” is weak.
* No baseline or ablation comparisons: The paper introduces several post-hoc metrics (CTIS, AURC, curvature weighting) but does not compare them against simpler alternatives (e.g., using raw persistent homology, PCA instead of UMAP, or unweighted Betti differences). As a result, it is unclear what value CTIS adds.
* Weak connection to neuroscience or ML relevance: While the “intervention” framing is promising, the work does not clearly link its findings to real neuroscientific hypotheses or machine-learning implications. It reads more as an exploratory RNN case study than a framework addressing a specific gap.
* Limited interpretability and missing illustrations: Given the geometric and topological focus, the paper should include schematic diagrams showing how manifold structures deform or collapse under lesions. The absence of visual intuition makes the framework hard to follow.
* Writing clarity: The exposition is serviceable but somewhat descriptive; motivation and framing could be tightened to emphasize the research question and the intended scope of generalization.

### Questions
1. How robust are the results to UMAP parameters (n_neighbors, min_dist) or to alternative embeddings (e.g., PCA, Isomap)?
2. Could you provide a clear interpretation of the sharp CTIS drop between 0.4–0.5 lesion severity in Fig. 1A?
3. Have you tested whether your results hold without curvature weighting or without UMAP (i.e., using direct persistent homology on high-dimensional trajectories)?
4. What specific neuroscientific or ML problems could CTIS help diagnose beyond this toy low-rank RNN setup?
5. Can you include schematic figures to illustrate intuitively how manifold collapse is detected and what CTIS represents?

### Soundness
2

### Presentation
2

### Contribution
2
