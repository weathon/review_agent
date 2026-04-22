# Angular and Shell-Aware Deep Potential Energy Model for Molecular Dynamics

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Angular information, especially involving the first and second coordination shells, is critical for accurately describing the potential energy surface (PES) in molecular systems. 
However, existing machine learning PES models either neglect this information or indiscriminately process it from all neighbors, blurring the critical contributions of distinct shells and compromising their predictive accuracy.
In this work, we propose the Angular and Shell-Aware Deep Potential (ASDP), a novel architecture designed to overcome this limitation. 
Based on the DPA-1 attention mechanism, ASDP integrates a specialized encoding module that selectively processes angular information confined within the first two coordination shells. 
This shell-aware approach allows for a more physically meaningful representation of the local atomic environment. 
Experimental results show that by capturing crucial shell-specific angular dependencies, ASDP represents the PES of various molecular systems with the \textit{ab initio} quantum mechanics (QM) accuracy,
outperforming many existing methods and offering a new direction for creating highly accurate and robust machine learning potentials.
Our code can be found in \url{https://anonymous.4open.science/r/ASDP-ICLR-code}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Authors highlight that existing machine-learning potentials often treat angular information uniformly across all neighbors within a large cutoff, ignoring its limited physical range. To address this, they propose the new architecture  Angular and Shell-Aware Deep Potential (ASDP) that explicitly encodes angular features within the first and second coordination shells through an additive attention bias. This design aims to suppress distant, uncorrelated angular noise. The model builds upon DPA-1 and is benchmarked on several datasets with molecular and periodic systems showing improved energy and force prediction accuracy.

### Strengths
- The work introduces a physically motivated inductive bias into attention-based deep potential models by explicitly restricting angular information to atoms within the first two coordination shells. This “shell-aware” cutoff excludes distant, angularly uncorrelated neighbors, reducing noise while preserving chemically relevant geometry. The motivation is clear, and the architectural diagrams effectively show how the additive angular bias is integrated into the attention mechanism.
- The approach is benchmarked on both molecular and crystalline datasets, indicating that the proposed mechanism can generalize across different chemical domains.

### Weaknesses
- The paper essentially proposes an attention mechanism modification for the DPA family of models (DPA-1 and DPA-2) and only experimentally validates the effect of the modification on a single model (DPA-1). This raises serious concerns about the generalizability of the proposed method and the value of the paper to the broad scientific community.
- The paper proposes two changes to the attention mechanism: incorporating angular information via bias and limiting angular information to atoms within the second coordination shell. However, no ablations (to the best of my understanding) demonstrate the significance of proposed modifications individually. 
- The first four angular features were chosen without clear theoretical or empirical justification. The ablation study only compares 4 vs 6 features, never testing which of the first four are essential. This makes the feature design appear ad-hoc rather than physically motivated.
- The shell neighbor number $s = N_2$ is fixed manually for each dataset based on "chemical intuition". This prevents transferability across systems and requires manual tuning for every new domain. For molecules or disordered systems with variable coordination, such as static or lattice-based, the criterion is questionable. Moreover, The neighbor count method is density-independent and may not adapt to systems with changing local density (e.g., liquids or amorphous phases).
A continuous, distance-based smooth cutoff approximating the boundary of the second coordination shell would likely provide a more transferable and physically consistent solution. I suggest the authors evaluate this alternative in additional experiments.
- The molecular benchmarks are limited. Evaluation of the method on more recent datasets like SPICE2.0, nablaDFT or OMol25 with more complex organic and biomolecular systems would better test generalization.
- The chosen baselines (DeepPot-SE, DPA-1/2, NequIP, Allegro) don’t include stronger modern equivariant GNNs and force-field models (e.g.,DimeNet++, PaiNN, MACE ,GemNet). The paper compares mostly within DeepMD-style descriptors, so claims of state-of-the-art performance are weakly supported.

### Questions
- How sensitive are results to the manual choice of $N_2$ if we vary it by ±1? Could a small smooth spherical cutoff for angular bias perform comparably?
- Why were datasets with larger molecules or condensed-phase configurations omitted?
- Could the additive bias formulation in ASDP be integrated into existing attention MLFFs for a fairer comparison?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new model architecture for neural network based potential energy surfaces (PESs). Such PESs ideally hard code as much physics as possible (for example to improve robustness or minimize the necessary number of training samples) without introducing unphysical biases. 

It is argued that previous models introduce such a bias by including angular information in terms of cosine similarities. This way certain important physical properties are not modeled. Moreover, the different impact of angular information depending on proximity to the particle that defines the neighbourhood is not taken into account. 

Based on these observations a new model is constructed that overcomes these issues. Experiments suggest that the new model retains the accuracy of previous sota models while offering increased robustness.

While the concrete changes might be somewhat incremental, they have a concrete physical interpretation and offer new and general insights on the construction of robust PES models.

### Strengths
The paper contributes to an important field. It identifies a critical shortcoming of previous approaches and fixes it by introducing a new architecture. This is a nice and significant contribution.

### Weaknesses
The shell parameter s is fixed and requires either domain expertise or tuning. However, it is written that adaptively choosing s is subject to future work by the authors. 

Moreover, one could argue that the architectural changes are somewhat incremental to previous model.

### Questions
none

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes ASDP, a descriptor-based ML potential for molecular dynamics that injects angular information as an additive bias into attention, while being “shell-aware”: it only encodes angles formed within the first and second coordination shells. This is meant to emphasize chemically relevant geometry (directional bonds in the first shell; many-body/torsional effects in the second) and avoid noisy long-range angles. Across six molecular benchmarks, ASDP reports competitive or improved energy/force errors vs. DPA-1 and some equivariant GNNs, with good training stability.

### Strengths
1. The key innovation of ASDP lies in introducing an explicit angular inductive bias into neighborhood attention, something not explored in prior descriptor-based potentials. By encoding three-body angular information as an additive bias within the attention mechanism, the model guides the attention layer to respect local geometric structure without overcomplicating the representation. 
2. The paper includes reasonable ablations to justify design choices. In Table 2, they show that using both first and second shells (s=N2) is generally better than using no angular information (s=0) or an overly large cutoff (s=30)
3. The paper is easy to read and follow

### Weaknesses
1. A key concern is that the paper does not establish how ASDP scales to much larger or more diverse datasets. All six benchmarks have at most on the order of 10^5 training frames, while the mainstream MLIPs (eSEN, UMA, GemNet, etc) has been trained on dataset with more than 100M samples (such as OC, OMat, OMol). At least showing generalizability on a few million sample of diverse dataset, such as OMol 4M, would be helpful for understanding the usability of such approach.
2. The proposed shell-aware angular bias is a sensible innovation, but it is largely an incremental modification of the existing DPA-1 framework rather than a fundamentally new paradigm. Prior models (e.g. DimeNet, GemNet) have also incorporated angular terms and even dehegral terms into learned descriptors. The main novelty here is the additive form of the bias in attention. While the paper argues this fixes some issues of DPA-1, one could question whether this change alone warrants a full new architecture. In particular, the need for explicitly encoding angles at all (versus letting a sufficiently expressive graph network learn them) is a design choice that trades off generality for built-in structure. 
3. ASDP’s performance gains are not uniform across tasks. For some benchmarks (e.g. AlMgCu and ANI-1) the original DPA-1 model actually had lower energy errors than ASDP. The paper does not deeply discuss these cases, instead focusing on where ASDP wins (SSE-PBE, organic reaction, etc.). This raises questions: why does the more complex model underperform on some datasets? Is it due to overfitting the chosen shells, or noise from added features (as hinted by the 4 vs 6 feature ablation)? The lack of consistency suggests the benefit of ASDP’s module maybe context-dependent.
4. ASDP requires setting the shell cutoffs in advance (the number of neighbors to consider). The authors fix this per dataset from chemical knowledge. In practice this means a user must know how many neighbors form each shell for their system, which may not be obvious for new materials. Moreover, the need to tune or guess this hyperparameter for each new task reduces the model’s few-shot usability. The ablation study shows that choosing s suboptimally (too small or too large) can degrade accuracy, so its setting is crucial. The paper could be strengthened by providing guidance on selecting or learning this parameter, but as-is this reliance on domain-specific priors is a limitation.
5. Relatedly, the experimental scope, while diverse, is still limited. All datasets are from materials/chemistry; none are, say, biological macromolecules or other domains where ML potentials are used. The model’s behavior in very large-scale MD (many thousands of atoms) is not assessed. There is also no evaluation of MD simulation outcomes (e.g. energy conservation, observables) Finally, the reported results lack uncertainty measures or statistical variance.
6. Implementing the angular encoding is non-trivial: one must compute angles for all neighbor-triplets in the first two shells and generate multiple trigonometric features. This adds code complexity and computational overhead. Although the authors show inference is still competitive, training time and GPU memory requirements are not reported. It is unclear how much longer ASDP takes to train compared to DPA-1. The reviewer is concerned that for very large systems or datasets, the combinatorial cost of angles (O(s^2) per atom) could become significant. In practice, the need to hand-tune features (4 vs 6) and manage these computations could be a barrier.

### Questions
1. The paper fixes s (the second-shell cutoff) based on prior chemical knowledge. How sensitive are the results to this choice? Could s be learned or adapted automatically? It would be useful to know how ASDP performs if s is mis-specified, or if one tries a data-driven selection.
2. The paper reports inference times but does not detail training efficiency. How much longer (in wall-clock time or epochs) does ASDP take to train compared to DPA-1 on the same hardware? 
3. The authors use 6 features per angle (cos, sin, sin(2\theta), von Mises, and two distance sums/differences). Is this feature set universal, or tailored to these tasks? Could simpler or different features suffice?
4. How does the parameter count and memory footprint of ASDP compare to DPA-1 and the equivariant models? I.e. is the experiment controlled on the number of parameters?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces ASDP, which is an upgrade to the recent DPA-1 potential for learning interatomic potentials (forces). DPA-1 is a rather simple transformer based architecture based on Euclidean relative position vectors and (implicitly) dot products between them. The present paper explains that this approach has inherent limitations in terms of its expressive power. The simplest concern is that when two position vectors are perpendicular, their dot product is zero so they will not contribute to the attention scores at all. Instead, ASDP uses a rather elaborate mechanism to reoncode angular information with internal parameters in a special bias matrix that is added to the attention weights. Empirical results show that this is a possible way to improve DPA-1, although the results are far from unequivocal.

### Strengths
- Learning interatomic potential is an important, DPA-1 is a strong competitor in this field and ASDP represents one way in which it could be further improved
- The mechanism by which angular information is encoded is novel and might inspire new ideas for other architectures as well
- The architecture is motivated by some chemical intuition

### Weaknesses
- The mechanism by which the bias matrix is computed involving sines and cosines and a two-layer MLP (which is the paper calls "sophisticated") is very very ad-hoc.
- Full SO(3)-equivariant (or SE(3)-equivariant) architectures based on the representation theory of the underlying groups, which the paper refers to as spherical-harmonic based potentials, have the advantage that they do not need such extra devices because they do not suffer from the representational limitations of just dot products. While most of the SE(3)-equivariant potentials cited in the paper are classical convolutional types architectures, there exist transformer variants of these as well. For a fair comparison ASDP should really be compared to these potentials.
- Relatedly, the authors should evaluate ASDP on the standard benchmarks that the community has been using going back to QM9, not just on the six hand-picked molecular systems appearing in Table 1.
- It seems like there is a hard radial cutoff between atoms that are deemed to be in the first two coordination shells vs the rest. The existence of such a hard cutoff introduces a discontinuity in the learned representation in the space of possible configurations, which can be a problem.
- Overall, the proposed modification to DPA-1 is interesting, but a little unconvining because it is very heuristic and not thoroughly compared to the competitors.

### Questions
n/a

### Soundness
3

### Presentation
4

### Contribution
3
