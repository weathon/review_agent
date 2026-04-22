# Riemannian Variational Flow Matching for Material and Protein Design

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
We present Riemannian Gaussian Variational Flow Matching (RG-VFM), a geometric extension of Variational Flow Matching (VFM) for generative modeling on manifolds. Motivated by the benefits of VFM, we derive a variational flow matching objective for manifolds with closed-form geodesics based on Riemannian Gaussian distributions. Crucially, in Euclidean space, predicting endpoints (VFM), velocities (FM), or noise (diffusion) is largely equivalent due to affine interpolations. However, on curved manifolds this equivalence breaks down. We formally analyze the relationship between our model and Riemannian Flow Matching (RFM), revealing that the RFM objective lacks a curvature-dependent penalty -- encoded via Jacobi fields -- that is naturally present in RG-VFM. Based on this relationship, we hypothesize that endpoint prediction provides a stronger learning signal by directly minimizing geodesic distances. Experiments on synthetic spherical and hyperbolic benchmarks, as well as real-world tasks in material and protein generation, demonstrate that RG-VFM more effectively captures manifold structure and improves downstream performance over Euclidean and velocity-based baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Riemannian Variational Flow matching (RG-VFM), a method that is designed for conducting generative modeling on manifolds. RG-VFM extend variational flow matching to general geometric objects. RG-VFM replaces the velocity matching objective in RFM with a distribution matching objective that is analogue to VFM, but with the distribution defined on a Riemannian manifold. The authors then compare the Jacobi fields of RG-VFM and RFM and show theoretically that such a reparameterization of regression objective is capable of capturing the high-order curvature of the manifold. Empirical results suggest the success of RG-VFM in capturing curvature information and modelling geometries.

### Strengths
- The formalization and mathematical derivations are lucid and well-presented.

- The manuscript is well-structured with a clear logical progression, making the core arguments easy to follow.

- The novelty of the proposed method and its key distinctions from prior literature are clearly articulated.

### Weaknesses
My primary concerns relate to the empirical evaluation and the clarity of the motivation.

- Empirical Results: The significance of the empirical results is not entirely convincing. In the real-world protein backbone generation benchmarks, the high standard deviations suggest that the performance improvement over existing models may not be statistically significant. Furthermore, for the structure prediction results, standard deviations are not reported, which makes it difficult to ascertain the significance of the reported performance gains.

- Motivation: In the abstract, the authors state that the equivalence of endpoint, velocity, and noise prediction breaks down on curved manifolds. It is not immediately clear how RG-VFM alleviates this issue, or how this observation directly motivates the proposed method. I would appreciate clarification if this point was addressed in the manuscript and I overlooked it.

### Questions
- Motivations: Could the authors elaborate on how RG-VFM addresses the breakdown of equivalence between endpoint, velocity, and noise prediction on curved manifolds? Furthermore, how does this challenge directly motivate the design of RG-VFM over other approaches?

- More clarification on the incorporation of high-order curvature. Can I get more clarification on what is the practical impact of the higher-order curvature term on the learning problem? For example, does its inclusion confer specific advantages when learning on heterogeneous manifolds against homogeneous ones?

- Computational Complexity: Could the authors comment on the change of computational complexity when learning with RG-VFM? If learning the higher-order curvature information using RG-VFM induces more computations, relative to methods like RFM?

- Empirical Reults (Table 1): Why does the external view of RG-VFM exhibit significantly better performance when learning on the hyperboloid manifold? Is this phenomenon specific to learning on Riemannian manifolds, or could it also occur in Euclidean settings under certain conditions?

- This question is for my own curiosity: The relationship between RG-VFM and RFM appears analogous to that between Transition Matching [1] and Flow Matching in Euclidean space. Could the authors comment on this analogy? Specifically, is the model effectively learning a conditioned transition kernel, approximating a jump from $x_t$ to $x_1$?


[1] Transition Matching: Scalable and Flexible Generative Modeling. N. Shaul et al. NeurIPS 2025

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper shows theoretically that training (variational) flow matching on end-point prediction in Riemannian geometry yields a more precise loss, that better accounts for the structure of the manifold. They validate empirically their losses on a variety of datasets, including large-scale ones with applications for science.

### Strengths
- The paper is well-presented: the text is clear and simple.
- The proofs provided in the appendix are well-detailed.
- Proposition 4.1 (and 4.3) is an interesting contribution, showing a limitation in Riemannian Flow Matching [2] – the baseline method for Riemannian generative modelling – and demonstrating the method’s superiority in accounting for curvature terms, theoretically.
- The proposed fix is very simple.
- The empirical validation is rather extensive and convincing for the given examples, including large-scale ones. Overall, the method improves (arguably slightly) over the provided baselines.

### Weaknesses
- I am not certain about the initial trichotomy (or the impression that is given when stating is as follows) of “endpoint (VFM), a velocity (FM/RFM), noise (diffusion)”. Diffusion can be trained all three ways, and noise prediction was introduced in DDPM [1], as it empirically produced better performance. Same goes for flow matching. (Indeed, though, because of the linearity in the path, all reparameterisations are *theoretically* equivalent.)
- It seems to me that the words “intrinsic” and “extrinsic” are not used properly. If I am not mistaken, intrinsic (informally) “live directly on the considered space”; extrinsic coordinates depend on the ambient space. For instance, intrinsic on the 2-sphere would be the angles $\theta$ and $\phi$; extrinsic would be the usual 3D coordinates. Here, the authors seem to use it for projected on the manifold or not. (See Figure 2.) Modelling intrinsically is known to be difficult.
- I would argue that the paper is not particularly novel on the methods front (not that it is without contributions), mostly (variational) flow matching on Riemannian geometries.
- Sampling from the Riemannian Gaussian is not trivial on certain manifolds. It seems that the authors have omitted this discussion in the paper.
- Perhaps I have missed out on this, but it seems that there is no discussion on the hypotheses made about the manifold (homogeneity and existence of closed form geodesics).
- Training for end-point prediction is not new [3].

### Questions
- Did I misunderstand your usage of the words “intrinsic” and “extrinsic”?
- What is the novelty in your method [3], beyond the theoretical guarantees: isn’t end-point prediction a rather well-known “trick”?
- Does homogeneity alongside existence of closed form geodesics not trivialise the structure of the manifolds considered? (I am not certain at all, I would like to discuss.)
- Have you had the opportunity to try out your method on manifolds where the higher order terms you mention appear (non-constant curvature)? I understand that you might not have, as the examples could be rare, but in which case the contribution of the paper is less significant, practically speaking. This question can also be understood as “Why does Riemannian Flow Matching work so well if it misses those important terms?” (or are they not so important therefore?)

Overall, I think the paper is rather sound and a good read, but the contribution feels relatively small.

### References

[1] Jonathan Ho, Ajay Jain, Pieter Abbeel. “Denoising Diffusion Probabilistic Models”

[2] Ricky T.Q. Chen, Yaron Lipman. "Flow Matching on General Geometries"

[3] Bowen Jing, Bonnie Berger, Tommi Jaakkola. "AlphaFold Meets Flow Matching for Generating Protein Ensembles"

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces **Riemannian Gaussian Variational Flow Matching (RG-VFM)**, a geometric extension of Variational Flow Matching (VFM) for generative modeling on manifolds.  
The authors argue that on curved manifolds, endpoint prediction (as in VFM) provides a stronger learning signal than velocity prediction (as in Flow Matching, FM, or Riemannian FM, RFM) because it directly minimizes **geodesic distances**.  

Key contributions:
1. **A variational flow matching objective on general Riemannian manifolds**, employing Riemannian Gaussian distributions with closed-form geodesics.  
2. **A formal analysis linking RG-VFM to RFM** via Jacobi fields, showing that curvature-dependent terms naturally emerge in the RG-VFM objective but are absent in RFM.  
3. **Experiments** on synthetic (spherical/hyperbolic checkerboards), **materials (MOF)**, and **protein backbone generation** tasks. Results indicate improved geometric consistency and generation quality over Euclidean and velocity-based baselines.

### Strengths
1. The paper provides a clear and well-motivated theoretical bridge between **variational** and **geometric** generative modeling. The idea of introducing curvature-dependent terms via Jacobi fields to analyze flow-matching losses is novel and elegant.  
2. The derivation is rigorous, with propositions and proofs connecting RG-VFM and RFM. The mathematical treatment of curvature effects is insightful, especially Proposition 4.3, showing curvature-dependent correction terms.  
3. The paper is well written and technically organized, with clear mathematical notation and well-illustrated figures (e.g., Fig. 1–3 showing geometric intuition).

### Weaknesses
1. The method is limited to manifolds with **closed-form geodesics**. This assumption restricts applicability to simple spaces (e.g., \(S^n\), \(H^n\)), leaving open how to handle more general manifolds.  
2. While results are positive, the experiments lack deeper ablations (e.g., sensitivity to curvature magnitude, loss variants, or effect of variance parameter σ in the Riemannian Gaussian).  
3. The paper briefly mentions that RG-VFM maintains simplicity of linear flows but does not analyze training/sampling cost relative to RFM or diffusion models.

### Questions
1. How would the approach generalize to manifolds without analytic exponential/log maps (e.g., learned manifolds from data)?  
2. Could the curvature-dependent term be approximated or regularized for arbitrary manifolds, enabling broader applicability?  
3. In Table 3, could the authors provide variance/error bars for the protein design metrics to quantify statistical significance?  
4. Is there any insight on the trade-off between extrinsic (RG-VFM-R³) and intrinsic (RG-VFM-M) variants beyond the empirical performance?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on extending Variational Flow Matching (VFM) to Riemannian manifolds through the proposed Riemannian Gaussian Variational Flow Matching (RG-VFM). The authors argue that endpoint prediction provides a stronger learning signal by directly minimizing geodesic distances. The method formulates a variational flow matching objective based on Riemannian Gaussian distributions, applicable to manifolds with closed-form geodesics. They further demonstrate with experiments on spherical, hyperbolic, and real-world datasets that RG-VFM can better captures manifold geometry.

### Strengths
- Very well-written paper with excellent explanations and visuals.
- Variational extension of FM is well motivated; and this paper lifts this idea to Riemannian manifolds.
- The derivations and design choices are well-justified and appear natural.
- Good empirical results in two different real-world domains and on one synthetic dataset.

### Weaknesses
- The work appears somewhat incremental, as extending Riemannian flow matching to a variational formulation is relatively straightforward. Formally establishing a connection between RG-VFM and RFM through Proposition 4.1 is useful, but not very surprising.
- The discussion of intrinsic versus extrinsic viewpoints is interesting, yet it remains unclear how the extrinsic perspective offers a clear advantage. While it provides additional flexibility, it essentially represents a trade‑off rather than an optimal method. Similar to the difference between VFM and FM, the primary benefit seems to arise mainly from supervision at the endpoints.
- Although the ability to choose an arbitrary variational distribution is often presented as a main strength of variational flow matching, this flexibility is rarely reflected in practice, since most experimental setups still rely on simple distributional choices. It is not easy for the reader to establish a direct connection between this generic motivation and the experimental setups (esp. for readers less familiar with these benchmarks). It would be much nicer if the authors could elaborate how this type of generality is used in practical setup.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
4

### Contribution
2
