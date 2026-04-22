# Generalised Flow Maps for Few-Step Generative Modelling on Riemannian Manifolds

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Geometric data 
and purpose-built generative models on them have become ubiquitous in high-impact deep learning application domains, ranging from protein backbone generation and computational chemistry to geospatial data. 
Current geometric generative models remain computationally expensive at inference---requiring many steps of complex numerical simulation---as they are derived from dynamical measure transport frameworks such as diffusion and flow-matching on Riemannian manifolds. In this paper, we propose Generalised Flow Maps (GFM), a new class of few-step generative models that generalises the Flow Map framework in Euclidean spaces
to arbitrary Riemannian manifolds. We instantiate GFMs with three self-distillation-based training methods: Generalised Lagrangian Flow Maps, Generalised Eulerian Flow Maps, and Generalised Progressive Flow Maps. We theoretically show that GFMs, under specific design decisions, unify and elevate existing Euclidean few-step generative models, such as consistency models, shortcut models, and meanflows, to the Riemannian setting. We benchmark GFMs against other geometric generative models on a suite of geometric datasets, including geospatial data, RNA torsion angles, and hyperbolic manifolds, and   
achieve state-of-the-art sample quality for single- and few-step evaluations, and superior or competitive log-likelihoods using the implicit probability flow.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Generalised Flow Maps (GFM), a unified framework for few-step generative modeling on Riemannian manifolds. It generalizes the notion of flow maps to non-Euclidean geometries such as spheres, tori, and Lie groups.
The core idea is to replace the Euclidean linear interpolant with geodesic-based Riemannian interpolation and to define manifold-consistent flow maps that jump between points along the probability flow ODE.
The authors derive three generalized self-distillation objectives, Generalised Lagrangian, Eulerian, and Progressive Flow Maps.
Empirically, GFM achieves state-of-the-art few-step generation on diverse geometric domains, such as protein torsion angles, geospatial data, and synthetic rotations.

### Strengths
* The paper extends flow-map-based generative modeling from Euclidean to general Riemannian manifolds, encompassing multiple geometric structures (tori, spheres, Lie groups, hyperbolic spaces).

* The paper is theoretically rigor.

* The paper tries to bridges recent few-step generation paradigms (Consistency Models, Mean Flows, Shortcut Models) with the geometric setting via G-LSD, G-ESD, and G-PSD objectives, which is a appealing story.

### Weaknesses
A potential weakness of this work lies in its limited empirical validation on large-scale or high-dimensional settings, where the proposed few-step framework would be most impactful. The observed gains are primarily evident at very few inference steps (as shown in Figures 3 and 4) and on relatively small datasets, making it unclear how well the method scales to more complex manifolds or realistic applications.
Unlike Euclidean diffusion or flow matching models, where few-step sampling is crucial for accelerating large-scale tasks such as protein generation or real-time image editing, the computational benefits in Riemannian domains remain less compelling.
In particular, for manifolds like torsion or angle spaces, existing diffusion or flow models already achieve fast inference, so the advantage of adopting a few-step Riemannian formulation is not yet convincingly demonstrated.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces the Generalized Flow Maps (GFM) framework, which extends the flow map matching framework to Riemannian manifolds. It further generalizes the self-distillation loss from previous works to the Riemannian setting and demonstrates that each variant can be efficiently learned by neural networks. Experiments on RNA backbones, protein side-chains, and various toy datasets validate the empirical performance of the proposed methods.

### Strengths
1. The paper is well-written, and the proposed method is easy to follow and understand.
2. It extends the Flow Map Matching framework [1] to general Riemannian manifolds.


[1]. Boffi, Nicholas Matthew, Michael Samuel Albergo, and Eric Vanden-Eijnden. "Flow map matching with stochastic interpolants: A mathematical framework for consistency models." Transactions on Machine Learning Research (2025).

### Weaknesses
1. **Limited experimental scope.** The experiments are restricted to relatively simple toy datasets. As shown in previous works, Riemannian generative models have broad applications in various domains, such as material generation [1], molecular conformer generation [2], and protein backbone generation [3]. As mentioned in the paper, the inference processes of these models are often inefficient. Therefore, additional experiments on real-world applications would strengthen the paper’s contributions and demonstrate that the proposed methods can indeed improve the inference efficiency of Riemannian diffusion models.

2. **Applicability to general manifolds.** The proposed method appears to be applicable only to well-known manifolds with closed-form geodesics. In addition, it relies on an embedding of the manifold into Euclidean space. Consequently, for more complex manifolds without such prior geometric knowledge, the method would be difficult to apply in practice.

3. The results appear less innovative, as the proposed method can be seen as a direct generalization of existing approaches in Euclidean space [4, 5]. However, this would be acceptable if the method proves to offer meaningful benefits in real-world applications (see Weakness 1).


[1]. Miller, Benjamin Kurt, et al. "Flowmm: Generating materials with riemannian flow matching." arXiv preprint arXiv:2406.04713 (2024).

[2]. Jing, Bowen, et al. "Torsional diffusion for molecular conformer generation." Advances in neural information processing systems 35 (2022): 24240-24253.

[3]. Yim, Jason, et al. "SE (3) diffusion model with application to protein backbone generation." arXiv preprint arXiv:2302.02277 (2023).

[4]. Boffi, Nicholas Matthew, Michael Samuel Albergo, and Eric Vanden-Eijnden. "Flow map matching with stochastic interpolants: A mathematical framework for consistency models." Transactions on Machine Learning Research (2025).

[5]. Geng, Zhengyang, et al. "Mean flows for one-step generative modeling." arXiv preprint arXiv:2505.13447 (2025).

### Questions
1. Does the proposed method require the geodesics of the Riemannian manifold to have a closed-form expression? In addition, does it rely on an embedding of the manifold into a Euclidean space? I would like to confirm whether my understanding is correct.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Generalised Flow Maps (GFM), a novel framework for few-step generative modeling on Riemannian manifolds. It addresses the high computational cost of geometric diffusion and flow-matching models by directly learning the flow map (integrator), enabling single or few-step high-quality sampling. The approach generalizes existing Euclidean few-step models and achieves state-of-the-art sample quality across various geometric datasets.

### Strengths
GFM is tested on TNA torsion, Earth dataset, SO3, and Hyperbolic spaces. The exponential results are well proposed.

### Weaknesses
The experiments are demonstrated on relative low-dim spaces, unlike the Euclidean algorithms, e.g., Meanflow, that can be applied to large datasets.

In the case with relatively large NFE, e.g., NFE=8, there is almost no improvement. (Fig. 4)

### Questions
Why in Tab. 3 volcano dataset the proposed algorithm worse than several existing algorithms? Is it because lack of data since the volcano has only <1000 data.

The Euclidean version that requires fewer NFE algorithms, in my view, highly depends on the flat property of the space. Will the curved space itself lead to some difficulty in GFM? Do you have some intuition?

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
This paper introduces Generalised Flow Maps (GFM), a new class of generative models for Riemannian manifolds that enables fast, few-step inference. It generalizes Euclidean flow map principles by proposing three self-distillation training methods: Generalised Lagrangian, Eulerian, and Progressive Flow Maps (G-LSD, G-ESD, G-PSD). These methods are derived from three equivalent theoretical conditions that characterize a manifold-constrained flow map. Experiments on various geometric datasets (tori, spheres, SO(3), hyperbolic) show GFMs achieve state-of-the-art sample quality in single- and few-step evaluations, outperforming traditional geometric generative models that require many simulation steps.

### Strengths
**Originality & Significance**: The paper's core contribution is the novel and significant generalization of flow map learning from Euclidean spaces to arbitrary Riemannian manifolds. This directly addresses the critical bottleneck of slow inference in existing geometric generative models.

**Quality & Clarity**: The work is technically sound, providing a rigorous theoretical foundation for the new GFM variants. The empirical validation is comprehensive, testing across a diverse set of non-trivial manifolds and demonstrating clear, state-of-the-art performance in few-step sampling. The paper is exceptionally well-written and easy to follow.

### Weaknesses
**Unexplained Performance Gaps**: The three GFM variants are derived from theoretically equivalent conditions but show vast empirical performance differences (e.g., G-LSD is far superior to G-ESD in Table 2). The paper notes this but lacks an in-depth analysis of why, which is a key practical limitation.

**Missing Training Cost Analysis**: The paper focuses exclusively on inference speed. However, the GFM objectives add a self-distillation loss to the standard RFM loss, implying a more expensive training procedure. This training cost trade-off is not discussed or quantified.

**Validity of NLL Metric**: The NLL is computed using the "instantaneous velocity" (the implicit vector field $v_{t,t}$). Given that GFMs are trained to learn the global flow map $X_{s,t}$ for few-step sampling, how valid is the NLL of the instantaneous flow as a primary measure of model quality?

**Missing Protein Backbone Experiments**: The introduction prominently features protein backbone generation ($SE(3)^N$) as a key high-impact application. Why were experiments on this crucial manifold omitted in favor of simpler torsion angle datasets?

### Questions
See **Weaknesses**.

### Soundness
2

### Presentation
3

### Contribution
3
