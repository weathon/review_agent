# ASMG: Data Structure-Aware Routing via Incremental Subspace Learning for MoE

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 4, 2

## Abstract
Mixture-of-Experts (MoE) models scale model capacity efficiently by selectively
routing inputs to a subset of specialized experts. However, their performance
critically hinges on the gating mechanism, which is typically implemented as
a shallow linear projection followed by a softmax or sigmoid activation. This
minimal design lacks the representational capacity to capture structural variations
in the input, often resulting in weak expert specialization and suboptimal routing.
To address this limitation, we propose Adaptive Structure-Aware MoE Gating
(ASMG), a data-driven gating mechanism that dynamically interpolates between
a standard learnable gating matrix and an evolving principal subspace learned
via the Generalized Hebbian Algorithm (GHA). By tracking input structure with
iterative basis updates, ASMG enables the gating function to remain both task-
supervised and structure-aware throughout training. We validate our method
through (i) a highly controlled synthetic task based on multinomial HMMs and (ii)
extensive real-world benchmarks spanning multiple domains and training regimes,
including both finetuning and pretraining. Across a wide range of evaluations,
ASMG achieves consistent gains over strong MoE baselines. Moreover, optionally
enabling unsupervised GHA updates at test time further improves robustness under
distribution shifts, offering an online adaptation mechanism that enhances standard
gating with stronger OOD resilience.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores a method for MoE gating by computing a gating weight matrix on the principal components of the inputs.  The (approximate) principal components are learned online using GHA, and the gating matrix decomposed into RV where R is in the principal component space, limited to K components.  This is combined with a vanilla gating matrix using linear combination with a learned combination parameter.  The behavior of this method is explored first on two synthetic toy data distributions (gaussian classification and HMM language stand-in), and then applied to five language tasks from GLUE and four vision datasets, with small but likely consistent improvements.

### Strengths
Limiting gating to principal components makes intuitive sense as a way to increase gating robustness, especially earlier on in training, as it could quickly align experts to work within buckets of the largest variance directions.  The approach has promise, with good analyses in the two synthetic toy sections.  The analyses in the toy data examples clearly illustrate a source of potential.

### Weaknesses
While the real-world data seems to have some consistent improvements, these are very small right now.  The mixture coefficient between plain gating and GHA gating goes towards GHA, this is also a small change (at least for the current training time and schedule).

It's also unclear how this interacts with balancing, and if the source improvements of this method is separate or overlapping from effects of balancing.  Balancing constraints for MoE are common and important, and these also lead to better expert specializations and usage, for example by reducing assignment collapse.  So it's important to understand how this method interacts with these techniques.  Does it have similar effects and reduce the need for them?  Or is it separate and a potential source of gain on top?

Overall, while there are some good ideas here, the work still seems preliminary.  Is the mixture really needed, or can it be reduced to the GHA side more aggressively (and can this or other small changes lead to larger gains)?  Does this method work with multiple expert layers (right now gains are small enough in the real-world experiments that it's unclear how much it's actually enabling itself)?

### Questions
3.2.3  says Z = RV is initialized with random R and "subsequently trained" (l.195), but then in the next sentence says "R is fixed and not updated".  This seems conflicting --- if R isn't updated, what is updated in Z?

3.2.5:  what is the cosine similarity computed over?  the text just says it's similarity between experts.  is it between gating vectors?  or something with the outputs, or expert weights?

3.3.4:  "Since hidden representations evolve dynamically throughout training, it is no longer feasible to directly optimize the full GHA-derived gating matrix Z".:  I don't see why this would necessarily be the case.  What are the issues that happen learning the GHA gating from the start?  also what is alpha initialized to?

Fig 6:  is the y axis alpha or sigmoid(alpha)?

* What are the distributions of assignments to experts for each strategy (naive, SVD/GHA, ASMG)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Adaptive Structure-Aware MoE Gating (ASMG), a new gating mechanism for MoE models designed to capture structural variations in the input, thereby addressing the limitations of conventional shallow linear routers. The method leverages the Generalized Hebbian Algorithm (GHA) to incrementally learn an evolving principal subspace of the input distribution during training and, optionally, at inference. The final routing mechanism is an interpolation between this data-driven subspace and a standard learnable gating matrix. Analyses on synthetic data, alongside experiments on real-world language and vision benchmarks, confirm that ASMG yields improvements over standard MoE baselines.

### Strengths
- The paper is well-written, clearly structured, and easy to follow.
- The application of GHA is novel, enabling the routing mechanism to adapt during both training and inference. 
- A key strength of ASMG is its capacity for online adaptation at test time, which offers a practical solution for improving model robustness under distribution shifts.
- The empirical evaluation across both language and vision tasks  demonstrates that the proposed method surpasses conventional gating strategies.

### Weaknesses
- The motivation for introducing the mixing coefficient matrix $\mathbf{R}$ is not well-established. The paper would be strengthened by an ablation study demonstrating the necessity of this component, as well as a clearer intuitive explanation for why a learnable linear combination of principal components is preferable to using the components directly.


- The proposed method constructs its structure-aware basis using GHA to find the top-K principal components of the input distribution. This formulation inherently constrains the number of basis vectors (and thus experts, $K$) to be less than or equal to the input's dimensionality $d$, i.e. $K \leq d$. This constraint runs counter to a primary advantage of sparse MoE, which is the ability to scale the number of experts far beyond the model's hidden dimension. This limitation could hinder ASMG's applicability in scenarios that require a very large pool of experts.

- The first synthetic task, Gaussian mixture classification, may be overly simplistic. Given that the data is generated from linearly separable clusters and ASMG's router is initialized by extracting principal directions from the entire training set as a pre-processing step (Section 3.2.3), the problem is significantly simplified. Consequently, the superior performance of ASMG in this setting is expected and may not generalize. The validation would be more compelling if conducted on a synthetic task with non-linear decision boundaries or more complex data structures.

- Furthermore, the insightful analyses on expert routing, representation, and collaboration are confined to this simplistic synthetic task. The paper would be more impactful if these analyses were extended to the real-world language and vision experiments to demonstrate that similar specialization behaviors occur in more complex settings.

- The paper lacks a critical ablation study on the necessity of the interpolation mechanism itself. An analysis comparing the full ASMG model to variants using only the GHA-driven matrix or only the standard learnable gating matrix would be essential to quantify the benefits of the proposed hybrid approach.

### Questions
In addition to the points raised above, I have the following questions for the authors:

- While Section 3.2.3 mentions that $\mathbf{R}$ is a fixed random matrix for the first synthetic task, it is not explicitly stated whether $\mathbf{R}$ is fixed or learnable in all experiments. Could the authors clarify this? Additionally, have you investigated whether a learnable $\mathbf{R}$ would be more beneficial than a fixed one?

- How does the proposed mechanism ensure balanced expert utilization? This is a critical factor in MoE models to prevent expert under-utilization or collapse.

### Soundness
2

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
4

### Summary
Focusing on specialized MoE routing, this paper proposes a novel interpolation method via iterative GHA. It shows good performance in various tasks and great OOD adaptation. The research problem is significant in MoE community, but there are no experiments on LLMs, which weakens the contributions.

### Strengths
- The research question of specialized gating is critical to MoE models.
- Although the iterative update costs additional computations, it brings test-time adaptation for OOD settings.
- The overall performance is good and brings new ideas to the MoE field.

### Weaknesses
- It’s a pity that this paper only conduct experiments on encoder-only models (e.g. BERT). It would be much more significant if the authors could conduct experiments on small-sized MoE models for language modeling.
    - Or, if it is impossible for you to train a model from scratch, could you convert a well-trained MoE model to GHA gating?
    - For example, OLMoE-1B-7B is a good start, and it would be better to extend to DeepSeek-V2-Lite, and Qwen3-30B-A3B (2~10B tokens would converge if only routers are trained). I understand the GHA would be compatible with current LLMs.
    - We (or at least me) really don’t care about finetuning-style GLUE at all in such an LLM era.
- The computational analysis may be biased. Although the most computational cost lies in the expert forward pass, GHA ( O(B((m+2)Kd)) + O(K^2d) ) is greater than the vanilla routing ( O(BKd) ). And your baseline should be vanilla MoE instead of DynMoE and cosine gate in Table 3.
- Algorithm 1 should be placed in the main content. The whole bunch of texts in section 3.1 really do not help readers understanding the whole process.

### Questions
- line 53: develope → develop
- line 101~104, \eta in the equation is not properly defined.
- What if the GHA is not utilized during inference? (i.e. set \sigma(\alpha) to 1.0 in evaluating GLUE benchmarks)

### Soundness
3

### Presentation
3

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
The paper proposes a novel gating mechanism for mixture-of-experts (MoE) layer which interpolates standard and principal component-aware gating mechanisms.  Generalized Hebbian Algorithm is used to approximate top-K PCs of data and allows online updates during the inference time as well. Through synthetic and real data benchmarks, they show that the proposed method can be more reliable compared to standard baselines.

### Strengths
1. **Clarity:** the method and motivation behind the method is quite clearly presented.
2. **Originality:** data structure-aware routing using GHA is an interesting touch to routing in MoE which also enables test-time adaptation at an acceptable computational cost.
3. **Significance:** Although the OOD performance improvement does not seem very substantial, I believe it is a right step forward in that robustness direction in terms of methodology.

### Weaknesses
1. **Motivation behind $R$:** The authors introduce a learnable mixing parameter $R \in \mathbb{R}^{K \times K}$. I think it is not sufficiently well-motivated AND/OR the explanation is a bit misleading. 
- (i) Why don't we want direct alignment scores with PCs and rely on the naive gating term in terms of task specific alignment? 
- (ii) Line 156 *"This creates a latent gating basis that spans the same routing subspace"*... this claim may not be true in general since $span(RV) \subseteq span(V)$ where equality holds iff $R$ is full-rank (which doesn't seem to be enforced). Would enforcing full-rankness of $R$ improve performance then?
2. **Analyzing the effect of test-time GHA:** As mentioned in Strengths section above, I think the test-time adaptation of $V$ is an attractive approach to OOD. However, I believe the OOD performance may not always prefer adapting $V$ to the test input over using a fixed $V$ from the pretraining stage. There are two ways to make this claim more convincing:
- (i) by including both train-time and test-time GHA versions in Figure 8. This is because the improvement in Figure 7 is not quite conclusive on its own since the improvement is modest and only average performance is reported.
- (ii) by testing on robustness benchmarks which has a *corruption strength* parameter such as ImageNet-C [1] and showing that test-time GHA performance degrades at a slower rate as the corruption severity increases.

___

Overall, I believe the paper proposes an interesting design to structure-aware and adaptive MoE gating mechanism with promising empirical results. Therefore, I am open to increase my score upon satisfactory responses to the concerns raised in the Weaknesses section.

___

### References

1. Dan Hendrycks, Thomas Dietterich. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR 2019

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a novel routing method in MoE models whereby the standard routing matrix of gating vectors is replaced with a set of principal basis vectors derived from the generalized Hebbian algorithm (GHA). This construction allows the router to better capture the true structure of the input distribution by better aligned with the leading principal components of the data, obtaining a data-structure-aware router which enhances routing assignments and expert specialziation.

### Strengths
1. The use of the GHA to capture the structure of the distribution and enrich the router with data-aware dynamics is intuitive, novel, and highly interesting.

2. The notation and method is excellently presented and discussed.

### Weaknesses
**Limited experimental validation**. To properly validate the efficacy of the proposed method, the authors should dedicate more of the paper to real experiments with real data. The authors only present two benchmarks, GLUE and DomainBed, and a single backbone for each task. Furthermore, despite proposing a novel router, the authors only mention one alternate routing baseline, which is the cosine router. Additionally, use of an MoE-version of BERT is quite far, in my view, from current contemporary and frontier MoE models. To better validate the empirical benefits of the authors' method, I would strongly recommend using widespread, frontier backbones such as OLMoE [1] or DeepSeekMoE [2], or even older variants such as Switch  Transformer [3], and then validating ASMG against vanilla routers and a selection of alternative baseline routing methods such as expert-choice [4] and stable MoE [5],  to name a few. If the authors are compute constrained, all of the mentioned models are available in small sizes. As it stands, however, it is difficult to be properly assess the performance of ASMG given the limited baselines, backbones, and tasks. 

**Empirical benefit is highly marginal** For what real experiments we do have, the results  seem to display extremely limited performance gains. For example, we see just 0.2% gain in the OOD setting and just 0.1% relative to GMoE. My concern is then that much of the reported gains are potentially not statistically significant. 

[1] OLMoE: Open Mixture-of-Experts Language Models (Muennighoff et al, 2024)

[2] DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models (Dai et al, 2024)

[3] Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Fedus et al, 2021)

[4] Mixture-of-Experts with Expert Choice Routing (Zhou et al, 2022)

[5]  StableMoE: Stable Routing Strategy for Mixture of Experts (Dai et al, 2022)

### Questions
1. I'd suggest the authors reduce the emphasis on synthetic experiments and focus more on assessing their method on real tasks with frontier models and baselines. The synthetic experiments are highly comprehensive and serve as an interesting case study and motivator, but, in my view, do not need to take up such a significant portion of the paper, especially if that comes at the expense of real experiments, which are more helpful for demonstrating the true performance of the method.

2. What are the consequences on load balance? Intuitively, if we're aligning the gating vectors with the principal components of the data distribution, I would be concerned that whichever experts are most aligned with the leading components will then be assigned the majority of the tokens, thereby necessarily introducing quite steep load imbalance. Is there reason why conceptually this won't happen, or some empirical results on this?

### Soundness
3

### Presentation
3

### Contribution
2
