# Should Bias be Eliminated? A General Framework to Use Bias for OOD Generalization

- Decision: Reject
- Scores: 4, 6, 4, 0, 6

## Abstract
Most approaches to out-of-distribution (OOD) generalization learn domain-invariant representations by discarding contextual bias. In this paper, we raise a critical question: Should bias be eliminated? If not, is there a general way to leverage it for better OOD generalization? To answer these questions, we first provide a theoretical analysis that characterizes the circumstances in which biased features contribute positively. Although theoretical results show that bias may sometimes play a positive role, leveraging it effectively is non-trivial, since its harmful and beneficial components are often entangled. Recent advances have sought to refine the prediction of bias by presuming reliable prediction from invariant features. However, such assumptions may be too strong in the real world, especially when the target also shifts from training to testing domains. Motivated by this challenge, we introduce a framework to leverage bias in a more general scenario. Specifically, we employ a generative model to capture the data generation process and identify the underlying bias factors, which are then used to construct a bias-aware predictor. Since the bias-aware predictor may shift across environments, we first estimate the environment state to train predictors under different environments, combining them as a mixture of domain experts for the final prediction. Then, we built a general invariant predictor, which can be invariant under label shift, to guide the adaptation of the bias-aware predictor. Evaluations on synthetic data and standard domain generalization benchmarks demonstrate that our method consistently outperforms both invariance-only baselines and recent bias-utilization approaches, yielding improved robustness and adaptability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper questions the conventional treatment of bias in out-of-distribution (OOD) generalization, proposing instead that bias can be constructively leveraged for improved performance. The authors provide a theoretical framework for identifying when and how biased features contribute positively and introduce a generative model-based approach to disentangle invariant content from bias. They further propose a bias-aware prediction scheme that incorporates environment routing and adaptive bias correction with learnable label priors. Extensive experiments on synthetic data and established domain generalization benchmarks demonstrate that their approach (BAG) achieves robust improvements over both invariance-only and prior bias-utilization methods.

### Strengths
1.	The paper provides a strong theoretical analysis of the role of bias in ODD generalization, which inspires the design of the proposed framework BAG.
2.	Experiments on two benchmarks prove the effectiveness of the proposed method.
3.	The main ideas, motivations, and methodological details are generally well-presented.

### Weaknesses
1.	Theoretical validation is insufficient. Beyond performance results, the theoretical conclusions in the paper lack quantitative experimental support. I recommend adding empirical analyses that directly substantiate the theoretical claims, which would also help reinforce the motivation of the proposed method.
2.	Missing comparison with SOTA DG methods, like [1,2,3]. In addition, I am curious whether the proposed methods can be effectively applied on top of SOTA DG methods.
3.	The empirical results are only demonstrated on relatively modest-scale benchmarks(PACS and OfficeHome). Experiments on larger datasets (like DomainNet or TerraIncognita) should also be provided.

[1] Zou, Yingtian, et al. "Towards robust out-of-distribution generalization bounds via sharpness." arXiv preprint arXiv:2403.06392 (2024).

[2] Zhang, Qingyang, et al. "The best of both worlds: On the dilemma of out-of-distribution detection." Advances in Neural Information Processing Systems 37 (2024): 69716-69746.

[3] Chen, Jinggang, et al. "Gaia: Delving into gradient-based attribution abnormality for out-of-distribution detection." Advances in Neural Information Processing Systems 36 (2023): 79946-79958.

### Questions
No supplementary material that provides code, which is mentioned in L1255 that “The full code are provided in the supplementary material”.

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
2

### Summary
The paper asks whether “bias” should always be removed for out-of-distribution (OOD) generalization and proposes BAG (Bias-Aware Generalization): (1) learn disentangled content/bias features via a VAE; (2) build a bias-aware predictor using an environment-routing mixture of experts with a learned domain estimator; and (3) perform bias correction with an adaptive label prior, yielding an “invariant predictor” term that remains valid under label shift. Experiments on synthetic data, PACS, and Office-Home show gains over ERM/IRM/ACTIR and recent bias-utilization SFB.

### Strengths
1. The paper formalizes when bias can help rather than hurt, via identifiability and “unblocked influence” conditions, then operationalizes it in BAG.
2. The adaptive label prior and decomposition (Eq. 5) address a key weakness of prior SFB-like methods that assume invariant

### Weaknesses
1. Conditions A1–A4 (smooth densities, conditional independence factorization, domain variability/linearly independent shifts) may be hard to meet in complex vision tasks; practical diagnostics are not provided.
2. While block-wise identifiability is cited from prior work, the paper’s empirical section offers limited stress-tests for partial violations (e.g., non-identifiable generators, entangled factors)

### Questions
1. How sensitive is performance to the number of domain experts and embedding dimensionality? Any heuristics or validation procedure beyond source-domain accuracy?
2. If p(e∣b) is wrong (e.g., target domain outside the convex hull of sources), how does BAG degrade? Any fallback to invariant-only predictions?

### Soundness
3

### Presentation
3

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
This paper challenges the common assumption in OOD generalization that spurious or bias features should always be removed. The authors provide theoretical conditions under which bias can be identifiable and beneficial for prediction, particularly when there exists an unblocked causal path between bias and label. Building on this, they propose Bias-Aware Generalization (BAG) — an end-to-end framework that disentangles invariant and bias representations via a VAE with an independence regularizer, routes predictions through inferred environments, and corrects bias using an adaptive, learnable label prior. Experiments on synthetic and standard DG benchmarks (PACS, OfficeHome) show modest gains over ERM.

### Strengths
* **Interesting and original perspective** — The paper takes a refreshing stance by arguing that bias or spurious features are not always detrimental. This idea of *leveraging* bias for better OOD generalization is both conceptually interesting and relevant, especially given how dominant the “bias elimination” mindset has been in this field.

* **Solid theoretical framing** — The authors provide a clear theoretical justification for when and why bias can be beneficial, using identifiability conditions and causal reasoning. This helps ground the idea beyond intuition and makes the argument more rigorous than many prior discussions on bias utilization.

* **Coherent and well-structured framework** — The proposed BAG model integrates disentanglement, environment routing, and adaptive label priors into a unified end-to-end architecture. The framework is logically consistent and easy to follow, with each component corresponding naturally to a part of the theoretical motivation.

### Weaknesses
* **Lack of ablations and component analysis** — The paper does not provide sufficient ablation studies to isolate where the performance gain comes from. For instance, it’s unclear how much each part—the content predictor, bias predictor, environment routing module, or adaptive prior—actually contributes. The absence of results like *C-only vs. B-only*, or *with vs. without routing and prior correction*, makes it hard to judge whether the added modules are necessary or if the model mainly reproduces SFB’s effect in a different architecture.

* **Limited experimental scope and dataset diversity** — The experiments are conducted only on synthetic data, PACS, and OfficeHome. These are relatively small and saturated benchmarks that no longer reflect the challenges of modern OOD evaluation. The paper should consider following the **DomainBed** codebase and benchmark protocol (VLCS, PACS, OfficeHome, TerraIncognita, DomainNet) to ensure fair hyperparameter tuning and reproducible comparisons. In addition, it would be helpful to include datasets that explicitly test spurious correlations—such as **CMNIST**, **Waterbirds**, **CelebA**, **MetaShift**, or **BAR** (as used in *Project-Probe-Aggregate: Efficient Fine-Tuning for Group Robustness*)—to more convincingly demonstrate the benefit of leveraging bias. Currently, the performance improvements appear substantial on synthetic data but modest on real-world settings.

* **Overlap with prior work and unclear novelty** — The overall framework, especially the test-time refinement step and bias correction mechanism, is conceptually very close to Stable Feature Boosting (SFB). While the paper adds disentanglement and environment routing, it is not yet clear whether these bring substantial new insights or measurable benefits. Without stronger theoretical or empirical differentiation, the work feels more like an incremental extension rather than a clear step forward.

### Questions
1. Could the authors provide detailed abations to show the independent effect of the content predictor, bias predictor, environment routing module, and adaptive label prior? Which component contributes most to the final improvement?

2. Is the environment predictor $p(e|b)$ truly necessary? Would directly predicting $p(y|b)$ yield similar performance? How sensitive is the model to the number of experts $M$ in the routing mechanism?

3. How does the proposed predictor refinement differ from SFB’s bias correction step, both theoretically and empirically? Could the authors clarify the key advantage that BAG offers over SFB beyond architectural integration?

4. Are the experiments implemented within the DomainBed codebase for fair comparison? Do the authors plan to include larger or spurious-correlation datasets (e.g., Waterbirds, CelebA, CMNIST) to validate the claim that bias utilization helps in realistic settings?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper challenges the typical view that bias should be minimized for robust out-of-distribution (OOD) generalization. The authors propose Bias-Aware Generalization (BAG), a framework that leverages both stable and bias features to improve prediction performance. Through experiments on synthetic and real-world datasets, BAG outperforms traditional methods like ERM and IRM. The approach combines stable feature predictors with bias terms and uses a mixture of experts model, enhanced by soft labeling for test-time adaptation, showing that bias, when carefully managed, can enhance OOD generalization.

### Strengths
(1) The fundamental idea is reasonable. Covariate bias is indeed influential in classification and OOD-related tasks. I believe the authors’ perspective is justified. However, I believe this can be both helpful and risky. it may improve performance on certain tasks but could also lead to biased errors in other scenarios. 


(2) The authors conduct experiments on both synthetic and real-world datasets, demonstrating the effectiveness of their proposed method compared to existing approaches.

### Weaknesses
I have reviewed this paper for NeurIPS 2025, in which 5 reviewers unanimously decided to reject the paper.

In my previous review, I outlined several issues, but I found that the authors did not make substantial revisions. Even minor issues, such as typographical errors in symbols and writing, were not corrected.

Given these considerations, I have decided to reject the paper. In today’s world, where the volume of peer review work is high, I strongly recommend that the authors take each reviewer’s feedback seriously and make substantial improvements addressing the weaknesses of the work before resubmitting. The fllowing reviews also included some previous review comments that I agree with, for reference.

(1) The decomposition of content and bias representations is not empirically validated. It remains a verbal argument. How can one verify that the learned representations actually correspond to the intended content and bias factors? This is a key issue.

(2) A central part of the method’s separation of bias and content relies on the independence loss. However, this only enforces a necessary, not sufficient, condition for conditional independence.

(3) The core of the paper lies in validating the benefit of explicitly modeling and utilizing bias, which seems to be evaluable without using test-time adaptation setting. I do not see the necessity of including test-time adaptation component. If I understand correctly, the proposed method is the only one that uses test time adaptation, which gives it an unfair advantage in the empirical evaluation.

(4) I recommend that the authors use the same set of baselines across the benchmarks on which they conduct experiments. Although the authors mention that SFB did not report performance on the Office-Home dataset, it still seems feasible to compare the performance of baselines like GMDG on the PACS dataset.

(5) Large part of the claimed contributions of the paper have been derived in Stable Feature Boosting (SFB) already.

(6) Similarly, a mixture of experts method that is not too dissimilar to what the authors propose has been proposed in Prashant et al. ‘Scalable out-of-distribution robustness in the presence of unobserved confounders’ (2025), but is not referenced and at least conceptually compared to.

(7) In line 1200 of the pseudocode, is there a typo? Should it be f_c(\mathbf C) instead of f_c \mathbf C?

(8) I suggest the authors visualize e b y c x in the clearest way possible. I cannot distinguish the difference between e and b. You assume b ⊥ c | y (line 207), and also assume y is not independent of e given c (lines 245-246), but I fail to understand the rationale behind these assumptions.

(9) The writing is unclear, and there are some necessary explanations for certain concepts that would be meaningful. For example, in Lemma 2.1, what does positive support refer to, and why is this emphasized? Isn't positive support self-evident? What does $\mathcal{B}$ mean in the context of domain variability? This is not explained. Additionally, what does linear independence mean, and why is it reasonable?

(10) it is unclear what "RE" in BAG-RE refers to (w/o Regularization of IND?), and what BAG-VAE stands for. Isn't your BAG method based on VAE already? what is BAG without VAE refers to? None of these are clearly explained, and the writing is very unclear.

### Questions
see above

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper challenges the conventional view in out-of-distribution (OOD) generalization that bias must always be removed. Instead, it argues that bias can sometimes help generalization and proposes a framework (BAG) that explicitly models and leverages bias.
The authors provide a theoretical analysis identifying conditions under which bias can aid prediction,  bias-aware generative model that disentangles invariant content and bias features, and a mixture-of-experts mechanism for bias-aware prediction, guided by an adaptive label prior to handle both covariate and label shift.

### Strengths
- The paper has a fairly decent novelty contribution, challenging entrenched assumptions in DG research about bias should be eliminated or not and reframes bias as a potentially beneficial signal. Furthermore, the formalization of when bias helps prediction (via “unblocked influence”) and the proofs of identifiability and performance are useful for technical depth and novelty. The use of causal graphs and conditional independencies is well-motivated.
- BAG unifies multiple strands: causal representation learning, mixture-of-experts modeling, and adaptive label calibration — into a coherent probabilistic framework.
- Experiments are well-structured and make sense for the pipeline.
- Paper is well written and structured.

### Weaknesses
- Only tests the problem on very small datasets, making it difficult to understand or interpret whether this framework is generalisable.
- The theoretical identifiability conditions (A1–A4) require smooth, positive densities and independent latent dimensions given (e, y). These are rarely satisfied in high-dimensional deep representations. A discussion of approximate or empirical identifiability would help.
- Although disentanglement is central, the paper lacks visualizations or examples showing what the learned bias and content dimensions represent in image space. It is a bit difficult to interpret some of the results due to this oversight.
- The gains seem to be mainly on the Synthetic case, where the non-synthetic examples have minimal gain that is often within the confidence threshold.

### Questions
A discussion of the theoretical assumptions (A1-A4) would be useful to understand where this is applicable, and where it is not (aka, which datasets/domains does this make sense to apply to, and which do not make sense?)

### Soundness
3

### Presentation
4

### Contribution
3
