# From One to Many: Trajectory Invariant Learning for Multimodal Large Language Model Editing

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Knowledge editing emerges as a crucial technique for efficiently correcting incorrect or outdated knowledge in large language models (LLM). Existing editing methods for unimodal LLM rely on a rigid parameter-to-output mapping, which causes causal-underfit and causal-overfit in cascaded reasoning for Multimodal LLM (MLLM). In this paper, we reformulate MLLM editing as an out-of-distribution (OOD) generalization problem, where the goal is to discern semantic shift with factual shift and thus achieve robust editing among diverse cross-modal prompting. The key challenge of this OOD problem lies in identifying invariant causal trajectories that generalize accurately while suppressing spurious correlations. To address it, we propose ODEdit,a plug-and-play invariant learning based framework that optimizes the tripartite OOD risk objective to simultaneously enhance editing reliability, locality, and generality.We further introduce an edit trajectory invariant learning method, which integrates a total variation penalty into the risk minimization objective to stabilize edit trajectories against environmental variations. Theoretical analysis and extensive experiments demonstrate the effectiveness of ODEdit. Our code is available at
https://anonymous.4open.science/r/ODEdit-2756.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This article formalizes the knowledge editing of Multi Modal Large Models (MLLM) as a cross modal OOD generalization problem for the first time, pointing out that traditional "single modal" editing methods can lead to causal underfit and causal overfit in MLLM. To this end, the author proposes the plug and play framework ODEdit, which explicitly suppresses false associations related to the environment while maintaining cross modal semantic consistency through triple OOD risks (reliability, locality, generalization) and editing trajectory invariant learning (ETIL) constraints. A large number of experiments have shown that ODEdit consistently improves four indicators of multiple baselines (WISE, MEND, T-Patcher, UniKE) on the MMEdit benchmark, and has theoretical convergence guarantees.

### Strengths
- Provides a novel theoretical perspective by reformulating MLLM editing as an out-of-distribution (OOD) generalization problem , introducing insightful concepts like "causal underfit" and "causal overfit"  to diagnose editing failures.

- Proposes a comprehensive optimization framework based on a tripartite OOD risk objective (reliability, locality, generality) , implemented via NLL , KL divergence , and MMD.

- Introduces the innovative ODEdit framework, featuring Edit Trajectory Invariant Learning (ETIL) , which provides a novel, theoretically-grounded solution by integrating Invariant Risk Minimization (IRM) and a Total Variation (TV) penalty to stabilize editing trajectories .

- Designed as a "plug-and-play" framework , ODEdit is empirically shown to consistently improve performance across diverse existing editing baselines, including both parameter-adjusting and model-extending methods

### Weaknesses
- The ODEdit framework introduces significant computational overhead. It requires expensive MMD calculations (with multi-scale kernels) and a complex primal-dual optimization for the IRM-TV objective . This complexity far exceeds that of baselines like MEND or IKE , yet the paper provides no quantitative analysis (e.g., wall-clock time per edit, peak memory usage) of this practical cost.
- The empirical validation is constrained to relatively small-scale and early-generation MLLMs (BLIP2-OPT 2.7B and MiniGPT-4 7B) . The framework's effectiveness and scalability on more recent, larger, or architecturally different MLLMs (e.g., LLaVA, Qwen-VL) remain unverified .
- Experimental validation is confined to the MMEdit benchmark, which primarily consists of VQA and image captioning tasks . The framework's performance on editing more complex, free-form, or unstructured knowledge (e.g., benchmarks like MMKE-Bench, VLKEB, or MIKE) is not assessed. Furthermore, its applicability to specialized domains (e.g., medical, legal, financial) is untested. The following work is worth the author's participation in experiments and discussions:

[1] Du Y, Jiang K, Gao Z, et al. Mmke-bench: A multimodal editing benchmark for diverse visual knowledge[J]. arXiv preprint arXiv:2502.19870, 2025.

[2] Xu D, Wang J, Chai Z, et al. MedMKEB: A Comprehensive Knowledge Editing Benchmark for Medical Multimodal Large Language Models[J]. arXiv preprint arXiv:2508.05083, 2025.

[3] Li J, Du M, Zhang C, et al. Mike: A new benchmark for fine-grained multimodal entity knowledge editing[J]. arXiv preprint arXiv:2402.14835, 2024.

[4] Zhang J, Zhang H, Yin X, et al. Mc-mke: A fine-grained multimodal knowledge editing benchmark emphasizing modality consistency[J]. arXiv preprint arXiv:2406.13219, 2024.

[5] Huang H, Zhong H, Yu T, et al. Vlkeb: A large vision-language model knowledge editing benchmark[J]. Advances in Neural Information Processing Systems, 2024, 37: 9257-9280.

- The framework introduces multiple new hyperparameters that appear sensitive and complex to tune. This includes the parameters of the TV penalty $\lambda$ (learned via an MLP) , the MMD kernel bandwidths $\sigma_q$, and the primal-dual learning rates $\gamma_1, \gamma_2$. This complexity may hinder practical adoption.
- The generality risk ($\mathcal{R}_{gen}$) calculation is critically dependent on "rephrase counterparts" generated by diffusion models. The method's overall effectiveness may be highly sensitive to the quality, diversity, and semantic fidelity of these generated samples, a dependency that is not systematically investigated in the paper.

### Questions
- What is the precise computational overhead (e.g., latency and memory) of ODEdit compared to baselines like MEND and UniKE, and how does this overhead scale with the complexity of the optimization (e.g., number of optimization steps)?
- How does the performance of ODEdit scale when applied to larger or more recent MLLM architectures like LLaVA or Qwen-VL, and do the "causal-underfit/overfit" issues  manifest differently in these models?
- How does ODEdit perform on benchmarks requiring more complex, free-form knowledge editing (e.g., MMKE-Bench) or on specialized domains (e.g., MedMKEB), which go beyond the VQA/captioning tasks in MMEdit?
- Given the framework's sensitivity to hyperparameters like the learning rates ($\gamma_1, \gamma_2$) and the TV penalty $\lambda$, how stable is the primal-dual optimization , and are there principled guidelines for setting these parameters across different models and datasets?
- How robust is the generality risk ($\mathcal{R}_{gen}$)  calculation to the quality and diversity of the "rephrase counterparts" generated by diffusion models? What happens if the generator model produces low-quality or semantically drifted samples?

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
3

### Summary
This paper frames the model editing problem in multimodal large language models as an out-of-distribution generalization problem. Through derivations, they finally have a loss function that is generalizable to the unseen domains. Extensive experiments also support their claims about the generalization of the editing method.

### Strengths
- In my understanding, the motivation of the research is to formulate the editing problem as an OOD generalization problem and adopt IRM to solve this problem. The main contribution comes from the derivation of the final optimization problem (Equation 13).
- The paper is well structured, and the writing is generally good.

### Weaknesses
1. Missing or Uncited Related Work
The paper overlooks several recent studies relevant to dynamic balance and continual editing in multimodal or LLM-based model editing.

E.g. [1],
[1] Guo D, Hu M, Guan Z, Hartvigsen T, Li S. BalancEdit: Dynamically Balancing the Generality-Locality Trade-off in Multi-modal Model Editing. arXiv preprint arXiv:2505.01343. 2025.

2. Lack of Causal Grounding

The authors discuss output variation within multimodal LLMs as being shaped by “cascaded reasoning that integrates unimodal perception, inter-modal alignment, and shared semantic space modeling.” However, this claim lacks causal modeling support.

3. Unclear Definition of “Semantic Shift” and “Factual Shift”

The terms semantic shift and factual shift appear to be used descriptively rather than formally defined.

4. Dataset Access Assumptions

The paper assumes access to three datasets — $D_{in}$, $D_{se}$, and $D_{out}$ — without discussion of their realism or comparability with prior work.

Traditional model editing frameworks (e.g., UniKE) typically rely on paired input–output data or factual triples but not necessarily three distinct datasets.

The need for separate in-domain, side-editing, and out-of-domain datasets may limit applicability in real-world continual learning or model update scenarios.

5. Missing Discussion on Long-Term Editing Behavior

The paper does not address life-long editing (how edits accumulate and interact over time) or editing consumption (the efficiency of the editing method).

Relevant precedents include:
Hartvigsen et al., A Unified View of Model Editing: Towards Causality and Generalization, ICLR 2024.
Mitchell et al., Memory-Based Model Editing at Scale, ICLR 2022.

### Questions
See above.

### Soundness
2

### Presentation
2

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
This paper proposes ODEdit, a plug-and-play invariant learning framework for knowledge editing in MLLMs. Building on the perspective that MLLM knowledge editing is an out-of-distribution (OOD) generalization problem, ODEdit introduces a tripartite OOD risk objective to concurrently address reliability, locality, and generality. It further develops an edit trajectory invariant learning method incorporating a total variation penalty, grounded in theoretical analysis and optimized via a primal-dual approach. Extensive experiments on standard MLLM benchmarks, along with ablation studies and visualizations, aim to demonstrate the robustness and effectiveness of ODEdit compared to prior methods.

### Strengths
1. The formalization of the editing challenge as an OOD generalization problem is interesting and provides a different perspective.
2. The paper tackles the pressing issue of robust and adaptive knowledge editing in MLLMs, a setting of increasing importance as these models proliferate in real-world.
3. The presentation is good and can demonstrate the author's insights.

### Weaknesses
1. While ODEdit often shows incremental gains, for several metrics and baselines, improvements are relatively modest and in some cases, the gains in one dimension come with trade-offs in another.
2. The ablation study in Table 2 shows that the effect of $R_{gen}$ is limited and has led to a decline in the vast majority of metrics. I wonder why such a phenomenon occurs.
3. I think adding more MLLMs (e.g. Qwen-VL) and datasets is better for proving the effectiveness of ODEdit.

### Questions
See Weaknesses

### Soundness
2

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
3

### Summary
The authors address the problem of knowledge editing for multimodal large language models (MLLMs). They reformulate editing as an out-of-distribution (OOD) generalization problem: the editing mechanism must succeed across a variety of cross-modal prompting “environments”, not just the original prompt. To address this, they propose ODEdit, a plug-and-play invariant learning framework that learns invariant edit trajectories so that edits generalize across prompt/background modalities.

### Strengths
1. Casting editing as an OOD generalization problem is interesting.
2. The authors provide a theoretical analysis linking their invariant constraint.

### Weaknesses
1. ODEdit requires generating multiple versions of the same edit query to construct diverse environments, whereas several baselines operate on a single prompt without such augmentation. This difference introduces additional supervision and may render the comparison with baselines less fair.
2. Although the paper acknowledges the potential tension between locality and generality, it does not provide a systematic analysis of this trade-off. Moreover, the reported improvements in generality are relatively marginal, which weakens the empirical support for the claimed advantage.
3. The visualization results for out-of-distribution generalization show only subtle differences, making it difficult to clearly perceive the claimed improvement. A more quantitative or visually distinct analysis would better support the argument.

### Questions
Please refer to the Weakness.

### Soundness
3

### Presentation
2

### Contribution
2
