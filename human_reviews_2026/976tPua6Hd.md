# Baleen: Self‑Interpretable, Robust SSMs with Stochastic Selective Memory

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
We introduce \textit{\textbf{Baleen}}, a family of state space models that unifies \textbf{stochastic selection} with \textbf{information bottleneck} to build interpretable and robust long‑context learners. Unlike Mamba/Mamba2’s deterministic gates, Baleen treats selection as a random variable and regularizes it with a closed‑form KL to a sparsity prior: (i) \textit{\textbf{\bib}} samples Bernoulli state‑transition gates; (ii) \textit{\textbf{\gib}} samples Exponential  time‑intervals. This yields an explicit trade‑off between retention and compression and exposes token‑level selection heatmaps at inference for self‑interpretation. On language benchmarks, \ib  improves average accuracy over Mamba2 by +0.95 at 370M pretraining and +1.38 at 7B finetuning. Baleen delivers stronger robustness to localized perturbations and adversarial attacks: under CIFAR‑10 sequence perturbation, prefix damage falls to 0.6\% vs 26.5\% for Mamba2 (average under attacks 0.542 vs 0.385). Finally, Baleen’s self‑interpretations outperform IG/Grad‑CAM on average fidelity across four text classification tasks. We will release our Baleen‑7B models on Hugging Face with code, checkpoints, and an interactive selection‑heatmap demo.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper introduces an improvement to the state space model, introducing Beleen families. This is attributed to a careful analysis of Mamba from the aspect of the information bottleneck. Baleen studies the selection mechanism from Mamba and has two new variants of the selection strategy. Experiments show the performance enhancement brought about by the new selection methods. Other aspects, including interpretation and performance robustness, are also studied.

### Strengths
- Clear motivation: This paper clearly revisits the information-theoretic pitfalls of Mamba and proposes the improved Baleen architecture.
- Plug-and-play designs: The proposed designs can be incorporated into existing Mamba models.
- Comprehensive experiments: Various aspects of the Balle model are investigated, including language benchmarks, inherent interpretation, and robustness.

### Weaknesses
- Inexact terms: Section 4.3 discusses the model’s performance robustness to perturbations. This does not constitute an adversarial scenario, as there is no clear attacker or malicious intent.
- Missing baselines: For the language benchmark, what about the performance of transformer-based models of equal size?

### Questions
Will the authors release their codes to reproduce the experimental results?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper implements a variant of state-space models inspired by the information bottleneck principle. Thy argue that this information bottleneck principle can result in an inherently robust and interpretable architecture, whereas standard MAMBA does not have any incentives to select only useful parts of the context and can "overfit" in various ways with longer contexts. They propose a variant of the infromation bottleneck for the sequence modelling problem and provide a variational upper-bound which can be realized via stochastic transition dynamics in Mamba. They provide two formulations:  Baleen-B samples transitions from a bernoulli distributions and Bernoulli E samples the time-interval parameter randomly. They show that Baleen improves language modelling performance relative to MAMBA, can yield inherent interpretability, and appears to induce superior robustness to adversarial attacks/perturbations.

### Strengths
I think that this paper attacks an important problem facing state-space models: their seeming difficulty in rejecting irrelevant historical information in Mamba models. This has, for example, been documented in [1] where they attribute that a inability to learn to forget information contributes to performance degradations when Mamba models are tested beyond their training context size. Additionally, I find that the connection to information bottleneck principle is interesting and they clearly explain the derivations along the way to their variational upper bound and training algorithm. I believe that it is interesting that they derive two possible instantiations of their approach based off Bernoulli sampling of the transition matrix and . They also perform evaluations on a range of model properties, including interpretability, robustness, and general performance -- which I find are interesting.
[1] Chen, Yingfa, et al. "Stuffed Mamba: Oversized States Lead to the Inability to Forget." arXiv preprint arXiv:2410.07145 (2024).

### Weaknesses
**Motivations for the Method Could Be Presented Better**: The authors introduce their problem, in my opinion, far too abstractly. They comment on there being "no incentive" for the MAMBA model to compress irrelevant contextual information out. They provide some vague intuitions that this may be connected to worse performance/adversarial vulnerability etc. However, these claims are largely intuitive and do not seem entirely convincing.  I believe that the authors could better motivate their methods by discussing more direct evidence that MAMBA models detrimentally fail to remove knowledge. For example, connecting with the observations in [1] and/or providing more detailed results tying together the inability to forget with adversarial vulnerability/overfitting could be very beneficial for this paper.

**Better Clarity in the Implementations** Although the authors do a good job of presenting the mathematical derivations behind their method, the implementation details could be more clear. In fact, I would argue that too much time is spent on the mathematical derivation, relative to implementation details. After reading the paper, for example, I was a bit confused on several implementation details. For example, it was not clear to me how is inference performed in Baleen? The authors mention that the expectation of the stochastic matrices is used for interpretation, but it remained unclear if this is for the general forward pass or just the interpretations. Similarly, how are the hyperparameters for the prior distributions chosen? This seems to be important for the reproducability of the work but is highly unclear from the paper.

**Show Results Across Architecture and Sequence Length** The authors performed some experiments at a 20B token scale and superior performance along some baselines. However, I would be interested to know more about whether these benefits only arrive in a small-scale training setting. For example, in [1] they note that the relationship between the state parameters and the sequence length control how much forgetting is learned. Thus, it would be interesting to learn more about whether the benefits of BALEEN are amplified or mitigated depending on the relationship between Architecture v.s. Sequence Length. It would be good to haver more clarity about in what regimes the information bottleneck is particularly helpful, as this would improve our understanding of the method and its benefits.


**Provide Qualitative Examples of the Difference Between Standard Mamba and BALEEN Interpretations**  Currently, the paper provides some examples. of the BALEEN interpretability an argues that they are superior to Mamba. There are some quantitative results on this as well. However, I feel that the it would help my understanding of the paper if there were qualitative examples of where a comparable setup with Mamba fails.


[1] Chen, Yingfa, et al. "Stuffed Mamba: Oversized States Lead to the Inability to Forget." arXiv preprint arXiv:2410.07145 (2024).

### Questions
(1) Could the authors please clarify whether their problem statement is related to that shown in the Stuffed Mamba paper? 
(2) Could the authors clarify the hyper-parameter tuning details for the prior parameters?
(3) Could the authors mention how is stochasticity of the parameters utilized at inference time.
(4) Could the authors mention how they expect the benefits of BALEEN to scale at different sequence lengths/state sizes?

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
3

### Summary
This paper proposes Baleen, a new state space model that combines information bottleneck principles with stochastic variables to improve interpretability and robustness in SSMs. The authors propose two variants: Baleen-B, which models the state transition matrix $A_t$ as a Bernoulli RV, and Baleen-E, which models the discretization parameter $\Delta_t$ as an exponential RV. Both variants include analytical KL divergence terms (or bounds) as regularizers. 
Experiments demonstrate improvements over Mamba-2 on language benchmarks using 370M trained from scratch on 20B tokens, and 7B finetuned on 0.5B tokens. Baleen models also show better adversarial robustness, and self-interpretation capabilities derived from the transition matrix $A_t$.

### Strengths
Things that I liked in this paper include:

- Casting SSMs as an information-bottlenecked sequence model (Eq. 5) is a clean way to show that MLE-trained Mamba can suffer with spurious context and is sensitive to perturbations. I think the paper connects Mamba to the special case (\beta = 0) and builds from there. This provides a principled framework for understanding the trade-off between retention and compression in linear recurrent models.

- The Exponential-gated timestep (indirectly on both (A_t) and (B_t)) is well-motivated and look less ad-hoc.

- I think the adversarial robustness setting is an interesting angle to SSMs, and Baleen seems to handle that well. For example, the position bias attack experiments (Table 3) show convincing improvements, with Baleen-E maintaining a higher accuracy than Mamba-2. The prompt attack experiments also demonstrate consistent robustness improvements.

- The method is easy to implement and cause re-use existing efficient kernels. This makes Baleen a easy drop-in improvement for Mamba and also other linear recurrent models.

### Weaknesses
- I think that the claim that Mamba "just does MLE" is an overstatement. That is, the argument that Mamba = SSIB with $\beta = 0$ is  true only for an idealized training loss, without regularization. In practice, large-scale Mamba training uses regularization (dropout, weight decay, data noise, possibly distillation) that already limits I(H;X).

- For me, the Bernoulli parameterization of $A_t$ is not fully convincing. The paper says "it is natural to model $A_t$ as Bernoulli" because each entry decides keep/drop. But in Mamba-2 the reason $A_t \in (0,1)^{N \times D}$ is the specific discretization, which leads to entries being viewed as *decay factors*, not probabilities. Turning *every* entry into an i.i.d. Bernoulli with parameter $p_t$ is very a strong modeling assumption (e.g., it allows exactly zeros which implies full forgetting about the context). 

- For Bernoulli the paper says that Gumbel–Softmax (Binary Concrete) is used, which is fine. For the Exponential variant there is no information about the differentiability of the estimator.

- The "importance" score simply averages $A_t$ over both the state dimension ($N$) and the feature dimension ($D$). That is exactly the kind of averaging MambaAttention/MambaLRP later criticized as washing out channel-specific effects. 

- For interpretability, the evaluation would be stronger if the paper includes other SSM-specific methods, such as MambaAttention [1], MambaLRP [2], and LATIM [3]. The current baselines are somewhat weak and outdated. 

- On 370M / 20B-token pretraining, the improvements over Mamba-2 are small and vary across tasks. On datasets like MMLU and WinoGrande at that scale the noise is high, so it is hard to attribute the gains to the IB formulation. The 7B finetuning comparison is also slightly unfair because Baleen is finetuned on the extra 0.5B Crystal data but the baseline Mamba-2 is not.

[1] https://arxiv.org/abs/2403.01590 
[2] https://arxiv.org/abs/2406.07592
[3] https://aclanthology.org/2025.acl-long.1194/

### Questions
- What is the Crystal dataset used for 7B instruction tuning? I could not find further information about it.

- For the Exponential case, what is the exact gradient estimator used?

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
This paper introduces Baleen, a new family of SSMs that integrates stochastic selective memory with an information bottleneck objective to improve interpretability and robustness. Extending Mamba, Baleen models state-selection gates as random variables with a variational KL regularizer, balancing context retention and compression while preserving linear-time recurrence. Two variants based on Bernoulli gating and exponential time sampling, Baleen-B and Baleen-E, yield self-interpretable token-level importance scores derived from expected gate values. The formulation provides an information-theoretic refinement of Mamba, explicitly controlling how much input history is retained, and promoting minimal sufficient representations that enhance robustness and explanation fidelity.

### Strengths
1. The paper presents a principled integration of information bottleneck theory into SSMs, yielding stochastic gate regularization that improves both interpretability and robustness.

2. The proposed Bernoulli and exponential gating variants are theoretically well-motivated and retain Mamba’s linear computational complexity.

3. The method demonstrates strong robustness under input perturbation and adversarial settings.

### Weaknesses
1. While formulating a variational information bottleneck on Mamba’s gating is interesting, the conceptual novelty is incremental. The method essentially adds a KL-based sparsity regularizer to encourage selective gating, similar to variational dropout or selective masking. The core SSM architecture remains unchanged, and the stochastic gates are parameterized similarly to Mamba’s deterministic ones. Thus, Baleen may be viewed as “Mamba2 + KL regularizer,” raising the question of whether similar gains could be achieved with simpler methods (e.g., L0​/L1​ penalties or tuned dropout). Nonetheless, the formalization through the IB framework provides a theoretically grounded lens on sparsity and interpretability, which may justify the contribution despite its incremental nature.

2. Baleen inherits the fixed hidden-state constraint of SSMs, which limits the amount of information it can store and retrieve. This becomes problematic for tasks requiring the distributed recall of multiple facts across long contexts. The IB objective improves information quality but may over-compress, discarding useful signals needed later. Since the number of independent bits is bounded by the state dimension, Baleen cannot fully overcome memory saturation when the sequence length greatly exceeds the state size. Without adaptive or external memory, it may still struggle with tasks requiring long-term or multi-fact reasoning.

3. Although Baleen consistently outperforms prior SSMs, its empirical gains are modest. Improvements vary across tasks, with some regressions, and certain baselines (e.g., Gated-DeltaNet) outperform it on individual benchmarks. Moreover, comparisons are limited to SSM and linear RNN families; Transformer baselines, or hybrid models of comparable scale, are missing. Without such results, it remains unclear whether Baleen approaches or still lags Transformer-level performance.

### Questions
1. How sensitive are Baleen’s results to the choice of the IB regularization strength (the weight β in KL term in Eq. 4)?

2. Can Baleen effectively handle tasks requiring many pieces of information to be retained over long sequences?

3. Since the gates are stochastic, have the authors examined whether the expected gate values are stable across different runs, and whether the top-selected tokens align with known important words or ground-truth rationales?

### Soundness
3

### Presentation
3

### Contribution
2
