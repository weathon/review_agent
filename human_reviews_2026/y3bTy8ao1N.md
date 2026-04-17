# A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Steering Vectors in Large Language Models

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Large language models (LLMs) possess impressive generative capabilities but remain opaque and can exhibit unsafe or undesired behaviors. Existing control methods rely on supervised fine-tuning or curated prompt-response datasets, which limits their scalability. We propose SteerCLR, an unsupervised method that simultaneously discovers a bank of diverse and disentangled steering vectors directly from unlabeled prompts. By optimizing a novel contrastive objective over internal model activations, SteerCLR learns vectors that correspond to distinct behavioral shifts. Injecting these vectors into a frozen LLM enables fine-grained, low-latency control over generation, including suppressing toxicity, modulating sentiment, and uncovering subtle stylistic dimensions, without relying on labeled data, classifiers, or attribute-specific supervision. We demonstrate that optimizing for activation magnitude and activation diversity yields a rich set of interpretable directions. Experiments on instruction-tuned Llama-2-13B-chat model show SteerCLR discovers diverse interpretable steering vectors in a single training run, significantly advancing the scalability of mechanistic interpretability and enabling practical interventions for safety, alignment, and model auditing.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SteerCLR, an unsupervised contrastive learning framework that discovers a bank of diverse, interpretable steering vectors in an LLM. By maximizing activation impact and enforcing cross-vector diversity, SteerCLR learns disentangled behavioral directions (e.g., sentiment, style, refusal) without labeled data. Experiments by injecting these steering vectors to a frozen LLM show that the discovered vectors are diverse and could enable fine-grained and composable control over model behavior.

### Strengths
1. **Conceptually simple and efficient.**
- The methodology is fully unsupervised and scalable to discover many behavioral vectors in one run. 

2. **Novel and well-motivated formulation.**
- The contrastive objective elaborated in Section 3.2 that combines and balances magnitude and diversity is elegant and effective.

3. **Well-supported ablations and analysis.**
- The authors have done comprehensive ablations on layer sweep and hyperparameter tuning (Appendix B) to justify their reported results.
- They also conduct both quantitative (Section 4.4) and qualitative (Section 4.5) analysis to show that the discovered steering vectors are meaningful and generalizable across prompts.

### Weaknesses
1. **Justification of the separation of subspaces.**
- Although the authors have analyzed the differences when applying different directions on positive, neutral, and negative (quantitative), as well as the generated output responses (qualitative), I believe more quantification of the specifics of the separability of steering vectors are needed. 
- For example, 1) taking the discovered steering vectors and using linear probes to test their separability, or 2) providing descriptive statistics (similarity, distance, etc) on the vectors in the subspace.

2. **Ablate on the two components of the contrastive objective.**
- The authors propose that both the magnitude and diversity matter in the optimizing process. It would be interesting to ablate on the two components to provide a fine-grained look into what is the effect of each on the outcome.

3. **Compare with baselines beyond CAA**, such as SEA [1] and BiPO [2].

---
**References**

[1] Qiu, Yifu, et al. "Spectral editing of activations for large language model alignment." Advances in Neural Information Processing Systems 37 (2024): 56958-56987.

[2] Cao, Yuanpu, et al. "Personalized steering of large language models: Versatile steering vectors through bi-directional preference optimization." Advances in Neural Information Processing Systems 37 (2024): 49519-49551.

### Questions
An advantage of supervised, labeled prompts to extract steering vectors is its targeted control over specific attributes. Although SteerCLR shows that certain directions can produce particular behaviors, the control is descriptive rather than intentional, which means that one can amplify an effect but not specify what effect to pursue. 

I hope the authors could discuss how this framework could either:
1) already allow more practically controllable and useful for alignment applications, or 
2) additionally incorporate mechanisms for goal-conditioned or guided discovery (e.g., light supervision or constraint-based objectives).

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes an unsupervised LLM activation steering method where several steering vectors are discovered from unlabeled prompts by optimising a contrastive objective and injecting them at an early layer to maximise activation change at a deeper target layer. The results show the method is comparable and improves over the Contrastive Activation Addition baseline across multiple behavioral evaluations. While most of the experiments are run on Llama-2-13B-chat, supporting qualitative results are shown on Qwen2.5-7B-Instruct.

### Strengths
- the paper builds on previous steering vectors and activation addition literature.
- benchmarks results from "Steering Llama 2 via Contrastive Activation Addition" are reused which is good for reproducibility. But the model used is very outdated.

### Weaknesses
- All experiments were conducted on a very outdated model (Llama-2-13B-chat) which puts into question its applicability.
- How the fluency or relevance of the responses may be affected by the proposed method is not studied and only addressed qualitatively.
- There is no study of the computational cost of using this method or overhead it applies.
- It is only compared to one baseline, CAA. The authors could use as an additional comparator e.g. https://arxiv.org/pdf/2310.01405 for instance.
- A wide set of interpretable behaviors (sentiment, toxicity, verbosity, reasoning style, jailbreak, gamification) is mentioned but only assessed qualitatively. This analysis is not strong enough to make claims such as "exceeds single-vector baselines in coverage". No sentiment or toxicity focused benchmarks are used.

### Questions
Can you rerun your experiment and your single baseline with a newer model of the same size?
Can you perform a quantitative study for some of the intepretable behaviours you are mention in your paper? (sentiment, toxicity, verbosity, reasoning style, jailbreak, gamification)
Can you compare your method to other inference time steering methods that do not use activation addition?

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
3

### Summary
This paper proposes SteerCLR, an unsupervised method for discovering steering vectors within a frozen LLM. The authors show that the proposed approach yields broader and more distinct behavioral directions than previous work (CAA) in quantitative evaluations. They also demonstrate stylistic variations and interpretable behavioral shifts in qualitative analyses.

### Strengths
1. Unlike prior methods, SteerCLR performs single-pass optimization and is conceptually straightforward, which makes it efficient from an implementation perspective.
2. Despite being unsupervised, it achieves overall stronger quantitative performance than supervised or paired-data baselines.

### Weaknesses
1. The selection of source (lower) and target (deeper) layer indices requires stronger justification. Because different layers in an LLM encode distinct types of knowledge, the rationale for choosing a specific layer pair should be grounded in prior evidence rather than limited ablation studies. Otherwise, each new model would necessitate redundant ablations, undermining the claimed efficiency of the single-pass design.
2. The interpretation of discovered steering vectors through LLM-based labeling appears unreliable. The number of sampled prompt pairs (only ten) is too small, and there is no guarantee that the evaluating LLM accurately understands or consistently identifies the semantics of each vector.

### Questions
Please respond to the weaknesses mentioned above.

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
5

### Summary
This paper introduces a new technique for learning a set of steering vectors given activations over a dataset in an unsupervised manner, using the combination of the circle loss (a contrastive loss from prior literature) which encourages diversity and an impact loss which maximises steering effect. The vectors found seem to be better than a supervised baseline (CAA) on a small set of tasks, and analysing the vectors reveals interesting and clear signals about various aspects of language.

### Strengths
- The method proposed in novel and amenable to scaling up, and I'd be excited to see the quality of the steering vectors it finds at scale / the scaling trends in general. I actually think an interesting framing of this approach would be unsupervised dictionary learning, as a steering-focused alternative to SAEs. Compare [Arad et al. (2025)](https://arxiv.org/abs/2505.20063).

### Weaknesses
- The evaluation dataset is too small and the baselines compared against are too weak to draw strong conclusions about the efficacy of the method. See more recent work like [Wu et al. (2025)](https://arxiv.org/abs/2501.17148) (a large-scale benchmark for steering methods), [Arad et al. (2025)](https://arxiv.org/abs/2505.20063) (a strong baseline for steering), etc.
- I'd be interested in LM-judge-produced labels for all the steering vectors learned by the method rather than a qualitative assessment of a few features. In general, this paper ought to consider more of the ramifications for scaling this technique.

### Questions
- How hard was it to tune the $\alpha$, $\beta$, etc. hyperparameters? Some appendices with experiments would be nice if possible.

### Soundness
2

### Presentation
3

### Contribution
2
