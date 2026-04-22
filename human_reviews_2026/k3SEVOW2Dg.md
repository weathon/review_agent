# Latent Concept Disentanglement in Transformer-based Language Models

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
When large language models (LLMs) use in-context learning (ICL) to solve a new task, they must infer latent concepts from demonstration examples. This raises the question of whether and how transformers represent latent structures as part of their computation. Our work experiments with several controlled tasks, studying this question using mechanistic interpretability. First, we show that in transitive reasoning tasks with a latent, discrete concept, the model successfully identifies the latent concept and does step-by-step concept composition. This builds upon prior work that analyzes single-step reasoning. Then, we consider tasks parameterized by a latent numerical concept. We discover low-dimensional subspaces in the model's representation space, where the geometry cleanly reflects the underlying parameterization. Overall, we show that small and large models can indeed disentangle and utilize latent concepts that they learn in-context from a handful of abbreviated demonstrations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates how LLMs (specifically, Gemma-2-27B) represent and manipulate latent concepts when performing ICL. The authors combine mechanistic interpretability methods with controlled synthetic tasks to explore whether models infer and compose hidden concepts (e.g., countries, radii) in structured ways. The authors explore two major experimental setups: two-hop reasoning tasks and geometric tasks, and conclude that transformers naturally form disentangled, low-dimensional concept representations that are causally and geometrically interpretable.

### Strengths
* The division between discrete and continuous latent concepts is conceptually clean and helps generalize across types of reasoning.

* Identifying sparse, interpretable sub-circuits for latent concept composition is relevant to understanding compositional generalization.

* The authors situate their findings well within recent discussions of task vectors, linear representation hypothesis, and in-context reasoning circuits.

* Demonstrating low-dimensional manifolds for latent numerical parameters (and the ability to steer them) is a compelling visualization of representational structure.

### Weaknesses
* While synthetic setups isolate mechanisms effectively, they may oversimplify real-world linguistic reasoning. It’s unclear whether these disentanglement findings generalize to naturalistic contexts.

* The discrete reasoning experiments rely on Gemma-2 models, while continuous ones use small custom transformers. The gap in architecture and scale complicates unification. Moreover, would the discrete reasoning results replicate with other models?

* Activation patching is compelling, but causal claims might be overstated without broader ablation or counterfactual consistency checks.

* While qualitative visualizations (e.g., PCA plots) are persuasive, more statistical rigor (variance, replicability, robustness to seeds) would add credibility.

* The discussion of how these insights could enhance controllability, generalization, or interpretability in applied settings remains mostly speculative.

### Questions
* Do you observe similar disentanglement patterns in larger, more naturalistic multi-hop datasets (e.g., HotpotQA, factual chains beyond two hops)?

* Are the same attention heads consistently responsible for “bridge” resolution across tasks or prompts, or does specialization vary by context?

* How might identifying latent concept circuits inform real-world model editing, debiasing, or steering applications?

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
3

### Summary
This paper studies the problem of whether and how transformers represent latent concepts as part of their computation in a an ICL setup.

Key ideas

* Use activation patching (a.k.a Causal Mediation Analysis) to determine if the model is indeed representing the latent concept in a discrete multi-hop setting.
* For the numerical latent variables, the authors propose the use of linear probing the intermediate model embeddings to assert the existence of task vectors.
* Use PCA and project the task vectors onto the first two principle components and then study the geometry of the resultant manifolds along with the ordering of the points. Using steering (or interpolation) based methods to show that model captures concept’s geometry.

Main contributions

* Use of causal and correlation techniques to show that for tasks requiring memorized world knowledge, a sparse set of attention heads are responsible for resolving intermediate latent concept, thereby rejecting the hypothesis that the model takes a shortcut (directly maps the input to the corresponding output without any intermediate steps). This shows that the model is indeed performing abstract reasoning (chained reasoning) in its hidden activations.
* Show how the above phenomenon plays out in case of smaller language models are used (2B, instead of 27B) and larger number of ICL examples.
* In alignment with the previous work on Linear Representation Hypothesis (LRH), the authors establish the existence of task vectors. In addition to that, the authors show that these representations reflect the geometry of the latent variable.

### Strengths
Originality

* The unorthodox evaluation methodology of using activation patching with counterfeit examples is creative.
*  The technique of using interpolation (referred in paper as steering) in this setting is innovative and establishes the geometry of the underlying latent variable.
* The paper discusses the findings of previous studies in the literature survey section and clearly state the novel contributions of this work.

Quality

* Overall, I find the paper to be of good quality, the methodology is sound and the results are well presented. The experiments are conducted on a variety of datasets which demonstrates the robustness of the claims.

Clarity

* The paper is well written and easy to read. The ideas are explained clearly with supporting experiments. 

Significance

* This works brings us closer to understanding how LLMs represent intermediate concepts, perform chained reasoning in ICL setup.
* It uses tasks such as add-k, circular trajectory, rectangular trajectory to empirically prove that for numerical latent variable problems, tasks vectors indeed represent the geometry of latent variables.
* The paper strengthens the broader hypothesis that LLMs engage in multi-step reasoning rather than relying solely on direct input-output associations.
* The authors also highlight the influence of model size and the number of ICL examples on the latent structures.

### Weaknesses
* In the “memorized world knowledge” task, the authors claim that the LLM relies on step-by-step composition of latent concept. However, all their experiments are of the kind where the latent concept has a one-to-one mapping with the output class (for instance country → capital). While, there is no problem with this choice, it might be helpful to study different scenarios (such as country (bridge) → famous politicians from that country)
* In Figure 10, for the circular trajectory problem, even with high value of beta (0.8), we find that MSE corresponding to original and opposite radius is nearly equal. Isn’t this counter-intuitive?

### Questions
* In the “Memorized world knowledge” thread - While the existing work uses 2-hop reasoning tasks, it would be interesting to study higher order transitive tasks (3 hops or more). It would be interesting to tease out the impact of model size  on such tasks. One potentially interesting finding could be that LLMs are able to deal with k-hops (k>>2) reasoning tasks much easily as compared to small language models (SLMs). It would be interesting to see, if LLMs indeed contain sparse attention heads which represent all the intermediate (bridge) concepts.
* The authors have used two variants of Gemma 2 model family - 27B and 2B. It would be helpful to conduct experiments using models of varying capacity - including a few with multilingual support, so that claims/findings can be strengthened further.
* In section 3, add-k problem, I did not understand the reason behind the constraint k_{i+1} - k_i = 3. What happens when we relax the constraint? Will the claims be still valid, if this constraint is not met?
* The authors mention that ideal scaling constant for bridge intervention is 2.0. My understanding is scaling is a way to boost the importance of the signal (bridge information). Is that correct? If that is so, it is understandable that the performance saturates at 4.0. But, what could be the reason for deterioration?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper systematically studies how large language models represent, disentangle, and utilize latent concepts in in-context learning. The authors use a series of controlled tasks combined with mechanistic interpretability methods to provide both causal and correlational evidence, revealing the internal reasoning and representation mechanisms of Transformer models when solving ICL tasks.

### Strengths
The exploration of how LLMs understand and reason at the conceptual level is interesting and provides valuable insights for the community. The use of causal mediation analysis and PCA visualization to validate the claims is persuasive under the synthetic tasks presented in the paper. The experimental design is rigorous and highly interpretable. In particular, the demonstration of low-dimensional manifold structures in the mode’s representation space (i.e., the geometric interpretability of latent variables) is a novel perspective.

### Weaknesses
The tasks and experimental setups are overly idealized, relying almost entirely on highly synthetic toy tasks, which do not represent real-world natural language reasoning tasks. Of course, this is a common issue in the interpretability field.

The analysis of model scale and generalization is insufficient. The paper only compares Gemma-2-27B and 2B, without systematically examining whether the same mechanisms hold across different architectures (e.g., LLaMA, Qwen series) or larger-scale models. It is also unclear why the experiments in Section 3 are conducted on a small GPT-2–style model rather than the previously used Gemma models, and what the motivation for this model substitution is.

The paper provides only empirical analysis. It would be more compelling if any theoretical or mechanistic explanation were offered for why Transformers naturally form linear geometric task vectors or bridge concepts.

Are the bridge concepts unique? The experimental design in the paper is overly simple, but in real-world scenarios, the relationships between concepts are much more complex. With only a few in-context examples, there may exist multiple potential bridge concepts between the source and target, a possibility that the paper does not discuss.

Another concern is how such concept-level empirical interpretability analyses can provide meaningful value for real-world applications. In other words, when would we actually need concept-level explanations? Even if we know which attention heads encode certain concept representations, how should we intervene or utilize them in practice? Could specific application scenarios be provided as examples?

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper analyzes how LLMs represent and combine latent "concepts" during ICL. To analyze these interactions, they employ causal mediation analysis to identify which components of the model’s computation mediate the effect of particular input concepts on final predictions.

### Strengths
- The paper takes a systematic and well-executed approach to analyzing internal mechanisms of ICL.
- The approach offers interpretable, fine-grained insight into how contextual information propagates, complementing existing representation-based probing techniques.
- The study spans synthetic geometric reasoning tasks and natural language problems, demonstrating that the framework generalizes across distinct domains of conceptual structure.  
- The visual analyses (mediation heatmaps and intervention results) are well presented and help illustrate which attention paths encode specific concepts.  
- The work contributes methodologically by adapting tools from causal inference to the study of neural mechanisms in LLMs.

### Weaknesses
- The paper treats latent concepts as interpretable dimensions or attention patterns but never defines them formally.  
- The causal model in the introduction ($F = R \circ C$) is not linked to the implemented CMA pipeline. $R$ and $C$ are just introduced without definitions. There is no derivation showing that the empirical mediation quantities estimated from activations correspond to components of this decomposition.  
- Experiments are mostly conducted on synthetic or simplified reasoning datasets (two-hop relations at most).  
  These provide clarity but limit conclusions about behavior on realistic compositional or linguistic tasks.  
- The paper occasionally presents correlations in mediation strength as evidence of conceptual disentanglement. In my opinion, stronger validation, like causal interventions or counterfactual ablation, would be needed to corroborate these claims.  
- No stability analysis or statistical testing is reported. It is unclear how consistent the discovered mediators are across runs or models.

### Questions
1. How are concept variables defined or selected for mediation?
2. How stable are the identified mediating paths across random seeds or different model checkpoints?  
3. Intuitively, can the mediators identified via CMA be related to known attention-head clusters or activation subspaces discovered in prior ICL localization work?

### Soundness
3

### Presentation
3

### Contribution
3
