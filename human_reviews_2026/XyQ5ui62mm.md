# Belief Dynamics Unify In-Context Learning and Activation Steering

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Large language models (LLMs) can be controlled through prompts (in-context learning) and internal activations (activation steering), but a unified theory explaining these methods is lacking, making their application often reliant on trial-and-error. Here, we develop a unifying predictive account of LLM control from a Bayesian perspective, proposing that both context- and activation-based interventions impact behavior by shifting the \textit{model's belief in latent concepts}. Under our framework, steering operates by shifting concept priors, and in-context learning leads to an accumulation of evidence. This theory predicts three key phenomena we verify empirically: (i) sigmoidal learning curves as in-context evidence accumulates, (ii) predictable shifts of these curves with activation steering, and (iii) additive effects of both interventions, creating distinct behavioral phases. Our framework yields a closed-form model that is highly predictive of LLM behavior across context- and activation-based interventions in a set of five domains inspired by prior work on many-shot in-context learning. Crucially, this model also predicts the precise crossover boundaries where these interventions trigger sudden behavioral shifts. Taken together, our framework offers a unified account of prompt-based and activation-based control of LLM behavior, and a methodology for empirically predicting the effects of these interventions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a unified Bayesian framework for understanding how in-context learning (ICL) and activation steering control large language model behavior. The authors argue that both methods operate by updating an LLM's beliefs over latent concepts: ICL accumulates evidence through likelihood functions while activation steering modifies concept priors. They derive a predictive model that captures sigmoidal ICL curves, steering-induced curve shifts, and additive interaction effects. Experiments across three LLMs on five persona-manipulation tasks show the model achieves high predictive accuracy.

### Strengths
The theoretical unification of ICL and activation steering through Bayesian belief updating is conceptually elegant.

### Weaknesses
Circular dependency in steering vector construction: 

The paper claims that activation steering modifies priors p(c) independently of context (Section 3.2). However, steering vectors are constructed using the CAA method (Eq. 2), which requires processing demonstrations Dc and Dc' through the model—the same demonstrations that could be used directly for ICL.
This creates a conceptual problem: if steering vectors are derived from demonstrations, then the "context-independent prior modification" actually depends on contextual information. The distinction between "steering modifies priors" versus "ICL accumulates evidence" may simply be two different ways of using the same information from demonstrations, rather than genuinely independent mechanisms.
The paper should clarify whether steering can be constructed without demonstrations, or if the theoretical framework is primarily describing how demonstration-derived information can be "pre-compiled" into steering vectors for later application.

### Questions
Statistical testing only contains correlations, lacking confidence intervals, significance tests, or comparisons to simpler baseline models.

### Soundness
3

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
4

### Summary
This paper studies the interaction between modulating an LLM's behavior through in-context learning and activation steering on multiple-choice-based persona evals. Importantly, the authors develop a Bayesian theory of the effects of these two interventions, with the prediction being that steering modulates the prior odds of a particular concept, and ICL leads to the accumulation of evidence for a particular concept. The paper finds that the resulting model is highly predictive of the behavior we observe in practice.

### Strengths
The paper is conceptually elegant, clearly written, and well-executed. Specifically, the central idea that steering modulates priors and ICL accumulates evidence is quite natural (perhaps in hindsight!), and the accompanying theory appears to be sound given the assumptions it makes. The quality of the writing and presentation is high – I found the diagrams to clearly communicate the structure and goals of the method, and the numerical figures support the claims being made. While the current execution has some limitations, I think this work has the potential to have a strong impact, both on the ICLR community and at large, as understanding the relationships between the different test-time interventions we can make on LLMs is arguably one of the most important topics in LLM research today.

### Weaknesses
- I think the paper would be strengthened by showing that the predictive model still works well in additional settings beyond the 5 shown. At the moment, I am unsure how general the method is along the "task" axis. If I recall correctly, there are 30-50 persona evals in Perez et al. Would we expect the results to hold up for all of them?
- Relatedly, I am curious if the authors have any thoughts or experimental evidence on whether we expect the broad behaviors (but not the theory) to generalize to more complex settings beyond concept learning. In particular, many recent works have studied how language models learn in-context in simple decision-making settings such as bandit tasks [1,2,3,4], where the latter of those finds that steering vectors can modulate an agent's belief state about what action is best to take. Could the authors share any thoughts on how we could understand steering vs. ICL for decision-making?
- Diversity of results: When I look at figures 3, 4, and 5, I feel like I'm essentially looking at different views of the same data, which makes the paper feel a bit thin from a results perspective. I think the paper would be strengthened by another set of experiments: e.g. understanding how some parameters of the task affect the downstream relationship between steering and ICL, or perhaps looking into the effects of finetuning models in the same setting – I'd be very curious to understand how it would differ from steering or ICL in changing models' behavior.

[1] Binz, Marcel, and Eric Schulz. "Using cognitive psychology to understand GPT-3." Proceedings of the National Academy of Sciences 120.6 (2023): e2218523120.
[2] Krishnamurthy, Akshay, et al. "Can large language models explore in-context?." Advances in Neural Information Processing Systems 37 (2024): 120124-120158.
[3] Nie, Allen, et al. "Evolve: Evaluating and optimizing llms for exploration." arXiv preprint arXiv:2410.06238 (2024). 
[4] Rahn, Nate, Pierluca D'Oro, and Marc G. Bellemare. "Controlling Large Language Model Agents with Entropic Activation Steering." ICML 2024 Workshop on Mechanistic Interpretability.

### Questions
- It looks like the great majority of tasks studied saturate quite quickly in n_icl_shots, c.f. the plots in figure 5 are very red. Do the authors know of tasks where this doesn't happen? Would we expect the results to be different?
- How was the dataset constructed for CAA?
- Out of curiosity – it seems that one interesting thing here is that the ICL examples come from some dataset of examples, and vary in number, while the CAA vectors are constructed from some other (potentially overlapping) dataset, and are fixed. Do you think there is a correspondence between a particular set of in-context examples, and some steering vector? I.e. is there a steering vector which affects the model's beliefs in a way that is precisely equivalent to including those examples in context?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a unified Bayesian framework that connects in-context learning and activation steering in large language models. The authors argue that both methods operate by updating an LLM’s beliefs about latent concepts: ICL provides evidence through context, adjusting likelihoods of concepts. Activation steering modifies priors over these concepts by manipulating hidden representations.

They formalize this connection through a Belief Update Model, predicting three key phenomena: Sigmoidal learning curves for ICL as evidence accumulates. Shifts in these curves due to activation steering. Additive effects between context-based and activation-based interventions.

Empirical experiments on five “persona” datasets (e.g., Machiavellianism, Psychopathy, Narcissism, Moral Nihilism) confirm these predictions. The results show that steering magnitude and context length jointly shape model behavior, revealing phase boundaries in LLM belief states.

### Strengths
- The used method is interesting in that it links two major inference-time control methods, prompting and activation steering under a single Bayesian belief-update paradigm.

- The Bayesian framing builds upon prior cognitive and interpretability theories, offering a formal probabilistic model of LLM behavior.

- Empirical Validation: Robust experiments across multiple models (Llama-3.1-8B, Qwen-2.5-7B, Gemma-2-9B) support the theory’s predictive accuracy.

### Weaknesses
- The model assumes linear representation of belief and separable concept directions, which may not hold for complex or abstract concepts.

- The experiments focus on binary, persona-style datasets. Generalization to open-ended or multi-concept tasks remains uncertain.

- While the Bayesian model predicts empirical data well, it is unclear how it performs out-of-domain or on unseen LLM architectures.

### Questions
Please see the weaknesses, and:

- Are the observed phase boundaries consistent across different LLM sizes (e.g., 70B+ models), or do scaling laws alter belief dynamics?

- To what extent do steering effects persist across layers?do certain layers dominate belief updates?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper provides a hypothesis for ICL and activation steering: they both manipulate the internal feature representations.
With assumptions, the paper derives a formulation for the relationship between such representation change and concept probability.
Based on the derivation, the paper makes three predictions: (1) sigmoid change of concept during ICL; (2) steering will have a linear effect on log-prior odds for concept ; (3) log posterior odds will be additively impacted by varying in-context examples and steering magnitude.
The paper couples the prediction with experiments.

### Strengths
(1) The paper gives a mathematical derivation of the relationship between representation change and concept probability.

(2) Mathematical analyses are coupled with well-designed experiments. The experiments support the predictions from the mathematical formulations.

### Weaknesses
(1) The paper assumes in L294: $\frac{p(c|v)}{p(c'|v)}=\frac{p(c)}{p(c')}$. Could the author give an explanation of this assumption?

(2) Equation (6) is not clear to me. Could the author give a step-by-step derivation?

(3) The term "concept-consistent label" is used but not defined. What is the meaning of "concept-consistent label"? what does $l_i=y_i^c$ mean?

(4) The considered datasets are "we used a selection of datasets that represent relatively low-probability concepts in LLMs, yet consist of behaviors that a sufficiently capable LLM would be able to follow accurately." While this is a suitable choice for experiments, it also limits the dataset's diversity. In other words, we do not know whether the "sigmoid" phenomenon happens for a high-probability concept.

### Questions
(1) In visualizations of Figures 3 and 4, some curves look like a sigmoid, and some are not (such as Figure 4, bottom right). How do we convince ourselves that the curve is sigmoid?

### Soundness
3

### Presentation
2

### Contribution
2
