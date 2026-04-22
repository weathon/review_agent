# The Geometry of Reasoning: Flowing Logics in Representation Space

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
We study how large language models (LLMs) "think" through their representation space. 
We propose a novel geometric framework that models an LLM's reasoning as flows---embedding trajectories evolving where logic goes. 
We disentangle logical structure from semantics by employing the same natural deduction propositions with varied semantic carriers, allowing us to test whether LLMs internalize logic beyond surface form. 
This perspective connects reasoning with geometric quantities such as position, velocity, and curvature, enabling formal analysis in representation and concept spaces. 
Our theory establishes: (1) LLM reasoning corresponds to smooth flows in representation space, and (2) logical statements act as local controllers of these flows' velocities. 
Using learned representation proxies, we design controlled experiments to visualize and quantify reasoning flows, providing empirical validation of our theoretical framework. 
Our findings indicate that training solely via next-token prediction can lead LLMs to internalize logical invariants as higher-order geometry in representation space, challenging the "stochastic parrot" argument.
Experiments across Qwen and LLaMA model families further suggest the presence of a general, possibly universal, representational law underlying machine understanding and human linguistic regularities, largely independent of specific training recipes or model architectures. 
Our work serves as both a conceptual foundation and practical tools for studying reasoning phenomena, offering a new lens for interpretability and formal analysis of LLMs' behavior.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper analyzes “reasoning” in LLMs, particularly the reasoning flows (trajectories) in representation space. 
Logic is posited to act as a local controller of velocity along these flows, with Menger curvature used to capture second-order structure. 
The authors construct a dataset that keeps formal logical skeletons fixed while varying topical and linguistic “carriers”, then extract hidden states from Qwen3/LLaMA3 to compare position, first-order/second-order differences and similarities. They report that while positions cluster by surface semantics, velocity/curvature similarities align by logic, supporting the claim that logic governs flow dynamics.

### Strengths
- An interesting work on reasoning, the authors view logic as a carrier-invariant framework for reasoning, and they test if LLMs have learned these structural invariants in their own embedding space.

- Empirical comparisons across model sizes support qualitative claims.

- Some interesting findings, for example, in the representation space, sentences on the same topic cluster together. However, when looking at the differences of curvature, logical structure emerges as the dominant factor even across unrelated topics and languages.

### Weaknesses
- The paper is not easy to follow. Many key notions are introduced before being properly defined or motivated, which makes it difficult for readers outside the narrow intersection of geometry and logic to follow the argument. For example, the term “flow” appears early (Abstract, intro) for many times, but its formalization as a sequence of hidden states, or its mapping to logical entailment steps, is only explained several pages later (Sec. 4.2). Similarly, “menger curvature” is used as a core analytic measure before the geometric intuition or connection to reasoning trajectories is established. Also, it is often mentioned without details: what the Menger curvature is, how it is calculated, or at least some intuition should be provided. A short schematic or example trajectory would help. Recent ICLR/ICML papers have handled similar conceptual density more clearly, see, for instance, “The Geometry of Categorical and Hierarchical Concepts in LLMs” (Park et al., ICLR 2025) and “Tracing the Representation Geometry of Language Models from Pretraining to Post-training” (Zhang et al., ICML 2025).

- The paper cites several strands but does not sufficiently contrast contributions with recent, closely related geometry/trajectory work at ICLR/ICML.

- Its evidential basis is correlational and methodologically fragile in its current form. For example, the smoothness hypothesis needs independent validation. Hyp. 4.6 is asserted with a construction in App. C.1, but the fitted smooth curve could be an artifact. Maybe consider reporting results on shuffled/phase-randomized controls as baseline. 
Maybe also consider adding causal tests, stronger statistical treatment, external benchmark dataset validation, and clearer positioning/discussion vs. recent ICLR/ICML work; this submission could become a citable contribution to the geometry-of-reasoning literature. But I'm not sure if it can or will be done in the CR.

### Questions
- It is not clear that the dataset isolates formal logic from its semantic context. There is a lack of systematic analysis, e.g., statistical analysis, to demonstrate the separation. In other words, how to ensure the disentanglement of the logic format and the semantic context?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a framework to analyze reasoning in large language models by viewing a chain of thought as a trajectory in representation space. The trajectory is described with three geometric descriptors: position, velocity, and curvature. Using this framework, the authors analyze reasoning traces in Qwen3 and LLaMA3 models. They synthetically generate multi-step logical sequences that share the same logical skeleton while varying topic and language. Applying their analysis, they find that velocity and curvature are more similar for traces that follow the same logic and remain low across different logics even when topic and language match. By contrast, position is more similar for inputs written in the same language. The authors interpret this as evidence that models encode logical structure in the dynamics of their representations, while static position reflects surface form.

### Strengths
- The paper Introduces a relatively novel geometric perspective on reasoning, modeling chains of thought as trajectories in representation space.
- The authors provide a clear formalization of curvature and reasonable metrics to estimate it.
- The paper includes convincing evidence that first- and second-order properties (velocity, curvature) track logic, while position reflects surface form or semantics.

### Weaknesses
1. Lack of scaling analysis. The paper does not analyze how similarity patterns should evolve with model size. Table 1 shows no clear trend across Qwen3 0.6B, 1.7B, 4B and LLaMA3 8B. Position, velocity, and curvature similarities fluctuate rather than improving systematically with scale. The authors should discuss whether this is expected and what theory or diagnostics would predict size effects.
2. Traces are not self-generated by the evaluated models. The reasoning sequences are produced by a much larger model and then fed to smaller models for analysis. This can introduce a distribution shift and may mask effects that would appear if each model were analyzed on its own traces. The paper should discuss how results might change under self-generated or mixed setups.
3. Limited link to performance and disentanglement. Related to point 2, the paper does not examine whether models with stronger reasoning performance exhibit a clearer disentanglement between logic-driven dynamics (velocity, curvature) and surface-driven position. A mild suggestion is to correlate the geometric measures with task accuracy and to repeat the analysis on self-generated traces to see if better-performing models show stronger disentanglement.

### Questions
1. How should similarity patterns for position, velocity, and curvature change with model size? Table 1 shows no clear trend across Qwen3 0.6B, 1.7B, 4B and LLaMA3 8B. Is the absence of a trend expected? 
2. How would the results differ if trajectories were computed on self-generated traces rather than sequences authored by a much larger model? 
3. Do better-performing models exhibit stronger disentanglement between logic-driven dynamics (velocity, curvature) and surface-driven position? Can you correlate the geometric measures with task accuracy and repeat the analysis on self-generated traces to test this?

### Soundness
3

### Presentation
4

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
The paper introduces a new framework to mathematically model what happens when LLM's reason. 
It is hypothesized that reasoning traces can be characterized by the geometric properties of the path that the reasoning tokens trace through an LLM's latent space.
Evidence is presented suggesting that the latent 'reasoning flows' for similar kinds of reasoning (following a certain template) also exhibit similarities on the second and third order geometric properties of velocity and curvature.
This evidence is based on studying the given geometric properties for activations of various models when forwarded on a novel dataset where the same logical template is used to construct different chains-of-thought in various domains and languages.

### Strengths
Interesting new framework for thinking about reasoning in LLMs.

The concepts that are appealed to are made mathematically precise, and the math is introduced step-by-step in a way that is easy to follow (relative to the complexity of the subject matter).

The results show that when an LLM reasons according to the same logical pattern/template (but about different topics and in different languages), its activations move through latent space with similar curvature (how small or large is the angle made between triples of subsequent token activation vectors). And, to a lesser extent also with similar velocity (how far apart is each subsequent token activation vector). This is an interesting finding.

### Weaknesses
There is one claim I believe is too strong, on line 429-431: "Together, these results show that LLMs internalize latent logical structure beyond surface form. They are not mere stochastic parrots (Bender et al., 2021): whereas humans formalized logic only in the 20th century (Bochenski & Thomas, 1961), LLMs acquire it emergently from large-scale data—a hallmark of genuine intelligence."  I'm not sure these results really provide much evidence against the 'stochastic parrot theory'. The results, as I understand them, show that the same logical structure elicits similar trajectories (in the second- and third-order geometric sense) through latent space. But the logic is not just similar, it is identical. Wouldn't we need to show that the LLM correctly identifies the (vast majority of) instances as instantiations of the logical problems (in an exact, not fuzzy sense).  And as for the second part, the experiments use Qwen models, which are not merely pre-trained on large-scale data, but also mid- and post- trained, right? so the observations might be due to those training stages, which quite likely involve supervised/reinforcement training on logic problems. Finally, the juxtaposition between discovering-from-data and human-discovery, seems mistaken. Even if the model 'discovers' it from the data, that is likely due to the human discovery being described/present in the data.

If this is addressed, I'm happy to increase my score.

### Questions
What can we learn from the fact that logics B and C are very similar in curvature? Does this suggest that the LLM uses the same mechanisms for both? Is the similarity a surprise to you, would you expect it from looking at the templates?

Presumably, not the entire latent space is relevant for (logical) reasoning. Would you expect there to be a subspace in which the correlation between (the velocity/curvature of) samples of the same logic template is even stronger, perhaps much stronger (close to one)?

What are the individual rows/columns within the blocks (L:E through L:A) in Figure 2, are they the different topics and languages?

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
4

### Summary
This paper introduces a geometric framework for analyzing reasoning in LLMs. The core idea is to treat reasoning as smooth trajectories (“flows”) in representation space, where logical operations act as differential controllers of the embedding’s local velocity and curvature.

### Strengths
- the topic of reasoning as continuous geometric flows within embedding space is interesting and the proposed formulation combining mapping and alignement provide a good abstraction linking reasoning to symbolic representation. 

- The construction of continuous C^1 trajectories via the relaxed prefix-mask mechanism (Proposition C.4) is both novel and technically precise

- the proposed dataset and empirical results show good results.

### Weaknesses
- the proposition 4.10 that connects logic to the integral of velocity. It is not clear on how it maps between inference rules and specific vector-field constraints on v(s). It is not clear  how logical connectives translate into geometric operations or basis directions in representation space?

- in the mapping A = \Psi \circ \Gamma^{-1}, does\ Gamma injective? 

- how the mapping D_R : \Psĩ \mapsto (\Delta y_t) preserve logical equivalance?

- The logical dataset is generated with GPT-5 templates. How the validation is performed and logical equivalence are guaranteed.

- How empiracly C^1 interpolation is validaded?

### Questions
see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
