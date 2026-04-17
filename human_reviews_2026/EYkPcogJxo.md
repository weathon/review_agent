# Cognitive Structure Generation via Diffusion Models with Policy Optimization

- Decision: Reject
- Scores: 6, 4, 8, 2

## Abstract
Cognitive structure (CS), a student's construction of concepts and inter-concept relations, has long been recognized as a foundational notion in educational psychology, yet remains largely unassessable in practice. Existing approaches such as knowledge tracing (KT) and cognitive diagnosis (CD) simplify and indirectly approximate CS, but they intertwine representation learning with prediction objectives, limiting generalization, interpretability, and reuse across tasks. To address this gap, we propose Cognitive Structure Generation (CSG), a task-agnostic framework that explicitly models CS through generative modeling. Based on educational theories, CSG first pretrains a Cognitive Structure Diffusion Probabilistic Model (CSDPM) and then applies reinforcement learning with SOLO-based hierarchical rewards to align generation with genuine cognitive development. By decoupling cognitive structure  representation from downstream prediction, CSG produces interpretable and transferable cognitive structures that can be seamlessly integrated into diverse student modeling tasks. Experiments on four real-world datasets show that CSG yields more comprehensive representations, substantially improving performance while offering enhanced interpretability and modularity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work frames student modeling as Cognitive Structure Generation (CSG): given a student’s learning history, generate a personalized graph. The nodes encode concept mastery and edges encode inter-concept relations. It instantiates this with a Cognitive Structure Diffusion Probabilistic Model (CSDPM) trained in two stages: (i) pretraining on simulated cognitive structures inferred from logs via a rule-based procedure, and (ii) policy optimization of the reverse diffusion process to align generated graphs.

### Strengths
- Overall, I like the idea and the problem setting behind the proposed methods. Modeling dynamic, inter-concept graphs is both important and challenging,  especially given the limited data available per student, the lack of concrete evaluation methods, and the difficulty of optimizing over discrete structures. I also appreciate the effort to combine KT and CD, since prediction and interpretability are equally important in the educational domain.

- The method’s use of a discrete graph diffusion model combined with policy optimization guided by the SOLO taxonomy is interesting. The pipeline, i.e., pretraining on simulated structures followed by reinforcement learning (RL) alignment, is novel within the education modeling space.

- The paper is very well-presented. I enjoyed both the clarity of the writing and the quality of the figures.

### Weaknesses
- The training signal for Stage-I comes from rule-based simulation computed from the same interaction logs used downstream, with Gaussian noise added t the Q-matrix for robustness. The mapping $f_{\mathrm{UOC}}, f_{\mathrm{UOR}}$ is basically weighted correctness; then values are rounded and one-hot encoded, which discards uncertainty and may bake in label-like information before KT/CD. A stronger justification or comparison against alternative simulators (Bayesian knowledge tracing, IRT-based posteriors, or human-elicited maps) is needed.
- The 8:1:1 split at the interaction level seems somewhat inconsistent with the paper’s introduction section about “generalization” and “cold-start”. It would be informative to see how the models perform under smaller data regimes, especially given that several heuristics are already used so the method should work with small data. 
- The edge construction $f_{\text {UOR }}$ uses co-occurrence within items as evidence of a relation between concepts. This conflates test design with student cognition; edge identifiability is questionable. Visual cases are interesting but anecdotal. A human-judged metric (or even prompt LLMs to evaluate edges) would make it more concrete.
- Three datasets have only 20 items, and the largest uses 57 concepts but task heterogeneity is unclear. It’s hard to conclude that CSG scales to hundreds–thousands of concepts or to more realistic, noisy Q-matrices. Reported generation times are low, but training cost of diffusion + RL may be substantial.
- The graph size is fixed (predefined by the number of concepts). In real-world educational settings, graph structures are dynamic: concept sets can expand or contract, and nodes often exist at different abstraction levels. For example, “linear algebra” as a field, “dot product” as a concept, and “specific exercises” as instances, which I think are all presented in student's mind. How does this method extend to these situations?

### Questions
See weaknesses.

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
The authors propose a new student modeling framework named CSG, which aims to explicitly generate a student’s Cognitive Structure (CS) through a diffusion model. Unlike traditional KT and CD methods that implicitly model students’ concept mastery states, this work generates cognitive graphs to represent both the concepts and the construction process of relationships between concepts.

### Strengths
1. The study formalizes the cognitive structure modeling problem as a graph generation task and introduces a diffusion model combined with a RL framework, which is innovative.
2. Experiments are conducted on public datasets with comprehensive results. The proposed method significantly outperforms baselines across multiple metrics (AUC, ACC, RMSE). Ablation studies (V1–V5) further validate the effectiveness of each module.
3. The generated cognitive structures improve performance on both KT and CD downstream tasks, demonstrating strong generalization ability of the learned structures.

### Weaknesses
1. The quality of the cognitive structures is evaluated indirectly through KT/CD task performance, without qualitative validation of the generated structures themselves.
2. There is no direct comparison with other graph generation paradigms (e.g., VAE, GraphGAN,), making it difficult to justify the necessity of the diffusion model.
3. The rule-based “Cognitive Structure Simulation” component relies on handcrafted empirical formulas (e.g., Eq. (1)(2)) without verification against real student thinking patterns, which may introduce bias.

### Questions
1. Are the rule functions (Eq. (1)–(2)) in the simulated cognitive structure still effective across different subjects or question types? Have the authors considered learning these weights from data instead of manually defining them?
2. Could the authors visualize the temporal evolution of the generated cognitive graphs?
3. Why did the authors choose the diffusion model over other generative methods such as VAE?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors present a way to generate personal latent representations called cognitive structures, using diffusion model. They show on several datasets that they outperform either knowledge tracing or cognitive diagnosis techniques.

### Strengths
This is a unique approach that draws a link between knowledge tracing and cognitive diagnosis, two popular research communities in the literature.
It addresses the shortcomings that not everyone may have the same representations for a domain.

The mathematical description of the proposed approach is very clear.

### Weaknesses
To me, the link with SOLO theory is a bit far-fetched: the fact that there would be exactly 5 levels is arbitrary. But that's a nice story.

The presentation contains many LLM-generated sentences: "align generation with genuine cognitive development". It is yet to be proven that the learned representations correspond to the actual cognitive development of students. "authentic levels of cognitive growth" is a bit too much too.

### Questions
Could you please elaborate more about your use of LLMs for writing the paper?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a Cognitive Structure Generation (CSG) task and a corresponding CSDPM method to address it, aiming to improve model generalization and interpretability. Experiments on Knowledge Tracing (KT) and Cognitive Diagnosis (CD) tasks demonstrate the effectiveness of the proposed approach.

### Strengths
The paper attempts to provide a unified method applicable to both KT and CD tasks and reports some empirical improvements.

### Weaknesses
1. The paper overclaims novelty in defining the CSG task. Similar ideas have been explored in prior works such as MSKT (ESWA, 2024) and DiffCog (TLT, 2024).

2. The interpretability analysis is not convincing. While interpretability is stated as a key contribution, the paper lacks quantitative metrics to support this claim. The main body include few interpretability analyses, most of which are placed in the Appendix.

3. The paper provides no theoretical justification or proof to support the soundness of the proposed method.

4. The computational cost of the proposed method is not discussed, leaving questions about its scalability and efficiency.

5. It remains unclear how the method performs on large-scale datasets, especially those with a greater number of knowledge concepts.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
