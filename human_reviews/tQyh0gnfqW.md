# Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation

- Decision: Accept (Poster)
- Scores: 6, 3, 8

## Abstract
Transporting between arbitrary distributions is a fundamental goal in generative modeling.
Recently proposed diffusion bridge models provide a potential solution, but they rely on a joint distribution that is difficult to obtain in practice.
Furthermore, formulations based on continuous domains limit their applicability to discrete domains such as graphs.
To overcome these limitations, we propose Discrete Diffusion Schrödinger Bridge Matching (DDSBM), a novel framework that utilizes continuous-time Markov chains to solve the SB problem in a high-dimensional discrete state space.
Our approach extends Iterative Markovian Fitting to discrete domains, and we have proved its convergence to the SB.
Furthermore, we adapt our framework for the graph transformation, and show that our design choice of underlying dynamics characterized by independent modifications of nodes and edges can be interpreted as the entropy-regularized version of optimal transport with a cost function described by the graph edit distance.
To demonstrate the effectiveness of our framework, we have applied DDSBM to molecular optimization in the field of chemistry.
Experimental results demonstrate that DDSBM effectively optimizes molecules' property-of-interest with minimal graph transformation, successfully retaining other features. Source code is available [here](https://github.com/junhkim1226/DDSBM).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Adapt diffusion Schrodinger bridge matching to the discrete domain and instantiate it through molecular optimization

### Strengths
1. This method adapts diffusion Schrodinger bridge matching which opens a new avenue for this task.
2. The experimental setting for molecular optimization is a very relevant downstream task for the discrete domain, and it builds up some baselines for that task.
3. The work provides corresponding theoretical guarantees.

### Weaknesses
1. Globally, the structure is complete but the writing is not easy to follow. The writing can be enhanced at least in the following way:
* The square brackets around the citation should not appear twice.
* The name of section 2.2 'SOLUTION METHOD' needs to be optimized.
* The supplementary material also contains the body part of the paper.
* The structure section 2/3 is a bit confusing:
  * There should be an introduction to the continuous methods (IPF, IMF) as prior knowledge since the method highly dependent on them, then a transition to the method for the discrete domain. Here, section 2 starts by discussing the discrete domain and introducing the definitions together, which makes it a bit confusing which parts come from previous work and which part comes from this work.
  *  Consequently, the insights about the transition from continuous domain to discrete domain are missing.

2. Under the claim of proposing a discrete model for DSBM, the results only contain 2 molecule datasets with few supportive ablation experiments being given. It will support better the contribution of this work to give toy results for examples except for molecule optimization such as toy datasets with discrete features, or results on other types of graphs, or more ablations, or pure generation results (as in continuous SB papers), or more visualizations of the resulting optimization chain. No need to have them all, but similar supportive evidence would definitely make the experiments more complete.

### Questions
How significantly does the choice of graph-matching algorithm impact the overall performance of the method?
If it has a substantial influence, there is another reason that experiments for some simpler modalities without such complex matching may help to clarify the contribution.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces discrete diffusion Schrödinger bridge model using continuous-time Markov chains. The proposed model was validated on molecular optimization problem.

### Strengths
Strong performance compared to diffusion bridge model.

### Weaknesses
Overall, this paper failed to distinguish its own contributions from existing work and to compared itself with relevant works. First of all, the proposed discrete diffusion Schrödinger bridge matching (DDSBM) highly relies on continuous-time Markov chains (CMTC). However, this paper failed to refer flow matching models [1, 2] which introduce CMTC for discrete data domains. 

[1] Campbell, Andrew, et al. "Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design." Forty-first International Conference on Machine Learning.

[2] Gat, Itai, et al. "Discrete flow matching." arXiv preprint arXiv:2407.15595 (2024).

And, this is not the first paper that solving Schrödinger bridge (SB) problem in discrete state spaces. There are works for discrete SB [3, 4]. However, the relevance and difference between the proposed method were not discussed.

[3] Harchaoui, Zaid, Lang Liu, and Soumik Pal. "Asymptotics of discrete Schrödinger bridges via chaos decomposition." Bernoulli 30.3 (2024): 1945-1970.

[4] Chow, Shui-Nee, et al. "A discrete Schrodinger bridge problem via optimal transport on graphs." calculus of variations 20.33 (2021): 34.

### Questions
1. What is the relevance and difference between the proposed method and existing discrete SB models in Weakness?

2. How does the performance of molecular optimization compare with discrete flow matching models?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work presents Discrete Diffusion Schrödinger Bridge Matching (DDSBM), a framework that adapts continuous-time Markov chains (CTMCs) to address the Schrödinger Bridge (SB) problem within high-dimensional discrete spaces. Using the Iterative Markovian Fitting (IMF) technique, DDSBM enables optimal graph modifications by minimizing structural changes, which is particularly valuable for molecular optimization tasks in drug and material discovery. In this context, the framework aligns with graph edit distance (GED), allowing efficient property-driven modifications while preserving the molecule's structure. Experiments show DDSBM achieves minimal structural shifts and successfully maintains desirable molecular properties, outperforming traditional graph translation methods.

### Strengths
- The Schrödinger bridge problem is well studied in the continuous-state diffusion literature, but has, to the best of my knowledge, not yet been applied to the discrete setting. The paper nicely bridges this gap, hence proposing a valuable contribution to the discrete diffusion literature.
- The paper looks sound and technically strong.
- Empirical results seem to indicate that this approach outperforms previous baselines by a large margin.

### Weaknesses
- The choice of the FCD metric the evaluate the graph structure distributions is surprising, since it is typically used to assess distribution learning from a chemical perspective. Why not use NSPDK for example ?
- I’d like to see some error bars on your results.
- The method description is quite technical, and might be hard to understand for the average graph machine learning practioner. It could be worth it to include a short tutorial on the Schrödinder Bridge problem to ease the reader understanding.

### Questions
- What is the Hungarian algorithm ? You mention it in section 3.3. I'd appreciate if you could drop a reference or give some details in appendix.
- The baselines outperform your approach in terms of Validity. Even though the results on FCD and NLL indicate that your method is stronger, I'd appreciate if you could comment on this metric.

### Soundness
3

### Presentation
3

### Contribution
3
