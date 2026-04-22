# Case-Guided Sequential Assay Planning in Drug Discovery

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 4

## Abstract
Optimally sequencing experimental assays in drug discovery is a high-stakes planning problem under severe uncertainty and resource constraints. A primary obstacle for standard reinforcement learning (RL) is the absence of an explicit environment simulator or transition data $(s, a, s')$; planning must rely solely on a static database of historical outcomes. We introduce the Implicit Bayesian Markov Decision Process (IBMDP), a model-based RL framework designed for such simulator-free settings. IBMDP constructs a case-guided implicit model of transition dynamics by forming a nonparametric belief distribution using similar historical outcomes. This mechanism enables Bayesian belief updating as evidence accumulates and employs ensemble MCTS planning to generate stable policies that balance information gain toward desired outcomes with resource efficiency. We validate IBMDP through comprehensive experiments. On a real-world central nervous system (CNS) drug discovery task, IBMDP reduced resource consumption by up to 92\% compared to established heuristics while maintaining decision confidence. To rigorously assess decision quality, we also benchmarked IBMDP in a synthetic environment with a computable optimal policy. Our framework achieves significantly higher alignment with this optimal policy than a deterministic value iteration alternative that uses the same similarity-based model, demonstrating the superiority of our ensemble planner. IBMDP offers a practical solution for sequential experimental design in data-rich but simulator-poor domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
- Formulates the experimental assay selection problem into an RL planning task, where different assays are actions (each with varying cost and informativeness about the target outcome) as well as the action to stop testing. 
- Proposes a new framework called an "Implicit Bayesian Markov decision process" under the assumption that similar molecular representation $x$ yield similar target properties via the defined similarity weights. Use the weights to construct a similarity-based transition function and use this to perform MCTS search with a tree ensemble.

### Strengths
- Originality and significance: This reviewer believe these areas are lacking (see weakness). In particular, the generalizability of the proposed method to other domains or other experiments seems lacking unless the proposed weight function can be justified through prior literature or strong empirical performance with the real-world dataset. 
- Quality: Experiments could be improved. 
- Clarity: The current presentation of the ideas in the methods section is clear. Some connections to related work (especially regarding the transition assumption for model-based RL, e.g., "While conventional frameworks assume access to either environment simulators or transition data," (Line 1023)") could be clarified, since typically model-based RL does not assume that all transition tuples are available or that a transition model can be perfectly reconstructed from the historical data, but rather it also operates under the assumption that the historical data is collected from a behavioral policy (often different from the optimal or the new policy being tested) [Liu et al., 2020].

 Liu et al., 2020. Provably good batch reinforcement learning without great exploration

### Weaknesses
# Method:
- Lacking justification for the assumption about the similarity weights: It's unclear (at least the current manuscript is lacking justification from the relevant field) that similar molecular structures behave similarly in terms of the target property, and whether the proposed similarity function accurately captures this relationship. This requires either grounding in molecular structure literature or rigorous justification with real world experiments which the current paper does not provide.
- If the weight functions in Equations (4) and (5) cannot be justified by prior work in this domain, then they should be clearly stated as the authors’ own design choice, which may not generalize to other settings of the assay sequencing problem and certainly not to other domains (e.g., pixel-based state representations or most high-dimensional states) where the similarity of the pixel states does not necessarily mean similarity in the underlying states. 
- The novelty of MCTS with double progressive widening is not clearly established -- if there's any modification or technical contribution within the MCTS search framework that the authors are proposing as new, then that modification should be highlighted. The current work's novelty is mainly driven by the similarity-based weight function, which this reviewer has questions about.

# Experiments:
- Figure 2 shows the trade-offs between state uncertainty and action cost. However, state uncertainty is measured under the assumed similarity weights and transitions. It is therefore unclear how to interpret these results or how they depend on the validity of the assumed weight functions. In order to justify the proposed weight functions, the authors could demonstrate whether the assumed weights can explain the transitions observed in the real dataset and compare the policy values of the learned policy versus the behavioral (from the historical dataset) or the heuristic policy. 
- For Table 2, authors could add a baseline of learning the dynamics model (without similarity weights) and applying MCTS on top of that model. This would be a reasonable MCTS baseline to include to compare with the proposed weighted transition-based MCTS.
- The current experiment result with 100 independent trials has a small sample size to understand whether the results are statistically significant. 
- An ablation comparing the performance of an ensemble of MCTS versus a single tree would be useful to include, especially since this paper mentions that the "robust ensemble" is one of the main contributions (Line 82), but currently it's unclear whether and how the ensembling helps with "robustness" of the method.

### Questions
Questions about the justification for the similarity based weight function and possible additions to the experiments are elaborated in Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents the Implicit Bayesian Markov Decision Process (IBMDP), a model-based reinforcement learning framework for sequential assay planning in drug discovery. The main contribution is an implicit transition model that uses similarity-weighted sampling to form a nonparametric belief distribution over historical cases, enabling Bayesian belief updating as assay evidence accumulates. The authors prove that it is equivalent to POMDP belief updates where the hidden state represents a latent index over historical compounds. Planning employs ensemble MCTS-DPW to generate robust policies balancing information gain with resource efficiency. On real-world CNS drug discovery data, IBMDP reduced costs by up to 92% versus rule-based heuristics.

### Strengths
1. This paper studies an important but underexplored problem, sequential experimental design when only static historical data exists, without simulators or explicit transition tuples, and it is well-motivated by real pharmaceutical constraints where mechanistic models are unavailable.

2. The authors provide theoretical analysis by formalizing the proposed approach as a POMDP and proving that similarity weight updates implement exact Bayesian belief updates.

3. The authors conducted experiments on the real-world drug discovery dataset and demonstrate that the proposed method achieve substantial cost reductions.

### Weaknesses
1. The proposed method highly depends on the quality of the historical dataset $\mathcal{D}$. Unlike model-free RL, it cannot discover strategies that are not present in the historical data, so any gaps or biases in the data can lead to suboptimal decisions. It would be better if the authors could discuss whether this is a valid concern and how to address it.

2. The authors' primary claim of practical utility is on the real-world case study, but the only baseline used is a rule-based decision strategy. It is unclear if IBMDP's impressive cost reduction is due to its multi-step planning or if any simple computational method (e.g., a greedy one) would have achieved similar results over the non-computational heuristic.

3. Figures and tables in the experiments section are difficult to read and interpret. It would be better if the authors could revise the presentation of the experiments section to explain clearly what Figure 2 and Table 1 mean. The caption of all the tables in the paper also overlaps with the tables, and the authors should also fix them.

### Questions
1. Please see the comments in the Weaknesses part.

2. The proposed method uses a variance-normalized Euclidean distance to calculate the similarity between compounds, but it might be violated by complex biological realities, such as nonlinear or threshold effects. Could the authors elaborate more on this part?

3. The paper states MCTS-DPW is "particularly well-suited"  but does not experimentally justify this specific choice over other, simpler planners. Could the authors discuss more on the choice?

4. How does the framework perform when a candidate compound $x_*$ is "out-of-distribution" (i.e., has low similarity to all cases in $\mathcal{D}$)?

### Soundness
2

### Presentation
1

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
The paper addresses the problem of planning with a static database of historical outcomes under simulator-free settings, and proposes an Implicit Bayesian Markov Decision Process (IBMDP), which is a model-based RL framework for sequential assay planning when no transition data tuples or simulators exist. IBMDP builds an implicit transition dynamics model using similarity-weighting historical outcomes and updates weights with Bayesian belief updating. The experiments on diverse tasks show that the proposed method significantly reduces resource consumption compared to baselines and provides significantly high alignment with an optimal policy of a synthetic benchmark environment.

### Strengths
- The paper addresses an important problem: decision-making without simulators is practically useful, and the paper is well-motivated.

- The proposed method provides similarity-weighted sampling that is intuitive and computationally tractable.

- The paper is generally well-written.

- The empirical evaluation of IBMDP includes both a real-world drug discovery task and a synthetic benchmark.

### Weaknesses
- For the real-world drug discovery task, the baselines are insufficient. Comparisons are performed against rule-based heuristics; e.g., kNN-Thompson alternatives compatible with the same posterior predictive and constraints would strengthen the effectiveness of IBMDP.

- Theoretical analysis: 

- - No convergence guarantees or regret bounds are provided. Unlike other Bayesian RL methods with proven regret bounds, IBMDP offers empirical robustness.

- - The provided consistency proof is weak. Theorem in D.6 only holds for the synthetic linear case with independent features, this is where the method is least needed.

- The reported saving claim does not include distribution statistics. See questions below.

- Missing ablations: 

- - The sensitivity of the method to thresholds $ε, τ$, metric, and kernel choice is not sufficiently explored. 

- - What is the contribution of ensemble vs. single MCTS?

- MINOR: 
- - The blank space after Table captions (Table 1 and 2) should be corrected.

### Questions
- How is the sensitivity of savings and the accuracy of IBMDP to the design choices stated above (see Weaknesses)?

-  Can you provide bootstrap confidence intervals for cost savings across many compounds, for the 92% reduction?

- See the Weaknesses section above.

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
The paper tackles the challenge of sequential assay selection in drug discovery under simulator-free conditions i.e., when no explicit transition data (s, a, s′) is available. Traditional RL approaches fail here because they require explicit simulators or learned transition dynamics. The authors introduce the Implicit Bayesian Markov Decision Process (IBMDP), a model-based RL framework that constructs a case-guided implicit model of transition dynamics using historical assay outcomes. Instead of learning P(s′|s, a), it uses similarity-weighted sampling from past compound outcomes to implicitly simulate plausible next states.

### Strengths
The paper clearly identifies a gap of planning without simulators and formalizes a principled approach via implicit dynamics. The appendix convincingly reinterprets the similarity mechanism as Bayesian belief updating in a POMDP where the latent variable indexes historical prototypes. This elevates what could be seen as heuristic into a grounded probabilistic framework. The approach bridges case-based reasoning, kernel RL, and Bayesian experimental design, offering a coherent hybrid that is both intuitive and effective in domains like drug discovery. Using ensembles (MLASP) is also a thoughtful design choice to address stochastic variance from sampling-based transitions.

### Weaknesses
- There is a lack of comparison with existing causal bayesian optimization approaches and the authors do not cite relevant work such as Durand et al 2025 https://arxiv.org/pdf/2503.19554 and other CBO works. 

- The approach is highly dependent on historical coverage: that is, it can only sample from observed compound profiles. This means it cannot generalize beyond the chemical or assay distribution of the historical dataset. This is acknowledged but severely limits applicability in novel discovery spaces.

- The paper uses a variance-normalized Euclidean kernel across heterogeneous features (QSAR predictions, assay outcomes). There is no principled validation that this metric reflects biological similarity or assay informativeness.

- The choice of lambda_k weights is ad hoc without sufficient explanation

- The mapping between kernel similarity and likelihood is conceptually elegant but mathematically thin in terms of how good the bayesian updating approximation is. There’s no rigorous treatment of calibration or sensitivity to the temperature.

- Constraint handling is similarly not adequately discussed since feasibility constraints and terminal uncertainty are hard-coded thresholds and their influence on pareto efficiency isnt analysed.

- The CNS case study involves only ~220 compounds and a few assays. Comparison is only to rule-based heuristics and a deterministic similarity VI baseline without any comparison with active learning, BOED, or uncertainty-driven planning methods.

### Questions
1. How sensitive is IBMDP to the representativeness of the historical database? Would performance degrade sharply when the test compound lies outside the convex hull of D? Could you adapt the method to incorporate uncertainty about the similarity weights themselves (e.g., via Bayesian kernel hyperpriors)?

2. Why assume Gaussian noise in the implicit likelihood (Eq. 10)? Biological assays are often heavy-tailed or discrete. 

3. Have you compared the variance-normalized Euclidean distance to molecular fingerprints or learned embeddings? 

4. How robust are decisions to the choice of the temperature parameter? Did you conduct ablation or sensitivity analysis?

5. Would a multi-objective formulation (e.g., Pareto front optimization in cost–uncertainty–likelihood space) be more interpretable than fixed constraints?

6. How would IBMDP compare to Bayesian optimization or active learning with uncertainty-based acquisition?

7. For larger datasets or higher-dimensional features, how feasible is the O(|D|·d) similarity computation? Have you explored approximate nearest-neighbor methods or mini-batching?

8. Could the implicit transition model be coupled with learned latent dynamics (e.g., via VAEs) to interpolate between historical cases?

9. How might IBMDP integrate active learning to identify which new compounds to assay next, not just which assays to run?

### Soundness
3

### Presentation
2

### Contribution
2
