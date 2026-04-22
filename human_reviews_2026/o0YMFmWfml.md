# MODE: Learning compositional representations of complex systems with Mixtures Of Dynamical Experts

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 6, 2, 4, 6

## Abstract
Dynamical systems in the life sciences are often composed of complex mixtures of overlapping behavioral regimes. Cellular subpopulations may shift from cycling to equilibrium dynamics or branch towards different developmental fates. The transitions between these regimes can appear noisy and irregular, posing a serious challenge to traditional, flow-based modeling techniques which assume locally smooth dynamics. To address this challenge, we propose MODE (Mixture Of Dynamical Experts), a graphical modeling framework whose neural gating mechanism decomposes complex dynamics into sparse, interpretable components, enabling both the unsupervised discovery of behavioral regimes and accurate long-term forecasting across regime transitions. Crucially, because agents in our framework can jump to different governing laws, MODE is especially tailored to the aforementioned noisy transitions. We evaluate our method on a battery of synthetic and real datasets from computational biology. First, we systematically benchmark MODE on an unsupervised classification task using synthetic dynamical snapshot data, including in noisy, few-sample settings. Next, we show how MODE succeeds on challenging forecasting tasks which simulate key cycling and branching processes in cell biology. Finally, we deploy our method on human, single-cell RNA sequencing data and show that it can not only distinguish proliferation from differentiation dynamics but also predict when cells will commit to their ultimate fate, a key outstanding challenge in computational biology.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a unified perspective to address key limitations of existing methods for modelling complex mixtures of overlapping behavioural regimes: the inability of flow-based models to handle branching dynamics, the computational complexity of switching system inference, and the lack of interpretability in many neural approaches. The resulting Mixture Of Dynamical Experts (MODE) framework shows improved performance over baselines like Gaussian Mixture Models, Neural ODEs, etc. across a range of idealised and real-world benchmarks.

### Strengths
1. The paper clearly identifies the failure of traditional flow-based models (like NODEs) in handling complex, overlapping dynamical regimes, using Figure 1 effectively to show how these models improperly "average" distinct fates.
2. The MODE framework is great for its simplicity, combining an intuitive Mixture of Experts (MoE) approach with SINDy-based regressors. This grounds the model in interpretable, sparse symbolic equations, which is a significant advantage for scientific applications.
3. The "Related Work" section is thorough, correctly positioning the paper by contrasting it against standard dynamical systems, specific computational biology flow models, switching systems, and other MoE approaches.

### Weaknesses
I am not an expert in computational biology, so here are the weaknesses I underline, mostly relating to the machine learning methodology:

1. A priori $K$ selection: The most critical limitation of this work is that the number of experts, $K$, must be manually specified corresponding to the number of dynamic regimes. This is impractical for real-world discovery tasks where $K$ is a primary unknown.
2. The paper's solution to trajectory crossing (a known failure of NODEs) is clear. It would be strengthened by contrasting its discrete mixture-based approach with continuous, augmentation-based methods like ANODE [1].
3. Gradient-based optimisation of the gating network is a known challenge (see MixER [2]). The paper's regularization solution (Eq. 4) to prevent the gating network from collapsing is interesting. It could be compared to other methods, such as the K-Means initialization used in MixER.
4. A claim in L297-298 that MODE "does not rely on phase space geometry" appears to contradict the model's formulation, which explicitly uses the state $x$ (the geometry) to parameterize both the gating function $\pi(x)$ and the expert dynamics $f_{\Theta_{s}}(x)$. This needs clarification.



### Minor issues:
- L235: "dynamical"
- L236: "on"

### References:
- [ 1 ] Dupont et al., "Augmented Neural ODEs", NeurIPS 2019
- [ 2 ] Nzoyem et al., "MixER: Better Mixture of Experts Routing for Hierarchical Meta-Learning", SCOPE@ICLR 2025.

### Questions
1) Is the claim that NODEs "average" switching zones (L067) an empirical observation of a common training failure, or a more fundamental theoretical limitation of fitting a single vector field to multi-modal velocity data? Please compare with ANODE for instance?
2) There is minimal quantitative comparison to Neural ODEs, even though there are the subject of much of the criticism in the Related Work. This relates to my question in 1)
3) Eq (4) displays two losses that aim to achieve two opposite things. I have two questions concerning this:
	- It is not clear how the load balancing term prevents expert collapse (despite the additional definition in L1055)
	- Could you please provide an ablation for this?
4) Regarding ablation studies, one examining the polynomial basis of SINDy is much needed, especially when the oscillator uses functions outside the dictionary (Goldbeter experiment).

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
The paper introduces MODE (Mixture Of Dynamical Experts), a mixture-of-experts framework for snapshot dynamical data that jointly (i) clusters heterogeneous dynamical regimes and (ii) forecasts across regime switches/branching. Each expert is a sparse symbolic regressor, combined by a gating distribution and per-expert isotropic noise. Empirically: (i) on elementary dynamics, MODE yields NMI/ARI ≈ 0.96–1.00, strongly outperforming GMM and spectral clustering and approaching a supervised MLP; (ii) on synthetic forecasting tasks, MODE achieves lower Wasserstein distances than MLP/SINDy and commits to branches; (iii) on U2OS scRNA-seq, a 2-expert MODE matches FUCCI cycle vs. exit AUC = 0.98 over 10 seeds.

### Strengths
1. This paper proposes a snapshot-trained MoE for dynamics with interpretable SINDy-style experts, with gating regularizers to avoid collapse and stochastic rollout that commits to fates. Which I believe is new for this field.
2. The method is evaluated on strong and fair elementary benchmarks, which confirms its performance.
3. The paper is well-written, such as the objective (Eq. 3), regularizers (Eq. 4), rollout (Eqs. 5–6) and data generation (appendices) are explicit.

### Weaknesses
1. The decomposition between expert field and stochastic term is not probed.
2. Results focus on low-D synthetic (2–3D) and PCA-5 for scRNA. Please add OOD tests.

### Questions
1. Please consider adding switching-model baselines (Switched Flow Matching; Neural MJP; mixture-NODEs with gate). Use the same K and similar parameter counts. It would be interesting to see the performance gaps.
2. MODE improves W2 but slightly loses W1,x to MLP (0.1363 vs 0.1284). Please explain the trade-off and add per-axis ablations.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present MODE (Mixture of Dynamical Experts), a framework that decomposes complex and noisy biological dynamics into components. MODE enables unsupervised discovery of dynamical regimes and accurate gene expression forecasting across regime transitions. Applied to synthetic and single-cell RNA sequencing data, MODE effectively distinguishes proliferation and differentiation dynamics and predicts cell fate commitment.

### Strengths
1.	The manuscript is clearly written and well organized.
2.	The experiments cover both simulated and real-world datasets, providing comprehensive validation.

### Weaknesses
1.	The advantage or the main goal of developing MODE is not that meaningful. There are many RNA velocity models, like VeloVAE (Gu et al, ICML 2022, LatentVelo ), that are explicitly designed to model cell lineage bifurcation. This significantly reduces the novelty and practical impact of this study. 
2.	The result from the benchmarking is not convincing, since GMM and spectral methods are quite simple and may not be suitable for complicated data. The authors should compare the performance with other methods mentioned in the related work, like flow matching based methods (Meta Flow Matching) and maybe some RNA velocity models like scVelo (Bergen et al, 2020). And it will also be helpful to compare with other MoE models, like DynMoE (Guo et al, 2025).
3.	The scalability is questionable, which is crucial for real single cell RNA sequencing data. The U2OS cell line dataset contains only about 1,000 cells, which is a very limited number in reality. It will be helpful if the authors could run the method on a larger dataset, like the mammalian organogenesis dataset (Cao et al, 2019), which also has branching differential trajectories for this dataset. And I also suggest including metric like cross-boundary direction correctness (CBDir, Qiao et al, 2021) for the real dataset.

### Questions
1.	Why didn't you just use traditional precision, recall and F1 metrics for evaluation?
2.	How much compute time does the MODE model need in each of your experiments?
3.	When generating the data, how will different noise levels affect the model performance?

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
4

### Summary
The paper introduces MODE (Mixture of Dynamical Experts), a new framework designed to model complex systems, like those in computational biology (e.g., cell differentiation) where the behavior shifts dramatically over time or across different states. Unlike traditional models (like Neural ODEs) that assume a single, smooth governing rule, MODE uses a mixture-of-experts approach. It learns multiple simple, interpretable equations (i.e., experts) and a gating function that dynamically chooses the right equation for a given state. This allows MODE to successfully model systems that bifurcate or transition between different operating regimes.

### Strengths
- MODE’s formulation as a mixture of sparse dynamical regressors with neural gating is conceptually simple yet powerful. It bridges classical sparse regression (e.g., SINDy) and modern mixture models, resulting in interpretable and flexible dynamics decomposition.

- Challenges introduced in L37-43 with an interesting example of RNA sequencing make sense. Also, the problem statement (Consequently, a large body of research at the intersection of computational biology and data-driven dynamical systems has been devoted to the modeling of snapshot data. Regard the action error diagnosis as a node classification task.) is reasonable.

- The model’s ability to discover latent regimes without supervision is well demonstrated, highlighting its potential for uncovering hidden cellular states or transitions from high-dimensional, noisy biological measurements.

- The evaluation spans from controlled synthetic systems (bistable, predator–prey, Lorenz) to biological switching processes and real single-cell data, which demonstrates the model’s versatility and robustness.

### Weaknesses
- Figure interpretation and clarity. It is unclear whether the x-axis in Figure 1 represents time. The visualization appears to suggest that blue cells evolve into red and green cells over time. In the overlapped region (left panel), do the red and green cells physically interact, or does their spatial overlap simply obscure individual dynamics? I would appreciate further clarification on how overlapping dynamical regimes introduce modeling challenges and whether this overlap is a visualization artifact or a true physical mixture.

- Is the research problem novel? The general problem of learning systems governed by multiple, switching, and partially unknown dynamics has been extensively studied in the literature (e.g., Graph Switching Dynamical Systems, ICML 2023). It is not fully clear what unique challenge this paper addresses beyond existing frameworks for switching or hybrid dynamical systems.

- Is MODE technically novel? The MODE framework appears conceptually similar to standard mixture-of-experts models with sparse MAP estimation. The objective in Equation (3) essentially corresponds to a weighted MAP formulation, and the comparison baselines (GMM, supervised MLP, NODE) are arguably too weak or mismatched for a fair assessment. It would strengthen the paper to demonstrate MODE’s advantage over stronger baselines explicitly designed for switching or compositional dynamics.

- Motivational gap regarding branching systems. The paper motivates MODE by emphasizing that biological systems may bifurcate into multiple branches. However, it is not clear why existing models could not be independently trained for each branch (e.g., fitting separate NODEs for red and green trajectories). If MODE’s advantage lies in discovering branches without supervision, this distinction should be emphasized more clearly.

- How do we know how many experts will be required? The number of experts (K) seems to correspond to the number of distinct dynamical regimes, which in practice may require prior domain knowledge. It remains unclear how MODE performs when K is misspecified or when the true number of regimes is unknown, which could be an important consideration for real-world biological applications.

- In Line 185, the authors assume that each sample is governed by one expert, implying no interaction across regimes. If that assumption holds, why not simply learn from data after regime transitions are completed, rather than during overlap? Clarifying this design choice would help justify the need for mixture modeling during ambiguous transitions.

- Equation (4) introduces pi_s(x) as the expert assignment distribution, but it is unclear how this distribution behaves in practice. Are the mixture weights highly non-uniform across regimes, and how sensitive are results to the entropy or balancing regularizers?

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a new method called MODE (Mixture of Dynamical Experts) to address the challenge of modeling complex systems, particularly in computational biology. It focuses on the difficulties posed by sparse and unordered `snapshot’ data that often exhibit multiple overlapping behavioral regimes and branching dynamics. MODE tackles this gap by using a gated mixture model that decomposes complex, ambiguous flows into distinct dynamical components. The authors demonstrated that MODE outperforms existing approaches in unsupervised classification of dynamical populations under noise and limited sample sizes, using both synthetic and real-world single-cell RNA sequencing data.

### Strengths
1) The paper abstract, introduction and related work are written very well and motivate their method.
2) They discussed and prioritized method interpretability which is crucial for scientific applications.
3) They applied the method to both real world and synthetic data.
4) They explicitly explained the distributions and the assumptions about (most) variables.

### Weaknesses
1) The method itself should be more detailed, i.e. you state the model, but the fitting procedure / algorithm could be more clearly explained.
2) Model-wise, from what I understand, it seems like the authors had an implicit assumption that each snapshot reflects a single state (i.e., comes from one most likely expert). I would assume that during its cycle, a cell's evolution may be governed by multiple experts simultaneously, e.g., experts that capture division-related signals as well as growth signals that may reflect different dynamics and can be combined for a more flexible expert.
3) While I recognize that this paper's main goal is to describe the method, I miss a short discussion about what the method tells us about the biology (rather than only e.g. predictions). E.g. How much in advance in time can you predict future differentiation? 
4) In the related work, there is a missing discussion on decomposed dynamical system models [1,2] and their extension to long term forecasting [3].
4) With respect to figure 4, you talk about SINDy but I cannot see it presented.
5) The method assumes that the basis functions that define the dynamics (Z) are known and are polynomials. It is built on setting these a priori, which requires choosing/knowing the appropriate polynomial basis.
6) Can you clarify what is B? (unless I missed it, I cannot find where it is defined)
7) Small typo: title of section 3 should be method, not methods.


[1] Mudrik, N., et al. (2024). Decomposed linear dynamical systems (dlds) for learning the latent components of neural dynamics. JMLR

[2] Chen, Y., et al. (2024). Probabilistic decomposed linear dynamical systems for robust discovery of latent neural dynamics. NeurIPS

[3] Mudrik, N., et al. (2024). LINOCS: Lookahead Inference of Networked Operators for Continuous Stability. TMLR

### Questions
1) How would you model a case where each snapshot evolves by the rules of multiple co-active experts that govern its dynamics?
2) Did you try to look at the 2nd or 3rd most likely expert in every time point? Maybe you will reveal differences within the apparently the same cell state across snapshots that can further reveal their fate even earlier in time?
3) How much does the degree of the polynomials affect the result? Is there a way to fit / infer the basis Z rather than setting it as a hyperparameter?
4) What can the method tell us about the biology / biological processes in future datasets that existing methods cannot?

### Soundness
3

### Presentation
3

### Contribution
2
