# Automatic and Structure-Aware Sparsification of Hybrid Neural ODEs with Application to Glucose Prediction

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Hybrid neural ordinary differential equations (neural ODEs) integrate mechanistic models with neural ODEs, offering strong inductive bias and flexibility, and are particularly advantageous in data-scarce healthcare settings. However, excessive latent states and interactions from mechanistic models can lead to training inefficiency and over-fitting, limiting practical effectiveness of hybrid neural ODEs. In response, we propose a new hybrid pipeline for automatic state selection and structure optimization in mechanistic neural ODEs, combining domain-informed graph modifications with data-driven regularization to sparsify the model for improving predictive performance and stability while retaining mechanistic plausibility. Experiments on synthetic and real-world data show improved predictive performance and robustness with desired sparsity, establishing an effective solution for hybrid model reduction in healthcare applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents HGS, a method for simplifying the structure of mechanistic neural ODEs. The authors point out that the mechanistic models often used may be too complex for the small datasets we typically have in fields like healthcare, which can lead to overfitting. Their proposed solution includes three steps: first to modify the model's graph structure based on its topology by collapsing cycles and adding shortcuts, and then to use regularization to learn which connections to prune. The method is evaluated on synthetic data and a real world glucose forecasting task.

### Strengths
1. The work is well motivated. Trying to bridge the gap between complex mechanistic models built by experts and data-driven methods is a critical research area, and the paper does a good job of framing the problem.
2. The idea to first perform a structural simplification of the graph before applying a data-driven pruning is quite clever. It's a nice way to inject some domain agnostic heuristics to constrain the learning problem.
3. The set of experiments seems comprehensive. The authors have benchmarked their method against a good range of strong baselines, and the ablation study clearly shows that each part of their pipeline contributes to the final result.

### Weaknesses
My main concerns are with the real-world application, and I would need the authors to address these points before I could reconsider my score.

1. My primary concern is how the model handles patient variability. The T1DEXI dataset includes 105 different people, and it's a physiological fact that glucose dynamics differ significantly between individuals. From my reading of Equation 1, the MNODE learns a single set of dynamics for everyone, i.e., an "average patient" model. This seems to sidestep an important challenge in this domain. Could you clarify if this is indeed the case, and if so, explain the rationale behind this choice? To be clear, this has been discussed extensively in recent years ([1-4] for example).
2. The paper mentions that the cross validation splits were created from a "random permutation" of the 342 time series. To avoid data leakage, it's standard practice in clinical ML to split data at the patient level (i.e., all data from a single person stays in one fold). Could you confirm whether your evaluation followed this practice? If not, the reported performance might not accurately reflect generalization to new, unseen patients.
3. The paper uses the term MNODE, but it seems that the functional forms of the UVA-Padova model are discarded, with only the graph structure being retained. The actual dynamics are then learned by MLPs. This is a very weak form of mechanistic prior. It would be helpful to discuss the trade-offs here.
4. It's worth noting that in the synthetic experiments, the TCN model starts to outperform HGS on the RMSE metric as the sample size grows to N=1000. While HGS remains more robust in terms of Peak RMSE, this suggests that its strong inductive bias might become a disadvantage when more data is available. Could you comment on this trade-off?

[1] Generative ODE Modeling with Known Unknowns

[2] Physics-Integrated Variational Autoencoders for Robust and Interpretable Generative Modeling

[3] Learning Physics Constrained Dynamics Using Autoencoders

[4] CONFIDE: Contextual Finite Difference Modelling of PDEs

### Questions
Please address the weaknesses above.

### Soundness
1

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
3

### Summary
The authors propose a sparsification method for Neural ODEs parameterized by DAGs / relaxed DAGs, and benchmark on predictive accuracy for held-out samples from time-series data.

### Strengths
The authors get good results on their synthetic and empirical benchmarks, and their graph reduction seems like a reasonable solution to the problem.  The comparison to other graph reduction methods makes the argument in favor of the current setup more convincing.

### Weaknesses
I’m a little confused by the motivation of the setup.  In most instances I’m familiar with, the incentive for parameterizing a causal model or a dynamical system with a graph is interpretability, but the authors insist that the setup isn’t identifiable and the purpose is only for predicting future states.  In that case, what is the rationale for insisting on DAG / RDAG structure at all?  One could still encode prior knowledge by constraining dependence of one state variable on others with masking.

The third part of the model seems to just be the inclusion of L1/L2 regularization, which I think is straightforward enough to not be a novel portion of the proposed method.

The highly varying y-axes across graphs makes it much much harder to gauge which performance improvements are substantial and which ones are almost negligible.  I would suggest the authors use tables in some cases so a reader can more easily tell.  For example, many bar plots in Figure 2b are somewhat misleading because the scale is so small.

### Questions
To my understanding, only the MNODE based methods are given the prior graph determining the underlying dynamics?  In that case it seems the comparisons to the black box NODE models are somewhat uninformative, without encoding that prior into the model (even as a form of regularization).

### Soundness
3

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
The paper targets an application of Hybrid ODE models in healthcare. Hybrid neural ODEs combine neural ODEs with mechanistic models to incorporate domain relational knowledge. However, mechanistic models in physiology and medicin tend to become excessively large when capturing wide ranging dynamics.  These models may contain  many latent states for a handful of observable states, and interactions among states can lead to inefficiency and overfitting. Model reduction techniques often require domain knowledge. Data-driven reduction techniques, on the other hand, are completely devoid of any domain knowledge and do not preserve mechanistic structure or constraints.

The paper proposes a method for automatic state selection and structure optimization in mechanistic neural ODEs by combining graph modification based on domain knowledge with data-driven regularization that improve predictive performance and stability while remaining mechanistically plausible. The approach combines domain-knowledge informed graph modification with a mix of L1 and L2 regularization . The graph modification bases itself on classical reduction methods to retain key topological structures. While the regularization step allows gradient based pruning during training, making the processes efficient.

The architecture of the model is an encoder-decoder on where the encoder takes in history and produces an initial state. The decoder takes the initial state and graph representation of the ODE system and future inputs evolves the state features as a Neural ODE.

The actual reduction algorithm has three steps. Step 1 merges maximally strongly connected components into supernodes with self loops. Step 2 augments the graph with short-cuts using partial transitive closure, and step 3 applies a data-driven mixture of L1 and L2 regularization.

Experiments are performed on healthcare data showing improved prediction and robustness with desired sparsity.

### Strengths
The paper is very detailed regarding the method and well organized.

The method combines classical and data-driven approaches for the state reduction and sparsification problem.

Experiments are done both on synthetic and real data. Ablation experiments are done for all the steps of the method.

The paper targets biomedical domains where methods must work with complex models and limited data and interpretability is significant.

### Weaknesses
Rationale of step 1 says that transforming the original graph into and RDAG reveals essential causal structure. What is the evidence for this? It also says that this improves training stability. Has this been shown in the experiments? What is the reason for this improvement in stability?

How do self-loops allow more flexible modeling of intra-component dynamics? It is said to be a key principle of hybrid modeling (line 190), but I am unsure how this is established.

The formulation of step 2 is too formal which leads to a decrease in clarity of the paper. It would be useful to start from intuition and build from there. For instance, it would be useful to first describe what is the intuitive description of a partial closure.

The introduction to the paper claims that the method allows incorporating domain knowledge reduction of the graph. As far as I can tell in steps 1 and 2, the graph reduction methods are more structural (not data-driven) rather than incorporating domain knowledge.

Typos etc.

line 167. “maximally strongly” should be maximal strongly …
There appears to be a formatting problem where line spacing has changed starting from page 5.

### Questions
In the motivation of the method the paper states that the method allows domain-knowledge informed graph modification. I don’t follow how the method allows this in the steps of the method. How is this domain-knowledge used in the experiments?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses a key bottleneck in hybrid neural ODEs (i.e., over-parameterization and over-fitting arising from combination of mechanistic model complexity, addition of data-driven components and sparse data) by proposing an automatic structure-learning pipeline. The method combines domain-informed graph modifications with data-driven regularization to identify essential states and interactions in mechanistic neural ODEs, thereby improving training efficiency, prediction accuracy, and robustness. The authors evaluate performance on synthetic and a real healthcare dataset, showing clear improvements over existing baselines while yielding sparser and more interpretable models. Overall, this is a strong paper. The problem is relevant and timely, the methodology is technically sound, and the experiments convincingly support the claims.

### Strengths
Originality: The paper addresses an important challenge in hybrid neural ODEs: automatic reduction of mechanistic state space and structure. The authors propose a creative combination of domain-guided graph pruning and regularization-driven sparsification, contributing to the emerging literature on structure-learning for scientific ML. Although I must admit I am not an expert in graph-based techniques so I am not able to assess the novelty of these claims, but the authors acknowledge a large body of literature.

Quality: The authors present a solid methodological formulation with thoughtful integration of mechanistic priors and data-driven sparsity and strong empirical validation across both synthetic and real healthcare datasets, confirming generality. Moreover, they perform a comprehensive comparison against state-of-the-art baselines.

Clarity: The problem is well-motivated, and writing is generally clear and well-structured. The authors make clear statements about what this method is and what it is not (e.g., does not guarantee true model discovery). Overall, the model architecture and training procedures are communicated effectively, multiple repetitions are performed for all results to confirm robustness and claims are aligned with results.

Significance: This paper positions hybrid modeling for more deployment-ready use in data-scarce healthcare contexts, which is a domain where interpretability and robustness are essential. In addition, automated model reduction has broad relevance beyond the presented domain and opens a promising direction for robust structure learning in hybrid models.

### Weaknesses
Figures readability: Many figures contain very small titles/legends/axis labels, making interpretation difficult. Increasing font sizes and improving layout will improve accessibility and impact.

Interpretation depth: While results are strong, the text describing figures and drawing insights is brief. Slightly reducing methodological narrative to provide richer interpretation would improve readability and scientific impact.

Limited discussion of learned functional forms: The paper does not clearly articulate whether the recoverable mechanistic relationships must be linear w.r.t. parameters (as in LASSO-type sparsity) or whether the framework can support nonlinear mechanistic hypotheses. Clarifying this assumption and its implications for general healthcare systems would strengthen the positioning.

Computational considerations not fully reported: Training time, scalability to larger mechanistic graphs, and computational savings from sparsification could be discussed more concretely in Appendix. Would this be a concern in the practical implementation and use in the types of healtcare applications of interest?

### Questions
Model class expressiveness: Does the proposed sparsification framework assume linearity in the mechanistic model parameters? If so, how does this limit applicability to nonlinear systems common in healtchare and other fields?

Generalization to larger mechanistic graphs: How does performance scale with increasing state dimensionality or interaction complexity? Any expected limitations?

Practical deployment: What guidance can you provide for practitioners on choosing sparsity regularization strengths and graph-modification thresholds?

Training efficiency: Can you quantify computational savings (e.g., reduced training time, memory use) achieved through the learned sparsity?

Interpretability: Are there examples where the discovered reduced model aligns with clinical or mechanistic knowledge? If so, including brief commentary would strengthen the healthcare relevance.

### Soundness
4

### Presentation
3

### Contribution
4
