# InvarGC: Invariant Granger Causality for Heterogeneous Interventional Time Series under Latent Confounding

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Granger causality is widely used for causal structure discovery in complex systems from multivariate time series data. Traditional Granger causality tests based on linear models often fail to detect even mild non-linear causal relationships. Therefore, numerous recent studies have investigated non-linear Granger causality methods, achieving improved performance. However, these methods often rely on two key assumptions: causal sufficiency and known interventional targets. Causal sufficiency assumes the absence of latent confounders, yet their presence can introduce spurious correlations. Moreover, real-world data typically comprise only time series from multiple environments, without prior knowledge of interventions. It is difficult to distinguish intervened environments from non-intervened ones, and even harder to identify which variables or timesteps are affected. To address these challenges, we propose Invariant Granger Causality (InvarGC), which leverages cross-environment heterogeneity to mitigate the effects of latent confounding and to distinguish intervened from non-intervened environments with edge-level granularity, thereby recovering invariant causal relations. In addition, we establish the identifiability under these conditions. Extensive experiments on both synthetic and real-world datasets demonstrate the competitive performance of our approach compared to state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper develops a model that generalizes nonlinear granger causality to a setup where there are unobserved latent confounders and unknown interventions. The current methods in the field of nonlinear granger causality relies on the assumption of causal sufficiency, and the current methods of causal discovery including latent confounders do not handle unknown interventions. 
The paper introduces the invarGC algorithm along with identifiability theorems, and then test it on synthetic and real world datasets.

### Strengths
* As far as I know this paper is the first to extend Granger Causality to the case where there are nonlinear relations, unobserved latent confounders, and unknown interventions

* This paper gives an algorithm along with an identifiability theorem, an experiment on a toy dataset, and an experiment on an almost real world dataset

* At least equivalent to the tested models for the given task, and better when there are hidden confounders and interventions

### Weaknesses
* Some assumptions seem to be stronger than needed and others weaker than needed.

* Many relevant state of the art methods are excluded from the experimental setup

* In the real world dataset for benchmarking, the confounding factors are artificially introduced. The confounding factors are therefore not natural ones.

### Questions
* Several state-of-the-art methods which are highly relevant in this setting have not been compared to, such as CD-NOD (Huang et al, 20), RegimePCMCI (Saggioro et al, 2020), JPCMCI (Günther et al, 2023) (these methods assume causal sufficiency similarly to GC, Dynotears, ... but unlike them  they can handle multiple regimes). There is also FCI-JCI (Mooij) which can handle multiple regimes as well as hidden confounding. It is true that FCI-JCI was not introduced  directly for time series but can be easily adapted (using the same strategy as varFCI (Malinsky)) to time series while taking into account instantaneous relations. Is there a reason for not including them?

* Assumptions (1-4) are not explicitly introduced as Assumptions but rather as conditions A1,...,A4 within a Theorem, which is confusing. Later in the text they were refer to as Assumption 1, ..., Assumption 4.

* Don’t you also need the faithfulness assumption to rule out deterministic relationships, not just to exclude path cancellations? (In other words, shouldn't you use the formal definition of Faithfulness which exclude both)

* Since you assume no instantaneous relations and all edges are oriented using time, can't you replace faithfulness with adjacency faithfulness? 

Minor:
I’ve always thought of Granger causality as a weaker notion of causality, more about prediction than true causal influence. However, under certain assumptions, doesn’t Granger causality actually correspond to genuine causality? If that’s the case, why do we still use the term Granger causality rather than simply referring to it as causality, given that it is traditionally considered useful mainly for predictive purposes ?

References:

Biwei Huang, Kun Zhang, Jiji Zhang, Joseph Ramsey, Ruben Sanchez-Romero, Clark Glymour, Bernhard Scholkopf.
Causal Discovery from Heterogeneous/Nonstationary Data. JMLR, 2020

Elena Saggioro, Jana de Wiljes, Marlene Kretschmer, Jakob Runge
Reconstructing regime-dependent causal relationships from observational time series. Chaos, 2020


Wiebke Günther, Urmi Ninad, Jakob Runge. Causal Discovery for time series from multiple datasets with latent contexts
UAI, 2023.

Joris M. Mooij, Sara Magliacane, Tom Claassen.
Joint Causal Inference from Multiple Contexts. JMLR, 2020.

Daniel Malinsky, Peter Spirtes.
Causal Structure Learning from Multivariate Time Series in
Settings with Unmeasured Confounding. KDD workshop on causal discovery, 2018.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes an algorithm to recover an invariant Granger causal graph from heterogeneous interventional time-series in the presence of latent confounding and unknown intervention targets. Under specific assumptions, the authors establish identifiability of the Granger causal graph, the subspaces spanned by latent confounders, and edge-level interventions.

Methodologically, the approach combines networks for latent-confounder modeling, intervention identification, and invariant predictor learning, and recovers the graph via next-step prediction augmented with some regularizations to enforce invariance and identifiability.

Empirically, on both synthetic and real-world datasets, the method has performance that is competitive with or superior to strong baselines.

### Strengths
1. The paper tackles an important gap in the literature by proposing an algorithm that recovers an invariant Granger causal graph under latent confounders and unknown interventions.

2. The paper is well written and logically organized.

3. The work provides both theoretical guarantees and empirical validation.

4. The experimental results show impressive performance, often matching or outperforming the baselines.

### Weaknesses
1. The theoretical guarantees are not solid.

1.1 Assumption A4 is vague. Combining the main paper and the appendix, “...interventions are sufficiently diverse to distinguish true causal parents from non-parents.” means that, for $X^j_t\in PA(X^i_{t+1})$, that "there exists at least one environment in which the mechanism of the edge $ j \rightarrow i$ differs from its invariant form.", it also means that for any latent variables connected with the target variable, "there exists an environment in which the mechanism along that direction differs from the invariant one. ".That is, by saying “...interventions are sufficiently diverse to distinguish true causal parents from non-parents,” although the intervention targets are unknown, they must intervene on enough edges so that every parent and latent variable for the target variable is identifiable. This is a very strict assumption.

1.2 With Assumption A3, the latent-to-observed mechanism is invariant. Variables $X^j_t\notin PA(X^i_{t+1})$ induced by such invariant spurious edges could be identified as parents, as there is no edge between $X^j_t$ and $X^i_{t+1}$ and hence such an edge cannot be intervened on, resulting in an invariant spurious edge between $X^j_t$ and $X^i_{t+1}$. I wonder, “by Assumption 4 there exists an environment in which the mechanism along that direction differs from the invariant one,” how this is true for latent variables, as interventions are only for observed variables.

2. Important details about the algorithm are missing.

With finite samples, the hyperparameter $\lambda$, $\alpha$ in equation 12 are important. As claimed in the paper, "For regularization parameters in a standard non-degenerate range, the increase ∆ dominates any penalty saving", what is the practical choice of these hyperparameters? Will the performance be sensitive to these parameters?

3. Scalability and running time are not reported.

It would also be beneficial to clearly list the number of nodes and confounders for each experiment. For instance, what is the number of nodes used for different types of the Causal-Rivers dataset?
 
4. Ablation results need quantitative clarity.

It would be clearer to include quantitative results demonstrating the decrease in, for instance, AUROC and AUPRC without LCMCI or under other misspecifications. The current visualization is not straightforward, and it is not clear whether it reports a single trial or an average performance. For example, in Figures 3 and 4, what is the ground truth, what is the value in each cell, and why are some cells dark blue even though they are not extremely large or small compared with other cells?

### Questions
1. Could you please clarify the meaning of Assumption A4 with a toy example? More generally, what does this assumption require in terms of the number of interventions and the corresponding intervention targets?
2. Are the other baselines also applicable to edge-level interventions? If the setting does not match their assumptions, how does that affect their performance?
3. Is the connection restricted to $Z_t \rightarrow X_{t+1}$ with lag 1? By default, does the confounder connect two variables both at time  $t+1$?
4. Should there be a noise term in equation 1?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper considers the problem of Granger causal discovery from time series data. Specifically, the authors consider the presence of both latent confounding and multi-domain heterogeneity, where the causal relations among variables (i.e., the causal graph) are the same across all environments, but the causal mechanism (functional relations) among observed variables may vary. The authors show that, when the model is linear and the lag size is one (i.e., $X_{t+1}$ does not depend on $X_{1:t-1}$ and $Z_{1:t-1}$ conditioned on $X_t$ and $Z_t$), the true causal graph can be uniquely identified, which can be recovered by minimizing the loss function in the proposed recovery algorithm. The authors evaluate the performance of the algorithm on both synthetic and real datasets.

### Strengths
1. The problem is well-formulated and considers a very realistic and under-studied setting.
2. The authors conduct extensive simulations to demonstrate the effectiveness of the proposed algorithm, especially on real-world datasets.

### Weaknesses
1. The notation is a little bit confusing and hard to follow. For example, the subscript of $W$ includes both numbers and variables. It would be better if it can be unified (say use $W_{0, 1:d}$ instead of $X_{0,X_{t+1}X_t}$).
2. Some of the technical details are not clearly explained, such as mathematical formulation of certain assumptions and technical details (see Q1 and Q4 below). 
3. It seems to me that the identification results considers a much simpler setting than the model described in Section 3.2 (see Q5 and Q6 below).

### Questions
1. Is there a mathematical formulation of "sufficiently diverse" in (A4)?
2. Are there any restrictions on the "minimality" of interventions in Theorem 3? For example, suppose there are three environments and two observed variables, where environment 1 is invariant and the node $x_2$ is intervened in environments 2 and 3. Then there exists another model where environment 2 is invariant and the node $x_2$ is intervened in environment 1 and 3.
3. In line 359, does "two environments remain purely observational" imply that these two environments share exactly the same causal mechanism?
4. In Equation (7), is LCIM a neural network? If yes, how are the parameters optimized?
5. How do the causal effects among latent variables (i.e., W_{Z_{t+1}Z_t}) affect the performance of the recovery algorithm?
6. Do the theoretical results presented in Section 4.3 only hold in linear setting?

### Soundness
3

### Presentation
2

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
This paper proposes InvarGC, a framework for Granger causality discovery in heterogeneous interventional time series subject to latent confounding. InvarGC identifies invariant causal relations by using data heterogeneity across environments, infers latent confounders through a dedicated inference module, and distinguishes intervened from non-intervened environments at the edge level. The authors offer formal identifiability guarantees of the recovered causal graph. The authors also conduct comprehensive experiments with synthetic and real-world datasets to support their claims.

### Strengths
- Tackles a relevant setting: unknown interventions + latent confounding, and the problem setup is clear.
- Identifiability results (graph, latent subspace, edge-level interventions) with explicit assumptions.
- Good experimental results vs. strong baselines across synthetic and real data; sensible ablations on $L$.

### Weaknesses
1. The largest real-world example (TEP) uses 33 variables, and Causal-Rivers uses node subsets. While nontrivial, this leaves open whether InvarGC scales effectively to higher-dimensional (>100 variables), longer sequences, or truly networked time series encountered in domains such as neuroscience, genomics, or industrial process control. No runtime or computational complexity results are reported either.
2. Although the ablation study in Figure 3 analyzes the effect of the number of latent confounders ($L$) and regularization weights, the empirical analysis is somewhat superficial. There is insufficient exploration of how robust the method is to hyperparameter misspecification in practice, especially under lackluster prior knowledge of the true confounder count. It is unclear how challenging tuning becomes as the dataset grows, or whether the method is stable across a realistic hyperparameter grid.
3. Edge-level detection uses an ad-hoc threshold; no uncertainty or sensitivity analysis.

### Questions
1. What are training/runtime and memory costs as $d$ and $T$ grow?
2. Can you provide more details on the choice and parameterization of the non-linear functions $f_i(\cdot)$, $g_{k,i}(\cdot)$, and $h_{k,i}(\cdot)$? Are these always neural networks, and how sensitive are your results to their depth/width or activation choices?
3. Is edge-level intervention detection always threshold-based? Would a probabilistic approach or inclusion of uncertainty quantification improve detection stability, particularly for weak interventions?

### Soundness
3

### Presentation
2

### Contribution
2
