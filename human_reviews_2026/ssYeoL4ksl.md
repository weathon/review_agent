# Identification of Causal Relationships in Linear Cyclic Models with Latent Variables

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 8, 4, 4

## Abstract
Causal discovery aims to model the intricate mechanisms underlying complex systems. Numerous methods have been proposed to identify causal relationships from observational data, but they often assume that the causal model is acyclic and all variables are observed. Those methods risk yielding misleading or spurious causal relationships when confronted with the challenges posed by cycles and latent variables. To address these challenges, we propose a novel method that leverages higher-order cumulants to recover the causal structure among observed variables, even in the presence of cycles and latent variables. Specifically, we construct two cumulant matrices that incorporate various (joint) cumulants of the observed variables. By utilizing these matrices, we provide identifiability theories that determine the existence of cycles and latent variables based on the rank differences of the constructed cumulant matrix, and determine the causal relationship between two observed variables. This innovative method provides a robust framework for accurate causal discovery in complex systems with inherent cyclic and latent structures. Experimental results in simulated and real-world data demonstrate the effectiveness of our proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a causal discovery method utilizing higher-order statistical features that can simultaneously identify no edge, unidirectional causality, and feedback loops between two variables using only observational data, while accounting for potential confounding factors; it outperforms multiple baselines on both simulated and real-world data.

### Strengths
- The research problem is important and practically significant. By uniformly handling feedback loops and latent confounding, it is applicable to complex systems in biology, economics, and other domains.
- Through constructing J(k1,k2) and M(k1,k2), the paper provides identifiability-related theory based on rank criteria, distinguishing multiple scenarios including no edge, unidirectional causality, and cycles.
- The experiments are well-replicated, covering diverse scenarios including acyclic/cyclic/mixed cases and various non-Gaussian noise distributions.

### Weaknesses
- Currently serves as a pairwise variable identifier; no extension strategy from local to full graph, conflict resolution mechanism, or complexity analysis is provided.
- The theorems require k to grow with l, and experiments show that large sample sizes are needed for stable high accuracy; uncertainty quantification or robustness explanations are lacking for real-world small-sample cases.
- The paper's assumptions are quite strict; analysis of robustness and sensitivity to near-Gaussian noise, nonlinearity, and near-unstable systems is absent.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates how to infer the causal relationship between two variables while accounting for potential cycles and a known number of hidden confounders, under the assumption of a linear non-Gaussian model. The main idea is to exploit patterns in higher-order cumulants to infer causal directionality. The authors begin with an intuitive explanation for the case of a single hidden confounder: if the joint cumulant is identical to that of X but not to Y then X->Y, whereas if the joint cumulant matches both X and of Y then X<->Y. Then the authors extend this to the case with l hidden confounding.

### Strengths
New important theoretical contribution for causal discovery.

The authors validated their results on simulated data and applied it on real data.

### Weaknesses
The only limitation I can identify is that the paper focuses exclusively on the bivariate case. However, this seems reasonable given the complexity of the problem: starting with a simple setting is a natural first step. Hopefully, these results will pave the way toward a multivariate extension in future work.

### Questions
* It is not entirely clear to me how you distinguish between the case where X causes Y and Y causes X (with a hidden confounder) and the case where neither X nor Y causes the other, but there is a single hidden confounder affecting both. Could you please provide some intuition on how these two situations are differentiated? (In other words, how do you distinguish between Figure 1a and Figure 1e?)

* Can you please describe the contribution of the paper with respect to other known causal discovery algorithm like [1] [2] in the cyclic setting ?

* The arXiv version of [3] is cited in the paper. Can you please cite the UAI version instead? 

* As far as know the CCD algorithm was introduced in [4] not in [5]. The algorithm introduced in [5] is LING-D.



References:

[1] Jin, Ni, Spence, Rubin, Xu. Directed Cyclic Graphs for Simultaneous Discovery of Time-Lagged and Instantaneous Causality from Longitudinal Data Using Instrumental Variables. JMLR, 2025


[2] Mooij, Claassen. Constraint-Based Causal Discovery using Partial Ancestral Graphs in the presence of Cycles. UAI, 2020.

[3] Joris Mooij and Tom Heskes. Cyclic causal discovery from continuous equilibrium data. UAI. 2013.

[4] Richardson. A Discovery Algorithm for Directed Cyclic Graphs. UAI. 1996

[5] Lacerda, Spirtes, Ramsey, Hoyer. Discovering Cyclic Causal Models by Independent Components Analysis. UAI. 2008

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a novel algorithm for causal discovery in the presence of cycles and latent confounders, based on cumulant–matrix rank analysis.

It provides theoretical guarantees for identifying direct edges, their directions, and the existence of cycles, with confounding.

Empirically, the method outperforms baselines in four two-variable settings (with and without confounders), and it includes a real-world case study.

### Strengths
1. The paper is well written and presents both theoretical guarantees and empirical demonstrations, including a real-world case study.

2. It tackles a challenging setting for causal discovery where both latent confounders and cycles may be present.

3. Across four two-variable scenarios (with/without a confounder), the proposed algorithm achieves competitive or superior accuracy.

### Weaknesses
1. Although the method iteratively increases $m,k_1,k_2$, there is no theoretical guarantee under misspecified orders. Ideally, none of the three rank scenarios should trigger when the orders are wrong; if they still do, that would be problematic. The paper does not clarify whether this can occur. 

1.1 Empirically, the iterative procedure appears favorable in the reported experiments because the initialization matches the ground truth; consequently, when this holds, no substantive iteration occurs. Please first verify this by reporting the order values at termination for the conducted experiments. Please then further provide evidence that the method still works when the initial values are misspecified.

2. The proposed algorithm focuses on pairwise causal relations, leaving its applicability to general causal graphs unclear. Are there fundamental challenges that prevent extending it to a unified algorithm for arbitrary graphs?

3. Following the above, both the synthetic and real case studies are limited to two-variable settings, raising concerns about scalability. Additionally, the reported experiments also restrict the number of latent variables to one.

4. It would be helpful to report computational cost and running time.

Minor comments:

1.  Some symbols in Section 2 are used without first being introduced. For example, $\alpha,\beta$,  $X_a$ and $n$. Please define them upon first use or introduce them right after the first use.

2. When stating, “Although some methods can simultaneously address cycles and latent variables, they are constraint-based and have limitations in identifying certain causal edges” (line 68), please add the corresponding citations.

### Questions
1. From the experimental results, the CCI baseline appears sensitive to the noise distribution, which is reasonable given its reliance on conditional independence tests. What explains the pattern where the Gamma distribution sometimes works (Cases 1–2) but fails in Cases 3–4, while the other two distributions show the opposite behavior? If this is driven by test assumptions (as noted in the paper), one would expect consistency across all cases. Could you clarify this?

2. Empirically, is $n=5000$ the lower bound for effective sample size? With only two variables, 5000 observations seem relatively large compared to what other causal discovery methods typically require.

3. It stated: “This binary accuracy measure is assessed at significance levels ranging from $\alpha = 0.2$ to $\alpha = 0.01$." What does varying the significance level mean in this context? Additionally, how were the baseline parameters selected or tuned?

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
The paper studies identifying causal relations between two observed variables under linear non-Gaussian models that may contain cycles and latent confounders. The key idea is to build two higher-order cumulant matrices, $J^{k_1,k_2}$ (from joint cumulants) and $M^{k_1,k_2}$ (augmenting with marginal cumulants), and then use their ranks to decide: (i) whether there is an edge, (ii) the edge’s direction, and (iii) whether there is a cycle. Under the standard stability condition for cyclic SEMs and non-Gaussian independent noises, the paper proves rank-based necessary and sufficient criteria (Theorems 1–3) and summarizes identifiability (Theorem 4). An algorithm then iterates over $k_1,k_2$ and an estimated number of shared latents to classify the $X,Y$ relation. Experiments on synthetic cases (no edge/ single edge/cycle, with or without a latent) and a small psychological dataset illustrate the approach, with comparisons to DirectLiNGAM, a cumulant-based acyclic method, ReLiNGAM, and CCI.

### Strengths
- Unified handling of cycles and latents from observational data in the bivariate case. The same rank framework covers “no edge,” “directed edge,” and “cycle”.
- The paper explains how marginal vs joint higher-order cumulants share (or don’t share) noise terms across structures, which motivates the rank criteria.

### Weaknesses
- All theorems target a pair of observables at a time. There is no procedure (or guarantee) for constructing a globally consistent multivariate graph from pairwise decisions.

- High-order cumulants are statistically noisy; the method needs large sample sizes before ranks stabilize. The paper gives little guidance on regularization/thresholding for near-rank decisions.

- The manuscript needs a careful language edit.

### Questions
- How do you set numerical thresholds for deciding ranks of $J$ and $M$ from noisy cumulant estimates? Any bootstrap, or shrinkage scheme to stabilize near-boundary cases?

- The algorithm increases $m$ and adjusts $(k_1,k_2)$ (e.g., start with $(3,4)$, then $k_1 = m+2$, $k_2 = 2m+2$). Why this schedule, and how sensitive is performance to alternate choices (e.g., using multiple $(k_1,k_2)$ and aggregating decisions)? Please clarify lines 4-26 of Algorithm 1.

- For $p>2$ observables, how do you ensure a globally consistent graph when pairwise decisions disagree?

Minor comment: Please fix typos (“identifiablity,” “acylic,” “RCREPRODUCIBILITY,” etc.)

### Soundness
3

### Presentation
2

### Contribution
2
