# Explainable Evidential Clustering

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Unsupervised classification is a core problem in machine learning. Because real-world data are often imperfect, non-additive frameworks, such as evidential clustering, grounded in Dempster-Shafer theory, explicitly handle uncertainty and imprecision. These frameworks are particularly well suited to high-stakes decisions, which tend to require both interpretability and cautiousness. However, while decision-tree surrogates have enabled transparent explanations for hard clustering, explainability for evidential clustering remains largely unexplored. We address this gap by formalizing representativeness, a utility-based criterion that captures decision-makers' preferences over explanation misassignments, and introducing evidential mistakeness, a loss function tailored to credal partitions. Building on these foundations, we propose the Iterative Evidential Mistakeness Minimization (IEMM) algorithm, which learns decision-tree explainers for evidential clustering by optimizing representativeness under uncertainty and imprecision. We provide theoretical conditions for effective explanations in both hard and evidential settings and show how utility parameters can be set to reflect different decision attitudes. Experiments on synthetic and real-world datasets demonstrate that IEMM improves the performance of existing methods by producing representative and preference-aligned explanations of evidential clusterings, supporting cautious, transparent analysis in the presence of imperfect data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Explainable Evidential Clustering, a framework that extends decision-tree–based explanations to clustering models that account for uncertainty. The authors formalise two new theoretical constructs: Evidential Representativeness, a utility-based criterion that quantifies how well an explanation reflects decision-maker preferences, and Evidential Mistakeness, a loss function capturing representativeness errors in evidential settings. The paper provides theoretical analysis, algorithmic formulation, and experiments on small synthetic and real-world datasets.

### Strengths
The paper addresses an underexplored problem on explainability for evidential clustering, where model outputs include uncertainty and imprecision. Its main originality lies in adapting the decision-tree explainer paradigm to this evidential setting and introducing formal definitions. 

The methodological quality is solid in its theoretical formulation, with proofs, definitions, and algorithmic description. The paper is well written and structured.

In terms of significance, the paper’s contribution is relevant for research on explainability under uncertainty and could serve as a useful foundation for future work in evidential learning. However, its empirical and comparative evaluation is limited, relying on small datasets and a single older baseline, which restricts its practical and experimental impact. Overall, the work is conceptually original and clearly presented, but its experimental and empirical depth remains a key weakness.

### Weaknesses
The paper’s main limitation lies in its narrow and weak empirical validation. All experiments are conducted on small, classical datasets (e.g., Iris, Wine, Diabetes, and simple credal datasets), which do not convincingly demonstrate scalability or generalizability. The evaluation includes only one baseline (IMM, 2020), making it difficult to assess whether the proposed method actually advances the state of the art. 

The methodological novelty is also limited since IEMM is largely an incremental adaptation of IMM to an evidential setting. 

In terms of clarity and framing, while the theoretical exposition is rigorous, the paper occasionally leans too heavily on formalism. It does not provide concrete real-world examples where evidential clustering explanations would offer tangible benefits over simpler alternatives. The related work section is not critical or up-to-date, omitting recent methods for explanations for clustering models.

### Questions
1. The related work overview lists some studies but does not critically analyse methodological differences or advances since 2020. Could the authors expand the discussion to include more recent developments (2021–2025) in explainable clustering and evidential learning?
2. Why is the experimental comparison limited to IMM (2020)? Have you considered including other, more recent interpretable clustering baselines?
3. The paper motivates cautiousness in high-stakes domains (e.g., healthcare), but the evaluation does not include such real-world data. Can the authors provide a realistic use case demonstrating IEMM’s interpretability or decision support value?
4. The paper evaluates explanation quality only through the proposed metrics, which measures fidelity to the evidential clustering under a chosen utility. It is not clear why no human-centred or standard explanation quality measures were included. How can one ensure that higher representativeness actually corresponds to more understandable or useful explanations for end users?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method to generate interpretable, decision-tree-based explanations for evidential clustering. This is a clustering paradigm that models uncertainty and imprecision using Dempster-Shafer theory. The authors introduce new utility-based notions of representativeness and evidential mistakeness, and design the Iterative Evidential Mistakeness Minimization (IEMM) algorithm to greedily optimize decision trees for explaining credal (i.e., evidential) partitions. Theoretical analysis generalizes previous results for hard clustering to the evidential setting and provides utility-based frameworks reflecting different decision-maker attitudes towards caution and error. Experimental validation on synthetic and real-world datasets demonstrates that IEMM delivers more utility-aligned, representative explanations than adapted baselines.

### Strengths
1. The paper tackles the largely unexplored challenge of explainability for evidential clustering, a problem that is of interest in high-stakes and risk-averse domains.

2. It reconstructs IMM’s logic for hard cluster explainers, formalizes representativeness in the evidential regime, and proves that minimizing evidential mistakeness yields maximal representativeness under a stakeholder utility.

3. The use of a stakeholder-specific utility to mediate caution versus specificity is conceptually and practically meaningful.

### Weaknesses
1. The DSClustering paper (Hovhannisyan, 2025) recently proposed a system that also leverages Dempster–Shafer theory to generate interpretable, rule-based cluster descriptions and to communicate uncertainty to end users. Given this development, the authors of the current submission risk slightly overstating the claim that “no one has addressed interpretability in evidential clustering.” While their approach remains distinct, the paper should avoid implying exclusivity in combining Dempster–Shafer reasoning with interpretability. Instead, the authors should explicitly acknowledge DSClustering as a concurrent but methodologically different effort, emphasizing that their work addresses the post-hoc explanation of existing evidential clusterings.

2. The experiments rely only on small, low-dimensional tabular datasets (Iris, Wine, Diabetes). The absence of high-dimensional, noisy, or real-world examples (e.g., medical or industrial data) limits the demonstration of scalability and generality. Similarly, the baseline (IMM with collapsed labels) cannot express ambiguous metaclusters, making IEMM’s superiority somewhat tautological under the chosen utility. Including stronger baselines, such as CART-style trees trained to maximize expected utility directly, or rule-based interpretable evidential clustering methods, would provide a more compelling comparison.

3. The introduction of a stakeholder-specific utility function U(A,B) is conceptually strong and central to the paper’s stakeholder-aligned framing. However, the paper does not specify how this utility is to be elicited, parameterized, or grounded in user preferences. In practice, different stakeholders, such as clinicians, regulators, or engineers, may have distinct preferences and tolerance levels for ambiguity or misclassification.

4. The exposition in Sections 3–4 is dense and symbol-heavy, which obscures the paper’s otherwise logically theoretical structure. Definitions of representativeness and evidential mistakeness are presented abstractly before any motivating examples, forcing readers to decode complex notation (mass functions, focal sets, utilities) without an intuitive grounding. This could be improved by introducing a running toy example early in the section, visually showing how cautious explanations evolve with different parameters before formal derivations. The paper is overall good and will consider improving the score after clarifications in the rebuttal.

Reference:

Interpretable Clustering Using Dempster–Shafer Theory, Hovhannisyan 2025.

### Questions
1. How large can the focal set family |F| become in practice, and do you prune focal sets before running IEMM?

2. Are shallow axis-aligned trees essential for your theoretical guarantees, or could oblique or deeper trees achieve higher representativeness without sacrificing interpretability?

3. How do you envision practical elicitation of the stakeholder utility  𝑈(𝐴, 𝐵)?

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
4

### Summary
The paper presents an approach for explaining evidential clustering by training a surrogate decision tree whose leaves are labeled with the 
focal elements of the credal partition. To construct the decision tree, the IMM algorithm used for explaining centroid-based methods is 
appropriately extended and the Iterative Evidential Mistakeness Minimization (IEMM) method is proposed and evaluated.

### Strengths
S1. It seems to be the first approach to explain evidential clustering.
S2. Several novel notions and definitions are included (e.g. cautious explainer)
S3. The proposed IMM extension is well-formulated.

### Weaknesses
W1. Evidential clustering has not been widely accepted, especially in real applications 
(compared for example to fuzzy or probabilistic clustering methods). Hence an explainer specialized to that framework has limited reach.
W2. If the number of focal elements is large, it seems difficult to interpret the results.

### Questions
Q1. Are there approaches that build decision trees to explain fuzzy or probabilistic clustering solutions 
taking into account uncertainty in cluster membership? If yes, they should be included in the comparison.  
Q2. It would be nice to present the trees obtained for the real datasets considered.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
Explainable clustering is a problem that has received considerable theoretical attention in recent years. The setup is the following: Given a dataset and its (hard) cluster assignments, can we find an axis-aligned partition of the data that has competitive performance to the original clustering. This line work started with the IMM algorithm (Moshkovitz et al, 2020), and there has been multiple extensions of the problem setting and methods, but the main focus of this line of work has been on the theoretical guarantees (the so-called price of explainability, and its variations).

The present paper generalises the problem and IMM algorithms to the setting of evidential clustering. In evidential clustering, each data instance is mapped to a probability mass function over the power set of all clusters. This generalises hard clustering, Bayesian clustering (where only singleton clusters have non-zero probabilities), and categorical clustering (where each data is mapped to a single subset of clusters), as illustrated in Figure 5.

In explainable evidential clustering (as posed in the present work), one is given a data set and corresponding mass functions and the goal is to find axis-aligned decision tree based partition of the data such that each data is mapped to a subset of clusters. The authors propose a generalised notion of mistakes, using the mass functions, and extend the IMM algorithm of Moshkovitz et al to obtain an IEMM algorithm, which they empirically evaluate on Gaussian mixtures and few small UCI datasets.

### Strengths
The evidential clustering framework is quite interesting, which I believe already provides more interpretability than hard clustering (since the mass function encodes relevant uncertainty in the clusters). The problem of fitting axis-aligned decision trees provides an additional layer of explainability to the problem. 

The problem of explainable evidential clustering is also mathematically interesting (at least to those working on explainable clustering) because the criteria from clustering changes from k-means/k-median cost to a more complex setting, where the uncertainty information/mass functions are available.

### Weaknesses
The main drawback of the paper is that the problem is not well-formulated. A few important missing pieces are noted below
- The paper proposes IEMM as an approach for explainable evidential clustering without precisely stating the problem (that is, what do we want to minimise while ensuring explainability). One can contrast this with IMM and the corresponding line of work, where the explainable k-means clustering is posed as a problem of achieving low k-means cost while ensuring explainability. As a result IEMM is a heuristic that does not precisely minimise a well-defined loss. This is of the same flavour as hierarchical clustering heuristics, which did not have any sound theoretical basis until Dasgupta's seminal works in 2010s.
- A natural consequence of above is that there are no theoretical guarantees for the proposed IEMM algorithm, which is in contrast to all existing works on explainable clustering that I am aware of. The appendix includes some theoretical results on representativeness, but they do not provide any formal guarantees on the performance of IEMM.
- More fundamentally, the IEMM takes mass functions as input for each data point, but returns only a subset of clusters and not a mass function. Hence, there is loss of information after one approximates the evidential clusters for decision trees (in other words, this method does not seem relevant for explainable Bayesian clustering).
- The experiments are too limited and focus only on simple data, where explainability does not have any consequence. I would be fine with limited experiments if there was strong theoretical contribution. In particular, the explainable clustering literature has been primarily of theoretical interest (with limited use in practice). However, without a strong theory, the paper needs to demonstrate the practical impact of such an approach, where explaining clusters with axis-aligned trees would matter.
- The overall presentation is quite poor, and not written for a general machine learning audience. A large number of concepts are introduced, but not used much in the paper. This makes the paper quite hard to read. I believe the presentation can be significantly simplified by making it more direct. For example, I feel the notion of representativeness is presented in a quite complicated way, making it too formal to even understand how it is useful. The same can be said about the setup, where one can get a better idea only after reading the appendix.

### Questions
There are no specific questions. This paper needs to be significantly reworked and rewritten. Please see weaknesses

### Soundness
2

### Presentation
1

### Contribution
2
