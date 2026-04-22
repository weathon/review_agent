# Score-based Greedy Search for Structure Identification of Partially Observed Causal Models

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
Identifying the structure of a partially observed causal system is essential to various scientific fields. Recent advances have focused on constraint-based causal discovery to solve this problem, and yet in practice these methods often face challenges related to multiple testing and error propagation. These issues could be mitigated by a score-based method and thus it has raised great attention whether there exists a score-based greedy search method that can handle the partially observed scenario. In this work, we propose the first score-based greedy search method for the identification of structure involving latent variables with identifiability guarantees. Specifically, we propose Generalized N Factor Model and establish the global consistency: the true structure including latent variables can be identified up to the Markov equivalence class by using score. We then design Latent variable Greedy Equivalence Search (LGES), a greedy search algorithm for this class of model with well-defined operators, which search very efficiently over the graph space to find the optimal structure. Our experiments on both synthetic and real-life data validate the effectiveness of our method.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes LGES (Latent variable Greedy Equivalence Search), a score-based greedy algorithm to learn the causal structure of partially observed linear SEMs with latent variables. The authors prove global consistency under some assumptions. They design an efficient two-phase search with tailored operators and show empirical results versus constraint-based baselines on synthetic and real data.

### Strengths
Overall, the paper tackles an important gap: scalable score-based discovery for latent-variable models with identifiability guarantees. A greedy score-based method with identifiability guarantees in the presence of latent variables is nontrivial. Linking equality constraints, minimal dimension, and GNFM to MEC identifiability. Strong synthetic performance vs. constraint-based baselines.

### Weaknesses
I am not sure if the conditions, such as GNFM structure, generalized faithfulness, and linear/noise are satisfied.
About experiments, I see that on Big Five and Burnout, LGES fits worse than hand-designed CFA structures (Table 3) you should discuss why and provide intuition. Also, since Ng et al. (2024) is cited as an exact search score-based approach for latent models, a direct comparison would be informative to position LGES.
Please clarify the notation and correct the expression in equation 3.
Typos/edits:
“propogation” → “propagation”;
“eal-life” → “real-life”;

eal-life

δ definition has conflicts in Section 5.2, Appendix C.4

### Questions
In which real domains do you expect the 2|Lp| pure-children and group-wise uniform adjacency assumptions to hold?
What happens when there are fewer than 2|Lp| pure children, or when pure children do not exist for some groups? Do you degrade gracefully (e.g., to equivalence classes over merged latents)? You argue the identifiability story extends because equality constraints over covariance remain. But then the method does not exploit higher-order moments.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a score-based causal discovery for latent variable models, more precisely "Generalized n factor models", that is inspired by GES. It builds upon a MLE score that is proven to be consistent. Moreover, the presented algorithm LGES also provably recovers (up to permutation) the equivalence class of the true graph.

### Strengths
The paper studies an important topic: causal discovery without assuming causal sufficiency of the observed variables. The proposed approach for generalized n factor models is novel and the techniques non-trivial. The theorems for asymptotic consistency are sound and additionally the algorithm is evaluated on synthetic data. Generally, the writing in the paper is clear (apart from a significant number of small typos and grammatical errors).

### Weaknesses
Relation to other recent work:
While the paper gives a broad overview of causal discovery (in the presence of latent variables), the papers (Ng et al, 2024) and (Dong et al, 2024) appear to be closely related. In particular for the results in Section 3.1, it is not clear to me how novel the contributions in the paper are (Theorem 1 is stated as "inspired by Ng et al (2024)"). The paper would benefit from distinguishing itself more clearly here.

GES as a score-based algorithm in contrast to constraint-based methods:
In the paper it is argued that score-based methods compare favorably to constraint-based methods in particular because the latter are prone to error propagation. I agree with the sentiment, but to me it seems that the proposed algorithm suffers from exactly the same problem. If I understand correctly, the algorithm starts with a graph that (at least in some parts) is complete (similar to constraint-based methods), and then deletes edges which lead to no worsening of the MLE score. An incorrectly deleted edge can never be added back into the graph (thus errors propagate) and also there is an intuitive connection to the independence tests that the algorithm aims to avoid (the edge can be deleted asymptotically because some conditional independence is found to hold and delta could be interpreted as some cutoff value). Another thing is that "Greedy" is maybe an unfortunate name, because it usually suggests that each step is done to be locally optimal wrt the score but it doesn't appear to be the case here.

Assumption of the generalized n factor model:
Generally, it is of course clear that restrictive assumptions are needed to recover latent variables. Thus, I find it fair to confine the analysis to a model such as the generalized n factor model even though it may reduce applicability. The paper could do a better job though of explaining why the constraints in Def. 2 are needed. With the proofs deferred to the appendix I find this rather opaque.

Code availability:
The code doesn't appear to be part of the submission, which I find unfortunate. 

Grammatical errors and typos:
While I found the paper generally easy-to-read and thus do not weigh this issue much, the paper contains a significant grammatical errors, small inconsistencies and typos, which could have been avoided by careful proofreading (a list of suggestions for improvement is under questions).

### Questions
1. Can you clarify the relation to recent work by Ng et al. (2024) and Dong et al. (2024)? In particular, the novel aspects of Section 3.1.

2. Can you comment how the approaches distinguishes itself from constraint-based methods that start with a complete graph and subsequently delete edges? How does the proposed method avoid issues with error propagation?

Suggestions:
- line 020: "propose *the* generalized..."
- line 020: "establish *its* global consistency"
- line 022: "by using score" ?
- line 024: "class of *models*" and "which *searches*"
- line 033: "are being made *towards*..."
- line 041: "on *the* discovery of *the* entire structure
- line 042: "on rank or tetrad constraints linearity assumption" ?
- line 048: "statistical *tests*"
- line 053: "most classical"
- line 060: "As a consequence, considerable attention has been given..." please provide references to substantiate this statement
- line 063: "What is the core of structure identifiability by using likelihood scores?" I don't understand what this means
- line 068: "We characterize how *a* likelihood..."
- line 071: "We propose *the*..." and below "*the* score"
- line 072: "MEC" this abbreviation has not been introduced thus far
- line 074: "develop *the* Latent..."
- line 076: "identifiability *guarantees*"
- line 096: "v-structures" this term is not introduced
- line 107: "formulation of *the* likelihood"
- line 139: "*algebraically* equivalent"
- line 159: "*the* number of edges and *the* number *of* measured variables
- line 180: "by making use of score" ?
- line 200: "how *the* likelihood score"
- line 218: "we propose *the* generalized"
- line 266: establishes
- line 281: "we follow the traditional wisdom GES Chickering (2002)"
- line 317: "In Definition 3, it implicitly requires"
- line 338: remove superfluous "."
- line 403 and 407: Def. 1 and Def.2 (also please be consistent with abbreviations)
- line 481 to the end: please remove or replace by a proper section or conclusion

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
4

### Summary
This work proposes a score-based observational causal discovery algorithm in the presence of latent variables for linear SCM. It provides identifiability guarantees for the causal structure involving latent variables. This paper aims to identify the whole underlying causal structure among both observed and latent variables. It also propose Generalized N Factor Model and can identify the structure up to the Markov equivalence class by score.

### Strengths
- This paper seems to extend the work by Ng et al. 2024. The major difference between this paper and Ng et al. 2024 seems to be greedy search vs. exact search. Particularly, this work proposes an algorithm named LGES, a two-phase GES-style greedy search, operating directly over MEC. This is the first score-based greedy search method for the identification of structures involving latent variables with identifiability guarantees.

- By introducing the generalized N factor model, LGES proves that algebraic equivalence implies Markov equivalence, providing a clean route to global consistency up to MEC via a greedy method. That’s a nontrivial step on top of the SALAD’s algebraic equivalence by Ng et al. 2024.

- The description of LGES is clear.

- The paper provides some real-world datasets to motivate the design of the algorithm.

- Proofs seem to be sound. 

- The proposed algorithm can incorporate Verma-type constraints.


Reference:

Ng, Ignavier, et al. "Score-based causal discovery of latent variable causal models." Forty-first International Conference on Machine Learning. 2024.

### Weaknesses
- The algorithm is limited to linear Gaussian SCM. 
- Since this paper is highly related to Ng et al. 2024,  it should give a more in-depth discussion between this work and Ng et al. 2024 in the introduction or related work to highlight the differences and contributions of this work.
- The organization of the paper needs improvement. Some terms such as equality constraints are mentioned in line 136, but it does not provide a brief explanation what that means at the same place of the paper. Rather, it refers to the appendix of the paper to see Definition 6. In terms of organization, this makes the reading more difficult. Similarly for Definition 7. Also, line 197, inequality constraints appear without any explanation about what the means. Line 214 does not provide any contexts about Definition 2 before it says ‘the number of all possible graphs that satisfy Definition 2’. Definition 2 only shows up later in the paper.
- Besides the organization of the paper, the clarity of the paper also needs improvement. For example, GNFM and GNF are used interchangeably. 
- Lines 253-254 are very hard to follow: ‘At the same time, if the number of observed children is insufficient, we can still gather more relevant observations or measurements of the underlying system.’
- The efficiency of the proposed algorithm lies critically on the proposed generalized N factor model. It is debatable whether this model assumption is realistic or not. In practice, there should be way more latent than observed. 
- The paper should introduce some basic terms in causal graphical models such as CPDAG.
- It is confusing to use the same notation $\mathcal{S}$ to represent both the state and a CPDAG.
- The paper should cite the exact theorem number in line 299.

Reference:

- Ng, Ignavier, et al. "Score-based causal discovery of latent variable causal models." Forty-first International Conference on Machine Learning. 2024.

### Questions
- What is $V$ in line 223?
- Lines 246-247: Why does the partitioning make the requirement of $2|\mathbf{L}_{p}|$ observed variables being minimal?
- What is GNF in line 248? Is it GNFM?
- Lines 253-254 say ‘At the same time, if the number of observed children is insufficient, we can still gather more relevant observations or measurements of the underlying system.’ Do the authors mean the number of observed children is sufficient instead of insufficient?
- Why does the CPDAG in the initial state need to explicitly list out all the latent variables in the graph?
- Why is SALAD by Ng et al. 2024 not compared in the experiment?
- Do all the baselines output CPDAG? Are they compared against a ground truth CPDAG in the experiment?
- Can authors report the performance only based on the causal structure among the observed variables, including the F1 scores for arrowheads?
- In the experiment, it says 10 random seeds, but there are 20 randomly generated DAGs. Does that mean each DAG is used 10 times to generate 10 different datasets for each sample size?
- Which CI test is used for GIN in the experiment? What is the alphas used? 
- Why not compare the performance with the baselines when the ground truth graphs do not satisfy Definition 2 (generalized N factor models), instead of the violation of normality and linearity, as the first experiment has already used uniform noise terms?
- Why is the performance only reported for sample sizes 500 and 1000 for the model specification case in the experiment? What about the sample size of 100?

- Why is LCD [1] for latents not compared against LGES?

Reference:

[1] Rohekar, Raanan Y., et al. "Iterative causal discovery in the possible presence of latent confounders and selection bias." Advances in Neural Information Processing Systems 34 (2021): 2454-2465.

### Soundness
3

### Presentation
2

### Contribution
2
