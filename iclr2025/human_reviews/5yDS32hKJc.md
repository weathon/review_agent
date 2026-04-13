## Human Reviewer 1

### Summary
This paper proposes Earliest Disagreement Q-Evaluation (EDQ), a method for estimating causal effects of sequential treatments with irregular timing. The key contribution is adapting Q-learning approaches to continuous-time settings while maintaining model-free properties. The work focuses on healthcare and financial applications where treatment timing is crucial.

### Strengths
1. The paper addresses a crucial and previously underexplored problem in causal inference: estimating the effect of interventions on treatment timing in sequential decision-making. This problem has practical implications in healthcare and finance, where treatment timing often has substantial impact on outcomes. The authors provide compelling motivation and clearly articulate why existing methods are insufficient.

2. The proposed EDQ method demonstrates impressive scalability and flexibility by being compatible with modern deep learning architectures like transformers. This is a significant advancement over existing methods that either require time discretization or don't scale to large models, making the approach more practical for real-world applications.

### Weaknesses
1. The paper's title promises insights into "When and What to Do," but primarily delivers a method for evaluating pre-specified timing policies. While the proposed EDQ method effectively handles off-policy evaluation of treatment timing effects, it provides no framework for discovering optimal timing strategies.  The experimental section further highlights this gap, focusing solely on estimation accuracy rather than demonstrating practical utility in finding better treatment schedules. Additionally, despite the title's suggestion, the paper explicitly omits treatment selection (the "what" aspect) to focus on timing, making the scope narrower than advertised.

2. The experimental validation relies on overly simplified simulation settings, particularly in modeling treatment delays with an exponential distribution  $\delta\sim exp(\lambda_a)$. Although the method's theoretical guarantees don't depend on specific distributional assumptions, the paper would be stronger with experiments using more realistic delay distributions and multiple vital signs. This limitation raises questions about the method's practical performance in real-world healthcare settings where treatment timing follows more complex patterns.

3. The technical presentation has several gaps that affect its accessibility and reproducibility. The paper lacks analysis of computational complexity and clear guidelines for practical implementation.

### Questions
1. Definition 5, which is the most important definition, is not clear. For instance, what does $\tilde P_t^a$ mean? I cannot find its definition throughout the whole article including appendix.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper presents a novel approach for handling irregular timing in off-policy evaluation within sequential decision-making frameworks. It extends the classical Bellman equation in fitted Q-evaluation (FQE) to account for cases with irregular time intervals and introduces the concept of measuring the "earliest disagreement time" in Q-function updates.

Overall, the paper is well-written, with a natural flow that guides the reader from foundational ideas to specific methods. While reading the introduction, I was somehow expecting a real-data analysis that could demonstrate the approach, perhaps with an example like ASCVD control or another scenario where OPE might reveal insights under varied intervention timings. I understand the challenges of conducting OPE with observational data, so this is not a request for inclusion in the rebuttal. However, such an analysis could add an interesting dimension to showcase differences in performance in real-world applications and offer potential interpretations. Some terms and assumptions could benefit from clearer explanations, but overall, this paper provides a practical and effective method for handling irregular time in OPE without imposing heavy modeling or causal assumptions.

### Strengths
1. The paper’s motivation is well-illustrated with examples. I appreciate the effort to clarify the setup in Section 2-3, which makes the new approach easier to understand. The comparison with classical FQE and the challenges outlined at the end of Section 3 are also helpful.

2. The simulation studies demonstrate promising results, effectively showcasing the benefits of the proposed EDQ by comparing it with FQE and MC methods.

### Weaknesses
1. Some notations are unclear on first mention. Here are a few areas where clarification could improve readability (please correct me if I missed something):

 (a) In Definition 4, Line 170, does $\ll$ in $P\ll P_{obs}$ denote absolute continuity? It might be helpful to define this symbol for readers who may not be familiar with the concept.
 
 (b) In Theorem 1, Line 251, assumptions 1-3 don’t appear to be clearly defined or referenced in Section 2.2.

 (c) In Algorithm 2, Line 279, the loss function $l(Q_t,y)$ seems to be undefined. While it may be user-specified and flexible, adding a brief note about its role and possible choices could provide helpful context.

### Questions
1. How does the computational time of your algorithm compare with classical FQE? 

2. In cases where stages are regularly spaced in time, would EDQ reduce to classical FQE? A brief discussion or proof in the paper about how EDQ behaves in the limit as time intervals become regular would suffice.

3. In practical applications, a common workaround for irregular timing is to include the interval between the $k-1$th and the $k$th stages as a component of the state $\boldsymbol{X}$, then proceed with classical FQE for Q-function estimation. While this may lack methodological novelty, industry practitioners often find it to be a straightforward and effective way to account for "irregular time" in value estimation. Could the authors elaborate on the motivation for considering "earliest disagreement times" and explain situations where this approach might perform better?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces off-policy evaluation through the framework of stochastic point processes and presents Earliest Disagreement Q-evaluation (EDQ) as a method for estimation. Under the causal validity assumption, the authors demonstrate that EDQ can accurately identify the true policy value. Additionally, this assumption is elucidated using a local independence graph. Experiments conducted on simulated data confirm the effectiveness of the proposed approach.

### Strengths
**1**. The formulation of sequential causal effect estimation using point processes is novel, effectively addressing the irregular nature of treatments and covariates in various real-world applications.

**2**. The EDQ method for estimation is compelling, with corresponding consistency results rigorously demonstrated in Theorem 1.

**3**. The assumptions required for causal identification are elucidated through a local independence graph, providing an intuitive understanding of their implications.

### Weaknesses
Experiments should be conducted on real-world datasets or, at the very least, on simulated datasets generated from real-world data to provide more convincing validation.

### Questions
**1**. To my knowledge, consistency and the SUTVA assumptions are also essential in causal effect estimation. Why are these assumptions not required in your work?

**2**. What is the impact of different sequential modeling architectures (such as Transformers, RNNs, regression models, etc.) on your method? Is the estimation sensitive to the choice of architecture and hyperparameters?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper proposes a method for estimating the causal effect of policies using Earliest Disagreement Q-Evaluation (EDQ), where a policy determines both the action and its timing. Specifically, the authors aim to estimate the outcome of a target policy by leveraging dynamic programming, extending fitted Q-evaluation from discrete to continuous time. The update function is driven by the earliest point of disagreement between the behavior policy and the target policy.

### Strengths
- The paper considers the problem of evaluating a policy using continuous-time causal inference, focusing on off-policy evaluation.
- To tackle the challenges of policy evaluation in continuous time, where the conventional Q-function collapses, the paper leverages the property of countable decision points in a point process and introduces a simple yet efficient method based on earliest disagreement times.
- The paper also provides identifiability conditions for the causal estimands, ensuring accurate estimation.

### Weaknesses
- Could the authors discuss the connection between the Bellman equation in discrete time and Theorem 1? Equation (1) is equivalent to the Bellman equation for a finite-horizon problem in discrete times, and Theorem 1 appears to be a direct extension using differential equations and stopping times.
- The proposed framework allows the trajectory outcome to be a sum of time-specific outcomes (lines 79, 204-205). Are there any constraints on the number of observations in the trajectory? If the total number of $k$ is indefinite, the scale of $Y$ will vary with the number of observed time-specific outcomes, potentially affecting the outcome’s interpretation and stability.
- Writing improvement suggestions
  - In the second paragraph of Section 1, the motivation could be clearer by explicitly stating the challenges related to sequential treatment and irregular timing, with a particular emphasis on the primary challenge about decision times. In the current introduction, it is not immediately clear to me which difficulties are being discussed in lines 37-39.
  - In the third paragraph of Section 1, it would help to introduce the components of an intervention. From my understanding, there are two key components: the timing of the intervention and the intervention itself. Otherwise, readers might not yet understand whether treatment timing in line 49 is determined by the environment or chosen by the policy.
  - In line 79, $Y_k$ has not been defined yet.
  - At the end of line 88, should $d N(t)$ be $d N^l (t)$?
  - Could you clarify the distinction between a marked decision point process in Definition 1 and a marked temporal point process (Upadhyay et al. 2018)?
  - A period is missing from line 103.
  - In line 105, should each trajectory be indexed by $i$ or $m_i$? It seems that $m$ is the total number of trajectories and $i$ takes values from $[m]$.
  - What are assumptions 1-3 in line 251?
  - In equation (2), should the right-hand side condition on $\mathcal{H}_t$ as in equation (1)?

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 5

### Summary
The paper aims to tackle the temporal irregularity when estimating causal effects of policies, by proposing a method called EDQ (earliest disagreement Q-evaluation) based on dynamic programming. The simulation demonstrates more accurate estimations of EDQ.

### Strengths
- The work investigated an important problem, i.e., when to intervene with irregular times.

- The work is well motivated, given many human-related backgrounds such as healthcare and finance requires a carefully evaluation for when to provide proper intervenes.

- The paper is well structured and polished.

### Weaknesses
- Given the work is called into category of off-policy evaluation. Related works in off-policy evaluation should be thoroughly discussed in the paper, either in Introduction or Related work. I understand there can exist some key differences between traditional off-policy evaluation and EDQ, but should be carefully and thoroughly discussed and compared in experimental settings. Also, there exists work regarding when-to-treat problem (e.g., [1]). A further comparison and discussion regarding EDQ and those works would be great. 

- Since one major motivation of the work is human-related when-to-treat problem, and the paper used a lot of healthcare examples (which is comprehensive). I’m curious about whether the work can be examined on some related settings. It’s understandable that running real-world experiments would not be feasible and high-stake, but it would be more impressive to provide experiments on some empirical motivated settings, e.g., sepsis [2], autism [3], etc.




- Minors:

It would be great if code and/or data can be released for reproducibility.

Line 99. a was defined as action and then represented number of multivariate unobserved process

Line 185. FQE needs careful citations, given it’s a well-established work in off-policy evaluation and optimization.

Line 308, 355, 363. Lower cases for bolded words?

[1] Learning When-to-Treat Policies

[2] Counterfactual Off-Policy Evaluation with Gumbel-Max Structural Causal Models

[3] Off-policy Policy Evaluation For Sequential Decisions
Under Unobserved Confounding

**Update after rebuttal**:

The authors conducted further healthcare analysis to support their major contribution, and further discussions regarding related works.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 6

### Summary
The authors propose a method for estimating the effect of a continuous-time treatment policy, defined as a treatment intensity function (when to treat) coupled with a mark distribution (what treatment to select), from observational data. Their approach, Earliest Disagreement Q-Evaluation (EDQ), is based on the principle that effects of observed versus proposed policies should not diverge until the point when the two policies disagree. The authors first review the identifiability of causal effects in this setting, then present the proposed estimator and associated algorithm. They construct a simulation setting in which a supposed vital sign is responsive to treatment, with diminishing returns. They show that EDQ estimates the effect of alternative (non-observed) treatment policies more accurately than comparator methods in this setting.

### Strengths
The authors tackle a novel problem and present an elegant solution. Identifiability of causal effects follows from previous results, and is presented clearly. The related work section is outstanding: comprehensive, highly informative, and well-written. The simulation setup is intuitive and clearly presented, and the proposed method outperforms alternatives.

### Weaknesses
- The key weakness is that the experimental results are underdeveloped. I would have liked to see more variations on the simulation results as well as application to at least one real dataset.

- I was confused by the notation and presentation in Definition 5, which then made it difficult to understand the details of EDQ. Specifically, I am not clear on how $\tilde{P}_t^a$ is defined or how to sample from it. I have lowered my score for this reason, but I'm happy to increase it if the authors clarify the presentation and/or if other reviewers did not have similar difficulty.

- While defining the policy as an intensity (i.e. rate) is interesting, I have a hard time imagining a realistic scenario where it would make sense to sample treatment times rather than deciding whether/how to treat at fixed or given intervals.

- (minor) The title ("When and What") is a bit misleading. I take the point that learning when to treat (i.e., the rate portion of the policy) is the challenging part, but saying both when and what in the title is misleading, since the paper and experimental results focus only on the former.

- (minor) There are quite a few typos / minor errors, particularly in later sections of the paper.

### Questions
- Line 89: Why do these assumptions imply that the process is self-exciting? Please elaborate.
- How well does the algorithm scale? Could the authors comment on the computational complexity?

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3