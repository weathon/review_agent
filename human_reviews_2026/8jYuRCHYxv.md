# Missingness-MDPs: Bridging the Theory of Missing Data and POMDPs

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
We introduce missingness-MDPs (miss-MDPs); a subclass of partially observable Markov decision processes (POMDPs) that incorporates the theory of missing data.
Miss-MDPs capture settings where, at each step, features of the current state may go missing, that is, the state is not fully observed.
Missingness of state features occurs dynamically, governed by the missingness function, a restricted observation function.
In miss-MDPs, we distinguish three types of missingness functions: 
missing completely at random (MCAR), missing at random (MAR), and missing not at random (MNAR).
Our problem is to compute a policy for a miss-MDP with an unknown missingness function from a dataset of observations. 
We propose probably approximately correct (PAC) algorithms that, from a dataset, approximate the missingness function and, thereby, the true miss-MDP.
We show that, for specific missingness functions, the policy computed on the approximated model is epsilon-optimal in the true miss-MDP. 
The empirical evaluation confirms these findings and shows that our approach becomes more sample-efficient when exploiting the type of the missingness function.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- This work addresses a particular type of POMDP, called a "miss-MDP," where certain state features are missing at random or through dependencies with other state features, and proposes an approach for estimating the missingness function from a historical dataset and computing a near-optimal PAC policy.
- They evaluate the proposed algorithm (and its variants) using an ICU dataset and a toy domain under different assumptions about the missingness function and compare their performance to the oracle and the uniform action-selection baselines.

### Strengths
- Originality and significance: This work extends existing research on POMDPs and MDPs with missing state observations by introducing an estimation method for the missingness function under different assumptions about missingness. 

- Quality: The paper includes both a toy domain and a clinical dataset.

- Clarity: Overall, the writing is clear, with a few outstanding questions regarding the distinction between MNAR and non-simple MAR (which the reviewer elaborates on in the Weaknesses section). Figure 2 effectively illustrates different cases of missingness and is helpful to include.

### Weaknesses
Originality and significance: 
- While the problem itself is interesting and important, the analysis and applicability of the proposed algorithm are limited to tabular states, and it is unclear how the method could be extended to high-dimensional states. Adding to this point, the experimental domains have fairly small state spaces (2 for the predator task and 4 for the ICU task) whereas in real-world settings, where the motivation for POMDPs and missingness comes from, state observations are typically high-dimensional, which limits the applicability and significance of this work. 

- Although the inclusion of a PAC analysis is valuable, it is fairly standard and not technically novel enough to constitute an independent contribution. 

Experiments: 
- The current experiments include only variants of the authors’ own algorithms (besides the uniform and the oracle baselines). The authors could consider modifying or including baselines from prior work, such as deep variational methods by Igl et al., 2018, since it is difficult to calibrate the empirical advantages without comparisons to other approaches, even if those methods are based on different assumptions about missingness. (Igl et al., 2018. Deep variational reinforcement learning for POMDPs. There may be other works that can be adapted to the problem setup of this paper and can be included as comparisons.)

Clarity:
- The distinction between non-simple MAR and MNAR discussed in Section 4.2 (where the authors state that the non-simple MAR case satisfies the three conditions of independence, no self-censoring, and positivity) is unclear. In particular, the example provided for "MAR but not simple MAR" is: “Now, the missingness probability of feature 1 depends on the value of feature 2 (only if observed), while feature 2 itself misses with probability 0.5.” This example appears to correspond to the MNAR case shown in Figure 2, where $S_2$ affects $R_1$ (so feature 1 depends on the value of $S_2$ if it's observed), while $S_2$ itself can be missing as suggested by $R_2$. 

- This confusion continues with Lines 200-201, where the authors describe MNAR as cases where "missingness probabilities may depend on the values of missing functions" and provide a self-censoring example for MNAR. However, in the later sections (particularly 4.2), MNAR is described as non-self-censoring, independent, and positive. The authors could consider providing a different example of MNAR that satisfies these conditions to clearly distinguish between non-simple MAR and MNAR. Without further clarification, the description of MNAR may mislead readers into interpreting it as cases where missingness depends on the feature’s value (e.g., when temperature is too high or too low to be recorded properly, making the missing value itself informative about the measurement).

### Questions
- Question about the distinction of MNAR in section 4.2 and non-simple MAR is raised in Weaknesses.

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
3

### Summary
This paper formalizes miss-MDPs and gives PAC-type guarantees for learning missingness under simple MAR and MNAR with independent incidators and no self-censoring, then use a standard POMDP method. Experiments suggest convergence to optimal policy with reasonable data sizes.

### Strengths
- Formalize the miss-MDPs to bridge between missing data taxonomy and sequential decision making.
- Shows under MAR, belief updates do not depend on missingness probabilities.

### Weaknesses
- AsMAR estimates the set $\hat{I}_{always}$, but Theorem 1's PAC statement does not condition on this set. 
- Counting set for AIMI is too restrictive. $Z_s^{i,r_i}$ requires all features $j\neq i$ be observed and exactly match $s$, discarding any sample where any other component is missing. This can devastate sample efficiency, even under positivity, and the paper gives no guidance on the resulting rates. 
- By design, $M$ depends only on $s$ (not $a$ or $t$), and your POMDP observation in prelims is also action-independent. Many practical missingness processes (e.g., clinical testing) are action-selected.

### Questions
See weakness

### Soundness
2

### Presentation
3

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
This paper presents a subclass of POMDPs where the state can be observed with certain components missing. Standard assumptions on missingness from the theory of missing data are investigated. Theoretical analysis is provided for estimating the missingness model given a dataset of trajectories sampled by running a so-called fair policy, and the consistency of a policy computed using the estimated model. The theoretical insights are empirically validated on two simple problems.

### Strengths
* Incomplete state observations can arise in practice. Formalizing and analyzing this are interesting.
* The writing is clear and easy to follow.

### Weaknesses
* The main contribution of the paper is in the theoretical analysis, but this applies to POMDPs with finite state and action spaces only, which is not particularly surprising.
* Empirical experiments are only done on simple small problems.
* A minor comment is that PAC as used in this paper is different from the standard concept of PAC as introduced by Valiant, which can be confusing.
* Another minor comment is that "observation" and "trajectory" are used synonymously at some places (e.g. the abstract mentions "a dataset of observations", while it means a dataset of trajectories).

### Questions
* Does the analysis cover the case when some states are not reachable from some other states?
* In Theorem 1 and Theorem 2, presumably the result holds for not just a particular $n^{\*}$, but for any sufficiently large $n^{\*}$, is it?
* The experiments show that exact missingness function can be estimated. It is surprising that probabilities can be estimated without any error using random datasets. Did I miss something?

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
4

### Summary
This paper presents a novel formalism, so-called missingness Markov decision processes (miss-MDPs), which is a subclass of partially observable MDPs (POMDPs). The main contribution of the paper is to introduce three types of missingness functions, which are missing completely at random (MCAR), missing at random (MAR),
and missing not at random (MNAR), into MDPs. The authors prove finite-time PAC guarantees for estimating missingness functions and ensuring \epsilon-optimal policies.

### Strengths
The paper gives an interesting formalism for a subclass of POMDPs and has PAC-bounds on learning policies for miss-MDPs.

### Weaknesses
Overall, I believe the paper has an interesting theoretical approach, but it lacks motivation in using the miss-MDP formalism instead of POMDPs, as the authors did not show a specific computational advantage either in theory or in practice. I will have some questions in my review.

### Questions
What is the main advantage of using miss-MDPs instead of simply treating them as a special observation function in POMDPs?

What new capabilities or insights does the miss-MDP formalism provide that POMDPs do not? You only showed finite-time PAC bounds in your paper.

Can you provide some examples where this approach may provide better theoretical and practical guarantees compared to just using POMDPs?

How does admittability affect learning or belief updates later? Could multiple states admit the same observation z? If so, how does this ambiguity influence identifiability?

What is the rate at which the error in \hat{M} translates into policy suboptimality \epsilon? Is it possible to establish bounds as a function of the number of states, observations, and the missingness function?

### Soundness
3

### Presentation
2

### Contribution
2
