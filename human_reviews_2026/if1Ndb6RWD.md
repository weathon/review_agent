# Information-based Value Iteration Networks for Decision Making Under Uncertainty

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 2

## Abstract
Deep neural networks that incorporate classic reinforcement learning methods, such as value iteration, into their structure significantly outperform randomly structured networks in learning and generalization. These networks, however, are mostly limited to environments with no or very low uncertainty and do not extend well to partially observable environments. In this paper, we propose a new planning module architecture, the VI$^2$N (Value Iteration with Value of Information Network), that learns to act in novel environments with high perceptual ambiguity. This architecture over-emphasizes reducing uncertainty before exploiting the reward. VI$^2$N can also utilize factorization in environments with mixed observability to decrease the computational complexity of calculating the policy and to facilitate learning. Tested on a range of grid-based navigation tasks, each containing various types of environments with different degrees of observability, our network outperforms other deep architectures. Moreover, VI$^2$N generates interpretable cognitive maps highlighting both rewarding and informative locations. These maps highlight the key states the agent must visit to achieve its goal.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces a new DNN architecture for RL. This architecture builds on the idea of Value Iteration Networks, extending it to better contend with the partial observability of POMDPs using the idea of Pairwise Heuristic.

### Strengths
1. The authors are very clear and direct about the limitations of the paper, which I greatly appreciate.
2. I found the presentation for the most part very clear.
3. The idea seems very natural for the purpose of applying VI nets to POMDPs.

### Weaknesses
1. Limited evaluation: only navigation tasks. No RL experiments - only learning from an expert. 
2. Limited novelty: the paper takes two developed ideas (Value Iteration Networks (VIN) and Pairwise Heuristic (PH)) and combines them in an apparently straight forward manner.
3. HP tuning: The HP tuning process for the method and the baseline QMDP are (I believe) unspecified. I.e. was there any? Was it for both methods? Was it under equivalent conditions (i.e. same amount of compute dedicated to the tuning of each?) and motivation for the decisions are missing.
4. Statistical significance: There is a measure of stat. signif. in the tables. Is it standard div.? SEM? Other? I believe it is unspecified (unless I've missed it). Is it over different seeds (how many)? Is there a reason seeds are not necessary here? (random init of the DNN seems to justify different seeds, to me).
5. Introduction to an RL audience could be expanded (see questions / comments). Specifically, I would have liked a brief overview of the training process (dataset with expert actions? learning through interaction with the environment? other?). If the authors could please include a brief explanation in the rebuttal, and add a description into the main paper.

### Questions
I'm open to changing the review score/s. Specifically:
1. If the authors could motivate well that: the evaluation is sufficient for the method to convincingly dominate the baseline in a major set of tasks. Alternatively, increasing the number and types of tasks in the evaluation.
2. If the authors could motivate well that the combination of VIN and PH is not straightforward.
3. If the authors will add and motivate the HP tuning proces and stat. signif. evaluation.

Additional comments:
1. Uncertainty can refer to different things, that are traditionally dealt with very differently, in RL: the partial observability of POMDPs, the epistemic-uncertainty that drives exploration in sparse-reward domains, and the stochasticity of reward / transitions ("aleatoric" uncertainty) that makes everything more challenging. Although I think the presentation is rather clear in its focus on POMDPs exclusively, I would have liked it to be even more explicit - preferably as early as possibly (i.e. intro), on the types of uncertainty that will / will not be addressed in this work.
2. Lines 019-020: "Tested on a diverse set of domains". Since the evaluation is limited to navigation tasks in grid envs. with goal / position known / unknown, I do not think that this can be considered "a diverse set of domains". I would be more comfortable with a claim along the lines of "a range of grid-based navigation tasks".
3. I would have liked the QMDP baseline to be presented in more detail (perhaps in a related work section?), so that the reader can better understand and contrast the method and the (only, previously SOTA?) baseline.
4. In Section 3 there are multiple references to "the value V". It's not clear to me whether this refers specifically the value of the optimal policy $V^*$, the value of a general policy $V^\pi$, or specifically the value of some expert policy $V^{\pi_e}$. Could the authors specify (And add to the paper)?
5. Table 2 and Table 3 seem to measure the same thing (success rate) in different units (% and out of 1). Have I misunderstood? If not - is there a reason they are not presented in the same units?
6. Non-VI-based baselines for POMDP solving (i.e. standard popular algorithms with an LSTM) would put the results in better context.
7. Can the authors include additional results / discussion of compute cost contrast between QMDP and their method (comparable / one is significantly more expensive than the other?)

Additional  minor comments
1. Line 011: In my opinion (/understanding), value iteration is a dynamic programming method, not an RL method (it relies on knowing the full model of the env., not on learning from a dataset of interactions). I do not mean to nitpick, it is more that it would have been easier for me to follow the narrative had DP rather than RL been the term used.
2. Line 017: "This architecture over-emphasizes..". Over means "too much" -> bad. Is that the intention? Perhaps simply "emphasizes"?
3. Equation 1 sums across s' s', probably should be across s' s. I'd also denote the set $o \in Z$ for the first sum, to improve clarity.

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
2

### Summary
The paper seems to extend partially observable VIN, in particular the method QMDP-Net, with a pairwise heuristic for solving POMDP. Experiments on simple gridworld tasks show their approach solves more randomly generated tasks than QMDP-Net.

### Strengths
The pairwise approach seems to be especially well suited for tasks where the state space factors in observable and unobservable variables.

### Weaknesses
I recommend to reject this paper, because I simply fail to understand it. I am familiar enough with the original VIN approach, but without reading the QMDP-Net paper (which is quite old by now) I doubt many will be able to understand this paper. What is needed here is a complete rewrite that explains (in equations) how the kernels are learned, how the belief is updated, and how the resulting VIN is actually solving the POMDP. In particular the main innovation, the pairwise approach, must be explained much more for the presented equations to make sense. I tried to read Section 3 a couple of times, but I am still unsure what is computed here, why it is computed, and how this is supposed to solve a POMDP.

If other reviewers disagree with this statement, I am happy to be convinced otherwise, but I believe an ICLR paper should be accessible even to non-expert of a field like this. 

**Detailed Comments**
- To solve the induced belief-MDP, one needs to do value iteration over the space of all possible belief distributions. I do not understand how this can be achieved in VIN, which only seems to work over discrete state spaces (beliefs are continuous).
- The term uncertainty is ambiguously used: I believe you mean partial observability, and sometimes noisy observations, but uncertainty is more often associated with stochastic environments (aleatoric), or incomplete sampling (epistemic).
- Some symbols are never or insufficiently defined, like $Q$ in Equation 5. This extends to fairly important concepts as the belief distribution $b(s,s')=b(s)b(s')$, where it is never defined how these $b(s)$ are updated or why they are independent from each other. 
- The notation often changes during the text. For example, the reward function is defined (and first used) as $R(s,a)$, but then later used as $R(s)$ or $R(s,a,z)$ without defining these terms formally.
- Equation 1 defines whether $s$ and $s'$ are *distinguishable*, but contains a sum over $s$ and $s'$, which does not seem to make any sense.
- It is unclear to me how the values $V(s, s')$ and $V(s_v, s_h, s'_h)$ are actually represented in the architecture.
- The experiments are missing baselines, i.e., other approaches to solve POMDPs. Just comparing to QMDP-Net is not enough for a top-tier conference.

### Questions
- Why should the actions selected in Equation 5 solve a given POMDP?
- What are the standard deviations in the tables over? Did you train the kernels multiple times and this is the STD over the random seeds? Is this about multiple runs of one set of kernels with noisy observations?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper "Information-based Value Iteration Networks for Decision Making Under Uncertainty" proposes to combine Value-Iteration networks with Partially Observable MDPs (POMDPs) to account for uncertainty in the environment. To overcome the computational complexity of obtaining the optimal policy for POMDPs they adapt a solver that uses the pairwise heuristic that estimates the value $V(s,s')$ for states $s,s'\in S$. Notably, the authors argue that this heuristic is only necessary for the features that are uncertain, which can drastically decrease the amount of required computations. They provide empirical evaluation on two original gridworld datasets and show that their approach notably outperforms QMDP in environments with high uncertainty. Finally, they present "information maps" for each state that appear to make model decisions more interpretable than the value function.

### Strengths
The combination of the solid theoretical foundation of value iteration with MOMDPs to account for uncertainty in the environment is inspiring. The theory developed in this paper was very clearly stated and easy to follow, even without significant knowledge on partial observability or pairwise heuristics for such problems. Furthermore, the observable goal, unknown position environments were motivated by real-world applications (like robots with sonar sensors), justifying their relevance. The emperical results show the benefits of this approach for difficult navigation environments. On top of that, the analysis highlights that both reward exploitation and resolving uncertainty are essential for the model's success. Finally, the obtained information maps provide an impressive insight into the agent's capabilities.

### Weaknesses
As mentioned in the discussion, the lack of evaluation beyond 2D navigation tasks is unfortunate, since it would help to understand the algorithms capabilities in other reinforcement learning domains. Specifically, you state that you successfully tested noisy environments; it would therefore have been interesting to see quantitative results for these experiments, since they are especially relevant for real-world applications. Furthermore, though QMDP-Net performs better than unconstrained networks, a comparison to at least one state-of-the-art baseline not specifically built for partial observability would have been beneficial.

So, the main issues are:
- Limited experiments on relatively toy 2d environments 
- Lack of comparison against other established methods

### Questions
I am unsure how the threshhold $\lambda$ has to be selected by a domain expert and why it should be close to 1. The reasoning behind this choice would be an interesting note to fully grasp the presented approach.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces new a differentiable planning architecture ($VI^2N$) for decision making under uncertainty in partially observable environments. The method extends Value Iteration Networks (VINs) by integrating a pairwise heuristic mechanism (from prior works) that aims to explicitly model and reduce uncertainty before exploiting rewards. The authors also proposed how $VI^2N$ can leverage Mixed Observable MDP (MOMDP) factorisation to reduce computational complexity and improve scalability. They then present various experiments in several simple grid-world domains with different degrees of uncertainty (covering cases where the agent’s position or the goal’s position is unknown) and shows that $VI^2N$ outperforms QMDP-Net (the most directly related prior work) in success rate. An ablation study on the recurrence depth of $VI^2N$ further highlights the importance of the model’s planning horizon.

### Strengths
* **Significance and originality:** The paper tackles an important limitation of previous value iteration-based networks (VINs, QMDP-Nets) by explicitly integrating information-gathering behavior into planning under uncertainty. This is highly relevant for domains such as robotic navigation and grid-world planning tasks.
* **Experimental coverage:** The study provides experiments in several partially observable gridworld settings, including tasks with unknown agent position / known goal and known agent position / unknown goal, capturing a different types of uncertainty.
* **Comparative results:** The paper demonstrates consistent and often substantial improvement over QMDP-Net, the closest relevant baseline, showing that $VI^2N$ can perform better in environments with significant perceptual ambiguity.
* **Insightful ablation:** The analysis on the number of recurrences in the VI and $VI^2$ modules reveals how performance depends on planning depth, providing useful interpretability and model understanding.
* **Clarity and theoretical grounding:** The exposition of the pairwise heuristic is conceptually clear and builds logically on established literature in POMDPs.

### Weaknesses
* **Limited generality:** Like QMDP-Net and other VIN-based models, $VI^2N$ assumes a discrete environment with spatially invariant transition kernels and discrete actions, which restricts applicability to continuous or large-scale real-world domains.
* **Figure clarity:** The main architecture diagram (Figure 1) is dense and difficult to interpret, lacking a proper legend and a lot of missing arrow direction indicators, which makes following data flow challenging.
* **Restricted evaluation scale:** Experiments are limited to small binary grid-worlds, which makes unclear how it will scale to larger maps or continuous domains (e.g., those used in the prior VIN or QMDP-Net works).
* **Outdated baselines:** The authors only compare against QMDP-Net, which is a fairly old baseline (Karkus et al., 2017). They justify this by saying that the QMDP-Net paper showed that they perform significantly better than unconstrained networks. However there are several more recent POMDP baselines other than RNNs/LSTMs and behavior cloning (which is what Karkus et al. compared against), like transformers (e.g Decision transformers, TrXL, etc). Additionally, given that the experiments are all trained offline using expert trajectories, there are several more recent offline RL baselines (like CQL, IQL, etc).
* **Weak evidence for claims:** It looks like the authors did not average their results across several training runs (or the number of seeds used is not reported). Hence it is unclear if the results are significant (let alone statistically significant). There are also no plots/results to validate the claim that $VI^2N$ focuses on reducing uncertainty *before* exploiting rewards. Finally, it is unclear if QMDP-Net also got/used the same factorised representation in Task 2 with the fully observable agent’s location (so the belief should similarly be only over unobserved variables).
* **Little ablations and analysis:** Only one ablation (recurrence depth) is provided. No analysis of other hyperparameters and model failure modes is given. For example it is unclear how performance varies with $\lambda$, and what tasks/situations are problematic for $VI^2N$ due to the changed bellman equation (Equation 4).
* **Lack of robustness tests:** Although the architecture is designed for high uncertainty, the paper does not test performance under systematic increased stochasticity. E.g.
  - Increasing action slip probabilities
  - Noisy observation models like the classical “noisy TV” scenarios where observations become uninformative (e.g. where transitions into a specific grid position gives uniformly random observations).
* **Scope of generalisation:** All experiments use offline expert demonstrations. Hence it is unclear how the method will be affected by non-expert demonstrations or the online RL settings. 

Some of these limitations are acknowledged in the last section (Discussion), including a couple additional ones. However given the severe lack of analysis as mentioned above (and no theory), the authors really should have included some of the experiments they leave to future works.

### Questions
Please see the weaknesses above. Mainly:

1. **Baselines:** Why were no more recent POMDP or offline RL baselines included? Could you compare against transformer-based or uncertainty-aware architectures/algorithms (like CQL) to contextualise performance?
2. **Scalability:** How does $VI^2N$ scale computationally and in performance for larger or continuous environments? Have you tested on larger grid-worlds or 3D navigation tasks as suggested in discussion, since you claim it is easily doable?
3. **Statistical significance:** How many training seeds were used? Could you report mean ± std over multiple runs to assess (or at least illustrate) the significance of the improvements?
4. **Robustness to noise:** How would $VI^2N$ perform under increased stochasticity (e.g., transition noise or noisy observations)? Does the network maintain its advantage over QMDP-Net in such cases? Does the plots of informative areas through the value function of pairs change as one would expect? 
5. **Ablation depth:** Beyond recurrence count, could you ablate other design elements (e.g., factorisation, pairwise distinguishability threshold λ) to clarify what drives improvements and the corresponding tradeoffs? Are there failure cases as a result of the way the rewards and Bellman equations are modified (or can the authors prove they maintain optimality)? In general could the authors analyse failure cases?
6. **Interpretability consistency:** Are the “informative area” maps consistent across runs and different environment structures, or do they vary significantly depending on training initialisation?
7. **Generalisation to non-expert settings:** Given that experiments rely on expert trajectories, how would $VI^2N$ perform under non-expert trajectories?

### Soundness
1

### Presentation
3

### Contribution
2
