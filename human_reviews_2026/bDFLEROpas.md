# Back Propagation through Auctions: First-Order Policy Gradient for Auto-Bidding

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6, 4

## Abstract
In online advertising, auto-bidding agents compete in high-frequency auctions by setting a bidding parameter for each time interval, that scales estimated impression values into actual bids. 
While prior work has framed this sequential decision problem as a reinforcement learning (RL) task, we identify that standard RL methods overlook key structural properties of the auto-bidding environment: agents receive fine-grained, impression-level feedback, and the objective is nearly differentiable due to the high density of impressions within each interval.
We leverage this structure to propose First-Order policy gradient for auto-Bidding (FOB), a method that directly computes policy gradients by smoothing historical auction data and backpropagating through the sequential auctions. 
FOB leverages Myerson's lemma, a cornerstone of auction theory, to explicitly derive gradients.
We validate FOB on  AuctionNet, a public auto-bidding environment, where it consistently outperforms standard RL baselines and domain-specific auto-bidding methods, achieving superior performance with greater stability and faster convergence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies the problem of automated bidding at the step level in fast-paced online ad auctions. The central insight is that the auto-bidding environments are different from typical MDPs in standard reinforcement learning problem settings, as they provide detailed feedback for each impression (i.e. values and clearing prices for individual auctions). Also, since there are so many impressions packed into each time step, the relationship between the bidding parameter and the step's reward/cost is more smooth. To resolve these problems, the authors develop ``FOB'', a first-order policy-gradient method that backpropagates through actual historical auction instances. At each step, they smooth out the piecewise-constant reward function $r_t(a)=\sum_i v_{t,i}\mathbf{1}{a,v{t,i}>p_{t,i}}$ into a differentiable version $\tilde r_t(a)$ and calculate $\nabla_a \tilde r_t(a)$ using either piecewise-linear slopes between breakpoints or a Savitzky–Golay fit. For the step cost $c_t(a)$, they apply Myerson's lemma to enforce $c_t(a)=a,r_t(a)-\int_0^a r_t(\alpha),d\alpha$, which gives them the analytical gradient $\nabla_a \tilde c_t(a)=a,\nabla_a \tilde r_t(a)$. The policy gradient across multiple steps then comes from the chain rule, with earlier actions affecting later ones through the remaining budget $B_t$. They treat budget depletion steps specially by differentiating with respect to $B_u$ using $\partial r_u/\partial B_u=1/a_u^\star$ where $a_u^\star=c_u^{-1}(B_u)$. On the theory side, they offer: (i) a reduced state abstraction result (Theorem 3.1) proving that $s_t=(t,B_t,B)$ is sufficient when step arrivals are independent, (ii) a derivation through Myerson establishing the identity they use for cost gradients, and (iii) a generalization showing the cost/reward linkage works for any single-parameter truthful mechanism. In experiments on AuctionNet (48 steps per episode; roughly $10^5$ impressions per episode), FOB beats PPO, TD3, SAC, and a specialized DDPG variant (USCB) in normalized return across budgets $B$ from 150 to 400, with quicker and more stable training. Ablation studies reveal comparable performance with different smoothing methods, and a partial-feedback variant (observing only prices of won impressions) using one-sided gradients performs nearly as well as full-feedback FOB. The code is promised, and the paper explicitly declares LLM usage for writing and visualization code.

### Strengths
(1) Clear identification of domain structure (counterfactual replay, high impression density) enabling first-order gradients.

(2) Succinct algorithm (actor-only; no TD/critics); stable training; competitive wall-clock.

(3) Solid empirical showing on AuctionNet; ablations on smoothing and partial feedback variant.

### Weaknesses
(1) Assumption on feedback completeness and realism: The approach assumes access to the clearing prices $p_{t,i}$ for losing auctions to construct the breakpoint set for $r_t(a)$ (and exact offline-optimal normalization). While the paper does discuss partial feedback and provides one-sided estimators, major platforms do not always release losing prices to advertisers; even if the platform itself trains on behalf of advertisers, this assumption should be delineated more precisely. Please quantify how performance degrades as the fraction of missing losing prices increases beyond the one-sided case (e.g., random hiding or biased hiding w.r.t. $p/v$), and whether calibration tricks can recover gradients.

(2) Smoothing bias and sensitivity: The surrogate $,\tilde r_t(a)$ is a smoothed version of a step function. The paper provides two smoothing methods (piecewise-linear and SG), but there is no analysis of the bias introduced by smoothing choices and hyper-parameters (e.g., SG bandwidth $h$, polynomial degree $d$, number of points, etc). How sensitive is FOB to these choices across different traffic regimes (e.g., fewer impressions per step, heavy-tailed $v$ or $p$)? An ablation across impression densities (e.g., subsampling auctions per step to simulate sparse steps) would help verify the “high-density” assumption’s necessity.

(3) Generality beyond second-price truthful auctions: The Myerson identity and the clean gradient link hinge on single-parameter truthful mechanisms. The paper mentions extensions, but first-price and GSP do not directly admit the same identity. The method would then need explicit smoothing for cost (and potentially different chain rules). It would be good to either include a small experiment or a derivation sketch for first-price auctions showing how FOB adapts and whether the observed stability persists.

Of course, it is impossible for me to complete a very in-depth inspection on the soundness or technical details on a paper whose topic is out of my personal research domain, due to the 2-week reviewing time window versus 5 assigned submissions most of which are with extensive derivations of equations. I just reported what I found and what I thought.

### Questions
Your one-sided estimator uses only winning breakpoints. How does FOB behave when the policy changes significantly between epochs (e.g., shifts $a_t$ upward) and the relevant left-neighborhood has very few wins? Do you employ any exploration or smoothing safeguards to avoid vanishing gradients in such cases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Assuming that the impression sequence is independent of the bidding actions, the paper proposes a auto-bidding optimization algorithm that computes first-order policy gradients through differentiable auction dynamics. The proposed method is evaluated on AuctionNet, where it achieves superior performance and stability compared to standard reinforcement learning baselines.

### Strengths
1. Effectively leverages the assumption that the impression sequence is independent of bidding actions to design a more sample-efficient reinforcement learning algorithm.
2. Provides reproducible results with clear implementation details and open-sourced code.
3. Conducts comprehensive experiments on the public AuctionNet benchmark, demonstrating strong empirical performance.

### Weaknesses
The assumption that the impression sequence is independent of bidding actions is overly strong and unrealistic. In real-world online advertising systems, auctions are typically conducted in two stages, where advertisers can only observe impressions after entering the second stage (see [1]). Consequently, the observed impression sequence often depends on the advertiser’s bidding behavior. While the paper presents an interesting approach under idealized conditions, its practical contribution remains limited.

[1] Mou et al. "Sustainable online reinforcement learning for auto-bidding." NeurIPS 2022.

### Questions
How would the proposed method perform in a more realistic auction setting where the impression sequence depends on the bidding actions?

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
This paper leverages the high-intensity nature of impression arrivals in step-level real-time bidding to observe the cost and reward functions continuous and smooth. Building on this, it applies Myerson’s lemma to derive Proposition 4.1, which establishes the relationship between the first-order derivatives of cost and reward. The smoothness property is then exploited to approximate these derivatives, and the Adam optimizer is employed during backpropagation to update deep neural networks for bidding. This FOB approach enhances the stability of reinforcement learning training. As a result, the proposed method achieves approximately 82% of the optimal cumulative reward obtained from hindsight linear programming (Equation 1).

### Strengths
1.	The authors utilizes the classic results from the auction theory to improve reinforcement learning, resulting in a novel method.
2.	The code is well written and very readable.

### Weaknesses
Please refer to the question section.

### Questions
1.	In line 231, why does the state information $S_t$ does not include the past second highest price $p_t$? Is it $\gamma_{\tau}$ instead of $v_{\tau, i}$? 
2.	Sufficiency of the state space: Theorem 3.1: it is not realistic to assume $\gamma_1, \dots \gamma_T$ to be mutually independent. At least, they are temporally correlated.
3.	Regarding Question 2, how would your experiment results differ if you enlarge the state space to include all the observed information in the past?
4.	As the author mentioned in the limitation, it is not realistic to assume the true value $v_{t,i}$ is observable. Consider, if we can only observe a noisy but unbiased version $\hat{v}_{t, i}$ of $v_{t,i}$, how will it affect your experiment results?
5.	Just curious, why the ``evaluate`` method in class Actor(nn.Module) in fob.py, ppo.py, sac.py, td3.py and uscb.py are slightly different? A follow up question, are the experiment results robust to Actor specification, i.e. rank consistent?
6.	More baseline experiments are suggested, such as online linear LP evaluated in AuctionNet: A Novel Benchmark for Decision-Making in Large-Scale Games baseline program. I saw the implementation of OnlineLpBiddingStrategy. But why are its results not included in the experiment sections?

The reviewer is willing to reconsider the rating if all the above questions are properly addressed.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces FOB (First-Order policy gradient for auto-Bidding), an algorithm for optimizing agents in online ad auctions. The authors show that standard reinforcement learning methods inefficiently handle the environment's information, overlooking two key properties: (1) availability of impression-level feedback and (2 smoothness of reward/cost functions due to high impression density. FOB exploits this structure by directly computing low-variance, first-order gradients by smoothing historical auction data and backpropagating through the sequential auctions. By leveraging Myerson's lemma to analytically link cost and reward gradients , FOB achieves "superior performance with greater stability and faster convergence" than standard RL baselines on the AuctionNet benchmark.

### Strengths
- The paper's primary strength is identifying and exploiting the "nearly differentiable" structure of the auto-bidding problem. Moving from a high-variance, black-box, zeroth-order optimization (standard RL) to a low-variance, first-order optimization by leveraging domain-specific structure is a powerful and well-justified leap.
The application of Myerson's lemma to derive the cost gradient directly from the reward gradient in particular is a clean and insightful theoretical contribution.

- The resulting FOB algorithm is simpler than the deep RL baselines it competes against. The fact that this simpler, more direct method achieves superior performance and stability is a very strong and compelling result.

- The paper includes valuable analyses that strengthen its claims. The ablation on gradient smoothing methods (Table 2) shows the choice isn't critical.

### Weaknesses
- The main method assumes full knowledge of the instance $\mathcal{I}$, including the prices $p_{t,i}$ of auctions the agent lost. This is a very strong assumption and often not practical? While this is well-addressed in Appendix C, this limitation should be more prominent in the main paper.

- The method trains by optimizing the expected return over a buffer of historical instances. This implicitly assumes that the distribution $P$ of instances is stationary where s real-world ad auctions, this is often false. Standard online RL methods, while higher variance, may be more adaptive to such non-stationarity.

- The true reward/cost functions $r_t(a)$ and $c_t(a)$ are piecewise-constant step functions, and their true gradient is zero almost everywhere. The paper introduces smoothing (piecewise-linear or SG filter) to create a differentiable surrogate $\tilde{r}_t(a)$. This smoothing introduces an approximation error and a biased gradient. While it works well empirically, the paper lacks a theoretical analysis of this gradient bias or a discussion of its potential impact. *Please correct me if I'm wrong on this!*

### Questions
- Why can a user win an auction and then spend up to their budget? Shouldn’t their bid always be capped by the budget?

- Explain the depletion step more and why it is critical to the analysis? Esp compared with just capping bids by the budget.

- In Figure 2, do cumulative costs and rewards converge? Does this imply 0 revenue?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces an RL based method to optimize budget constrained advertising campaign when the auction is second price. 
The algorithm relies on the observations that: the system has access to information that allows for counterfactual estimation; "fluid" approximation is possible: one can compute derivative with respect to the bid multiplier because of the scale involved. 
The algorithm is tested against RL baselines on auction Net data and shows superior performances.

### Strengths
The algorithms displays superior performance against RL baselines.
The idea of dividing the timeline into different steps allows for counterfactual estimation. 
The methodology is simple and could be generalized to other settings.

### Weaknesses
- the fact that the highest bid and the value are observed should be explained. Is it because the platform sees everything and the value is derived from a machine learning prediction? Related to that, the experimental setup should be better explained: what is the data made off (columns of the datasets, etc...). 
- is RL the right tool? I am surprised the experiment does not include other baselines such as PID, basic heuristics and no regret approaches 
- the paper does not provide any theoretical contribution, and it feels like it is overcomplicating some aspects (for example, invoking Myerson's lemma for the second price auction).
- link between J and the derivative of its approximation would be appreciated
- Looking at the experimental setup it is not clear if the problem is  MARL or RL because the authors mention multiple players.
- The realism of the assumption should be better discussed, for instance,

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2
