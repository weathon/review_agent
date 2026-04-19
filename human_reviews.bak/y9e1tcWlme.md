# Tackling Decision Processes with Non-Cumulative Objectives using Reinforcement Learning

- Decision: Reject
- Scores: 6, 6, 3, 5

## Abstract
Markov decision processes (MDPs) are used to model a wide variety of applications ranging from game playing over robotics to finance. Their optimal policy typically maximizes the expected sum of rewards given at each step of the decision process. However, a large class of problems does not fit straightforwardly into this framework: Non-cumulative Markov decision processes (NCMDPs), where instead of the expected sum of rewards, the expected value of an arbitrary function of the rewards is maximized. Example functions include the maximum of the rewards or their mean divided by their standard deviation. In this work, we introduce a general mapping of NCMDPs to standard MDPs. This allows all techniques developed to find optimal policies for MDPs, such as reinforcement learning or dynamic programming, to be directly applied to the larger class of NCMDPs. Focusing on reinforcement learning, we show applications in a diverse set of tasks, including classical control, portfolio optimization in finance, and discrete optimization problems. Given our approach, we can improve both final performance and training time compared to relying on standard MDPs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a decision-making framework that allows for the optimization of non-cumulative objectives, by mapping such objectives to the regular cumulative formulation, such that optimizing the cumulative objective results in the optimization of the non-cumulative objective.  The authors show that this MDP framework satisfies important theoretical properties that make it possible to find an optimal policy, and provide several empirical experiments in support of this.

### Strengths
This paper provides a theoretically-sound and easy-to-use framework that makes it possible to optimize non-cumulative objectives, by mapping such objectives to the regular cumulative formulation, such that optimizing the cumulative objective results in the optimization of the non-cumulative objective. This framework clearly has several potential use-cases in various domains. The paper is polished, well-written, and the technical concepts are explained in a relatively clear manner.

### Weaknesses
I have two concerns regarding this paper:

The first of which is that while the authors do a good job explaining how their framework can be used, they fail to properly formalize the theoretical conditions that they are operating in. For example, they do not address whether the state and action-spaces are or can be finite/infinite, if there is a restriction on the type of policy that can be used (stationary, stochastic, etc.), as well as whether there are implications of their framework being used in off-policy settings. Because of this, the extent to which their framework can be utilized is unclear.

My second concern is related to the experiments. In particular, the comparison of non-cumulative objectives to the cumulative objects seems unintuitive, given that no-where in the paper do the authors claim the non-cumulative objectives are better than the cumulative objectives. Rather, it would be more intuitive, and consistent with the rest of the text, if the experiments showed that the proposed methods indeed are able to find the optimal policy for the non-cumulative objective. As such, it would appear that the experiments in the current draft of the paper do little to support the claim that the proposed methods can be used to optimize non-cumulative objectives.

As such, I am open to increasing my score if the authors can properly formalize the theoretical conditions that they are operating in, and either 1) the authors can convince me that the experiments in the current draft are adequate, or 2) the authors improve the existing experiments in such a way that they address my concerns, or 3) the authors can provide additional experiments that address my concerns.

### Questions
1) What are the off-policy implications of this method?
2) A regular average would appear to be a common and useful non-cumulative objective to include in the paper. Was there a reason that it was not included?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors consider the problem of control in finite-horizon Non-cumulative MDPs (NCMDPS). They propose a general mapping of NCMDPs to standard MDP, and hence the problem can be solved by existing standard RL methods. They use a series of numerical experiments to illustrate the NCMDP problems and the proposed transformations.

### Strengths
* The authors consider the problem of control in NCMDPs, which is important and understudied in the RL community.

* The proposed mapping of NCMDPs to standard MDPs is simple and intuitive, and can effectively solve many problem instances of NCMDPs.

* The empirical studies are concrete, showing both the necessity of considering the problem of control in NCMDPs and the effectiveness of the proposed mappings.

### Weaknesses
* The paper has technical flaws, i.e., the function $f$ is ill-defined. In the original problem statement, $f$ is defined as a function on $\mathbb{R}^T$. But later on, the authors also use notations like $f(r_1,...,r_t)$, where $f$ should be treated as a function on $\mathbb{R}^t$. I think it may be helpful to define $f$ as a function of a set ($f(\\{r_1,...,r_t\\})$, $\\{r_1,...,r_t\\}$ is the set of $t$ rewards, $t$ can be any integer between $1$ and $T$). This works for all problem instances in Table 1.

* The authors overstate their contributions. They claim their methods work for arbitrary functions $f$. However, as indicated in the previous comment, Definition 1 is invalid for general $f$ on $\mathbb{R}^T$. Even if $f$ is treated as a function of a set, it turns out only for certain choices of $f$ there exist fix-size $h_t$ and corresponding functions $u$ and $\rho$.  I think the authors should try to find a classification of functions f with constant-size state adaptions. Even some preliminary results such as sufficiency conditions or necessary conditions would be helpful.

* I think in empirical studies the authors should compare their methods with prior works on the control of NCMDPs with a specific choice of objectives and I am curious if the proposed method (which is more general) can produce comparable performances.

### Questions
* Can the proposed mapping of NCMDPs to MDPs be generalized to infinite-horizon MDPs? (when $f$ is defined as a function of set.)

* It seems that if we transformed an NCMDP to a standard MDP, the resulting standard MDP is governed by the dynamics of $s_t$ and $h_t$ can be viewed as an auxiliary state. I am curious whether there are some connections with the framework called MDPs with latent dynamics (see [1], [2]).

[1] Simon Du, Akshay Krishnamurthy, Nan Jiang, Alekh Agarwal, Miroslav Dudik, and John Langford. Provably
efficient RL with rich observations via latent state decoding. In International Conference on Machine
Learning, 2019.

[2] Amortila, Philip, et al. "Reinforcement Learning under Latent Dynamics: Toward Statistical and Algorithmic Modularity." arXiv preprint arXiv:2410.17904 (2024).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper studies the problem of finding an optimal policy for a special class of decision problem called non-cumulative Markov decision process (NCMDPs) where instead of the sum of rewards, it aims to maximize the expected value of an arbitrary function of the rewards. The paper solves NCMDPs by mapping it to standard MDPs, allowing direct application of MDP solver for NCMDPs. It also performs numerical experiments of classical control, portfolio optimization, and discrete optimization that use NCMDP objectives, and shows that their method improves both training time and final performance compared with standard MDP solvers with cumulative reward.

### Strengths
- The NCMDP setting considered in this paper fits many applications that does not directly fit into the MDP setting with cumulative rewards. Some examples include the weakest-link problem in network routing which maximizes minimum reward, the Sharpe ratio in finance which maximizes the mean divided by standard deviation. 
- The paper provides a straightforward solution to NCMDPs by first map an NCMDP to a standard MDP, which allows direct application of black box MDP solvers. 
- It provides comprehensive empirical results applying this method to a range of environments with non-cumulative reward objective, including classical control, portfolio optimization with Sharpe ratio, discrete optimization with lost cost objective.

### Weaknesses
- The mapping from NCMDP to MDP provided in equations (3)-(5) augments the state with $h_t$ that represent necessary information for reward history, and is updated as $h_{t+1}=u(h_t,\tilde{r}_t)$ and satisfies $r_t=\rho(h_t,\tilde{r}_t)$. For arbitrary reward function $f(r)$, an essential factor to ensure the mapping to MDP is of reasonable size is to find an efficient functional form of $u$ and $\rho$ that summarizes this information from reward history. However, the paper only shows a list of examples of special examples of $f$ and does not specify how to construct $u$ and $\rho$ for arbitrary function $f$ in general, and does not provide an bound to the resulting state dimensions, which at worst case must contain the set of all historical rewards. I think this makes the overall contribution of the paper limited. 
- In the experimental section, for each environment, the paper compares their method of MDP reduction (with modified state and reward) with an RL agent with cumulative objective. This does not seem like a very fair comparison given the cumulative objective will be the wrong objective in each of these environments, and it is clear that correct specification of the objective will lead to better performance.

### Questions
- The paper gives very specific examples of construction of $u$ and $\rho$ that allows an efficient reduction from NCMDP to MDP of small augmented state size. Can you give examples of more general classes of non-cumulative objective function $f$ that induces $h_t$ that is of reasonably small dimensions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
In traditional MDPs, the agent is tasked with the optimisation of the policy value function that is the expected sum of the rewards. In a non cumulative markov decision process, the value function is replaced by the expected value of an arbitrary function of the rewards. 
The authors propose a method to translate an NCMDP into a regular MDP and show the expected returns of the two are equal and apply MDP solving methods to NCMDPs.

### Strengths
Definition 1 and the possible applications of the method are interesting.

### Weaknesses
The variables u, h and $\rho$ are discussed in the paragraph following the statements of equations 3,4,5. You should introduce them before.
Also, their description is too vague. For example "This can be achieved by extending the state space with ht , which preserves all necessary
information about the reward history" does not give me a good sense of what the function u should be. 
I suggest the statement of theorem 1 be rearranged to: assumptions then conclusion, instead of the current: assumption then conclusion then more assumptions.

### Questions
It seems to me that if the update function u of definition 1 is a constant function, no information is preserved and the constructed MDP could not have the same value function as the NCMDP. Is this right? If it is, this sort of choice should be excluded in definition 1.
Later in the same paragraph, the authors note that an h_t recording all reward history is sufficient to construct an MDP but the state space is exponentially large so there must be some tradeoff between state space and "Markovness". How much information can be lost from the full history before we lose the Markov condition? Is there a characterisation of the functions u that satisfy this property?

### Soundness
3

### Presentation
3

### Contribution
2
