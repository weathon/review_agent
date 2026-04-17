# Incentivizing Truthfulness in Fully Decentralized Learning with Guaranteed Accurate Convergence

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Decentralized learning has gained significant attention due to its advantages in scalability, privacy, and fault tolerance. In this paradigm, multiple agents collaboratively train a global model by exchanging parameters only with their neighbors, without the assistance of a centralized server. However, a key vulnerability of existing decentralized learning approaches is their implicit assumption that all agents behave honestly during gradient updates and information sharing. In real-world scenarios, this assumption often breaks down, as selfish or strategic agents may be incentivized to manipulate gradients or share false information for personal gain, ultimately compromising the final learning outcome. In this work, we propose a fully decentralized payment mechanism that, for the first time, guarantees both truthful behaviors and accurate convergence in decentralized stochastic gradient descent algorithms. This represents a significant advancement, as it addresses two major limitations of existing truthfulness mechanisms for collaborative learning: 1) reliance on a centralized server for payment collection, and 2) the tradeoff between ensuring truthfulness and maintaining convergence accuracy. In addition to characterizing the convergence rate under convex or strongly convex conditions, we also prove that our approach guarantees the cumulative gain that an agent can obtain through strategic behavior remains finite, even as the number of iterations approaches infinity—a property unattainable by most existing truthfulness mechanisms. Experimental results on several machine learning applications confirm the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new payment scheme to incentivize honest gradient reporting in decentralized SGD. For agents that only care about the quality of the learnt model on their own distribution,  the proposed mechanism adresses two kinds of "attacks": Noise injection and gradient inflation. It works by having agents compare their squared gradient norms, and the agent with the larger one making a payment proportional to the difference. Unlike prior works, the proposed mechanism does not require a central server, is budget-balanced, and is claimed to yield similar convergence guarantees to the fully honest case, even without convexity assumptions.

### Strengths
- The paper tackles an interesting and relevant problem
- The empirical results seem to show that the propsed mechanism succeeds
- The writing is mostly easy to follow

### Weaknesses
- **Main issue**: The proof of Lemma 1 seems to be highly incomplete (and the statement is likely wrong at the level of generality the papers considers). The proof appears to only show that an agent that cares about model quality at time t+1 is not incentivized to manipulate gradients at time t, but this does not seem to establish that the same is true for agents that care about the model quality at time T+1 (which is what determines utility). 
   - It seems like some of the improvements over prior work, such as the extension to non-convexity are only possible due to this incomplete treatment of delayed effects. 
- Some of the claimed contributions are unclear: The paper claims that various prior works do not achieve eps-incentive compatibility and accurate convergence, but does not provide much support for the claim. In a similar vein, it is unclear whether the extension to the non-convex case is due to an improvement in the mechanism, or refined analysis.
- Relatedly, the discussion mostly focuses on connections to VCG mechanisms, but the  proposed mechanism appears to be heavily inspired by Chakarov et al. Specifically, in a fully connected graph, the proposed mechanism seems to be equivalent to Chakarov's up to rescaling. 
- As in Chakarov et al, the attack model is somewhat restricted. In particular without the seemingly arbitrary assumption of $\alpha\geq 1$, it seems like under the proposed scheme some agents could be incentivized to report zeros for the gradients in order to collect payments, while "free-riding" on other players' gradient information. 
- Nitpicks
  - The notation for $\lambda$ is overloaded (step size and Eigenvalues)
  - The notation for the mechanism could be simplified by allowing for negative payments and simply setting agent i's payment to agent j as the difference in squared gradient norms.

### Questions
- Can you elaborate in more detail, how most of the cited mechanisms are based on VGC? I am no expert on this, but after looking at the definition, the connection seems unclear for some works like Chakarov et al.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduced the incentive mechanism for decentralized learning, so that all nodes send the true stochastic gradient and do not skew it to maximize their reward. Then, in Theorem 1, this paper shows that using the proposed method, the optimal strategy to maximize the reward is sending the true gradient.

### Strengths
* The paper is well written and easy to follow. The reason to introduce the incentive into decentralized learning is clear.
* Roughly speaking, in Theorem 1, this paper shows that using the proposed method, the optimal strategy to maximize the reward becomes that nodes send the correct gradient. This result fits the motivation explained tin he introduction.

### Weaknesses
* The reviewer feels that introducing the incentive mechanism for decentralized learning is interesting, but how can we guarantee that the nodes follow this mechanism? For instance, the set of actions, Eq. (3), seems to be limited, and there would be other choices. For instance, ignoring the parameters received from neighboring nodes in Eq. (2) or increasing $w_{ii}$ and decreasing $w_{ij}$ would also be possible to increase the reward. Besides this, there is also some degree of freedom in the choice of reward function, $R_i$, and nodes can use different reward functions. How can we guarantee that all nodes use the same reward function? The statement of Theorem 1 only holds when nodes follow this setting. The reviewer feels that there remains a possibility that nodes can work maliciously to maximize the accuracy of their dataset. This is my main concern for this paper.
* It is a bit hard to understand the intuition of Eq. (4). Why do we need to minimize $\sum_t P_{i,t}$? As far as I understand, this paper considered the case where each node tried to minimize its own loss function. Thus, it is natural that Eq. (4) is designed so that Eq. (4) increases as the reward increases. However, it is unclear why Eq. (4) is designed so that Eq. (4) increases as $\sum_t P_{i,t}$ decreases.
* In Theorem 1, the dependence on other parameters, $L_f, H, \sigma, \rho$, is hidden in the convergence rate. Can the authors show the convergence rate in Theorem 1 more precisely?
* I think it is not a critical weakness, but assuming that $f_i$ is Lipschitz continuous is a bit stronger compared with the many decentralized learning papers. It should be clarified.

### Questions
See the weakness section.

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
4

### Summary
This paper proposes a fully decentralized payment mechanism that guarantees both truthful behavior and accurate convergence in decentralized stochastic gradient descent algorithms. The theoretical analysis appears sound; however, my main concern is that several key assumptions are unrealistic. I will reevaluate the paper after considering the authors’ rebuttal.

### Strengths
This paper proposes a fully decentralized payment mechanism that guarantees both truthful behavior and accurate convergence in decentralized stochastic gradient descent algorithms. The theoretical analysis appears sound; however, my main concern is that several key assumptions are unrealistic.

### Weaknesses
My detailed comments are as follows:


1.	My primary concern is that, although agents are allowed to behave strategically, the authors assume they will truthfully follow the proposed payment mechanism. This assumption seems unrealistic — rational agents motivated by self-interest would likely attempt to exploit or manipulate the mechanism to maximize their own benefit. The authors should incorporate additional mechanisms or safeguards to prevent such behavior, rather than relying on an unrealistic assumption of honesty.


2.	In Definition 2 and Lemma 1, the main results rely on the assumption that the neighbors of agent $i$ are truthful. This is impractical, as agent $i$ cannot ensure or verify the honesty of all its neighbors in a decentralized setting. The authors should consider more robust assumptions that better reflect real-world conditions.


3.	In Assumption 1, the authors assume that the loss function is Lipschitz continuous. However, this assumption conflicts with the strongly convex setting, rendering the corresponding results questionable. Moreover, the assumption is overly strong and fails to account for data heterogeneity. The authors should adopt more realistic assumptions that explicitly consider data heterogeneity, as commonly used in the literature.

### Questions
N/A

### Soundness
2

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
4

### Summary
This paper addresses a core problem in fully decentralized learning: how to incentivize truthful information sharing among selfish or strategic agents. Specifically, the authors consider a scenario where agents can manipulate their gradients (e.g., by scaling them) to bias the model. They propose a novel payment mechanism where agents with larger gradient norms must pay agents with smaller norms. The authors provide theoretical proofs showing that by carefully designing the decay rate of the payment coefficient ($C_t$), the optimal strategy for agents converges to "full truthfulness" over time. They also prove that DSGD converges even in the presence of this (diminishing) strategic manipulation.

### Strengths
* To the reviewer's knowledge, this is the first work to propose a fully decentralized payment mechanism to address truthfulness in this setting, which is a significant contribution.
* The theoretical results, particularly the the convergence (despite manipulation) and finite cumulative gain, provide strong theoretical support for the proposed method.
* The experimental results align with the theory, demonstrating that the presence of the payment mechanism effectively mitigates the drop in test accuracy caused by strategic manipulation.
* The paper is well-written, clear, and provides a thorough background on the problem.

### Weaknesses
1.  The proposed payment mechanism may inadvertently incentivize free-riding.
    The reviewer understands that this paper focuses on the "Truthfulness" problem—preventing *active* participants from manipulating gradients for personal gain. However, it does not address (and may worsen) a related and important incentive problem: free-riding.
    Consider a free-rider agent $i$ that submits a zero-gradient (or a gradient with a very small norm), i.e., $m_{i,t} = 0$. According to Mechanism 1, if this agent's honest neighbor $j$ computes a valid, non-zero gradient, then $||m_{j,t}|| > ||m_{i,t}||$. This forces the honest agent $j$ to *pay* the free-riding agent $i$. Consequently, the mechanism does not penalize this "passive" non-participation and, in fact, provides a direct financial incentive for agents to submit low-norm gradients.

2.  The payment mechanism appears vulnerable due to its simplicity.
    The mechanism's reliance *only* on the reported gradient norm makes it susceptible to strategic decoupling. The paper's core “implicit” assumption is that agents will honestly report the norm $||m_{i,t}||$ for the payment calculation. A strategic agent can separate its update from its payment. For example, an agent $i$ could use a highly manipulated gradient $m_{i,t}$ in its local update (Eq. 2) to influence its $\theta_{i,t+1}$ (which is then shared). Simultaneously, to avoid payment, it could falsely report a "reasonable" norm $||m'_{i,t}||$ (e.g., a norm similar to its neighbors' or its own true gradient) to the payment mechanism. In this scenario, the agent gains the benefit of manipulation without incurring the cost.

### Questions
1.  Clarity on Convergence Rate Comparison:
    The main convergence results are presented in Big-O notation, which obscures some constants and makes direct comparison difficult. Theorem 1 shows a convergence rate of $\mathcal{O}(T^{-(1-v)})$ for the non-convex case, where $v \in (1/2, 2/3)$. This rate (between $\mathcal{O}(T^{-1/3})$ and $\mathcal{O}(T^{-1/2})$) appears slower than the $\mathcal{O}(1/\sqrt{T})$ rate achievable by standard DSGD (e.g., Koloskova et al., 2020). For a clearer comparison, could the authors provide a convergence rate in a form that is more directly comparable to standard DSGD rates, such as Theorem 2 in Koloskova et al. [1]?

2.  Impact of Data Heterogeneity on Honest Agents:
    The paper's experiments rightly suggest that high data heterogeneity *increases* the incentive for gradient manipulation. However, this raises a question about the mechanism's fairness. In a highly non-IID setting, an *honest* agent's true gradient norm $||g_{i,t}||$ could naturally be an order of magnitude larger or smaller than its neighbors' norms, simply due to different data distributions.
    Does Mechanism 1 risk misinterpreting this "natural difference" as "strategic manipulation"? Could this lead to the mechanism unfairly penalizing honest agents whose data distributions are simply unique or outliers compared to their neighbors?

### References

[1] Koloskova et al. A Unified Theory of Decentralized SGD with Changing Topology and Local Updates. ICML 2020.

### Soundness
3

### Presentation
3

### Contribution
3
