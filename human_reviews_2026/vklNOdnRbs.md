# COBRA: Contextual Bandit Algorithm for Ensuring Truthful Strategic Agents

- Decision: Reject
- Scores: 0, 2, 6, 6

## Abstract
This paper considers a contextual bandit problem involving multiple agents, where a learner sequentially observes the contexts and the agents' reported arms, and then selects the arm that maximizes the system's overall reward. Existing work in contextual bandits assumes that agents always truthfully report their arms, which is unrealistic in many real-life applications. For instance, consider an online platform with multiple sellers; some sellers may misrepresent product features to gain an advantage, such as having the platform preferentially recommend their products to its users. To address this challenge, we propose an algorithm, COBRA, for contextual bandit problems involving strategic agents that disincentivize their strategic behavior without using any monetary incentives, while having incentive compatibility and a sub-linear regret guarantee. Our experimental results also validate our theoretical results and the different performance aspects of COBRA.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
**Disclaimer**: I reviewed this same paper for NeurIPS 2025. The main reason that this paper gets rejected from NeurIPS 2025 is because there seem some fundamental flaws in its approaches and proofs and some explicit proof bugs were also identified (see my detailed description below). There was a very long technical discussion during the rebuttal phase of NeurIPS 2025, but ultimately the discussion did not resolve the technical issues/bugs identified by the reviewers there. The ICLR 2026 draft did a bit re-writing, e.g., by re-phrasing some assumptions into a “LOOM-compatible” definition, but the rephrasing did not seem to change the earlier issues and just make it appear differently. 

This paper develops a new algorithm for strategic contextual bandit problem where each arm is a strategic agent and can misreport their feature vector, though cannot manipulate reward. This problem was proposed and studied by Kleine Buening et al. NeurIPS'24, but the current paper develops a different algorithm inspired by VCG. Specifically, the paper proposes to use all agents' information, excluding arm a's, to estimate a reward function for agent a (in linear bandit case, this means estimate the theta parameter for a). The paper argues that this, coupled with an optimistic-pessimistic inequality called LOOM condition, can help to induce incentive compatibility.

### Strengths
Interesting research question.

### Weaknesses
The paper’s writing has some clarity issues. For instance, in the definition of “LOOM-compatible contextual bandit algorithm”, it is unclear why “any contextual bandit algorithm” would always have an “estimated function”. We know UCB has a natural estimate function of its reward, but if we do Thomas Sampling, do we call that updated distribution mean as the “estimate”? What’s the definition of “estimated function”? How do you ensure that “any” bandit algorithm will have an “estimated function”? 

What is the difference between Theorem 1 and Theorem 4?  

The above are some smaller technical issues I found, but **my biggest concern is that there seem major flaws in the proposed approach, as well as in the technical proofs,** which makes the paper not acceptable. In particular, the paper's approach of using all other agent's information to estimate a parameter and reward for arm a could not work for linear contextual bandit. The no regret proof for standard linear context bandit, which this paper adapts from, crucially depends on a "self-normalization lemma", which roughly says if we use data along some direction d to estimate parameter theta, then in the future when we see feature roughly along direction d, our reward estimation error will be small. However, this paper's approach of using other arms' information to estimate i's arm reward basically excluded the possibility of such "self-normalization".

**A potential counterexample**  Let us consider the standard multi-armed bandit as a special case of linear contextual bandit. This is a special case because we can view x_{a,t} as always along the canonical basis e_a direction, with rescaling factor as agent’s private information at each round t. In such special cases, LinUCB algorithm effectively becomes estimating the parameter theta_a (a'th dimension of theta) along direction e_a separately and independently using arm a's previous information. This can be verified by plugging in x_{a,t} as a scaled version of e_a into LinUCB descriptions in Line 300 to 306. However, using this paper's proposed approach, as explained in LOOM Condition in Equation (2) and algorithm description of COBRA, COBRA will use information orthogonal to direction e_a to decide whether e_a’s report is correct or not. This simply is impossible since the information from any arm other than a will not be useful for estimating \theta_a. Concretely in Equation (2), the LCB for an arm a, estimated used all other arms orthogonal to a, will always much larger than the UCB on the right hand side, and will never change over the execution of the algorithm. 

The paper’s technical writing is a bit difficult to follow, but I tried to check the detailed proof to see where concrete proof bugs may appear in the paper. I think one concrete bug is the following, inherently due to the issue I mentioned above. 

 In the proof of Theorem 4 in the Appendix, I think the inequality from Line 1102 to Line 1105 is incorrect. The paper claims that a Det/Det term is always upper bounded by a constant C. This is not correct. In the standard multi-armed bandit special case where arms are canonic basis vectors, this term is lower bounded by the number of times arm a is pulled, which can be \Omega(T) and cannot be upper bounded by a universal constant C. That makes the regret linear.

Another thing that strikes me as odd when diving deep into the proof of Theorem 4 is that I don’t see where they use the LOOM strategy of banning a misreported arm at all. If the proof were correct, they could arrive at the regret bound in equation (18) without ever using an argument about the NE and the regret bound would hold for ANY arm strategy?! I might be missing something here, but otherwise this obviously can’t be true (or their assumptions are overly strong so that there is generally no point in doing mechanism design in their setting).

### Questions
Feel free to respond to my concerns about proof bugs above. I am happy to be convinced and revise my ratings, but currently I do not see a way to overcome the intrinsic issue of the approach.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper considers a contextual bandit problem with strategic arns who misreport contexts to maximize their number of selections over $T$ rounds. The authors propose a VCG-inspired algorithm that uses optimistic and pessimistic estimates of each agent’s reward to eliminate misreporting agents. It is shown that under the proposed algorithm truthfulness is an approximate Nash equilibrium and the authors establish sublinear regret bounds given that the arms play in NE. The authors also provide empirical results on synthetic problem instances to support their theoretical claims.

### Strengths
1. The studied problem is interesting, and the intersection of online regret minimization under uncertainty with mechanism design is a challenging but interesting domain. 
2. The authors motivate the model and the work well.

### Weaknesses
I have concerns about the correctness of Theorem 4. Firstly, there a various typos in Appendix B.2.2 which make the proof of Theorem 4 hard to read. For example, Lemma 4 has various typos and it is unclear  what $a_a$ is, what the $x$ in the definition of $UCB_{t, -a} (x_{s, a_a})$ in line 996 is, etc. 

Following this, I am confused about line 1067 in the proof. You plug-in Lemma 4, but what is the $x$ in $\lVert x \rVert_{V_{t, -a}^{-1}}$. As far as I can tell, this $x$ should be $x_{t,a}$. However, then bounding $\sum_{t=1}^T \lVert x_{t,a} \rVert_{V_{t,-a}^{-1}}$ is hard and I believe that you shouldn't be able to bound this the way you do. Consider the case where the context vectors of arm $a$ are linearly independent of the context vectors of all other arms $a'$ (for all time steps). Then, there is no good bound for the sum over $\lVert x_{t,a} \rVert_{V_{t,-a}^{-1}}$. In other words, if you are only using every other arm's data, your exploration bonueses can stay arbitrarily large. I have difficulties following the reasoning on the top of page 21 (partly due to typos), but I suspect that the issue is there. 

Another sign that something is possibly wrong is that you arrive at your regret bound (18) on page 21 without ever using LOOM or any guarantee about the arm strategies. What would be the point of mechanism design when you can get the same regret guarantee without using that the arms play a NE under COBRA? 

Other weaknesses: 
- The presentation could be improved, and particularly Section 4 is difficult to read as it is extremely dense. Related to the issues in the presentation, you introduce COBRA(TS) and then present your theoretical guarantees as if they also hold for COBRA(TS). As far as I can tell you only prove results for COBRA(UCB); see e.g., proof of Theorem 2. It is often quite unclear what algorithm you are referring to when you just write COBRA.

I'd be happy to increase my score if my concerns can be resolved.

### Questions
1. In line 131, you say that the previous work's [1] method may not be practical when the true reward function is unknown. Could you please be precise about what you mean by this? The algorithm in [1] appears to be designed specifically for the case where the true reward function is unknown. 
2. Line 336: It should be COBRA(TS) instead of COBRA(UCB). 

[1] Thomas Kleine Buening, Aadirupa Saha, Christos Dimitrakakis, Haifeng Xu; Strategic Linear Contextual Bandits, NeurIPS 2024

### Soundness
1

### Presentation
2

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
This paper considers incentive-copmatible contextual bandits, where one learner is interacting with strategic agents controlling the arms' contexts.

After observing the context $c_t$, the learner selects $a_t$, resulting in a stochastic reward with mean $f(x_{t,a_t})$ where $x_{t,a} = \phi(c_t, a) \in R^d$ is the feature vector associated with $c_t$.
It is unclear, but I assume that neither $c_t$ or $\phi$ are known to the agent.

A linear version of this problem was previously studied by Buening et al, but the techniques used in this paper are substantially different.

The main idea is to test if agents are over-reporting and exlude them. The threat of exclusion provides the incentive. For that reason they define the following estimates:  $f_t$, which is based on everybody's estimate, and $f_{t,-a}$ which excludes $a$'s reports.

The algorithm *could* be combined with a large class of context bandit algorithms. Of course, this requires carefully looking whether various conditions are satisfied. This is captured in Assumption 1 where they asume that:
  (a) $f(x) \leq UCB_{t,a}(x)$,
  (b) $UCB_t(x_{t,a}) \leq UCB_{t,-a}(x_{t,a})$
where
$UCB_{t,a}(x) = f_{t,a}(x) + \epsilon_t$
$UCB_{t,-a}(x) =  f_{t,-a}(x) + \epsilon_{t,-a}$,
with $\epsilon$ denoting appropriate confidence intervals around estimates.

The assumption is examined page 30. Case 2, where one agent over-reports, is the basic scenario. But given, how central this assumption is to the proofs, this is really an inadequate proof for me. I guess the $-a$ interval should be wider than the $a$ one, but this should be proven more rigorously. 

The other quantity is $LCB$, which structured differently:
- $LCB_{t,-a} = f_{t,-a} - \epsilon_{t,-a}$.
- $LCB^{(x)}_{t,a} = \sum_{s=1, a_s = a}^t LCB_{t,-a}(x_{s,a_s})$,
i.e. it is the sum of lower bounds through other agents reports, where $x$ now defines a sequence of rewards (confusingly).

This is complemented with $UCB^{(y)}$, an upper bound on the total reward which does *not* use the contexts. Perhaps this notation is a bit counterintuitive. In any case the  LOOM condition can be summarised as follows:

If the lower bound, constructed through context-dependent estimates using agents reports, exceeds the upper bound constructed only through observed rewards from agent $a$, then $a$ is over-reporting.

### Strengths
+ The main idea is nice and intuitive
+ The results are an improvement and extension of previous work

### Weaknesses
- The presentation could be improved. I spent more time to understand what is going than I should have had to. 
- Assumption 1 can be quite restrictive. It should at least be cleanly proven for some special cases, but the discussion in Appendix D is inadequate. Intuitively, it should hold for the linear case based on the reasoning given.

### Questions
? The paper also says it is inspired by VCG, but the connection is not
spelled out. I assume it is because $f_{t,-a}$ uses the reports of the
remaining agents.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses contextual bandit problems where strategic agents may misreport their features to maximize their selection probability. The authors propose COBRA (Contextual Bandit Algorithm for Ensuring Truthful Strategic Agents), which uses a Leave-One-Out-based Mechanism (LOOM) inspired by VCG mechanisms to detect and disincentivize misreporting. The authors propose that reporting arm features truthfully is the best/dominant strategy for the agents which is achieved via LOOM and COBRA. Experimental results validate their theoretical results of sublinear regret.

### Strengths
1) Misreporting in contextual bandits is clearly motivated (food delivery/marketplace settings) and is an interesting/practically relevant problem.
2) LOOM provides a theoretically grounded, drop-in mechanism compatible with common contextual bandit algorithms.
3) The proofs in the appendix are well structured.

### Weaknesses
1) Only synthetic evaluation. Considering that real world applications are well motivated and reiterated throughout the paper it would have been nice to see some experiments on real world data. 
2) The scale of the synthetic experiments is quite small as well. Having just 5 agents with only one of the agents over reporting (line 465) is a bit unsatisfactory in terms of scale. It would be better to see experiments on a larger scale particularly larger $d$ and $N$ than those found in the appendix, and with more than one over reporter.
3) A limitations sections would benefit the paper. The term dominant strategy is only applicable for the case of over reporting and can be misleading considering the paper assumes that collusion and under reporting cannot happen (which are strong assumptions in their own right). The failure conditions of LOOM should also be mentioned in the main paper instead of being scattered around in remarks and in the appendix. 
4) The complexity of LOOM + COBRA should be discussed, especially in the case of agents with multiple arms. It would be great if the authors could go into more detail about how their algorithm scales to agents with multiple arms instead of the short paragraph in the appendix to determine application feasibility.

### Questions
Questions: 
1) Modern applications including the ones motivated in the paper routinely handle thousand to millions of items at once. What is the computational complexity of LOOM+COBRA? Considering this and the dependence of the regret on $\sqrt{N}$, do you believe it is applicable to this scenario ?
2)  How does the complexity change if each agent picks multiple arms?
3) Would it possible to add a semi synthetic or small real world experiments at a larger scale $(d \ge 50, N\ge 50$, more than one over reporter) than those in the appendix
4) See 3) from Weaknesses.

Minor Corrections/Presentation issues: 
The paper has few typos and grammatical errors   
- line 103: non-leaner --> non-linear 
- line 113,118: study an --> studied an 
- line 239: "Note that the assumptions underlying contextual bandit algorithms need to satisfy in our setting" (should be rephrased) 
- line 336: "propose a TS-based variant COBRA(UCB)" should be COBRA(TS) 
- line 373: drive --> derive and quite a few more in the appendix. 
 
Additionally it would be great if the authors could state/explain the notation used in Table 1 of section C of the appendix, in the paper itself. I would also prefer if the core equations were set on their own display lines rather than wrapping across text lines.

### Soundness
3

### Presentation
2

### Contribution
2
