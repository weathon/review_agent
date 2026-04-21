# Social Bayesian Optimization for Building Truthful Consensus

- Avg Score: 5.25
- Decision: Reject
- Scores: 5, 5, 6, 5

## Abstract
We introduce *Social Bayesian Optimization* (SBO), a query-efficient algorithm for consensus-building in collective decision-making. In contrast to single-agent scenarios, collective decision-making encompasses group dynamics that may distort agents' preference feedback, thereby impeding their capacity to achieve a truthful consensus. We demonstrate that under standard rationality assumptions, reaching truthful consensus—the most preferable decision based on the aggregated latent agent utilities—using noisy feedback alone is impossible. To address this, SBO employs a dual voting system: cost-effective but noisy public votes, and more accurate, though expensive, private votes. We model social influence using an unknown social graph and leverage the dual voting system to efficiently learn this graph. Our findings show that social graph estimation converges faster than the black-box estimation of agents’ utilities, allowing us to reduce reliance on costly private votes early in the process. This enables efficient consensus-building primarily through noisy public votes, which are debiased based on the estimated social graph to infer truthful feedback. We validate the effectiveness of SBO across multiple real-world applications, including thermal comfort optimization, team building, travel destination discussion, and strategic alliance in energy trading.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper studies an online setting of learning the most favorable candidate among agents. The utility of the agents are unknown and need to be elicited from query on binary relations between candidates. Moreover, agents reported preferences may be influenced by their neighbors on an unknown social network. While no non-dictator aggregation rule can reach the correct candidate under the influence of noisy preference, this paper circumvent the barrier by introducing extra costly yet truthful query. These query can reveal the social influence and therefore infer the true preferences from the noisy queries. There theoretical results bound the regret of their algorithm. They also run experiments to demonstrate the robustness efficiency of the algorithm on synthetic and real-world data.

### Strengths
The paper studies an important question under a highly non-trivial model. While many paper in our literature adopt the utility model to analyze the welfare, it is always worth a question whether people really know quantitively about their utility.  The paper also well motivates why this problem is non-trivial and why the challenges are not addressed by the previous work. 

The research has a clear and complete structure. The introduction motivates the significance and non-triviality of the problem. Impossibility results distinguish their contribution from previous work. A constructive contribution (the algorithm) is proposed with both theoretical guarantee and empirical tests.

### Weaknesses
However, I have doubt on the soundness of theoretical results in this paper. I check the proof Theorem 3.3 and Theorem 3.9 and the statement of Theorem 3.12, and have questions on all of them. 

Theorem 3.3
For the |X| = 2 (binary decision) case, the proof directly show that any aggregation rule is dictatorship, which is very counterintuitive. Following Arrow's definition, an aggregation rule A is dictatorship if [there exists an agent i such that for any alternative (x, x') and any utility profile u, x >_i x' iff x >_{A(u)} x'}]. Now just consider A be the majority rule (one two candidates). Then any agent i cannot be a dictator when he/she is on the minority. (I think it is more reasonable to show that non-dictatorship and group-thinking-proof can not exist together.)

For the |X| > 2 case, the proof restrict the result to a subcase where every two agents have distinct favorite candidate. This directly requires |X| > |V| (the number of candidates is larger than the number of agents), which is very restrictive (especially in voting scenarios). Therefore, the theorem claims more than it actually proves. (On the other hand, I doubt whether it is necessary to have such restriction.)

Theorem 3.9
Lemma F.3 is very counterintuitive. Firstly, w1 and w2 is not defined. More importantly, I don't think this inequality holds without strong     assumption unlisted. For example, when u(x) = 0,  u(x') < 0, and w1 < w2, lower{w}(u(x) − u(x′)) ≤ w1u(x) − w2u(x′) is apparently violated, as 0 > lower{w}  * u(x') > w2 * u(x'). The other inequality can be easily violated similarly. 

Theorem 3.12
I didn't check the proof, but your high-level claim does not match your statement. The statement gives to upper bounds, which only shows that ''the worst convergence rate of one is faster than the other'', but not '' the convergence rate is faster'', which you should probably prove a lower bound to claim. 

Moreover, despite the well written and well motivated introduction, the technical part of this paper is poorly organized and hard to follow by  a non-expert. Many important yet basic setting is not introduced (MLE, MAP, and the online model). I realized that this paper adopts an online-setting model and care about regret or the first time only at the end of the page 6! Many fancy models, assumptions and priors are introduced without justification or references. People with no expertise on MLE or MAP may not understand what the algorithm is doing. (I guess it is updating the confidence interval B?) Given the problem itself is important and highly non-trivial, I think it is more important to let the audience to understand the basic notions from the beginning, which the more technical parts (such as Dietrich Prior) should go to the Appendix. 

Minors:
L110: t is not defined
L231: Graph prior is not referred. 
L258: B^{u,a,v} is not defined but used multiple times throughout the paper. 
L1197: I don't think the definition of non-dictatorship for (27) is correct. The ($\forall$ x, x') and ($\not\exists$ $i$) should be swapped. There also should be ''for any utility profile''.

-----------------
Update: Theorem 3.3 and 3.12 are addressed. Theorem 3.9 is beyond my expertise now. I raise the score to 5 and decrease my confidence to 2.

### Questions
***1. Please defend yourself on the soundness of the theoretical results. Please address every point I proposed for all three theorems, justify your proof is correct or can be made correct with reasonable assumptions. If the soundness can be justified or amended, I am happy to raise my score. 

2. In the algorithm Line 4, you maximize the acquisition function by finding the pair of candidates where agents overall have strongest preferences on one to the other. What is the intuition behind this? Firstly, this sound like greedily finds the local maximum in every step, which usually does not find the global maximum (unless the function is convex). Secondly, if we want to elicit as much information in each step, we should select the most informative pairs. I tend to believe that the pairs are close to each other are more informative, because revealing the difference between then help us better understand the utility functions. 

3. For Theorem 3.9, why there is an $n$ factor in the regret of oracle? If we know the adjacency matrix A and its inversion, elicit public votes by an oracle should be the same as elicit private votes by "None", and the regret should also be the same.

4. You study an online setting and consider about regret. Will the offline setting make a difference? For example, if we can elicit a full ranking from each agent, will things change?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces Social Bayesian Optimization (SBO), an innovative algorithm designed to build consensus in group decision-making by mitigating social biases that often distort individual preferences. Traditional consensus-building approaches struggle under social dynamics like the bandwagon effect, which leads to non-truthful feedback. SBO overcomes this by combining two types of votes: noisy but cheap public votes and private, costly votes, free from social influence. The algorithm leverages a social graph model to efficiently estimate social influence, allowing it to correct public votes and reduce the reliance on costly private feedback.

### Strengths
1. The paper introduces a voting system in Social Bayesian Optimization (SBO) that effectively separates noisy public votes from more reliable private feedback. This approach allows the algorithm to debias public responses, which is particularly impactful in social contexts where group dynamics often distort true preferences.

2. The authors provide a well-structured theoretical framework, including a proof that achieving groupthink-proof consensus is impossible without the dual voting mechanism.

### Weaknesses
1.  The algorithm’s performance heavily depends on accurately modeling social influence through a predefined social graph, which may not reflect the complexity of real-world interactions. 

2. While the dual voting system improves consensus accuracy, it also increases computational and operational costs due to the need for costly private votes. 

3. The paper assumes certain types of social influence, like bandwagon effects, but does not account for other common group dynamics (e.g., peer pressure or hierarchical influence). 

4. The model assumes agents have consistent and rational preferences, which may not align with the variability found in real-world settings. 

5. The algorithm depends on several parameters, such as the decay rate and aggregation weight, but the paper lacks a thorough sensitivity analysis on how variations in these parameters affect performance.

### Questions
1.  The algorithm’s accuracy relies on estimating the social influence graph effectively. How do estimation errors in the social graph affect the convergence rate and regret bounds of SBO? Can the authors quantify a bound on the cumulative error introduced by inaccurate social graph estimation?

2. The paper models social influence as a linear function within the social graph. How would the algorithm’s convergence and regret bounds be affected if the influence model were non-linear? Is there a theoretical guarantee that SBO can handle non-linear influence functions without compromising efficiency?

3. The paper provides sublinear regret bounds for SBO. However, given the dual voting system’s reliance on private votes to debias public ones, can the regret bounds be formally decomposed to show how each voting system independently contributes to the cumulative regret? How would an imbalance in the quantity of public versus private votes affect these bounds?

4. The algorithm assumes that agents’ preferences are rational and consistent. If agent feedback were inconsistent or stochastic, how would this impact the estimation of utilities and the associated convergence rate? Could the authors provide an error bound quantifying the effect of inconsistent feedback on the final consensus?

5.  SBO’s performance depends on parameters like the decay rate and the aggregation function’s weight. Can the authors derive optimal parameter values that balance the trade-off between convergence speed and the accuracy of social influence estimation? What are the theoretical limits on these parameters for guaranteed sublinear convergence?

### Soundness
2

### Presentation
3

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
This paper considers the Social Bayesian Optimization (SBO) for consensus-building, which is to estimate n people's utility functions from their votes. The paper proves an axiomatic impossibility result to have both groupthink-proofness and non-dictatorship. The paper then describes its proposed mechanism that iteratively finds the better social outcomes.

### Strengths
- This is my first time reading literature on this topic and the paper does a fairly good job in explaining key notions, challenges and techniques in this problem.
- The paper includes several experiments under the real world context.

### Weaknesses
- The paper introduces several assumptions that are perhaps too strong to be realistic. For example, Assumption 3.4 requires private votes to reflect the truthful utility, but it is unclear that such truthful utility exists and can be elicited easily. If such private votes exist, why not use as many private votes as possible?
- The paper lacks sufficient justification on the limitations of their proposed mechanisms.

### Questions
I have some research experiences in mechanism design, social choice theory and statistical learning theory, but I have little knowledge in this BO, especially in its application to preference aggregation. Hence, I would like to understand how the preferential bayesian optimization techniques have been applied. Does it have any real world application? How many samples does it require to work under the typical noise condition in practice? --- for example, it is not quite realistic to elicit human preference with >10 samples, your proposed mechanism could take too long to run.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes a consensus-building inspired learning problem.  Specifically, given a graph $G$, each agent $i$ has a true utility over a set of items $x$, $u(x,i)$, and influenced utility $v(x,i)$ which depends on both its true utility as well as it's neighbor's in $G$.  They first show that there does not exist an aggregation function $A$ so that $$argmax A(u,x) = argmax A(v,x)$$ (groupthink-proof) and not dictatorship.  Then they provide a public and private vote setting that can query noisy comparison based on either influenced utility $v$ (public vote) or true utility (private vote) from each agent and design a learning algorithm to bound the error and sample complexity.  Finally, they conduct some experiments.

### Strengths
This paper's model is ambitious that try to use learning techniques to address consensus building problem.

### Weaknesses
## Unclear contributions
1.  I am not convinced that the social Bayesian optimization, SBO (Theorem 3.9), offers a meaningful solution to consensus building.  The introduction suggests that consensus building is challenging due to social dynamics, and "even with access to each agent's utility, constructing a social utility that fairly represents all agents is impossible under mild rationality constraints." My understanding is that these challenges stem from the discrepancy between influenced utility and true utility—a problem that becomes irrelevant if direct estimation from the true utility is possible. However, their algorithm requires direct access to private votes based on all agents' true utility.  A more convincing justification for SBO would focus on its sample complexity benefits.  For instance, is SBO superior to using private votes alone? To what extent does incorporating public votes enhance the learning process?
2.  The impossibility result in Theorem 3.3 is strange, particularly in its nonstandard definition of dictatorship.  Unlike traditional definitions, where one dictator determines the order of all items, Definition 3.2 allows for different dictators for each item pair. Under this definition, even majority voting or a simple average function might satisfy their dictatorship definition.  This nonstandard definition undermines the impossibility result. 
## Convoluted writing
The paper should improve its readability.  The problem formulation remains unclear until Assumption 3.4, making it difficult to follow the main argument initially. Additionally, the paper presents several facts that appear unrelated to the core problem (e.g., Arrow’s impossibility results on line 40 and Proposition 2.4 on GSF).

## Minor comments
- In line 46, please provide either citation or your results for the second challenge.  
- In line 98, assumption 2.2 is the Bradley-Terry model.  You should introduce it here instead of in line 242.
- In line 107, the definition of linear aggregation function is unclear.  Specifically, does the weight $w$ depend on $u$?  If $w$ is independent of $u$, linear aggregation functions do not contain egalitarian nor the GSF in equation (2).  If $w$ can be any function of $u$, any function of $u$ satisfies the definition.  
- Equation (4) is not clear to me.  Instead of scalar $\kappa_i$, should the parameter of Dirichlet $n$-dimensional?
- in line 252, is $\hat{v} = \hat{A}\hat{v}$ a typo?

### Questions
Is SBO superior to using private votes alone? 
To what extent does incorporating public votes enhance the learning process?

### Soundness
3

### Presentation
2

### Contribution
3
