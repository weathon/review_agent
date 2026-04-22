# Bandit Learning in Matching Markets with Switching Cost

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 6

## Abstract
We study the bandit learning problem in two-sided matching markets. While existing works successfully derive sub-linear bounds for the player-optimal regret, they typically assume cost-free switching and may incur up to $O(T)$ switches over a time horizon of length $T$. Such frequent reassignments are impractical in real-world applications since switching is usually costly and disruptive. To address this limitation, we explicitly incorporate switching costs into the decision-making process and aim to minimize player-optimal stable regret under a switching-cost budget. 
We first consider a setting with unit switching cost, where each switch incurs a fixed cost. 
We propose a cost-aware algorithm that achieves the same regret bound of $O(\log T/\Delta^2)$
as previous approaches while reducing the total number of switches to $O(\log T)$, where $\Delta$ is the players' minimum preference gap. Furthermore, we show that by slightly relaxing the regret to $O(\sqrt{T/\Delta^2})$, the total number of switches can be reduced to $O(\log \log T)$; in the extreme case, with only $O(1)$ switches, the algorithm still guarantees a regret of $O(T^{2/3})$. 
We also generalize this approach to heterogeneous switching cost setting by leveraging the shortest Hamiltonian Circuit orderings and provide analogous theoretical guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work studies bandit learning in two-sided matching markets. Prior work achieves sublinear regret but incurs $O(T)$ switching costs over a horizon $T$. Such frequent reassignments are impractical in real-world applications due to operational and coordination costs. To address this issue, the paper aims to minimize regret while keeping switching costs low.

Under unit switching costs, the authors claim to match existing regret bounds while reducing the number of switches to $O(\log T)$. Moreover, they show that by further reducing switching to $O(\log \log T)$, the regret becomes $O(\sqrt{T}/\Delta^{2})$. In the extreme case of only $O(1)$ switches, they obtain a $T^{2/3}$ ($\alpha$-fraction) regret bound. The paper also discusses a generalization of their approach to heterogeneous switching costs.

### Strengths
This work provides the first algorithm that achieves low switching costs for bandit learning in two-sided matching markets.

### Weaknesses
1. A primary concern is the tightness of the regret bounds. In the literature on batched bandits, it is well known that near-optimal regret can be achieved with only $O(\log \log T)$ switching cost. However, in this work, the proposed algorithm attains a suboptimal regret of $\tilde{O}(\sqrt{T}/\Delta^{2})$ under the same switching budget.

2. Another concern relates to the novelty of the algorithmic contribution. Although it is interesting to explore low-switching algorithms in two-sided matching bandits, it is unclear what new ideas are introduced beyond existing epoch-based batched bandit techniques and prior work on matching-market bandits.

3. Finally, in the heterogeneous switching cost setting, although the paper claims to obtain theoretical guarantees analogous to the unit-cost case, the regret bounds are not clearly specified. A more explicit statement of the regret order in this setting would significantly improve clarity.

### Questions
1. Could the authors better clarify the novelty of their algorithmic contributions? In particular, it is not fully clear what elements go beyond standard epoch-based batched bandit techniques and prior matching-bandit algorithms.

2. Could you elaborate more on the regret guaranteed under heterogeneous switching costs?


3. In Algorithm 1, it is unclear how the total number of rounds is controlled. Specifically, Line 8 runs $(t_{l+1} - t_l)/K$ consecutive pulls per arm in each phase, which appears to exceed the total horizon $T$ when summing across all phases. A clarification of how the time budget aligns with the horizon would be helpful.

### Soundness
2

### Presentation
3

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
The paper investigates the bandit learning problem in a two-sided matching market where one side (the players) has unknown preferences (utilities) regarding the other side (the arms). The optimal stable matching must be learnt over a time horizon T. Crucially, the authors move beyond the standard assumption of cost-free switching, which allows for potentially O(T) reassignments. By explicitly modeling switching costs, the paper aims to minimize the player-optimal stable regret under a predefined switching cost budget. To address this problem, the authors propose new algorithms (such as SCAM) and provide corresponding theoretical regret analyses.

### Strengths
**1. Theoretical Rigor and Coherence:** The paper is well-written and logically sound. The arguments are self-contained, and the proofs appear complete. I have read through the proofs and did not find any fundamental flaws or fatal errors in the analysis.

**2. Meaningful Motivation:** I endorse the paper's motivation. The concept of switching cost in matching markets is essential; in real-world scenarios (e.g., job assignments, resource allocation), switching a match involves various extra overheads, disruptions, and administrative costs. Incorporating this constraint makes the model more practical than existing works.

### Weaknesses
**1. Lack of Experimental Validation:** The paper is purely theoretical, and the absence of any experimental evaluation on real-world datasets is concerning. Without experiments to demonstrate the performance of the proposed algorithm (especially concerning the trade-off between regret minimization and cost budget adherence) against baselines in real-world scenarios, the structure of the article feels incomplete and unpolished. The lack of experiments also raises a strong suspicion regarding the real-world applicability of this model. It leaves the impression that the scenario modeled in this paper might be purely hypothetical and not grounded in a verifiable practical context.

**2. Insufficient Technical Novelty:** The two-sided market bandit model, the incorporation of switching costs, and the analytical techniques employed are all fairly conventional and do not offer substantial novelty for a paper submitted to ICLR 2026. The core algorithmic ideas—namely, grouping exploration rounds for the same arm and adopting exponentially increasing epoch lengths (the doubling trick) to reduce switching—are standard practices in the broader bandit literature when addressing switching cost constraints. While the paper is clearly written and the analysis is sound, I did not find the technical contributions particularly innovative. Overall, the work appears to be a rather straightforward application of existing methodologies developed for matching market bandit problems.


**3. Minor**  There appears to be a structural issue in the description of Algorithm 1. Specifically, the loop in Line 7 seems to be incorrectly nested within the loop in Line 5. This is, however, a minor issue and should be easy to fix.

### Questions
1. Could you please give an example and explain in detail what real-world data set you would choose if you were to do experimental verification, how you would process the data, and how you would design the experiment?
2. Have you considered how to derive results concerning the regret lower bound, and how to construct a hard instance to prove it?

### Soundness
3

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
This paper considers matching in bandit markets with switching costs.
The algorithm uses a phased design, similarly to successive elimination in bandit problems. It also consecutive plays each arm before moving on to the next arm. I wonder if this rather obvious adaptation is taken into account when considering existing algorithms. Let's take the ETC algorithm from Lui et al. It performs $hK$ matchings, and so an equal number of switches. If, instead of cyclic mathings, it performed them in blocks, then it would have only $h$ switches. Setting $h = O(\ln(N T))$. So,  $N,K$ do not appear outside the logarithm, which makes sense.

In general, I find these 'decentralised' extensions of the matching problem to be pseudo-decentralised. They rely on an initial agreement to follow a protocol which has to be centrally managed in order for arms and players to obtain indices. In this setting, I do not see why players could simply not re-propose until they get matched, for example: with the matchings taking a very short amount of time, relative to the time needed to obtain the reward. In the end, though, why do we even get rid of the matching platform? The whole point of the matching problem is to make sure it matches everybody according to their preferences.

The basic algorithm is simple enough. After indexing to avoid conflicts, there are phases within which each player can sample up to K arms. To simplify, at subphase l, player i is playing arm s(i,l) so that none are playing the same arm. This setting is a rather straightforward extension.

The general cost case requires solving a parallel TSP problem.
If we are talking about sample complexity, switching and regret, then I think we do not really care about computation: I think the discussion about optimal PTSP solutions is beside the point: the authors could have simply said they assume they obtain an approximate PTSP solution somehow. Again, this problem would not have arisen in the centralised case, and it somehow seems to me as though they authors are trying to make the problem more convoluted for no good reason.

### Strengths
+ Very well written
+ OK overview of prior work, including switching.
+ The general switching cost setting has some more complex elements.

### Weaknesses
- The prior work ignored the switching problem, but it was trivial to fix in most cases. In some sense, this misrepresents the prior work.
- The extension then seems almost trivial in the uniform cost setting.
- While general cost setting is valid, it becomes contrived once you factor in decentralised proposals.

### Questions
Why do we need to be 'decentralised'?
Why did you not take into account the trivial method for taking switching into account in prior work?
Why do you complexify the general cost case and not simply assume you have an epsilon-optimal solution?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper considers the variant of the matching markets bandit problem where there is switching costs. The model is motivated by practical applications and other related works in online learning. The observation the paper makes is that most algorithms for matching markets work in two phases -- an exploration phase where every player learns their preferences over arms and an exploitation phase where players use the Gale Shapely matching based on their estimated arms. Existing algorithms incur a large switching cost in the exploration phase as they essentially employ round robin. The paper proposes an alternative for the exploration through carefully repeating each arm sufficient number of times. The exact number of times to play each arm in a phase is given by a geometric series. Through this analysis, the authors show that $O(K \log(T))$ switches can yield $O(K \log(t)/\Delta^2)$ regret. Further, the authors also generalize it to the case when switching costs are non-uniform. They show that a heuristic -- where they solve the problem for a single player and then offset the starting point for the other players, can yield similar results.

### Strengths
The paper has several theoretical contributions. It identifies a simple way to mitigate the round-robin style exploration common in all prior algorithms. Further, it shows a simple way of extending the analysis to the heterogeneous case, by utilizing connections to finding Hamilton paths in the cost graph. They solve the collision avoidance by a simple heuristic of computing the tour for a single player and then producing an offset for others. They have an approximation ratio showing that this is worse off compared to the optimal only by a fraction which is the ratio of the largest to the smallest switching-cost.

### Weaknesses
The paper's writing can be improved in two ways. First, it will make it better to read even with a few illustrative figures, especially for the Hamilton path heuristic and their core algorithmic contribution in the unit cost case. Second, it was not immediately obvious to me that if all costs are uniform, the generalized algorithm reduces to that of the unit-cost case. Having a section, maybe even in the Appendix can help readers parse and understand the algorithm better. Without these aids, the algorithms are quite hard to understand and parse.

### Questions
Can you explain/show, how the generalized algorithm reduces to the one you had for unit- switching cost case? This will help me understand the contributions of the generalized algorithm better.

### Soundness
3

### Presentation
2

### Contribution
3
