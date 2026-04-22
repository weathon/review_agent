# Making A Trade-Off Between Cost and Distance By A Differentiable Way

- Avg Score: 4.00
- Decision: Reject
- Scores: 8, 4, 2, 2

## Abstract
The Cost-Distance problem, introduced by Meyerson, which is a natural abstraction for modeling UAV logistics networks, seeks a network design that simultaneously minimizes construction cost and the weighted routing distances from multiple sources to a designated root. Existing methods exhibit a strong dependence on the number of sources and are difficult to parallelize, which hinders their scalability on large graphs. We propose Cost-Distance Policy Gradient (CDPG), the first gradient-based framework for this problem. CDPG relaxes the discrete subgraph selection into a probabilistic adjacency matrix and formulates the Cost-Distance objective as an expectation, enabling efficient optimization via policy gradients. Our algorithm achieves the time complexity of $\mathcal{O}(m\log n)$, faster than the previous fastest approximation algorithm's $\mathcal{O}(|S|(m+n\log n))$ in graphs with dense sources. Extensive experiments across 9 real-world Unmanned Aerial Vehicle (UAV) logistics scenarios in the Guangdong-Hong Kong-Macao Greater Bay Area demonstrate that CDPG significantly outperforms approximation algorithms, continuous relaxation baselines, and heuristic search methods. Our code is available at: \url{https://anonymous.4open.science/r/iclr_cdpg-8737}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper considers the cost-distance problem, in which a connected subset of a graph must be selected such that an objective function consisting of edge-wise costs and routing distances be minimised. The routing distances are defined with respect to a root node and (possibly several) source nodes. The authors propose a learning-based approach based on a probabilistic relaxation that estimates the probability of each edge being part of the solution. These probabilities gives rise to an RL policy, whose value function (and hence estimate of routing distances) can be computed in a differentiable way via matrix computation.  Masking is applied to retain acyclicity and hence solution validity. The authors study the algorithm's properties theoretically, showing advantageous computational complexity. The method is also extensively validated empirically, showing better performance than competing methods including a recent approximation algorithm, and fast runtimes.

### Strengths
S1. The paper studies a well-motivated practical problem.

S2. Both the theoretical and experimental analyses are solid and results obtained are excellent.

### Weaknesses
W1. The clarity in parts of the manuscript could be improved and some of the choices should be better justified.

### Questions
C1. I don't think Theorem 2 deserves to be a theorem. The matrix form that is given is simply the *policy evaluation* algorithm for a given policy $
\pi$, with the reward function substituted with the one for this MDP. This is very well known in the RL literature.

C2. It is unclear to me why Soft Actor-Critic, with a stochastic policy, is used as a baseline. While there are indeed RL methods that sample actions at evaluation time, the much more standard RL setup would be to train a policy and draw actions from it *greedily* at evaluation time (i.e., always choose highest probability action). Stochasticity seems to hurt here and lead to worse results. Can you provide a baseline with greedy evaluation of the policy?

C3. I find it odd to refer to the methods in the results as e.g. "NOTEARS", since this is just CDPG but using a different acyclicity method in the whole algorithm pipeline. NOTEARS itself cannot be applied to this problem. Different names for these CDPG variants (incl. GS, MEP) would be clearer.

C4. Source code includes only the proposed method; all baselines and experimental evaluation code should be provided for reproducibility. Ditto for the datasets (only 2 are given)

C5. Small comments:
- Check use of \citet and \citep, textual citations are used where parenthetical citations should be used e.g. L34-39
- Typos: "parameterized" -> "parameterizes"; "its" -> "it is" (L226)
, "leaded" -> "reached" (L234), "denoted" -> "denotes" (L248), "satiesfies" -> "satisfies" (L270), "improve" -> "improvement" (L389)
- Simulated annealing references: author names duplicated

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CDPG, the first gradient-based framework for the Cost-Distance problem in network design, which minimizes both construction costs and weighted routing distances. CDPG relaxes discrete subgraph selection into a probabilistic adjacency matrix optimized via policy gradients, using novel components including bilinear edge policy, acyclic masking, and MDP-based routing distance formulation. The method achieves O(m log n) complexity versus O(|S|(m+n log n)) for existing approximation algorithms when sources are dense. Experiments on 9 UAV logistics networks show 2.5%-27.67% improvements over baselines.

### Strengths
1. **Novel Methodology**: First gradient-based approach to Cost-Distance problem with creative MDP formulation for routing distance (Theorem 2), enabling differentiable optimization of shortest-path objectives.
2. **Strong Theory**: Comprehensive analysis including Lipschitz continuity (Theorem 4), error bounds (Theorem 5), convergence guarantees (Theorem 6), and complexity improvement to O(m log n) (Theorem 7) with detailed proofs.
3. **Well-Designed Components**: Acyclic mask, unreached penalty, and bilinear embeddings are well-motivated with convincing ablation studies demonstrating each component's contribution.
4. **Practical Contribution**: Real-world UAV logistics dataset from Greater Bay Area with careful construction of node/distance/cost weights; comprehensive experiments across multiple baseline categories.

### Weaknesses
1. **No Approximation Guarantees**: Critical gap—provides convergence to stationary point but no bounds on solution quality vs. optimal. Held algorithm has O(log |S|)-approximation; CDPG offers no comparable guarantee, making it unclear when to use in practice.
2. **Limited Applicability Analysis**: Minimal characterization of when CDPG fails (sparse sources, Graph 9). Improvements range 2.5%-27.67% but no guidance on what graph properties determine performance. "Significantly outperforms" is overstated.
3. **Hyperparameter Concerns**: Sensitivity analysis only on Graph 3; uses fixed values (η=0.06, l=8, T=300) across all graphs without justification. Unclear if problem-specific tuning is needed. Initialization strategy (H,Q~N(0,1), K←0) appears arbitrary.
4. **Incomplete Baselines**: Only compares one approximation algorithm (Held) despite citing others (Meyerson 2008, Chekuri 2001). No simple greedy heuristics. Continuous relaxation baselines are adaptations, not direct competitors.
5. **Reproducibility Gaps**: Rounding/pruning procedure underspecified; no convergence plots/criteria; initialization sensitivity not analyzed; GPU speedup claimed but not demonstrated; no scalability experiments beyond provided graphs.
6. **Presentation Issues**: Notation overloading (X, P); Algorithm 2 parameters (η=0.08, l=6) contradict stated defaults (η=0.06, l=8); unclear connection between Theorem 1 and actual algorithm; figures difficult to read.

### Questions
1. Can you provide approximation ratio bounds (theoretical or empirical via LP relaxations)?
2. What graph properties beyond |S| determine when CDPG outperforms Held? Can you provide selection guidelines?
3. How were default hyperparameters chosen? Do they transfer to new instances or require tuning?
4. Please specify the exact rounding and pruning algorithms. How does rounding affect solution quality?
5. Why weren't Meyerson et al. (2008) and Chekuri et al. (2001) algorithms compared? What about greedy heuristics?
6. What is initialization sensitivity? Have you tried multiple restarts? What are actual convergence criteria?
7. Can you provide GPU speedup measurements and scalability to 50K+ node graphs?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work tackles a challenge in planning efficient delivery networks for drones. The goal is to design a network that keeps both construction costs and travel distances low.  To solve this, the authors developed Cost-Distance Policy Gradient (CDPG) an approach that uses ideas from reinforcement learning/Markov decision processes to find better network designs more quickly. Instead of working directly with fixed connections, CDPG treats possible connections as probabilities, allowing the system to “learn” which ones work best through trial and error. The authors evaluate the proposed approach using drone delivery scenarios.

### Strengths
+ MDPs are powerful tools for optimization problems and this paper presents an interesting example of potential applications of these techniques.

### Weaknesses
- Unfortunately, the paper is not convincing in terms of modelling. It seems to the reviewer that the problem itself (UAV routing)should not be studied as a cost-distance problem, since being a UAV network, the cost is the same for all the routes (they are in the air): there is a single minimization cost, i.e., distance.

- Formula (1) does not appear correct since the authors are not weighting the different importance of cost and distance (but please note that the cost is the same, so it can be ignored in this situation; it is a constant).

- The method of “bilinear logics” is not well introduced. The authors do not provide a sufficiently convincing motivation in terms of the actual theoretical foundations of the method.

- Theorem 6 is not specific to the problem under consideration; it is much more generic.

- The evaluation is based on datasets that are not “controlled”. The impact of specific characteristics of the graphs is not studied. For example, the authors should have studied different graph structures, in my opinion (such as random graphs with different probability of links, etc.).

- It is difficult to explain the results in Figure 2. Why do you have the constant cost with an increasing number of nodes? That is very difficult to explain. The computational cost must increase given the algorithm implemented by the authors.

### Questions
In my opinion, the paper has a major flaw in terms of actual modelling. Unfortunately, the authors have to reconsider the actual modelling decisions that they made - for this paper, it is not a matter of clarifying some specific points.

However, the reviewer is curious to understand why the results show that the computational costs are constant even in presence of a larger number of nodes. This is very difficult to explain given the design of the algorithm.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The work addresses the problem of designing a network for UAVs by relating it to the cost-distance problem where the task is to minimize construction cost and the weighted routing distances from multiple sources to a designated root. The work claims that existing methods have high runtime complexity, and their work has better parallelism to allow for scalable and fast optimization. Experiments are shown on 9 graphs.

### Strengths
The problem, inspired from graph theory, seems interesting. The proposed contribution, cost-distance policy gradient seems non-trivial and empirically seems work well on the 9 tested graphs.

### Weaknesses
There are several concerns that require addressing. Some are listed below.

The work needs a better motivation for addressing the cost-distance problem in relation to designing the UAV network. The abstract and intro cite the UAV network planning as the main motivation for their study of the cost-distance problem. However, very little attention is paid to how to realistically model UAV network design problem to the studied formalism. I can only see one or two paragraph in the intro to the somewhat high-level connection of UAV logistics and cost-distance problem. There are several unclear points such as what are root, source nodes, edges in UAV terminology? Rarely a real world problem such as UAV logistics translates itself into a clean abstraction such as the cost-distance framework. The work should provide an clear, and detailed connection to UAV logistics by discussing appropriate background work in UAV network design and cost-distance problem.

There needs to be more effort as to why the cost-distance problem is relevant to the ICLR community. Most of the prior work in this problem is in the theoretical computer and OR community. Most of the baselines are also from the OR and approximation math optimization literature (Held and Perner). A main issue is that there is not a clear and agreed upon definition of the UAV network design problem available based on the cited work. Thus, an effort should be made to connect the cost-distance problem to other well-studied problems (with relevance to ICLR) which have baselines and datasets available.

The writing of the paper requires substantial improvements. Technically, several dense mathematical terms have been introduced without clear intuition and explanations. Some examples are below:

– The paper straightway goes into the problem formulation from the third para of introduction. It should be moved to the problem formulation section

– In Eq 1, it is not clear if (i, j) belong to X or X’ in \sum_{i, j}

– Same confusion for \sum_i in Eq 1

– Why Eq4 guarantees P is acyclic, no citation or intuition is provided 

– What are different terms in Eq 5, what does the p^\theta . X . A provides us? How this mask really works? There needs to be a clear working example.

– The definition of \phi in line 191 is not fully explained. In particular, how different terms in \phi definition justify the logic explained in lines 191-196

– Given that computing W* is NP-hard, it is quite strange to approximate it using a constant matrix (line 204). This seems adhoc.  Why that can be a good approximation for real world problems is far from clear.

– The penalty term in in Eq11 is not fully explained. It seems quite dense, and its exact role and understanding is not clear. 

These are only few of the concerns in technical section. As a result of these issues, the reader is unable to grasp the relevance and significance of contributions clearly.

The authors claim their approach is faster. However, unlike previous works by Meyerson, they do not provide what kind of bounds their method can achieve. In general, even a random approach can be faster, but it of course may not provide any quality bounds. Thus, the authors should clarify this point when discussing the runtime benefits of their method.

Empirical evaluation is somewhat limited. Results are on synthetic instances that are constructed by authors themselves by deriving them from a part of the greater bay area. However, these are not real world UAV logistic design problems that are widely accepted in the community. There are several design assumptions made as noted in section D to interpret them as UAV problem. 

I also find baselines methods quite limited. The author claim that method by Held and Perner is most relevant. However, this work seems like a arxiv report, I could not verify if this was published in a conference/journal. Thus, it does not inspire confidence to have comparisons against such unpublished work.

Total 9 graphs is also quite limiting. Ideally, the authors should chose a large, publicly available UAV design problems not constructed by them. It is quite difficult to draw any firm conclusion from this small dataset provided by the authors themselves.

### Questions
See above.

### Soundness
2

### Presentation
1

### Contribution
2
