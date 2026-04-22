# Risk-Sensitive Agent Compositions

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
From software development to robot control, modern agentic systems decompose complex objectives into a sequence of subtasks and choose a set of specialized AI agents to complete them.
We formalize agentic workflows as directed acyclic graphs, called agent graphs, where edges represent AI agents and paths correspond to feasible compositions of agents.
Real-world deployment requires selecting agent compositions that not only maximize task success but also minimize violations of safety, fairness, and privacy requirements which demands a careful analysis of the low-probability (tail) behaviors of compositions of agents.
In this work, we consider risk minimization over the set of feasible agent compositions and seek to minimize the value-at-risk and the conditional value-at-risk of the loss distribution of the agent composition where the loss quantifies violations of these requirements.
We introduce an efficient algorithm which traverses the agent graph and finds a near-optimal composition of agents.
It uses a dynamic programming approach to approximate the value-at-risk of agent compositions by exploiting a union bound.
Furthermore, we prove that the approximation is near-optimal asymptotically for a broad class of practical loss functions.
We also show how our algorithm can be used to approximate the conditional value-at-risk as a byproduct.
To evaluate our framework, we consider a suite of video game-like control benchmarks that require composing several agents trained with reinforcement learning and demonstrate our algorithm's effectiveness in approximating the value-at-risk and identifying the optimal agent composition.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies risk-sensitive agent composition in multi-agent workflows. It models the system as a directed acyclic graph (DAG), where each edge represents an agent and each path corresponds to a feasible composition. The goal is to minimize the Value-at-Risk (VaR) of losses that capture safety, fairness, or privacy violations, focusing on tail risks. To do this, the authors propose a dynamic programming (DP)–style algorithm, BucketedVaR, that uses a union bound and a discretized risk budget to estimate an upper bound on the VaR. They prove asymptotic near-optimality under independence assumptions, and experiments on compositional RL benchmarks show the method can recover the optimal paths with accurate VaR estimates.

### Strengths
1. The paper presents a technically sound and mathematically coherent formulation of risk minimization for agent compositions. 

2. The theoretical development is detailed and carefully reasoned, with clear proofs supporting the approximation guarantees. 

3. Overall, the work provides both a mathematical framework and an algorithmic contribution that meaningfully advance the study of risk-aware agent systems.

### Weaknesses
1. The paper’s presentation is quite dense, and the abstract and introduction do not clearly communicate the main ideas, which makes it difficult to identify the core contribution without considerable effort. 

2. The approach also depends on several strong assumptions, such as the DAG workflow structure and the independence of agent losses, which limit its generality and practical relevance. 

3. Finally, the work remains mostly theoretical, and it is not clear how the proposed framework could be applied or integrated into real-world multi-agent systems.

### Questions
1. The framework assumes that the overall task can be represented as a DAG of agents. In many real-world scenarios, agent interactions may not follow a strict directional or decomposable structure. How restrictive is this assumption, and could the proposed approach be extended beyond DAG workflows? 

2. The theoretical analysis seems to rely on several independence and directionality assumptions, as well as implicitly fixed outputs between connected agents. Could the authors clarify which of these are essential for the algorithm’s validity and which could be relaxed? 

3. The loss function is defined only over the trace T, while the output Y appears not to influence the risk. In realistic settings, the output could affect downstream traces. How would the framework handle such dependencies? 

4. In Figure 4b, the 16-Rooms benchmark exhibits a noticeable spike when using 5–20 buckets. Could the authors explain this behavior? 

5. While the theoretical and algorithmic contributions are clear, the practical applicability of the framework remains uncertain. The paper would benefit from a discussion of potential real-world scenarios where the proposed method could be directly used or integrated into existing agentic systems.

I am not primarily working on the formal side of risk-sensitive optimization, but these questions reflect my current understanding of the framework and its assumptions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a risk-aware framework for optimally selecting agent compositions that are represented as directed acyclic graphs, called agent graphs. The algorithm BucketedVaR utilizes dynamic programming to traverse the agent graph and find the optimal path that minimizes the value-at-risk of losses by dividing the risk budget $\alpha$ among the agents and applying a union bound. Theoretical analysis shows that asymptotically, the quantile estimated by the algorithm is near-optimal when the loss variables along every path are independent or are only loosely correlated, scaling polynomially with the number of agents. The algorithm is tested on a series of reinforcement learning benchmarks to evaluate its effectiveness in identifying optimal paths and quantifying tail behaviors.

### Strengths
- This work connects risk-sensitive optimization and compositional agent systems using directed acyclic graphs. The presented BucketedVaR algorithm is polynomial in the number of agents, thereby avoiding exponential enumeration (as seen in baselines).

- The topic presented is relevant and has significance for measuring risk sensitivity in safety-critical AI agent composition. The experiments in various RL benchmarks demonstrate that BucketedVaR successfully identifies the same optimal path as the baseline algorithm across all benchmarks, while providing tight estimates of VaRα.

- The paper is clearly written and well-organized; the agent graph formalism and examples presented help illustrate the ideas effectively.

### Weaknesses
- In real AI agent chains, losses are often correlated via shared context or sequential dependence; the paper should analyze or empirically test how violation of independence could affect performance.


- Since the work only tested on RL benchmarks but described LLM examples as potential use cases, missing validation on LLM agentic pipelines where sampling cost and judge noise matter is concerning. For LLMs, losses (e.g., amount of hallucinated information) may be subjective, noisy, or non-numeric. Additionally, black-box sampleability may be significantly more expensive than in RL, where thousands of samples per edge can be easily obtained.


- VaR ignores tail severity and is not a coherent risk measure; the authors should justify this choice of using VaR versus CVaR or provide comparative results.

### Questions
1. How sensitive are the algorithm’s guarantees/empirical performance to deviation from independence and to noise in loss measurements?
2. Why was VaR chosen over CVaR?
3. How would the method adapt when only a few samples per agent are feasible, such as LLM-based agents?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper formalises risk-sensitive composition of agentic workflows represented as DAGs, where path loss is the max over edge-level losses. It proposes BucketedVaR, a dynamic-programming algorithm that allocates a risk budget across edges via a union bound to upper-bound the path value at risk and select a path in polynomial time. Theoretical results include a finite-sample guarantee based on DKW and an asymptotic near-optimality bound under independence of edge losses. Experiments on compositional rl tasks show tight empirical coverage and agreement with an exhaustive baseline on small graphs.

### Strengths
1. Clear formulation & motivation: Risk of max loss is appropriate for safety/privacy violations in composed systems.

2. Simple, scalable idea: Union-bound budgeting with DP avoids path enumeration; complexity $O\left(n(d+1)^2|V|^2\right)$.

3. Theory with interpretable slack: Asymptotically, the selected path is within an $\alpha^2 / 2$ quantile-level slack under independence (Thm. 2)

4. Empirical evidence: Tight coverage across benchmarks; agreement with exhaustive baseline; clear sensitivity to buckets/sample.

### Weaknesses
1. Choice of VaR over CVaR: VaR ignores tail severity; CVaR is often preferred for safety. The paper mentions CVaR as future work but gives no partial result or empirical check.

2. Missing baselines and ablations. No comparison to chance-constrained shortest path or to dependence-aware relaxations. No experiments that explicitly vary tail dependence (e.g., shared noise seeds, correlated disturbances) to show robustness/failure modes.

3. Finite-sample conservativeness is unclear. Thm. 1 shows the returned threshold q satisfies coverage $\geq 1-(\alpha+\gamma)$ (Eq. 5), i.e., it provides an upper bound on $\text{VaR} _{\alpha+\gamma}$, not necessarily on $\operatorname{VaR} _\alpha$. Hence $q$ can underestimate the true $\operatorname{VaR} _\alpha$ when $\gamma>0$-problematic for safety guarantees. The text claims the estimate is "at least as large as the true VaR" asymptotically, but the finite-sample statement doesn't ensure one-sided conservativeness at level $\alpha$.

4. Graphs are small (e.g., 16 paths), and a is relatively large ( $\geq 0.05$ ). Safety practice often targets $\alpha \leq 10^{-3}$. Sample sizes of $10^4$ per comparison are heavy and may be infeasible for LLM/robotic agents. There is no runtime/scaling analysis on denser DAGs or rarer tails.

### Questions
Q1 Dependence-robust, conservative guarantees. How can BucketedVaR be modified to deliver non-anti-conservative VaR ${ }_\alpha$ guarantees under unknown edge-loss dependence without path enumeration? 

Q2 What is the empirical sensitivity to tail dependence? Please inject controlled correlation between edge losses (e.g., shared disturbance processes, comonotonic sampling) and report coverage error and path changes versus your baseline.

Q3. Some suggestions for baselines: (i) chance-constrained shortest path surrogates; (ii) CVaR of max; (iii) an independence-aware exact path solver (using order statistics / product CDFs) where feasible.

### Soundness
3

### Presentation
3

### Contribution
3
