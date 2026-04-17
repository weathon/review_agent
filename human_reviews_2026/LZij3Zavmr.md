# Evading Overlapping Community Detection via Proxy Node Injection

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Protecting privacy in social graphs requires preventing sensitive information, such as community affiliations, from being inferred by graph analysis, without substantially altering the graph topology. We address this through the problem of \emph{community membership hiding} (CMH), which seeks edge modifications that cause a target node to exit its original community, regardless of the detection algorithm employed. Prior work has focused on non-overlapping community detection, where trivial strategies often suffice, but real-world graphs are better modeled by overlapping communities, where such strategies fail. To the best of our knowledge, we are the first to formalize and address CMH in this setting. In this work, we propose a deep reinforcement learning (DRL) approach that learns effective modification policies, including the use of proxy nodes, while preserving graph structure. Experiments on real-world datasets show that our method significantly outperforms existing baselines in both effectiveness and efficiency, offering a principled tool for privacy-preserving graph modification with overlapping communities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the Community Membership Hiding problem in the challenging, realistic setting of overlapping communities. The authors propose a deep reinforcement learning framework that injects proxy nodes and learns an optimal edge modification policy to hide a target node's affiliation. Experiments show this method significantly outperforms baselines.

### Strengths
1. The paper addresses the significant problem of privacy in graph analysis by formally defining and tackling Community Membership Hiding in the realistic setting of overlapping communities.

2. The proposed solution combines a deep reinforcement learning framework with a proxy node injection technique to learn effective graph modifications.

3. The paper provides an empirical evaluation on real-world datasets, demonstrating that the proposed method significantly outperforms baselines.

### Weaknesses
1. The paper's claim to be the "first to formalize and address CMH under overlapping community detection" may need to be revisited. The problem of hiding membership in overlapping communities has been explored in prior works [1][2][3]. The authors should refine their contribution statement to more clearly distinguish their work from these existing studies and conduct a more comprehensive survey of related literature.

2. The current baselines are primarily variations applied within the proposed proxy-injection framework. The comparison should include established, state-of-the-art methods specifically designed for overlapping community hiding.

3. The proposed ODRL method requires repeated queries to a community detection oracle, which can be computationally intensive. The current experiments are conducted on relatively small graphs. To demonstrate practical viability for real-world applications, it would be beneficial to evaluate the method on larger-scale datasets and discuss potential optimization strategies to mitigate the computational overhead.

4. The paper could be further strengthened by including theoretical analysis to provide a more formal justification for the proposed ODRL framework's design and efficacy.

References:

[1] Liu D, Yang G, Wang Y, et al. How to protect ourselves from overlapping community detection in social networks[J]. IEEE Transactions on Big Data, 2022, 8(4): 894-904.

[2] Yang G, Wang Y, Chang Z, et al. Overlapping community hiding method based on multi-level neighborhood information[J]. Symmetry, 2022, 14(11): 2328.

[3] Han Z, Shi L, Wang Y, et al. CoHide: Overlapping community hiding algorithm based on multi-criteria learning optimization[J]. Peer-to-Peer Networking and Applications, 2025, 18(2): 94.

### Questions
1. Could the authors clarify the specific novelty of their problem formulation or technical approach that distinguishes it from prior work which also appears to address overlapping community hiding?

2. To better contextualize the method's performance, could the authors elaborate on why established, external baselines from the literature were not included?

3. Given that the proposed framework requires repeated calls to a community detection oracle, could the authors provide the practical training and inference runtimes for the experiments? Furthermore, could the authors discuss the optimization strategies for scaling this approach to much larger, real-world graphs?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper formalizes community membership hiding (CMH) for overlapping community detection, where nodes naturally belong to multiple communities. The authors propose ODRL, a deep reinforcement learning agent employing Proximal Policy Optimization (PPO) with a factored policy to inject proxy nodes and modify edges, concealing a target node's community affiliation. Experiments on five real-world datasets demonstrate substantial F1 score improvements over heuristic baselines in both symmetric (same detector for training/testing) and asymmetric (different detectors) settings. While the problem formulation is novel and results promising, concerns regarding experimental rigor, scalability, detectability analysis, and practical deployment warrant careful consideration.

### Strengths
- Section 4 The formalization of CMH for overlapping detection is a meaningful contribution. Figures 2-3 compellingly show that naïve proxy injection achieves F1 scores of 0.6-0.9 for non-overlapping detectors but drops below 0.3 for overlapping detectors, clearly establishing problem non-triviality.​
- Section 5: The factored policy decomposition reduces computational complexity from O(|V|²) to O(|V|) while enabling action masking. The 4-layer GCN with jumping knowledge connections and GRU temporal representation reflects careful design. The reward function (Eq. 3) balances immediate success with incremental progress via similarity differential δtsim​.
- Tables 2-3, Figures 4: The asymmetric setting (training on ANGEL, testing on DEMON) demonstrates cross-algorithm generalization crucial for real-world deployment. ODRL maintains strong performance with 20-70 percentage point F1 improvements over baselines on large datasets.​
- ODRL's advantage increases with graph size and complexity, achieving 60-90% F1 scores on pow and nets where baselines approach zero, suggesting meaningful structural pattern learning.​

### Weaknesses
- Baselines use fixed k=μ while ODRL explores k∈{0.5μ, μ, 2μ}, making it unclear whether gains stem from the learned policy or expanded budget exploration. Missing are comparisons with Silvestri et al. (2025)'s gradient-based optimization and sophisticated node injection attacks (GANI, TDGIA, G-NIA) from adversarial graph learning. Critical ablations are absent, no evaluation of ODRL without proxies (P=∅) or flat vs. factored policy comparisons.​​
- While acknowledging expensive oracle calls, no wall-clock times, oracle queries per episode, or memory requirements are reported. Excluding fb-75 because experiments "could not be completed within a reasonable timeframe" is concerning for modest-sized graphs. No computational comparison with prior work (Bernini et al., 2024; Silvestri et al., 2025) is provided.​
- Evaluation uses only two label propagation algorithms (ANGEL, DEMON) with fixed φ=0.8. A single asymmetric test (DEMON) provides insufficient evidence for "detector-agnostic" claims. Sensitivity to merging threshold φ is unexplored.​​
- PPO selection over TRPO/A3C/SAC lacks justification. No sensitivity analysis for λ=0.1, 4 GCN layers, d_h=32, or proxy edge probability p=0.5. The threshold τ=0.5 remains constant despite Eq. 1 indicating difficulty scales with τ.
- No convergence guarantees, sample complexity bounds, or characterization of graph properties (density, modularity, clustering coefficient) affecting success rates.

### Questions
Questions are majorly if they could explain the weaknesses.

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
The paper introduces the problem of Community Membership Hiding (CMH) under overlapping community detection, aiming to prevent inference of sensitive affiliations in social graphs. 
The authors propose a Deep Reinforcement Learning (DRL) framework, called ODRL that injects proxy nodes and optimally rewires edges to hide a target node's community membership.
The approach is evaluated on multiple real-world graph datasets and compared to heuristic and prior DRL-based baselines.
The method achieves higher F1 scores, balancing success rate and community preservation, and empirically shows generalization to unseen detection algorithms.

### Strengths
* **Well motivated research direction.**
The paper convincingly shows the shortcomings of naive, common strategies in the overlapping setting they consider.
The factored policy seems to be an effective approach for efficiency.


* **Promising empirical results**.
The empirical evaluation shows good results, with the proposed ODRL method generally outperforming the considered baselines with significant margin.
The approach works solidly on the asymmetric setting considered (`angel` $\to$ `demon`).

### Weaknesses
* **Presentation and clarity.**
The paper is very difficult to parse for someone not already very familiar with related work. 
A number of methods (see next comment) are simply mentioned without any explanation or intuition of their inner workings and advantages/disadvantages. 
For instance, `angel` and `demon` are presented as community detection algorithms with no details, mentioning that they are considered with a "merging threshold" of 0.8.
If not familiar with the two methods, it is difficult to attach any meaning to similar statements.
This heavily affects the clarity of the exposition and makes the contribution more difficult to judge.

* **Presentation of experimental results.**
Building up on the previous point, while I appreciate the usefulness of presenting experimental results early on, the current display of results is difficult to parse.
In Figure 2, the methods and datasets are presented without description.
For instance, "Greedy", "Louvain", and "Walktrap" are not introduced or referenced.
Similar considerations hold for Figure 3.

* **Claims of scalability.**
You discuss how your approach shows more notable advantages in large-scale graphs.
At the same time, you mention that scalability is one of the limitations your approach faces.
I think this point should be more thoroughly addressed. 
The datasets you consider have a relatively small number of nodes.
And while I would not expect you to provide results on graphs with "billions of nodes", it is unclear whether the graphs you consider are the largest you could practically consider.
At the same time, you do not quantify the computational/time effort your experiments require.

* **Policy optimality.**
In your contributions, you mention that you obtain an "optimal policy" with your DRL-based approach.
First, the term "optimal" is unclearly defined for me.
Then, it is unclear to me how some notion of optimality can hold, given the large search space.

 ### Other comments
 
 * The acronym ODRL is never introduced.
 * In line 137, should it be $>1$ rather than $\ne 1$? Or is the case where a node belongs to no community also considered?
 * It is unclear whether the $\lambda$ in Equation 3 is the same as in line 363.

### Questions
* How computationally expensive is it to consider larger graphs? 
* Can the proxy node injection be detected in adversarial settings? How can this be defended against?
* How does the choice of $\lambda$ affect the reward function?
* Can you expand on the claims of optimality for your policy? Is there any formal justification to the claim I may have missed?

### Soundness
2

### Presentation
2

### Contribution
2
