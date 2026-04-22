# Revisiting the Role of Homophily in Fair Graph Representation Learning

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 2

## Abstract
Graph Neural Networks (GNNs) can propagate sensitive signals via message passing, especially on homophilous graphs where edges preferentially connect nodes sharing sensitive attributes. We revisit fairness through the lens of homophily using CSBM-S, a synthetic model that independently controls label homophily $h_y$ and sensitive homophily $h_s$, enabling precise and repeatable evaluation of group fairness. CSBM-S reveals two key observations: (i) group disparity peaks when $h_y \approx 0.5$; and (ii) bias consistently diminishes as $h_s \rightarrow 0.5$. Guided by these insights, we propose FairEST, which enforces $h_s \approx 0.5$ by flipping the sensitive attribute and its most correlated features during training. Across diverse benchmarks and backbones, FairEST attains the lowest bias on most encoder-dataset pairs with comparable accuracy, yielding average absolute reductions of 1.63% ($\Delta \mathrm{SP}$) and 1.28% ($\Delta \mathrm{EO}$) over the prior state-of-the-art. Together, CSBM-S and FairEST provide a homophily-centric toolkit for analyzing and mitigating bias in graph representation learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce CSBM-S, a controllable synthetic benchmark that decouples label homophily (h_y) and sensitive homophily (h_s), enabling precise evaluation of fairness mechanisms.
CSBM-S identifies two empirical trends: Group disparity peaks when label homophily and Bias tend to decrease as sensitive homophily
Then, based on these insights, they propose FairEST, a method that enforces ~0.5 by flipping sensitive attributes and correlated features during training to mitigate bias. Experimental results show consistent improvements in fairness metrics across baselines.

### Strengths
I like the idea of fairness in GNNs through a homophily view, offering a new conceptual angle on how topology affects bias propagation. Also, FairEST is conceptually straightforward, model-agnostic, and integrates easily into existing GNN pipelines.
The observations linking fairness with specific homophily ranges could inform future fairness-aware graph design.

### Weaknesses
- To me, GNNs inherently rely on the homophily principle, learning from neighboring nodes under propagation. Therefore, attributing fairness issues primarily to homophily may oversimplify the problem. The root causes of unfairness might instead come from global structural factors, such as community topology or node identity, rather than local structural properties like node degree or neighborhood similarity.

- The rationale for flipping sensitive attributes may appear heuristic. That is, its connection to causality or representation disentanglement could be better articulated.

### Questions
- Could the authors clarify whether the feature flipping operation might leak or distort semantic information critical for downstream tasks?

- How does FairEST perform on heterophilous graphs where h_y and h_s  are both low? Does the method still yield fairness gains?

- How sensitive is FairEST to incorrect or noisy sensitive attributes?

### Soundness
3

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
5

### Summary
This work aims to study fairness in GNNs from the perspective of homophily. Specifically, the authors focus on notions of label and sensitive attribute homophily, assessing which neighborhood patterns cause fairness degradation. Through their CSBM-S model, a synthetic graph model that controls the label and sensitive attribute homophily, the authors demonstrate that group fairness degrades as label homophily tends towards 0.5, while group fairness improves as sensitive attribute homophily tends towards 0.5. To use these findings, the authors present FairEST, a method which aims to optimize the sensitive attribute homophily at training-time to 0.5. Generally, FairEST is able to achieve decent fairness metrics, but does incur a performance cost.

### Strengths
1. I found the method sections of the paper relatively easy to read given that each section naturally follows from the previous and logically builds on one another. 
2. The empirical results, at least for the fairness metrics, tend to be decent.
3. The methods simplicity as a training-time augmentation makes it amenable to different backbones and settings.

### Weaknesses
1. My main concern about this work is that it largely uses methods and insights already well established in the literature. Moreover, the authors do not offer enough arguments as to how their work explicitly differs from these methods, sometimes missing citations all together. A few examples are: 
    
    - Homophily and fairness are already well connected in the literature. As far as I can tell, the authors do not explicitly address how their work builds on, or "revisits", these previous findings [1, 2, 3].
    - Beyond just connecting homophily and fairness, the proposed CSBM-S model and analysis in section 4 are highly similar to that in [2], both from the model design of manipulating label and sensitive attribute homophily, as well as the resulting takeaways. 
    - The idea of flipping the sensitive attribute, aiming to "debias" the message passing process, does not seem sufficiently different from previous methods which manipulate the graph structure to encourage different treatments across sensitive attributes [3, 4].

2. For the majority of the experimental results, while the fairness metrics are decent, the accuracy drops are sometimes quite large. Given my points above, I think significantly more effort needs to go in to remedy this issue and establish more novelty in the method. 

3. While trying to assess whether the author justified the performance drops, I realized there are instances in section 6.2 which do not seem to correspond to Table 1. For instance, on line 364, the reported accuracy numbers changes are significantly higher than the drops seen in the table (e.g., authors report -2.1% on GIN-bail, yet it would appear the drop is closer to 5.5% in table 1). 

In all, I think this work needs quite a bit more effort to both sufficiently ground itself in the literature and also improve presentation in the experimental section. 

[1] Wang et al. “Improving Fairness in Graph Neural Networks via Mitigating Sensitive Attribute Leakage”
[2] Loveland et al. “On Graph Neural Network Fairness in the Presence of Heterophilous Neighborhoods” 
[3] Li et al. “On Dyadic Fairness: Exploring and Mitigating Bias in Graph Connections”
[4] Rahman et al. “Fairwalk: Towards Fair Graph Embedding”

### Questions
Please see my weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies group fairness in GNNs through the lens of label homophily $h_y$ and sensitive homophily $h_s$. It introduces CSBM-S, a synthetic generator that independently controls $h_y$ and $h_s$ to analyze bias under message passing, observing that disparity peaks near $h_y \approx 0.5$ and diminishes as $h_s \to 0.5$. Building on this, the authors propose FairEST, which iteratively edits the sensitive attribute $s$ and its most correlated features to steer neighborhoods toward $h_s \approx 0.5$, with an auxiliary group-fairness loss. Experiments across multiple datasets and GNN backbones show reduced group-fairness gaps with comparable accuracy.

### Strengths
+ The motivation is clear. The paper formalizes node-level $h_y$, $h_s$, and standard group-fairness metrics ($\Delta\mathrm{SP}$, $\Delta\mathrm{EO}$), then analyzes how message passing amplifies or attenuates disparities. It further employs CSBM-S to vary $h_y$ and $h_s$ independently. Grid sweeps and a mean-field analysis yield interpretable patterns that motivate the method. 
+  FairEST is backbone-agnostic and easy to implement. Edits to $s$ and correlated features, combined with a fairness loss, reduce bias without architectural changes.
+ The experimental study is extensive, covering multiple datasets/backbones with ablations and hyperparameter analyses that reveal both gains and failure modes.

### Weaknesses
- The method assumes the sensitive attribute $s$ is observed, which may be unrealistic in high-stakes settings. Please evaluate fairness when $s$ is hidden or unavailable, e.g., by comparing to adversarial/invariant approaches and to a setting where $s$ is predicted from proxies.
-  The algorithm greedily balances neighborhood homophily $h_s$ toward $0.5$ using node-wise majority and a fixed iteration cap. It is unclear whether these local flips provably reduce global disparity or instead induce distributional shifts in $P(s)$. 
- The paper currently targets binary labels and a single binary sensitive attribute. How about the generalization to multi-class or multi-attribute settings?
-  Only group fairness is evaluated. Individual fairness is discussed conceptually but not assessed. Many applications may require both.

### Questions
Please refer to the above weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies the relationship between homophily and fairness in GNNs. They claim that the degree of label homophily $h_y$ and sensitive homophily $h_s$ significantly impacts bias amplification during message passing. To analyze this, they propose CSBM-S, a synthetic graph model that decouples label and sensitive homophily, allowing controlled experiments. They further introduce FairEST, an algorithm that enforces $h_s \approx 0.5$ to improve fairness by iteratively flipping sensitive attributes and correlated features. Experiments on several benchmarks and GNN baselines show modest improvements in fairness metrics with comparable accuracy.

### Strengths
1. The paper identifies an intersection between fairness and graph homophily. They show how label homophily and sensitive homophily could shape fairness under the message passing of GNNs.

2. The paper introduces CSBM-S as a controlled simulator for fairness studies, which allows disentangling the effects of different homophily levels in a reproducible manner, potentially benefiting future research.

3. Experiments include multiple models and datasets, and the authors conduct ablations, sensitivity analyses, and noise robustness tests.

### Weaknesses
1. The idea that message passing propagates sensitive signals via edges with attribute correlation is well-established (Wang et al., 2022; Dong et al., 2022; Dai & Wang, 2021). The notion of balancing sensitive attribute distributions (making $h_s \approx 0.5$) is just a graph-level rephrasing of feature decorrelation or resampling. FairEST’s “flip and reflect” procedure is essentially a stochastic data augmentation trick, not a theoretically or algorithmically novel approach. As such, the manuscript somewhat overstates the conceptual novelty of the approach by framing it as a “homophily-centric fairness framework,” when the underlying idea remains relatively straightforward.

2. In Section 4.4, the analysis appears to restate known results from mean-field diffusion analysis. The authors find that bias is largest when label information is weak ($h_y \approx 0.5$) and when sensitive channels dominate (extreme $h_s$). While this observation is intuitively consistent, i.e., bias tends to increase when sensitive features drive predictive signals. It does not seem to offer new theoretical insight into why or how GNN architectures amplify fairness issues.

3. Despite proposing CSBM-S, the paper does not use it to uncover deeper causal or structural insights about fairness dynamics in graphs. The synthetic experiments are mainly limited to grid-sweep heatmaps and a few straightforward observations, without quantitative analyses of robustness, sensitivity to graph topology, or comparisons with alternative fairness mechanisms. As a result, the proposed “homophily-centric toolkit” currently functions more as a synthetic data generator than as a framework for deeper theoretical understanding.

### Questions
1. How does FairEST compare with trivial strategies such as random node feature shuffling, attribute dropout, or standard reweighting schemes?

2. Are the fairness improvements statistically significant across runs? Please report confidence intervals or p-values. 

3. The method assumes full access to sensitive attributes during training and precise correlation estimation, which is rarely met in practical settings. How would FairEST operate under partial or uncertain sensitive attribute availability?

4. Have the authors tested on larger or more realistic graphs (e.g., OGB datasets)? The current experimental setup lacks scalability evidence.

5. Does enforcing $h_s \approx 0.5$ actually remove causal influence of the sensitive attribute, or does it merely mask correlations?

### Soundness
2

### Presentation
2

### Contribution
2
