# Adversarial Graph Neural Network Benchmarks: Towards Practical and Fair Evaluation

- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Adversarial learning and the robustness of Graph Neural Networks (GNNs) are topics of widespread interest in the machine learning community, as documented by the number of adversarial attacks and defenses designed for these purposes. 
While a rigorous evaluation of these adversarial methods is necessary to understand the robustness of GNNs in real-world applications, we posit that many works in the literature do not share the same experimental settings, leading to ambiguous and potentially contradictory scientific conclusions. 

In this benchmark, we advocate for standardized, rigorous evaluation practices in adversarial GNN research. We perform a comprehensive re-evaluation of seven widely used attacks and eight recent defenses under both poisoning and evasion scenarios, across six popular graph datasets. Our study spans over 437,000 experiments conducted within a unified framework.

We observe substantial differences in adversarial attack performance when evaluated under a fair and robust procedure. Our findings reveal that previously overlooked factors, such as target node selection and the training process of the attacked model, have a profound impact on attack effectiveness, to the extent of completely distorting performance insights. These results underscore the urgent need for a standardized evaluation framework in adversarial graph machine learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new benchmarking protocol for adversarial robustness in GNNs. The authors argue that the lack of standardized evaluation protocols in existing literature has led to inconsistent and potentially misleading conclusions about the robustness of GNNs. To address this, they propose a rigorous and unified framework and use it to re-evaluate seven popular adversarial attacks and eight defense mechanisms across six datasets. The study also introduces a simple baseline attack, L1D-RND, which achieves surprisingly competitive results,

### Strengths
The analysis of target node selection is particularly impactful, demonstrating that high-degree nodes are significantly more robust to attacks. The introduction of the L¹D-RND baseline is a simple but powerful methodological choice. Its strong performance against both vanilla and defended models serves as a crucial sanity check.

### Weaknesses
While the topic is suitable for NeurIPS/benchmarks, some framing still reads like a broader methods survey plus pipeline paper.

### Questions
Q1: The performance of the naive L¹D-RND baseline is surprisingly strong, especially against defended models. Do you have a hypothesis for why this simple, non-optimized method is so effective?

Q2: The paper shows that the performance ranking of attacks like Nettack and FGA changes dramatically between homophilic and heterophilic datasets. Does your analysis provide any intuition why?

### Soundness
3

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
This paper presents a large-scale benchmark study for adversarial robustness in Graph Neural Networks (GNNs). The authors re-evaluate seven commonly used adversarial attacks (e.g., Nettack, FGA, PR-BCD, SGA, GOttack) and eight defense methods under unified experimental conditions, spanning six datasets (Cora, Citeseer, Pubmed, Chameleon, Squirrel, OGB-Arxiv). The study performs over 437,000 experiments to identify inconsistencies and biases in previous literature and proposes a standardized evaluation framework for fair, reproducible comparison of adversarial GNN methods.
Notably, the paper introduces a simple baseline attack, L1D-RND, which surprisingly achieves competitive performance, revealing that many existing complex attacks might be over-claimed.

### Strengths
* Strong motivation and clarity of purpose. The authors clearly identify the reproducibility and fairness problems in existing adversarial GNN research.

* Massive and rigorous experimental effort. Over 400k runs under controlled conditions is impressive, and the inclusion of both evasion and poisoning settings makes the study comprehensive.

* Methodological contribution. The risk-assessment-based evaluation pipeline, incorporating model selection and random splits, significantly improves fairness and reliability.

* Insightful analysis. The findings on the effects of node degree, model selection, and dataset type (homophily vs. heterophily) are valuable and highlight overlooked pitfalls in prior works.

### Weaknesses
* Limited novelty beyond benchmarking. The work does not propose new algorithms or defense strategies; its main contribution lies in evaluation methodology rather than new technical innovation.

* Analysis depth on defenses is limited. While the results include many defenses (e.g., GNNGuard, RUNG), the paper does not provide deeper insights into why some defenses succeed or fail under different graph structures.

* Overemphasis on L1D-RND baseline. Although the naive attack’s competitiveness is an interesting observation, it occupies a disproportionate part of the discussion, while more meaningful interpretive analysis could be expanded.

* Scalability constraints remain. The paper points out that most existing attacks cannot scale beyond moderate graph sizes, yet it does not propose or analyze new scalable solutions.

* Writing issues in some sections. Certain parts of the experiments and tables are overly dense, making the presentation hard to follow; the narrative could be more concise and reader-friendly.

### Questions
See weaknesses

### Soundness
3

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
5

### Summary
The authors are interested in the subject of adversarial attacks for Graph Neural Networks. Specifically, the work tackles the challenge of fair comparison and evaluation of the current available GNNs and the proposed defense methods, by proposing a unified experimental setup, taking into account different factors.

### Strengths
- The tackled problem of GNN’s adversarial robustness is important to ensure better adoption of these methods.
- Some methods and evaluation consider different settings than the baselines, creating therefore some difference in terms of performance which is not due to the method itself but rather to hyper-parameters - unifying therefore the experimental setup is of great gain to the community.

### Weaknesses
- The authors considers the adversarial accuracy (or attacked accuracy) as the main point of comparison. While this is already a good element, I believe that in some specific settings, one should also take into account the complexity of the proposed defense method. In this perspective, I would suggest adding a Table of comparison in terms of complexity or training/evaluation time of each method. Such direction could give some credits to other methods that are rather aiming to have a simpler and yet effective approaches to defending adversarial robustness (for instance NoisyGNN [1])
- While a number of hyper-parameters and other elements are considered in the paper, I think important missing element to ensure the claimed and desired fairness is the initialization. As previously studied in [2], changing the initialization can have a great impact of the performance of a method. I was wondering if it’s possible to discuss such direction and how you ensured fairness in this aspect. 
- The paper focuses on node classification with the topological attacks. While it’s already a big and important part of current research, I was wondering if there was a plan to extend to node feature-based adversarial attacks (in which different methods have been proposed [3])?

- [Minimal] The presentation format of the results could be enhanced. For instance through visualization. While I liked the table with all the results, it’s not very easy to compare all the methods and understand the ranking. By providing some visualizations (for instance a Radar-Chart plot), it could be easier to see the difference between defenses when subject to an attack for each datasets. 

Typically, I would be very interested in seeing the discussion regarding the initialization and the time complexity in the revised manuscript for the rebuttal. I would also expect the authors to better cern their perspective on extending their framework and evaluation to the context of feature-based attacks. 

—

[1] A Simple and Yet Fairly Effective Defense for Graph Neural Networks. - AAAI 2024.

[2] If You Want to Be Robust, Be Wary of Initialization. - NeurIPS 2024.

[3] Graph Neural Networks with Adaptive Residual. - NeurIPS 2021.

### Questions
- Could you provide clarification element regarding ensuring fairness when taking into account initialization? 
- Can you provide benchmark regarding the performance in terms of complexity, to better understand the trade-off performance/complexity? 
- Are there any plans to extend the study for node feature-based adversarial attacks? 
- Could you also provide performance of the different methods in terms of certifications to better understand their theoretical performance?

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
4

### Summary
This paper presents a comprehensive and rigorous benchmark for evaluating adversarial attacks and defenses on Graph Neural Networks (GNNs). The authors identify critical issues in the current literature, such as inconsistent evaluation protocols and biased experimental setups, which lead to unreliable or incomparable conclusions. To address this, they propose a standardized framework that incorporates realistic model selection, diverse target node sampling strategies, and evaluation across multiple random data splits.
Within this framework, the authors conduct a large-scale re-evaluation of seven prominent attack methods and eight defense mechanisms across six diverse graph datasets (including homophilic, heterophilic, and large-scale graphs) under both poisoning and evasion settings.

### Strengths
1.	The paper’s decision to focus on creating a fair and standardized benchmark directly addresses a fundamental issue hindering progress in adversarial GNN research. This methodological contribution is highly valuable, as it provides the community with a tool to filter out unreliable results and foster more reproducible science.
2.	The experimental framework itself is a key strength. Its design is exceptionally rigorous, incorporating crucial elements like proper model selection and diverse target node sampling that are often overlooked. This approach ensures that the comparisons are not only fair but also reflect more realistic application scenarios.
3.	The discovery that target node selection and victim model tuning can drastically alter outcomes is a major finding. The paper provides compelling evidence that many prior works may have systematically overestimated attack effectiveness. This insight is powerful and forces a necessary re-evaluation of the community's collective understanding of GNN fragility.

### Weaknesses
1.	In the paper, it seems that the evaluation of adaptive attacks is somewhat limited. While including an adaptive variant of one attack is a good step, a truly robust defense should be tested against attackers that are specifically designed to circumvent its core mechanism. It may help that the authors could discuss the scope of their adaptive evaluation and acknowledge that stronger, tailored attacks might exist for some of the tested defenses.
2.	In the experiments, the analysis on the large-scale OGB-ARXIV dataset seems to be constrained by computational resources. This is understandable, but it does mean that conclusions about scalability are based on a limited set of successful runs. At this time, this might temper the strength of these specific claims. I would encourage the authors to be more explicit about this limitation in the main text.
3.	The paper reports that many attacks failed on the large graph due to timeouts, which is an important finding. However, the reasons for these failures are not deeply analyzed. It would be more insightful to understand if the bottleneck is memory, a specific computational step, or another factor. A more detailed breakdown of these failure modes would provide valuable lessons for designing future scalable algorithms.

### Questions
1.	I am curious about your findings on adaptive attacks[1]. Did you observe if making an attack "adaptive" (i.e., aware of the defense) had any impact on its general attack strength? For instance, does an attack become less effective overall when it is modified to bypass a specific defense? A brief comment on this potential trade-off would be very interesting.
2.	The findings on the OGB-ARXIV dataset are very insightful. Beyond reporting the timeouts, do you have any further analysis on the specific bottlenecks that prevent methods from scaling? Understanding whether the primary limitation is memory usage or computational time for specific operations would be very helpful for future algorithm design.
3.	The results clearly show that high-degree nodes are inherently more robust to the attacks you tested. What implications does this have for the design of future defense mechanisms? Does it suggest that defenses should focus their efforts on protecting low-degree or fringe nodes, rather than applying a uniform protection strategy across the entire graph?

[1]Felix Mujkanovic, Simon Geisler, Stephan Günnemann, and Aleksandar Bojchevski. Are defenses for graph neural networks robust? Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 2022.

### Soundness
3

### Presentation
3

### Contribution
3
