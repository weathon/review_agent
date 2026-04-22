# Topology-Aware Knowledge Propagation in Decentralized Learning

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Decentralized learning enables collaborative training of models across naturally distributed data without centralized coordination or maintenance of a global model. Instead, devices are organized in arbitrary communication topologies, in which they can only communicate with neighboring devices. Each device maintains its own local model by training on its local data and integrating new knowledge via model aggregation with neighbors. Therefore, knowledge is propagated across the topology via successive aggregation rounds. We study, in particular, the propagation of out-of-distribution (OOD) knowledge. We find that popular decentralized learning algorithms struggle to propagate OOD knowledge effectively to all devices. Further, we find that both the location of OOD data within a topology, and the topology itself, significantly impact OOD knowledge propagation. We then propose topology-aware aggregation strategies to accelerate (OOD) knowledge propagation across devices. These strategies improve OOD data accuracy, compared to topology-unaware baselines, by 123% on average across models in a topology.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper empirically studies the impact of the topology of network, such as the degree of nodes, the number of nodes, and modularity, on the propagation of out-of-distribution (OOD) knowledge in distributed model training, finds that existing model aggregation methods can not propagate OOD knowledge effectively and proposes a topology-aware aggregate framework for enhancing the of propagation of OOD knowledge. The effectiveness of the proposed aggregation strategies are verified by extensive experiments.

### Strengths
* It is an important question that how network topology impacts the performance of distributed model training algorithms.

* This paper proposes new aggregation algorithms by incorporating the topology information, such as degree and betweenness centrality.

### Weaknesses
* The empirical findings in this paper is incremental in comparison with previous ones in [1]. It is better to provide some theoretical results. 

* There is a lack of proper definition regarding the performance measurement of algorithms (or the measure for knowledge propagation). In line 184, this paper uses the average AUC of all models in a topology on a given test set to measure the knowledge propagation. It is not a proper measurement. Since each device maintains its own local model, it is better to measure the test performance of each model on its own test data.

* There is a lack of theoretical analysis on the generalization error bound of the proposed algorithms. It is intuitive that the generalization error bound should depend on the topology of network, such as the connectivity of network, the degree of nodes and the location of OOD datasets.

References

[1] Palmieri et al. Impact of network topology on the performance of decentralized federated learning. CoRR, 2024.

### Questions
Please refer to the above-mentioned weaknesses.

### Soundness
3

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
This paper tackles an important and challenging problem in decentralized learning: the effective propagation of out-of-distribution (OOD) knowledge across a network topology. The authors argue that traditional aggregation strategies, being unaware of the network structure, hinder OOD knowledge propagation, making performance highly dependent on the location of the OOD data source. To address this, the paper introduces "topology-aware aggregation strategies", which leverage network centrality metrics like node degree and betweenness to weight the model aggregation process, thereby accelerating the diffusion of critical knowledge. Through extensive experiments across various topologies and datasets, the authors demonstrate that their proposed methods significantly outperform existing topology-unaware baselines in improving OOD accuracy.

### Strengths
Strong Motivation:The problem of OOD knowledge propagation is highly practical and critical in real-world decentralized learning scenarios, such as IoT and edge computing.

Comprehensive Experiments:The experimental setup is very extensive, covering a wide range of variables and providing strong empirical evidence for the conclusions.

Theoretical Support:The spectral gap analysis in the appendix offers a plausible theoretical explanation for why the topology-aware strategies are superior, moving beyond purely empirical findings.

### Weaknesses
There is a lack of clear definitions for OOD and IID knowledge in the decentralized context, which obscures the relationship and distinction between them.

The paper provides insufficient justification for selecting degree and betweenness centrality over other potential topological metrics, leaving it unclear whether these are the optimal choices.

The analysis is confined to static topologies; applicability and potential overhead in dynamic networks are not discussed.

### Questions
There are some questions as follows：

1.Could you provide clearer definitions for IID and OOD knowledge in the decentralized context to better distinguish them?

2.What was the rationale for choosing degree and betweenness centrality over other metrics, and is there evidence they are optimal?

3.How would your method perform in dynamic topologies, and what is the expected overhead from re-calculating centrality?

4.Given that your topology-aware approach intentionally amplifies the influence of central devices, how do you propose to mitigate the heightened security risk of a compromised central device rapidly spreading malicious knowledge throughout the network?

5.Could you elaborate on the relationship between your specific OOD setup and more general Non-IID settings, like label skew across all devices?

6.Regarding contribution two, you state the sensitivity to topology is "a problem which does not exist in FL". What "problem" does this refer to, and why is it absent in FL?

7.How would your method's performance change if all devices had varying degrees of Non-IID data, instead of the current single-source OOD setting?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work studies the knowledge propagation in decentralized learning. In particular, the authors find that popular algorithms struggle to propagate out-of-distribution (OOD) knowledge to all devices. They further propose two topology-aware aggregation strategies to solve the problem. The experiments demonstrate the effectiveness of the proposed methods.

### Strengths
1. The proposed topology-aware aggregation strategies (Degree and Betweenness) are intuitive, simple to implement, taking in account the effect of topology on the learning effect.

2. The authors conduct experiments across five different datasets and multiple varied topologies and systematically study the impact of topology degree, modularity, and node count, providing a comprehensive characterization of the solution's performance.

### Weaknesses
1. The main concern is about the assumption of the data distribution and OOD data definition.I suspect the practicality of only one node having the OOD data. Besides, the experimental setup defines OOD data by a "backdoor" methodology (i.e., inserting triggers), as mentioned in Appendix B.2.2. This setup is very narrow and seems artificial. Also, this definition is different from the commonly used term OOD. The findings in this work may be hard to generalize to more natural setups.

2. The topology in this work is assumed to be static. However, in practice, the connection in IoT or edge computing networks is usually dynamic. When the topology changes, the definition of Betweenness may also vary.

3. This work relies on a synchronous communication model, which is a strong assumtion that does not reflect many real-world decentralized system.

### Questions
1. Given that the OOD data is defined as a backdoor, how can the method generalize to natural OOD scenarios, such as a node holding an entirely new data class?

2. If OOD data is not concentrated on a single node but is sparsely distributed across multiple random nodes, would the topology-aware strategy still be effective?

3. Since the method accelerates the propagation of all knowledge, how does it prevent being exploited to accelerate the spread of malicious backdoor or model poisoning attacks?

4. Did the study measure the impact of node churn (joining and leaving), which is critical for IoT or edge computing scenarios, on OOD knowledge propagation?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of propagating OOD knowledge across devices in decentralized learning. Unlike federated learning’s centralized aggregation, decentralized learning models only exchange updates with neighbors in a communication graph. This paper shows that existing topology-unaware aggregation methods struggle to disseminate OOD information: OOD accuracy lags IID accuracy by a large margin and is highly sensitive to both where OOD data originate and the network structure. To remedy this, this paper introduces simple topology-aware aggregation strategies that weight neighbor models by centrality metrics.

### Strengths
1. Clear identification of OOD propagation as a distinct and critical problem in decentralized learning.
2. Elegant, low-overhead methods that seamlessly integrate with existing gossip algorithms.
3. Comprehensive evaluation: five datasets, three topology families, varying OOD locations, and spectral-gap theory.

### Weaknesses
1. The paper assumes a single-device OOD “worst case” but does not analyze scenarios with multiple OOD sources, which may arise in practice and interact nonlinearly.
2. Only degree and betweenness are studied, yet other centrality metrics (e.g. eigenvector, closeness) might offer better trade-offs; appendix C claims negligible cost but no profiling of alternative metrics is shown.
3. Betweenness computation, even if amortized,scales superlinearly (O(nm + n² log n)), and while Table 3 reports <1 s for ≤1,000 nodes, real networks can be much larger; no discussion of dynamic topologies or incremental updates.
4. Hyperparameter τ is fixed at 0.1 for both metrics without ablation; it’s unclear how sensitive performance is to temperature choice.
5. The backdoor approach (10% red-square or token triggers) may not generalize to natural OOD shifts; more realistic distribution shifts (e.g. novel classes or domain changes) are not evaluated.
6. While memory/round messages stay the same, degree-based weighting implicitly assumes knowledge of global centrality, requiring initial full-graph exchange or central computation; the mechanism to share topology information securely is not detailed.
7. Beyond spectral-gap bounds, there is no formal convergence or generalization guarantee for non-convex objectives under topology-aware weighting, especially for heterogeneous data.
8. This work identifies and tackles OOD knowledge propagation in decentralized learning via two simple topology-aware aggregation schemes. The extensive empirical results validate their benefits, but broader applicability and scalability regarding multiple OOD sources, dynamic graphs, realistic shifts, and parameter sensitivity remain to be addressed.

### Questions
please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2
