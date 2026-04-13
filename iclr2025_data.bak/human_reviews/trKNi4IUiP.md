## Human Reviewer 1

### Summary
The paper introduces RIGBD, a novel framework for defending Graph Neural Networks (GNNs) against backdoor attacks. It leverages **random edge dropping** to identify poisoned nodes by detecting high prediction variance, and incorporates a **robust training strategy** to reduce the prediction confidence on these nodes, limiting the impact of triggers. The method is theoretically sound and empirically validated on real-world datasets, showing it can effectively decrease attack success rates while maintaining clean accuracy. Key strengths include scalability and robustness, though the method relies on hyperparameter tuning and may face challenges with indistinguishable poisoned nodes.

### Strengths
1. Effective Identification of Poisoned Nodes: The use of random edge dropping as a technique for distinguishing poisoned nodes from clean nodes is innovative and theoretically backed. The method efficiently handles large-scale graphs by reducing the computational burden that would otherwise be associated with processing each edge individually.

2. Stability for Clean Nodes: The approach ensures that clean nodes maintain prediction stability, which prevents sensitivity toward non-poisoned nodes.

### Weaknesses
1. Dependency on Hyperparameters: The effectiveness of the random edge drop rate (𝛽) and the number of iterations (𝐾) can vary significantly across different datasets, requiring careful tuning and potentially reducing generalizability. Additionally, there is no theoretical guarantee for selecting the optimal values for these hyperparameters.

### Questions
1. In the related work, the SBA method was mentioned. While I understand that SBA may have a low ASR, I am unsure about how your method defends against SBA specifically, as there are no experimental results regarding this method. Could the authors provide a simple experiment to illustrate the defense mechanism against SBA?

2. The paper does not mention any adaptive attacks. Are there any potential ways to design adaptive attacks targeting this method?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper shows that poisoned nodes exhibit high prediction variance with edge dropping. The authors propose random edge dropping, supported by theoretical analysis, to efficiently and precisely identify poisoned nodes. Furthermore, the authors introduce a robust training strategy to efficiently counteract the impact of the triggers, even if some poisoned nodes remain unidentified.

### Strengths
1. The structure of the paper is clear and easy to follow.
2. The paper conducts comprehensive experiments to demonstrate the performance of proposed method.

### Weaknesses
1. There are concerns about the efficiency of the method. First, the proposed method requires training the GNN encoder twice, which incurs a large overhead cost, especially in larger datasets such as obg-products[1]. In addition, when calculating the prediction variance, it is necessary to infer K times on the graph, which also brings a large overhead.
2.  The proposed method seems to rely on the homogeneity assumption. In heterophilic graphs, nodes tend to connect to nodes that share different labels with them. Would this result in normal nodes also having higher prediction variance after random edge dropping? It is recommended that the author can give the performance of this method on backdoor attacks in heterophilic graph datasets.

[1] Open Graph Benchmark: Datasets for Machine Learning on Graphs, NeurIPS 2020.

### Questions
1. Targeted attack generates attacks on specific nodes and aims to fool GNNs on these target nodes. Can the proposed method defend targeted attack strategies on graphs, such as nettack[1].
2. RIGBD minimizes the predicted confidence of the target class of the poisoned node in Eq. 6, and uses the remaining labeled nodes to calculate the cross entropy loss. If the number of attacked nodes is too large, will this cause insufficient supervision information?

[1] Adversarial attacks on neural networks for graph data, KDD 2018.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper presents Robustness-Inspired Graph Backdoor Defense (RIGBD) to defend against subgraph-based graph backdoor attacks. The authors demonstrate that poisoned nodes exhibit significant prediction variance when anomalous edges linked to them are removed, and propose a robust training strategy to mitigate the effects of backdoor triggers. Their method effectively identifies poisoned nodes, reduces attack success rates, and preserves clean accuracy across various datasets.

### Strengths
- RIGBD is specifically designed for graph-structured data, effectively leveraging the behavior of malicious edges linking trigger subgraphs to poisoned nodes. The core insight—that the implanted trigger influences prediction by transmitting malicious information through these edges—is particularly valuable in identifying poisoned nodes.

- This paper introduces an intriguing observation: simply removing backdoor triggers does not necessarily immunize the model against backdoor attacks. This insight opens up new directions for designing graph backdoor defense.

- The paper provides a thorough theoretical analysis that explains why random edge dropping can effectively distinguish poisoned nodes from clean ones.

### Weaknesses
- Though experiments demonstrate the superiority of RIGBD against subgraph-based backdoor attacks, its resistance against other forms of graph backdoors, e.g., injecting backdoors in the spectral domain [1], remains unevaluated.

- The method proposed for determining target nodes and labels seems to lack robustness. Specifically, the authors suggest ranking nodes in descending order based on prediction variance after random edge dropping, selecting the top-ranked nodes until a node label differs from the previous one. This approach could be susceptible to outliers or special node characteristics, such as nodes with few neighbors and high class-diversity among them, which may result in large prediction variance under random edge dropping. Such nodes may halt the poisoned node selection process or even lead to incorrect target class identification.

[1] Zhao, Xiangyu, Hanzhou Wu, and Xinpeng Zhang. "Effective Backdoor Attack on Graph Neural Networks in Spectral Domain." IEEE Internet of Things Journal (2023).

### Questions
- Would RIGBD be effective for non-subgraph graph backdoor attacks?
- How does RIGBD apply to all-to-all backdoor attacks, where different samples have different target labels?
- How does RIGBD adapt to the special case mentioned in weakness 2?

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper proposes a novel defense method against graph backdoor attacks, which is composed of poisoned nodes detection and robust training. In poisoned node detection, this paper first observes that edge dropping significantly influences the prediction of the poisoned nodes and then this has been theoretical verified. After this, this paper proposes to utilize random edge dropping to detect poisoned nodes efficiently based on poisoned GNN model. After detection, the poisoned model will be finetuned by the detected nodes to remove the influence of the backdoor attack.

### Strengths
1. The idea is novel and interesting, and is well evaluated empirically and theoretically.
2. In addition to the detection of poisoned node, this paper also proposes robust training to enhance defense performance further. This can safeguard GNN models against different kinds of attacks.
3. A well-written paper, and easy-to-follow.

### Weaknesses
1. To verify the effectiveness and generalization of the proposed defense, I suggest author deeply discussing the mechanism of dirty-label backdoor attacks against GNNs. 
2.  I find out that the discussion among baselines and the proposed defense is missing, e.g., why the proposed work only performs better on defending DPGBA. This also weakens the contribution of this work. Finally, I find that the baselines are not targeted at graph backdoor attacks, e.g., GNNGuard targets defending adversarial attacks while ABL targets traditional DNNs. It is important to compare with other defense strategies against graph backdoor attacks. 
3. I find that the idea of this paper is very similar to the motivation of the influence function which aims to measure the impact of deleting edges and nodes by influence function (in this paper authors use predicted probability) and does not require retraining or fine-tune the poisoned model. I suggest authors discuss and compare with backdoor defense strategies using influence function.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
2