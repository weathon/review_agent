# Sharing is Sabotaging: Cross-Client Poisoning Attacks on Federated Graph Learning

- Decision: Reject
- Scores: 2, 4, 8, 6

## Abstract
Federated Graph Learning (FGL) enables collaborative training of Graph Neural Networks (GNNs) without sharing raw data. While prior work has focused on privacy leakage risks, we reveal a more direct and structurally embedded threat: a gray-box poisoning attack that manipulates shared neighbor representations during training. Specifically, we show that auxiliary-information-sharing frameworks, which collaboratively train node generators, create novel and powerful attack surfaces. A malicious client can poison shared gradients during generator updates, causing benign clients to unknowingly incorporate compromised synthetic nodes into their local subgraphs. These poisoned nodes propagate corrupted signals during training, resulting in persistent model degradation. We formalize this threat model, propose a stealthy optimization-based attack, and demonstrate its effectiveness in degrading model performance while evading standard defenses. The attack is highly stealthy, model-agnostic, and remains effective across diverse datasets, partition strategies, and numbers of clients. Our findings highlight a critical vulnerability in FGL systems and underscore the urgent need for dedicated robustness mechanisms against attacks targeting the data-generation layer.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a new poisoning attack, Indirect Gradient Poisoning Attack (IGPA), targeting auxiliary-information–sharing frameworks in Federated Graph Learning (FGL), such as FedSage+. Unlike traditional attacks that directly tamper with model parameters or local data, IGPA manipulates the gradients exchanged during the training of neighbor generators (NeighGen). By injecting malicious updates into these shared gradients, an adversarial client can cause benign clients to generate poisoned synthetic nodes, which subsequently degrade the global model’s performance. The attack operates in a gray-box setting, maintains stealth by aligning adversarial gradients with benign ones, and remains effective across datasets (Cora, Citeseer, PubMed, MSA).

### Strengths
- This paper introduces an innovative poisoning vector exploiting gradient sharing in generator-based FGL, expanding the attack surface beyond known methods.
- This paper provides a rigorous problem formulation and detailed methodological breakdown (surrogate model, gradient crafting, stealth enhancement.

### Weaknesses
- The main concern is the weak attack performance reported in Table 1. The maximum attack performance does not exceed 50% for all four datasets. The reviewer is not sure whether such performance is convincing. In addition, as discussed in Section 3.5, the stealthiness seems to be highly sensitive to \belta and \lambda. Given the limited discussion on these two key parameters, the reviewer may be concerned about the effectiveness of the proposed attack.
- The proposed scheme seems to have a limited scope with FedSage+. If there is no available node generator in the framework, the proposed attack may lose its functionality.

### Questions
- What is the sensitivity of \lambda? 
- It suggests providing detailed defense results rather than similarity checks. 
- Since the reported attack performance is not that impressive, the reviewer may think a backdoor attack would succeed in such a scenario. That is to say, the system may keep a high overall performance while losing accuracy on specific tasks. It suggests further exploration.

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
The manuscript introduces a novel gray-box poisoning attack in FGL that exploits shared neighbor representations during training. By targeting auxiliary-information-sharing frameworks, a malicious client can inject compromised gradients during generator updates. This manipulation results in benign clients incorporating poisoned synthetic nodes into their local subgraphs, leading to persistent model degradation as corrupted signals propagate during training. The authors formalize this threat model and propose a stealthy, optimization-based attack that is model-agnostic and effective across varying datasets, partition strategies, and client configurations. Their findings expose a critical vulnerability in the data-generation layer of FGL systems and underscore the need for robust defense mechanisms.

### Strengths
The manuscript presents a compelling and technically rigorous contribution to the field of FGL by identifying a novel and underexplored security threat. The proposed IGPA is conceptually innovative, exploiting the shared neighbor-generation mechanism in auxiliary-information-sharing frameworks like FedSage+. Unlike conventional models or data poisoning, IGPA operates indirectly through gradient manipulation of generator updates, making the attack highly stealthy and persistent. This manuscript is methodologically sound, offering clear mathematical formalization, algorithmic design, and theoretical justification for both the attack and its stealth enhancements. The empirical evaluation is comprehensive, covering multiple benchmark datasets and varying numbers of clients. Results consistently demonstrate the attack’s effectiveness in degrading global model performance and its resilience under different configurations and stealth constraints. Visualizations and quantitative analyses of gradient similarity, node insertion ratios, and convergence behavior further strengthen the experimental credibility.

### Weaknesses
While the manuscript demonstrates strong methodological rigor and novelty, several limitations remain that could weaken its overall impact and generalizability.

The proposed IGPA is evaluated exclusively on FedSage+, which, although representative of auxiliary-information-sharing frameworks, may not encompass the full diversity of FGL architectures. Other frameworks (e.g., FedGNN, FedGCN, or personalized subgraph FGL) employ different data-sharing or aggregation mechanisms that might not be equally vulnerable. The absence of comparative evaluation across these settings restricts the claim of model-agnostic applicability.

The gray-box assumption, where the adversary has access to shared embeddings and generators, simplifies real-world complexities. In many practical deployments, such access may be limited or obfuscated by encryption, compression, or secure aggregation protocols. This manuscript does not thoroughly discuss how these protective mechanisms would affect the feasibility or strength of IGPA.

This manuscript convincingly exposes a new vulnerability but provides minimal insight into how such attacks could be mitigated. It neither analyzes the limitations of existing defenses in detail nor proposes preliminary countermeasures tailored to generator poisoning. This omission leaves the work somewhat one-sided, emphasizing offensive capability without offering a path toward practical system hardening.

The experiments focus primarily on node classification accuracy as the sole performance indicator. Other important metrics, such as communication efficiency, robustness–utility trade-offs, or computational overhead, are not reported. These would help characterize the full impact of the attack and its practical relevance under resource-constrained federated environments.

The evaluation uses well-known benchmark datasets (Cora, Citeseer, PubMed, MSAcademic) with relatively small graph sizes. The scalability and persistence of IGPA in large-scale or dynamic graph environments (e.g., evolving social or financial networks) remain unexplored. Real-world systems may introduce additional constraints, (e.g., heterogeneous connectivity, asynchronous updates, or variable communication delays) that could affect both attack feasibility and detectability.

While the attack’s formulation is mathematically defined, the manuscript offers little theoretical treatment of how IGPA impacts convergence properties of federated optimization. A more formal examination of gradient divergence, variance amplification, or convergence delay would strengthen the analytical depth of the work.

### Questions
Please refer to ``weaknesses`` part.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on a critical vulnerability in auxiliary-information-sharing Federated Graph Learning (FGL) frameworks, with FedSage+ as a representative example. It proposes a gray-box poisoning threat model for auxiliary-information-sharing FGL, where a malicious client poisons shared graph-related gradients to compromise the global model—without accessing other clients’ raw graph data or local parameters. Introduces the Indirect Gradient Poisoning Attack (IGPA), a gradient-guided poisoning method that approximates benign update directions and injects malicious signals. This attack reduces node classification accuracy while maintaining high stealth. This paper Empirically demonstrates IGPA’s performance across four standard datasets (Cora, Citeseer, PubMed, MSAcademic) and different client scales (3, 5, 10 clients). It shows the attack degrades model accuracy by over 10% (when β=1) and evades standard defenses through graph structure similarity and consistent loss convergence.

### Strengths
(1)The paper addresses an understudied vulnerability in auxiliary-information-sharing FGL frameworks. Unlike prior work focusing on privacy leakage, it targets active, indirect data manipulation via generator gradient poisoning—filling a critical gap in FGL security research.
(2)IGPA’s two-stage mechanism (training a surrogate classifier + optimizing an adversarial NeighGen) is technically sound. The use of pseudo-labels (refined via Exponential Moving Average and KL Divergence) and adaptive loss for stealth enhancement demonstrates careful engineering to balance effectiveness and undetectability.

### Weaknesses
(1)The threat model assumes a single malicious client, but real-world FGL systems may face multi-client collusion. The paper does not discuss whether IGPA scales to collusion scenarios or how attack effectiveness changes with more malicious clients.
(2)The paper only tests IGPA against basic server-side defenses (Krum, trimmed-mean) and simple anomaly detectors (cosine similarity/Euclidean distance). It does not evaluate against state-of-the-art FGL-specific defenses (e.g., robust generator training, gradient sanitization) or defenses designed for gradient poisoning in standard federated learning (FL)—leaving uncertainty about IGPA’s effectiveness under stronger protection.
(3)Figures 6, 8, and 9 show gradient similarity for specific clients but lack statistical summary (e.g., mean/standard deviation of similarity across all client pairs) or significance testing. This makes it hard to assess whether the observed similarity is consistent across different runs or datasets.

### Questions
Questions for the Authors：
(1)If multiple clients collude to launch IGPA (e.g., 2 or 3 malicious clients in a 10-client system), how does attack effectiveness and stealth change? Do collusion strategies (e.g., coordinated gradient poisoning) amplify the attack impact?
(2)Does IGPA disproportionately reduce accuracy for nodes with minority labels or low degree in the graph? If so, what mechanisms drive this disparity, and how might it be mitigated?
(3)How does IGPA perform against state-of-the-art FGL-specific defenses (e.g., robust NeighGen training with adversarial regularization) or gradient sanitization methods ? Can you provide experimental results for these defenses?
Additional Feedback to Improve the Paper：
(1)Add statistical summaries (e.g., boxplots of cosine similarity across all client pairs) and significance testing (e.g., t-tests comparing malicious vs. benign gradient similarity) to Figures 6, 8, and 9. This will strengthen the claim that IGPA evades gradient-based detectors.
(2)Include metrics like label-wise accuracy and test-time perturbation robustness to provide a more holistic view of IGPA’s impact.
(3)Add a small subsection or table explaining the motivation for choosing EMA+KL Divergence over alternative pseudo-labeling methods (e.g., confidence thresholds).
(4)Add experiments testing IGPA against 1–2 state-of-the-art FGL defenses to better contextualize the attack’s real-world threat level. This will help readers understand whether IGPA is a niche concern or a pressing vulnerability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper focuses on the FedSAGE+ federated graph learning framework and proposes a dedicated attack method for it. Unlike conventional attacks that target the framework’s classifier, the proposed method specifically aims at the data generator module—this design avoids direct interaction with the classifier and thereby significantly enhances the attack’s stealth. To validate the method, the authors conduct relevant experiments on typical federated graph learning scenarios, and the results consistently demonstrate the effectiveness of the proposed attack in compromising the FedSAGE+ framework.

### Strengths
1. The method integrates the attack into the data generator module, which cleverly bypasses the attack detection mechanisms deployed on the server side during the classifier training phase. This design not only reduces the risk of the attack being identified but also aligns with the operational logic of the FedSAGE+ framework, ensuring the attack’s feasibility in practical scenarios.
2. The paper demonstrates sufficient research effort: it not only proposes a complete set of attack methods but also proactively considers potential defense measures. By optimizing the attack strategy based on these defense mechanisms, the method achieves a balanced performance between attack effectiveness and stealth.
3. The paper is the first to identify a major security vulnerability in the FedSAGE+ framework. specifically, the neglect of security risks in the data generator module during the framework’s design. This finding provides a clear direction for subsequent defense-related studies.

### Weaknesses
1. The attack method is designed specifically for the FedSAGE+ framework, which limits its generality to other federated graph learning frameworks.
2. Experimental results show poor attack effectiveness under high stealth. For example, on the MSAcademic dataset with M=3, the model still maintains an accuracy of over 90%.
3. After data generation, the data needs to go through the classifier training process. For instance, the GAT-based GNN calculates attention scores for neighbor nodes, which weakens the impact of generated poisoned nodes on the final results.

### Questions
1. In eq 1 of the proposed attack method, if the "min" operation is replaced with a "max" operation, can the attack still achieve similar effectiveness? If not, what is the key reason for the difference?
2. How do different GNN architectures (e.g., GAT, GCN,…) affect the effectiveness of the proposed attack?
3. Under different β value settings, what is the probability that the client detects gradient anomalies?

### Soundness
3

### Presentation
3

### Contribution
2
