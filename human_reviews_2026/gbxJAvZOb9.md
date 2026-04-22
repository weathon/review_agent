# TRIDENT: An Efficient Data-Free Model Extraction Attack for Graph Neural Networks

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Graph Neural Networks (GNNs) are increasingly offered as Machine-Learning-as-a-Service (MLaaS) via APIs. However, this deployment model exposes them to model extraction attacks (MEAs), where adversaries aim to reconstruct or steal the proprietary GNN models by leveraging only query access to the service. Although recent work has developed data-free MEAs that have performed well in both transductive and inductive settings, the field still lacks a unified theoretical account explaining when and why such attacks succeed, and how an adversary should schedule queries under strict budget constraints. In this paper, we aim to bridge this gap by introducing the first theory-driven framework for MEAs against GNNs. This framework highlights that not all queries are equally effective, guiding more strategic, budget-constrained query scheduling. It also effectively leverages the surrogate model’s white-box accessibility to improve its alignment with the black-box victim. To specifically answer the key question of what kinds of queries enable effective MEAs, we formalize the extraction risk and derive a bound based on the generalization discrepancy between the query distribution and the victim model's unseen training distribution. Guided by this analysis, we propose TRIDENT, which strategically schedules queries, particularly under strict budget constraints. Extensive experiments on six real-world benchmarks and three GNN backbones show that our method achieves state-of-the-art performance. These results validate both the theoretical contributions and the practical efficiency of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The manuscript presents a theoretical and practical framework for MEAs on GNNs offered via MLaaS APIs. It formalizes extraction risk and derives a lower bound linking it to the victim model’s risk and the distributional discrepancy between the query and target data, revealing that minimizing this discrepancy is key to successful extraction. Building on this insight, the paper introduces TRIDENT, a data-free and query-efficient MEA that strategically schedules queries under strict budget constraints. Extensive experiments on six real-world datasets and three GNN backbones show that TRIDENT achieves state-of-the-art extraction performance.

### Strengths
The manuscript makes both theoretical and practical advances in MEAs on GNNs. It introduces a rigorous theoretical framework that formalizes extraction risk and derives a lower bound linking it to the victim’s intrinsic risk and the generalization discrepancy between query and target distributions, offering a clear design principle for attack optimization. Building upon this insight, the proposed TRIDENT method implements a data-free and query-efficient attack through two novel components: a budget-ratio-based graph generator for adaptive query scheduling and a surrogate-as-victim alignment module that refines surrogate learning without additional queries. Theoretical derivations are well integrated with algorithmic design, and extensive experiments across six benchmark datasets and three GNN architectures demonstrate state-of-the-art fidelity and accuracy, validating both the theoretical analysis and empirical effectiveness. The ablation study further substantiates each module’s contribution to performance and efficiency.

### Weaknesses
Despite its theoretical elegance and empirical rigor, the manuscript exhibits several limitations in clarity, scope, and practical depth that may restrict its impact and generalizability.

The paper’s theoretical sections are highly abstract and mathematically dense. The presentation relies heavily on many notations and several assumptions, which may obscure the underlying intuition behind the derived results. It may hinder accessibility for readers less familiar with formal risk analysis or statistical learning theory.

Although the proposed TRIDENT framework emphasizes query efficiency, the manuscript does not explicitly analyze the computational or memory overhead introduced by the dual-surrogate design and graph generation proces. For instance, maintaining both the main and light surrogate models, along with iterative graph synthesis and KL-weighted updates, could significantly increase training time and hardware demands. Without explicit runtime or complexity analysis, it remains unclear whether the claimed efficiency gains in query usage translate into practical computational efficiency in real-world MLaaS environments.

The experimental evaluation is confined to node classification benchmarks (e.g., Cora, CiteSeer, PubMed). This narrow focus limits the generality of conclusions regarding other graph learning paradigms such as link prediction, graph classification, or heterogeneous graphs. Moreover, all datasets are publicly available academic benchmarks, which may not fully capture the data complexity or noise characteristics of industrial GMLaaS deployments.

The study primarily compares TRIDENT with prior data-free attacks such as Type III attacks from Zhuang et al. (2024) but omits recent adaptive MEAs or defense-aware extraction techniques (e.g., dynamic watermarking, detection-based defenses, or distributionally equivalent attacks). This limits the ability to contextualize TRIDENT’s robustness under adversarially hardened environments or modern countermeasures.

Several assumptions in the theoretical derivation (e.g., i.i.d. sampling between query and target distributions, bounded losses, Lipschitz continuity) are strong and may not hold in practical black-box MLaaS settings. The manuscript does not investigate how deviations from these assumptions affect the tightness of the lower bound or the validity of the derived conclusions. This weakens the theoretical generality beyond controlled settings.

Although the ablation study is presented, it primarily focuses on validating the functional roles of individual modules. The manuscript, however, does not include parameter sensitivity analyses that examine how variations in key hyperparameters (e.g., the budget ratio, surrogate model size, or KL weighting factor) affect convergence behavior, fidelity performance, and query efficiency. Incorporating such analyses would offer a more comprehensive understanding of TRIDENT’s stability, robustness, and reproducibility under diverse experimental configurations.

### Questions
Please refer to the ``Weaknesses`` part.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TRIDENT, with theory-driven, data-free, and query-efficient model extraction attack for GNNs deployed in MLaaS. It formalizes extraction risk and derives a lower bound linking it to the victim model’s inherent risk and generalization discrepancy between query and target distributions, guiding the design principle of minimizing discrepancy. TRIDENT integrates a budget ratio based graph generator and a Surrogate-as-Victim alignment module to optimize query efficiency and surrogate model fidelity. Extensive experiments on six real-world datasets and three GNN backbones demonstrate its state-of-the-art performance in accuracy, and query efficiency compared to existing baselines.

### Strengths
1. First provides a unified theoretical framework for GNN MEAs, addressing the lack of rigorous theory in prior empirical works.

2.Proposes a novel and realistic setting, limited queries, black-box access, and no training data—well aligned with real-world MLaaS security scenarios.

3.Presents comprehensive experimental validation across diverse datasets and models.

### Weaknesses
1.The paper is overly formalized; the authors may consider simplifying some definitions. For instance, Definition 1 and Definition 4 appear similar, which reduces readability.

2.It would be valuable to discuss possible defense strategies. Under the setting of limited queries, black-box access, and no training data, it seems difficult for pre-trained model owners to mitigate such attacks.

3.Can the tightness of the theoretical lower bound be empirically validated due to the unobservability of the victim’s training data?

4.Under the no-training-data assumption, how is a suitable graph dataset obtained for querying? The initialization of the graph generator seems crucial. Moreover, TRIDENT requires training both a graph generator and two surrogate models, which introduces non-trivial local computational costs.

5.The paper appears to lack discussion of related works on graph model extraction attacks, such as [1].

[1] Extracting Training Data from Molecular Pre-trained Models

### Questions
See weakness.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The work investigates the subject of Model Extraction attacks, which consists of stealing a proprietary model by leveraging only query access, and specifically they focus on the field of Graph Neural Networks (GNNs). 
The main contribution of the paper is to try to bridge some absent theoretical investigation that is missing from the literature, which focused mainly on empirically understanding and approaching this problem. Based on theoretical insights, the authors provided TRIDENT, a new attack strategy to better schedule the queries in the specific context of tight attack budget.

### Strengths
- The general problem that is considered is very interesting and aiming for a theoretical analysis is rather interesting and clearly bridges some gaps within the literature. 
- The overall proposed theoretical analysis is interesting. While I have some questions regarding the computed bounds, it is still a good approach to extract insights. 
- The experimental results seems to be validating the worth of the proposed method.

### Weaknesses
- The problem formulation section is confusing - specifically, while I like the overall definition and introduction to the problem setting, I feel that it is rather adapted to a general aspect and not specific aspect of graphs. For instance, the authors refer to $\mathcal{X} \in \mathbb{R}^n$, but in the case of graphs, we have the adjacency space and the node feature space - and therefore an adaptation of this section could be of great value. Additionally, how is the graph data distribution defined in this case? 
- Proposition 1 is not very clear. While I get the main idea of trying to bound the risk, I don’t understand the statement and specifically the tightness of the low and upper-bounds constants. I have checked the proof to further understand, and while in the case of some losses, this bound seems to be small, in some cases it’s just degenerate. This remark also extends to Theorem 1.
- Some elements regarding the proposed methodology are not very clear. 
    - For instance this claim: “We expect that even with less queries for each generated graph, the generator would generate higher quality graphs due to more iterations.” — What is the main fundamental and thinking around it? I find it very hard to believe that actually with les queries, the generator would still perform an “acceptable” (based on some semantic/generation metric) generation. This latter point can be seen experimentally your experimental setting, namely Figure 3. 
    - From my understanding, you are interested in the problem of black-box attack, and at what point, you refer to white-box attack, which is a bit confusing when reading the paper. 
    - In the experimental setting, a lot of details are not very clear. Specifically:
        - How are the target and surrogate model chosen - do they have similar architecture (number of layers, hidden dimensions …)? If it’s the case, how does this relate to the black-box assumption? 
        - I couldn’t find details about the generator model, and also ablation study on how such choice affects the performance. 
        - In the paper, you claim results for OGB-Arxiv, but I couldn’t find them, am I missing something?

### Questions
- Could you better clarify Proposition 1 and Theorem 1 and the main claim and specifically try to better illustrate the considered constants in the upper-bounds? 
- Could you provide elements regarding the tightness of the provided bounds as it seems in that some cases they are vacuous and would argue worthless. 
-  Could you reformulate the methodology section such as to clarify where does the black-box and white-box paradigm occurs?
- Could you add details regarding the experimental details (please refer to the specific questions above).

### Soundness
2

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
3

### Summary
The core contribution is a new theoretical framework that formalizes extraction risk. The authors derive a lower bound for this risk, tying it directly to the victim model's own risk and, most importantly, the generalization discrepancy between the attacker's query distribution and the victim's (unseen) target distribution . This yields a clear design principle: an effective attack must minimize this discrepancy.

### Strengths
The paper's primary strength is its shift from a purely empirical to a theory-driven attack design. Formalizing the problem in terms of extraction risk and generalization discrepancy provides a solid conceptual foundation. The budget ratio is a simple but highly effective mechanism to manage the trade-off . The ablation studies  clearly show this component significantly boosts both performance and efficiency . The attack is thoroughly evaluated. The method shows consistent SOTA performance across different datasets  and different victim architectures. The attack also remains effective even against a label-flipping defense . The budget ratio is a simple but highly effective mechanism to manage the query-budget-to-training-iteration trade-off. The ablation studies (Fig. 3) clearly show this component significantly boosts both performance and efficiency .

### Weaknesses
The paper is vague on the specifics of the light surrogate model. It is described as smaller and can be "initialized from" the main surrogate.

### Questions
is there a principled way to pick $\alpha$ given a known budget B?

What prior knowledge is assumed for this "data-free" attack?

### Soundness
3

### Presentation
3

### Contribution
3
