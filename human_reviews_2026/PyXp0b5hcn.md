# Exploiting Client Heterogeneity for Forgetting Mitigation in Federated Continual Learning: A Spatio-Temporal Gradient Alignment Approach

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Federated Continual Learning (FCL) has recently emerged as a crucial research area, as data from distributed clients typically arrives as a stream, requiring sequential learning. This paper explores a more practical and challenging FCL setting, where clients may have unrelated or even heterogeneous tasks, leading to gradient conflicts where local updates point in divergent directions. In such scenario, statistical heterogeneity and data noise can create spurious correlations, leading to biased feature learning and catastrophic forgetting. Existing FCL approaches often use generative replay to create pseudo-datasets of previous tasks. However, generative replay itself suffers from catastrophic forgetting and task divergence among clients, leading to overfitting in FCL. To address these challenges, we propose a novel approach called \textbf{\underline{S}}patio-\textbf{\underline{T}}emporal gr\textbf{\underline{A}}dient \textbf{\underline{M}}atching with \textbf{\underline{P}}rototypical Coreset (STAMP). Our contributions are threefold: 1) We develop a model-agnostic method to determine subset of samples that effectively form prototypes when using a prototypical network, making it resilient to continual learning challenges; 2) We introduce a spatio-temporal gradient matching approach, applied at both the client-side (temporal) and server-side (spatio), to mitigate catastrophic forgetting and data heterogeneity; 3) We leverage prototypes to approximate task-wise gradients, improving gradient matching on the client-side. Extensive experiments demonstrate our method's superiority over existing baselines, particularly in scenarios with a large number of sequential tasks, highlighting its effectiveness in addressing the complexities of real-world FCL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes STAMP for FCL. It introduces temporal gradient alignment on clients and spatial gradient alignment on the server to handle task and client heterogeneity. At the same time, a prototypical coreset mechanism provides efficient replay without heavy memory usage. The method aims to align gradients across both temporal and spatial dimensions, improving generalization and reducing forgetting. Extensive experiments on diverse datasets demonstrate that STAMP outperforms existing FCL methods in accuracy, stability, and efficiency.

### Strengths
- Treating spatio-temporal gradient alignment jointly is a creative and well-motivated way to handle heterogeneity in FCL.
- The prototypical coreset approach is lightweight and avoids the heavy cost of generative replay, which is appealing for real-world edge deployments.
- The paper conducts comprehensive evaluations across multiple datasets and metrics, showing clear and consistent improvements.
- The inclusion of a theoretical generalization bound adds some rigor and helps contextualize the method’s stability and plasticity claims.

### Weaknesses
- The theoretical analysis remains largely qualitative and does not clearly demonstrate why STAMP achieves better gradient alignment in practice.
- While the proposed approach is efficient, the algorithmic pipeline (temporal + spatial alignment + coreset) can be conceptually heavy and may limit reproducibility.
- The improvements, although consistent, are relatively modest on some benchmarks, and the scalability on extremely large client pools is not deeply analyzed.

### Questions
- How sensitive is STAMP to the size of the coreset, and what is the trade-off between memory footprint and accuracy?
- Can STAMP handle asynchronous or partially participating clients in more realistic federated settings?
- How does STAMP perform when the degree of heterogeneity is extreme, e.g., when client label spaces are disjoint?
- Some related baselines like FedSSI and references should be added and compared [1-2].

[1] FedSSI: Rehearsal-Free Continual Federated Learning with Synergistic Synaptic Intelligence. The Thirteenth International Conference on Learning Representations, ICML 2025

[2] Unleashing the Power of Continual Learning on Non-Centralized Devices: A Survey. IEEE Communications Surveys & Tutorials, 2025

### Soundness
3

### Presentation
3

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
The paper proposes the use of gradient alignment technique across tasks and clients computed from a prototype coreset of samples (proposed in this paper) at each client to improve generalization in federated continual learning setting. The paper also provides bounds on the generalization gap between seen and unseen tasks.

### Strengths
- The paper provides theoretical justification for gradient alignment techniques and specifically for the coreset selection technique being used
- Generalization bound is derived and proposed technique is shown to have reduced the generalization gap.
- Performance is shown to be on par with popular prior work such as FedWeIT, while reducing the disk usage significantly.
- Ablation studies are quite extensive, the results shown are convincing to show that the method work reliably and provides the claimed gains. Datasets however are somewhat still more traditional datasets used in federated learning and does not contain natural continual learning datasets.

### Weaknesses
- The method relies on some assumptions such as requiring replay buffer at clients.
- Gradient alignment might slow the convergence, especially when updating the global model.
- The paper should at least in the appendix comment on how the coreset selection (combinatorial problem) is solved and what’s the additional worst-case complexity (O(n^2) or O(n log n), etc).
- The claims are a bit overstated as the generalization over tasks is not really the goal of the coreset selection. For example, assume a large volume of samples from a certain task/client that dominates a few training rounds. In those cases, the coreset will start to get dominated by samples from those tasks. As a result, the gradient alignment favors those tasks containing the most samples. Instead of task level, I suppose the claim is that the proposed technique will help in reducing generalization gap across the diverse data.

### Questions
- How is the coreset selection done? Even if this is a well-known and solved problem, it should clearly be stated including the computational complexity of it.
- Is the paper's goal really the generalization over all data or tasks? Check comments above, but the question is what happens when a single task has 1000x more samples compared to another task?

### Soundness
3

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
4

### Summary
This paper proposes STAMP, a federated continual learning (FCL) approach incorporating spatio-temporal gradient alignment across clients and tasks, along with a prototypical coreset to mitigate catastrophic forgetting with reduced memory usage. The idea of aligning gradients across the temporal (intra-client) and spatial (inter-client) dimensions is interesting, and the paper provides both theoretical motivation and empirical evaluations showing improvement over existing baselines on several vision datasets.

### Strengths
1.Novel idea of performing both temporal and spatio gradient alignment in FCL.
2.Prototypical coreset is a good alternative to generative replay and full memory buffers.

### Weaknesses
- Handling catastrophic forgetting: Although this work is rehearsal based, the work does not explicitly handle catastrophic forgetting (most of the rehearsal methods do, such as AGEM/GEM). However, the paper repeatedly claims that STAMP reduces catastrophic forgetting, but this is not explicitly demonstrated experimentally or theoretically. The provided plots focus only on temporal and spatio gradient alignment metrics. Good alignment does not necessarily guarantee reduced forgetting. More direct evidence such as forgetting curves and class/task-level retention is needed.
- The claim of Fig 1 is misleading. The difference indeed decreases, but at the cost of global accuracy in STAMP. How is this a good case? This also contradicts the claim of improved intra-client retention. The paper needs to explain this discrepancy.
- It is unclear how spatio and temporal gradient alignment preserve the gradient direction, especially in coparison with the memory data. The paper states that alignment prevents negative transfer, but lacks intuition or analysis on how this specifically preserves directionality of task gradients over time.
- It is not clear how the theoretical results show the impact of using coresets - a typical generalization result must encompass the effect of coresets. For instance, if one performed random sampling of points instead of using a coreset, how would the effect be reflected in Thm 2. 
- Storage cost concern for storing gradients. Calculating gradient alignment implicitly requires storing gradients from previous tasks and clients. This may become costly for large models.
- Ablation studies are insufficient. The method introduces multiple components (temporal GA, spatio GA, prototypical coreset, ProtoNet, MixStyle), yet ablations are limited. Add more like varying number of tasks, effect of different epochs per task, impact of removing the prototypical network, varying coreset sizes.
- \gamma used in the gradient alignment formulation is not defined in the main text.
- Possible error in Section 2.1. The description states that r is the current round of task t, but it should logically refer to the current round of task t+1.
- Figure 3 does not support the claim that heterogeneity helps generalization. The results do not show a clear benefit as heterogeneity increases. This contradicts the core hypothesis. The chosen values of \alpha are quite high, and does not indicate highly heterogeneous cases. 
Missing plots in Figures 5 and 6. No results for CIFAR100 with 2 classes/task for temporal gradient alignment in Fig. 5. No results for CIFAR100 with 20 classes/task for spatio gradient alignment Fig. 6. Some recent baselines with theoretical guarantees (CFLAG, AISTATS 2025) is not cited.

### Questions
Please clarify the points raised in weaknesses.

### Soundness
3

### Presentation
2

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
This paper focuses on the federated continual learning (FCL) scenarios where heterogeneous tasks assigned to clients. The authors propose Spatio-Temporal grAdient alignMent with Prototypical coreset (STAMP), which uses gradient alignment method and prototypes for mitigating the biased feature learning and severe catastrophic forgetting.

### Strengths
* The paper focuses on an important research problem of CFL with heterogeneous clients.

### Weaknesses
* Gradient Alignment method is not something new
* The theoretical results are either trivial (Lemma 1) or too complex to get meaningful insight (Theorems 1,2). Especially, it was hard to find some connection between the theoretical results and the empirical observations.
* It is unclear when/why STAMP works well.

### Questions
None

### Soundness
2

### Presentation
2

### Contribution
2
