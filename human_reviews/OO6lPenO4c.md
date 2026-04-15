# Federated Learning with Dynamic Client Arrival and Departure: Convergence and Rapid Adaptation via Initial Model Construction

- Decision: Reject
- Scores: 3, 5, 3, 3

## Abstract
While most existing federated learning (FL) approaches assume a fixed set of clients in the system, in practice, clients can dynamically leave or join the system depending on their needs or interest in the specific task. This dynamic FL setting introduces several key challenges: (1) the objective function dynamically changes depending on the current set of clients, unlike traditional FL approaches that maintain a static optimization goal; (2) the current global model may not serve as the best initial point for the next FL rounds and could potentially lead to slow adaptation, given the possibility of clients leaving or joining the system. In this paper, we consider a dynamic optimization objective in FL that seeks the optimal model tailored to the currently active set of clients. Building on our probabilistic framework that provides direct insights into how the arrival and departure of different types of clients influence the shifts in optimal points, we establish an upper bound on the optimality gap, accounting for factors such as stochastic gradient noise, local training iterations, non-IIDness of data distribution, and deviations between optimal points caused by dynamic client pattern. We also propose an adaptive initial model construction strategy that employs weighted averaging guided by gradient similarity, prioritizing models trained on clients whose data characteristics align closely with the current one, thereby enhancing adaptability to the current clients. The proposed approach is validated on various datasets and FL algorithms, demonstrating robust performance across diverse client arrival and departure patterns, underscoring its effectiveness in dynamic FL environments.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This work considers FL with dynamic participation where clients may leave or join the system in any global iteration and analyzes the model convergence in this setup. This work further proposes an adaptive initial model construction strategy that employs weighted averaging guided by gradient similarity, which is validated on various datasets.

### Strengths
(1) The considered dynamic setup is less explored in existing works but is quite crucial in practical applications.

(2) The theorem and analysis in section 4 are comprehensive.

(3) The proposed adaptive algorithm shows benefits compared with baselines.

### Weaknesses
(1) The theoretical results are under strong assumptions such as strong convexity (in Assumption 2), bounded gradient (in Theorem 1).

(2) Theorem 1 only captures a resursive relationship between two consecutive optimality gap. How does this theorem show the convergence of this dynamic system or in which condition may the FL algorithm diverge?

(3) With the initial model in (6), how does the convergence change from Section 4?

(4) The authors mentioned (Ruan et al. 2021a) in related works, but there is no comparison between this work and proposed method. Also, (Ruan et al. 2021a) and (Ruan et al. 2021b) are the same reference.

(5) The challenges in Line 50 mentioned that some classes may not be met in some iterations; can the client participation in Table 2 and Table 3 guarantee this discription? 

(6) The setups "half" and "partial-overlap" seem to be similar with those used in client participation works. Given this, it is more convincing to compare the proposed algorithm with those methods, such as (Gu et al.， 2021) (Jhunjhunwala et al. 2022).

### Questions
Besides of the above weakneess, could authors further explain the effect of scaling factor R in Line 380? What if we don't use this scaling factor?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a FL framework that addresses the challenge of dynamic client participation, a scenario in which clients frequently join or exit the training process. A probabilistic model is developed to quantify the impact of client arrival and departure on the FL optimization objective, providing an upper bound on the optimality gap under these dynamic conditions. Furthermore, a gradient-based approach is introduced for constructing an adaptive initial model that aligns more closely with the data distributions of the active clients.

### Strengths
+ This work addresses a key limitation in existing FL research by considering dynamic client participation, a more realistic scenario than the often-assumed static client set. The proposed framework, which adjusts for client variability, adds flexibility and may lead to increased model adaptability.

+ The theoretical derivation of an optimality gap bound, considering factors like stochastic gradient noise and non-IID data distributions, is a valuable contribution. This analysis provides a comprehensive understanding of how dynamic client changes impact convergence in FL.

+ The adaptive initial model construction strategy, leveraging gradient similarity, is innovative. This method enables the model to adapt more rapidly to the currently active clients by aligning more closely with their data distributions, potentially enhancing training efficiency.

### Weaknesses
- The theoretical analysis relies on a strong convexity assumption, which does not typically apply to deep neural networks commonly used in FL. Since neural networks are often non-convex, this assumption limits the practical applicability of the derived bounds in FL scenarios involving complex models.

- Using a single pilot model to initialize each new round, while computationally efficient, may introduce bias if client participation patterns shift significantly or if clients with distinct data distributions join later in training. A single pilot model might not fully capture the diversity of evolving data distributions, potentially affecting the model’s generalizability across client types.

- Although the experimental evaluation is thorough, it focuses solely on image classification tasks with a relatively small client population (10 clients). Including tests on non-image data, such as the Shakespeare dataset, and expanding the number of clients would provide solid evidence of the framework’s scalability and versatility.

- Lastly, the framework does not explicitly address the issue of catastrophic forgetting, which may occur when clients rejoin after an absence. Adapting exclusively to active clients could result in the loss of knowledge from previously encountered data distributions, thereby reducing performance for returning clients, especially if their data significantly diverges from that of current participants.

### Questions
- Could the implications of the strong convexity assumption for practical deep learning scenarios be discussed in more detail? Additionally, are there ways the analysis could be extended to accommodate non-convex settings commonly found in FL?

- What potential strategies could be employed to mitigate the bias introduced by using a single pilot model? For instance, would periodically updating the pilot model or utilizing multiple pilot models help in capturing the diversity of evolving data distributions?

- How does the proposed framework perform under different types of data (such as the Shakespeare dataset) or more complex datasets (such as the Tiny ImageNet dataset)? Additionally, further experiment on increasing the number of clients is necessary to demonstrate the proposed method’s effectiveness.

- What strategies could be integrated into the framework to mitigate catastrophic forgetting? Additionally, has there been any analysis or experiments on how the method performs when clients with previously seen data distributions rejoin the training process?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper tackles the problem of dynamic federated learning in which clients can join or leave during training. Unlike traditional FL approaches that assume a static set of clients, this paper addresses the challenges of an evolving client pool, which can disrupt convergence and slow adaptation. The authors propose a probabilistic framework to model client types and a dynamic optimization objective to maintain model relevance. They introduce an adaptive initial model construction strategy based on weighted averaging of gradients, prioritizing models aligned with current data.

### Strengths
1. The paper is well-written and easy to follow.

2. The authors provides the theoretical analysis, including convergence bounds and an upper bound on the optimality gap.

3. The proposed method is tested on diverse datasets, label distributoins.

### Weaknesses
1. I am uncertain about the novelty of this dynamic FL setting where clients join or leave during training. To me, this scenario seems fundamentally similar to federated learning with concept drift, as discussed in [1].

2. Although the theoretical analysis adds rigor, the proof relies on too many inequalities to approximate results, which makes the final theoretical outcomes feel quite loose. Does this theoretical analysis truly reflect the actual training dynamics?

3. The proposed Pilot Model requires storing all previous global models, which could lead to significant storage issues.

4. The experimental section is relatively weak; evaluating only FedAvg and FedProx does not sufficiently demonstrate the effectiveness of the proposed method. I think methods designed for FL with concept drift could also apply well to this dynamic FL setting.

[1] Federated Learning under Distributed Concept Drift

### Questions
Please see the weakness section.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper addresses a federated learning (FL) scenario where clients can dynamically join or leave the system. The authors propose a dynamic optimization objective tailored to online(active) clients, incorporating a probabilistic model to aid in convergence analysis for the new global optimization objectives. Experiments are conducted across various settings to validate the proposed approach.

### Strengths
The authors devote considerable effort to the convergence analysis, ensuring that the divergence between the realistic output $w^{(g)}$ and the global optimal point $w^{(g)*}$ at each round $g$ is bounded. Through recursive iterations, they demonstrate that the divergence for each round remains within an upper bound, enhancing the reliability of their proposed method.

### Weaknesses
The novelty of the system design warrants scrutiny. In a typical horizontal FL framework, at any round $g$, it is both natural and reasonable to compute the global model using the current active (or online) clients. If we denote the set of active clients as ${1,2,...,C}$ and the final local model for a client $c \in {1,...,C}$ as $w^c$, we can compute the global model $w^g = \sum_{c \in {1,...,C}} \eta_c w^c$ with the constraint $\sum_{c \in {1,...,C}} \eta_c = 1$, following standard aggregation rules. In the examples provided, the weight for each client is based on the ratio of the client's data samples to the total data samples of online clients, as described on line 214. Given this, it is unclear why the authors introduce a “transitive global objective” $w^{(g)}$ tailored solely for the active client set at round $g$. In a realistic horizontal FL setting, **the server can only compute the weighted average over active clients**, regardless of whether the clients hold heterogeneous data distributions. Therefore, introducing a changing objective in the realistic experiment setting does not appear to provide substantial novelty or practical benefit.

Regarding the experimental results, the accuracy improvements in Table 1 seem attributable to the dynamic initial model construction described in Section 5. However, the pilot model design appears independent of the proposed dynamic objective and could be implemented in any realistic FL scenario, provided the aggregated model from each round $g$ is retained. My concern lies in understanding how the dynamic objective impacts the final performance. This connection remains unclear, and I believe the emphasis in Section 5 and the experimental results downplay the significance of the proposed dynamic objective. Can the authors clarify this aspect?

### Questions
1. **Purpose of the Dynamic Objective**: For further clarification on the motivation behind the dynamic global objective, please refer to the detailed comments in the *Weaknesses* section.
2. **Clarity of Equations**: The presentation of Equations 1 and 2 is difficult to follow due to excessive notation. A more straightforward formulation with minimal notation might improve readability and comprehension.
3. **Convergence Condition in Theorem 1**: To guarantee convergence, the coefficient component in $(a)$, specifically $\mu\eta^{(g)}\left(\sum_{k \in \mathbb{K}^{(g)}}\frac{D^{(g)}_k e^{(g)}_k}{D^{(g)}}\right)$, must be $< 2$. How is this condition ensured? It would be helpful if the authors could provide a synthetic experiment with a clearly defined $\mu$, $\eta^{(g)}$, and other relevant hyperparameters to demonstrate that the objective function indeed converges under these settings.

### Soundness
2

### Presentation
2

### Contribution
1
