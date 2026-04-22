# Auto-regressive In-context Demonstration Selection

- Avg Score: 4.00
- Decision: Reject
- Scores: 8, 2, 2

## Abstract
Effective demonstration selection is crucial for maximizing large language model (LLM) performance in few-shot in-context learning. Due to influences such as recency bias, the effectiveness of demonstrations depends heavily on their context relationship to the specific query, and on the ordering in which they are presented, making demonstration selection a complex combinatorial problem. To address these two challenges, we introduce AutoSelect, a novel framework that formulates demonstration selection as an auto-regressive sequential decision process. At each step, AutoSelect embeds the query and previously selected demonstrations into matrix representations to preserve structural information, and a trainable policy model sequentially selects the next best exemplar. To navigate the factorial space of demonstration permutations, our framework formulates a Kullback-Leibler (KL) regularized optimization problem, from which an optimal policy induces an optimal Plackett-Luce (PL) ranking over all possible demonstration sequences. Our theoretical analysis provides a principled learning objective: we prove that minimizing a tractable policy-level Cross-Entropy (CE) loss provably bounds the worst-case discrepancy between our policy's induced PL ranking and the optimal one, enabling tractable prioritization of high-quality sequences. Empirically, AutoSelect outperforms existing heuristic and learning-based methods across nine diverse datasets, achieving up to an 11\% improvement over the strongest baseline. Our results are further supported by analytical studies and a case study, highlighting AutoSelect's key properties, as well as its transferability and generalizability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces **AUTOSELECT**, a novel framework designed to maximize the performance of large language models (LLMs) in few-shot in-context learning (ICL) by addressing the complex, combinatorial challenge of **demonstration selection and ordering**. The effectiveness of demonstrations is highly sensitive to their relationship with the query and their presentation order, due to factors like recency bias.

**Core Methodology:**

AUTOSELECT models demonstration selection as an **auto-regressive sequential decision process**.

1.  **Representation and Architecture:** At each step, the framework embeds the query and previously selected demonstrations into **2D matrix representations** to preserve structural information (such as token sequence order and inter-token relationships). A trainable, specialized **Vision Transformer (ViT)-based policy model ($\pi_\theta$)** processes these representations to synthesize a contextual representation ($z_t$).
2.  **Sequence Construction and Adaptive Stopping:** The policy model sequentially selects the next best exemplar or an explicit **End-of-Sequence (EOS) signal**. The EOS signal allows the model to dynamically learn the optimal demonstration sequence length for a given query.
3.  **Theoretical Optimization:** To navigate the factorial search space, the optimization is formulated as a Kullback-Leibler (KL) regularized Reinforcement Learning problem. The central theoretical contribution is proving that minimizing a tractable **policy-level Cross-Entropy (CE) loss** effectively bounds the worst-case discrepancy between the policy's induced **Plackett-Luce (PL) ranking** and the optimal PL ranking. This objective enables the policy to efficiently prioritize high-quality, ordered sequences without exhaustive enumeration.

**Contributions:**

Empirically, AUTOSELECT demonstrates consistent superiority over both heuristic and existing learning-based selection methods across nine diverse datasets, including complex reasoning tasks (Winowhy, AQuA). Performance gains reached up to **11% over the strongest baseline**. The framework's ability to select high-quality sequences over merely high quantity is also highlighted.

### Strengths
### Originality

*   **Auto-regressive Paradigm for Compositional Effects:** The paper introduces a highly original approach by framing the selection and ordering problem—which involves complex compositional effects—as an **auto-regressive sequence decision process**. This paradigm mirrors LLM generation, allowing the model to "compose" an effective sequence step-by-step, conditioning each choice on the previous selections and the query context.
*   **Synergistic Architecture:** The use of **2D matrix embeddings** (preserving token-level structure) paired with a specialized **ViT-based policy model** is a novel architectural choice demonstrated to be more effective than larger 1D sequence models like a GPT-2 variant for this task.

### Quality

*   **Strong Theoretical Foundation:** The optimization is rigorously derived from a KL-regularized RL framework. The proof that minimizing the conditional Cross-Entropy (CE) loss on collected trajectories provably bounds the deviation from the optimal Plackett-Luce (PL) ranking provides a **principled and justifiable learning objective** for sequence prioritization [Theorem 4.2].
*   **Comprehensive Empirical Validation:** The framework is tested extensively on nine diverse tasks, including challenging reasoning benchmarks like Winowhy, Epistemic Reasoning, and AQuA, demonstrating **robust and consistent performance improvements**. The performance superiority holds across different LLM families and scales.

### Clarity

*   **Well-Defined Mechanism:** The sequence generation process, including the sequential decision-making conditioned on the prefix ($\tau_{<t}$), the specialized rollout procedure to efficiently collect full and sub-trajectories, and the reward aggregation metric for smoother feedback, are clearly articulated in both text and algorithms (Alg. 1 and 2).
*   **Adaptive Stopping Mechanism:** The incorporation of the $\text{e}[\text{EOS}]$ signal provides a transparent way for the policy to **dynamically learn the optimal sequence length**, balancing performance and inference cost, a mechanism empirically validated by the visualization of varying average trajectory lengths across tasks.

### Significance

*   **Effective Solution to the Combinatorial Bottleneck:** AUTOSELECT successfully models and optimizes the critical compositional effects between demonstration content and ordering, overcoming a key bottleneck in ICL that heuristic and simple retrieval methods overlook.
*   **Practical Efficiency Trade-Off:** The framework makes a worthwhile, one-time offline investment in policy training, which results in significant savings and superior accuracy during inference time compared to methods that require heavy online evaluation (like greedy-oracle).

### Weaknesses
### 1. Need for Updated Related Works and Contextualization

The field of ICL selection is rapidly evolving. Although the paper effectively compares against strong baselines from 2022 and 2023 (ActRL, CEIL, EASE), incorporating discussion on the latest advancements is essential for ACL/ICML submissions. Recent work, such as those focusing on efficient RL methods, sample-efficient learning, or incremental evaluation techniques, might share goals or constraints with AUTOSELECT.

*   **Suggestion:** The authors should incorporate and contrast their work against the core ideas presented in very recent or forthcoming literature, potentially addressing themes like:
    *   Incremental/Fair evaluation methods (e.g., "Let The Jury Decide: Fair Demonstration Selection for In-Context Learning through Incremental Greedy Evaluation (ACL, Findings 2025)").
    *   New RL paradigms or sample-efficient selection strategies (e.g., "Demonstration Selection for In-Context Learning via Reinforcement Learning (ICML 2025)" or "Sample Efficient Demonstration Selection for In-Context Learning (ICML 2025)").
    *   New architectural or strategy comparisons (e.g., "Revisiting Demonstration Selection Strategies in In-Context Learning (ACL 2024)").

### 2. Quantitative Analysis of Training Cost vs. Greedy Oracle

While the paper justly criticizes Greedy-oracle for its extremely high computational cost during selection time (requiring exhaustive enumeration and evaluation of remaining candidates at every step), AUTOSELECT's training phase also necessitates querying the task-solving LLM to compute the sequence-level reward $r(\bar{x}, \tau)$.

*   **Suggestion:** A quantitative comparison is needed: What is the **total number of LLM API calls** required to train the AUTOSELECT policy (e.g., 400 episodes $\times$ 3 rollouts $\times$ 5 aggregated samples) versus the estimated calls required by Greedy-oracle to select demonstrations for the **400 testing samples**? Providing this quantitative trade-off would fully justify the "modest, one-time offline investment".

### 3. Scaling Limitations for Long Context Applications (RAG)

The paper acknowledges that extending the auto-regressive selection paradigm to **long-context applications** like Retrieval-Augmented Generation (RAG) is a challenging direction due to potential computational complexity. Since the core paradigm's complexity scales with sequence length, this current limitation suggests the method's scope is currently constrained primarily to few-shot ICL where $T$ is small (e.g., $T=4$ in the main results).

*   **Suggestion:** The authors could briefly discuss potential pathways or architectural modifications (e.g., using sparse attention or focusing on key points in the context prefix $\tau_{<t}$) that might be explored to make the auto-regressive policy model tractably scale to significantly larger selection lengths (e.g., $T>32$) in future work.

### 4. Deeper Insight into ViT/2D Matrix Synergy

The paper asserts that the ViT-based policy operating on 2D matrix embeddings is superior because it better captures **structural information** and **inter-exemplar relationships** compared to a GPT-2 variant operating on flattened vectors.

*   **Suggestion:** While the ablation confirms the performance gain (Fig. 11), the explanation of *why* the ViT architecture is fundamentally better suited to capture these specific text relationships remains somewhat abstract. Providing qualitative analysis or visualization (e.g., attention maps showing how the ViT processes the relationships between the query matrix and previously selected exemplar matrices) would offer deeper insight into this unique architectural strength.

### Questions
1.  **Selection of Maximum Trajectory Length $T$:** The main results (Table 1) use a maximum length $T=4$. However, the parameter sensitivity analysis (Fig. 8, left) shows that increasing $T$ to $7$ and potentially beyond **consistently improves performance** across tasks. Why was $T=4$ chosen as the default setting for the primary comparison, and would the relative performance advantage of AUTOSELECT over other learning-based baselines (like CEIL and EASE) hold if they were all evaluated using the superior $T=7$ setting?
2.  **Robustness of EOS Perturbation ($\lambda$):** The End-of-Sequence (EOS) signal embedding is initialized using the average exemplar embedding plus a small random perturbation $\lambda$. While Table 4 shows stable performance around $\lambda=0.01$, could the authors elaborate on the theoretical or empirical benefit of introducing this perturbation *instead of* setting $\lambda=0$? Does the small noise prevent the policy from overfitting to the mean embedding, thereby encouraging broader exploration, and is this effect consistently beneficial across all LLM backbones?
3.  **Technical Hurdles for Zero-Shot Cross-Domain Generalization:** Figure 5 demonstrates that simple adaptation (A50/A100) significantly improves transferability, achieving near-optimal results on target tasks. In the case of **direct transfer (No Adaptation, NA)**, the performance gains are smaller. What is the single biggest technical challenge currently preventing the policy $\pi_{\theta}$ from achieving robust zero-shot generalization across diverse tasks (e.g., from reasoning to classification) without requiring any target task adaptation episodes? Does addressing this require a fundamental shift in the policy architecture (e.g., meta-learning or a more complex task embedding mechanism)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors propose auto-selector, a novel framework for selecting demonstrations in few-shot in-context learning that formulates the problem as an auto-regressive sequential decision process. The key innovation lies in treating demonstration selection similarly to how LLMs generate text - building sequences one exemplar at a time while conditioning on the query and previously selected demonstrations. The framework employs a Vision Transformer (ViT) operating on 2D matrix embeddings to preserve token-level structural information, and optimizes a policy using a theoretically-grounded Cross-Entropy (CE) loss that provably bounds the discrepancy between the learned policy's induced Plackett-Luce (PL) ranking and the optimal ranking. The method is evaluated across nine diverse datasets and shows consistent improvements over existing baselines, with particularly strong gains on challenging reasoning tasks.

### Strengths
1. This paper introduces the demonstration selection problem in the context of Plackett-Luce ranking optimization with theoretical support. 
2. This paper enables the dynamic termination where the model can learn to optimize the sequence lengths for different queries and tasks.

### Weaknesses
1. The main concern lies in the computational complexity cost. The autoregressive approach can only rank the given exemplar in a step-by-step fashion. This method could lead to a high computation cost compared to the traditional point-wise ranker.
2. The performance improvement is rather marginal. While consistent, improvements on some datasets (AGNews: -0.4%, Amazon: +0.8%) are quite modest, raising questions about practical significance versus computational cost trade-offs.
3. The method is primarily evaluated on short sequences (T≤4, with extensions to T=16), but scalability to much longer sequences relevant for retrieval-augmented generation or longer context scenarios remains unclear.

### Questions
1. How does the training and inference time scale with larger exemplar pools (e.g., 500-1000 candidates)? What are the practical limits of the approach?
2. How sensitive is the method to the size and quality of the validation set used for reward computation? What happens when validation data is limited or noisy?
3. Beyond the analysis provided, how sensitive is performance to other key hyperparameters like the aggregation size |D_aggr|, temperature scheduling, or replay buffer size?
4. How does the method perform when extended to much longer sequences (T>16) that might be relevant for document-level tasks or retrieval scenarios?

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
5

### Summary
The AUTOSELECT framework selects  few-shot demonstration samples by formulating it as an auto-regressive sequential decision process. At each step, a trainable policy model sequentially selects the next best exemplar. The policy is trained by minimizing policy-level Cross-Entropy loss, which efficiently enables the policy to align its induced Plackett-Luce (PL) ranking with the optimal ranking, prioritizing high-quality demonstration sequences based on sequence-level rewards.

### Strengths
The claimed novelties are:
- Matrix representation for the policy.
- The "constrained" RL problem that the authors solve to get the optimal policy from the ranking set of generated policies.
- The authors compare with previous order-dependent selection methods only, EASE and CEIL..

### Weaknesses
- The authors compare only using qwen 2.5 3B model resulting in performance score that woefully lower than state of the art methods. For example, the authors report 40% acc to AQUARAT dataset, whereas the SOTA is close 80% on the same. The experiemntal results are not at all convincing.

- The authors emphasize the Plackett-Luce (PL) ranking objective that basically ranks the trajectories on the actual performance, and then optimizes the parameters using the collection. This seems like a simple policy gradient algorithm. Also, the authors call the KL divergene regularized objective as a "constrained" RL problem. I dont think the nomeclature in standard.

- FInally the Aurhors have completely missed the literature on order-independent selection methods from the literature survey and comparison. Since their model is more general, they should show that their model outperforms these methods. e.g.

Purohit, Kiran, V. Venktesh, Raghuram Devalla, Krishna Yerragorla, Sourangshu Bhattacharya, and Avishek Anand. "EXPLORA: Efficient Exemplar Subset Selection for Complex Reasoning." In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 5367-5388. 2024.

### Questions
- Why have the authors not performed experiments with a more recent model like qwen3 8B ?

- What is the architecture of the policy model?

- How many trajectory rollouts were needed to train the policy model?

### Soundness
2

### Presentation
2

### Contribution
1
