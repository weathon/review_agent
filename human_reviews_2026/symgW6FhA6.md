# Triple-BERT: Do We Really Need MARL for Order Dispatch on Ride-Sharing Platforms?

- Avg Score: 6.50
- Decision: Accept (Oral)
- Scores: 8, 6, 6, 6

## Abstract
On-demand ride-sharing platforms, such as Uber and Lyft, face the intricate real-time challenge of bundling and matching passengers—each with distinct origins and destinations—to available vehicles, all while navigating significant system uncertainties. Due to the extensive observation space arising from the large number of drivers and orders, order dispatching, though fundamentally a centralized task, is often addressed using Multi-Agent Reinforcement Learning (MARL). However, independent MARL methods fail to capture global information and exhibit poor cooperation among workers, while Centralized Training Decentralized Execution (CTDE) MARL methods suffer from the curse of dimensionality. To overcome these challenges, we propose Triple-BERT, a centralized  Single Agent Reinforcement Learning (MARL) method designed specifically for large-scale order dispatching on ride-sharing platforms. Built on a variant TD3, our approach addresses the vast action space through an action decomposition strategy that breaks down the joint action probability into individual driver action probabilities. To handle the extensive observation space, we introduce a novel BERT-based network, where parameter reuse mitigates parameter growth as the number of drivers and orders increases, and the attention mechanism effectively captures the complex relationships among the large pool of driver and orders. We validate our method  using a real-world ride-hailing dataset from Manhattan. Triple-BERT achieves approximately an 11.95% improvement over current state-of-the-art methods, with a 4.26% increase in served orders and a 22.25% reduction in pickup times. Our code, trained model parameters, and processed data are publicly available at https://github.com/RS2002/Triple-BERT .

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Triple-BERT, a centralized single-agent reinforcement learning approach tailored for the order dispatch problem on ride-sharing platforms, aiming to address the challenges faced by traditional multi-agent reinforcement learning (MARL) methods in handling large-scale order dispatch scenarios. The paper introduces an innovative network architecture and training strategy that significantly enhances the efficiency of order dispatch and platform performance. Overall, the research is innovative, and the experimental results are impressive, offering substantial theoretical and practical contributions to the ride-sharing domain.

### Strengths
1. Clear Problem Definition
2. Triple-BERT innovatively addresses the large action and observation spaces through an action decomposition strategy and BERT network  The incorporation of the QK-Attention mechanism significantly reduces computational complexity.

### Weaknesses
While the experimental results are significant, the theoretical analysis in the paper is relatively weak. For instance, a deeper exploration of why the action decomposition strategy works effectively and its theoretical foundations could be further elaborated.
In the paper, the literature review on related work appears to be somewhat limited. For instance, significant contributions such as "Future Aware Pricing and Matching for Sustainable On-Demand Ride Pooling" , and studies exploring "MARL-Based Pricing Strategy via Mutual Attention for Mobility-on-Demand (MoD) Systems with Ridesharing and Repositioning," as well as "Mutual Information as Intrinsic Reward of Reinforcement Learning Agents for On-Demand Ride Pooling,"

### Questions
No

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a centralized single agent reinforcement learning method for large-scale order dispatching on ride-sharing platforms. In particular, the authors provide a two-stage training strategy to achieve good performance, i.e., 1) the first stage uses IDDQN for MARL pre-training that yields good feature extraction; 2) the second stage employs TD3 for SARL fine-tuning to get the final order dispatch result. In addition, the network backbone design consists of BERT, MLP and QK-attention. Experimental results validated the effectiveness of the proposed method.

### Strengths
(1) The motivation is well presented of using a two-stage training strategy for order dispatch on ride-sharing platforms.

(2) The explanations and illustrations are mostly clear and intuitive of the action space, the actor sub-network, the critic sub-network, and the solving algorithms.

### Weaknesses
(1) The main contribution is the two-stage training strategy but some other possible approaches are not analyzed. In particular, the title claims MARL is not required but used as pre-training, why not directly consider SARL without MARL pre-training.

(2) On Page 5, Line 236-237, the positional embedding is discarded due to permutation invariance of the input sequence. This is not a typical practice in the literature, it could be better if an ablation study is conducted for the difference. Besides, if permutation invariance is important for the whole framework, using GNN to model the input could be a better choice as done in [16,34].

(3) On Page 5, Line 269, this guarantees a unique solution. No proof or explanation on this point.

(4) On Page 6, Line 323-324, the authors assume that unsatisfactory independent MARL can provide a good starting point. This point is not proved or analyzed to some extent.

(5) On Page 6, Line 299-300, the authors claim that IDDQN is chosen for the stage-1 pre-training. However, in Table 1, the authors present the proposed Triple-BERT uses TD3 as a difference to comparison methods. It is a bit confusing since the whole framework is a two-stage approach and it is inappropriate to only highlight the stage-2 as a contribution.

### Questions
No.

### Soundness
3

### Presentation
3

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
This paper addresses the real-time order dispatching problem in large-scale ride-hailing platforms, where numerous drivers and passengers must be matched dynamically under uncertainty. The authors propose Triple-BERT, a centralized single-agent reinforcement learning approach designed to overcome the scalability and coordination limitations of multi-agent RL (MARL) frameworks. The method extends the TD3 algorithm with (i) an action decomposition strategy that factorizes joint action probabilities and (ii) a BERT-based attention architecture to handle large observation spaces efficiently via parameter reuse and contextual embedding of driver–order relationships.
Experiments on a real-world Manhattan ride-hailing dataset show improvements of roughly 12% in total performance metrics, with additional gains in served orders (+4.3%) and pickup times (–22%). Code and models are made publicly available.

### Strengths
1. Well-written and clearly structured: The paper is technically sound, logically organized, and easy to follow. The motivation for replacing MARL with a single-agent formulation is well articulated.

2. Practical relevance: Large-scale order dispatching remains a key benchmark problem in reinforcement learning for logistics and mobility; the paper contributes to this ongoing line of research.

3. Architectural innovation: The integration of a BERT-based network to manage relational structure and parameter reuse is elegant and practically valuable.

### Weaknesses
1. Limited discussion of related MARL coordination strategies: The paper does not engage with recent advances in local/global reward allocation or communication mechanisms that also address coordination challenges in MARL. Without such comparisons, it remains unclear whether the proposed single-agent framework is superior conceptually or primarily an engineering simplification.

2. Scope of novelty: While the combination of TD3 and BERT-based embedding is technically solid, the conceptual contribution lies mainly in neural architecture design rather than introducing a fundamentally new RL principle. The innovation should be framed more as an architectural contribution.

3. Statistical robustness: Results are averaged over only three random seeds, which limits statistical reliability. Given the inherent stochasticity of dispatching and RL training, more repetitions would strengthen confidence in the reported improvements.

4. Clarification on matching procedure: The algorithm ultimately uses a bipartite matching layer for stabilization, suggesting that global coordination is still enforced externally. The authors should clarify to what extent Triple-BERT learns full end-to-end dispatching policies versus providing improved matching scores.

### Questions
1. How does Triple-BERT compare to MARL approaches that use reward decomposition or credit assignment to improve coordination?

2. Could the authors elaborate on the role of the final bipartite matching step — is it part of the learning policy or a post-processing stabilization?

3. How sensitive are the results to hyperparameters and random seeds? Would results remain consistent under higher variance conditions?

4. Does the model generalize to other cities or datasets beyond Manhattan?

5. What are the main computational trade-offs of the BERT-based encoder compared to simpler attention or graph-based alternatives?

### Soundness
3

### Presentation
3

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
This paper proposes Triple-BERT, a centralized single-agent reinforcement learning (SARL) framework for large-scale ride-sharing order dispatch. It addresses MARL limitations (e.g., poor cooperation, high dimensionality) via a BERT-based architecture with self-attention to model driver-order relationships globally, QK-attention to reduce computational complexity, and action decomposition to simplify joint decisions. A two-stage training strategy (decentralized pre-training + centralized fine-tuning) mitigates SARL sample scarcity. Experiments show Triple-BERT outperforms MARL baselines by ~15% in reward, serving more orders with lower latency (<0.2s), and better scalability to thousands of agents.

### Strengths
- BERT’s self-attention effectively captures global driver-order interactions, enhancing cooperation.
- QK-attention reduces complexity from multiplicative to additive, enabling real-time dispatch.
- Handles 1,500–2,000 drivers/orders without retraining, outperforming MARL in high-concurrency scenarios.
- Two-stage training resolves SARL data scarcity via MARL pre-training.

### Weaknesses
- Action decomposition relies on an approximate "independent action probability" assumption, which may not fully reflect inter-dependencies.

### Questions
What challenges might be encountered when deploying this framework in practical applications?

### Soundness
3

### Presentation
3

### Contribution
3
