# RESCHED: Rethinking Flexible Job Shop Scheduling from a Transformer-based Architecture with Simplified States

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Neural approaches to the Flexible Job Shop Scheduling Problem (FJSP), particularly those based on deep reinforcement learning (DRL), have gained growing attention in recent years. However, existing methods rely on complex feature-engineered state representations (i.e., often requiring more than 20 handcrafted features) and graph-biased neural architectures. To reduce modeling complexity and advance a more generalizable framework for FJSP, we introduce \textsc{ReSched}, a minimalist DRL framework that rethinks both the scheduling formulation and model design. First, by revisiting the Markov Decision Process (MDP) formulation of FJSP, we condense the state space to just four essential features, eliminating historical dependencies through a subproblem-based perspective. Second, we employ Transformer blocks with dot-product attention, augmented by three lightweight but effective architectural modifications tailored to scheduling tasks. Extensive experiments show that \textsc{ReSched} outperforms classical dispatching rules and state-of-the-art DRL methods on FJSP. Moreover, \textsc{ReSched} also generalizes well to the Job Shop Scheduling Problem (JSSP) and the Flexible Flow Shop Scheduling Problem (FFSP), achieving competitive performance against neural baselines specifically designed for these variants.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ReSched that emphasizes a minimalist approach by redefining the state representation of FJSP, by only using four features compared to the current state-of-the-art method DANIEL, which uses a total of 26 features.  The paper also shows that many of the features of DANIEL are redundant and that performance increases when some are removed. Additionally, the paper introduces a novel network architecture, whereby they show through extensive testing the effect of the components of the network architecture. Their results indicate that ReSched outperforms DANIEL and other baselines, without a significant increase in the evaluation runtime.

### Strengths
The authors show that existing DRL work uses too many features to train, and they also show an ablation study for the features of DANIEL, further strengthening their claims for a minimalist approach.

The Ablation study of ReSched in Table 5 is well done, and it shows the effect of each introduced component of their network architecture.

### Weaknesses
The use of the REINFORCE algorithm over PPO is not well-justified. It is well-known that the vanilla REINFORCE method suffers from high variance, and the paper does not report training across multiple seeds. This raises doubts about the results presented in the paper, as training over multiple seeds might reveal that DANIEL outperforms ReSched. While DANIEL also trained on only a single seed, they used PPO, which is known to be more stable.

REINFORCE is also known to be highly sample-inefficient. The paper states that ReSched trains for smaller instances for 2000 epochs, with 1000 instances each epoch, with a batch size of 50. This is significantly longer than baselines, as both HGNN and DANIEL, who also uses 1000 instances, but with a batch size of 20, train in a single “epoch”, meaning they only use these instances once. This means that DANIEL and HGNN will have $1000 \times 20= 20000$ different episodes to train on, whereas ReSched has $2000 \times 50 \times 1000= 100000000$. This means that ReSched has 5000 times the training data compared to the baselines, which it compares to. This could mean that HGNN and, especially, DANIEL could outperform ReSched whenever they are trained on the same amount of data.

The contribution of ReSched is marginal in my opinion, since it only introduces a new state representation and network architecture. Moreover, due to previously stated weaknesses, I am not fully convinced about the results and, consequently, the significance of these novel parts.

The paper uses Definitions, Propositions, Corollaries, and Remarks quite liberally. For example, Corollary 1 is completely based on unproven Proposition 1. I can understand that Proposition 1 is likely true, but I would not state it in this way without giving a formal proof.

### Questions
Could the authors explain why $\gamma=0.99$, when it is normally set to $\gamma=1$, such as for L2D and DANIEL?

Could the authors maybe show if their proposed network architecture works well, with PPO or REINFORCE with a custom baseline? Their current motivation of REINFORCE being simpler is not strong, given that we use methods like PPO for a reason, like better sample efficiency and more stable training.

Could the authors show the performance of ReSched when trained on a similar amount of data as DANIEL and HGNN? 

In Table 3, it is unclear whether ReSched uses the greedy or sampling evaluation for JSP. Could you please clarify which one is used?

### Soundness
2

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
4

### Summary
This paper presents a new neural network architecture and training procedure for different types of scheduling problems, with a particular focus on the FJSSP. The authors also discuss applications to the classical JSSP and the Flow Shop Scheduling Problem.

The main idea of this work is to employ a simplified feature space as input encoding for a transformer-based architecture that uses a non-learnable positional encoding. In contrast to previous research that relies on extensive and often redundant feature representations, the authors deliberately restrict the input to a small set of features that effectively capture the current system state without incorporating information about past actions. This enables the network to focus on a specific sub-problem at each action step.

Through extensive experiments, the authors demonstrate that their method outperforms several existing approaches and is capable of establishing new benchmark results across a variety of problem instances and sizes. The model is trained using deep reinforcement learning with the REINFORCE algorithm, where advantage estimates are incorporated into the policy gradient computation.

The authors also state that they plan to release their implementation publicly.

### Strengths
- The paper strikes a balanced narrative between reviewing foundational concepts of the FJSSP and introducing the method. Explanations are concise and accessible, enabling readers to follow both theoretical background and model design decisions.

- Although the transformer architecture itself is not new, the authors extend it with relevant innovations (such as rotary positional encodings) and tailor the input representation to the scheduling context. Further, the authors introduces a deliberate feature reduction strategy, focusing only on state-relevant features, a nontrivial design choice in scheduling, where feature selection is often a bottleneck. In sum, the application and evaluation of these architectural modifications in FJSSP (and its extensions) represent a creative adaptation of transformer models to a new domain.

- The paper provides a solid theoretical justification for using a reduced feature space as input to the transformer network. The claims are supported by a direct comparison with another SotA method that uses a richer feature representation, showing improved performance under feature restriction. Comprehensive ablation studies isolate and quantify the contribution of architectural components.

- The proposed method consistently outperforms strong baselines in extensive experiments including both greedy and sampling-based inference strategies. The authors demonstrate a generalization to classical JSSP and FSSP indicating high potential for broader applicability beyond the initial problem formulation. The results contribute actionable insights to the research community, emphasizing the importance of carefully chosen feature representations in neural approaches to scheduling.

### Weaknesses
- The experimental evaluation lacks a comparison of computation time between the proposed method and other neural network–based approaches. This is particularly relevant because the method supports sampling-based inference, which may introduce higher computational cost. A runtime analysis (such as wall-clock time per episode or inference step) would clarify the trade-off between solution quality and computational efficiency and help position the method against deep learning baselines.

- In Figure 1a, the use of plus signs suggests that both solutions are combined to generate a schedule, although the solutions seem independent and not actually aggregated. This may mislead readers. In Figure 2, certain elements, especially the graphs in the upper part of the decision-making module, are not clearly explained in the text, making their meaning and role ambiguous. Enhancing these figures or adding brief explanations in the caption would improve interpretability.

- In the appendix, the authors show that the DANIEL network improves notably when fewer features are used. Table 1 further indicates that DANIEL is generally the second-best method and even outperforms the proposed approach in certain cases. However, the paper does not evaluate how a feature-reduced DANIEL network would perform directly against the proposed model. Since the central contribution of the paper is the effectiveness of a reduced feature space, a head-to-head comparison with a feature-restricted DANIEL baseline would be essential to validate that the performance gain is due to the proposed method, and not merely an effect of feature reduction.

- Appendix C presents a key–value cache mechanism, but its relevance to the main paper is unclear, as no inference-time results are reported. Without linking the cache mechanism to performance measurements, the section feels isolated rather than integrated into the contribution. Including inference-time evaluation, or explicitly discussing when and how the cache is beneficial, would strengthen the connection to the main content.

### Questions
1. In Appendix B.3, could you clarify whether the machine embeddings are entirely removed, or whether only the global descriptors are excluded? If no machine features are present, it is unclear how the Dual Attention Network would function. Were these embeddings perhaps replaced with placeholder values, such as zero entries?
2. How does your method compare to the DANIEL network when both use a reduced feature space? In the appendix, you demonstrate that the DANIEL network improves in performance when fewer features are used. Furthermore, Table 1 shows that the DANIEL network generally performs as the second-best method and even outperforms your proposed approach in certain cases. How would a feature-reduced version of the DANIEL network compare directly to your method?
3. The paper presents several ablation studies analyzing various components of the proposed model, but the choice of the REINFORCE algorithm appears to have been assumed without direct justification. Since all competing methods were trained using a Proximal Policy Optimization algorithm, was this alternative also tested? If so, did it yield inferior results? Including such a comparison in the ablation studies would strengthen the empirical analysis.
4. How does the formulation of the reward function impact the training process? This question is particularly relevant since the DANIEL method learns based on the change in the current makespan rather than the change in the expected lower bound.
5. Why was the inference speed presented in only a single figure and not compared across all experiments?
6. Why were no ablation studies conducted on the network size, such as the number of attention heads or Transformer block dimensions? This seems highly relevant when introducing a new architecture.
7. Why was the RoPE positional encoding chosen? Were other positional encoding methods considered, and how might their use impact learning performance, particularly regarding inference speed and makespan?
8. A variety of metaheuristics exist for solving the FJSP. Why were none of these methods investigated or compared against the proposed approach?
9. The FJSP, JSSP, and FFSP represent relatively limited and less complex scheduling problems. How could this approach be extended to handle a broader range of scheduling problems, including those with more diverse and realistic constraints encountered in real-world applications?

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
4

### Summary
The paper claims that a minimal Markov-sufficient state for FJSP can be formed using four core features. It tries to remove historical dependencies, redundancy, and auxiliary variables from node features, and introduces a dual-branch Transformer: an operations branch with self-attention enhanced by RoPE and a machine branch with cross-attention that injects edge features and self-connections to stabilize operation–machine imbalance. The method reports SOTA on FJSP with promising transfer to JSSP/FFSP.

### Strengths
The paper shows that strong performance is possible with a small, domain-aware feature set.

Structure-aware Transformer well aligned with precedence and operation-machine eligibility.

It demonstrates sample efficiency and generalization across instance sizes in FJSP, one of the most challenging COPs.

### Weaknesses
The “minimal MDP” claim (proposition 1) lacks a formal proof; it relies mainly on empirical results.

Related work on feature minimization is incomplete: Lee & Kim (2024) already aimed to remove historical dependencies and relative time feature (global minimum available time subtraction) in JSSP; conceptually aligned with “State: SubProblem” design (line 229 in this paper). Authors analyze only DANIEL (Wang et al., 2024b) features in FJSP, but feature minimization in JSSP had already been explored. 
- Lee, J. H., & Kim, H. J. (2025). Graph-based imitation learning for real-time job shop dispatcher. IEEE T-ASE.

JSP comparisons are incomplete and relatively weak. While FJSP results include several recent methods, JSP needs stronger baselines. For example:
– Park, J., Bakhtiyar, S., & Park, J. (2021). ScheduleNet. arXiv:2106.03051.
– Chen, R., Li, W., & Yang, H. (2022). TII 19(2):1322–1331.
– Iklassov, Z. et al. (2023). IJCAI, pp. 5350–5358.
– Lee, J. H., & Kim, H. J. (2025). Graph-based imitation learning for real-time job shop dispatcher. IEEE T-ASE.

### Questions
1. Can you provide a formal proof that the four-feature state is Markov-sufficient (proposition 1)?

2. You excel on FJSP but are less competitive on recent JSP baselines. What structural or distributional differences make JSP harder for your method?

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
4

### Summary
The paper proposes RESCHED, a deep reinforcement learning framework for solving scheduling problems. It proposes a simplified state representation and Transformers to solve Job Shop Scheduling Problems. The experiments show better performance than DANIEL, and RESCHED is even surpassing OR-TOOLS on 40x10 instances. The architecture used is a decoder only architecture and the reinforcement learning algorithm is REINFORCE. It learns a policy that is then used either greedily or with sampling.

### Strengths
The paper has good results on large instances.
A new state representation and architecture are proposed.

### Weaknesses
The writing is not polished (Experments, Dateset, ...)
The training times of the different DRL approaches are not compared.
It would be better to also have OR-TOOLS on Taillard benchmark.

### Questions
Can you detail the sampling algorithm used in Table 1?

### Soundness
3

### Presentation
2

### Contribution
3
