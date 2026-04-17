# Iterative Distillation for Reward-Guided Fine-Tuning of Diffusion Models in Biomolecular Design

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
We address the problem of fine-tuning diffusion models for reward-guided generation in biomolecular design. While diffusion models have proven highly effective in modeling complex, high-dimensional data distributions, real-world applications often demand more than high-fidelity generation, requiring optimization with respect to potentially non-differentiable reward functions such as physics-based simulation or rewards based on scientific knowledge. Although RL methods have been explored to fine-tune diffusion models for such objectives, they often suffer from instability, low sample efficiency, and mode collapse due to their on-policy nature. In this work, we propose an iterative distillation-based fine-tuning framework that enables diffusion models to optimize for arbitrary reward functions. Our method casts the problem as policy distillation: it collects off-policy data during the roll-in phase, simulates reward-based soft-optimal policies during roll-out, and updates the model by minimizing the KL divergence between the simulated soft-optimal policy and the current model policy. Our off-policy formulation, combined with KL divergence minimization, enhances training stability and sample efficiency compared to existing RL-based methods. Empirical results demonstrate the effectiveness and superior reward optimization of our approach across diverse tasks in protein, small molecule, and regulatory DNA design.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper

### Strengths
In general, the exposition is well done. I don't have a ton of experience on modern RL approaches for training diffusion models and I found the explanations fairly approachable.

The evaluation setup is from prior work from papers that are more application-specific. In general, creating reward functions that are cheap computational surrogates for wet-lab experiments is difficult. I work in this area. I think the ones that are used are reasonable choices.

### Weaknesses
It was hard for me to understand which parts of Section 4 are novel. See question below.

The main paper provides few experimental results. However, the supplement has some really interesting follow-on results. It would have been nice to see some of these ablations and analyses featured in the main paper (e.g., fig 4).

The 'approximate soft-value functions' approach makes approximations that may be risky to make in practice. I would have liked to see a contrast with a sampling-based approach to estimating this.

### Questions
I found the explanation of 'Approximation of soft value functions' (line 8 in Alg 1) a little too concise. In what sense is this a posterior mean? Why is it possible to evaluate a discrete reward function on this 'mean'?
 
It was hard for me to understand which parts of Section 4 are novel. Can you please provide more details about the relationship to prior work? Also, are there special cases or modifications of your framework that would make the method similar to other RL methods (e.g. if the roll-out policy was the same as the roll-in policy)? Drawing these connections can help readers understand things better. I appreciate section 6, but it was hard for me to understand if there is other related work that is similar to yours instead of a policy gradient method.

To what extent is the use of a diffusion model orthogonal to the particular RL approach? The inverse process of the diffusion model is just just a generic policy. Why is your presentation specific to diffusion models? Are there other non-diffusion policies where your approach would be sensible?

What if, for example, you applied your approach to an autoregressive protein language model instead of a discrete diffusion model? I'm not suggesting you do this experiment. I'm just trying to understand why diffusion models are the focal point.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on addressing the challenge of reward-guided fine-tuning of diffusion models in biomolecular design. Diffusion models are effective at modeling the complex distributions of biomolecules like proteins, small molecules, and regulatory DNA. However, real-world applications require optimizing for core rewards—such as physics-based protein binding affinity, small molecule docking scores, and science-driven protein secondary structure matching scores—that are non-differentiable, making traditional gradient-based methods inapplicable. Existing reinforcement learning (RL)-based fine-tuning approaches (e.g., DDPO, PPO) face issues including training instability, low sample efficiency, and mode collapse due to their on-policy nature and reliance on reverse KL divergence.

To tackle these problems, the authors propose VIDD (Value-guided Iterative Distillation for Diffusion models), a framework that treats fine-tuning as a policy distillation task, enabling optimization for any type of reward (including non-differentiable ones) through three iterative stages. First, the Roll-in stage collects off-policy trajectories by mixing samples from the pre-trained diffusion model (to explore a wide range of biomolecular design spaces) and a stable roll-out policy (to leverage high-reward regions), decoupling data collection from policy updates to improve sample efficiency. Second, the Roll-out stage simulates a soft-optimal "teacher policy" that balances reward maximization and compliance with the pre-trained distribution. This policy is weighted by a value function approximated using the diffusion model’s prediction of the clean sample , eliminating the need for reward gradients. Third, the Distillation stage updates the "student model" (the fine-tuned diffusion model) by minimizing the forward KL divergence between the soft-optimal policy and the current model policy, which prevents mode collapse compared to the reverse KL divergence used in RL methods.

Experiments across four biomolecular design tasks—protein secondary structure matching, protein-target binding, DNA enhancer design, and small molecule docking—demonstrate that VIDD consistently outperforms baseline methods (e.g., DDPO, DDPP, Best-of-N) in terms of reward metrics. For instance, it achieves 83% β-sheet content in proteins, 8.28 Pred-Activity in DNA enhancers, and a 9.4 docking score for small molecules. Meanwhile, it maintains biomolecular naturalness, as shown by metrics like pLDDT (a measure of protein structure confidence) and 3-mer correlation (a measure of DNA sequence naturalness).

### Strengths
1.Effective Adaptation to Non-Differentiable Rewards: VIDD bypasses the need for reward gradients by approximating soft value functions using the diffusion model’s x₀ prediction. This allows it to directly optimize core non-differentiable rewards in biomolecular design, such as binding affinity scores from AlphaFold and docking scores from QuickVina2, filling a key gap in fine-tuning diffusion models for scientific applications.
2. Training Stability and Anti-Collapse Ability: Unlike on-policy RL methods, VIDD uses off-policy data in the Roll-in stage, reducing sensitivity to noisy trajectories. By minimizing forward KL divergence instead of the reverse KL divergence used in RL, it forces the model to cover the distribution of the soft-optimal policy, avoiding mode collapse. For example, in protein binding tasks, VIDD maintains higher diversity than DDPO, and in small molecule design, it achieves a lower negative log-likelihood (NLL), indicating better preservation of biomolecular naturalness.

### Weaknesses
1.	Unaddressed Impact of Value Function Approximation Errors: VIDD relies on the diffusion model’s x₀prediction to approximate soft value functions, but it does not analyze how errors in x₀prediction—such as those arising from long protein sequences or complex DNA structures —affect the distillation process. Additionally, it fails to compare this approximation method with alternatives like Monte Carlo sampling, leaving uncertainty about whether this is the optimal approach.
2.	High Hyperparameter Sensitivity and Lack of Adaptive Strategies: Key hyperparameters—such as the roll-in mix ratio (βₛ), the roll-out policy update interval (K), and the value weight coefficient (α)—require manual tuning and have a significant impact on performance. For example, in DNA tasks, using K = 5 results in much higher activity than using K = 20. However, VIDD does not propose adaptive strategies (e.g., dynamically adjusting K based on reward progress), which increases the difficulty of applying it across different tasks.
3.	Insufficient Exploration of Robustness to Noisy Rewards: Biomolecular rewards often suffer from "proxy bias"—for instance, a predicted binding affinity score (ipTM) only approximates real binding affinity, and DNA activity predicted by Enformer may differ from actual cellular activity. VIDD does not test its performance under conditions of reward noise (e.g., random fluctuations) or bias, leaving unanswered questions about whether it will over-optimize for proxy objectives (e.g., producing proteins with high predicted binding affinity but no actual binding function).

### Questions
1.	Value Function Approximation and Error Correlation: VIDD uses x₀ prediction to approximate soft value functions, but the accuracy of x₀ prediction tends to decrease for long biomolecules (e.g., large protein complexes or DNA sequences with over 500 bases). Can you provide additional experiments to show the correlation between x₀ prediction errors (e.g., structural similarity between the predicted x₀ and the real x₀) and VIDD’s reward performance? Or can you compare this approximation method with alternatives such as Monte Carlo sampling or regression-based value estimation to verify its validity?
2.	Adaptive Hyperparameter Optimization: Hyperparameters like the Roll-in mix ratio (βₛ) and Roll-out update interval (K) have a major impact on performance. Manual tuning is time-consuming—for example, βₛ=0.8 optimizes protein diversity. Do you plan to design adaptive strategies, such as adjusting K based on the rate of reward improvement or optimizing βₛ based on the diversity of Roll-in data? Can you also provide heuristic rules for hyperparameter selection across different biomolecular tasks (e.g., recommended ranges for α in protein vs. small molecule tasks)?
3.	Robustness to Noisy and Biased Rewards: Biomolecular rewards often contain noise—for example, fluctuations in Enformer-predicted DNA activity or small molecule docking scores. How would VIDD perform if rewards are perturbed (e.g., ±10% Gaussian noise) or biased (e.g., systematically underestimating high-reward samples)? Are there calibration methods (e.g., multi-reward fusion) to reduce the impact of proxy bias?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposed VIDD, a novel reinforcement-learning-based method for diffusion models in biomolecule design, including protein, small molecule and DNA sequecnces. The authors found the limitations of the original methods, including mode collapse, computational inefficiency, and reward hacking during training. Then, VIDD solves them step by step with corresponding techniques. In experiments, results from three systems demonstrated the effectiveness of the general design.

### Strengths
1. The motivation and the insight for the method design make sense, and the proposed solutions are simple and easy-to-implement. 

2. It involves many biological systems, demonstrating the effectiveness and the robustness of the algorithm in different tasks.

3. VIDD unfied diffusion and value-weighted MLE with clear objective function and implementation, the framework could combine any type of non-differentiable reward functions, which is of high application range in biomolecule design.

### Weaknesses
1. The most concerning point from me is that the papre did not prove the effectiveness of each component through ablation studies, which makes the Method section not solid. More results on different biological systems are needed. 

2. Only limited baseline works were discussed and compared. Some baselines from discrete flow matching/diffusion on biological sequence design are of high correlation with this work [1-4]. 

3. More strict theoretical proof/analysis are needed (See Questions). 

4. There are irrational design of experimental validation (See Questions).

References: 

[1] Tang, S., Zhang, Y., & Chatterjee, P. PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion. In Forty-second International Conference on Machine Learning.

[2] Zhao, Y., Uehara, M., Scalia, G., Kung, S., Biancalani, T., Levine, S., & Hajiramezanali, E. Adding Conditional Control to Diffusion Models with Reinforcement Learning. In The Thirteenth International Conference on Learning Representations.

[3] Cao, H., Shi, H., Wang, C., Pan, S. J., & Heng, P. A. (2025). GLID $^ 2$ E: A Gradient-Free Lightweight Fine-tune Approach for Discrete Sequence Design. In ICLR 2025 Workshop on Generative and Experimental Perspectives for Biomolecular Design.

[4] Tang, S., Zhang, Y., Tong, A., & Chatterjee, P. (2025). Gumbel-softmax flow matching with straight-through guidance for controllable biological sequence generation. ArXiv, arXiv-2503.

### Questions
1. The results in DNA design system of DRAKES is different from the ones from original paper. Is there any setting changes?

2. In Section 5, the paper presents a theorem showing that the PPO-style objective is equivalent to a reverse KL divergence to the soft optimal trajectory distribution. However, for VIDD, it only states that it is “closer to a forward KL,” without providing a theorem or assumptions under which the empirical objective (7) is strictly equivalent to a forward KL.

3. Eq (5) and Eq(7) fall under the offline-RL framework. Could you please clarify the coverage condition for hte roll-in distribution. Moreover, can you analyze the coveragence under mixture roll-ins?

4. Can you do ablation studies on K, $\alpha$, to demonstrate the stability if lazy updates?

5. Also for Eq (5) and Eq (7), what is the differences/novelty of VIDD compared to exising reward-reweighted SFT or value-weighted MLE in offline RL?

6.For the “off-policy + forward-KL”, is there empirical evidence showing that VIDD maintains its claimed stability advantage under the same roll-in coverage and identical KL regularization budget?

7. Can you provide a formal proof/connection between VIDD and the mentioned inference-time methods?

8. I found that when the roll-in distribution and target teacher distribution are mismatched, Eq. (7) does not perform explicit importance correction. Will it bring biased estimation? If so, how is there any sols?

9. There should be general (mean/median) evaluation for sampling rather than simply using Best-Of-N.

10. More ablation studies should be added. 

11. Using AlphaFold-Multimer’s ipTM as the sole metric can be biased by the predictor itself. Can you try to report another metric to demonstrate the effectiveness, such as pTM, pDockQ?

12. For protein experiments, I think there would be potential data mismatch between EvoDiff and VIDD. Would you please clarify it?

13. Could you please show some insights towards the training distillation and test-time sampling tradeoff?

14. Can you also report log-likelihood in DNA system? It is because sometimes the 3-mer correlation cannot capture accurate simiarity among distributions.

### Soundness
2

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
4

### Summary
This paper presents **VIDD (Value-guided Iterative Distillation for Diffusion models)** — a reward-guided fine-tuning framework for diffusion models in biomolecular design tasks where reward functions are often non-differentiable.
VIDD reformulates RL-style fine-tuning as **off-policy, value-weighted maximum likelihood estimation** with a **forward KL** objective, addressing instability and mode collapse common in PPO-like methods.
It iteratively distills soft-optimal denoising policies through three phases: off-policy roll-in, reward-weighted roll-out, and KL-based distillation.
Experiments across **protein**, **DNA**, and **molecular** design tasks show strong and stable improvements over DDPO, DDPP, and other baselines, including differentiable settings.

### Strengths
1. **Principled Distillation Framework:** The paper introduces a principled distillation-based alternative to PPO-style fine-tuning, effectively addressing instability issues commonly seen in on-policy reinforcement learning for diffusion models.

2. **Handling Non-differentiable Rewards:** The proposed method applies to realistic biomolecular tasks where reward gradients are unavailable, bridging diffusion modeling with practical scientific design scenarios.

3. **Stability via Lazy Updates:** The lazy-update strategy allows training to use mostly stable, older data while periodically refreshing the roll-out policy to capture model improvements — striking a balance between stability and adaptivity.

4. **Comprehensive Evaluation:** The experiments are comprehensive, covering multiple scientific domains (protein, DNA, molecule) and providing ablations and comparisons with strong baselines.

### Weaknesses
1. **Heuristic Value Approximation:** The soft-value approximation via a single forward reward prediction (Algorithm 1, line 8) may introduce bias or variance, but the paper lacks quantitative analysis of its impact.
2.  **Limited Theoretical Rigor:** The forward-KL interpretation is intuitive yet only algebraically shown (Appendix B) without a formal derivation or stability proof.
3. **Comparisons on Differentiable Rewards:** For differentiable cases (e.g., DNA enhancer tasks), comparisons with gradient-based fine-tuning methods such as DRAKES are limited; more systematic evaluation would strengthen claims of generality.
4. **Overlap with Prior Work:** The framework is conceptually related to value-weighted MLE and offline RL distillation and the paper could better position its novelty beyond these precedents.

### Questions
1. How sensitive is VIDD to the choice of α (temperature) and lazy update interval K?

2. How does the algorithm behave when the reward oracle is noisy or misaligned (e.g., false-positive docking scores)? Does it still remain stable, or collapse due to biased reward?

3. Can VIDD be combined with differentiable reward gradients (hybrid)? For differentiable domains, could both gradient and value-weighted distillation terms coexist?

### Soundness
3

### Presentation
4

### Contribution
3
