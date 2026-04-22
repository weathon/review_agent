# Beyond RLHF: A Theoretical Framework of Alignment as Distribution Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 4, 2, 2

## Abstract
Alignment via reinforcement learning from human feedback (RLHF) has become the dominant paradigm for controlling the quality of outputs from large language models (LLMs). However, the standard RLHF objective lacks formal justification and incentivizes degenerate, deterministic LMs in the asymptotic regime. We ask under what assumptions can we derive RLHF or other novel objectives with rigorous learning-theoretic guarantees, without relying on an \emph{a priori} notion of reward maximization. To this end, we reframe alignment as \emph{distribution learning} from pairwise preferences, formalizing our approach with a probabilistic assumption describing how preferences reveal information about the target (oracle) LM. This leads us to propose three principled alignment objectives: preference maximum likelihood estimation, preference distillation, and reverse KL minimization. We prove that all three approaches enjoy strong non-asymptotic $O(1/n)$ convergence to the target LM, naturally avoiding degeneracy. In particular, reverse KL highly resembles the RLHF objective, providing strong justification for RLHF with a minor correction. Furthermore, our theory confirms the empirical finding that on-policy objectives (e.g., RLHF) often outperform likelihood-style objectives (e.g., DPO). Finally, we empirically show that our proposed methods consistently matches or outperforms baselines across various tasks and models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work proposes 3 LLM alignment objectives from distribution learning perspective, i.e., preference maximum likelihood estimation, preference distillation, and reverse KL minimization.

### Strengths
The proposed three methods look well derived and motivated. The idea is presented clearly. There seems sufficient experimental details in the appendix.

### Weaknesses
The relationship and comparison among the three methods are not clear to me. In the introduction, could you briefly compare the proposed three methods and when to use which? Also, the experimental performance improvement does not look significant.

### Questions
(1) In the introduction, could you briefly compare the proposed three methods and when to use which? 

(2) Line 94: "where as" to "whereas". 

(3) Line 143: What is "cyclic preference"? I have not found this term in existing literature, so you could explain in the paper. Also, how can your model (1) allow cyclic preference?

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
3

### Summary
This work reviews language model alignment as a distribution learning problem rather than reward maximization. Instead of relying on the RLHF objective, the authors assume a probabilistic Bradley–Terry preference model and derive three principled methods—Preference MLE, Preference Distillation, and Reverse KL Minimization. For each method, O(1/n) convergence to the target model is established. The Reverse KL formulation further shows that RLHF can be viewed as an approximation of minimizing the KL divergence between the learned and ideal distributions.

### Strengths
1. This work provides a new view for alignment with theoretical justifications. 
2. The proposed method has empirical improvement over RHLF, reliable method and DPO.

### Weaknesses
While the paper provides a theoretical framework for viewing alignment as distribution learning, its empirical validation remains limited. The experiments are conducted on relatively small models and limited number of tasks. The performance gains over DPO, RELABEL and RLHF are modest. Evaluations on larger language models and more diverse tasks would be necessary to fully demonstrate the scalability and practical advantages of the proposed approaches.

### Questions
Where are assumptions (i) and (ii) in lines 140–151 formally stated? It would be clearer if the authors presented them in a more explicit way.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper approaches the alignment problem from a distribution-learning perspective rather than the traditional RLHF framework, aiming to provide a deeper understanding of model behavior and theoretical guarantees for convergence toward a target distribution as the sample size increases. Assuming the existence of a target language model that assigns higher likelihoods to preferred responses, the authors propose three algorithms—PMLE, Preference Distillation, and Reverse KL—whose solutions provably converge to the target model. Experimental results show that these methods consistently match or outperform baseline approaches across diverse tasks and model settings.

### Strengths
- The paper provoide an interesting distributional learning perspective for LLM Alignment.
- The experimental results demonstrate that the proposed methods can mitigate the alignment tax problem and improve performance compared to prior methods.

### Weaknesses
* **Lack of theoretical justification for the distribution-learning perspective.** The paper assumes the existence of an optimal language model that determines the preference probabilities in Eq. 2. However, such an optimal policy can also be derived under a reward-maximization objective with entropy regularization [1]. It remains unclear whether this distribution-learning perspective truly goes beyond the notion of reward maximization in alignment.

* **Unclear implications of the theoretical rate improvement.** The theoretical analysis derives a $1/n$ upper bound, improving upon the $1/\sqrt{n}$ rate from prior works. However, a major challenge in alignment is reward over-optimization [4,5], and simply scaling the sample size $n$ does not mitigate this issue. Prior pessimism-based approaches indicate that avoiding over-optimization requires ensuring single-policy coverage [2,3]. The paper does not demonstrate whether the three proposed algorithms achieve such coverage, nor whether improving the convergence rate to $1/n$ alone meaningfully mitigates over-optimization in practice.

* **Strong similarity to prior reward-maximization methods.** The three proposed methods closely resemble previous reward-based objectives: PMLE parallels DPO with a Reverse KL term [6], Preference Distillation resembles Online-DPO [7], and Reverse KL aligns with RLHF with an added entropy bonus.

## References
[1] Maximum Entropy Inverse Reinforcement Learning. AAAI 2008.

[2] Correcting the Mythos of KL-Regularization: Direct Alignment without Overoptimization via Chi-Squared Preference Optimization. ICLR 2025 Spotlight.

[3] REBEL: Reinforcement Learning via Regressing Relative Rewards. NeurIPS 2024.

[4] Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms. NeurIPS 2024.

[5] Scaling Laws for Reward Model Overoptimization. ICML 2023.

[6] The Importance of Online Data: Understanding Preference Fine-tuning via Coverage. NeurIPS 2024.

[7] Direct Language Model Alignment from Online AI Feedback. CoRR 2024.

### Questions
- Can PMLE and the other two objectives achieve a single-policy coverage coefficient?

- Why use Reverse KL instead of Forward KL (SFT-style) regularization? What is the theoretical justification for applying Reverse KL in all three objectives from the distribution-learning perspective? Proposition 12 suggests that Reverse KL is not required to achieve the $1/n$ sample rate, so what additional role does it play?

- What distinguishes PMLE from DPO + Reverse KL, and Preference Distillation from Online-DPO? Can DPO + Reverse KL and Online-DPO achieve the same $1/n$ sample rate? If not, why?

- How do PMLE and Preference Distillation compare to DPO + Reverse KL and Online-DPO as the number of samples increases?
Does PMLE mitigate over-optimization as sample size grows, compared to $\chi$-PO [2]?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Proposes a novel perspective on alignment by framing it as distribution learning from pairwise preferences.,Introduces three new algorithms: preference maximum likelihood estimation, preference distillation, and reverse KL minimization, which are not present in existing literature.,Provides rigorous learning-theoretic guarantees for the proposed methods, which is a significant advancement over traditional RLHF approaches.

### Strengths
Addresses a critical gap in the theoretical justification of RLHF objectives, which has been largely taken for granted in prior works.,The proposed methods potentially improve the safety and effectiveness of LLMs by avoiding degeneracy associated with traditional RLHF.,Empirical validation showing that the new methods outperform existing baselines indicates practical relevance and applicability.

### Weaknesses
1. Novelty: Although the objectives are distributional, the strong convergence analysis has been developed by several works before [1,2,3], but the authors neither cite those works nor highlight the novelty of their analysis in the work. The authors need to state the novelty of their analysis.

Reference:
[1] Zhao, H., et al., Logarithmic regret for online kl-regularized reinforcement learning
[2] Wu, D., et al, Greedy Sampling Is Provably Efficient for RLHF
[3] Zhao, H., et al, Sharp analysis for kl-regularized contextual bandits and rlhf

2. The idea of distributional learning has also been studied before [1,2] ..., named as generative reward model (GRM). The authors also do not mention this line of work. Could they discuss the relationship of their work with GRM?

Reference:
[1] ZHong, H, er al,DPO Meets PPO: Reinforced Token Optimization for RLHF
[2] Dakota Mahan, et al, GENERATIVE REWARD MODELS

### Questions
1. The applications of their distributional objective seem to be limited compared to general RLHF, since they need first to learn a good policy distribution and then approximate that distribution. Besides, there will exist an approximation error between the surrogate policy and the true optimal policy. This error is neglected in their theoretical bound.

### Soundness
3

### Presentation
3

### Contribution
2
