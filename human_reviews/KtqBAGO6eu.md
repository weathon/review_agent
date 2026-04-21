# Privacy Preserving Generative Feature Transformation

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 3

## Abstract
Data-Centric AI (DCAI) aims to use AI to get better data for better AI. Feature transformation, as one of the essential tasks of DCAI, can augment the data representation and has garnered significant attention. Existing methods have demonstrated state-of-the-art performance on advancing predictive tasks. However, these methods can lead to serious privacy leakage. For example, sensitive features in original data can be inferred by models trained on transformed data, exposing vulnerabilities in the privacy-preserving capabilities of these methods. To address this issue, we introduce a privacy-preserving feature transformation framework that transforms data representation while preserving privacy from a generative modeling perspective. Specifically, our framework includes two phases: 1) privacy-aware knowledge acquisition and 2) privacy-preserving feature space generation. In the knowledge acquisition phase, we develop an information bottlenecks guided reinforcement learning system to explore and collect privacy-aware feature sets as a knowledge base in token sequence form. In the feature space generation phase, we develop a generative model to encode the knowledge base into a privacy-aware latent space, where the best latent representation is identified and decoded into the optimal privacy-preserving feature space. We solve the optimization via projected gradient ascent that maximizes predictive performance and minimizes privacy exposure. Finally, we present extensive experiments on eight real-world datasets to evaluate how our method can navigate both performance and privacy. The code is available at https://anonymous.4open.science/r/anonymous-2B53/.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents a two-phase Privacy-preserving Feature Transformation (PFT) framework that integrates concepts from generative AI, the privacy information bottleneck approach, and gradient-based optimization. In the first phase, multi-agent reinforcement learning is utilized to generate privacy-aware feature sets, where the agents aim to maximize the mutual information between the transformed features and downstream tasks while minimizing the mutual information between the transformed features and sensitive attributes. In the second phase, generative models are employed to create privacy-preserving feature spaces that also maintain utility. This is achieved through a constrained gradient update, where the objective function optimizes performance metrics while the constraints ensure that the sensitive attributes become less exposed with each update iteration. Experiments on real-world datasets show that PFT enhances the balance between performance and privacy metrics, consistently achieving better results across all datasets based on the average of privacy and utility performance metrics.

### Strengths
1) This work addresses the challenge of privacy-preserving feature transformation, demonstrating that the two-phase PFT framework outperforms non-private feature transformation methods and even simple classic differentially private adaptations of these methods.

2) The paper introduces a novel combination of generative AI, the privacy information bottleneck approach, and gradient-based optimization to develop the framework.

3) The first three figures (Figures 1-3) effectively illustrate the concept of the paper. While some parts of the main text lack clarity, these figures assist in understanding the framework.

4) The diverse experiments, particularly those in Section 4.3.1 (correlation of features with downstream tasks and sensitive features) and Section 4.4 (ablation study), are well-motivated and provide valuable insights into the effectiveness of the PFT framework.

### Weaknesses
1) On page 4, it is stated that PFT maximizes the incremental performance of the feature space, which serves as a lower bound of the mutual information. However, the paper does not provide a justification for why it optimizes this lower bound instead of the mutual information itself. Additionally, in Equation (3), the derivation of this lower bound has issues. Specifically, inequality (c) is unclear, as the definition of information entropy suggests that for inequality (c) to hold, the following must be true:
\begin{equation*}
   \sum\limits_{F_i, y} p(F_i, y) \log(p(F_i \mid y)) \geq \sum\limits_{F_i} p(F_i) \log(p(F_i \mid y))
\end{equation*}
   However, this transformation generally does not hold, as the left-hand side uses the joint probability while the right-hand side uses the marginal probability.

2) It is mentioned in the appendix, that the parameter $\alpha$ that controls the trade-off between privacy and utility in the experiments is chosen from different values, with the best performance reported for PFT. However, for the baseline methods, parameters are taken directly from the original papers. This approach may not provide a fair comparison, as PFT's parameters are fine-tuned to maximize the average objective function reported in Tables 1 and 2, whereas the baseline methods, particularly the differentially private benchmarks, are not similarly optimized for this objective.

3) There are some misrepresentations and errors, particularly regarding the minimization and maximization of objectives. In Equation (1), while $\mathcal{L}$ appears to represent the loss function, it seems that the $ \min $ and $\max $ operators may have been interchanged. The utility loss function should be minimized, while the loss function for predicting the sensitive attribute should be maximized.

### Questions
1) While it is mentioned in the paper that privacy might be unmeasurable for the specific task of privacy-preserving feature transformation, is there a reason why differential privacy is not considered as a standard for measuring privacy?

2) I cannot follow Equation (3)(c). Could you please provide a justification for it? Additionally, I suggest using an index in Equation (3) for the summation, as it is unclear what the summation is over (which parameter).

3) In line 283, what exactly is $ v $? I know that in the appendix it is mentioned that pairwise accuracy can be used as a performance metric, but it is not clear from the main text how $\Psi_{pe} $ is trained. Could you please explain this further?

Here is a minor comment:

4)  In line 128: "We minimize the mutual information between the new feature space and downstream tasks while minimizing the mutual information between the new feature space and sensitive features." I believe "minimization" should be changed to "maximization."

If my concerns are addressed, I would be happy to consider increasing my score.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a privacy-preserving feature transformation pipeline. The pipeline consists of two parts: 
1. privacy-aware knowledge acquisition: A reinforcement learning framework is used to transform the feature set in a way that maximizes the mutual information (MI) between the transformed features and the downstream task and minimizes the MI between the transformed features and the sensitive features. 
2. privacy-preserving feature space generation: The transformed feature set from part 1 is treated as a sequence of tokens and map it to a latent representation using a sequential encoder. A decoder is used to map this representation back to the feature space. Performance and privacy evaluators are built to estimate the utility and privacy risks from the latent representation. The latent representation is modified using a constrained update rule in a way that improves utility while reducing the privacy leakage.

Evaluations are performed on multiple datasets to show that the proposed method can provide high performance on downstream tasks while making it harder to recover sensitive features.

### Strengths
1. The paper studies an important problem of privacy-preserving feature transformation.
2. Experiments seem extensive.

### Weaknesses
1. The proposed method does not provide strong theoretical privacy guarantees. Note that empirical evaluations of privacy is insufficient to claim that the proposed feature transformation technique can protects against all future privacy leakage attacks. 
2. In table 1, the proposed method seems to have high "SF" (prediction accuracy on sensitive features). Doesn't this mean that the proposed method is worse-off in terms of privacy compared to other methods? I also don't understand why a higher "Avg"  of "DT" and "SF" is desirable.

### Questions
1. Is there a mistake in Eq. 9? $\psi_{pe}$ does not appear in either the objective or the constraint.
2. Can you provide any guarantees for worst case privacy leakage for your proposal?
3. It's not clear why the proposed method has better privacy from the evaluations (see weakness #2)

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes a framework for performing feature transformation with privacy preservation. The framework first extracts privacy-aware knowledge from the dataset using information bottleneck-guided multi-agent reinforcement learning. It then generates a privacy-preserving feature space. The resulting feature space not only maintains privacy but also enhances the accuracy of downstream tasks.

### Strengths
1. The paper presents a valuable idea for feature transformation that effectively incorporates privacy preservation.
2. It includes clear visual explanations and sufficient mathematical details. The authors also provide comprehensive experimental results that demonstrate the effectiveness of their approach.

### Weaknesses
1. The paper lacks sufficient background information on reinforcement learning and the DQN structure, which could make it difficult for readers unfamiliar with these concepts to fully understand the framework.

2. The presentation could be improved. The related work section would be more effective if placed earlier in the paper, and some figures, such as Figure 5, are missing important details. Additional explanations are needed to make the visual information clearer and more informative.

### Questions
1. In Section 3.2.1, what specific DQN structure is used? It would be helpful to have more background information on this. Could you also explain why you chose this particular structure for your framework?

2. What does the X-axis represent in Figure 5? Could you provide more details on how this figure was generated and explain what it illustrates?

3. I assume that "sensitive features" refer to features that contain private information. How accurately does your method identify these sensitive features? In the evaluation section, could you explain in more detail how you assess the accuracy of sensitive feature detection?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper studies techniques to transform the feature space of the dataset while preserving privacy. A new method is proposed that consists of two steps 1) privacy-aware knowledge acquisition using multi-agent reinforcement learning  2) privacy-preserving feature space generation by mapping from feature space to a privacy-preserving latent space. The main objective function is to optimize of the transformation space to maximize the mutual information between new feature and downstream task while minimizing the mutual information between the private task and the new feature space.

### Strengths
The problem of optimizing over the transformation space is interesting and the objective function seems reasonable.

### Weaknesses
The presentation and the problem formulation are not clear to me.
For example, the motivation of the problem is deferred to Appendix E, while I believe this is important to explain why targeting this problem.
Also, I read Appendix E, and couldn't clearly understand the challenging problem. It seems in Figure 7 that downstream task performance w/o sensitive features is close to the now w sensitive features. 

The problem statement is not clear as well. For example, what is the original feature F? is it real space of d dimensions? also what are A_pr and A_pe models? how they are different? Also could you please correlate this problem statement to a real scenario?

### Questions
see above

### Soundness
3

### Presentation
2

### Contribution
2
