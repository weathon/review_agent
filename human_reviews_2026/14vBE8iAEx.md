# Beyond Reward Hacking: Causal Rewards for Large Language Model Alignment

- Decision: Reject
- Scores: 8, 2, 6, 2

## Abstract
Recent advances in large language models (LLMs) have demonstrated significant progress in performing complex tasks. While Reinforcement Learning from Human Feedback (RLHF) has been effective in aligning LLMs with human preferences, it is susceptible to spurious correlations in reward modeling. Consequently, it often introduces biases—such as length bias, sycophancy, conceptual bias, and discrimination—that hinder the model’s ability to capture true causal relationships. To address this, we propose a novel causal reward modeling approach that integrates causality to mitigate these spurious correlations. Our method enforces counterfactual invariance, ensuring reward predictions remain consistent when irrelevant variables are altered. Through experiments on both synthetic and real-world datasets, we show that our approach mitigates various types of spurious correlations effectively, resulting in more reliable and fair alignment of LLMs with human preferences. As a drop-in enhancement to the existing RLHF workflow, our causal reward modeling provides a practical way to improve the trustworthiness and fairness of LLM finetuning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper addresses a fundamental problem in Reinforcement Learning from Human Feedback (RLHF) — reward hacking due to spurious correlations in reward modeling. Traditional RLHF reward models often conflate superficial correlations (e.g., response length, agreement, demographic cues) with genuine human preferences, leading to biases such as length bias, sycophancy, concept bias, and discrimination bias (see Section 1). To overcome this, the authors propose a Causal Reward Model (CRM) that incorporates counterfactual invariance to ensure that reward predictions are independent of irrelevant variables (Section 3.2–4.1). The method employs Maximum Mean Discrepancy (MMD) regularization to enforce statistical independence between latent representations and spurious factors (Eq. (1), Section 4.1). Experiments across four settings — sycophantic, length, concept, and discrimination bias — demonstrate that CRM significantly mitigates spurious correlations and improves fairness and robustness without sacrificing model utility (Sections 5.1–5.4, Tables 1–3, Fig. 3). The approach integrates seamlessly with existing RLHF pipelines, offering a practical improvement to current alignment workflows.

### Strengths
Originality: The paper introduces a novel causal regularization framework for reward modeling within RLHF, which is an underexplored direction. While prior works address specific biases (e.g., length bias via PoE or WARM), CRM provides a unified causal formulation that generalizes across multiple types of spurious correlations (Section 2.2). The concept of counterfactual invariance adapted from causal inference to LLM alignment is new.

Quality: The theoretical foundation is clearly established with causal diagrams (Figure 1, Section 3.2) and well-defined objectives (Eq. (1)–(2), Section 4.1). The experimental design is comprehensive and evaluates diverse bias types. Results (e.g., Table 2 and 3) consistently demonstrate the superiority of both conditional and unconditional CRMs over baseline reward models. The empirical validation across multiple bias domains substantiates the method’s robustness and generalizability.

Clarity: The paper is well-structured and written clearly. Figures and tables (e.g., Fig. 2 and 3) effectively visualize comparative performance trends.

Significance:
Reward hacking remains a major bottleneck for safe deployment of RLHF-trained LLMs. Hence, this paper’s contribution is relevant.

### Weaknesses
Hyperparameter Sensitivity: The choice of MMD coefficients (λ) significantly affects results (Fig. 3), but the paper does not deeply explore tuning stability or sensitivity. This could impact deployment robustness in practice.

### Questions
Causal Assumptions: How sensitive is CRM to incorrect causal graph specifications?

Scalability: How does the computational cost of MMD regularization scale with large datasets and high-dimensional embeddings?

Hyperparameter Selection: How was λ chosen in practice? Did you observe stable optima across tasks or require dataset-specific tuning?

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
3

### Summary
The paper introduces a method called CRM to address spurious correlations and reward hacking in RLHF. The central idea is to remove the confounding effect of response length so that the reward model more accurately reflects the true quality of the response. The authors leverage MMD to design a regularizer for reward model training. Experiments are conducted to evaluate the performance of the proposed method.

### Strengths
The paper introduces a new perspective by leveraging the method in counterfactual invariance. The proposed CRM serves as a regularizer that can be integrated into standard reward model frameworks.

### Weaknesses
- I am concerned about the novelty of this paper, as it largely builds on the technique proposed in [1], which already demonstrates that MMD can serve as a causal regularizer. In comparison, the present work appears to mainly combine and reapply techniques from that paper.

- The paper is not well written, causing unnecessary confusion in its methodology.

- Line 212: “the causal graph reveals that $T^{L, \perp}$ is independent of $Z$,” but Figure 1 shows that $Z$ directly influences $T^{L, \perp}$.

- I believe Equation (1) contains the key insight of the paper, but its description is delayed, and the preceding discussion seems disconnected from it. 

- Is $T^{L, \perp}$ well-defined?

- The description of MMD could be moved to the Preliminaries section to make the methodological contribution clearer. 

- Why is it necessary to bin $Z$? Why not directly measure the dependence between $f(T)$ and $T$ using the Hilbert–Schmidt Independence Criterion (HSIC)? These questions further aggravate my concern about the novelty of the paper, as it closely follows the approach in [1].

- The definition of $p'_m$ is also unclear.

- I am skeptical about the effectiveness of the proposed formulation. From a statistical perspective, MMD only measures dependence; it does not imply causality. Intuitively, applying MMD regularization in (1) cannot ensure the causality claimed in this paper.

- How is $Z$ chosen in practice? What if $Z$ includes a factor that does reflect the true reward of the response? Naively applying Equation (1) in such cases could hinder effective reward learning.

- The experimental comparison is limited. The authors should consider including more recent methods, for example:
  - https://arxiv.org/abs/2403.19159
  - https://arxiv.org/abs/2502.00814

[1] https://arxiv.org/abs/2106.00545

### Questions
See Weaknesses section

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposed a new approach called ‘Causal Reward Model’ to mitigate the problem of spurious correlation and reward hacking that is common in RLHF. Traditional methods use spurious shortcuts to deal with hacking coming from factors like long length. The paper, instead, treats the factors like length and tone as spurious variables $Z$, and learn a reward predictor whose internal representation is independent of $Z$. A maximum mean discrepancy based on the reward predictor is then added to the loss function as the regularisation.

### Strengths
-The paper is based on causal language and counterfactual invariance, which is novel compared to previous methods using supurious shortcuts. The CRM is then added as a normal regulariser which can be embedded into common model frameworks easily.
- Empirical results are also good.

### Weaknesses
- The method can only debiases the explicit spurious factors. If an unknown latent shortcut hack the reward, it won’t fix it. 
- The benchmark is PPO, but what about other methods that address the shortcuts, the author should compare with these methods as well.

### Questions
- In practice, how to decide the spurious variables Z? 
- Does CRM stay robust after the fine-tuning, which may cause the policy’s distribution shift against the preference dataset used to train CRM?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a causality inspired regularizer in Reward Modelling.

### Strengths
The idea seems promising and novel. The idea is well motivated, and the solution is presented with clarity.

### Weaknesses
The experimental section seems to have missed out on providing some details. 

1) For any one of the dataset mention how you are creating the bins.
2) How are you calculating p_m and p_m^'. For what values of Z is this being computed?
3) What are the values that Z can take in any one of the datasets?
4) Describe the conditional CRM in more detail.

### Questions
See above.

### Soundness
3

### Presentation
1

### Contribution
3
