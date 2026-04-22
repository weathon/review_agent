# Token-Importance Guided Direct Preference Optimization

- Avg Score: 6.50
- Decision: Accept (Oral)
- Scores: 8, 6, 8, 4

## Abstract
Aligning Large Language Models (LLMs) with human preferences is crucial for safe and effective AI interactions. While popular methods like Direct Preference Optimization (DPO) have simplified alignment, they remain sensitive to data noise and overlook the differential importance of individual tokens. Existing token-level approaches often rely on probability prediction or simplistic weighting schemes to obtain token importance, which still cannot fully address these issues. To solve this problem, we propose the Token-Importance Guided Direct Preference Optimization (TI-DPO), a framework that achieves fine-grained semantic control through two synergistic innovations. 
First, we propose a novel hybrid weighting mechanism that combines gradient attribution with a Gaussian prior, ensuring both the accuracy and robustness of token importance scores. Second, we employ a triplet loss to provide structured guidance for the optimization, explicitly guiding model outputs to approach preferred responses and diverge from non-preferred ones. Experimental results show that TI-DPO achieves higher accuracy and stronger generative diversity, providing more stable and computationally efficient solutions compared with DPO and other RLHF methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses a technique for token-level alignment in preference signal processing within Direct Preference Optimization (DPO). Unlike typical DPO methods that focus on sequence-level alignment, this research explores token-level alignment. Existing approaches, such as TIS-DPO, rely on probability estimation from contrastive language models, while other methods use repeated sampling.


In this study, the authors introduce gradient-based signals to assess token importance in relation to positive and negative examples. They also incorporate a triplet loss term to enhance training stability, providing detailed mathematical derivations for the loss function, gradients, and a tighter loss bound. Additionally, they discuss experimental results across a comprehensive evaluation suite.


To determine token importance weights, the authors calculate the maximum logit for the final token in both positive and negative sequences and then compute gradients of the preceding tokens' embeddings that maximize this final token max logit. This process helps identify the importance of tokens towards maximizing confidence in the final token. The authors develop a weighting scheme for previous tokens, integrating it with a Gaussian prior that softly biases towards middle tokens. This refined weighted objective is used for training, with the expectation of learning token-level importance.



Furthermore, the authors incorporate triplet loss by generating an anchor response and adding a triplet loss term. This term guides the anchor closer to positive examples and away from negative ones, contributing to the combined training objective.


The paper presents a series of experiments and ablation studies, demonstrating progressive improvements with each method. The results indicate strong performance on various evaluation tasks.

### Strengths
- The paper effectively introduces the concept of using gradient-based signals to determine token importance, enabling a more detailed understanding of token-level significance.
- The authors conduct thorough ablation studies, demonstrating that each component of the proposed method enhances the final evaluation metrics.
- The presentation of choices and the algorithm is exceptionally clear.
- There is a robust mathematical analysis and derivation concerning the mechanics of the new objective, including stronger loss bounds.
- The associated code repository clearly outlines the training data and the methodology mechanics.
- The paper includes additional information in the Appendix, such as token weight distribution across various evaluation datasets and mathematical derivations.

### Weaknesses
- Providing additional unbiased examples and analyses could illuminate the nature of token weights identified across various datasets. The paper does not clearly convey a qualitative understanding of the token weights the model is learning, except for one example.

### Questions
- The Gaussian prior seems somewhat counterintuitive, particularly for question-answer prompts where key tokens may not be centrally located. How do you interpret the effectiveness of this prior?
- The approach to determining token weights relies primarily on the final token logits, which may not always provide the most significant signal. For instance, the final token could be akin to a closing word, potentially lacking influence on the overall quality of the response. How do you address this observation, and do you have any studies that explored this phenomenon?
- Is there potential for information loss when calculating weights by taking the maximum of the final logits?
- What is the additional computational cost associated with this method, considering the extra gradients and anchor responses involved?
- Could you provide more illustrative examples to clarify the nature of token weights and their impact?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TI-DPO, a new framework for aligning large language models (LLMs) with human preferences. This framework performs human preference alignment for LLM at a token-level granularity rather than the conventional sequence-level used in DPO and RLHF. The authors observe that existing sequence-level preference optimization methods ignore the varying importance of individual tokens, resulting in unstable training and coarse-grained supervision. To tackle this problem, they further identify two key challenges. First, the model needs to accurately identify the key tokens that influence human preference. Second, the optimization objective should capture continuous and fine-grained preference differences. To address the first issue, they propose the Hybrid Weighting Mechanism to combine gradient attribution with a Gaussian prior to compute token-level importance weights. To tackle the second issue, the authors utilize the Triplet Loss to guide the model’s outputs toward preferred responses and away from non-preferred ones, thereby enabling continuous preference learning in semantic space.

### Strengths
1.	Originality: The paper introduces a novel perspective by extending preference optimization from the traditional sequence-level to a token-level framework, allowing fine-grained semantic control. The proposed Hybrid Weighting Mechanism utilizes gradient attribution to determine the contribution of each token to the model’s output, with a Gaussian prior incorporated as a regularization mechanism to enhance training stability. The use of the Triplet Loss for fine-grained preference alignment and continuous gradient of preference learning further distinguishes this work from prior DPO-based approaches.

2.	Quality: The theoretical analysis demonstrates that TI-DPO achieves a tighter loss bound than DPO. The experimental evaluation is comprehensive. The results demonstrate superior accuracy compared to baselines on the HumanEval, TruthfulQA, and IFEval benchmarks.

3.	Clarity: The paper is generally well written and the motivation is clearly explained.

4.	Significance: TI-DPO provides a token-level human preference alignment framework that enhances the performance of LLMs and offers more stable training.

### Weaknesses
1.	Limited empirical validation of claimed advantages: Although the paper claims improvements in robustness (e.g., line 20, line 58, line 125), generative diversity (e.g., line 24), and computational efficiency (e.g., lines 24-25), these aspects are not discussed or quantitatively evaluated. There is a lack of experiments that measure robustness under noisy or perturbed preference data. The paper also does not discuss or evaluate the claimed diversity gains. For example, evaluate generative diversity using metrics such as Self-BLEU, Distinct-n or Diversity Entropy. Reporting quantitative comparisons of GPU hours or providing computational complexity analysis would also strengthen the claim of efficiency.

2.	Lack of details in the Gaussian prior design: The paper describes the use of a Gaussian-shaped prior in the Hybrid Weighting Mechanism to assign higher baseline importance to middle tokens. The paper only mentioned “we define a Gaussian-shaped prior distribution $\mathcal{P}_{\text{prior}}$ centered on the sequence” (lines 228-234), but it does not specify its exact settings. This lack of detail hinders not only the reproducibility but also the evaluation of how much the Gaussian prior actually contributes to the method’s effectiveness.

3.	Ambiguity in anchor generation: The paper states that the anchor response is “dynamically created by the model using the preferred response as a contextual starting point” (lines 262-264), without further details of how this response is generated. The lack of the anchor generation procedure makes it difficult to evaluate the triplet loss setup and determine whether the anchor truly represents an intermediate state between positive and negative samples.

4.	Incomplete comparison and missing analysis of performance gaps: While the paper claims to address two major challenges in token-level preference optimization (lines 52-55), it does not include sufficient comparisons with other token-level alignment methods, such as TIS-DPO [1], cDPO [2] or Logic-RL [3], which are mentioned in both the Introduction and Related Works sections. Moreover, TI-DPO underperforms baselines, such as GRPO or TPO, on key benchmarks, including MMLU, GSM8K, and GPQA (as shown in Tables 1 and 7–9). Yet, the paper provides no analysis or discussion to explain these results.

5.	The ablation analysis is insufficient: The current study only examines the removal of the triplet loss and the use of uniform or random weights, which is not enough to fully demonstrate the contributions of the proposed components. Specifically, the paper lacks ablations on (a) the relative effects of gradient attribution versus the Gaussian prior in the Hybrid Weighting Mechanism, and (b) the sensitivity of model performance to the weighting coefficient λ (eq. 7) and the triplet margin α (eq. 12). In addition, there are many predefined hyperparameters (e.g. α, β, γ, λ) utilized in this paper. The paper does not provide detailed hyperparameter values, which may hinder reproducibility. Moreover, in Algorithm 1 (line 346), the hyperparameter γ is not listed among the inputs, making it unclear whether it is user-defined or internally fixed.

6.	Weak connection between theory and experiments: While Lemma 1 and Theorem 2 are mathematically sound, the paper does not clearly explain how the “tighter loss bound” are related to improvements in accuracy. A short discussion linking the theoretical results to empirical results would enhance the reading flow. Moreover, Sections 4.2 and 4.4 are overly dense, with long mathematical derivations but limited commentary. Adding clearer links between theory and experiment results, together with concise textual explanations, would significantly improve both clarity and readability.

7.	Lack of broader discussion and limitations: The paper does not sufficiently discuss token-level preference optimization in different scenarios, such as scalability to very long sequences or potential bias amplification when importance weights overemphasize frequent tokens. For example, when applied to tasks with very long responses (e.g., multi-turn dialogue or chain-of-thought reasoning), computing gradient-based token importance may become computationally expensive. In addition, if the preference data contains biased correlations such as “she” frequently appearing with “nurse” and “he” with “surgeon”, the weighting mechanism might assign higher importance to these patterns, thus reinforcing gender stereotypes rather than mitigating them. It would be valuable if the authors further analyzed whether their framework can address such issues or provided a brief discussion of potential solutions.

[1] Liu, Aiwei, et al. "TIS-DPO: Token-level Importance Sampling for Direct Preference Optimization With Estimated Weights." The Thirteenth International Conference on Learning Representations.

[2] Lin, Zicheng, et al. "Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM’s Reasoning Capability." Forty-second International Conference on Machine Learning.

[3] Xie, Tian, et al. "Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning." arXiv preprint arXiv:2502.14768 (2025).

### Questions
1.	Quantitative evidence for claimed advantages: Could you provide experiments or analysis to substantiate the claims of robustness, generative diversity, and computational efficiency? 

2.	Comparison with other token-level baselines: The paper mentions that TI-DPO provides fine-grained alignment compared with several token-level alignment methods (e.g., TIS-DPO [1], cDPO [2], and Logic-RL [3]) but does not include them in the experimental comparison. Could the authors provide experiments or analysis to discuss their methods with other token-level methods?

3.	Design of the Gaussian prior: Could you clarify the exact settings of the Gaussian-shaped prior used in the Hybrid Weighting Mechanism (lines 228–234)? How sensitive is TI-DPO’s performance to the design settings of this prior?

4.	Anchor generation in Triplet Loss: The generation process of the anchor $y$ is not specified. Could you explain how the anchor response is generated and how to ensure an intermediate position between preferred and non-preferred responses?

5.	Analysis of performance gaps: TI-DPO underperforms GRPO or TPO on MMLU, GSM8K, and GPQA (Tables 1 and 7–9). Could you discuss why this occurs? 

6.	Ablation studies and reproducibility: Could you provide further ablations to discuss (a) the individual effects of gradient attribution and Gaussian prior, and (b) the sensitivity to key hyperparameters such as λ (Eq. 7) and α (Eq. 12)? Additionally, is it possible to clarify the values of predefined hyperparameters (e.g., α, β, γ, λ) and how these choices are selected?

7.	Scalability and potential bias amplification: Does the token importance computation in this work introduce computational overhead? When applying TI-DPO to tasks with long responses (e.g., multi-turn dialogue or chain-of-thought reasoning), how does the computational cost scale with model size and sequence length? Furthermore, has the model been evaluated for potential bias amplification, such as reinforcing biased correlations like “she-nurse” or “he-surgeon” in preference data?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes an enhancement to Direct Preference Optimization (DPO) through token-level importance alignment. The authors incorporate token-level weighting based on gradient attribution, coupled with a Gaussian prior to promote robustness. The method extends DPO with a modified token-level objective and an additional triplet loss for fine-grained alignment. Theoretically, the paper shows that the standard DPO objective upper bounds the proposed per-token loss. Empirical results demonstrate that this approach strongly outperforms vanilla DPO and achieves better performance compared to other RLHF-based methods across multiple benchmarks.

### Strengths
1. **Well written and clearly presented.** The paper is well structured and makes effective use of figures and visualizations to support understanding.  
2. **Novel contribution.** Introduces a new approach that weights token importance and incorporates a triplet loss for more fine-grained alignment.  
3. **Sound theoretical analysis.** Provides theoretical guarantees showing how the proposed formulation relates to and improves upon vanilla DPO.  
4. **Strong empirical results.** Experimental results demonstrate notable performance improvements over standard DPO and related baselines.

### Weaknesses
1. **Limited generalizability.** The method is presented as DPO specific, which limits its applicability to newer and potentially more effective alignment approaches such as GRPO and DRPO.  
2. **Incomplete methodological details.** Some aspects of the triplet loss implementation are insufficiently described, making it difficult to fully understand how this component contributes to the overall improvement of the method.

### Questions
While I really enjoyed this paper, some remaining questions and suggestions that would benefit the paper:
1. Looking at Table 2, Tl-DPO shows a substantial improvement compared to vanilla DPO. Can this method be extended to other alignment frameworks such as GRPO or DRPO? Including such extensions and experiments would greatly strengthen the paper.  
2. Have you conducted ablations on the impact of the Gaussian prior compared to using no prior or alternative priors (e.g., uniform, softmax probabilities)?  
3. Regarding the relationship with vanilla DPO, is there any equivalence between the vanilla and token-level DPO formulations? Specifically, for Equation 8, are there particular values of \( w \) that would make it equivalent to vanilla DPO?  
4. Line 266 mentions that for the triplet loss, each response is first mapped to an embedding. How is this mapping performed?  
5. Line 298 states that “Tl-DPO will be **significantly** lower than the original DPO loss.” However, Theorem 2 only proves that \( L_{\text{DPO}} \) upper bounds \( L_{\text{Tl-DPO}} \). Without additional arguments, this claim appears overstated.  
6. Line 470 claims that Tl-DPO “effectively bridges the alignment gap between LLMs and human value systems.” This statement is too strong and should be moderated.  
7. While Theorem 2 provides a useful upper bound for Tl-DPO, its practical significance without further discussion is unclear, since a lower loss does not necessarily imply better downstream performance. Are the Tl-DPO and DPO losses comparable under certain conditions or specific values of \( w \) (see Question 3)?  

**Minor Comments That Do Not Affect Rating**
1. For the triplet loss, \( y \) is being optimized. What about the embeddings for \( y_w \) and \( y_l \)? Is the network that produces these embeddings trained as well?  
2. Have you tested alternative contrastive losses beyond the triplet loss? For example, using a SimCLR or SogCLR-style approach, one could form positive pairs between \( y \) and its corresponding \( y_w \), and negative pairs between \( y \) and all other \( y_l \). This might provide a stronger signal from negative responses through hard-negative weighting.  
3. Equation 3 is described as showing token-level DPO, but the equation currently corresponds to vanilla DPO. Additionally, \( y^{<} \) is defined but never used.  
4. Line 190 mentions that “\( m \) is the count of tokens,” but this variable is not used in the formulation.  
5. Line 9 in Algorithm 1 references \( \Delta r_{\text{triplet}} \), which has not been defined earlier. Equation 10 defines only \( \Delta r_{\text{token}} \). The definition of \( \Delta r_{\text{triplet}} \) should likely appear near Equation 12.  
6. Section 5.4 should explicitly refer to Figure 2 for clarity and consistency.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose the Token-Importance Guided Direct Preference Optimization (TI-DPO) framework, a novel method to align large language models with human preferences. TI-DPO employs a hybrid weighting mechanism that assigns importance scores to tokens and a triplet loss that provides a more structured optimization signal. Theoretically, the authors show that TI-DPO achieves a tighter loss bound than DPO. Extensive experiment results indicate that TI-DPO surpasses existing RLHF methods across benchmarks, such as HumanEval, TruthfulQA, and IFEval.

### Strengths
- The paper introduces a novel Token-Importance Guided Direct Preference Optimization (TI-DPO) framework that integrates gradient-based token attribution with a triplet loss objective, aiming to achieve finer-grained preference alignment.
- Theoretical analysis provides a formal derivation suggesting that TI-DPO attains a tighter loss bound than standard DPO, offering a potentially more stable optimization objective.

### Weaknesses
- The motivation for employing a Gaussian prior in the hybrid weighting mechanism is insufficiently justified. The assumption that salient tokens cluster near the center of a sequence lacks both empirical evidence and theoretical grounding. Alternative priors or adaptive distributions are not discussed.
- Although the authors claim that TI-DPO achieves a tighter loss bound than DPO, the paper does not provide quantitative evidence to demonstrate the practical significance of this theoretical improvement. Moreover, the potential implications for additional benefits, such as convergence speed or generalization performance, remain unclear.
- The experimental evaluation omits comparisons with recent token-level DPO variants, such as TIS-DPO [1].

[1] Liu, Aiwei, et al. "Tis-dpo: Token-level importance sampling for direct preference optimization with estimated weights." arXiv preprint arXiv:2410.04350 (2024).

### Questions
- Could the authors clarify the procedure for generating the anchor response $y$ used in the triplet loss?
- What is the theoretical or empirical rationale for selecting a Gaussian prior for token-importance weighting over alternative distributions?
- How sensitive is the model performance to the weight-mixing hyperparameter $\lambda$?

### Soundness
2

### Presentation
3

### Contribution
2
