# Rotation Control Unlearning: Quantifying and Controlling Continuous Unlearning for LLM with The Cognitive Rotation Space

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 2, 2

## Abstract
As Large Language Models (LLMs) become increasingly prevalent, their security vulnerabilities have already drawn attention. Machine unlearning is introduced to seek to mitigate these risks by removing the influence of undesirable data.  However, existing methods not only rely on the retained dataset to preserve model utility, but also suffer from cumulative catastrophic utility loss under continuous unlearning requests. To solve this dilemma, we propose a novel method, called Rotation Control Unlearning (RCU), which leverages the rotational salience weight of RCU to quantify and control the unlearning degree in the continuous unlearning process. The skew symmetric loss is designed to construct the existence of the cognitive rotation space, where the changes of rotational angle can simulate the continuous unlearning process. Furthermore, we design an orthogonal rotation axes regularization to enforce mutually perpendicular rotation directions for continuous unlearning requests, effectively minimizing interference and addressing cumulative catastrophic utility loss. Experiments on multiple datasets confirm that our method without retained dataset achieves SOTA performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes Rotation Control Unlearning (RCU), a novel approach for continual machine unlearning in Large Language Models that addresses two major challenges: Cumulative utility degradation across multiple unlearning requests and a Lack of precise control and quantification over the unlearning process. RCU reinterprets LoRA-based updates as rotations in a cognitive representation space. The method introduces:  1) a skew-symmetric loss to model LoRA updates as rotational transformations, 2) an orthogonal rotation axis loss to ensure perpendicular update directions across sequential unlearning requests, thereby minimizing interference and catastrophic forgetting, and 3) a distributional shift compensator that produces rotational salience weights, enabling precise auxiliary quantification of unlearning effects. RCU is tested on ScienceQA and TOFU benchmarks, achieving effective continual unlearning without relying on retained datasets.

### Strengths
1. By modeling unlearning as rotations in a latent space, RCU introduces a mathematically grounded and interpretable framework for tracking and controlling knowledge removal.

2. Experiments on TOFU and ScienceQA demonstrate RCU’s unlearning efficacy and utility preservation.

3. The introduction of rotational angles and salience weights provides a precise, tunable metric for unlearning strength.

### Weaknesses
1. The training procedure involves a combination of losses for LoRA and OOD detection, making the optimization process relatively complex. However, the paper does not provide sufficient details on the choice of loss weights (Lambda values). A sensitivity analysis should be conducted to demonstrate the model's robustness to different hyperparameter settings and justify the selected values.

2. The paper would benefit from improved writing quality. The rationale for using rotation angle is not clear, and there are several typos and inconsistencies, for example, in Table 4, the word "physics" appears inappropriately or misspelled.

### Questions
Could the authors provide justification for the choice of the lambda hyperparameters used in the loss functions? Given that the training objective combines multiple loss terms (e.g., for LoRA tuning and OOD detection), the values of these weights significantly affect model behavior. A robustness analysis—such as an ablation or sensitivity study—should be conducted to evaluate how performance varies with different lambda settings across tasks or datasets.

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
4

### Summary
The paper proposes Rotation Control Unlearning (RCU), a novel method for Large Language Model (LLM) unlearning that precisely quantifies and controls the unlearning degree during continuous unlearning requests.
They propose the RCU method, which reformulates the LoRA update process as rotations within a "cognitive rotation space". This approach allows RCU to effectively unlearn information continuously without needing a retained dataset.
The method introduces 3 new key components:
1. Skew Symmetric Loss: A loss function designed to construct the cognitive rotation space. This constraint ensures the LoRA parameter updates behave like rotations.
2. Rotational Salience Weight: A metric derived from an OOD detector that is used to precisely quantify and control the degree of unlearning (i.e., the rotation angle) for any given request.
3. Orthogonal Rotation Axes Regularization: A loss function that enforces mutually perpendicular rotation directions for consecutive unlearning requests. This minimizes interference between different unlearning tasks and directly addresses the problem of cumulative catastrophic utility loss.

### Strengths
**1. Fine-grained Control over Unlearning Intensity:**

The paper introduces a novel perspective by modeling the unlearning strength as a controllable variable through rotational scaling. This design allows the model to precisely adjust the degree of forgetting at a fine-grained level. Such a mechanism provides more flexibility and interpretability for selective and controlled unlearning.

**2. Theoretical Novelty through the Cognitive Rotation Space:**

The proposed Cognitive Rotation Space offers a strong theoretical contribution. By formulating unlearning as a spectral-space rotation governed by skew-symmetric transformations, the authors connect geometric orthogonality with cognitive forgetting behavior. This framework is both conceptually elegant and mathematically grounded, bringing a fresh theoretical lens to the study of machine unlearning.

**3. Strong Empirical Performance and Robustness:**

The experimental results are impressive; the method demonstrates consistently strong unlearning performance across two benchmarks while significantly alleviating cumulative catastrophic utility loss in continual unlearning settings. The comprehensive evaluation convincingly shows that the proposed approach achieves an excellent balance between forgetting precision and retention stability.

### Weaknesses
**1. Limited Orthogonality Scope:**

The current design of the orthogonal rotation loss \mathcal{L}_o appears to only enforce pairwise orthogonality between the current and the immediately preceding rotation axes. However, it does not ensure global orthogonality across all historical rotations. As the number of unlearning requests T grows, earlier forgetting directions may be reprojected onto new ones, potentially leading to knowledge leakage or cumulative utility degradation. Have the authors considered enforcing global orthogonality, and is it necessary for maintaining long-term stability in continual unlearning?

**2. Granularity of the Rotational Salience Weight $\beta$:**

In the RCU algorithm, is the $\beta$ obtained through the OOD detection and Distributional Shift Compensator shared among all forget data within one unlearning process, or is a separate $\beta$ computed individually for each data sample?

**3.Computational Overhead and Efficiency Concerns:**

Since the proposed approach relies on OOD detection to obtain a sample-dependent $\beta$ value, it seems that an additional forward pass through the OOD module may be required for each input during inference. Will this introduce noticeable latency? Moreover, during training, the optimization process involves multiple components and loss terms, which might increase computational complexity and training time. A more detailed discussion or empirical analysis of the computational overhead would strengthen the paper.

**4. Results Not Consistent with Prior Baselines.**

After reviewing the O³ paper, I notice that the two papers report identical numbers for the other baselines, but this paper’s results are worse than those reported in O³ specifically for the O³ method itself. Given that the remaining unlearning methods appear unchanged across both papers, could the authors clarify the source of this discrepancy?

### Questions
Questions are included in the weakness section.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the problem of machine unlearning in LLMs. The authors noted that existing methods rely on retained dataset to preserve utility, and become impractical for larger and more practical settings, and could lead to catastrophic utility loss in continuous setting.The authors propose rotation control unlearning (RCU) that interprets LoRA updates as rotations within a specific cognitive rotation space. The authors claim that rotation angle corresponds to the degree of forgetting. The authors also introduce orthogonal rotation axes loss to allow multiple unlearning steps. The authors conduct experiments on a QA and TOFU benchmark on a single Llama non-instruct model.

### Strengths
The studied problem is important and the proposed method shows its effectiveness to some extent.

### Weaknesses
Overall, there are several weaknesses the authors should address. Most importantly, why can we quantify the degree of unlearning using rotation angle? This paper does not provide conceptual, empirical, or mathematical justification for why unlearning itself should correspond to a rotation angle in rotation space. Also, the writing needs significant improvement. Many times throughout the paper, the authors use a term extensively without explanations (for example, the cognitive rotation space, what is that?) The authors should explain some of these terms in the preliminary section. The current preliminary section is essentially a ‘related work’ section. The method section also reads as a list of techniques without explaining why they are needed or how they relate to the core problem of unlearning. Overall, the paper does not convincingly argue that RCU is an effective or principled approach to unlearning.

### Questions
I also have concerns regarding experiments. Moving into 2026, this setting feels too limited to demonstrate practical relevance. The authors should try larger models on way more unlearning requests to prove effectiveness of their methods.Also, the evaluation metrics are not well justified and not properly defined. For the TOFU task, why not use the eval metrics provided by the original authors? For example, the p-value statistical test and the average of Rouge, Answer Probability, and Truth Ratio? The authors should provide more contexts about these evaluations and justify their choices. 

While [1] is a recent work, [1] has achieved ideal unlearning by masking out the training signal of TOFU in their corpus. According to [1] results, an ideally unlearned model will achieve ideal forget quality vs utility trade-off. Also, forgetting ROUGE isn’t always better. This translates to the authors choice of SU and DU, where the authors are claiming that lower is always better. I strongly recommend the authors to use TOFU’s eval metrics and compare their results with LMLM from [1].  Also, the authors should compare computational costs, since we do want a simple and efficient algorithm for unlearning. 

Lastly,  [2] has pointed out that unlearning isn’t always robust. More experiments on the robustness of their methods should be presented in the paper.

[1] Zhao, L., Zalouk, S., Belardi, C. K., Lovelace, J., Zhou, J. P., Weinberger, K. Q., ... & Sun, J. J. (2025). Pre-training Large Memory Language Models with Internal and External Knowledge. arXiv preprint arXiv:2505.15962.
[2] Łucki, J., Wei, B., Huang, Y., Henderson, P., Tramèr, F., & Rando, J. (2024). An adversarial perspective on machine unlearning for ai safety. arXiv preprint arXiv:2409.18025.

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
4

### Summary
The paper improves on the continual framework of unlearning named $O^3$ [1], which consists of orthogonal LoRA adapters for different unlearning requests, a OOD detector trained with a novel contrastive entropy loss and utilizes a global-aware scoring mechanism, and a soft inference algorithm to detect unlearned data during inference.

The paper introduces rotation matrices for updating the LoRA adaptors and constrain the rotation angles to be orthogonal for every unlearning request. They update the consequent OOD detector to take this modification into account.


[1] Gao, Chongyang, et al. "On large language model continual unlearning." ICLR 2025

### Strengths
- The paper is well-written for the most part. 
- The empirical results look strong, even though its not completely convincing.

### Weaknesses
- Please improve the figures. They are hard to see without a lot of zooming in. 
- The metrics in Table 1 for $O^3$ do not match with the original paper. In this case, please include multiple runs to demonstrate the improvement properly. This is a major weakness of this paper because of how closely it follows $O^3$ and claims improvement. 
- Please include the $U^2R$ metric from the $O^3$ paper to demonstrate improved utility preservation over  $O^3$ . 
- From Equation 4, 5 : $C = BA$ and $R = exp(C) = I + BA$.  This seems very strange. Is there any evidence why this approximation may hold , can you please provide evidence that $C << I$ from your experiments ?
- The above point makes it very hard to understand whether the method legitimately works due to the rotation aspect or the performance is still an artifact of $O^3$ due to the high similarity of constraints and other frameworks.

### Questions
- Should Eq. 3 be $W = W + BA$ ?
- One of the central claims of the paper is that the idea of updating LoRA parameters using rotation matrices will improve the catastrophic utility degradation. Why is it so ? Why is this better than $O^3$? Is there any intuition ?
- In Eq. 11, why is the use of relative rotation space needed ? Can we directly use $R_t$ and $R_{t-1}$ ? 


Please address the Weaknesses and the above questions.\
I am willing to raise my score based on the clarification.

### Soundness
2

### Presentation
3

### Contribution
2
