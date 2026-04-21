# Rethinking Invariance Regularization in Adversarial Training to Improve Robustness-Accuracy Trade-off

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
Adversarial training often suffers from a robustness-accuracy trade-off, where achieving high robustness comes at the cost of accuracy.
One approach to mitigate this trade-off is leveraging invariance regularization, which encourages model invariance under adversarial perturbations; however, it still leads to accuracy loss.
In this work, we closely analyze the challenges of using invariance regularization in adversarial training and understand how to address them.
Our analysis identifies two key issues: (1) a "gradient conflict" between invariance and classification objectives, leading to suboptimal convergence, and (2) the mixture distribution problem arising from diverged distributions between clean and adversarial inputs.
To address these issues, we propose Asymmetric Representation-regularized Adversarial Training (ARAT), which incorporates asymmetric invariance loss with stop-gradient operation and a predictor to avoid gradient conflict, and a split-BatchNorm (BN) structure to resolve the mixture distribution problem.
Our detailed analysis demonstrates that each component effectively addresses the identified issues, offering novel insights into adversarial defense.
ARAT shows superiority over existing methods across various settings. Finally, we discuss the implications of our findings to knowledge distillation-based defenses, providing a new perspective on their relative successes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work addresses the fundamental robustness-accuracy trade-off in adversarial defenses through a novel invariance regularization approach. They proposed the method, AR-AT, achieves state-of-the-art performance by systematically addressing two key challenges: the gradient conflict between classification and invariance losses, and the mixture distribution problem in adversarial training. They resolved the gradient conflict through a strategic stop-gradient operation, while implementing a split batch normalization structure to handle the mixture distribution challenge.

### Strengths
This work is well-written and easy to understand. The perspective on gradient conflict is particularly interesting, connecting previously disparate threads in adversarial robustness research.

### Weaknesses
First, the paper's central premise about gradient conflicts requires deeper theoretical examination. Similar gradient conflicts arise in various scenarios, such as when optimizing the same loss across different mini-batches, yet these conflicts don't necessarily impact model generalization (or adversarial robustness). The paper doesn't provide sufficient theoretical justification for why resolving gradient conflicts specifically improves adversarial robustness.

Second, there's a critical gap in the analysis between local optimization dynamics and global distributional properties. The gradient conflicts observed at the mini-batch level may not accurately reflect the underlying conflicts in the full input distribution. Without this theoretical bridge, it's unclear whether the proposed solution addresses the fundamental cause of the robustness-accuracy trade-off.

Third (minor), the experimental results may not be the SOTA. Ref https://github.com/wzekai99/DM-Improves-AT

### Questions
Ref Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose AR-AT, which uses asymmetric invariance loss with stop-gradient operation and a predictor to avoid gradient conflict, and a split-BatchNorm (BN) structure to resolve the mixture distribution problem, to improve robustness-accuracy trade-off. The experiments demonstrate the effectiveness of this method.

### Strengths
1. The paper is easy-to-follow

2. The motivation of the proposed method is clear

### Weaknesses
1. In Figure 3, can you explain why $||z-z'||_2$ increases over time even though you add a regularization term? and why the use of stop-grad exacerbates this issue?

2. Line 200-201 & Figure 2: Plotting curves of only minimizing the second term can enhance your claim.

3. Table 17 indicates that solely resolving the mixture distribution problem can already improve the performance to a large extent. However, addressing gradient conflict alone does not contribute much to robustness, especially for (1)vs(2) and (5)vs(6). Thus, you should also report the standard deviations to make your results more convincing.

### Questions
1. Where is Figure 4?

2. What is the computational overhead of your method compared to other baselines?

### Soundness
4

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
3

### Summary
This paper proposed a bag of tricks to address two issues in siamese-based adversarial training, i.e., ``conflict gradients'' and the mixture distribution problem.

### Strengths
1.  The paper is well organized. The target issues are clear and each step of the solution is motivated and well represented.
2.  Experiments clearly show the effectiveness of each technique they proposed.

### Weaknesses
In Table 3, some baseline methods provide error bars, but others do not.
 
Moreover, considering the huge body of literature on adversarial training, I have concerns about whether the presented comparison with baseline methods is sufficient and comprehensive.

### Questions
1. Since the proposed method applies invariance regularization to multiple layers, I wonder if the baseline methods also adopt such multi-layer regularization.  

2. Does your method introduce additional computation costs compared to baseline methods?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel method called Asymmetric Representation-regularized Adversarial Training (AR-AT), aimed at improving the robustness-accuracy trade-off in adversarial training. AR-AT achieves this goal by addressing two key issues in invariance regularization: (1) the “gradient conflict” between invariance loss and classification objectives, and (2) the mixed distribution problem caused by the differences in input distributions between clean and adversarial samples. AR-AT introduces asymmetric invariance regularization, a stop-gradient operation, a predictor, and a split BatchNorm (BN) structure. Experimental results show that AR-AT outperforms existing methods across various settings and provides new insights into knowledge distillation-based defenses.

### Strengths
(1) This paper is well written and easy to follow.

(2) The paper validates the effectiveness of AR-AT through experiments on multiple datasets and model architectures, demonstrating superior performance in the robustness-accuracy trade-off.

(3) The paper not only presents a new method but also provides an in-depth analysis of the “gradient conflict” and mixed distribution problems in invariance regularization, offering new theoretical insights for the field of adversarial defense.

### Weaknesses
(1) The novelty of this paper maybe limited. The proposed stop-gradient operation and predictor are often adopted in self-supervised learning 

(2) More evidence and experiments on mixed distribution problem should be claimed. 

(3) There are a few minor errors, such as "clean representation z^' " in line 191.

### Questions
(1) The novelty of this paper may be somewhat limited, as the proposed stop-gradient operation and predictor are commonly utilized in self-supervised learning.

(2) Additional evidence and experiments addressing the mixed distribution problem should be provided.

(3) There are a few minor errors, such as the notation "clean representation Z^' " in line 191, which should be corrected.

(4) The ImageNet dataset should be incorporated into the experiments to enhance the robustness of the results.

### Soundness
3

### Presentation
3

### Contribution
2
