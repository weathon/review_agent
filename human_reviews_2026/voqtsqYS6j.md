# Don't Shift the Trigger: Robust Gradient Ascent for Backdoor Unlearning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Backdoor attacks pose a significant threat to machine learning models, allowing adversaries to implant hidden triggers that alter model behavior when activated. Although gradient ascent (GA)-based unlearning has been proposed as an efficient backdoor removal approach, we identify a critical yet overlooked issue: vanilla GA does not eliminate the trigger but shifts its impact to different classes, a phenomenon we call trigger shifting. To address this, we propose Robust Gradient Ascent (RGA), which introduces a dynamic penalty mechanism to regulate GA's strength and prevent excessive unlearning. Our experiments show that RGA effectively removes backdoors while preserving model utility, offering a more reliable defense against backdoor attacks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies a trigger shifting problem in the gradient ascent (GA) based backdoor unlearning methods in the text classification domain.
Then, it proposes a GA-based unlearning method (RGA) with a decaying loss and L2 regularization to prevent trigger shifting. The paper shows the effectiveness by using three backdoor attacks with three datasets on three models.

### Strengths
- The paper shows a side-effect of unlearning that is previously overlooked.
- The paper presents a clear threat model and the problem.
- The paper uses CUBE (backdoor detection method) by assuming unknown poison samples to reflect the real-world scenarios before unlearning the backdoor by the proposed method.
- The paper is well written and easy to follow.

### Weaknesses
- The attacks in the experiments (BadNets, AddSent, HiddenKiller) are not recent. The papers do not consider the recent attacks, such as,

[1] https://aclanthology.org/2024.findings-acl.468/

[2] https://arxiv.org/abs/2412.18975

- The models used in the experiments are also not recent. The effectiveness of RGA in recent models is not known.

- In Section 4.2, the multi-class classification case is referred to Appendix A.2 theoretically. The paper does not show empirical results except AG (4 classes).

- If the trigger overlaps with the frequent concepts, there may be some collateral damage. The effectiveness of RGA on multi-class classification is not clear, and whether there is collateral damage on multi-class classification.

Minor:
- In Section 6.1 (Unlearning Baselines), the items do not follow a parallel structure. The (1) and (2) are noun phrases, and the third one (3) is a sentence.

- The numbers in Tables 2, 7, and 8 are too small.

### Questions
- In Table 6, DGA is better in some cases, especially in the HiddenKiller case, where the gap is significant. Is it related to the size of the dataset? Why is regularization bad in this case?

- Have you tried using actual poisoned examples rather than the ones detected by CUBE? If so, what is the performance gap?

- Does trigger shifting exist in other modalities, such as image?

- Can RGA be applied to generative tasks in LLMs?

### Soundness
3

### Presentation
4

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
This paper addresses the problem of removing backdoor effects from poisoned language models through a new method called Robust Gradient Ascent (RGA). The authors first identify and formalize the trigger-shifting problem (a situation in which the backdoor is not eliminated but merely moved into another class) that arises when applying unlearning approaches to remove the influence of backdoor attacks. To mitigate this, the proposed RGA introduces an adaptive penalty term that dynamically modulates the unlearning process to prevent divergence and preserve model utility. Experimental evaluations are conducted on three text classification datasets, three model architectures, and under three distinct backdoor attacks. The results demonstrate that RGA stabilizes the unlearning process and prevents trigger-shifting.

### Strengths
The paper is well written and organized. The motivation is clearly stated, the mathematical formulation is elegant, and the proofs are concise and well integrated into the appendix.

The methodology is conceptually simple yet effective. The adaptive penalty strategy in RGA is a neat and intuitive idea that provides theoretical grounding for stability and convergence while maintaining interpretability.

The experimental setup is comprehensive and convincing. The authors evaluate across multiple datasets, architectures, and attack types, providing a strong empirical validation. 

Each research question is well defined, and the experimental section is systematically designed to address it.

### Weaknesses
**Multiclass setup.** The experimental evaluation is limited to two-class or, at most, four-class classification problems. The theoretical framework for addressing trigger shifting is well motivated in the binary case, where the gradient ascent direction can be clearly interpreted as moving away from one class and toward another. However, it remains unclear whether this analysis holds when extending to a larger number of classes, where the gradient effects may be distributed across multiple class directions, potentially reducing the impact or stability of the unlearning process. To solve this limitation, the authors could include an additional experiment on a synthetic dataset with an increasing number of classes, or on a more complex multi-class setting, to examine whether the dynamics of RGA generalize beyond the binary scenario.

**Assumption on access to poisoned samples**. A minor recommendation is related to the assumption that the model owner has access to the poisoned data samples ( D_p ). This assumption is explicitly stated early in the text but not reinforced in Section 5, where the methodology is presented in detail. 

**Minor issues**:
- The abstract Figure 1 is visually appealing but not very clear at first glance. The figure could benefit from a simplified layout and clearer labels to better convey the main conceptual flow.

- The notation ( f_\theta(y|x) ) could be simplified to ( f_\theta(x) ) throughout the paper. Since the method does not explicitly model conditional distributions in the optimization, enforcing the conditional notation adds unnecessary complexity and may confuse the reader.

### Questions
How does RGA behave when the detected poisoned dataset ( D_p ) contains false positives or false negatives? Does the adaptive regularization remain stable or does it diverge under misidentified samples?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Robust Gradient Ascent (RGA), a novel framework that enhances the stability and reliability of GA-based backdoor unlearning. Specifically, this paper shows that GA does not necessarily eliminate the backdoor effect but can instead redirect it to a new backdoor effect. To address this, this paper proposes RGA, which introduces a dynamic penalty mechanism that adaptively regulates the strength of GA during backdoor removal. Extensive experiments demonstrate that RGA effectively eliminates backdoors without trigger shifting, while preserving model utility, and offers a more reliable GA-based defense against backdoor attacks.

### Strengths
1.This paper makes a significant contribution by presenting the discovery of trigger shifting. Based on this, this paper proposes RGA, with its innovative dynamic penalty mechanism based on KL divergence, as a principled and effective solution.

2.This paper conducted comprehensive experiments across diverse datasets, models, and attack methods. The introduction of the PACC and ΔPACC metrics is a novel and essential contribution.

3.The paper is well written and clearly structured.

### Weaknesses
1.This paper does not discuss or discuss other potential functions (e.g., linear decay, step functions, or other divergence measures) to prove that the design of this dynamic penalty mechanism is optimal.

2.This paper does not discuss the time comparison with other baseline methods, except for retraining.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
2

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
The authors identify and systematically demonstrate the problem of trigger shifting in traditional gradient ascent, and propose robust gradient ascent algorithm to alleviate the phenomenon.

### Strengths
(1) The observation of trigger shifting is very novel

(2)  The proposed unlearning method that avoids trigger shifting is elegant

(3) The proposed evaluation metric for this observation is well-defined

(4) The motivation is well presented (especially in Figure 1)

### Weaknesses
(1) The observation of trigger shifting is valuable; however, your theoretical explanations do not seem comprehensive. For example, intuitively, the trigger shifting issue is less severe as the number of parameters of models increase, but this is not reflected in your formula.

(2) Following (1), robust gradient ascent is not necessarily the best approach to resolve the issue.

(3) More advanced/SOTA models, such as the Qwen3 families should be evaluated to justify whether your observation holds with models exposed to more training data, or have more parameters.

Minor issue: Table 2 is really hard to read

### Questions
See weakness.

### Soundness
2

### Presentation
4

### Contribution
4
