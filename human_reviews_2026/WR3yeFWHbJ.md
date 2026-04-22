# Tight Robustness Certificates and Wasserstein Distributional Attacks for Deep Neural Networks

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 8, 6, 2

## Abstract
Wasserstein distributionally robust optimization (WDRO) provides a framework for adversarial robustness, yet existing methods based on global Lipschitz continuity or strong duality often yield loose upper bounds or require prohibitive computation. In this work, we address these limitations by introducing a primal approach and adopting a notion of exact Lipschitz certificate to tighten this upper bound of WDRO. In addition, we propose a novel Wasserstein distributional attack (WDA) that directly constructs a candidate for the worst-case distribution. Compared to existing point-wise attack and its variants, our WDA offers greater flexibility in the number and location of attack points. In particular, by leveraging the piecewise-affine structure of ReLU networks on their activation cells, our approach results in an \textit{exact} tractable characterization of the corresponding WDRO problem. Extensive evaluations demonstrate that our method achieves competitive robust accuracy against state-of-the-art baselines while offering tighter certificates than existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes a new type of adversarial attack along with an alternative approach for more precisely measuring model robustness.

### Strengths
Unlike existing adversarial attacks, the proposed method flexibly generates distributional adversarial examples, which could offer a different perspective on robustness evaluation.

### Weaknesses
The proposed attack does not appear to outperform existing methods when κ (kappa) is set to 1, and when transformer-based models are included in the experiments. This limitation diminishes the overall contribution of the work.

Specifically, the main purpose of AutoAttack (AA) is to reduce the influence of manually chosen hyperparameters, though it requires significant computational time to complete. The proposed Adaptive Auto Attack (A³ attack) aims to find competitive adversarial examples more efficiently. However, the generation time for adversarial examples has not been reported. Additionally, as shown in Figure 3, the proposed attack seems to struggle with selecting an appropriate step size. Together, these issues make it difficult to clearly assess the contribution of this work.

### Questions
* **Role of ReLU:** ReLU appears to play a central role in this method. I wonder whether the proposed attack can be extended to transformer-based models or to networks using other activation functions. If not, this limitation significantly reduces the generalizability and impact of the proposed method, especially compared with existing attacks. If it can be extended, please provide empirical evidence.

* **Jacobian Requirement:** If I understand correctly, the proposed attack requires computation of the Jacobian, which implicitly means it cannot be applied in black-box settings. Please clarify whether this limitation applies.

* **Target Selection (Algorithm 1):** The authors claim that “WDA consistently outperforms other single-method attacks” (lines 430–431). However, in Algorithm 1 (line 11), the target j* is altered in each iteration. Does this mechanism effectively play a role similar to that of ensemble attacks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper extends standard point-wise robustness evaluation to a distributional setting by introducing a Wasserstein Distributional Attack (WDA) parameterized by a parameter $\kappa$, which induces a strictly harder threat model. The paper further derives both upper and lower bounds on the worst-case loss, and attempts to give the sufficient conditions under which these bounds become tight. On the empirical side, this work further instantiates a concrete attack algorithm and shows that increasing the parameter kappa consistently reduces robust accuracy on CIFAR-10/100 under, thereby confirming that the theoretical bounds are consistent with practical robustness. Overall, the paper makes a practical contribution by pairing a stronger distributional threat model with bounds that are actually computable and reasonably tight.

### Strengths
One of the paper’s key strengths is how cleanly it upgrades point-wise robustness evaluation to distributional robustness with one intuitive $\kappa$ parameter, which naturally includes prior point-wise cases as its special case. 

In addition, the paper provides a concrete Wasserstein Distributional Attack and gives computable risk bounds, which turns WDRO into a practical tool. For ReLU networks, the activation-mask/Jacobian treatment even yields constructive (sometimes tight) certificates. The evaluation results further supports this claim: tuning the parameter $\kappa$ effectively controls the robust accuracy.

Taken together, the reviewer believes this work validates WDRO as a workable notion of robustness, and makes a strong case that future robustness certificates should target this distributional shifts rather than prior point-wise evaluation, and that defenses/training objectives should minimize this kind of distributional risk as in the paper.

### Weaknesses
The paper cites rLR-QP and BaB-Attack, but they don’t appear in the experiments. Is there a reason for omitting them (e.g., scalability or implementation constraints)? Since these are point-wise attacks—--i.e., the kappa=1 special case of your proposed WDA---it would be helpful to include them as baselines in the point-wise setting or clarify why they’re not considered.

Another minor typo: in the metrics description, the distributional accuracy is written as (1−1/kappa)⋅clean + kappa⋅adversarial, but given your mixture weights in Alg. 1/Eq. (10), the adversarial term should be 1/kappa instead of kappa.

### Questions
Please include rLR-QP and BaB-Attack as baselines in the evaluation or briefly explain why they’re not included.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new framework for distributional adversarial robustness under Wasserstein threat models while also providing theory along with it. More specifically regarding theory, it proposes a tight Lipschitz-based robustness certificates which are derived from the exact local geometry of ReLU and smooth neural networks. And regarding the experimental part, it proposes a novel Wasserstein Distributional Attack that generates distributional rather than point-wise adversarial examples.

### Strengths
- The paper has a good combination of both theory and experiments with many comparisons with other methods.
- Good presentation of the method and the experiments, and also an extensive ablation study.

### Weaknesses
- The paper is kind of hard to follow and read if you don't have experience with the Wasserstein distribution. But apart from that everything is well explained and presented.

### Questions
- I do think it is a good paper for the venue and fits the usual ICLR framework. Some questions that I would ask is how do the authors compute the step for every attack step?
- Regarding scalability can you provide attack runtimes and how it scales on bigger models that you are using here?

### Soundness
3

### Presentation
3

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
This paper studies tighter bounds for Wasserstein Distributional Robustness (WDRO). Based on these bounds, it proposes a new adversarial attack.

### Strengths
- The bounds for WDR are tighter.

### Weaknesses
- It is unclear the role of tighter bounds in developing Wasserstein distributional attack. As far as I see, Theorems 3.1 and 3.3 present to how to estimate two factors $l$ and $L$ relevant to the lower and upper bounds. After Corollary 3.2, the authors claim: "In the spirit of formulation 9, we propose a practical adversarial attack in Section 4 which aims to find adversarial direction $u^{(i)}$ for each sample i so that it maximizes the change of the logit function". Indeed, this is a typical and well-known way to craft adversarial examples without the theories from this paper.
- Although the paper aims to construct adversarial distribution (i.e., attack over distribution), in its Algorithm 1, it runs point-wise attack for each data example $X^{(i)}$. I cannot see any idea of distributional attack here because this requires the interaction of data examples during attack.  Moreover, it is unclear why the authors put clean examples $X^{(i)}$ in the adversarial distribution.
- No experiments on ImageNet which is the common standard for papers in adversarial attacks currently.

### Questions
- Why the robust accuracies increase when incorporating WDA to A3 in A3++?
- Wrong order of nested sets in Line 140?

### Soundness
2

### Presentation
2

### Contribution
2
