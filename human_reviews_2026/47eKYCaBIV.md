# Robust Federated Inference

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 4, 8, 8, 4, 4

## Abstract
Federated inference, in the form of one-shot federated learning, edge ensembles, or federated ensembles, has emerged as an attractive solution to combine predictions from multiple models. This paradigm enables each model to remain local and proprietary while a central server queries them and aggregates predictions. Yet, the robustness of federated inference has been largely neglected, leaving them vulnerable to even simple attacks. To address this critical gap, we formalize the problem of robust federated inference and provide the first robustness analysis of this class of methods. Our analysis of averaging-based aggregators shows that the error of the aggregator is small either when the dissimilarity between honest responses is small or the margin between the two most probable classes is large. Moving beyond linear averaging, we show that the problem of robust federated inference with non-linear aggregators can be cast as an adversarial machine learning problem. We then introduce an advanced technique using the DeepSet aggregation model, proposing a novel composition of adversarial training and test-time robust aggregation to robustify non-linear aggregators. Our composition yields significant improvements, surpassing existing robust aggregation methods by 4.7 - 22.2% in accuracy points across diverse benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the adversarial robustness of federated inference in the presence of corrupted clients. It demonstrates that robust averaging may be insufficient due to the non-continuity of the argmax operator. It also introduced a DeepSet-TM approach for robust non-linear aggregation by formulating the robust inference as an adversarial training problem. Experimental results demonstrate the superior robustness of the proposed DeepSet-TM framework over baselines in various attacking scenarios.

### Strengths
**Originality:** This work highlights the impact of the point-wise model dissimilarity and the probit’s margin on the robustness gap using linear averaging aggregation for federated inference. It also introduced a novel DeepSet-based framework to learn non-linear data-dependent robust aggregation function.

**Quality:** The provided counter example shows the prediction error induced by the non-continuity of the argmax operator. Theorem 1 theoretically analyzes the robustness gap of linear averaging aggregation for federated inference, followed by empirical validation in Figure 3. The robustness of the DeepSet-TM framework is also validated in the experiments using various attacks.

**Clarity:** The paper is well-written. The motivation behind robust federated inference is clearly discussed. The proofs of lemma and theorem are provided.

**Significance"** The problem of federated inference under potentially corrupted clients is highly significant in the context of real-world federated learning systems. A robust federated inference framework can largely improve the trustworthiness of the deployed systems.

### Weaknesses
(1) One major concern is the certification for DeepSet-TM under adversarial corruptions. Section 3 considers the linear averaging aggregation strategies and derives the theoretical result (Theorem 1) in shaping the robustness gap. Specifically, it highlights the impact of the corruption fraction, the model dissimilarity, and the probit’s margin. However, it is unclear how this robustness gap can be characterized in non-linear cases using DeepSet-TM. The highlighted potential factors, such as the point-wise model dissimilarity and the probit’s margin, are not explored in the non-linear aggregation of DeepSet-TM.

(2) Another concern is the computational efficiency of the proposed DeepSet-TM method in adversarial training (i.e., bi-level optimization problem). Though several simplification strategies are proposed, the running efficiency of DeepSet-TM is not well discussed compared to baselines.

(3) The impact of the simplification strategies for optimizing the robust DeepSet aggregator can be further discussed. With these simplification strategies (as well as ROBAVG operator in Eq. (10)), how will the derived solutions still approximate the optimal solution of the optimization problem in Eq. (9)?

### Questions
(1) Why is the assumption in line 241 required, i.e., if $[z]\_k = [z]\_{k'}$ for all $k,k'$, then $MARGIN(z) = \infty$?

(2) Lines 321-322 show that robust averaging is only incorporated at the inference time. Will the added ROBAVG operator change the optimality of the robust DeepSet aggregator defined in Eq. (8)? Why does it reduce the overall sensitivity of $\phi\_{\theta^*}$? 

(3) Why does the proposed DeepSet-TM approach consistently achieve inferior robustness under Loss Maximization Attack (LMA) in Table 2?

(4) DeepSet-TM replaces the point-wise loss function from the indicator function with the cross-entropy loss. Will it better handle the counter examples in lines 222-225?

(5) How is the function $\bar{h}(x)$ calculated in the counter example?

### Soundness
2

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
This paper systematically investigates a critical security issue in federated learning and edge computing: the robustness of federated inference. The authors point out that existing model ensemble and aggregation paradigms are highly vulnerable to simple attacks from malicious clients. They formally define the problem of "robust federated inference" for the first time, filling a gap in the field. Through an initial theoretical analysis of averaging-based aggregators, the paper reveals that the error of these methods depends not only on the proportion of malicious clients (corruption fraction) but also on the dissimilarity between honest client outputs and the prediction margin of the models for the two most probable classes. This analysis lays a solid theoretical foundation for subsequent method design.
To address the robustness problem in non-linear scenarios, the paper proposes a novel DeepSet-TM aggregator. This method ingeniously reframes the robust inference problem as an adversarial machine learning problem in the probit vector space and leverages the permutation invariance of the DeepSet architecture to significantly reduce the computational complexity of adversarial training. By introducing adversarial samples during the training phase and combining them with robust averaging (CWTM) mechanisms during the testing phase, DeepSet-TM ensures the aggregator's resilience to malicious inputs. The paper conducts rigorous empirical validation on multiple benchmarks, including image and text classification. The results demonstrate that DeepSet-TM achieves a significant improvement in worst-case accuracy, ranging from 4.7% to 22.2%, compared to existing techniques across various attacks (including the newly proposed SIA attack), strongly proving the practical utility and robustness of the proposed scheme.

### Strengths
This paper makes several substantial contributions in the field of robust federated inference, demonstrating high originality and significance. Firstly, the paper formally defines the critical problem of "robust federated inference" for the first time, clearly delineating the challenge of ensuring aggregator accuracy in the presence of malicious clients. This issue has long been overlooked in existing literature and possesses significant theoretical and practical value. Secondly, the paper conducts an in-depth theoretical analysis of averaging-based aggregators, revealing the complex relationship between their robustness, client output disparities, and prediction margins. This provides a solid theoretical foundation for understanding the limitations of existing methods and points the way towards more refined future designs. Most importantly, the proposed DeepSet-TM aggregator is highly original; it creatively reframes the robust inference problem as an adversarial machine learning problem and cleverly leverages the permutation invariance of the DeepSet model to address the computational complexity of adversarial training. This method, by combining adversarial training and robust averaging during the testing phase, effectively enhances robustness. The experimental section also demonstrates DeepSet-TM's superior performance across various datasets and attack types, with accuracy improvements ranging from 4.7% to 22.2%, thereby proving its strong potential and importance in practical applications.

### Weaknesses
Despite the paper's excellent performance in both conceptual and methodological aspects, there remain areas for improvement. Firstly, while the paper defines robust federated inference in the "Problem Formulation" section, some critical assumptions, such as those regarding the security of the client models themselves, could be elaborated upon in more detail. Currently, the paper primarily focuses on the robustness of the aggregation phase. However, if client models are susceptible to poisoning or backdoor attacks, this could impact the overall system robustness, a topic that receives relatively less discussion in the paper. Secondly, the computational cost and model complexity of the DeepSet-TM aggregator are potential weaknesses. Although the paper mentions leveraging permutation invariance to reduce the search space for adversaries, its actual overhead during training and inference (especially in large-scale federated deployments) still requires more detailed analysis and quantification compared to simpler averaging or median aggregators, such as providing end-to-end training time or inference latency comparisons for different aggregators. Furthermore, while the experimental results are impressive, it would be beneficial to further investigate DeepSet-TM's performance under varying degrees of federated heterogeneity (e.g., larger data distribution shifts among different clients) to comprehensively evaluate its generalization capabilities.

### Questions
1.The paper states that "the DeepSet-TM aggregator incorporates robust averaging (CWTM) at the inference time." Could you please elaborate on the specific mechanism of this "incorporation"? How are the outputs of DeepSet-TM combined with or selected alongside the results of CWTM during the testing phase, and how does this combination synergistically enhance robustness?
2.Regarding the computational efficiency of the DeepSet-TM aggregator, could you provide a more detailed analysis? Specifically, how much do the training time and inference latency of DeepSet-TM increase after introducing adversarial training and robust averaging at the testing phase, compared to a pure DeepSet or other non-linear aggregators? In large-scale federated deployments, is this increase acceptable in practice?
3.The paper highlights experiments on federated heterogeneity (controlled by the Dirichlet distribution parameter α), using α={0.3, 0.5, 0.8}. Could you further investigate DeepSet-TM's performance under higher heterogeneity (e.g., smaller α values, or other means of introducing larger data distribution shifts)? Does the design of DeepSet-TM have inherent advantages or limitations when dealing with extreme data heterogeneity?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides the first formal robustness analysis of federated inference, where a central server aggregates model predictions (probits) from multiple local clients to make a global decision. It shows that widely used averaging-based and trimmed-mean aggregators, though intuitively robust, can still fail even small deviations in logit (prob) space can flip the final argmax decision. The authors formally characterise the robustness gap and prove upper bounds showing that robustness depends on three key factors: the corruption fraction the inter-client dissimilarity of honest outputs, and the classification margin of the average prob vector. To go beyond linear aggregation, they cast robust federated inference as an adversarial learning problem over probability vectors, and propose a DeepSet-based neural aggregator that combines adversarial training with test-time robust averaging.

### Strengths
* The paper makes aa original theoretical contribution by providing the first formal robustness framework for federated inference, a setting previously treated mainly empirically.
* the theoretical strength is the counterexample construction that demonstrates the argmax discontinuity problem, showing that even infinitesimal perturbations in averaged probits can cause misclassification
* non-linear aggregation as an adversarial learning problem over probit vectors is conceptually novel. By recognizing that the adversarial space is structured (the probability simplex), the authors identify a tractable route to robustness.
* The proposed DeepSet-based robust aggregator then builds on this insight to combine permutation invariance, adversarial training, and inference-time robust averaging, achieving both theoretical justification and strong empirical performance.

### Weaknesses
* In REERM, the inner maximization is combinatorial and non-differentiable, requiring heuristic relaxations that undermine its minimax semantics. 
* The adversarial model is overpowered and underspecified, leading to unrealistic robustness assumptions and potentially excessive conservatism.
* Moreover, the learned DeepSet aggregator, though effective empirically, lacks analytical interpretability or formal robustness guarantees. Thus, REERM advances the conceptual framing of robust federated inference but leaves open key challenges in tractable optimization, principled approximation, and theoretical certification.
* What about KRUM aggregation as the robust aggregator?

### Questions
Kindly refer to weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper formally define and analyze the problem of robust federated inference, where a central server aggregates predictions from multiple local models in the presence of adversarial clients. They introduce a method combining adversarial training with test-time robust aggregation (e.g., trimmed mean), which improves robustness without increasing training cost. The proposed DeepSet-TM method outperforms existing robust aggregation techniques.

### Strengths
1. Systematically address robustness in federated inference (as opposed to training), with both theoretical and empirical contributions.
2. The DeepSet-based model with adversarial training and test-time robust aggregation is effective.
3. The consistent performance gains as the number of clients and adversaries increases

### Weaknesses
1. Adversarial training with DeepSet, while more efficient than full combinatorial search, is still computationally intensive, especially as n grows.
2. The analysis assumes f < n/2, which is common but may not hold in highly adversarial real-world settings. 
3. Some privacy concerns. The work focuses on robustness but does not discuss potential privacy implications of sharing probits or using server-side models.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of robust federated inference, where the goal is to design an aggregator capable of handling mixtures of clean and corrupted predictions from multiple clients in a federated setting. The authors propose a training paradigm that estimates the residual test error through an adversarial loss, enabling robustness against prediction corruption. The method is theoretically grounded and demonstrates strong empirical performance compared to existing baselines.

### Strengths
* Good writing, clear motivations
* Theoretically guaranteed proposal with outstanding performance.

### Weaknesses
* Theorem 1: How can the bound be controlled or estimated in practice? Since the left-hand side represents a loss function with an arbitrary value, how can we ensure it remains smaller than 1, given that the bound is a probability? The authors should provide a discussion after the theorem to clarify its practical implications.
* Applicability to Regression: How can the proposed framework (loss function, training algorithm, and theoretical results) be extended to regression tasks?
* Ablation Study: The authors should include ablation studies on the DeepSet aggregator to justify its necessity. Why are recently published aggregator architectures [1,2] not suitable? Would the findings remain consistent if a stronger or alternative aggregator were used?
* Experimental Settings: The current experimental setup appears somewhat simplified. In realistic scenarios, different clients may face distinct types of attacks. Have the authors evaluated the proposed approach under heterogeneous corruption settings to demonstrate its robustness and practical applicability?

[1] IIse et al., “Attention-based Deep Multiple Instance Learning”, arxiv 2018

[2] Xiang et al., “Exploring low-rank property in multiple instance learning for whole slide image classification”, ICLR’23

### Questions
See Weaknesses.

The authors should address the practical implications of theorem 1 and extending experiments / ablation studies.

### Soundness
2

### Presentation
3

### Contribution
3
