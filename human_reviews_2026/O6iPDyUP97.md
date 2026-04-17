# Treatment Responder Classification with Abstention

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Treatment responder classification seeks to learn a rule to classify individuals who will benefit from the treatment. This paper studies a new scenario in treatment responder classification when abstention is allowed, i.e., practitioners can opt out of making uncertain classification on some individuals for further investigation. By revealing the implicit relation between causal misclassification risk with abstention and Conditional Value at Risk (CVaR), we develop a doubly robust method named $\textbf{TRECA}$ to learn the classification rule under loose convergence conditions on nuisance parameters, and further extend it to deal with possible violation on key assumptions such as monotonicity and unconfoundedness. Rigorous theories and extensive experiments on two real-world datasets demonstrate the theoretical and experimental guarantee on our methods in learning treatment responders classification rules with low regret at the cost of limited abstention.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes an adaptation of abstention methods to treatment responder classification. 
The authors present a principled method grounded in theory to correctly estimate the functions of interest and evaluate their approach over two existing datasets.

### Strengths
The paper's main strengths:

1. The extension of abstention to the setting of treatment responder classification is very interesting
2. The presented method is theoretically grounded
3. The paper has the potential to be impactful

### Weaknesses
The main concerns I have are:

*Soundness*: The proofs rely on the unconfoundedness assumption, which is not properly discussed in my opinion. Indeed, a classical argument supporting abstention (as also acknowledged by the authors e.g., in lines 69-70; ) is the possibility of obtaining extra information on some difficult-to-classify instances. If new variables are involved, I think one might need an ad hoc notion of unconfoundedness.

*Clarity:* The paper notation is very dense, and sometimes it also seems not very consistent. For example, it seems to me that both $\mathbb{E}$ and $E$ are being used for denoting expected values (es within Eq. 1 and Proposition 1, respectively).


*Empirical Evaluation:* I am not sure whether I completely understand how the empirical evaluation was carried out. 
In particular, the authors state:

> For fair comparison, we randomly abstain the samples on baselines to ensure the metrics are computed on the same amount of samples.

I am missing the point here. Why not consider some naive "abstention" baselines rather than randomly abstaining? For instance, to better isolate the contribution of TRECA, it would be informative to include some abstention baseline that does not rely on the proposed doubly robust approach. For example, why not abstain when the uncertainty (e.g., using some ensemble for $\tau(x)$ estimation or some variance measure) is too large (i.e, above a certain quantile)? This would clarify whether the observed performance gains stem from the causal abstention risk itself or from the underlying CATE estimation quality.

*Relation to Prior Work:* I think the paper could benefit from adding another related work paragraph mentioning a few recent works that investigated (from different perspectives) abstention and causal inference. 
For instance, [1] proposes to consider a *counterfactual score* of an abstaining classifier, defined as the expected performance of the classifier had it not been allowed to abstain.
Moreover, some recent works connect causality and deferral (i.e., rather than simply abstaining, we involve another predictor in the decision): in [2] the authors propose a framework to "causally" evaluate classifiers that defer the decision to humans, while in [3] the authors propose an approach to learn deferral policy robust to confounding.

 I think discussing these works might better position the work in the current literature

**References**


[1] Choe, Yo Joong, Aditya Gangrade, and Aaditya Ramdas. "Counterfactually comparing abstaining classifiers." Advances in Neural Information Processing Systems 36 (2023): 28281-28293.

[2] Palomba, F., Pugnana, A., Alvarez, J. M., & Ruggieri, S. (2025, April). A Causal Framework for Evaluating Deferring Systems. In International Conference on Artificial Intelligence and Statistics (pp. 2143-2151). PMLR.

[3] Gao, R., & Yin, M. (2025, April). Confounding-Robust Deferral Policy Learning. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 39, No. 13, pp. 14238-14246).

### Questions
I would like the authors to discuss the above weaknesses.

I have a few other remarks I would like the authors to discuss:


*Technical Differences for Novelty* I appreciate the bridging of treatment responder classification and abstention. Still, I think the authors might emphasize better the differences between some of their results and the results from Franc et al., 2023. The main differences I see are some technical concerns related to the estimation due to the nuisance parameters, but in spirit, the optimal abstention strategy still consists of thresholding some specific conditional risk. Is it true or am I missing something?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work examines the problem of treatment responder classification when abstention is allowed. In Theorem 1, the authors derive a connection between their problem and another minimization problem that involves a well-known quantity, the Conditional Value at Risk (CVaR). Subsequently, they provide convergence guarantees for their methods both with and without a monotonicity assumption. Finally, they conduct a comprehensive experimental evaluation comparing their methods with several others on two real-world datasets.

### Strengths
1) Despite this problem is well known in Online Learning literature (see i.e. https://arxiv.org/pdf/1905.09561, https://arxiv.org/abs/2305.10564 and references therein) it’s yet unexplored in the causal inference literature.
2) The authors derived a nice connection of causal classification with abstention with Condition Value at Risk (CVaR), a well known quantity.
3) The authors propose two algorithms to solve this problem and establish theoretical results guaranteeing their asymptotic convergence.
4) The authors compare their experimental results against numerous benchmarks, and their methods demonstrate superior performance.

### Weaknesses
1. I believe several sections of the text require revision.
2. The authors should add an introductory paragraph explaining the notation used throughout the paper (i.e., the min and max operators, the O_p notation, etc.).
3. In lines 132-134, the authors claim that the constraint formulation ensures generalization performance for their classifier. I believe this section requires clarification: generalization is guaranteed only for the subsets of P where the algorithm does not abstain. In other regions, performance may be arbitrarily poor.
4. In Theorem 1, it is unclear why the abstention rule r^\star necessarily belongs to the hypothesis set \mathcal{R}.
5. I suggest that after introducing Assumptions 4 and 5, the authors include a brief discussion characterizing the strength of these assumptions and providing examples that satisfy them. For instance, I question whether Assumption 5 (monotonicity) holds in medical scenarios. Specifically, prescribing a drug instead of a placebo to a non-patient does not always produce positive effects.
6. In Theorem 2, the authors should explicitly state in the main text the dependence on problem-specific quantities such as \delta and c, and clarify why this result advances the existing literature. For example, the conditions under which the \epsilon_n quantity vanishes remain unclear.
7. Algorithm 1 is clearly presented and well-written; however, no theoretical guarantee (theorem) establishes its convergence or convergence rate.
8. In lines 312-313, the authors state that it is possible to derive differentiable upper and lower bounds for \rho(x) with respect to \tau, yet they subsequently employ \max{\tau/2, 0}, a function that is non-differentiable in \tau at 0. Why does this non-differentiability at zero not pose a problem?
9. In Theorem 4, the authors require that the predictor hypothesis class \mathcal{F} have VC dimension larger than n. What about simpler classes, such as linear or logistic models? Do you have experimental or theoretical evidence demonstrating the performance of your method in these cases? I would expect the theoretical results to be parameterized by statistical quantities characterizing both the predictor class \mathcal{F} and the abstention class \mathcal{R} (such as VCdim(F,R)).
10. In the experimental section (line 392), it remains unclear how the authors used the MLP to construct both the predictor and the rejector. How many parameters does this MLP contain? Additionally, why do the benchmark methods abstain at random locations rather than at the same points where your method abstains?

### Questions
1) See the above section.
2) I believe a natural extension would be to consider the dual problem: minimizing the number of abstentions subject to maintaining low misclassification rates on the training set. Consider a scenario where you have a stream of patients and wish to assign the correct treatment to each one without abstaining from too many cases.

Typos: line 104 Y_i(1) ,  Y(-1)

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the treatment responder classification problem, aiming to develop a classification rule that provides predictions for most individuals while permitting a small subset of samples to abstain from decision-making, thereby minimizing overall misclassification risk. The authors first establish the connection between causal misclassification risk and Conditional Value at Risk (CVaR). Building on this insight, they propose a doubly robust classification method named TRECA, which jointly estimates the propensity score and outcome models to achieve stable performance even under weak convergence conditions. Furthermore, they introduce an enhanced variant, TRECA+, which relaxes the monotonicity assumption to improve adaptability in non-monotonic scenarios. Comprehensive experiments on two real-world datasets (Twins and Jobs), along with their monotonicity-transformed counterparts, demonstrate that both TRECA and TRECA+ surpass various baseline methods in classification accuracy and stability.

### Strengths
(1) The paper formalizes the modeling of the Treatment responder classification problem based on the causal inference framework, with a systematic derivation process that possesses theoretical depth.

(2) The paper introduces an abstention decision mechanism into the causal responder classification task, enhancing the model's practicality from the perspectives of risk minimization and decision theory, with promising application prospects.

### Weaknesses
(1) The paper includes only Figure 1; however, Figure 3 is referenced on line 430 of page 8 in the main text, and Figure 5 is mentioned in the appendix. It is recommended that the authors carefully check and supplement the missing content.

(2) In the introduction section, the paper provides a clear overview of the treatment responder classification and abstention selection problems, but it does not adequately elaborate on the limitations of existing methods, making it difficult to distinguish the differences between the proposed method and other approaches.

(3) The methods section primarily focuses on mathematical derivations (such as loss functions, theorems, assumptions, and proofs), lacking an intuitive description of the overall model structure. For example, the paper mentions the predictor and rejector, but does not clearly explain their training process. 

(4) The paper lacks an overall framework diagram to intuitively illustrate the structure and process of the method.

(5) The methods section of this paper lacks discussion on time complexity or space complexity. It is recommended that the authors include corresponding analyses, particularly on the computational complexity in key steps such as the joint estimation of the propensity score model and outcome model, as well as the doubly robust method.

(6) The experimental setup section of the paper does not provide detailed descriptions of hardware and software configurations, such as the computing environment, GPU/CPU specifications, and so on, as well as exact parameter values, such as learning rate, batch size, optimizer type, etc.

### Questions
(1) Could the authors confirm whether Figures 3 and 5 are missing? Figure 3 is referenced on line 430 of page 8, and Figure 5 is mentioned in the appendix, yet only Figure 1 is provided. Could the authors either supply the missing figures or offer an explanation to enhance the completeness of the paper?

(2) The introduction clearly outlines the problems at hand, but it lacks detailed discussion on the limitations of existing methods. Could the authors elaborate on these limitations to provide a more comprehensive background?

(3) The methods section presents substantial mathematical content but does not provide an intuitive overview of the model, such as the training process for the predictor and rejector. Could the authors include such an overview to improve the clarity and accessibility of the section?

(4) There is no discussion regarding the time or space complexity of the proposed methods, particularly with respect to the joint estimation within the doubly robust framework. Could the authors address this aspect to inform readers of the computational efficiency of the approach?

(5) The experimental setup section omits key hardware and software details (e.g., computing environment, GPU/CPU specifications) as well as hyperparameters (e.g., learning rate, batch size, optimizer). Could the authors provide these details to ensure the experiments are reproducible?

### Soundness
2

### Presentation
2

### Contribution
2
