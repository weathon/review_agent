# FineFed: Forward-Only Federated Fine-Tuning for Many-Class Tasks under Non-IID Heterogeneity

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Federated learning (FL) on resource-constrained edge devices faces significant challenges when training large transformer models, particularly due to memory and computational limitations. While parameter-efficient fine-tuning (PEFT) methods help reduce memory usage, they still require back-propagation for gradient computation, which often demands more memory than storing model parameters. Forward-gradient (zero-order) FL offers a promising alternative by eliminating back-propagation, but existing methods suffer from computational inefficiency, poor performance on many-class tasks, and unstable convergence under non-IID data distributions.

We present \emph{FineFed}, an efficient forward-only FL framework that addresses these limitations through three key innovations: (i) \textbf{Forward-Only Head Tuning}, which enables exact gradient computation for many-class classification heads without back-propagation; (ii) \textbf{Uncertainty-Guided Forward Gradient Estimation}, which reduces computational cost by approximately $2.5\times$ via uncertainty-guided sample selection and micro-batch perturbations; and (iii) \textbf{Shared Momentum}, which ensures stable local updates and fast convergence under extreme non-IID data heterogeneity. Comprehensive evaluations across NLP and vision datasets demonstrate that FineFed achieves superior model accuracy and system efficiency compared to state-of-the-art methods, making forward-only federated learning practical for real-world deployment. 
Our code is available at \url{https://anonymous.4open.science/r/FineFed-0554/}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FineFed, a forward-only FL framework addressing the challenge of fine-tuning large transformer models on resource-constrained edge devices. Based on the dependence of existing methods on memory-intensive backpropagation, FineFed utilizes forward-only head tuning and uncertainty-guided gradient estimation to realize efficient fine-tuning with less computational costs. Besides, FineFed employs a momentum-sharing mechanism to overcome the non-converging issue of FL over non-iid data.

### Strengths
1. FineFed outperforms the baselines in the experiment.

2. This paper has good reproducibility.

### Weaknesses
1. This paper should not put the literature review in the appendix.

2. The design of momentum sharing is not new. Similar ideas have already been proposed in [1-3].

3. Uncertainty-based sample selection (section 3.2) employs a batch-wise sampling stragety. In this scenario, if the batches in a dataset are not will-shifted due to a bad choice of random seed, its efficacy may be compromised by unbalanced sample batches.  

4. The momentum sharing mechanism is orthogonal to forward-only head tuning and uncertainty-guided gradient estimation, and can be applied to other baselines as well.

5. Aside from model parameters, transmitting additional momentum may cause extra communication costs.

6. The presentation of this paper is not very clear.

### Questions
1. The presentation of this paper needs to be improved. For instance, concepts like “KB-level” communication (page 3) needs to be explained more clearly. Also, in lines 253-254, the authors state selecting 25% of total batches is sufficient. To support this claim, the authors should add a reference pointing at the corresponding empirical result. Lastlty, it is unclear how FineFed-HP works, and the difference between FineFed and FineFed-HP should be justified.

2. It seems that the forward gradient estimation (eq2) requires a closed-form representation of the gradient. Although such a representation is available for head layers with matrix-like shapes, for the complex feature extractors (e.g. convolutional layers), computing the closed-form representation of gradient is usually prohibitive.

3. FineFed is sensitive to the number of perturbation directions $P$ and the number of uncertain samples $k$. The authors should run some hyperparameter tuning experiments to show the effect of these hyperparameters.

4. The authors should add FedMeZO (discussed in the literature review) to the experimental baselines.

5. As shown in Algorithm 1 (lines 244-245), FineFed iteratively accumulates the gradient of PEFT, and I’m curious how to prevent the gradient-exposion problem in this scenario.

6. The authors need to specifically explain which parameters are PEFT parameters, and how to determine these parameters.

7. For fairness, the authors are supposed to evaluate the baselines with momentum sharing as well.


[1]On the Role of Server Momentum in Federated Learning, AAAI 2024.

[2]Momentum Benefits Non-IID Federated Learning Simply and Provably, ICLR 2024. 

[3]Accelerating Federated Learning via Momentum Gradient Descent, IEEE Transactions on Parallel and Distributed Systems, 2020.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the challenge of forward-only training in federated learning under many-class and strongly non-IID settings. It identifies that existing forward-only and zero-order gradient methods often suffer from unstable convergence and reduced accuracy when fine-tuning classification heads for heterogeneous clients. The proposed FineFed framework introduces a forward-only head tuning method, an uncertainty-guided forward gradient mechanism, and a shared momentum aggregation strategy. Experiments on NLP and vision benchmarks, combined with parameter-efficient tuning methods including BitFit, Adapter, and LoRA, demonstrate that FineFed’s effectiveness.

### Strengths
1.	While each component is individually known, the specific combination (forward-only analytical head gradients, uncertainty-guided perturbations, shared momentum) creates a robust and practical framework.
2.	Evaluations span multiple domains, tasks, and PEFT types, confirming generality and scalability.

### Weaknesses
1.	The proposed “FORWARD-ONLY HEAD TUNING FOR MANY-CLASS TASKS” is simple yet seems effective on large datasets such as CIFAR100 and ImageNet. However, it’s not clear why the proposed method converges well on these datasets, but the baselines such as FwdLLM and DeComFL cannot. More discussions and analysis are necessary to improve the theoretical depth.
2.	Despite the success on combining three key modules, the paper lacks convergence or variance-bound analysis for the combined use of single-sided forward gradients and shared momentum. 
3.	The main FineFed model still requires MB-level uplink/downlink bandwidth, comparable to FedAvg-M. Competing forward-only methods (FwdLLM, DeComFL) use KB-level communication. Although FineFed-HP (head perturbation) achieves lower communication, it doubles computation cost and slightly reduces accuracy.
4.	Experiments use ≤100 clients with fixed-class (pathological) non-IID partitions. It remains unclear whether FineFed scales to thousands of clients or more realistic Dirichlet partitions.
5.	More experimental setups such as random seeds should be consolidated in the appendix for easier reproducibility.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents FineFed, a forward-only federated fine-tuning framework designed to make FL of large models feasible on resource-constrained clients. The authors aim to address the limitations of backpropagation-based fine-tuning and previous forward-only methods.

### Strengths
++The paper targets an important problem, i.e., making large model fine-tuning feasible on memory-constrained devices, which is a key bottleneck for future FL deployments.
++The description of forward-only head tuning is clear and rigorous.
++The paper presents diverse experiments across multiple domains, datasets, and PEFT strategies. Ablation studies isolate the impact of each component.
++The inclusion of code availability, hyperparameters, and detailed appendices improves reproducibility.

### Weaknesses
++The paper lacks any formal convergence or variance analysis for FineFed’s forward-gradient estimators or shared momentum dynamics.
++While combining the three techniques is novel in integration, each component is built on existing concepts with limited theoretical innovation. Therefore, the novelty is incremental.
++The accuracy-communication-compute trade-offs between FineFed and FineFed-HP are not fully analyzed under larger model scales.

### Questions
Please address all the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper aims to improve the performance of forward-only (zeroth-order) in federated learning. The authors propose FineFed, a hybrid framework that is based on three primary components: 1) Forward-Only Head Tuning (FineFed-HT), 2) Uncertainty-Guided Forward Gradient Estimation and 3) Shared Momentum.

The paper combines the memory-saving benefits of ZO for the large backbone with the fast-convergence of exact analytic gradients for the small but critical head. The empirical results show that FineFed outperforms prior forward-only baselines on many-class tasks and achieves accuracy comparable to backpropagation-based methods such as FedAvg-M.

### Strengths
1- The authors have clearly explained the existing methods in forward only methods in FL and their shortcomings.

2- Focusing on Head tuning especially for multi-class datasets is interesting.

3- The experiments are comprehensive and demonstrate the method's effectiveness.

4- The ablation study shows the importance of different design choices.

### Weaknesses
1- The authors mention that this is a forward-only (zeroth_order) paper but the "Forward-Only Head Tuning"  is the exact analytic first-order gradient of the cross-entropy loss with respect to the head parameters. 

2- Authors explain that one of the benefits of forward-only methods is their communication efficiency. However this method requires MB-scale communication (Table 2), as it must transmit the full PEFT parameters and momentum state, just like FedAvg-M. 

3- Overstated Novelty of Components: Some choices such as Shared Momentum or Uncertainty-Guided Sampling are very common and well-established techniques in machine learning to improve the performance or make the training more efficient.

4- The scope of the paper is limited to classification tasks.

### Questions
Please check out the previous section.

### Soundness
2

### Presentation
2

### Contribution
2
