# Communication-Efficient Federated Learning with Adaptive Number of Participants

- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
While communication efficiency is a central challenge in Federated Learning (FL), standard protocols typically rely on a fixed, heuristically chosen number of participating clients per round. This rigid approach often leads to redundant communication in easy optimization stages or insufficient aggregation in heterogeneous regimes. In this work, we propose Intelligent Selection of Participants (ISP), an adaptive algorithm that dynamically optimizes the number of active clients to maximize communication efficiency without compromising convergence. Theoretically, we derive a convergence bound for the non-convex setting, revealing that the required number of participants scales with the gradient heterogeneity, rather than the total number of devices in the network. Guided by this insight, ISP speculatively adjusts the participation budget based on real-time training dynamics. ISP achieves consistent communication savings of up to 30\% while matching the final accuracy of full-budget baselines. Furthermore, detailed ablation studies highlight the robustness of our adaptive criterion, establishing the dynamic selection of client count as a critical, distinct optimization task in federated systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Intelligent Selection of Participants (ISP), a dynamic mechanism that adaptively determines the optimal number of clients to involve in each communication round of federated learning. Unlike conventional approaches such as FedAvg and FedProx that assume a fixed client count, ISP formulates participant selection as an optimization problem to minimize communication costs while maintaining model improvement. It integrates seamlessly with standard federated algorithms and operates without modifying client optimizers. Extensive experiments on CIFAR-10, Tiny-ImageNet, and real-world ECG data show that ISP consistently reduces communication costs by up to 30%, without sacrificing model accuracy.

### Strengths
+ Introduces the Intelligent Selection of Participants (ISP) framework, which formulates client-count determination as an optimization problem balancing convergence and communication cost.
+ Empirically validated across diverse datasets (CIFAR-10, Tiny-ImageNet, and ECG), showing up to 30% communication savings and no degradation in accuracy.
+ Offers a general solution applicable to both standard and advanced FL scenarios, including gradient compression.
+ The paper is well-structured and easy to follow.

### Weaknesses
- The paper’s major weakness lies in its lack of theoretical analysis. Despite formulating client-count selection as an optimization problem, the authors provide no convergence guarantees, communication–computation trade-off bounds, or proofs of optimality for the ISP mechanism.
- The optimization in Equation (3) is treated heuristically via Monte Carlo approximation, without analytical discussion of its stability, variance, or expected bias.
- Without formal complexity or asymptotic analysis, it remains unclear whether the observed 30% communication savings are due to principled algorithmic efficiency or empirical tuning effects.
- The intermediate communication step (τ + ½) introduces synchronization and computation overhead, which might partially offset communication savings, especially in large or unstable networks. However, this overhead is only discussed qualitatively and not measured quantitatively.
- The Monte Carlo estimation of loss reduction has no analysis of its variance, bias, or sample efficiency, raising concerns about ISP’s decision reliability across rounds.
- In the CIFAR-10 experiments, results show that POW-D outperforms ISP-POW-D in terms of test loss and communication cost balance. This inconsistency undermines claims that ISP universally enhances all sampling strategies. 
- The paper does not include statistical significance testing of improvements, leaving uncertainty over whether the reported gains (especially marginal ones) are meaningful.
- The evaluation scope is narrow, focusing on image and ECG classification tasks, and other domains like NLP are not explored, limiting the method’s generalizability.
- Experimental settings (e.g., client sampling distributions, training parameters) are split between the main text and appendices, reducing accessibility and reproducibility for readers.

### Questions
1. Can you provide any theoretical justification or convergence analysis for ISP? Even partial proofs or convergence bounds would strengthen the paper.
2. How stable is the Monte Carlo loss estimation? Include results showing its variance or reliability across runs.
3. How costly is the intermediate step (τ + ½) in practice? Please measure and report this overhead quantitatively.
4. Are the 30% communication savings due to ISP’s design or specific hyperparameter tuning?
5. Why does POW-D outperform ISP-POW-D on CIFAR-10? Analyze when ISP may underperform or over-select clients.
6. Can you test ISP on other domains (e.g., NLP or speech tasks)? This would show its generality beyond image and ECG data.
7. How would ISP perform under client dropouts or asynchronous updates? A short sensitivity test would make the results more practical.
8. Discuss how ISP could work with differential privacy or personalized FL frameworks.

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
This paper introduces Intelligent Selection of Participants(ISP), a dynamic mechanism that adaptively determines the optimal number of clients to participate in each communication round of Federated Learning (FL). ISP treats client number as a tunable variable and selects the minimal count needed to guarantee expected model improvement. This paper conduct experiments on CIFAR-10, Tiny-ImageNet, and a large real-world ECG dataset.

### Strengths
- This paper proposed the approach to determine the optimal number of participating clients per round. This adaptive viewpoint expands the optimization scope of FL and directly addresses communication bottlenecks.
- Extensive experiments on both standard benchmarks (CIFAR-10, Tiny-ImageNet) and a large-scale, real-world ECG dataset substantiate ISP’s practical relevance. The reported 30% communication reduction with comparable or better accuracy is a outcome.
- The authors include detailed ablations, for example, depth, windows, delta etc., theoretical derivations enhancing reproducibility.

### Weaknesses
- The proposed intermediate full-client communication step (Algorithm 2) contradicts the goal of communication efficiency, especially for large-scale networks. Though amortized over delta rounds, it still introduces a potential scalability concern.
- ISP partially mitigates communication costs, but its performance still depends on user-defined parameters, which may affect the results of dynamic selection.

### Questions
- For Table 2, please define the unit of measurement used in the *Communication* column. It’s unclear whether it represents the total number of communication rounds, total transmitted updates, or another metric.
- In Table 2, the average training time for ISP-FedCor appears about one hour longer than its baseline FedCor result. Could the authors clarify this discrepancy? Since ISP is expected to select fewer clients per round, a shorter overall training time would normally be expected. Please explain the underlying cause of the slower runtime.
- Regarding Figure 2, the plots suggest that ISP consistently outperforms its corresponding baselines. However, in Figure 2(b), we observe that as the number of clients increases, the model performance also improves across algorithms. Given that ISP often selects fewer clients, could the authors elaborate on how ISP maintains or enhances performance under reduced client participation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes ISP (Intelligent Selection of Participants), an adaptive mechanism for federated learning (FL). ISP dynamically determines the optimal number of clients per training round to enhance communication efficiency without compromising model accuracy. Unlike existing FL methods, which assume a fixed number of participants, ISP formulates the round-wise client count selection as a constrained optimization problem, selecting the minimum number of clients needed to achieve the expected loss reduction. The authors validate ISP across diverse setups, including vision transformers, real-world ECG classification, and gradient-compressed training. In these cases, ISP consistently achieves communication savings of up to 30% without degrading performance.

However, the work is limited by insufficient theoretical support, unaddressed hyperparameter sensitivity, and incomplete comparisons with the latest baselines. The computational overhead from intermediate communications also requires further optimization for resource-constrained environments.

### Strengths
1. ISP formalizes the round-wise choice of the number of participants as a constrained problem, selecting the smallest value that achieves expected loss decrease.
2. The framework is compatible with popular FL algorithms and requires no changes to client optimizers, enabling easy integration with standard FL pipelines.

### Weaknesses
--Unmitigated ISP overhead restricts edge use. The Monte-Carlo approach in ISP introduces heavy computational overhead, which is not fully mitigated and may limit applicability in resource-constrained edge environments.

--No comprehensive analysis of ISP hyperparameter impact. ISP relies on multiple hyperparameters (e.g., window Δ, momentum β, resolution ω), but the paper lacks a comprehensive analysis of how these parameters affect performance across different FL scenarios.

--Limited Theoretical Justification. The optimization formulation for client count selection is primarily validated empirically, yet theoretical analysis on convergence guarantees and the trade-off between communication efficiency and model accuracy is insufficient.

--A lack of comparison with advanced baselines. The work compares ISP with classic and some state-of-the-art client sampling methods but overlooks recent advanced adaptive FL frameworks, making it hard to assess ISP’s competitiveness against the latest techniques.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles an under-explored question in federated learning which is finding out how many clients should participate per round. Existing methods fix this number, focusing only on which clients to sample. The authors propose Intelligent Selection of Participants (ISP), an adaptive mechanism that dynamically adjusts client count based on observed training progress to improve communication efficiency. ISP periodically runs a lightweight intermediate round to estimate how model performance changes with different client counts. It then selects the smallest number of clients that still improves the loss and smooths updates over time to prevent oscillation. The method integrates with standard FL algorithms (FedAvg, FedProx, SCAFFOLD) and techniques like client sampling and gradient compression. Experiments on CIFAR-10, Tiny-ImageNet, and real ECG data show fewer communication rounds without accuracy loss.

### Strengths
- **S1:** Clearly motivated and easy to integrate into existing FL systems.
- **S2:** Consistent gains in communication efficiency across tasks and datasets.
- **S3:** Works with client selection and compression methods.
- **S4:** Thorough experimental coverage, including a large ECG setup.

### Weaknesses
- **W1:** Requires synchronized or full-client intermediate rounds, which limit scalability.
- **W2:** No theoretical analysis of convergence or stability.
- **W3:** ISP highlights a neglected dimension of FL optimization: dynamically adjusting client count to balance efficiency and performance. While conceptually straightforward, it’s practical and general, requiring no client-side changes. The contribution is incremental but relevant to real-world FL deployments. 
- **W4:** Modest conceptual novelty given prior adaptive participation work.

### Questions
- **Q1:** How does ISP behave under highly variable client availability, such as mobile or cross-device FL?
- **Q2:** How sensitive is the algorithm to its tuning parameters such as particularly the interval between updates and the smoothing coefficient?
- **Q3:** Could the intermediate probing phase be replaced by a lighter-weight estimation, such as tracking gradients or validation loss trends?
- **Q4:** How does the approach compare to existing adaptive participation heuristics, such as linear decay or reinforcement-learning-based scheduling?
- **Q5:** Could the approach handle asynchronous updates, where clients finish at different times?

### Soundness
2

### Presentation
3

### Contribution
2
