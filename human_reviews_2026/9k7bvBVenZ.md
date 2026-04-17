# FedMuon: Federated Learning with Bias-corrected LMO-based Optimization

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 2

## Abstract
Recently, a new optimization method based on the linear minimization oracle (LMO), called Muon, has been attracting increasing attention since it can train neural networks faster than the existing adaptive optimization methods, such as Adam.
In this paper, we study how Muon can be utilized in federated learning.
We first show that straightforwardly using Muon as the local optimizer of FedAvg does not work since the LMO is a biased operator.
We then propose FedMuon, which can mitigate this issue and can converge to the stationary point.
We also analyze how solving the LMO approximately affects the convergence rate and find that, surprisingly, FedMuon can converge for any number of Newton-Schulz iterations, while it can converge faster as we solve the LMO more accurately.
Through experiments, we demonstrated that FedMuon can outperform the state-of-the-art federated learning methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies how to adapt the recently proposed Muon optimizer, an LMO-based method, to the federated learning setting. The authors show that directly applying Muon as a local optimizer fails to converge due to the inherent bias of the LMO operator. To address this, they propose FedMuon, which introduces bias correction via control variates similar to SCAFFOLD. The paper provides a detailed convergence analysis under both exact and approximate LMOs, showing that FedMuon converges for any number of Newton–Schulz iterations, and converges faster when the LMO is solved more accurately. Experiments on FashionMNIST and CIFAR-10 demonstrate superior performance in both homogeneous and heterogeneous settings.

### Strengths
1. The paper provides the first formal analysis of bias introduced by LMO in the federated setting and rigorously proves that straightforwardly applying Muon fails to converge.
2. The author establish the convergence under both exact and inexact LMOs, and clarifying how the number of Newton–Schulz iterations affects the rate.
3. The experiments align well with theoretical findings.

### Weaknesses
1. The algorithm section (Section 4) only offers an intuitive, SCAFFOLD-like rationale for bias mitigation, but lacks formal theoretical support. It would be helpful to include a short proof sketch or lemma explicitly showing which terms or steps contribute to bias reduction.
2. Appendix D includes an example to prove why LocalMuon does not converge, but it would be more convincing to show that FedMuon can indeed converge under the same problem setting.
3. While illustrating the bias issue with a divergence example is interesting, similar bias phenomena commonly appear in other contexts such as bilevel optimization. A typical and simple approach in those settings is to communicate the momentum across clients in LocalMuon. However, the paper lacks discussion on why the authors chose not to adopt such straightforward momentum communication and instead opted for a control-variate-based correction mechanism.
4. There is no discussion of heterogeneous settings in the theoretical section. Given that the experiments include heterogeneous data distributions, the theoretical analysis should also address how FedMuon behaves under such non-IID conditions.
5. The empirical validation is confined to two small-scale vision datasets (FashionMNIST, CIFAR-10). The empirical validation should include large-scale or non-vision tasks to better demonstrate how FedMuon scales to realistic federated learning workloads or LLM fine-tuning scenarios.
6. Figure 2 provides an example of how the number of Newton–Schulz iterations $T$ affects performance. However, prior works such as [1] and [2] commonly fix $T=5$, and [3] shows that increasing beyond 5 yields no performance gain but adds unnecessary wall-clock time. Thus, the absence of $T\geq5$ in Figure 2 makes the empirical validation less convincing.

[1] Jordan et al., Muon: An optimizer for hidden layers in neural networks, 2024.

[2] Refael et al., Sumo: Subspace-aware moment-orthogonalization for accelerating memory-efficient llm training, 2025.

[3] Semenov et al., Benchmarking Optimizers for Large Language Model Pretraining, 2025.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FedMuon, a federated optimization algorithm that extends the recently proposed Muon optimizer—an LMO (Linear Minimization Oracle)-based method—into the federated learning setting. The authors identify that directly applying Muon in FedAvg (termed LocalMuon) fails to converge due to the bias of the LMO operator. To mitigate this, FedMuon introduces a bias-correction mechanism inspired by SCAFFOLD, along with a theoretical convergence analysis that also accounts for inexact LMO computations via the Newton–Schulz iteration. The authors provide both theoretical results and empirical validation on benchmark datasets (FashionMNIST, CIFAR-10) demonstrating FedMuon’s improved convergence and accuracy over existing methods such as FedAvg, FedAdam, and SCAFFOLD.

### Strengths
1.Novel and technically solid contribution.
The paper identifies a subtle but important issue (bias of the LMO in federated settings) and provides a theoretically justified correction method. The proofs (Theorem 1–3) are rigorous and clearly stated.

2.Theoretical generality.
The analysis handles both exact and inexact LMOs, showing convergence under any number of Newton–Schulz iterations—a strong and non-trivial result rarely seen in this line of work.

3.Experimental validation.
The experiments convincingly show FedMuon outperforming strong baselines under both homogeneous and heterogeneous data distributions. The empirical trends match the theoretical predictions (faster convergence as LMO accuracy increases).

4.Bridging two active research areas.
The paper effectively connects recent LMO-based optimization (Muon) with federated learning theory, which is timely and relevant for distributed LLM optimization

### Weaknesses
1.Limited experimental diversity.
The experiments are conducted only on small-scale benchmarks (FashionMNIST and CIFAR-10). Demonstrating results on larger or more realistic FL setups (e.g., cross-device or NLP tasks) would strengthen the empirical claims. ViT model or roberta-base model，imagenet,CIFAR100 data.

2.The proposed method simply combines the MUON and SCAFFOLD algorithms, and its communication cost is about twice that of the original approach.

3.The experimental results do not show a significant advantage.

4.The theoretical analysis of the paper is solid, but there is still room for improvement in the writing and experimental sections.

### Questions
1. Can the communication cost be further reduced?

2.Is the method still effective for training and fine-tuning large models?

3.Can the code be open-sourced?

4.The algorithm’s line 8 is written differently from the MUON optimizer — why is it designed this way?

5.Is the number of local iterations too small（K=5） ?

### Soundness
3

### Presentation
2

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
This paper focuses on improving federated learning optimization by addressing the bias issues when applying the Muon optimizer, which uses Linear Minimization Oracle (LMO), in the FedAvg framework. The authors show that directly using Muon within FedAvg, referred to as LocalMuon, fails to converge due to the bias introduced by the LMO. To overcome this, they introduce FEDMUON, an optimization algorithm that incorporates a bias-correction mechanism inspired by SCAFFOLD, ensuring convergence. The theoretical analysis includes the convergence rate under inexact LMO computations, handled via Newton-Schulz iterations. Experiments on two benchmark datasets, FashionMNIST and CIFAR-10, demonstrate that FEDMUON outperforms several existing methods, including FedAvg, FedAvg (Adam), SCAFFOLD, and SCAFFOLD (Adam).

### Strengths
$\cdot$ The paper is well-motivated. Muon can outperform traditional adaptive optimizers like Adam in centralized settings, but its direct application to federated learning, specifically in FedAvg, fails to converge due to bias introduced by the LMO.

$\cdot$ The use of Newton-Schulz iterations to approximate the Linear Minimization Oracle (LMO) in FEDMUON is a key innovation, as it enables efficient bias correction while maintaining strong convergence guarantees even with inexact LMO solutions.

$\cdot$ The experimental results in the paper are highly consistent with the theoretical analysis, supporting the convergence guarantees of FEDMUON derived in the theory.

### Weaknesses
$\cdot$ It seems that the authors just incorporate the existing Muon to SCAFFOLD framework, which is just incremental. The authors should clarify their original contribution.

$\cdot$ The experiments are conducted on small datasets and simple models. Since optimizers like Adam perform better with larger models, comparisons on more complex models like Transformers would provide a better evaluation of FEDMUON's scalability and effectiveness.

$\cdot$ It would be better to report the time cost or communication cost of FEDMUON and baseline methods to empirically demonstrate that FEDMUON's advantages hold in terms of efficiency as well as performance.

### Questions
$\cdot$ Can FEDMUON be effective on larger models, such as transformers, and not just small-scale datasets and simple models?

$\cdot$ Besides SCAFFOLD from 2020, can the authors compare FEDMUON with more advanced federated learning algorithms？

$\cdot$ Could the authors report the communication or time costs of FEDMUON and baseline methods to demonstrate its efficiency advantages?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes FedMuon, an algorithm improving usage of Muon in federated learning. They propose Scaffold-like modification in Muon's momentum term before LMO and observe improvement in stability. They also show some theoretical analysis in spectrum norm.

### Strengths
The writing is clear and easy to follow.

### Weaknesses
I'm not convinced by the paper's solidity and novelty.
1. The analysis in Section 3 regarding Muon’s failure to converge in FL seems rather trivial. Similar reasoning could apply to any momentum-based optimization algorithm. This claim should be supported by experimental evidence. In fact, based on the only experiments on LocalMuon shown in Figure 1, I don’t observe significant convergence issues. While there are some oscillations, I believe standard techniques such as learning rate or hyperparameter tuning could easily address them.

2. The modification is just including Scaffold-like updates in Muon, which seems quite limited.

3. The theoretical analysis is standard by combing Muon and Scaffold and classical SGD together. 

4. It’s already 2025, yet the experiments are limited to FashionMNIST and CIFAR-10 under very restricted settings, with only two baselines (FedAvg and SCAFFOLD). This level of evaluation would have been insufficient even several years ago. Moreover, the final accuracy shows almost no improvement over the baselines.

### Questions
What weight decay term was used in the experiments? This factor has a significant impact on the results, and certain related techniques (e.g., [1]) can help stabilize training.

[1] FedNAR: Federated Optimization with Normalized Annealing Regularization

### Soundness
1

### Presentation
2

### Contribution
1
