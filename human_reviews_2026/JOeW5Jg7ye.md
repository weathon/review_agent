# Mitigating Participation Imbalance Bias in Asynchronous Federated Learning Under Client Heterogeneity

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
In Asynchronous Federated Learning (AFL), the central server immediately updates the global model with each arriving client’s contribution. As a result, clients perform their local training on different model versions, causing information staleness (delay). In federated environments with non-IID local data distributions, this asynchronous pattern amplifies the adverse effect of client heterogeneity (due to different data distribution, local objectives, etc.), as faster clients contribute more frequent updates, biasing the global model. We term this phenomenon **heterogeneity amplification**. Our work provides a theoretical analysis that maps AFL design choices to their resulting error sources when heterogeneity amplification occurs. Guided by our analysis, we propose **ACE** ( **A**ll-**C**lient **E**ngagement AFL), which mitigates participation imbalance through immediate, non-buffered updates that use the latest information available from all clients. We also introduce a delay-aware variant, **ACED**, to balance client diversity against update staleness. Experiments on different models for different tasks across diverse heterogeneity and delay settings validate our analysis and demonstrate the robust performance of our approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the heterogeneity amplification problem in Asynchronous Federated Learning (AFL), where faster clients disproportionately influence the global model due to participation imbalance. The authors propose a theoretical framework that decomposes the MSE of the gradient estimation into three terms (i.e., sampling noise, bias, delay) and identify that the bias term arises from partial client participation. Building upon this insight, they propose ACE, an AFL algorithm that achieves full aggregation from all clients to eliminate bias error, and a practical delay-aware variant ACED that trades off between staleness and participation balance. Theoretical analysis shows that ACE’s convergence rate is independent of the data heterogeneity parameter, and experimental results show that ACE and ACED outperform FedBuff, CA2FL, and ASGD, particularly under high heterogeneity and large delays.

### Strengths
1. The paper provides a systematic MSE decomposition that unifies previous analyses of AFL and explicitly connects participation imbalance to bias error. 
2. The authors compare ACE against several strong baselines both theoretically and empirically.
3. I checked the proof of the main lemmas and theorem, and they seem reasonable.

### Weaknesses
W1. Although this paper is presented under the FL context, the proposed algorithms in fact operate more like distributed asynchronous SGD methods without local updates. Each client only computes a single gradient on the most recent global model and immediately uploads it, without performing any local training or multiple local steps. Therefore, the algorithmic structure is essentially equivalent to an asynchronous distributed optimization system rather than FL. 

W2. Full aggregation and per-client gradient caching may impose significant memory and communication costs on the server.

W3. For Table 1, ACE’s theoretical advantages rely on certain conditions, such as $m=n$ and $K=1$, instead of an intrinsic advantage of algorithmic design. When these conditions do not hold in reality, the theoretical advantages may not hold again.

### Questions
Q1. How does the server deal with memory and computation overhead when the number of clients $n$ is very large, given ACE requires caching all gradients? Can the authors estimate the costs? 

See weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work focuses on Asynchronous Federated Learning and proposes an algorithm for asynchronous federated optimization. The main claim is that the asynchronous pattern in FL amplifies the adverse effect of client heterogeneity (arising from differences in data distributions and local objectives). However, the theoretical analysis provided in the paper does not include any explicit characterization or quantification of data heterogeneity.

### Strengths
The authors study an important problem in federated learning. Asynchronous FL is indeed an interesting and practically relevant topic, as it reflects real-world deployment constraints in federated systems. 

The overall storyline and presentation of the paper are clear, which helps readers follow the main ideas and contributions effectively. 

The paper also provides extensive analysis to support its claims. However, some of these analyses are not rigorous upon closer inspection, which raises concerns about the soundness of the theoretical results.

### Weaknesses
1. This is a well-studied problem. I didn’t see any breakthrough made by this paper compared with existing works, e.g., [R1, R2].

2. The algorithm is basic. There are no additional procedures to handle data heterogeneity, so I am confused about how this algorithm can converge without a bounded data heterogeneity assumption. I checked the proof, which does not seem rigorous to me. For instance, in the characterization of the term C, you obtained a bound in Lines 1460–1463 that includes the norm of the gradient, but you omitted it in Equation (a6) (Line 1082). You can’t do this — the gradient can be unbounded. In fact, characterizing this gradient norm is fundamental in asynchronous federated algorithms. The issue here makes the entire theoretical result problematic.

3. The authors mention that the asynchronous pattern in FL amplifies the adverse effect of client heterogeneity (due to different data distributions and local objectives). However, I did not see this reflected in the theoretical results. I believe the reason is that the authors made mistakes in the proof, as discussed in the second point.

[R1] Tyurin, Alexander, and Peter Richtárik. "Optimal time complexities of parallel stochastic optimization methods under a fixed computation model." *Advances in Neural Information Processing Systems* 36 (2023): 16515–16577.

[R2] Maranjyan, Artavazd, Alexander Tyurin, and Peter Richtárik. "Ringmaster ASGD: The first Asynchronous SGD with optimal time complexity." *arXiv preprint* arXiv:2501.16168 (2025).

### Questions
No

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the challenge of participation imbalance in Asynchronous Federated Learning (AFL), where faster clients disproportionately influence the global model due to update staleness and client heterogeneity. The authors introduce the concept of heterogeneity amplification to describe this bias and develop a theoretical framework that decomposes the error in AFL updates into sampling noise, bias, and delay. To mitigate bias, they propose ACE (All-Client Engagement AFL), which aggregates updates from all clients using their latest available gradients, eliminating the bias term and improving convergence. They also present ACED, a practical variant that filters out excessively stale updates to balance diversity and freshness. Comparative analysis and experiments show that ACE and ACED outperform existing AFL methods in convergence speed and accuracy, especially under high heterogeneity and delay conditions.

### Strengths
+) Rigorous theoretical framework about AFL

+) The proposed method is novel to me

### Weaknesses
-) The paper does not deeply explore scalability of the proposed method for very large client populations or large models

### Questions
a) Could the proposed decomposition framework be extended to analyze other types of bias, such as adversarial updates?

b) How much overhead does gradient caching introduce?

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
4

### Summary
This paper addresses participation imbalance in Asynchronous Federated Learning (AFL), where faster clients contribute more frequent updates, biasing the global model toward their data, which the authors term as heterogeneity amplification. They propose a unified theoretical framework that decomposes AFL’s gradient error into noise, bias, and delay components, revealing that bias from partial client participation is the main source of heterogeneity amplification. To mitigate this, the authors introduce ACE (All-Client Engagement AFL), which performs all-client aggregation using cached gradients from every client, eliminating participation bias. A practical, delay-aware variant, ACED, includes only clients with sufficiently recent updates, balancing client diversity and staleness. Theoretical analysis shows ACE’s convergence rate does not require the bounded data heterogeneity assumption. Experiments on CIFAR-10 and NLP tasks demonstrate that ACE achieves faster and more stable convergence than FedBuff, CA²FL, and Delay-Adaptive ASGD, particularly under high heterogeneity and large delays.

### Strengths
**Clear theoretical framework:** The paper introduces a unified MSE-based decomposition (noise, bias, delay) that precisely attributes error sources in asynchronous FL and explains why participation imbalance leads to heterogeneity amplification. The proposed ACE and ACED algorithms directly emerge from this theory. It's also nice to see that the convergence does not require the bounded data heterogeneity assumption and Table 1 does a good job of comparing the impact of algorithmic elements on error terms. 

**Empirical validation:** Experiments across heterogeneity and delay regimes show consistent accuracy and convergence gains over prior AFL methods (FedBuff, CA²FL, Delay-Adaptive ASGD).

### Weaknesses
**Missing related work and novelty:** The idea of caching client updates and re-using them to eliminate partial participation variance has been well explored in FL literature (see [1], [2]). Firstly, it's problematic that the authors do not discuss these works at all. Secondly, the way I see it, the main idea of this work is extending this caching idea to the asynchronous setting. However, I do not feel this is significant novelty. As outlined in the strengths, I appreciate the clear theoretical framework motivating this approach, but the idea itself has been well-explored in the literature.

**Discussion on local steps:** The theoretical analysis gives the impression that doing more local steps (larger $K$) always worsens the error. However, this needs a more nuanced discussion. Yes, theoretically local steps increases the MSE error but it also helps decrease the initialization error faster (the term involving $F(w^{(0)}) - F^*$, see Theorem 2 in [3]). Furthermore, practically also we see benefits of using local steps. It seems that this work is restricted to the setting where $K=1$, which weakens the theoretical contribution.

**Limited experimental scope:** While results cover image and text tasks, the evaluation lacks large-scale, realistic FL benchmarks such as StackOverflow, Reddit or the federated split of Google Landmark-v2.


**References**

[1] Gu, Xinran, et al. "Fast federated learning in the presence of arbitrary device unavailability." Advances in Neural Information Processing Systems 34 (2021): 12052-12064.

[2] Jhunjhunwala, Divyansh, et al. "Fedvarp: Tackling the variance due to partial client participation in federated learning." Uncertainty in Artificial Intelligence. PMLR, 2022.

[3] Yang, Haibo, Minghong Fang, and Jia Liu. "Achieving linear speedup with partial worker participation in non-iid federated learning." arXiv preprint arXiv:2101.11203 (2021).

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2
