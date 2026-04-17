# SFedPO: Streaming Federated Learning with a Prediction Oracle under Temporal Shifts

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Federated Learning (FL) enables decentralized clients to collaboratively train a global model without sharing raw data. However, most existing FL frameworks assume that clients train on static local datasets collected in advance or that the data follows a fixed underlying distribution, which limits their applicability in dynamic environments where data evolves over time. A parallel line of research, online FL, removes all assumptions and adopts an adversarial perspective, but this approach is often overly pessimistic and neglects the structured, partially predictable nature of real-world data dynamics. To bridge this gap, we propose SFedPO, a streaming federated learning framework that incorporates a prediction oracle to capture the temporal evolution of client-side data distributions. We theoretically analyze the convergence bounds of SFedPO and develop two practical sampling strategies: a Distribution-guided Data Sampling (DDS) strategy that dynamically selects training data under limited storage by balancing historical reuse and distribution adaptation, and a Shift-aware Aggregation Weights (SAW) mechanism that modulates global aggregation based on client-specific sampling behaviors. We further establish robustness guarantees under prediction errors. Extensive experiments demonstrate that SFedPO effectively adapts to streaming scenarios with distribution shifts and significantly outperforms existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes SFedPO, a framework designed for FL under streaming data distributions by leveraging partial predictions about clients' data distribution shifts to guide both local data sampling and global aggregation. Based on a convergence upper bound, the authors develop two modules: Distribution-guided Data Sampling (DDS) and Shift-aware Aggregation Weights (SAW), which are claimed to jointly minimize the optimization error bound. Theoretical analysis provides convergence guarantees and robustness under prediction errors. Extensive experiments on multiple benchmarks show that SFedPO consistently improves test accuracy over existing FL baselines and can be plugged into various FL frameworks to further enhance performance.

### Strengths
1. This paper focuses on an important practical problem - federated learning with streaming and non-stationary data. 

2. The theoretical convergence upper bound drives the development of DDS and SAW modules. 

3. The experiment setup is reasonable, and the experiment results show the advantage of their proposed SFedPO.

### Weaknesses
W1. In lines 52-53 about the research question, the sudden transition to focusing on client sampling and server aggregation feels abrupt. From my personal perspective, the related background or explanation is not enough before coming up with the research question, which results in the research question not being convincing enough. 

W2. Theorem 1 has a lack of readability because it lacks explicit interpretation and further discussion, such as convergence rate analysis, the meaning of each term, the impact of some key factors (e.g., $\pi$), difference between your theoretical results and the result with static data distribution, and so on.  

W3. As for Equation (10), there are several issues: 

(W3.1) In lines 268–270, the authors state that “In realistic streaming environments, it is neither desirable nor feasible for a client to completely discard previously stored data or to entirely ignore new samples from a given state.” This viewpoint is not completely convincing because there are realistic situations where it is necessary to discard newly received samples with extremely high noise. Moreover, this assumption implicitly supports the validity of Equation (10) because it ensures that $\alpha_n$ cannot be zero. However, in practice, $\alpha_n$ can indeed be zero, in which case Equation (10) becomes undefined. 

(W3.2) The bounded gradient $G$ can be very large. As a very large $G$, $-a_1 d_m + b_1$ will be almost zero. Then, the score will not decrease with the state-specific heterogeneity bound $d_m$, which does not work as what they claimed. 

**Typo:** 
W4. Line 89: "a *date* evaluation metric" -> "a *data* evaluation metric"

### Questions
Q1. In the modularity experiment, does it use the same hyperparameter settings for baselines with and without SFedPO? 

Q2. Can the authors provide more explanation and discussion about the Theorems 1 and 2? Moreover, what is your theoretical contribution compared to the existing works? 

Q3: There are a lot of hyperparameters that need to be set in the experiment. Hence, I would like to know: how did the authors make sure that the ranges of $a_1$, $b_1$, $a_2$ and $b_2$ are reasonable? Moreover, when the heterogeneity score $s_n$ is large due to a large $G$, $a_2$ and $b_2$ may may not have a significant effect because they are dominated by $s_n$. Did the authors observe any related phenomena in your experiments?

See weaknesses above. I would adjusting my rating if the authors can address my concerns properly.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the problem of federated learning when clients receive stream of data. Due to having dynamic environment, the data distribution among clients may vary over time. The paper proposed adjusting data sampling for model training and obtained the convergence rate. Then to optimize the convergence, the paper proposes a client data sampling distribution and a server aggregation strategy. The proposed solution is based on the presence of reliable oracle that can predict states of clients. Experimental results show that the proposed algorithm achieves better results than other baselines.

### Strengths
- The problem of federated learning facing with stream of data is an interesting problem. The paper focuses on the cases where there is a possibility to predict the next state of clients.
- The paper provides convergence analysis for the proposed algorithm. Furthermore, the proposed data sampling and aggregation strategy technically sound which optimizes the convergence rate.
- The paper provides comprehensive the experimental results.

### Weaknesses
- The performance of the proposed algorithm depends on the quality of the predictive oracle model. However, an accurate oracle model may not always be available.
- The paper could benefit from including “Federated Learning for Data Streams” (Marfoq et al., 2023) as one of the baselines to strengthen the experimental study.
- A discussion on how the use of the oracle model can improve the convergence rate could also be added to enhance the paper.

### Questions
Based on my understanding, the paper focuses on scenarios where clients operate in dynamically changing environments, and the model training is adjusted accordingly to adapt to these changes. However, the paper does not appear to consider client heterogeneity. Could you elaborate on this aspect?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates federated learning under dynamic distribution shift scenarios. SFedPO avoids two extremes: traditional FL (assuming static distribution) and online FL (not using previous information at all)j by employing sampling strategy and client weighting mechanism. Both mechanisms are theoretically supported and experimental results demonstrate the superiority of SFedPO.

### Strengths
1. Research topic is timely and meaningful: federated learning using streaming data (dynamic data distribution).
2. Both ideas (sampling and client weighting) are theoretically supported.
3. Experimentally demonstrated performance improvement

### Weaknesses
1. Model architectures are too old: AlexNet and LeNet-5, meaning that updating only architectures can exceed the gain of the proposal. Utilizing at least ResNet is recommended.
2. Datasets are small and less challenging. Evaluating on CIFAR-100 is more common for federated learning. Dynamic environments can incur class changes in CIFAR-100, which makes challenges.

### Questions
Please address the weaknesses above.

### Soundness
3

### Presentation
3

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
The paper introduces SFedPO, a streaming federated learning (FL) framework designed to bridge the gap between conventional static FL and fully adversarial online FL settings. The authors assume that temporally evolving client data can be modeled as transitions among a finite set of latent states, each corresponding to a stationary distribution of data. By incorporating a prediction oracle that estimates the transition probabilities among these states, SFedPO dynamically adjusts local data sampling through a distribution-guided strategy (DDS) and adapts global aggregation via shift-aware weights (SAW). The authors provide convergence guarantees under this setup, demonstrate robustness to oracle prediction errors, and show that SFedPO consistently improves accuracy over baseline FL methods in simulated streaming settings.

### Strengths
The paper is logically structured, demonstrating a clear flow from problem formulation to theoretical analysis and practical implementation. The authors present a comprehensive convergence analysis and include robustness guarantees with respect to prediction errors. The framework is modular, meaning it can be integrated with multiple FL algorithms, and the experimental results demonstrate measurable improvements in accuracy across several baseline methods. The work also addresses an unfilled niche in FL literature by navigating between static and adversarial formulations with partial future knowledge.

### Weaknesses
The primary weakness lies in the practicality and realism of the assumptions. The use of a finite latent space with a known transition model and access to a prediction oracle may not reflect real-world data characteristics, and the experiments do not validate this setup beyond simulations. The partial-access experiment design may unfairly benefit SFedPO by clustering latent states in ways that other methods are not designed to exploit. Moreover, the feasibility of estimating parameters, such as the number of states or heterogeneity bounds, remains unclear, and no computational overhead analysis is presented for the proposed sampling and weighting schemes.

### Questions
What is the computational overhead of computing the sampling ratios $\alpha_{n,m}$ and aggregation weights pₙ during federated rounds, and how does this compare to the cost of local model training and communication?

Are there settings, either in terms of client availability, data dynamics, or oracle error, where SFedPO may underperform relative to classical FL methods such as FedAvg?

How should practitioners estimate or determine the number of latent states M if no prior structure is available in a real dataset?

In Equation (1), the authors propose updating the local distribution through a convex combination. Are there alternative techniques (e.g., kernel-based blending or Bayesian updating) that could better capture uncertainty or non-convex transitions across states?

### Soundness
2

### Presentation
3

### Contribution
2
