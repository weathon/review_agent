# Federated Learning with Profile Mapping under Distribution Shifts and Drifts

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Federated Learning (FL) enables decentralized model training across clients without sharing raw data, but its performance degrades under real-world data heterogeneity. Existing methods often fail to address distribution shift across clients and distribution drift over time, or they rely on unrealistic assumptions such as known number of client clusters and data heterogeneity types, which limits their generalizability.   We introduce **Feroma**, a novel FL framework that explicitly handles both distribution shift and drift without relying on client or cluster identity. **Feroma** builds on client distribution profiles—compact, privacy-preserving representations of local data—that guide model aggregation and test-time model assignment through adaptive similarity-based weighting. This design allows **Feroma** to dynamically select aggregation strategies during training, ranging from clustered to personalized, and deploy suitable models to unseen, and unlabeled test clients without retraining, online adaptation, or prior knowledge on clients' data. Extensive experiments show that compared to 10 state-of-the-art methods, **Feroma** improves performance and stability under dynamic data heterogeneity conditions—an average accuracy gain of up to 12 percentage points over the best baselines across 6 benchmarks—while maintaining computational and communication overhead comparable to FedAvg. These results highlight that distribution-profile-based aggregation offers a practical path toward robust FL under both data distribution shifts and drifts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents FEROMA, a FL framework that attempts to handle both distribution shift and drift without relying on client or cluster identity. The claimed contributions include
* a model aggregation strategy that adapts to client distribution profiles and can be extended to test-time adaptation without retraining
* a framework that works under both distribution shift and drift with minimal communication and computation  overhead, which is comparable to FedAvg
* theoretical bounds and empirical  measurements to validate the framework' practicality and privacy-preserving feature

### Strengths
* To my best knowledge, this is the first work that explicitly designed to address both distribution shift and distribution drift, during training as well as test time. [Disclaimer: (1) I have limited knowledge on the relevant literatures in the FL under distribution/concept drift. (2) Though the methods can be used for test-time adaptation, it's effectiveness is not fully justified.]

* The presentation is mostly clear and easy to follow and the visualizations help to understand the proposed method. 

* The experiments and ablation are comprehensive in studying distribution shift and drift accross different datasets.

### Weaknesses
1. Usages of math notations should be correct and consistent. They should be used to deliver clear messages and supplement natural languages, instead of further creating confusion. Examples of the abuse of notations and incorrect usages can be found in minor comments, which I can infer the messages. However, there are cases where the inconsistent (or incorrect) usage of math notations making the understanding of the key algorithm designs, hence the contribution, difficult. For example, the notation $(x^{(k)}, y^{(k)})$ is first introduced in line 123, which is defined as the local dataset and said to be **fixed**. Later, in Definition 3.1, $(x^{(k)}, y^{(k)})$ is then re-defined as $(x^{(k)}, y^{(k)})\sim P(X_t^{(k)}, Y_t^{(k)})$, making the $(x^{(k)}, y^{(k)})$ a random matrix, hence making $d_t^{(k)}$ a random vector. One may argue that $(x^{(k)}, y^{(k)})$ is deterministic and the randomness of $d_t^{(k)}$ comes from the fact that the $\phi_\psi$ is a random mapping. However, that's not the point. The point is that the readers should not guess what's the authors' actual messages. This further bothers me when attempting to understand the (R1) and (R3) right after the Definition 3.1. In (R3), it clearly states that $d_t^{(k)}$ is a "random variable" (actually a random vector), however in (R1), the way the equation is presented makes $d_{t_1}^{(k_1)}$ and $d_{t_2}^{(k_2)}$ look deterministic. Given $d_t^{(k)}$ is a "random variable", it only makes sense to me when the inequality is read as "it holds with probability 1" or "it holds under the expectation with respect to distribution ...". Since this part is not clear to me, I did NOT check any mathematical proofs.

2. While understanding the page limitation, I still think it's not appropriate to omit the results on the test-time performance in the main paper and to make claims on the performance of FEROMA based on its performance on MNIST(table 2). For example, in table 2, the proposed method can easily show double digits gain over other methods nearly across the boards. However, in other tables the gain may not be significant or some baseline methods may have better performances, e.g., Table 14 (MNITS, APFL wins) , Table 18 (Cifar-10, results are mixed), and Table 24 (Cifar-100, FEDRC and FEDEM wins under low non-iid level). In contrast, Table 1 is more compelling to support some claims. It would be better if the authors could clarify what cases are used to compute the average numbers. Simply mentioning "under varying drifting frequency, non-IID types and levels" is not enough. 


3. Some parameters on the experiment setups are not justified or not mentioned. 
	* Unjustified parameters: For example, the communication round is set to 20. This might be okay for datasets like MNIST since it's relatively easy, hence requiring less round to get decent performances. But for relative challenging datasets like cifar100, 20 rounds may not be sufficient. Will the observations and claims remain the same, when the communication rounds are set to 100 to allow all methods to converge? 

	* not mentioned: In the problem formulation, authors introduced $\mathcal{A}_{t}$ to denote active clients in round $t$. I would assume that only a fraction of client would participate in each round. But in the experiments, authors only mentioned "number of clients in 10, 20, 50, and 100". Does this mean all clients participate in all rounds?


I am willing to raise the score if my concerns are properly addressed.

### Questions
1. I don't understand how Global FL fallback works in Figure 3. But I guess, in the cases, $\bar w_{t}^{2,j}$ is a 0-vector. But why $\bar w_{t}^{k,j}\approx 1/|A_{t-1}|$ instead of $\bar w_{t}^{k,j}=1/|A_{t-1}|$?

2. Why there are N/As in tables for different methods under different settings, for example (ATP, CheXpert), in Table 1? There are no explanations. 

3. Will the observation and claims still be similar if the communication round is set to 100 or even 200? Do all clients participate in all rounds?

4. How the FEROMA compare to other test-time methods under the testing scenario mentioned in appendix D?

5. Can the PCA map $g$ be computed on the server side? Since the random seed is fixed and the ${m^{-}, m^{+}}$ are the same for all clients, I don't see why the computation of $g$ needs to be done on the client side.


**Minor Comments**
1. The difference between solid and transparent blue/red lines are not explained.
2. Equation (1) abuse the notations of weights to be optimize and the optimal solution. Similarly abuse use of notations also happen in R(1) when referring to the distribution over the dataset and $\mathcal{A}$ to denote active clients and measurable set for DP definition. matrix of latent vectors $s^(k)_t$ and the sample size $s^{(k)}$.
3. The notation $d_t'^{(k)} \subseteq d_t^{(k)}$ does not make sense to me. Maybe the authors intend to say $d_t'^{(k)}$ is a subvector of $d_t^{(k)}$.
4. Uncommon terms "Lipschitz-equivalent" is not properly defined. 
5. Figure 3. The weights for $\{d_{t}^{(i)}\}_{i=3}^{6}$ are all non-zero, which does not match the text "...exactly one weight is nonzero,...".
6. Some necessary details are deferred to the the Appendices, and one needs to jump back and forth to locate the details.

### Soundness
2

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
3

### Summary
Existing federated learning (FL) methods often fail to address distribution shift across clients and distribution drift over time, or they rely on strong assumptions such as known number of client clusters and data heterogeneity types, which limits their generalizability.
This paper aims to develop a new FL solution that can handle distribution shift and drift without relying on client or cluster identity.
This method builds on client distribution profiles that guide model aggregation and test-time model assignment through adaptive similarity-based weighting.

### Strengths
The proposed method introduces a unified approach that can handle both distribution shift and distribution drift. It covers scenarios that previous FL methods treat separately. This generalization is meaningful and provides a comprehensive theoretical foundation for dynamic non-IID conditions. Further, the way that this paper designed Distribution- Profile Extractor (DPE) with explicit differential privacy guarantees, bounded stochasticity, and theoretical validation supports a strong methodological foundation. 
The experiments utilize six, various non-IID scenarios, and drift frequencies, and large-scale client scenarios.

### Weaknesses
This reviewer has some concerns about weaknesses. 
The effectiveness of proposed method depends on the quality of the DPE. However, DPE itself requires a pretrained model. The manuscript even states “the effectiveness of FEROMA similarly depends on the quality of its DPE, which must generate reliable representations of client data distributions”.
 Poor representation quality can undermine profile fidelity and mapping accuracy, and ultimately make the method less reliable in low-resource scenarios.
The paper does not clearly quantify the robustness of proposed method against unseen or out-of-support distributions.
While the manuscript states the following, it does not clearly provide theoretical analysis of convergence guarantees or sensitivity of performance to hyperparameters:   “FEROMA dynamically selects the best aggregation strategy—ranging from clustered to personalized or global—based on client distribution profiles, and naturally extends to test-time adaptation without retraining.”,

### Questions
- Can the authors revise the paper to enhance the FEROMA’s dynamic aggregation using more solid theoretical analysis?
- Would it be possible to make the FEROMA’s distribution in a way to be self-adaptive to evolving latent spaces?

### Soundness
3

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
This paper introduces FEROMA, a new Federated Learning (FL) framework designed to handle both distribution shift (heterogeneity across clients) and distribution drift (temporal heterogeneity within clients). Instead of relying on client or cluster identities, FEROMA constructs distribution profiles, including compact and differentially private summaries of local data, to guide model aggregation and test-time model assignment. FEROMA adaptively transitions between clustered, personalized, and global aggregation based on profile similarity, allowing it to generalize to unseen clients without retraining. Extensive experiments across six benchmarks (MNIST, FMNIST, CIFAR-10/100, CheXpert, and Office-Home) show up to 12% accuracy gain over 10 baselines with comparable overhead to FedAvg.

### Strengths
+ This paper contains extensive experimental evaluation. The paper includes thorough comparisons with 10 strong baselines across six datasets (including real-world ones like CheXpert and Office-Home). Performance gains are consistent (e.g., +14.1pp on MNIST, +10.3pp on CIFAR-10, Table 1). Scalability experiments up to 100 clients and drift-frequency ablations are convincing.

+ .The paper is well-structured and visually clear, with informative figures (e.g., Figures 1–3) and theoretical justification of the extractor’s properties (R1–R5).

+ The efficiency and practicality of FEROMA are good. FEROMA achieves low computation and communication overhead (profile size ≤ 0.35 % of model parameters). From the idea perspective, the adaptive similarity-based weighting and automatic aggregation mode selection are conceptually neat and practically beneficial.

### Weaknesses
- Writing quality is high, though a few sections are dense and could benefit from intuitive explanations or examples (e.g., DPE stochasticity (R3)). The paper includes an extensive set of experimental results across multiple datasets, baselines, and drift/shift scenarios, which demonstrates strong empirical effort and thorough evaluation. However, the presentation could be improved for readability. Currently, the dense amount of numerical results and tables makes it difficult for readers to extract the main insights quickly. I encourage the authors to streamline the narrative by emphasizing key findings in the main text (e.g., highlighting representative cases or summarizing trends) and moving secondary details to the appendix. Adding clearer transition sentences and summary paragraphs after major result sections would also enhance flow and help readers better connect the experiments to the paper’s main claims.

### Questions
- The paper claims computational and communication costs comparable to FedAvg. Could the authors quantify this in terms of total transmitted bytes or runtime for larger models (e.g., ResNet or Transformer architectures)?
- The paper includes many experimental tables and dense numerical results. Could the authors consider summarizing key findings in more digestible form? A short “Key Observations” paragraph at Section 4 might help readers connect results back to FEROMA’s core design principles.
- Does FEROMA automatically adapt this threshold during training, or is it fixed?

### Soundness
3

### Presentation
3

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
This paper proposes FEROMA, a FL framework designed to tackle both distribution shifts across clients and distribution drifts within clients over time.  FEROMA introduces privacy-preserving distribution profiles, a lightweight representations that capture each client's data distribution. Through these profiles, FEROMA adaptively maps client data to previous distributions, guiding both aggregation during training and dynamic, test-time model selection for unseen or unlabeled clients. Experiments across various benchmarks and real-world datasets show improvements.

### Strengths
1. FEROMA introduces a differentially private distribution profile for each client.
2. The method address diverse FL scenarios in a single framework.

### Weaknesses
1. This paper lacks some necessary to compare and discuss in the related work and experiments, e.g., [R1-R3]. Recent alternative approaches for handling distribution shifts and drifts in FL are omitted [R4-R5].
2. How would the DPE behave under significant violations of its underlying assumptions—e.g., highly multi-modal, long-tailed, or high-dimensional client feature spaces? Is there any empirical or theoretical work supporting extension of the current implementation to natural language or time-series data?
3. How robust is FEROMA to adversarial profile manipulation or dishonest client participation? If malicious clients attempt to game profile similarity, what is the effect on model assignment and privacy?
4. The DP mechanism claims negligible performance reduction at $\varepsilon=10$, but Table 9 shows nontrivial drops for stricter budgets. The risk model and privacy guarantees need a more critical security-oriented discussion. 

References:

[R1] Tan Y, Long G, Liu L, et al. Fedproto: Federated prototype learning across heterogeneous clients[C]//Proceedings of the AAAI conference on artificial intelligence. 2022, 36(8): 8432-8440.

[R2]Yang Z, Zhang Y, Li C, et al. FedGPS: Statistical Rectification Against Data Heterogeneity in Federated Learning[J]. arXiv preprint arXiv:2510.20250, 2025.

[R3]Chen G, Niu S, Chen D, et al. Cross-device collaborative test-time adaptation[J]. Advances in Neural Information Processing Systems, 2024, 37: 122917-122951.

[R4] Liao X, Liu W, Zhou P, et al. Foogd: Federated collaboration for both out-of-distribution generalization and detection[J]. Advances in Neural Information Processing Systems, 2024, 37: 132908-132945.

[R5] Bhope R A, Jayaram K R, Venkateswaran P, et al. Shift Happens: Mixture of Experts based Continual Adaptation in Federated Learning[J]. arXiv preprint arXiv:2506.18789, 2025.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
