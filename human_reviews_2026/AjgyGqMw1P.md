# FedSAGD: federated learning with stable and accelerated client gradient descent

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 8, 4

## Abstract
Federated Learning (FL) has become a promising paradigm for distributed machine learning. However, FL often suffers from degraded generalization performance due to the inconsistency between local and global optimization objectives and client-side overfitting. In this paper, we introduce global-update stability as an analytical tool to study generalization error and derive the stability bounds of mainstream FL optimization algorithms under non-convex settings. Our analyses reveal how the number of global update steps, data heterogeneity, and update rules influence their stability. We observe that momentum-based FL acceleration methods do not improve stability.
To address this issue, we propose FedSAGD, a new FL algorithm that leverages the global momentum acceleration mechanism and a hybrid proximal term to enhance stability. This design ensures updates follow a globally consistent descent direction while retaining the benefits of acceleration.
Theoretical analysis shows that FedSAGD achieves an advanced stability upper bound of $\mathcal{O}(1-(1-\Gamma)^T) (0 < \Gamma < 1)$  and attains a convergence rate of $\mathcal{O}(\frac{1}{\sqrt{sKT}})$ on non-i.i.d. datasets in the non-convex settings. Extensive experiments on real-world datasets demonstrate that FedSAGD significantly outperforms multiple baseline methods under standard FL settings, achieving faster convergence and state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to address the deteriorating generalization performance of FL caused by heterogeneous local data distributions and client-side overfitting. It proposes a global-update stability framework to better analyze generalization errors in FL. Additionally, the paper develops a novel FL algorithm named FedSAGD and demonstrates that this method can achieve superior stability and generalization. Finally, extensive empirical studies are conducted to verify that FedSAGD outperforms baseline methods in terms of faster convergence and better generalization.

### Strengths
The paper introduces and formalizes the concept of Global-Update Stability in FL, which is highly relevant for the practical setting of partial client participation.

### Weaknesses
1. The novelty is somewhat incremental. It appears to be a straightforward extension of traditional sample-level stability analysis to the client-level setting. 
2. The form of the generalization error bound is uncommon — the effect of sample size is not clearly provided in the bound.
3. The notation and writing in the problem setup are unconventional and need improvement — for example,  key quantities (loss function, empirical and population risks) are needed to define more clearly.

### Questions
1. Please explain the meaning of the symbols $\gamma$, $s$, $K$, and $T$ in the abstract, so that readers unfamiliar with FL can immediately understand them.
2. In the *Related Work* section, I suggest splitting the discussion of "Generalization and stability" into two sub-sections:
   - “Generalization in centralized learning” — introduce the traditional uniform-convergence framework, algorithm-dependent generalization (e.g., algorithmic stability, PAC-Bayes), deep-learning generalization theory, out-of-distribution/domain generalization theory, etc.
   - “Generalization in federated learning” — discuss algorithmic stability and PAC-Bayes / information‐theoretic generalization for FL, federated domain generalization, unseen‐client participation scenarios, etc.

   This clearer separation will help readers situate your contributions more precisely.
3. The *Problem Formulation* part is somewhat non-standard. I recommend the following structure: first define the loss function (on a single sample), then define the local empirical risk (on the local dataset) based on that loss, then define the local population risk (on the local data distribution) which is used for the generalization analysis. I believe that explicitly define the local loss for a sample and the risks will improve the clarity of this paper.
4. Regarding Assumption 3.1: In FL contexts it is common to assume that the variance of the local stochastic gradient (when sampling mini-batches from the local training set) is bounded. More precisely, the bounded local variance refers to the variance between the mini-batch‐SGD gradient and the full local empirical risk gradient (not between the SGD gradient and the true expected risk gradient w.r.t. local data distribution). 
5. Definition 3.4: What exactly does $A(\mathcal{S}^{i},\xi_j)$ denote? The notation is unclear—please provide a precise definition.
6. The generalization analysis in this paper appears unconventional. I suggest decomposing Equation (2) further to introduce the empirical risk explicitly, which will help expose how the sample size influences the generalization bound.
7. On the theoretical novelty/challenge: The concept of *Global-update stability* seems to mirror sample-level stability analysis by replacing “samples” with “clients”. Please clarify how this extension is non-trivial and what new technical difficulties you overcome.
8. Could the authors provide an intuitive explanation for *why* the proposed hybrid proximal term improves stability more than the standard FedProx regularization? A more intuitive commentary would strengthen the presentation.
9. I recommend presenting the generalization bound of your algorithm as a clearly stated theorem (rather than embedding it in the text). This helps readers locate and understand the key theoretical result.
10. For the experiments:
    - It would be beneficial to evaluate the robustness of the proposed method under feature skew (not only label skew), since real-world FL often suffers from heterogeneity in feature distributions.
    - Please report results averaged over multiple runs (e.g., mean ± standard error or provide error bars) to show statistical reliability of improvements.

### Soundness
2

### Presentation
3

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
This paper stuided federated learning and proposed a new federated optimization algroithm. The authors cliamed that FL suffers from degraded generalization performance due to the inconsistency between local and global optimization objectives and client-side overfitting. They introduced global-update stability as an analytical tool to study generalization error and derive the stability bounds of mainstream FL optimization algorithms under non-convex settings.

### Strengths
The authors performed extensive analyses, covering both their method and existing baselines. The experimental section includes a wide range of simulation results demonstrating the effectiveness of the methods.

### Weaknesses
1. This paper addresses a well-studied problem, but I did not observe a clear breakthrough over existing works. In fact, some of the results appear weaker than those of prior studies. For instance, the theoretical analysis in this paper relies on a **bounded heterogeneity assumption**, whereas existing methods such as **FedDyn** and **SCAFFOLD** do not require this restriction. Moreover, recent advances in **momentum-based optimization** [R1, R2] have established convergence guarantees **without assuming bounded data heterogeneity**. This raises concerns about the originality and strength of the theoretical contribution.


2. The authors claim that their work addresses the stability issue under partial client participation. However, the presented analysis is not convincing. Specifically, the convergence proof is established under **uniform random client participation** and measures convergence using the **gradient norm**, while the stability analysis switches to the **parameter distance** metric without justification. For non-convex problems, existing algorithms have already demonstrated convergence under uniform random participation, so it is unclear why this particular metric is introduced or necessary.

Additionally, the result presented in **Line 1468** appears counterintuitive. It is unclear how one can meaningfully characterize the parameter property in the context of **non-convex optimization**. Furthermore, note that $\Phi$ could be unbounded (infinite) under the definition in **Line 1002**, making the derivations there mathematically questionable and potentially invalid.

[R1] Cheng, Ziheng, et al. "Momentum benefits non-iid federated learning simply and provably." arXiv preprint arXiv:2306.16504 (2023).

[R2] Cheng, Ziheng, and Margalit Glasgow. "Convergence of Distributed Adaptive Optimization with Local Updates." arXiv preprint arXiv:2409.13155 (2024).

### Questions
See the weakness

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes FedSAGD, a FL algorithm that integrates global momentum and a hybrid proximal term to enhance both convergence and stability. By formalizing a new concept of global-update stability, the authors connect generalization guarantees to the sensitivity of global updates under partial participation. They provide rigorous theoretical proofs showing improved stability convergence, and validate the method through experiments on common FL benchmarks.

### Strengths
1. The paper introduces the novel concept of global-update stability and leverages it to design the FedSAGD algorithm, providing deep theoretical insights and improved generalization guarantees in federated learning, especially under client heterogeneity.

2. This provides a principled theoretical foundation linking generalization ability to stability, distinguishing it from previous works that focused primarily on convergence or variance reduction.

### Weaknesses
1. Sensitivity and Ablation Analysis Could Be Deeper: Although the paper examines parameter sensitivity for β, λ, and μ, it lacks detailed ablations isolating the effects of each component (momentum vs. proximal term). It’s unclear how much each part independently contributes to stability or convergence gains.

### Questions
1. Methods like SCAFFOLD also improves global stability with the control variates, and they do not require tuning on proximal terms. Would you think such design can also be incorporated to FedSAGD for improved results?

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
The paper starts from the observation that existing momentum-based acceleration methods in FL don’t necessarily improve stability. Building on this, it proposes FedSAGD, a new optimization framework designed to make FL training more stable under partial participation and heterogeneous data. The paper proposes a novel notion called global update stability,  which measures how sensitive the global model is to variations in the participating client set. Theoretical results show the link between this stability and generalization error, and suggest that FedSAGD achieves optimal convergence rates for non-convex objectives. Empirical evaluations on several standard FL benchmarks (CIFAR-10/100, EMNIST-L, Shakespeare) show consistent improvements over popular baselines such as FedAvg, FedProx, and SCAFFOLD.

### Strengths
1. The introduction of global update stability as a metric to quantify the effect of client sampling randomness is original and potentially impactful. It provides a new way to reason about generalization in FL beyond the traditional sample-level stability.
2. The paper is well-written, with clear definitions, lemmas. Most proofs are easy to follow.

### Weaknesses
1. Theoretical assumptions are overly idealized.
2. Communication and computation costs are not reported.

3. Security and privacy concerns are ignored.
4. Experimental evaluation lacks ablation and statistical rigor.

### Questions
1. The convergence and stability analyses rely on a set of strong assumptions—smoothness, bounded gradients, limited client heterogeneity, synchronous updates, and i.i.d. client sampling. These are rarely satisfied in real-world FL systems, especially in cross-device settings. As a result, the theoretical results cover only a small fraction of realistic scenarios, and it’s not obvious how much they translate to practical improvements.

2. The paper does not report communication or computation costs, even though FedSAGD adds extra components like the hybrid proximal term and global momentum updates. Without such analysis, it’s hard to judge the real efficiency gains. Is there any comparison of communication and computation overhead with baselines such as FedProx or SCAFFOLD? It would be useful to know how much extra cost the hybrid proximal term adds per round.

3. There’s no empirical evidence showing that smaller global stability actually leads to better generalization. It would be helpful to see a quantitative analysis or correlation study to support this claim.

4. The method requires the server to keep an exponential moving average of historical gradients.

   Without secure aggregation, this can expose gradient information and potentially lead to data reconstruction risks, an issue that runs counter to FL’s privacy goals.

5. How sensitive is FedSAGD to the choice of the global momentum coefficient $\beta$ and proximal weights $(\mu, \lambda)$ under strongly non-IID settings?

   Would performance drop significantly if the heterogeneity parameter \alpha (in Dirichlet partitioning) were smaller?

6. How does the proposed global update stability correlate with the empirical generalization gap in experiments? Was this relationship measured or analyzed quantitatively?

### Soundness
2

### Presentation
3

### Contribution
2
