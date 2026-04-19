# Enhancing Personal Decentralized Federated Learning through Model Decoupling

- Decision: Reject
- Scores: 6, 3, 5, 5

## Abstract
Personalized Federated Learning (FL) aims to produce many local personalized models rather than one global model to encounter an insurmountable problem -- data heterogeneity in real federated systems. However, almost all existing works have to face central communication burdens and the risk of disruption if the central server fails. Only limited efforts have been made without a central server but they still suffer from high local computation, catastrophic forgetting, and worse convergence due to the full model aggregation process. Therefore, in this paper, we propose a PFL framework through model decoupling called DFedMDC, which pursues robust communication and better model performance with a convergence guarantee. It personalizes the “right” components in the modern deep models by alternately updating the shared and personal parameters to train partially personalized models in a peer-to-peer manner. To further promote the shared parameters aggregation process, we propose DFedSMDC via integrating the local Sharpness Aware Minimization (SAM) optimizer to update the shared parameters. Specifically, it adds proper perturbation in the gradient direction to alleviate the shared model inconsistency across clients. Theoretically, we provide convergence analysis of both algorithms in the general non-convex setting with partial personalization and SAM optimizer for the shared model. We analyze the ill impact of the statistical heterogeneity $\delta^2$, the smoothness $L_u, L_v, L_{uv}, L_{vu}$ of loss functions, and communication topology ($1-\lambda$) on the convergence. Our experiments on several real-world data with various data partition settings demonstrate that (i) partial personalized training is more suitable for personalized decentralized FL, which results in state-of-the-art (SOTA) accuracy compared with the SOTA PFL baselines; (ii) the shared parameters with proper perturbation make partial personalized FL more suitable for decentralized training, where DFedSMDC achieves most competitive performance.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes a personalized federated learning (PFL) framework called DFEedMDC, which pursues robust communication and better model performance with a convergence guarantee. Besides, to promote the shared parameters aggregation process, the authors propose DFedSMDC via integrating the local Sharpness Aware Minimization (SAM) optimizer to update the shared parameters.

### Strengths
This work designs personalized local models and training schemes for decentralized federated learning. The authors present theoretical analyses for the convergence, which shows the negative influence of the statistical heterogeneity and the communication topology. Extensive experiments are conducted to evaluate the effectiveness of the proposed methods.

### Weaknesses
What do the "Grid" and "Exp" represent in Fig. 3? It would be easier for the readers to understand different communication topologies by visualizing them in the main test or in the Appendix.

In light of Theorem 1 and Theorem 2, the communication topology (i.e., the eigenvalue $\lambda$) has an impact on the DFedMDC and DFedSMDC methods. The reviewer suggests the authors report the $\lambda$ values of different communication topologies in Fig. 3 and discuss the influence of $\lambda$ on the test accuracy.

### Questions
The proposed DFedSMDC method, a variant of DFedMDC, achieves better performance by integrating the SAM optimizer into the local iteration update of shared parameters. The reviewer is curious if the incorporation of this optimizer could similarly enhance the performance of other baseline methods.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the personalized federated learning problem under fully decentralized setting. The framework of the considered personalized learning is the commonly used model decoupling with a globally shared model and personalized local models. DFedSMDC, an algorithm via integrating the local Sharpness Aware Minimization (SAM) optimizer to update the shared parameters, is proposed. Theoretical convergence results and numerical experimental results are both presented.

### Strengths
This paper studies the personalized federated learning problem under fully decentralized setting, and proposed DFedSMDC, an algorithm via integrating the local Sharpness Aware Minimization (SAM) optimizer to update the shared parameters.

### Weaknesses
1. The reviewer is quite doubt about the final results as shown in Theorem 1 and Theorem 2. I’ve checked the theoretical proof in the appendix and do not find the exact expressions for the final convergence results, but only the $\mathcal{O}$ expression. The first questionable part is that the right-hand side of Eqs. (3)-(4) will goes to 0 as the number of rounds $T$ goes to infinity, while in reality, this is not true for non-i.i.d scenarios. There will exists some constant terms related with heterogeneity that are irrelevant to $T$. Please explain this. 

2. The second part that may not be true in the theoretical results is that the convergence speed is monotonically related with the spectral gap $\lambda$. If this is true, it solves the challenging topology design problem of decentralized federated learning, since a fully-connected topology is the optimal topology according to the theoretical results in this paper. There is no discussion about this point in current manuscript and this leads to a doubtful result.  

3. Why is the convergence results not related with the number of workers? This is also a weird part. 

4. Why Theorem 1 is related with the cross Lipschitz constant $L_{vu}$, and Theorem 2 is related with $L_{vu}$? How about $L_{uv}$?

5. The results in Fig.3 are questionable according to the second comment. The reviewer is not sure if a fully-connected topology is the best.

6. What is the meaning of Fig. 4? Are multiple local epochs good or bad? How is it related with the theoretical results?

### Questions
See the weakness above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors present an innovative framework known as DFedMDC, which leverages model decoupling to address these issues and aims to provide robust communication and superior model performance while guaranteeing convergence. DFedMDC achieves this by personalizing the "right" components within modern deep models through alternate updates of shared and personal parameters, facilitating the training of partially personalized models in a peer-to-peer manner. To enhance the shared parameters aggregation process, the authors introduce DFedSMDC, which incorporates the local Sharpness Aware Minimization (SAM) optimizer to update shared parameters. SAM optimizer introduces proper perturbations in the gradient direction to mitigate inconsistencies in the shared model across clients.

The paper provides a thorough theoretical foundation, offering a convergence analysis of both algorithms in a general non-convex setting with partial personalization and SAM optimizer for the shared model.

### Strengths
1. The paper is well-written and exhibits a high degree of clarity, making it accessible and easy to comprehend.
2. The paper's strength is further underscored by its meticulous convergence analysis, enhancing its overall robustness.
3. The paper substantiates its claims with an exhaustive array of experimental results, effectively confirming the effectiveness of the proposed method.

### Weaknesses
1. A significant concern revolves around the novelty of the proposed method. The concept of model decoupling in personalized federated learning [1] and the application of Sharpness Aware Minimization (SAM) [2] to address model inconsistencies in decentralized federated learning have both been extensively explored in the literature. As such, the proposed method may appear to be a fusion of existing ideas (resembling an 'A+B' approach). It is essential for the authors to underscore their distinctive contributions in a more prominent manner.

2. In terms of experimental baselines, it is recommended that the authors include the most recent decentralized federated learning method ([2]) for a comprehensive comparison. This will enhance the paper's completeness and relevance in the context of the current state of the field.

3. Regarding the convergence analysis, it would be valuable to incorporate a discussion that compares the proposed method's convergence rate with the state-of-the-art (SOTA) approaches. 

[1] Exploiting Shared Representations for Personalized Federated Learning
[2] Improving the Model Consistency of Decentralized Federated Learning

### Questions
See weaknesses section above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes interesting methods DFedMDC and DFedSMDC for PFL, which simultaneously guarantee robust communication and better model performance with convergence guarantee via adopting decentralized partial model personalization based on model decoupling.

### Strengths
1. The study of personalized FL on decentralized FL is meaningful.
2. The experiments demonstrate that the proposed method is useful.

### Weaknesses
1. The proposed algorithm seems trivial and common in PFL. It seems its idea is the adoption of the method in DFL. Can you clarify what is the main novelty of this method?
2. Why introduce the SAM? It is unclear about the advantage of introducing this optimizer. Can you elaborate on it intuitively and theoretically?
3. In the theorem, why is it $V^{t+1}$, instead of $V^{t}$, and what does it mean?
4. The experiment results are a bit weird, in Table 1. Why do all baselines achieve better performance under larger heterogeneity? As I know, larger heterogeneity will usually lead to worse performance [1].
5. Regarding ``The test performance will get a great margin with the participation of clients decreasing’’: What will happen when the client number is less than 10, even 1? Does it mean no collaboration is the best?

[1]Karimireddy S P, Kale S, Mohri M, et al. Scaffold: Stochastic controlled averaging for federated learning[C]//International conference on machine learning. PMLR, 2020: 5132-5143.

Minors:

1.	It seems the hyperparameters of the proposed methods are finetuned (like $rho$ and local epoch for the personal part). Are the baselines’ results well finetuned? What's the used hyperparameter for baselines?
2.	What is the definition of $\sigma$ in Theorem 2?

### Questions
1. Could you give more explanation on Theorem 2? What is the difference/advantage compared with Theorem 1 as you introduce SAM into shared parameters?
2. Can you provide baseline results with more hyperparameter settings?
3. Could the authors provide more details about the experiment settings?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
