# DiffCPS: Diffusion Model based Constrained Policy Search for Offline Reinforcement Learning

- Decision: Reject
- Scores: 3, 8, 5

## Abstract
Constrained policy search (CPS) is a fundamental problem in offline reinforcement learning, which is generally solved by advantage weighted regression (AWR). However, previous methods may still encounter out-of-distribution actions due to the limited expressivity of Gaussian-based policies.  On the other hand, directly applying the state-of-the-art models with distribution expression capabilities (i.e., diffusion models) in the AWR framework is insufficient since AWR requires exact policy probability densities, which is intractable in diffusion models. In this paper, we propose a novel approach called $\textbf{Diffusion Model based Constrained Policy Search (DiffCPS)}$, which tackles the diffusion-based constrained policy search without resorting to AWR.  The theoretical analysis reveals our key insights by leveraging the action distribution of the diffusion model to eliminate the policy distribution constraint in the CPS and then utilizing the Evidence Lower Bound (ELBO) of diffusion-based policy to approximate the KL constraint. Consequently, DiffCPS admits the high expressivity of diffusion models while circumventing the cumbersome density calculation brought by AWR. Extensive experimental results based on the D4RL benchmark demonstrate the efficacy of our approach. We empirically show that DiffCPS achieves better or at least competitive performance compared to traditional AWR-based baselines as well as recent diffusion-based offline RL methods. Code will be made publicly available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors propose a new algorithm called DiffCPS for solving offline RL problems. They claim that DiffCPS can learn a diffusion-based policy avoiding the difficult density calculation brought by traditional AWR framework. They present theoretical justification for their approach and perform an empirical study to validate their proposed algorithms.

### Strengths
* The numerical experiments are well designed, with a detailed section of ablation study. Also, the toy example is intuitive and interesting.

### Weaknesses
* Corollary 3.1.1 and Theorem 3.2 are flawed. In the proof, the authors claim $J(\mu)$ is an affine function of $\mu$, which is **false**. In fact, $J(\mu)$ is **not** convex w.r.t. $\mu$ and the duality results can not hold. Consequently, the theoretical analysis presented in this paper is compromised, significantly undermining its contributions.

* The paper lacks clarity. Multiple mathematical objects are introduced without clear definitions. For example, $Q_\phi(s,a)$, $\epsilon_\theta(x_i,i)$, $f_\phi(y\mid x_i)$, $H(\cdot,\cdot)$, etc. . If the reader is not familiar with the literature on RL and diffusion models, she/he will definitely get confused by the undefined notations. Also, I think the authors should provide more explanations about the exact methods used to update $\mu$ and $\lambda$. It would be appreciated if the authors could give a complete and precise description of DiffCPSS. (If there are problems with page limits the authors can put it in the appendix.)

* The empirical performance of the proposed algorithm exhibits only marginal improvement when compared to the baselines.

### Questions
* The authors deploy diffusion-based policies in place of the traditional Gaussian policies to handle the multi-modal problem. I wonder whether there are choices other than diffusion models and naive Gaussians. For example, flow models can express a rich distribution class and have explicit densities. Is it possible to fit the flow models into the framework of AWR? 

* In the toy example, the authors claim that "SfBC incorrectly models the dataset as a circle instead of a noisy circle". I checked the image and did not find such a difference. Can the authors give an explanation?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors tackle a KL-constrained offline RL problem, where a RL policy is trained but additional constraints are put on the distribution shift between the original policy and the trained policy. The authors point out the weakness in using a unimodal Gaussian policy and propose to use a diffusion process as a policy parametrization. The authors show some desirable properties of using diffusion as a policy parametrization, and propose a primal-dual iteration algorithm to solve KL-constrained offline RL. The authors compare against baselines in D4RL and present ablation studies of the algorithm.

### Strengths
1. Overall, this is a good paper with strong theoretical and empirical results. 
2. The visualization for the problem of having unimodal policies is very intuitive, and motivates a richer class of policies that can model more multimodal distributions in offline RL. 
3. The ablation studies of the algorithm is extensive and the authors have done good research into best practices in training diffusion models.

### Weaknesses
1. The proposed baselines are entirely numerical and quantitative; while convincing, it would have been nice to see some qualitative behavior of DiffCPS compared to other baselines as well in order to strengthen the authors' claims. For instance, trajectory stitching is a popular example in offline RL where unimodal policies might possibly fail if done naively. In simple offline RL tasks such as the X data collection task in Diffuser [1], does DiffCPS succeed?

2. One interpretation of the authors' work is that when we use a generative model of the training data from the behavior policy, this has the effect of automatically constraining the distribution shift between the learned policy and the behavior policy. The authors are missing some relevant work in this direction in the context of model-based offline RL. For instance, [2] motivates a very similar objective with DiffCPS, where the cross-entropy term is penalized rather than constrained (with the difference that the distribution shift is on the state-action occupation measures rather than the action distribution in the model-based setting).

3. On a related note, offline RL also has model-free and model-based approaches, and the author's approach is model-free. The title might give the impression that this method is model-based, though I am aware that the authors' intention was to say it's based on a diffusion generative model. Maybe Diffusion based might be a better title?

[1] Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis", ICML 2022

[2] Suh et al., "Fighting Uncertainty with Gradients: Offline Reinforcement Learning with Diffusion Score Matching", CoRL 2023

### Questions
1. Have the authors considered using Augmented Lagrangian?
2. In Theorem 3.1, what is $d_{\pi_b(s)}$? Should this be the occupation measure of the behavior policy, which the authors previously denote using $\rho$? or the initial distribution of the behavior policy?
3. How specific are the authors' claims to diffusion models? For instance, if we had trained a denoising autoencoder model (assuming they can be trained well), would the theorems still hold?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies a constrained policy search in offline reinforcement learning. To increase the expressivity of Gaussian-based policies, the authors propose to use diffusion model to represent policy. The authors formulate a diffusion model based constrained policy optimization problem, and propose a constrained policy search algorithm. The authors also provide experiments to show the performance of this method.

### Strengths
- The paper is well organized, and the key idea is delivered. 

- The authors provide an example to show the expressivity limitation in standard advantage regression methods, which justifies the necessity of introducing diffusion models. 

- The authors use a popular diffusion model: DDPM to represent policy, and present a new constrained policy search method, which is intuitively simple and easy to implement. 

- Experimental results demonstrate comprable performance compared with state-of-the-art methods.

### Weaknesses
- The importance or motivation of propositions, theorems, and corollaries is not well explained. Most results can be more directly obtained using simple calculations that are known in diffusion model. 

- The use of diffusion model as policy in constrained policy search has been studied in offline RL. Due to the similarity, it is important to distinguish them in an explicit way. 

- The authors claim strong duality for Equation (12) according to the duality in the convex optimization. However, the constrained policy search is a non-convex problem. It is not justified if the strong duality still holds.  

- The main result is empirical. It is useful if the authors could provide performance analyses, which can strengthen the method with solid theoretical guarantees.

### Questions
- A large paragraph of this paper introduces known results. Can the authors highlight more new developments compared with existing methods?

- It is not clear to me the strong duality of Equation (12). Can the authors justify it? 

- How does the problem (18)-(19) can solve the original problem (12)? 

- Are there other diffusion models useful for constrained policy search? It is useful if the authors could discuss the generalization of this approach. 

- Training diffusion model can be inefficient. What are computational times for the methods in experiments?

- The examples in experiments are created in simulated environments. Any realistic offline dataset you can use to show performance of your algorithm?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
