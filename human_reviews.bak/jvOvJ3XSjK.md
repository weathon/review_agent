# Predict-then-Optimize  via Learning to Optimize from Features

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 3

## Abstract
Many real-world decision processes are modeled by optimization problems whose defining parameters are unknown and must be inferred from observable data. 
The Predict-Then-Optimize framework uses machine learning models to predict unknown parameters of an optimization problem from features before solving. Recent works show that decision quality can be improved in this setting by solving and differentiating the optimization problem in the training loop, enabling end-to-end training with loss functions defined directly on the resulting decisions. However, this approach can be inefficient and requires handcrafted, problem-specific rules for backpropagation through the optimization step. This paper proposes an alternative approach, in which optimal solutions are learned directly from the observable features by predictive models. The approach is generic and based on a simple adaptation of the Learning-to-Optimize paradigm from which a rich variety of existing techniques can be employed. Experimental evaluations show the ability of several Learning-to-Optimize methods to provide efficient, accurate, and flexible solutions to an array of challenging Predict-Then-Optimize problems.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces several ways to address a predict-then-optimize problem and investigates how to adopt learning-to-optimize methods to it. Through this, the paper presents a way to address predict-then-optimize problems by using learning-to-optimize methods.

### Strengths
The paper has a clear presentation and is easy to follow. The figures are helpful in understanding the content, and the existing concepts in related works are introduced properly.

### Weaknesses
The analysis and investigation in the paper is reasonable, but seems trivial. There is no rigorous mathematical analysis on the use of learning-to-optimize methods for predict-then-optimize problems, and there are no novel algorithmic improvements or suggestions as well. In summary, the idea of the paper is simply applying existing learning-to-optimize to predict-then-optimize problems. This proposal seems too simple.

### Questions
1. The proposal suggests that existing learning-to-optimize methods can be applied to predict-then-optimize problems. This idea (i.e., the application of learning-to-optimize for predict-then-optimize problems) seems simple to think of. What is the contribution of the paper?
2. In a similar context, is there any further discussion? For example, a variant of learning-to-optimize for the idea or a tailored loss function for effective learning. It seems hard to claim the idea is a new framework based on the current paper.
3. The authors state that the distribution shift issue is caused by a pretrained optimization proxy. Thus, a proper solution to the issue should be a way to resolve the distribution shift issue while using a pretrained proxy. In this regard, simply training the prediction part and learning-to-optimize part jointly, as the paper suggests, does not seem to be a solution to that problem, because it means that pretrained proxies cannot be used.
4. The proposed approach involves prediction due to its predict-then-optimize nature. So, it would be difficult to generate a pretrained model and utilize it for different predict-then-optimize problems. Then, even if the proposed approach can achieve the short inference time once the learning-to-optimize model is trained, it suffers from the training cost because of getting various $(z,x^*)$ samples in the process of collecting datasets for training.

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the predict-then-optimize framework where an optimization problem with part of the parameters missing and a feature correlated to the missing parameters are given. The conventional way of solving predict-then-optimize problems is to learn a machine learning model to map from features to missing parameters based on a training dataset. Then the model can be used to predict missing parameters from unseen features to solve optimization problems accordingly. This solution is known as two-stage learning.

Recently, an end-to-end predict-then-optimize learning framework, e.g., smart predict-then-optimize, decision-focused learning, decision-aware learning, was proposed to learn the machine learning model end-to-end with optimization problems in the training loop. The idea is to make optimization differentiable so that the solution quality and regret can be used to backpropagate gradients to learn the machine learning models. This end-to-end predict-then-optimize (EPO) framework has achieved many successes in different applications, but also suffering from its limitation on differentiability and computation cost.

The authors instead propose to generalize from the idea of learning to optimize (LtO) to achieve end-to-end learning. The idea of learning to optimize is based on learning a machine learning model to solve an optimization problem. This resolves the limitation of differentiability of optimization problems because machine learning models such as neural networks are differentiable by construction. This approach also reduces the computation cost of backpropagation because backpropagation in neural networks is efficient. The authors propose to jointly learn the learning to optimize framework and the prediction problem to achieve end-to-end learning in the predict-then-optimize problem. This idea leads to their solution “learning to optimize from features (LtOF)”.

Lastly, the authors run experiments on three different domains with convex and non-convex objectives and different loss functions to compare their LtOF solution with LtO, EPO, and two-stage learning. Their results show that LtOF can achieve better performance than two-stage learning while EPO sometimes can be better but suffer from limitation of differentiability and computation cost.

### Strengths
The authors clearly specify the challenges in end-to-end predict-then-optimize framework and propose solutions to handle them. The proposed method is also generic and can be used in many applications.

### Weaknesses
**Robustness**: I don’t think the LtOF approach can handle the issue of distribution shift that is encountered by LtO. When a different distribution of features and optimization parameters is present, the authors do not address why this approach is more robust to distribution shift. In addition, LtOF can be impacted by distribution shift more because it learns both optimal solution and parameter predictive model jointly, while EPO assumes given optimization problem and only learns the parameter predictive model. Given the same amount of training data, LtOF needs to learn more parameters and I believe this will lead to poorer generalization unless you have sufficient data. 

**Training v.s. testing sets**: I didn’t see any explanations on whether the experimental results shown in the paper are from the training set or from the testing set. If it is from the training set, then the results are not surprised given that LtOF jointly learns how to solve optimization and the predictive model. LtOF has more parameters and thus can overfit to the training set more than other methods. LtOF also uses the regret or decision quality as the training objective. So it is not surprised that LtOF outperforms two-stage learning. Please clarify if testing set is used and how you verify generalizability.

[updated] I noticed in the appendix that you briefly mention it. Please emphasize it in the main paper and clarify how the test sets are constructed.

**Nonconvex EPO**: In fact, EPO can handle nonconvex optimization objectives by locally approximating nonconvex objectives around the local minimum as convex objectives due to local optimality. The (heuristic) differentiability of nonconvex objectives was also mentioned in Amos et al. 2017, Agrawal et al. 2019a, Perrault et al. 2020, and Wang et al. 2020. The authors should also compare with EPO on problems with nonconvex objectives. The experimental results shown in the paper show that EPO can indeed be better in portfolio optimization in Table 2, but the authors claim that EPO is not available for nonconvex tasks. I think adding EPO results for nonconvex versions would help justify the authors’ claim.

**Synthetic dataset**: all the datasets used in the paper are synthetic with generated features. It is not clear if the proposed method would work in real-world applications and still outperform other existing methods.



**References for nonconvex EPO**:
- Perrault, Andrew, et al. "End-to-end game-focused learning of adversary behavior in security games." Proceedings of the AAAI Conference on Artificial Intelligence. 2020.
- Wang, Kai, et al. "Scalable Game-Focused Learning of Adversary Models: Data-to-Decisions in Network Security Games." AAMAS. 2020.

### Questions
Please refer to the weaknesses section.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the predict-then-optimize framework. Predict-then-optimize framework is where some parameters of an optimization problem are unknown and need to be predicted based on some features. There are mainly two types of previous work mentioned in this paper. One is a two-stage approach which first predicts the parameters as a supervised learning problem and then solves the optimization problem with the predicted parameters. The other type is End-to-End Predict-Then-Optimize (EPO) which gets the gradient through the optimization problem with respect to the input features. The two-stage approach does not consider the downstream optimization problem in the supervised learning task as the loss function cannot match the regret of the final optimization problem. This regret represents the gaps of the objective of the optimization problem with predicted parameters and true parameters.

Existing work uses deep learning methods to predict the solution to an optimization problem, i.e., a mapping from the parameters of optimization problem to optimal decisions. This paper applies some methods of training proxy models into the Predict-then-Optimize framework. The advantage of using this model in PtO is that the NN representation of the optimization problem is differentiable and fast compared to solving the optimization problem itself. However, a naive proxy model suffers poor performance if predicting on unseen data. A proxy model can predict some decision which does not satisfy the constraints and a naive projection to feasible region approach may not have good performance. This paper uses three recent works of learning proxy model, Lagrangian Dual Learning (LD), Self-Supervised Primal-Dual Learning (PDL), and Deep Constraint Completion and Correction (DC3) to address the generalization and constraints violations issues. They apply these proxy models techniques to the predict-then-optimize framework to get the Learning to Optimize from Features (LtOF). They paper evaluate LtOF with the previous two-stage and EPO approaches.

### Strengths
1. The use of LtO techniques in the PtO setting and the simultaneous training of the LtO and the PtO model is novel.
2. LtOF method performs better than the EPO and two-stage approaches in the nonconvex QP problem and nonconvex AC-optimal power flow problems and is very fast.
3. There are substantial technical difficulties involved with the problems they select due to their difficult constraint regions. Not having an explicit optimization solve makes getting feasible solutions difficult, but their models do pretty well on this score.
4. The paper is clearly written.

### Weaknesses
1. The contribution is kind of slight. There really isn't much to do to get LtOF—or the way the paper is written makes it seem like this combination is relatively straightforward. It gives the impression that, if there is a difficulty, is in the empirical details around hyperparameters and getting the LtO to satisfy the constraints, but the authors do not claim a contribution in this area.
2. The experiments don't seem to compare to strong EPO methods (or in problems where EPO is at least better than two-stage). For two of the three problems, the EPO comparison is worse than 2-stage. This leaves the impression that experimental validation does not fully evaluate how strong LtO is relative to EPO (it comes across more that it is stronger than two-stage in cases where EPO cannot be applied).
3. It would be useful to see the dependence on the amount of training instances available AND to have info about the amount of training time that is required (if I haven't missed them, these timing results are not present—it seems that a major cost of PtOF is the training cost.)

### Questions
1. Hope this paper could detail more about some techniques used in LtoF methods to reduce the negative effect of constraints violations. The Appendix C and figure 7 describe a projection method to find a feasible solution based on an approximate optimization solution for the AC Load Flow model. One of the LtOF methods uses the Deep Constraint Completion and Correction (DC3). Is the projection method only used in Lagrangian Dual Learning (LD) and Self-Supervised Primal-Dual Learning (PDL) LtOF but not DC3 LtOF? Is it possible that the projection method always needs to hand-craft a specific function for proxy models of different optimization problems? Or can learning proxy models of very different optimization problems can utilize a general projection method for the constraints violation issue?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a way to extend "Learning to Optimize" (LtO) approaches to the predict-then-optimize framework. Broadly, it combines the prediction $z \to \hat{\zeta}$ and optimization $\hat{\zeta} \to x^\star$ steps (using LtO) and proposes Learning to Optimize from Features (LtOF). It evaluates LtOF against 2-stage on three domains (and against EPO for one domain) and shows improved performance and, in one case, lower computational cost.

### Strengths
- **The paper is mostly well-written.** I found it easy to follow the main arguments.

### Weaknesses
- **The paper is missing important related work.** The main motivation for this method seems to be that it's hard to differentiate through the optimization problem for arbitrary optimization problems. However, there has been work in the predict-then-optimize literature that addresses this problem exactly. Specifically, Shah et al. [1] learn ``decision losses'' that learn the mapping from $\hat{\zeta} \to regret(x^\star(\hat{\zeta}), \zeta)$ (as defined in eq. (3)) to get around the issue of having to learn differentiable surrogates (there have also been more recent follow-ups [2,3] that make this more efficient). Even more generally, because you do not consider combinatorial optimization problems but rather continuous optimization problems, you could just use numerical differentiation methods to find the exact derivatives $\frac{\partial regret}{\partial \hat{\zeta}}$. For reasonable values of $|z| = 30, 50$ (as in this paper) along with some clever optimization tricks (e.g., warm-starting, GPU-acceleration, etc.) this may not be doable. There has also been a paper [4] that says you can get away with very simple surrogate gradients as well. All in all, these are just the tip of the iceberg. This space is not nearly as unexplored as the related work in this paper suggests, and it is important to compare LtOF to these alternate approaches.

- **The experiments are poorly designed and implemented.** I have several concerns with the experiments in this paper:
  1. *Two out of three domains do not have reasonable baselines:* Nonconvex QP and AC-OPF compare only to two-stage and not any EPO alternatives. As this paper has noted, the fact that you can do better than two-stage is well-known at this point. The question is about whether you can do better than other reasonable EPO alternatives like [1].
  1. *No comparison to prediction + pre-trained LtO:* One simple baseline would be to compare to optimizing over a pre-trained LtO model. In Figures 2 and 3 you argue that this is bad, but never show experimental results on it? If (a) you train the LtO model on the distribution of $\zeta$ and then (b) you pre-train the predictive model on a two-stage loss and then fine-tune it on a pre-trained LtO model, the distribution shift may not be too bad and perhaps it may not perform too badly? Is this the EPO with Proxy Regret baseline? It doesn't seem to be defined anywhere. Do you do any of the things suggested above to improve performance? How do you do hyperparameter tuning? Overall, this baseline seems to be a straw man.
  1. *Other concerns:* Where are the error bars? There seems to be an overload of $k$ as a measure of complexity of the mapping of $z \to \zeta$ and also a measure of complexity for $F_\omega$. Often for EPO, it works better to have a simpler predictive model $F_\omega$ due to overfitting issues; have you tried this? Also, it is often useful to "mix a little two-stage loss" into the EPO objective. Have you tried this?

- **I don't think that the *idea* is particularly novel/interesting.** Taken together, I believe that you *could* choose to solve predict-than-optimize problems using LtOF, and even that you could do better than two-stage. However, as the paper has noted as well, this has been studied time and again. Moreover, the idea of using LtO to solve for $x^\star(\zeta)$ is not particularly novel, imo. I think the more interesting question is about whether you *should* do it in this way and when, and I don't think the paper has answered this question. Specifically:
  1. *Generalization of LtOF to similar problems:* What happens if we want to change the number of stocks in our portfolio optimization problem at test time? In EPO, predictions are typically made per-stock $i$ using covariates $z_i \to \hat{\zeta}_i$. As a result, we can, for example, remove some of the stocks from our portfolio optimization problem at test-time without having to re-train.
  1. *Being able to use the optimal solver at test time:* From Table 2, we can see that the regret before restoration for the AC-OPF domain is *very* different than after restoration. This suggests that LtOF is not able to learn how to make good decisions. As a result, it seems like it would be useful to be able to use an optimal planner at test-time. This seems even more true in, e.g., shortest path or ranking problems where there is a lot of structure in the constraints. This begs the question, when should you (a) predict the intermediate parameters, e.g., using prediction + LtO separately, versus (b) predict decisions directly from features?
  1. *Interpretability/Fairness/Additional Considerations:* This is hard enough for prediction alone, does LtOF make this easier or harder in the predict-then-optimize case?

References:
1. Shah, Sanket, et al. "Decision-Focused Learning without Differentiable Optimization: Learning Locally Optimized Decision Losses." NeurIPS (2022).
2. Zharmagambetov, Arman, et al. "Landscape surrogate: Learning decision losses for mathematical optimization under partial information." arXiv preprint arXiv:2307.08964 (2023).
3. Shah, Sanket, et al. "Leaving the Nest: Going Beyond Local Loss Functions for Predict-Then-Optimize." arXiv preprint arXiv:2305.16830 (2023).
4. Sahoo, Subham Sekhar, et al. "Backpropagation through combinatorial algorithms: Identity with projection works." ICLR (2023).

### Questions
Please respond to the questions in the weaknesses section.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
