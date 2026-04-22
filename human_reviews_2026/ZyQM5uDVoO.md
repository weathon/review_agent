# Counterfactual Explanations for Time Series Data via Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Counterfactual (CF) explanations are a powerful tool in Explainable AI (XAI), providing actionable insights into how model predictions could change under minimal input alterations. Generating CFs for time series, however, remains challenging: existing optimization-based methods are often instance-specific, impose restrictive constraints, and struggle to ensure both validity and plausibility. To address these limitations, we propose a reinforcement learning (RL) framework for counterfactual explanation in time series. Our actor–critic agent learns a policy in the latent space of a pre-trained autoencoder, enabling the generation of counterfactuals that balance validity and plausibility without relying on rigid handcrafted constraints. Once trained, the RL agent produces counterfactuals in a single forward pass, ensuring scalability to large datasets. Experiments on diverse benchmarks demonstrate that our approach generates valid and plausible counterfactuals, offering a reliable alternative to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a reinforcement-learning framework for generating CF explanations for time-series classifiers. A actor–critic agent (DDPG style) perturbs an autoencoder’s latent representation to produce CFs that flip a black-box model’s prediction while staying close to the original instance by an L1&L2 proximity penalty. Inference uses a single forward pass through the trained actor and decoder. A notable contribution is efficient batch CF generation at inference (Algorithm 2), avoiding per-instance optimization. Experiments on 13 UCR benchmark datasets compare the RL approach against several baselines. The results show high validity and strong plausibility (Fig. 1 and Fig. 3).

### Strengths
1. The paper targets a practical challenge: producing CF explanations for time-series classifiers where optimization-based methods can be slow and difficult to ensure plausibility. The proposed RL framework shifts most computation to training from testing, therefore at inference, the actor can generate CFs in a single forward pass and supports batching.

2. The method is model-agnostic, requiring only access to the classifier’s predictions for reward computation, without access of gradient to the black-box model.

3. Empirically, the approach consistently flips predictions across 13 benchmark datasets and shows high plausibility and robust validity relative to baselines.

4. The manuscript is clearly structured and easy to follow.

### Weaknesses
### 1. (Major) Clarity on the RL framing and one-step formulation.
While the paper frames CF generation as a sequential decision process in Section 1, it later states that the setup is “equivalent to a Markov decision process with a one-step horizon,” which removes sequential structure. This raises questions about the motivation for a DDPG-style actor–critic framework in this setting. Specifically:

a) The setup is explicitly defined as a MDP with a one-step horizon, this removes the sequential aspect of the search, and it seems to me, is more resembling a contextual bandit rather than a full RL problem.

b) The critic loss is defined as the squared error between the critic's estimate and the immediate reward, so it slightly deviates from standard DDPG and in this one-step setting, the critic acts merely as a reward predictor rather than the estimator of a long-term Q.

c) The benefits of experience replay and the complex, full actor–critic model over a simpler direct mapping/generative alternative are not clearly demonstrated. If such complexity is required by the black-box setting (e.g., the reward function can be non-differentiable for a black-box model), please explicitly state such motivations in the paper.

### 2. (Major) Trade-off in Proximity and Minimality:
Compared to optimization-based baselines, the method tends to produce CFs that are farther from the original inputs under L1/L2 distances, which may be conflicting with the “minimal change” goal. Larger distances can also reduce actionability in some application settings.

### 3. (Major) Lack of Ablation Studies and Qualitative Analysis:
a) Most empirical results are presented as violin plots, additional per-dataset tables or simple statistical comparisons would strengthen the claims. Also, key design choices (e.g., experience replay, clipping) are not ablated, making it hard to evaluate their contributions.

b) The paper does not report training cost (for Algorithm 1) or compare inference latency (generation of CFs) against optimization-based baselines to validate their advantages in batching.

c) Providing visualizable case studies of the generated counterfactuals will be helpful to evaluate the qualitative performance.

### 4. (Minor) Table 1 in Appendix overflows the right margin.

**If any of these points reflect a misunderstanding on my part, clarification is welcome.**

### Questions
1. Did the authors evaluate a simpler direct mapping (e.g., a single actor network trained to produce $z_{CF}$ without a critic), and if so, how is the outcome?

2. It seems that the text defines the action as a perturbation applied to the state, but Algorithm 1 indicates the policy outputs a latent CF $z_{CF}$ (with added noise and clipping) rather than a separate $\delta$. Could the authors clarify this definition?

3. What is the motivation for clipping latents to [−1,1]?

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
This paper works on using RL to generate counterfactual inputs for time series classification tasks. The authors proposed to use DDPG that perturbs inputs in the latent space of a pretrained autoencoder, then decodes them back to the input space. The method is model-agnostic which only needs access to model predictions and operates entirely in latent space. The paper highlights four contributions: (1) an RL approach that scales via batch generation; (2) latent-space perturbations with proximity penalties to promote realism; (3) black-box applicability; and (4) opening RL techniques to counterfactual explanation

### Strengths
1. The paper introduces a novel approach that applies reinforcement learning to generate counterfactual explanations for time series data, which is an original and meaningful contribution to the field.  
2. The method is thoroughly evaluated on a large number of UCR time series datasets, providing strong empirical support for its effectiveness and generalizability.

### Weaknesses
1. The first section of the paper reads as if it may have been generated by an LLM, with generic phrasing and limited depth in motivation and related work discussion.  
2. From Figures 2 and 3, the performance improvement of the proposed method over existing approaches appears modest, raising concerns about the overall effectiveness of the method.  
3. The paper lacks a runtime analysis comparing the training and inference time of the RL-based approach with prior counterfactual generation methods, which is important for understanding its practical efficiency.  
4. The visual presentation could be improved. A clearer overview diagram of the proposed framework or a visual comparison between the RL-based approach and prior methods would greatly help readers understand the system design and key differences.

### Questions
1. Could the authors provide a runtime analysis comparing the training and inference times of the proposed RL-based approach with existing counterfactual explanation methods?  
2. What is the main motivation for using reinforcement learning in this setting, and why do the authors believe RL is particularly well suited for generating counterfactual time series?  
3. Have the authors considered evaluating the method on datasets beyond the UCR collection to further demonstrate its generalizability?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a reinforcement learning (RL) framework for generating counterfactual explanations (CFEs) in time series classification using batches.  The authors use an actor–critic RL agent to learn a policy in the latent space of a pre-trained autoencoder. 
This enables the generation of counterfactuals that are both valid (flip the classifier’s prediction) and  plausible (remain close to the data manifold) without handcrafted constraints. Once trained, the RL agent can generate counterfactuals in batches, making the approach scalable to large datasets. Experiments on 13 UCR benchmark datasets showed reasonable results. The approach is model-agnostic, requiring only access to classifier predictions.

### Strengths
- The method is model-agnostic; 
- metrics and their relevant are well defined;
- Experiments adopted standard datasets, and reasonable metrics with relevant baselines.

### Weaknesses
- Autoencoder performance and its impact on results is not discussed; 
- RL to predict Counterfactuals in time-series is not really a novelty, limiting the contribution to the batch predictions. '
(more on the questions)

### Questions
- Did you created visualization (or other analysis) on the time-series to illustrate if the timing (of flip) is preserved?
- How much can the autoencoder impact the overall method's performance? 
- Are there any datasets where the model performance is worse than expected? (Violin plots shows 
comparison with others, wondering if there are any datasets where the method is not performing well)
- How does it compare with methods that don't predict using the batch approach? what is the trade-off between
efficiency (at using batches versus single estimations) and the metrics?

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
5

### Summary
This paper proposes a counterfactual explanation method for time series classification based on a reinforcement learning (RL) framework. Inspired by prior work on tabular data, the authors extend the approach to time series and evaluate it on multiple real-world datasets.

### Strengths
Strength:
1 The topic is interesting, and to my knowledge, this is the first work that applies RL to generate counterfactual explanations (CFs) for time series classification.
2 The experiments are abundant.
3 The algorithm1 is clear and well introduced.

### Weaknesses
Weakness:
1 Although the paper introduces RL for generating CFs in time series classification, it does not clearly distinguish this task from CF generation in tabular classification. This distinction should be clarified.
2 he generated CFs satisfy the proximity principle through a regularization term, but the paper does not address whether the produced CFs are realistic or plausible in real-world time series scenarios.
3 The writing assumes familiarity with the definition of counterfactuals. For readers who are new to CFs, the lack of a clear formal definition for time series may reduce readability.
4 The related work section does not cite prior RL-based counterfactual generation methods in tabular classification.
5 The method does not discuss or address the challenge of high-dimensional action spaces in time series classification.

### Questions
Suggestions and Questions:
1 I suggest adding a formal problem definition section before Section 4, including:
    1.1 A clear mathematical formulation of counterfactuals in time series classification.
     1.2 A comparison with the tabular setting, highlighting what makes the time series case different.
2 The related work section should be expanded. There are several well-known papers that apply RL to counterfactual generation in tabular data; I list some of them below for reference:
   2.1 Chen, Ziheng, Fabrizio Silvestri, Jia Wang, He Zhu, Hongshik Ahn, and Gabriele Tolomei. "Relax: Reinforcement learning agent explainer for arbitrary predictive models." In Proceedings of the 31st ACM international conference on information & knowledge management, pp. 252-261. 2022.
   2.2 Nguyen, Tri Minh, Thomas P. Quinn, Thin Nguyen, and Truyen Tran. "Counterfactual explanation with multi-agent reinforcement learning for drug target prediction." arXiv preprint arXiv:2103.12983 (2021).

3 I have a question for formulat 4. As the action is taken step by step and it is easy to control the sparsity of the modified features. Why do you optime the L1 norm instead of the L0 norm directly?

### Soundness
3

### Presentation
2

### Contribution
3
