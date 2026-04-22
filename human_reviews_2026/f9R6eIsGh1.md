# Primal-Dual Direct Preference Optimization for Constrained LLM Alignment

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
The widespread application of Large Language Models (LLMs) imposes increasing demands on safety, such as reducing harmful content and fake information, and avoiding certain forbidden tokens due to rules and laws. While there have been several recent works studying safe alignment of LLMs, these works either require the training of reward and cost models and incur high memory and computational costs, or need prior knowledge of the optimal Lagrange multiplier. Motivated by this fact, we study the problem of constrained alignment in LLMs, i.e., maximizing the output reward while restricting the cost due to potentially unsafe content to stay below a threshold. For this problem, we propose a novel primal-dual DPO approach, which first trains a model using standard DPO on reward preference data to provide reward information, and then adopts a rearranged Lagrangian DPO objective utilizing the provided reward information to fine-tune LLMs on cost preference data. Our approach only needs to train two models rather than three as in prior works that need trained reward and cost models, which significantly saves memory costs, and does not require extra prior knowledge. Moreover, we establish rigorous theoretical guarantees on the suboptimality and constraint violation of the output policy. We also extend our approach to an online data setting by incorporating exploration bonuses, which enables exploration in the uncovered prompt-response space, and provide theoretical results that get rid of the dependence on preference data coverage. Experimental results on the widely-used preference dataset PKU-SafeRLHF demonstrate the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The author introduce a approach called Primal-Dual Direct Preference Optimization (PD-DPO). The key contribution: "rearranged Lagrangian DPO objective" (Eq. 14). Since preferences are collected separately for rewards ($r$) and costs ($c$), DPO cannot be directly applied to the Lagrangian $L(\pi; \lambda) = r - \lambda c$. The authors rearrange the optimality conditions to express the cost preference likelihood using the reward function. They then substitute the unknown reward $r$ with the implicit reward information captured by a policy $\pi_{\hat{r}}^{\ast}$, which is pre-trained using standard DPO on reward preferences. The proposed algorithm (Algorithm 1) iteratively updates the policy (primal step) using this rearranged objective and updates the Lagrange multiplier $\lambda$ (dual step) via projected subgradient descent. The dual update relies on estimating the current policy's cost using online binary human feedback.
For experiments, they demonstrate PD-DPO’s superior performance compared to SFT and SafeDPO, while improving harmlessness over SFT. However, it is significantly less safe than the computationally intensive Safe RLHF baseline (Beaver-v3.0).

### Strengths
1. The challenge of applying DPO when preferences are separated by objective (reward vs. cost) is large. The author clearly expresses cost in terms of reward and the optimal Lagrangian policy, and then substituting the reward using a pre-trained standard DPO policy is a clever solution (bypassing the need for explicit reward and cost modeling during constrained optimization).
2. Extension to the online setting. The reliance on comprehensive offline data coverage is a bottleneck for DPO. The incorporation of exploration bonuses directly into the rearranged DPO objective to guide exploration in a constrained setting is novel.
3. The adoption of a primal-dual framework for the algorithm is theoretically sound compared to methods that require prior knowledge or the sweeping of the Lagrange multiplier. By integrating the dual update directly into the optimization loop, the algorithm considers the optimal tradeoff between reward maximization and constraint satisfaction.

### Weaknesses
1. In Appendix B, lines 700-701, you mention that you fix the Lagrange multiplier $\lambda$ to 5 due to computational limits, instead of running the dual update. This might limit the empirical validation of the paper. Compared to Beaver-v3.0, the results in Figure 1 only validate the rearranged DPO objective with a fixed penalty. Was any experimental validation done on the primal-dual part for solving the constrained problem?
2. The cost estimation procedure required for the dual update is expensive in the context of real-world LLM training. require: sampling $N^{CE}$ responses and asking $M^{CE}$ human annotators for binary feedback at every iteration K. It is an synchronous demand for human interaction inside the training loop, which is slow, expensive, and likely high variance. The massive annotation cost that convergence would incur, as suggested by the obtained bound in Theorem 1, seems to outweigh the computational gains of avoiding reward/cost model training.
3. The results of this paper rely on the accuracy of $\pi_{\hat{r}}^{\ast}$. I assume this policy, trained by the standard DPO, is used as a fixed stand-in for the reward signal on Eq. 16 throughout the constrained optimization. Therefore, any error, bias, or overly optimistic compromise in $\pi_{\hat{r}}^{\ast}$ will be carried over to the PD-DPO training. You did not account for this in your experiments; what happens if $\pi_{\hat{r}}^{\ast}$ is a poor candidate?
4. Calculating these exploration bonuses in O-PD-DPO also requires constructing and inverting covariance matrices based on the feature representations $\phi(x,y)$ (also mentioned in Appendix D). This attempt to work in the high-dimensional space of LLMs is computationally infeasible.
5. PD-DPO also shows a harmfulness gap to Beaver-v3.0 (e.g., Elo score < 1100 vs ~1400); this trade-off is unacceptable in safety-critical applications. You argued that this trade-off is expected for the gains in efficiency,  but reduction in safety suggests the method struggles to enforce constraints strictly (perhaps exacerbated by the use of a fixed $\lambda$).

### Questions
1. Any experimental results that validate the full Algorithm to show the trajectory of $\lambda_k$, rewards, and costs across iterations K? 
2. Any clarification on the practicality of the cost estimation step (Algorithm 1, Line 4). Can you quantify the total human annotation effort required for convergence as suggested by your theory? Given this burden, how do you propose making the dual update feasible in a practical LLM pipeline? Have you considered using an offline-trained cost model just for the dual update estimation to avoid human-in-the-loop feedback?
3. How sensitive is the PD-DPO framework to the quality of the initial standard DPO model? Errors in this model directly impact the optimization objective (Eq. 16). Any ablation study on the performance of the final constrained policy when $\pi_{\hat{r}}^{\ast}$ is trained with varying amounts of reward preference data?
4. Any additional experiments conducted for O-PD-DPO? 
5. [related to weakness 5]] (and just some suggestions) Have you analyzed the Pareto front achievable by PD-DPO (e.g., by varying the initial $\lambda_1$ or the optimization parameters, assuming the full algorithm is run)? Is it possible to achieve higher safety with PD-DPO, even at the cost of some helpfulness?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a direct preference optimization like procedure for safety constrained alignment, that foregoes the use of trained reward models and constraint costs by using labels to directly construct a supervised Lagrangian objective. They provide near optimality and feasibility guarantees for problems solved by dual ascent.

### Strengths
All existing constrained alignment approaches explicitly train reward and cost models, learning directly from helpfulness and safety  preference binary labels is both novel in the context of constrained alignment and relevant. The paper provides near feasibility and optimality last iterate guarantees under standard assumptions.

### Weaknesses
The two stage approach fits a model using DPO and then uses the implicit reward given by this model in the training objective (e.q. 14). It then uses a labeling oracle for the safety cost. This is indeed a novel and reasonable approach. My main concern is that statement that the main advantage of their method is not fitting reward and cost models might be overstated in the sense that (1) the first phase is indeed fitting a reward model and (2) the second phase swaps a cost model for a labeling oracle.

The big memory gains with respect to using explicit reward an cost models disappear if the reward and costs are evaluated offline for the dataset of prompts and responses - responses are not sampled at each training step but at the end of each training epoch or primal update step. This is indeed the approach used in prior work (Huang et. al and Zhang et. al) that use DPO style losses.

Also, the formulation of Lagrangian maximization using DPO style losses was proposed at least in Huang et. al (referenced in the submission), and doing dual super-gradient descent on lambda by sampling the model and estimating the slack in Zhang et. al (also referenced in the submission) The only discussion about the distinction between this prior work and the proposed approach is the aforementioned lack of explicit cost and reward models. Although these prior works also have feasibility and optimality guarantees, there is no discussion how do the theoretical results compare to those.

Finally, the experiments present a single run with a single constrained baseline, where the method performs comparatively poorly in terms of safety. Without more experiments it is hard to evaluate how the proposed approach performs in terms of helpfulness/harmlessness trade offs (i.e. its pareto optimality) and, more importantly, wether it empirically succeeds at obtaining near feasible solutions to support the theoretical results.

### Questions
Can you provide additional experimental evidence supporting the performance of your method? The only point you provide does very badly in terms of the constraints compared to the baseline. The number of experiments is very limited, only a single run/performance is reported.

Can you point out the relation of your theoretical results to those in Zhang et. al (referenced in the submission) ? 

Can you comment on the choice of constraint threshold and perhaps do an ablation on its impact?

Can you include plots of dual variable dynamics and or their values in the appendix?

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
2

### Summary
For both offline and online constrained alignment problem of LLMs, this work develops a novel primal-dual DPO approach which does not require trained reward and cost models or prior knowledge of the optimal Lagrange multiplier. Then the performance of this algorithm is demonstrated by theoretical convergence rate of suboptimality and cost violation, and experimental results.

### Strengths
The targeted problem of LLM safety alignment is important. The algorithm design looks novel, by obtaining analytical solution of cost and reward. The presentation is clear.

### Weaknesses
The major problem is the costly algorithm and problematic convergence bounds, as shown in the questions 1 and 2 below respectively.

### Questions
(1) Algorithm 1 looks costly due to the following reasons. 

(1a) Eq. (15) trains $\pi _ { \hat{r} } ^ {\star}$ instead of $\hat{r}$. It seems that similar to algorithms that train $\hat{r}$, we also need to train and save two large models. What's the advantage? 

(1b) Does line 4 require human annotation in every iteration? Some online DPO-type algorithms use advanced LLM to automatically annotate newly generated samples. 

(2) Issues about the convergence bounds: The bounds in Theorem 1 contains $B$, whose last two constants contains algorithm generated variables $\pi_k$ and cannot be guaranteed small. The final term of $B^{\rm on}$ for Theorem 2 seems to go to $+\infty$ as $K\to+\infty$. Could you prove that it goes to 9 as $K\to+\infty$? 

**I'd like to raise my rating if questions 1-2 can be solved well.**

(3) Right above Eq. (8), it seems that safe RLHF has access to only $\mathcal{D}^c$ but not $\mathcal{D}^r$? 

(4) In the experiment, what evaluation model is used in model-based evaluation? Could you list the hyperparameters for the other algorithms? 

(5) Optional: Do you think the primal algorithm [1] will work on constrained alignment problem, which does not require Lagrange multipliers? 

[1] Xu, T., Liang, Y., \& Lan, G. (2021, July). Crpo: A new approach for safe reinforcement learning with convergence guarantee. In International Conference on Machine Learning (pp. 11480-11491). PMLR.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper studies a constrained alignment problem in the framework of direct preference optimization (DPO). The authors reduce this problem to a Lagrangian dual problem, where the Lagrangian maximizer is evaluated by minimizing a preference-based optimization objective function. A key idea is to introduce a preference-based presentation of a reward model. Based on this formulation, the authors propose a preference-based primal-dual algorithm: PD-DPO. Furthermore, the authors provide the optimality and constraint violation guarantees, both with and without the preference data coverage assumption. Finally, the authors conduct a Safe RLHF experiment to show effectiveness.

### Strengths
- The authors study a constrained alignment for LLMs based on preference data. This is an important direction for aligning LLMs with specific human values, since constrained generation is required in LLM applications. 

- The authors present a method that allows us to solve the Lagrangian dual problem in the DPO style. A key feature is that the proposed training algorithm: PD-DPO doesn't require direct knowledge of reward and cost models. 

- The authors provide iteration complexity guarantees of the proposed algorithm in terms of objective and constraint functions, both with and without the preference data coverage assumption.

### Weaknesses
- It would be helpful if the authors could have a table to compare the proposed method with previous preference-based methods in terms of model/algorithm assumptions and computational efficiency. For instance, the authors mention previous preference-based methods (Liu et al. (2024b); Huang et al. (2024); Zhang et al. (2025); Kim et al. (2025)) need to regenerate preference data.   

- The main idea of the proposed method is the policy-based representation of the combined reward and cost function in Equation (12). However, this assumes the Bradley-Terry model for a mixed human preference. It is important to explain in what extent this assumptions is valid in practice. 

- The proposed algorithm: PD-DPO utilizes the cost binary feedback from human annotators, which assumes an off-shell cost model. This type of labeling assumption is also used in previous works (Liu et al. (2024b); Huang et al. (2024)). It is useful to clarify their differences. 

- The iteration complexity guarantees seem to be limited to theoretical interest, since it assumes optimization steps of PD-DPO are solved exactly. 

- The data coverage assumption in Section 4.3 is strong, since it assumes coverage over polices for all iterations. The exploration bonus analysis in Section 5 assumes the Bradley-Terry model, which is not explicitly mentioned in the main paper.

- Another weakness of this paper is the limited experimental evaluation in terms of data sets, and baseline methods.

### Questions
Additional to suggestions in Weaknesses, below are some other questions.

- Notation c is abused in Section 4.1. 

- Since reward and cost are unknown, how to determine bound constants in (17) and (18)?

- What is rho in Assumption 1? How to choose it based on preference data?

- How large is the data coverage-related constants in Theorems 1 and 2?

- Math writing should be improved.

### Soundness
2

### Presentation
2

### Contribution
2
