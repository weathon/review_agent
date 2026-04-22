# Optimal Attention Temperature Enhances In-Context Learning under Distribution Shift

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Pretrained Transformers excel at in-context learning (ICL), inferring new tasks from only a handful of examples. Yet, their ICL performance can degrade sharply under distribution shift between pretraining and test data—a regime increasingly common in real-world deployments. While recent empirical work hints that adjusting the attention temperature in the softmax can enhance Transformer performance, the attention temperature's role in ICL under distribution shift remains unexplored. This paper provides the first theoretical and empirical study of attention temperature for ICL under distribution shift. Using a simplified but expressive “linearized softmax” framework, we derive closed-form generalization error expressions and prove that shifts in input covariance or label noise substantially impair ICL, but that an optimal attention temperature exists which minimizes this error. We then validate our predictions through extensive simulations on linear regression tasks and large-scale experiments with GPT-2 and LLaMA2-7B on question-answering benchmarks. Our results establish attention temperature as a principled and powerful mechanism for improving the robustness of ICL in pretrained Transformers—advancing theoretical understanding and providing actionable guidance for selecting attention temperature in practice.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper theoretically investigates how adjusting the attention temperature (τ) within the softmax attention mechanism of Transformers affects in-context learning (ICL), especially under distribution shift between pretraining and test data. Unlike prior works that fix τ, it demonstrates that an optimal τ exists that minimizes generalization error — thereby improving ICL robustness.

### Strengths
- Theoretical derivation is clear and matches empirical behavior.  
- Simple insight (temperature tuning) leads to measurable robustness gains.  
- Bridges the gap between theory and practical tuning of LLMs.

### Weaknesses
1. Temperature tuning seems to be a well-known method, and the contribution of this work seems to be limited to the theoretical analysis on a simplified linear regression model.

2. While the authors derive a closed-form of the optimal temperature of the linear regression case, the insight into how to tune the parameter and the quantitative analysis of the effect of the temperature are still missing. 

3. Section 4.2 is confusing. The discussion of different cases seems to be made without sufficient evidence, and I do not think Lemma 4.1 and Corollary 4.4 can support the discussion on distribution shift cases.

### Questions
See weakness.

### Soundness
3

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
4

### Summary
This paper theoretically characterizes the optimal attention temperature of Transformers with linear attention in ICL tasks. The authors analyze the generalization error when different kinds of distribution shift exist during testing and show that the optimal temperature can mitigate the impact of distributional shifts. Experiments are conducted to support the theory.

### Strengths
1. The proposed research about the optimal attention temperature is interesting and well-motivated. 

2. The theoretical results are solid.

### Weaknesses
1. Theorems 4.6 and 4.7 do not theoretically indicate how the generalization error and the optimal temperature are related to the distribution shifts. There is no discussion of how Eqn. (15) will change if some parameters of the distribution change.

2. I think the basic setting of linear attention by the simplification of Eqn. (7) is well studied in existing works. Only adding the learnable temperature is not theoretically challenging enough. This makes the theoretical contribution less significant. It is better to summarize the theoretical novelty of this work.

### Questions
1. If my understanding is correct, you set $\tau=1$ for pretraining, and a changable $\tau$ for testing. If so, why do you set $\tau=1$ for pretraining? Why not set $\tau$ as the temperature during pretraining and $\tau'\neq \tau$ as the temperature during testing?

2. In line 411, why does the optimal temperature $\tau\_{optimal}=c$ fully counteract the shift?

3. In line 453, you say, "noise effects diminish as the context length increases, consistent with our theoretical predictions." Which theorem do you refer to? Do you mean the $\sigma/l$ terms in Eqn. 11? If so, please specify which equation the experiment verifies in this statement.

4. What does Figure 3(a) imply? What is the trade-off between added context and accumulated noise by line 471? How is it related to your theory?

### Soundness
3

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
This paper analyzes how the attention temperature at inference time affects in-context learning (ICL) performance when there is a distribution shift between pretraining and test data (in the input, task, or noise distributions).
Using a simplified Transformer with linearized softmax attention, the authors derive a closed-form expression for the ICL generalization error as a function of the model parameters and the attention temperature.
They also provide an analytic construction of the model parameters that makes this simplified Transformer approximate the Bayes-optimal predictor for the training distribution (the “train-Bayes optimal”).
Under this setting, they show that adjusting the attention temperature at test time can provably minimize the generalization error and, empirically, can compensate for distribution shifts, sometimes even outperforming the unadjusted (train-Bayes-matched) baseline.

### Strengths
1. The paper is clearly written and well structured, with a precise statement of the problem and motivation.

2. The derivation of a closed-form expression for the ICL generalization error appears technically sound and provides theoretical insight.

3. The study of how pretraining affects in-context learning behavior is an important and timely question that the paper addresses thoughtfully.

### Weaknesses
1. **Limited practical applicability of the optimal temperature:** The theoretically derived optimal temperature $ \tau_{\text{opt}} $ depends on quantities from the true test distribution (e.g., $ \Sigma_x^{\text{test}}, \Sigma_w^{\text{test}} $), which are unknown in practice.  Therefore, the practical takeaway essentially reduces to tuning $ \tau $ by sweeping or validation, which is intuitive and already common.  The paper could better clarify whether any *predictive or data-dependent heuristic* for $ \tau $ can be inferred from their theory.

2. **Clarifying the intuition about shift magnitude and temperature:**  I would like to ask the authors whether the following intuition is consistent with their results:  
 – *If the distribution shift is small, the optimal $ \tau $ should remain close to 1. if the shift is large (e.g., much higher input variance or noise), $ \tau $ should increase accordingly.*  This seems qualitatively aligned with current LLM practices, where higher attention temperature can stabilize performance under distribution mismatch. This seesm to be the case in Figure 2, but how can one see this from equation 12? Any insights?

3. **Constructed vs. learned parameters:** The entire analysis relies on constructed parameters (Lemma 4.1) rather than parameters obtained by training, even in the simplified model. It remains unclear whether training via gradient descent would converge to these analytic forms or perhaps to different parameters that perform better than the train-Bayes estimator due to implicit biases of optimization. Since attention models are highly expressive, this reliance on a fixed construction limits how directly the conclusions translate to actual trained models.

4. **Relation to test-time adaptation (TTT):**  Allowing temperature adjustment at test time is essentially a *restricted form* of test-time tuning or adaptation (TTT).  Naturally, permitting any adaptation using in-context examples can improve performance; in this work, the authors allow only a single scalar ($ \tau $) to change.  In principle, broader adaptation (e.g., re-scaling attention weights) could approximate the *test-Bayes optimal* predictor.  This connection to TTT could be discussed more explicitly.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies in context learning using a stylized model (common in the literature) showcasing the ability of an attention layer to learn linear functions "in-context". The paper deviates from prior work by considering a ridge-regression type Bayes-optimal predictor (incorporating a gaussian prior on the unknown parameter) and a "linearized" softmax attention. The paper compars the performance of the optimal transformer, and considers the degradation of this performance under distribution shift (a known failure mode of ICL amongst softmax/linear attention), finally demonstrating that tuning the temperature parameter in the attention mechanism can help alleviate some of the said degradation.

### Strengths
The Llama/GPT-2 experiment is particularly interesting. It seems to have prescriptive advice for extracting the optimal behaviour from softmax-attention - by setting the temperature to be related to the ratio of the variance of the pre-softmax scores to the mean of those scores.

### Weaknesses
Showing that linearized-softmax can express something similar to the Bayes optimal predictor seems a little derivative of a similar analysis for linear attention.

Theorem 4.7 is difficult to parse, I think it would help if it could be written out in terms of the optimal parameter values from Lemma 4.1. I think something like this happens in Appendix J.

### Questions
I didnt understand the set-up of the "real world" experiments. Are the optimal temperatures calculated per attention layer per batch? What is special about ICL that this rule should only be applied in this setting (this doesnt seem to be related to any distribution shift)? I am giving a slightly lower score now. Once I have some more clarity on section 5.2 and how this can be applied to real models I will likely raise it. I apologize if my question has been addressed already in the paper.

Is there an interpretation of post-softmax attention weights with the optimal temperature? An example of the type of statement i would hope for is that that they are approximately a standard log-normal under some assumption about the pre-softmax distribution or that there is some gap between the highest and average post-softmax attention weights, etc.

Comparing Figure 6 to Figure 3(a). How is the non-monotonicity of Exact Match Score for optimal temperature to be compared to the monotonicity of gpt-2 for regression problems with optimal temperature?

### Soundness
3

### Presentation
4

### Contribution
2
