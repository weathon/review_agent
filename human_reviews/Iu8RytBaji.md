# Post-hoc Reward Calibration: A Case Study on Length Bias

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Reinforcement Learning from Human Feedback aligns the outputs of Large Language Models with human values and preferences. Central to this process is the reward model (RM), which translates human feedback into training signals for optimising LLM behaviour. However, RMs can develop biases by exploiting spurious correlations in their training data, such as favouring outputs based on length or
style rather than true quality. These biases can lead to incorrect output rankings, sub-optimal model evaluations, and the amplification of undesirable behaviours in LLMs alignment. This paper addresses the challenge of correcting such biases without additional data and training, introducing the concept of Post-hoc Reward Calibration. We first propose to use local average reward to estimate the bias term
and, thus, remove it to approximate the underlying true reward. We then extend the approach to a more general and robust form with the Locally Weighted Regression. Focusing on the prevalent length bias, we validate our proposed approaches across three experimental settings, demonstrating consistent improvements: (1) a 3.11 average performance gain across 33 reward models on the RewardBench
dataset; (2) improved agreement of RM produced rankings with GPT-4 evaluations and human preferences based on the AlpacaEval benchmark; and (3) improved Length-Controlled win rate (Dubois et al., 2024) of the RLHF process in multiple LLM–RM combinations. According to our experiments, our method is computationally efficient and generalisable to other types of bias and RMs, offering a scalable and robust solution for mitigating biases in LLM alignment and evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a calibration method for correcting bias (in particular length bias) in already trained LLM reward models (RM) on datasets. Assuming that the reward can be decomposed into an aligned reward and a bias reward term, the proposed method estimates the bias reward term using locally weighted regression (LWR) or a more robust variant (LOWESS). Then the correct reward is obtained by simply subtracting the estimated bias term. Experiments in RMBench, RM as Judge, and RM alignment show that the proposed method (RC-LWR) is able to effectively and robustly remove length bias compared with the common approach of adding a length penalty during RM training. Adding a length penalty requires a task-dependent penalty weight, whereas RC-LWR robustly adapts to different tasks and datasets.

### Strengths
- The proposed method, RC-LWR, is post-hoc and so does not require intervention during the training of the RM. It is also fast to compute. This means you can make many post-hoc corrections and sweep over hyper parameters all without needing to retrain. This is a huge advantage over adding a length penalty or other penalty term during RM training. This means you can try to correct for as many different biases as you want to do and try out combinations of them all post-hoc.

- RC-LWR is also robust to the curvature of the bias term due to using locally weighted regression, and only relies on the bias being locally close to linear.

- The paper is very clear and easy to follow.

### Weaknesses
- A slight weakness is the majority focus on length bias. The experiments pretty much all focus on length bias, with one generalization task to markdown features. 

- Another slight weakness is that only the combination of length penalty + RC-LWR was tested. A more interesting direction would be to test a combination of two biases, and whether you can apply the method multiple times to remove multiple biases.

### Questions
Typo between lines 224-225 “Based one LWR…”

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the reward modeling process in Reinforcement Learning from Human Feedback (RLHF). The authors highlight the issues of bias and spurious correlations in the training data and propose a method called post-hoc reward calibration to address these limitations. Specifically, they assume the reward model can be decomposed into two components and use the evaluation set to estimate the bias term through locally weighted regression. The experimental results demonstrate the method's effectiveness.

### Strengths
1.	The authors effectively identify common limitations in reward models and propose a method to enhance their reliability.

2.	The proposed method is claimed to be computationally efficient.

3.	Experimental results confirm the method's effectiveness.

### Weaknesses
1.	Assumptions: The proposed method depends heavily on several assumptions, such as (1) decomposition of the biased reward, (2) independence of the bias characteristic, (3) sufficient density, and (4) Lipschitz continuity. These assumptions may not hold in all scenarios.

2.	Difficulty in Determining $d$: In some scenarios, determining dd is difficult. While LWR can estimate Equation 5, linear approximation may often be inaccurate, necessitating further analysis.

3.	Simple Baselines: The baselines used are relatively simple. Comparisons with more complex baselines, such as reward model engineering in reference, are needed.

4.	Demonstrations Needed: Additional examples should be provided to showcase the method's effectiveness in various situations.

6.	Bias Exploration: It would be beneficial to explore other biases beyond 'length' and 'style', given that significant improvements are mostly seen in 'length bias'.

### Questions
1.	Bias and Spurious Correlation: How does the characteristic bias $c$ lead to spurious correlations? Examples would facilitate a better understanding for the reader.


2.	Hyperparameter Selection: What guidelines are there for selecting hyperparameters in different scenarios, such as $\beta_0$ and $\beta_1$?

3.	Other Types of Bias: Are there experiments demonstrating bias beyond length and style?

4.	Fig. 5 x-axis: What does the x-axis represent in Figure 5? Providing clarity would help readers interpret the results more effectively.

5.	There are several methods that can mitigate length bias, particularly in DPO. Have you considered simple baselines such as simPO?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on correction of spurious correlations in LLM-based reward models learned from preference data. The authors assume that the spurious feature (e.g., length) biases the rewards in a consistent away. They propose to estimate this consistent bias via a locally-weighted-regression method. Then, the estimated bias can be subtracted from learned reward in order to achieve a model without the spurious correlation. The authors focus on length as the spurious feature and compare to a baseline which subtracts a constant factor times the length from the reward. They evaluate a variety of reward models with various adjustments to eliminate length bias across several metrics, including RewardBench scores, correlation with human rankings of LLMs in ChatbotArena, and length-controlled win rate in AlpacaEval2. In general, the proposed method appears to better remove length bias.

### Strengths
The proposed method is relatively simple and is well-motivated conceptually. The fact that it can be applied post-hoc means that retraining is not required, which is helpful. I am not very familiar with the literature in this area but to my knowledge it is also novel. The results are extensive and seem to generally support that post-hoc reward calibration effectively removes length bias.

### Weaknesses
The main weaknesses of the paper include:

 * **Presentation:** I found the introduction of the method in Section 3 to be relatively clear, but the experiments in Sections 4 and 5 were much more difficult to understand. While I eventually could figure out the gist of the experiment design, it is often quite complicated and could be better presented. In particular, Section 4.2 is very dense. It describes multiple evaluation methods (AlpacaEval1, AlpacaEval2, ChatbotArena), which if I understand correctly in some cases used as both gold standard (AlpacaEval2) and as baselines (LC). The combination of many reward models, LLMs, calibration methods, and comparison rankings makes it quite important to carefully describe the purpose behind the experimental setup and why it is a good evaluation of the calibration methods. I think the current text requires several readings to understand these details. The rest of Sections 4 and 5 are also very dense and difficult to understand at first. It would be good to improve the presentation of these such that the results are easier to understand.

 * **Lack of insight into the result of reward calibration:** while the authors present many experiments on how reward calibration affects downstream performance, they do not present what the actual calibration process looks like for typical reward models. It would be helpful to see what the actual $\hat{r}\_\theta(c)$ function looks like over a range of values for $c$ for various reward models. This could help readers understand why RC-Mean and RC-LWR outperform length penalty, especially if the $\hat{r}\_\theta(c)$ function is quite non-linear. On the other hand, if it is close to linear, then it would raise the question of why not simply tune the constant $\alpha$ in the linear length penalty. Perhaps the length penalty simply does not use the right coefficient—in fact, it seems that it should always be possible to remove the length correlation measured in Table 3 by using a linear length penalty with the right coefficient.

 * Finally, the authors do not consider the extent to which the correlation between length and reward may not be entirely spurious—there may actually be some amount of weight that the reward models *should* place on length. 

Small issues:
 * "LLMs alignment" -> "LLMs' alignment"
 * Line 470 "board" -> "broad"

### Questions
With regards to the second weakness above, it would be helpful if the authors could plot the learned $\hat{r}_\theta(c)$ function for several reward models to see what kind of affect subtracting it has compared to the linear length bias $\alpha \\, c$. Also, it would be helpful to compare to a baseline of a linear length bias where the factor $\alpha$ is tuned per reward model.

### Soundness
3

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
4

### Summary
This paper proposes a novel method, “Post-hoc Reward Calibration,” to mitigate biases in reward models (RMs), particularly focusing on LLM output text length bias in RMs for large language models (LLMs). The method removes this bias by calibrating the reward score without additional RM retraining, making it computationally efficient. The approach assumes independence between the true reward and text length, constructs a locally linear regression model from text length to (proxy) reward values using the Robust Locally Estimated Scatterplot Smoothing (LOWESS), and calibrates the (proxy) reward with this regression model. The authors evaluate the approach on three tasks—RewardBench, RM as LLMs Evaluator, and RM for LLMs Alignment—and demonstrate that the calibrated RMs better align with human evaluations by providing fairer, length-independent scoring.

### Strengths
**Originality**: The paper proposes a unique, post-hoc calibration approach to mitigate reward model biases related to LLM output length without requiring additional data or retraining.

**Efficiency**: This method is computationally efficient, applying bias correction as a post-processing step, making it easily applicable across tasks without modifying reward models.

**Empirical Validation**: The approach is thoroughly evaluated on three tasks, effectively demonstrating its capacity to mitigate length bias.

### Weaknesses
**Assumption**:
The assumption of “Independence of the biased characteristic” may be overly restrictive in practical applications, especially given that this study focuses on text length as the biased characteristic. My understanding of this assumption (line 166) is that even when $x_1$ and $x_2$ are fixed at any arbitrary lengths $c_1$ and $c_2$, respectively, the true reward difference between $x_1$ and $x_2$ would be expected to be zero, implying no correlation between reward and text length. However, in practical applications, this assumption will often not hold. For example, in cases where prompts explicitly request brevity or detail, such as “respond concisely” or “provide a detailed explanation,” reward often depends on the response length. This assumption could thus limit the generalizability of the proposed method, especially for tasks where length naturally influences quality. A more flexible approach that adjusts for prompt-specific length expectations could enhance applicability.

**Baseline setting**:
The Length Penalty hyperparameter $\alpha$ seems to be fixed at 0.001, which may not be optimal across different reward models (RMs) and tasks due to variations in scale. While the proposed method benefits from adaptive $d$ tuning, the baseline might perform better with an α adjusted per RM or task, ensuring a fairer comparison. For instance, $\alpha$ could be proportional to the RM’s reward scale.

### Questions
How is the sensitivity of the length penalty hyperparameter to the performance compared to the proposed methods?

### Soundness
2

### Presentation
3

### Contribution
2
