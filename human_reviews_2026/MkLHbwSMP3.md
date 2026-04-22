# A Comedy of Estimators: On KL Regularization in RL Training of LLMs

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 6, 2

## Abstract
The reasoning performance of large language models (LLMs) can be substantially improved by training them with reinforcement learning (RL). The RL objective for LLM training involves a regularization term, which is the reverse KullbackLeibler (KL) divergence between the trained policy and the reference policy. Since computing the KL divergence exactly is intractable, various estimators are used in practice to estimate it from on-policy samples. Despite its wide adoption, including in several open-source libraries, there is no systematic study analyzing the numerous ways of incorporating KL estimators in the objective and their effect on the downstream performance of RL-trained models. Tang & Munos (2025) show that prevailing practices for incorporating KL regularization do not provide correct gradients for stated objectives, creating a discrepancy between the objective and its implementation. In this paper, we further analyze these practices and study the gradients of several estimators, revealing how design choices shape gradient bias. We substantiate these findings with empirical observations by RL fine-tuning Qwen2.5-7B and Llama-3.1-8B-Instruct with different configurations and evaluating their performance on both in- and out-of-distribution tasks. Through our analysis, we observe that: (1) estimator configurations with biased gradients can result in training instabilities; and (2) using estimator configurations resulting in unbiased gradients leads to better performance on in-domain as well as out-of-domain tasks. Overall, our findings provide useful takeaways for using KL-regularized objectives during RL post-training of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the role and implementation of Kullback–Leibler (KL) regularization in reinforcement-learning-based fine-tuning of large language models (LLMs). Specifically, it studies the effect of KL estimator design (e.g., K1, K3) and where the regularization term is applied (in the reward or the loss). The authors analytically derive gradient expressions for each configuration and demonstrate that some commonly used setups—especially those involving the Schulman (K3) estimator or placing KL directly in the loss—yield biased gradients relative to the intended reverse KL objective.
Empirical experiments with Qwen2.5-7B and Llama-3.1-8B-Instruct confirm that these biases manifest as instability and reduced out-of-distribution performance, whereas unbiased configurations (notably K1 in reward) produce more stable and generalizable models. The paper concludes with recommendations for correctly incorporating KL regularization in RL post-training pipelines.

### Strengths
Clear conceptual separation: Distinguishes between estimator type and where it is applied, which was previously conflated in many RLHF implementations.

Mathematically rigorous: Derivations are mostly correct and insightful, exposing how bias arises.

Theoretical and empirical alignment: The experimental outcomes directly reflect the analytic predictions.

Practical relevance: Provides actionable advice—using K1 in reward—for practitioners training LLMs with RL.

### Weaknesses
1. The theoretical analysis is mostly correct and logically coherent, but two small formulation issues were found:

Equation (14) (K3 in reward):
The derivation repeats the sequence-level log-ratio term inside the token-level summation, effectively multiplying it by T.
The correct form should place the log(pi_theta / pi_ref) term once, outside the sum.
This correction does not change the paper’s conclusions but fixes an algebraic error.

Equation (15) (K3 in loss):
The gradient is written as -E[r_t * grad log pi_theta], but the full derivative is (1 - r_t) * grad log pi_theta.
These are equivalent in expectation because E[grad log pi_theta] = 0, but showing the complete form would improve clarity and correctness.

Otherwise, the gradient decomposition in Equation (6) is correct and standard (score-function plus pathwise terms). The K1 results are accurate:

For K1 in reward, the gradient matches the true reverse-KL gradient.

For K1 in loss, the expected gradient is zero, since the pathwise term vanishes in expectation.
The analysis of K3’s bias and its effective behavior as a forward-KL regularizer is also correct.
Empirical results agree with the theory and are methodologically sound.



2. Experimental scope is somewhat narrow (math RLVR tasks only); results on other domains like code or dialogue would strengthen the claim.

3. The paper notes that adding KL to both reward and loss can recover unbiased gradients but does not test this empirically.

4. K3-in-loss behaves stably but with lower performance; more analysis (e.g., measuring entropy or forward-KL distance) would clarify why.

### Questions
Have you tested adding KL to both reward and loss for K3 or other estimators to confirm that it recovers unbiasedness as suggested analytically?

Why does K3-in-loss remain stable despite its bias? Can you quantify this by tracking token-level entropy or forward-KL measures?

How do off-policy updates or PPO/GRPO clipping interact with these KL configurations? Could stricter clipping stabilize biased cases?

Would using an adaptive or annealed beta (KL coefficient) schedule change which configuration is most stable?

How do the findings extend to learned-reward settings such as RLHF or RLAIF, where the reward model introduces noise?

You mention that popular libraries use biased KL objectives. Could you include a short table showing the default estimator, placement, and recommended correction for each?

### Soundness
3

### Presentation
3

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
This paper systematically analyzes various KL divergence estimators, especially on how they affect the training stability and algorithmic performance.

### Strengths
+ This paper studies an important problem.

+ It provides a detailed comparison between different KL estimators, and the empirical results are comprehensive.

+ Several takeaways are offered that may guide a more principled use of KL divergence.

### Weaknesses
+ The notation is a bit confusing. I had to go back and forth a couple of times to figure out what K1 and K3 actually stand for. The authors may consider name them in a more informative way.

+ The main observations are made empirically, it would be more insightful if further theoretical understanding are provided.

+ It remains unclear how different KL estimators interact with ratio clipping. That is, it does not fully isolate the effect of the KL estimator given the existence of the ratio clipping function.

### Questions
+ Do you think that we only need KL divergence or ratio clipping to ensure training stability, or it has to be both of them? Why?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper analyzes different KL regularization estimators used in RL training of large language models.
It shows that common implementations, such as placing the KL term inside the loss (e.g., K3-in-loss), lead to biased or unstable gradients, while only the K1-in-reward configuration yields an unbiased and stable update.
Both theoretical analysis and experiments on synthetic and LLM fine-tuning tasks (e.g., MATH) validate these findings.

### Strengths
1. **Systematic analysis and unified perspective.**
The paper provides a clear and thorough theoretical analysis of how different KL estimators behave when applied within RL training for LLMs. It systematically distinguishes the gradient properties of various configurations (e.g., K1-in-reward, K3-in-loss), offering a principled understanding of why some widely used implementations are biased or unstable.
The paper helps unify several inconsistent practices used across current RLHF frameworks. This unified perspective is not only theoretically clean but also highly relevant to practitioners, as it directly identifies which estimator–placement combinations produce unbiased gradients.

2. **Comprehensive empirical validation.**
The experiments, including both synthetic setups and LLM fine-tuning tasks (e.g., MATH with Qwen2.5 and Llama-3.1), provide consistent empirical evidence supporting the theoretical claims. The results clearly demonstrate that biased estimators can lead to collapse or divergence, while the unbiased configuration ensures stable convergence.

### Weaknesses
1. **Lack of practical cotribution.** The paper’s findings mainly reinforce practices that are already widely adopted—explicitly or implicitly—in existing RLHF implementations. As a result, the contribution feels more clarificatory than innovative, focusing on formalizing established patterns rather than proposing new directions.

### Questions
1. Could the authors elaborate on whether their analysis inspires any new algorithmic variants or training strategies beyond the clarification of existing practices?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies how reverse-KL regularization works in RL post-training of LLMs. The authors compare two token-level KL estimators. These are K1 (log-ratio) and K3 (Schulman’s approximation). They also compare two placements. One adds to the reward (score-function term). The other adds to the loss (pathwise term). They argue "K1-in-reward" is the only one that gives an unbiased estimator of the reverse $D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$ gradient. The other three setups are biased. This has real effects for stability and generalization (Table 1). Experiments used RLVR math tuning on Qwen2.5-7B and Llama-3.1-8B-Instruct. They report: (i) K1-in-loss creates instabilities or collapse; (ii) K3-in-reward collapses; (iii) K3-in-loss trains stable but does worse than K1-in-reward; (iv) a lower $\beta$ seems to perform better.

### Strengths
- The pathwise vs. score-function breakdown (Eq. (6)) is clean. It identifies which term each implementation estimates. It is correct as presented.
- Table 1 maps estimator × placement to bias and claimed behavior. This is usefull for practitioners.

### Weaknesses
1. The main theoretical point is already made by Tang & Munos (2025). The paper admits this. The point is that many common KL implementations do not yield the reverse-KL gradient. Much of the math repeats known results and blog-level derivations. This includes: K1 is unbiased in reward; pathwise K1 gradient is zero in expectation; K3 is an unbiased *divergence* estimator but a biased gradient in both placements. The paper’s main claimed novelty are the table systematization and a small empirical study. This is a small step.

2. The main story is “unbiased gradient = stability & OOD gains; biased gradient = instability/collapse”. This is not well isolated from other factors: 1) All main experiments use RLOO to “isolate” KL effects. But many real pipelines use PPO/GRPO with clipping, ratio caps, and more aggressive batching. The paper does not show the conclusions persist in those regimes it criticizes; 2) The paper assumes strictly on-policy sampling ($\omega=1$). But instability is known to explode off-policy. Just increasing minibatch updates from 1→4 collapses runs, no matter the estimator. This hurts the tidy “bias = collapse” story; 3) Key cross-setup comparisons are based on 250 steps of training on MATH. This is a tiny budget for modern RL post-training. Behavior might change at realistic budgets. No learning-rate sweeps or $\beta$-schedules were tried. 

3. Qwen baseline is evaluated with and without the chat template. The tuned models always use one. This makes comparisons noisier. Decoding uses temperature $=1.0$, $p=1.0$, and min-p $=1.0$ in vLLM. This is a very random regime for Pass@1. It can mask small gains or losses. The paper gives no sensitivity analysis to decoding or sampling seeds.

### Questions
1. Please implement the setup that adds the estimator to both reward and loss. This should realize the exact reverse-KL gradient in Eq. (6). Compare stability/accuracy to your four baselines. If this is noisy in practice, quantify the variance and cost.

2. Many pipelines use GRPO/PPO. Can you replicate the study under GRPO with clipping and typical off-policy minibatch updates? Your own results show that small off-policy collapses training no matter the estimator. Does K1-in-reward keep its claimed advantage there?

3. Re-evaluate with deterministic or low-temperature decoding. Report bootstrap CIs. Your current Pass@1 at temperature $1.0$ and $p = 1.0$ is probably underpowered to detect small but real effects.

### Soundness
1

### Presentation
1

### Contribution
1
