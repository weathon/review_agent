# The Best of N Worlds: Aligning Reinforcement Learning with Best-of-N Sampling via max@k Optimization

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
The application of Reinforcement Learning with Verifiable Rewards (RLVR) to mathematical and coding domains has demonstrated significant improvements in the reasoning and problem-solving abilities of Large Language Models.
Despite its success in single generation problem solving, 
the reinforcement learning fine-tuning process may harm the model's exploration ability, as reflected in decreased diversity of
generations and a resulting degradation of performance during Best-of-N sampling for large N values.
In this work, we focus on optimizing the max@k metric, a continuous generalization of pass@k.
We extend on-policy gradient estimate to the off-policy updates, a common element in modern RLVR algorithms, that allows better sample efficiency.
Empirically, we show that our objective effectively optimizes max@k metric in off-policy
scenarios, aligning the model with the Best-of-N inference strategy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a method to improve performance of models fine-tuned with RLVR on mathematical reasoning tasks. As many models are used with best-of-N sampling at inference time, the paper proposes to explicitly optimize the max @ k metric which is compatible with the best-of-N sampling technique. It builds on existing estimators for the policy gradient of max @ k reward, and extends them to the off-policy setting prevalent in RLVR methods. The paper presents a few experiments to support the usage of this optimization target.

### Strengths
The paper presents its ideas and proposed contributions clearly and simply, the reasoning and motivation are easy to follow into the proposed method and experiments.

The derivation of the policy gradient estimator seems sound and correct. 

Using the proposed estimator, it seems like the proposed method outperforms baselines on the targeted metric of max @ k for large k.

### Weaknesses
My main concern with this paper is that the proposed method and results seem incremental and slightly insignificant.  

As the authors themselves note, the unbiased estimator for the max @ k policy gradient was developed by Walder & Karkhanis. The motivation for deriving this gradient has also been described in previous papers. In addition, while many modern RLVR algorithms are indeed effectively off-policy, most are designed as on-policy algorithms and assumptions are only relaxed due to necessity of implementation. The same holds for the max @ k objective estimator for the on-policy case. Given all of this, it seems like the novelty of this paper is very limited. 

This compounds with the results being unimpressive - while found significant by the Wilcoxon test, it seems like there is very little gain over baselines. It is also lacking that results are not reported for k values other.than 128 (eg. similarly to the plot in Figure 1). Clarity in the way the baselines were run and measured would be a significant addition - see some questions below. 

Finally, it is worth wondering whether optimizing for best-of-N sampling, which in itself is necessary only because our models are imperfect, is good algorithmic design. It may border on the realm of Goodhart’s law - if we optimize for best-of-N, do we give up on improving top-1 performance?

### Questions
1. Figures 1/2: the effects do not seem very pronounced. How has the statistical significance of the motivating results been measured?
2. In the Off-Policy BoN approach (as presented in the experiments), is the policy gradient presented at the end of section 3 evaluated directly, of is it incorporated into GRPO? If it is the latter, how is it incorporated? 
3. Following up on Q2, is any variance reduction performed? Does it follow mean or LOO-1 (comparing to the on policy version)?
4. Does the derived policy gradient estimator incur higher computational complexity than standard reward? How does it compare to standard methods in terms of runtime/extra computation?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the limitation of current Reinforcement Learning with Verifiable Rewards (RLVR) methods, which optimize for single-sample accuracy but often reduce generation diversity, leading to suboptimal performance when applying Best-of-N (BoN) sampling strategies at inference time. Focusing on the code generation task, the authors propose a novel optimization objective based on max@k, a continuous generalization of the pass@k metric. They derive an unbiased gradient estimator for both on-policy and off-policy cases, enabling direct optimization of max@k. Empirical results on several code benchmarks demonstrate that the proposed off-policy BoN objective improves max@k performance over strong inference-aware baselines, showing its effectiveness in aligning RL optimization with BoN inference in code generation tasks.

### Strengths
- This paper proposes a novel optimization objective based on max@k which is a continuous generalization of the pass@k metric for aligning RL optimization with BoN inference in continuous reward settings like code generation tasks.
- This paper proposes an unbiased gradient estimator for both on-policy and off-policy cases, enabling direct optimization of max@k.
- The empirical results on several code benchmarks demonstrate that the proposed off-policy BoN objective improves max@k performance over strong inference-aware baselines, showing its effectiveness in aligning RL optimization with BoN inference in code generation tasks.

### Weaknesses
- This continuous max@k optimization method is a direct extension of the previous works on binary reward setting. The novelty is limited.
- The empirical results mainly focus on code generation tasks, and with only the Qwen2.5-coder-7B model. It is hard to conclude that the proposed method can generalize to other tasks and models.
- The max@1 score compared with the baselines is not significantly improved which raises the question of whether the proposed method can improve the single-sample accuracy.
- The results only demonstrate the effectiveness of the off-policy BoN objective in max@k while k=128. It lacks the understanding of the performance with different k values.

### Questions
- The experiments are limited to code generation tasks using only the Qwen2.5-Coder-7B model. Can you provide results on other domains (e.g., mathematical reasoning) or with different model architectures to validate its generalization？
- The improvement on the max@1 metric compared to baselines is not significant. Does your method inherently trade off single-sample accuracy for better BoN performance, or can it be tuned to improve both?
- The experiments focus on k=128 for evaluating the off-policy BoN objective. Could you provide results or analysis for different values of k (e.g., 256, 512) to better understand how the method scales with the number of samples?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Prior work derived an on-policy gradient estimator for max@k. This work derives an off-policy version, and gives experimental evidence that this can improve max@1 scores (and can remain competitive for max@k) across five RL with Verifiable Rewards (RLVR) coding benchmarks.

### Strengths
Simple

Clear with few exceptions (noted in weaknesses)

I am not familiar with the related works, but I believe the off-policy gradient estimator is novel

Four reasonable baseline algorithms in main results

Includes both code snippets in the paper, and full code in the supplementary materials

### Weaknesses
The abstract says "We derive an unbiased on-policy gradient estimate for direct optimization of this metric." But my understanding is that exact (on-policy) estimator was already derived in prior work (as noted at the start of section 3.2).

The bulleted list of contributions at the end of Section 1 also repeats that claim, but again I think it maybe shouldn't.

That same list of contributions includes "We show that optimization of continuous reward is crucial for successful RLVR application", but I think that claim should be worded a lot less strongly.

A big weakness is the `clamp_delta` hyperparameter. It's great that the code snippet is included, but it's bad that this hyperparameter is not mentioned anywhere else in the paper, and it also seems a potentially big downside of the algorithm.

Doesn't explain why those particular hyperparameters in Section 4.4 were chosen. I searched quickly just now, and a learning rate of 5e-6 is a common default, and it looks like the other hyperparameters might likewise be common defaults, but I think this should probably be mentioned in the paper.

### Questions
> Yue et al. (2025) investigate the effect of RLVR finetuning on math, coding, and visual reasoning benchmarks. They show that while RLVR improves pass@1 performance, the base model achieves higher pass@k scores for larger k.

If I understand this right, Yue et al. find more intuitive results for pass@1 than the maybe counterintuitive binary reward results found in this submission's Table 1? Why might that be? It might be helpful to note this contrast.

&nbsp;

> The policy gradient family of RL algorithms became standard in LLM post-training. Introduced by Stiennon et al. (2020), the key idea of these approaches is to consider the language model as a policy and train it with an RL algorithm.

Would it be fair to say Ranzato et al. ([https://arxiv.org/abs/1511.06732](https://arxiv.org/abs/1511.06732)) or Ziegler et al. ([https://arxiv.org/abs/1909.08593](https://arxiv.org/abs/1909.08593)) introduced policy gradient training of LLMs, rather than Stiennon et al.?

&nbsp;

> However, BoN can be too computationally expensive, and several works addressed this issue. A range of works address this limitation by fine-tuning the model to directly mimic the BoN distribution: [...]. This line of work is orthogonal to the present study since their inference strategy assumes generation of a single completion, as opposed to the generation of k completions employed here.

Just how orthogonal are they? Could any of those works be modified to generate $k$ completions in an optimized way?

&nbsp;

> Another line of work explores model’s optimization to maximize performance with an account for inference-time techniques, like BoN.

A bit ungrammatical

&nbsp;

> Applying the log-derivative trick

The $\theta$ after this part runs into $\pi$, it needs better spacing

A similar issue happens for $\theta$ and $\rho$, later, in Section 3.3, after "Similarly to on-policy scenario, we can calculate gradient of the objective"

&nbsp;

> All other deltas appear in the sum $\binom{i-3}{k-3}$

Should that be $\binom{j-3}{k-3}$?

&nbsp;

> Adam optimizer(Kingma & Ba, 2014)

missing space

&nbsp;

> with learning rate $5e – 6$

As rendered in the paper, the 5e-6 part looks like 5 times Euler's number minus 6

&nbsp;

$k$ vs k (math font vs plain text font) are used inconsistently to mean the same thing (throughout the paper)

&nbsp;

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates fine tuning Large Language Models (LLM) with Reinforcement Learning to perform well on coding tasks. In particular, the paper focuses on best of N sampling at inference time, where N solutions are generated and a verifier selects the best performing solution to be returned.

The authors show that fine tuning an LLM with a policy gradient algorithm (GRPO) reduces the diversity of its generations, where performance of a single generation improves. But if we generate N solutions and evaluate the best one, fine tuning with RL hurts performance.

The paper uses a continuous reward function defined as the ratio of unit tests passed on a generated program by the LLM instead of a binary signal to check if all tests have passed. This leads to the max@K objective as the expected value of highest reward among the generations. The authors continue to derive an estimate of this objective in on-policy and off-policy settings, to be used with GRPO. Finally, experiments on 5 coding domains show that fine tuning an LLM with the proposed algorithm outperforms other baselines in 4 out of 5 domains under max@K scheme but does not improve performance under max@1 scheme.

### Strengths
1. The paper’s writing and organization is clear and coherent, I could mostly follow the motivation for the research and how this work fits with the covered prior work.
2. The experiment section contains sufficient details regarding the methodology and evaluation scheme, the choice of hyperparameters, baselines, and statistical tests performed to evaluate the significance of the results.
3. The derivations of the GRPO objective adapted to max@K in section 3 is clear and easy to follow.

(I should point out that I am not an LLM expert, and therefore I do not say anything strong regarding the novelty or significance of this work to the LLM community but do raise a concern regarding the applicability of best of N sampling in the Weaknesses section.)

### Weaknesses
1. The authors interchangeably use K and N in the text. This should be unified and cleaned up.
2. I did not understand the first experiment (Table 1). The authors claim optimization of continuous reward function is crucial for fine tuning LLMs with RL. While this intuitively makes sense to me as an RL expert, I missed how this experiment suggests such a claim.
3. Perhaps the biggest shortcoming of this work, in my opinion, is conceptual:
  - Optimizing max@K performance of the model does improve its performance under best of N sampling, since the optimization objective better matches the evaluation scheme but it may also increase the variance of generations making each output less reliable.
 - While this issue may not show up in benchmark performance under best of N sampling, it does raise concerns for the applicability of such model in the real world. Not every coding problem in the world comes with a suite of verifiable tests. And the additional cost of computation and verification would limit the usability of such a model.
- Optimizing the performance of best of N generations changes the distribution of outputs such that given sufficiently large N, a very good solution is generated. This is an effort to produce a large set of generations that contain a good answer (like finding a needle in a haystack), rather than producing reliable confident answers (planning and reasoning to complete a difficult task).
- It would be good to include a discussion regarding the variance of generations and the distribution of quality of generated solutions to see whether optimizing the max may harm the performance of other solutions.

### Questions
Please respond to the raised issues in points (2) and (3) of the weaknesses section of this review.

### Soundness
2

### Presentation
3

### Contribution
2
