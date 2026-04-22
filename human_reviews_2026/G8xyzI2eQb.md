# Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large Language Models (LLMs) have demonstrated strong performance on reasoning tasks, but post-training optimization remains essential for aligning their behavior with specific task objectives. Existing reinforcement learning (RL) approaches often rely on costly human annotations or external reward models, limiting their scalability in real-world applications. To address this, we propose Reinforcement Learning via Self-Confidence (RLSC)—a method that uses the model’s own confidence in its outputs as the reward signal, without requiring human labels, preference models, or manually crafted reward functions. RLSC is also highly sample-efficient: it only needs 1 to 8 samples per problem, and typically converges within 15 to 30 training steps. Under the Pass@1 evaluation metric, Qwen-Math-7B achieves significant performance improvements across several mathematical benchmarks: AIME2024 +6.7\%, AMC23 +33.1\%, Math500 +32.3\%, Minerva +29.8\%.On average, RLSC delivers a 23.68\% improvement across these benchmarks. Notably, the effectiveness of RLSC is not limited to the Qwen series; it also leads to substantial performance gains on other mainstream models, including Olmo-7B, DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Llama-8B, Gemma-4B, and LLaMA-8B, etc. In summary, RLSC offers a simple, efficient, and scalable post-training method for pretrained language models, enabling significant performance gains with few training steps.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a new label-free algorithm for post-training of language models. The algorithm is generally increasing the model's confidence on its output, say $\mathbb E_{y\sim p}p(y)=\sum_{y\sim p}p^2(y)$, which is easy to implement following policy gradient theorem, and can be smoothed with a hyperparameter on the weight. The experiments are conducted on extensive datasets and various models, using the metric pass@1. The results of self-confidence approach exhibit observable advantages over other label-free algorithms. The authors also provide some analysis on the experimental result.

### Strengths
- Clear paper writing.
- Extensive experiments.
- The RLSC loss itself is novel to the reviewer.

### Weaknesses
*major*:
- The main claim, "This work demonstrates that high-quality post-training can emerge not from external labels, but
from a model’s internal signal - when that signal is derived with care.", is not novel, which has been revealed in many recent papers [1,2]. This claim is especially well-known for Qwen series models, where RLSC exactly exhibits the most prominent improvement; the improvement on llama is marginal, compared with that of Qwen, as shown in figure 3.
- The main metric, pass@1, is not a strong signal to reflect the model strength. It is intuitive that increasing self-confidence can sharpen the distribution, which would clearly benefit the pass@1 metric. However, as many papers like [3] point that, the performance on pass@k might not be improved. It would be better to cover results on pass@8 or pass@16 to see whether the advantage still exists.

*minor*:
- In Section 2.2, a coefficient $2$ on the gradient is missed.

[1] Wang et al. Reinforcement Learning for Reasoning in Large Language Models with One Training Example. https://arxiv.org/abs/2504.20571

[2] Shao et al. Spurious Rewards: Rethinking Training Signals in RLVR. https://arxiv.org/abs/2506.10947

[3] Yue et al. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? https://arxiv.org/abs/2504.13837

### Questions
- It is not clear to the reviewer that whether there exists a clear separation between adjusting temperature and conducting self-confidence training. By decreasing temperature, the distribution of the output can also be sharpened, which aligns with the intuition of RLSC. Could the authors conduct some comparison experiments on this point?
- As shown in figure 5, the entropy would decrease as the confidence increase. So what's the difference between increasing confidence and directly decreasing entropy? Would directly adding an entropy term in the training objective achieve the same effect as RLSC, or even better?
- Section 4.2 of [1] shows that increasing entropy can enhance the reasoning power of language models, which seems a conflict to this paper's claim. Could the authors provide any clarification?

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
4

### Summary
This paper proposes a method for fine-tuning LLMs by using the most likely output for a given prompt. The main claim is that this method is an improvement over RL fine-tuning because it does not require a ground truth reward or reward model, and can be used in a self-supervised fashion. For each prompt the method first samples a number of responses from the model under training, then gets the log probabilities of each response under the model. The loss is chosen to reinforce the responses with higher log-probability according to the model. Experiment results on math benchmarks show that the proposed method can improve model performance, on par with RL fine-tuning and other baselines from prior works.

### Strengths
1. The proposed method relies of model's own log-probabilities to improve performance, removing the need for a reward as in RL fine-tuning. This simplification is a valuable contribution in cases where rewards are difficult to obtain.

2. Experiments studied a wide array of base model classes (Qwen, llama, gemma, etc.) and several common math benchmarks. The results show that the proposed method can match RL fine-tuning in model improvement.

### Weaknesses
1. One key issue with the proposed method of reinforcing the highest probability response is that it cannot correct cases where the most-likely response is initially wrong. As mentioned in Sec. 2.1 the motivation behind the proposed method is biasing the probability distribution  towards the most-likely response, which cannot make the necessary correction.

Such cases is exactly why external information from a reward signal is needed to improve a model.

Is this method to be used as a final fine-tuning stage after RL fine-tuning (e.g., similar to applying majority voting to an RL fine-tuned model (Sec. 2.1)? If this is the intended usage, then it would still require a reward for the RL fine-tuning. 

It is interesting that by simply reinforcing the most likely response, model performance can be improved to match RL fine-tuning. However, this is more of a negative result for RL fine-tuning (and the benchmarks) than a positive one for the proposed method.

### Questions
In section 2.1 the paper states that "This expression is maximized when the distribution collapses to a delta function centered on a single most probable response.", then why not make the loss function for exactly this case? Clearly, this is not a desirable outcome for fine-tuning as the model's distribution would suffer entropy collapse. What does this say about the motivation of this paper? 

Why is the TTRL method a natural starting point for this paper? Is it the best reward-free fine-tuning method? 

In Sec. 3.1, step 2, what does "to ensure the original distribution remains unchanged" mean?

What does pass@k look like after fine-tuning using the proposed method? Is pass@1 approaching pass@k or is does pass@k collapse?

There are editing errors in the paper (for example duplicate text in Sec 3.2). Please proofread carefully.

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
4

### Summary
This paper introduces Reinforcement Learning via Self-Confidence (RLSC), a fine-tuning method for language models that uses model prediction confidence as reward signals. Unlike conventional RL approaches that require human annotations, external reward models, or manually crafted reward functions, RLSC is self-supervised, leveraging only the model's internal probability distributions.
RLSC is efficient, requiring only 1-8 samples per problem and converging in 15-30 training steps. The method achieves significant performance improvements across mathematical reasoning benchmarks and demonstrates effectiveness across various model architectures.

### Strengths
- The motivation behind the proposed method is sound.
- The paper is well-organized and easy-to-follow.
- Expermental results show that the proposed method is effective.

### Weaknesses
- Citation format employed in the paper requires revision. For example, "Models such as DeepSeek-R1 Guo et al. (2025)" should be "Models such as DeepSeek-R1 (Guo et al., 2025)".
- Notations in Table 3 seem unclear. LLaMA-8B should be written as LLaMA-3.1-8B; Qwen-Math-1.5B should be written as Qwen2.5-Math-1.5B; Gemma-4B-pt should be written as Gemma-2-4B-pt. And the similar issue occurs from Line 081 to Line 087.
- Missing references: 
[1] Wang, Yiping et al. “Reinforcement Learning for Reasoning in Large Language Models with One Training Example.” ArXiv abs/2504.20571 (2025).

### Questions
- There seems to be inconsistency in model selection across different backbones. For the Gemma-series model, you used the pre-trained checkpoint. And for OLMo-2-7B, which version did you use? The pre-trained version or the instruct-finetuned version? Why use Qwen2.5-Math-7B-GRPO instead of Qwen2.5-Math-7B or Qwen2.5-Math-7B-Instruct?
- During experiments, do you use any confidence calibration mechanism since models may be overconfident on some data instances?

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
2

### Summary
The paper proposes RLSC (Reinforcement Learning via Self-Confidence), which is a post-training method that treats a model’s output confidence as the reward, avoiding human labels, preference models, or handcrafted/verifiable rewards.
The key idea is to formalize mode sharpening, leading to a differentiable self-confidence objective and simple sequence-level losses that can be optimized efficiently. Experiments report the effectiveness of RLSC across multiple backbones on mathematical reasoning benchmarks.

### Strengths
- This paper converts self-confidence into a direct, differentiable objective, and training requires no external rewards or labels and uses few samples per question.  
- This paper reports convergence within 15–30 steps and strong efficiency compared to TTRL’s multi-sample majority voting. 
- Empirical results demonstrate that RLSC improves performance across multiple backbones and benchmarks.

### Weaknesses
- The method explicitly sharpens the output distribution, which could harm exploration/diversity or BoN performance. Pass@k metrics should be reported.
- Benchmarks are predominantly mathematical reasoning. Generalization to other domains (code and instruction following) is not demonstrated, limiting external validity.

### Questions
- The method sharpens the output distribution, which could harm exploration/diversity or BoN performance. Could you provide results on pass@k metrics to evaluate this aspect?
- The experiments focus on mathematical reasoning benchmarks. Could you provide results on other domains such as code generation or instruction following to validate the generalization of RLSC?

### Soundness
3

### Presentation
2

### Contribution
3
