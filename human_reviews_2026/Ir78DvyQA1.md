# Improve LLM Pre-training with RL-Guided Annealing

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Training large language models (LLMs) typically proceeds in two distinct stages: pre-training and post-training. However, the question of how to exploit these stages synergistically—particularly how post-trained models can inform and improve pre-training—remains underexplored.

We begin by analyzing training dynamics and identify the annealing (mid-training) phase as a critical turning point for the pre-trained base model’s capabilities. During this stage, high-quality corpora are introduced under a rapidly decaying learning rate, leading to a substantial shift in the base model’s probability distribution and a noticeable surge in performance.
Interestingly, while reinforcement learning (RL) during post-training induces only minor distributional shifts, it significantly enhances reasoning capabilities.
Motivated by this observation, we propose RL-Guided Annealing (RGA), a method designed to leverage RL-enhanced models, naturally produced during standard LLM training pipeline, to guide token weighting during the annealing phase.
Specifically, RGA transfers knowledge from the RL stage back to annealing by reassigning token-level importance weights based on the per-token loss differences between the base and RL models.
Notably, RGA does not require any specially trained teacher or reference model.

Across multiple model families, RGA consistently improves performance, achieving average gains of 5.21\%, 1.84\%, and 1.78\% on 10 pre-training benchmarks. It also boosts downstream performance after post-training by over 2\%.
These findings reveal a positive feedback loop between pre-training and post-training: RL-tuned models retroactively improve their foundational base models, which in turn support more effective RL—enabling a self-reinforcing path toward higher model quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose to guide the annealing stage of LLM pretraining with the RL-enhanced model, as they observed similarity between annealed and RL-enhanced models. They show that by using RL-enhanced model to produce per-token weight during pretraining, they are able to enhance the performance of LLM during annealing which persists after later stages of RL. Additionally, they show that the process can be applied repeatedly to further improve the performance.

### Strengths
1. Overall the presentation is clear and conveys a somehow reasonable story.

### Weaknesses
1. The scale of the experiment is limited. Furthermore, in the comparison, the improvement is significant for the 1B model, but the improvement is slight for 2B and 3B models.
2. While the specific setup might be novel, the general idea of distilling stronger models to improve the performance is long existing. Why use the RL-enhanced version of the same model instead of some larger/stronger models? How is the proposed per-token weight scheme compared to conventional soft-logit knowledge distillation? These should be discussed and compared.
3. Some part of the motivation is not entirely clear/convincing. Specifically, for the connection of annealing and RL, it is uncertain whether the similarity of annealed and RL-enhanced model is that strong/important as the authors claimed.

### Questions
1. Do you have results on larger models? OLMo 7B might be a good target, as the result on OLMo 1B is the most promising.
2. Do you compare with other conventional knowledge distillation methods? Does distilling from  the RL-enhanced version of the same model more helpful than using other RL-enhanced models.

### Soundness
3

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
3

### Summary
The paper aims to understand how post-trained LLMs can help inform and improve pre-training. To this end, the paper investigates the case of LLMs post-trained with RL. Specifically, the paper proposes to use the token log-likelihood of an RL-trained model to reweight tokens during pre-training of another model (detailed in Sec 3.1 and Algorithm 1). This is referred to as the RL-Guided Annealing (RGA). It is shown empirically that RGA-trained models (during pre-training) achieve better few-shot accuracy across multiple tasks (Table 1). Further, RGA-trained models, when going through further post training, also achieve better downstream performance (Fig 6).

### Strengths
The goal (i.e., improving LLM pre-training with an RL-trained model) is clearly conveyed and overall the paper is easy to read. Though, technical details can be better explained (see Weaknesses). How RL training (which happens after pre-training in practice) can help improve pre-training is not a well studied topic in the literature. This is an important research question to tackle. As far as I know, the specific token reweighting scheme proposed in Sec 3.1 is original (although the concept of token reweighting itself is an old idea).

### Weaknesses
The design choices in Sec 3.1 (for the proposed token reweighting mechanism) lack sufficient explanation and theoretical justification. Without further explanation, the proposed scheme appears arbitrary, and has limited novelty. Concretely, for instance, it is unclear why one has to center the mean of the log-likelihood differences by subtracting off $\\mu_\\Delta$ in (4). Similar questions for many design choices in (3), (4), (5) (more specific questions in Questions). 

No theoretical results are contributed. This is not a problem by itself if empirical results can support the claim. Unfortunately, I find the empirical results to be inadequate. Baseline methods are insufficient, and there ought to be more ablation study to see the effects of components in the proposed weighting scheme in (3), (4), (5). As a natural baseline, for instance, a quantitative comparison to standard distillation (from a teacher model) is missing. See more specific questions in Questions.

Overall, while the paper attempts to provide empirical evidence that the proposed method works, explanation on *why* it works is insufficient.

### Questions
**Questions (in order of importance)**

**Q1**:   L160 and Fig 3: 

> Moreover, many high-margin tokens are discourse connectives (e.g., ‘Therefore’, ‘So’, ‘gives’)”. 

Why are these tokens important for reasoning?  I also checked Fig 10 in the appendix. It is hard to grasp why the highlighted tokens have high relative weights according to the proposed weighting mechanism. Could you please elaborate?


**Q2**: L151: 

> The gap narrows substantially after annealing, indicating that the annealing stage is a critical turning point that drives a qualitative transformation of the base model.. 

Per experiments, I see that the proposed reweighting approach gives better downstream performance. But *why*? Understanding why is important for research. This question may be related to the next one.


**Q3**: What are the effects of each component of the proposed reweighting mechanism in Sec 3.1? Since little justification is given, these seem arbitrary. Concretely, 

**Q3.1**: Why center with $\\mu_\\Delta$?  Why not use the uncentered version? Why not even do standardization (i.e., divide by the standard deviation). To be clear, I am not asking the authors to run experiments with all these variants. I am asking why the particular choice was made. 

**Q3.2**: Why sigmoid in (5)? 

**Q3.3**: In (5), why set $\\epsilon=0.2$ in experiments? See line 302.

**Q3.4**: Why consider relative weight (i.e., the difference of log likelihoods of under two models) in (3)? Why not, say, consider only $-L_{RL}(x_t)$? That is, the token receives weight according to the log likelihood provided by the RL model. 

For all these questions, of course, one answer is “because it works in practice”. However, I would like to seek understanding and reasons for why it works. An empirical ablation study will help answer these questions. I think this kind of ablation study is missing. Also, it would be good to theoretically study for what objective these choices provide an optimal solution to.

**Q4**: *When* exactly do you do the proposed annealing algorithm? Is it after the end of pre-training (next-token prediction)? Or does one start doing annealing in the middle of pre-training? If the latter, when exactly? Related question: what exactly is the “Pre-Annealed” baseline at line 306?


**Q5**: In Fig 6, the paper claims that a model trained with the proposed RGA procedure gives better performance on downstream metrics after going through another post-training phase. Importantly, it is claimed that this gain transfer is regardless of the specific post-training technique used. Do I understand correctly? **My question**: is the gain transfer really specific to RGA-trained models, or does the gain transfer hold for any models that achieve good pre-training performance (regardless of whether they are trained with RGA)?


**Q6**: At a high level, the proposed method makes use of another trained LLM during pretraining. A natural baseline to compare to is thus distillation. In distillation, one pretrains an LLM by learning from the output token probability distribution of another teacher model (or a weighted combination of this and the ground-truth one-hot label) by minimizing the cross entropy. This natural baseline is missing. There are many distillation variants. See, for instance, this survey paper https://arxiv.org/pdf/2402.13116.


**Q7**: In the implementation, do you put a stop gradient operator on the weight $w_t^{(i)}$ in (6)?  Why or why not?


I appreciate that the authors are tackling this direction that has not received enough attention in the literature. However, there are missing details, justification, and results that prevent me from giving a higher  rating.


**Comments and suggestions**

* In the introduction,  the “annealing phase” or even the action of “annealing” itself are not made explicit. It is hard to understand the motivation of this work. It is not until Sec 3 that the reader finally understands what “annealing” refers to.

* Fig 2: Instead of token log probabilities, consider showing metrics that are more connected to downstream metrics. That way, it is easier to see the benefits.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces RL-Guided Annealing (RGA), a new LLM training method that bridges pre-training and post-training. Traditional pipelines train models through pre-training, fine-tuning, and followed by reinforcement learning. RGA proposes using the RL-tuned model to guide the mid-training during the pre-training phase (annealing stage where the learning rate decays and data quality increases). By comparing per-token losses between the base and RL models, RGA assigns token-weights, emphasizing those most influential to reasoning improvements. Experiments demonstrate consistent gain in pre-training benchmarks as well as downstream results.

### Strengths
* Intriguing idea using downstream training signals to influence the pre-training process, potentially closing the gap between stages that are usually isolated.
* The method is lightweight and easy to integrate. It introduces no new models or objectives, only a simple token-level reweighting mechanism that can fit neatly into existing training pipelines.

### Weaknesses
* Since pre-training typically targets broad, domain-agnostic language understanding, steering it with RL-trained signals might bias the model toward narrower reasoning or stylistic patterns, potentially harming generalization e.g. unseen tasks during the training including pre-training, fine-tuning and RL.
* The reported gains are relatively modest, especially given that the approach effectively requires a second pre-training phase, which can be computationally very expensive.

### Questions
* How sensitive is RGA to quality of the RL model used as reference. Does a weaker RL checkpoint still help, or could it misguide the annealing?
* What would happen if token reweighting were applied beyond the annealing stage, or throughout pre-training more continuously?
* How does RGA behave when the RL-tuned model is heavily domain-specific (e.g., math or code)—does it transfer well, or overfit to that domain’s distribution?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes RL-guided annealing (RGA) as a mid-training method aimed at better aligning the base model with the RL-tuned model, so that the annealed model becomes more receptive to later RL fine-tuning and achieves higher overall performance. The core idea is to use the RL model as a reference and re-weight tokens through importance sampling based on per-token loss differences, applied on a small set of high-quality corpora during the annealing phase. This alignment process between pre-training and post-training gives gains across several benchmarks.

### Strengths
The paper is generally well-written and easy to follow. The proposed RL-guided annealing method is novel and presents an interesting way to connect pre-training and post-training stages.

### Weaknesses
Please see Questions.

### Questions
1. While the method seems interesting and somewhat novel at first glance, there are a few concerns. It is not new to mid-train on data (for example, math) that the post-training procedure will later see (RL on math), in order to get a better base model for RL (as seen, for instance, in Qwen3). So it’s not entirely surprising that this setup gives some gains, which makes the extent of novelty here unclear. Moreover, Fig. 7 shows diminishing returns with successive RGA steps, raising questions about the scalability of the approach.

2. A crucial baseline seems missing. What happens if the compute spent on RGA is instead used to extend the RL phase? Also, it took me a while to realize that the “standard” baseline corresponds to mid-training on the same data without importance weighting — this could be stated more clearly.

3. It’s also not very clear under which context this method is actually useful. It feels like it might risk mode collapse, and Fig. 7 already hints at that. In practice, it’s hard to see when this would be preferable to simply spending the compute on standard mid-training or RL.

4. What is the motivation for Fig. 2? Since the annealing procedure explicitly aligns the model toward the RL model, that outcome seems almost tautological.

### Soundness
2

### Presentation
3

### Contribution
2
