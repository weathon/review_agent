# SkillFactory: Self-Distillation for Learning Cognitive Behaviors

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Reasoning models leveraging long chains of thought employ various cognitive skills, such as verification of their answers, backtracking, retrying by an alternate method, and more. Previous work has shown that when a base language model exhibits these skills, training that model further with reinforcement learning (RL) can learn to leverage them. How can we get models to leverage skills that aren't exhibited by base models? Our work, SkillFactory, is a method for fine-tuning models to roughly learn these skills during a supervised fine-tuning (SFT) stage prior to RL. Our approach does not rely on distillation from a stronger model, but instead uses samples from the model itself, rearranged to provide training data in the format of those skills. These "silver" SFT traces may be imperfect, but are nevertheless effective for priming a model to acquire skills during RL. Our evaluation shows that (1) starting from SkillFactory SFT initialization helps a model to generalize to harder variants of a task post-RL, despite lower performance pre-RL; (2) cognitive skills are indeed used by the model; (3) RLed SkillFactory models are more robust to regression on out-of-domain tasks than RLed base models. Our work suggests that inductive biases learned prior to RL help models learn robust cognitive skill use.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SkillFactory, a self-distillation method that trains small models to learn reasoning skills like reflection and retrying by rearranging their own outputs into structured “silver” traces. After supervised fine-tuning and reinforcement learning, the model shows improved reasoning and generalization on Countdown and OOD tasks without relying on larger teacher models.

### Strengths
1. The paper is well-written and clearly organized, making it easy to follow.

2. The data generation process is detailed, combining multiple prompting strategies to self-elicit diverse cognitive skills in small models as well as to ensure diversity.

3. The experiments are comprehensive, covering both in-domain and out-of-distribution evaluations, with detailed length analysis and ablation studies that strengthen the empirical support.

### Weaknesses
1. While the idea to equip models with cognitive skills is clear, the current formulation focuses on a small, pre-defined set of tagged skills (e.g., retry and reflection). As these skills are incorporated through explicit templates and tags, it limits the generalization to other cognitive behaviors beyond these predefined patterns.

2. SkillFactory involves both SFT and RL stages, whereas most baselines rely on only one of these training efforts (except STaR). The training effort is not consistent across these methods which leads to unfair comparison.

3. Although SkillFactory’s main strength emerges after the RL stage, its SFT performance remains notably below the R1-distilled baseline (Table 1). While R1-distilled traces are expected to yield stronger results even after RL, the paper does not report such comparisons (both in-domain and OOD), making it difficult to assess how close SkillFactory’s trajectories come to matching teacher-distilled SFT data.

### Questions
1. Countdown-3arg seems to be a relatively easy dataset, with performance saturating quickly after training. It seems that SkillFactory assumes the base model is at least sufficiently capable to generate several correct answers during sampling. Would the method still be effective if the base model struggles to produce correct outputs on more challenging datasets?

2. The experiments are conducted using a single model (Qwen2.5-1.5B-Instruct) trained on one dataset (Countdown). Have the authors trained on harder datasets or larger models to assess its generalizability?

3. How is the number of reflection/retry steps determined when composing a SkillFactory trajectory?

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
This paper presents a framework for teaching language models cognitive reasoning skills without requiring stronger teacher models. The key insight is that models can learn these skills from rearranged "silver" traces constructed from their own outputs. The method involves three stages: (1) sampling diverse solutions and reflections from a base model, (2) rearranging these into structured traces with explicit skill markers using tags, and (3) supervised fine-tuning followed by reinforcement learning. SkillFactory achieves substantial improvements on harder variants and better generalization to out-of-distribution tasks.

### Strengths
- The idea of creating structured "silver" training data by rearranging a model's own outputs is impactful. 
- The paper proposes a sound training pipeline that shows compelling generalization evidence to other tasks. Experiments are done in extensive settings. (RL Only, STaR, BOLT, R1 Distillation).

### Weaknesses
- The entire study uses only Qwen2.5-1.5B-Instruct. Larger models (7B+) may already exhibit these skills naturally. In addition, larger models might have better performance on some baselines’ settings (such as reinforcement learning). The generalizability of conclusions in this paper beyond 1.5B parameters is highly uncertain.
- Training exclusively on Countdown 3-arg (where solutions are easy to verify but hard to find) is an ideal scenario for reflection/verification skills. The paper lacks discussion of how SkillFactory's effectiveness changes for tasks where verification itself is difficult, subjective, or computationally expensive (e.g., creative writing, legal arguments).

### Questions
- Have you tested with at least one larger model (e.g., Qwen2.5-7)? This is critical to understand whether SkillFactory remains beneficial at practical scales where models may already exhibit some skills naturally. 
- Have you considered an experiment where SkillFactory is trained on a mixture of tasks (e.g., Countdown + GSM8K + Multiplication) rather than just Countdown? This would clarify whether the approach is domain-general or specific to search-like problems, which could change my assessment of the method's practical utility.

### Soundness
2

### Presentation
4

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
This work presents a self-distillation framework that constructs silver SFT traces by sampling multiple responses from a base model (without instruction following fine-tuning), forming a long context-style SFT data with reflections. The proposed method conducts experiments that use these silver SFT data as a warm-up of the RL stage. Experiments show the model trained with warm-up data can have better OOD performance on harder variants of toy and real tasks, such as Countdown, 3-digits multiplication.

### Strengths
1. The writing is clear and well-organized. 
2. Related work is comprehensive, and the authors carefully position SkillFactory relative to RL-only, distillation from stronger models, and self-distillation methods.

### Weaknesses
1. The title "SkillFactory" suggests a broad capability to learn diverse cognitive skills, but the implemented pipeline focuses on retrying and reflection. As L171 states, the proposed method has three steps: sampling diverse solutions, generating reflections, and assembling structured traces. It does not convincingly involve a wide variety of skills beyond long CoT with explicit verification and retry. 
2. Several prior works adopt a similar idea of sampling multiple attempts and assembling them into longer, structured traces with reflections, such as injecting "wait" to force the model to think more or combine multiple traces. The author discussed these methods in the related works, and makes the claim that "SkillFactory is similar to these methods, but focuses on generating data entirely from the base model and highlights that structure is key for the generalization of consistent skill use." Firstly, I appreciate that the author made a clear statement of the relation against prior works. However, this motivation may be insufficient for a top-tier conference paper. Why does a similar idea purely rely on the base model to make a new method?
3. The main results take the Qwen2.5-1.5B-Instruct and focus on toy tasks, like Countdown, digit multiplication, and Letter. This limits the strength of the claims about general-purpose reasoning skill acquisition. Results on widely accepted long-CoT benchmarks, such as AIME/AMC/GPQA-Diamond with 7B size models, would substantially improve the paper.

### Questions
see weakness

### Soundness
2

### Presentation
3

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
The paper proposes a synthetic data generation technique for self-distilling reasoning behaviors into language models. The authors show that their method, SkillFactory, helps LLMs learn these behaviors by training on the game of countdown. they also show transfer to other domains like Common Sense QA, GSM8k etc.

### Strengths
- The paper is clearly written and the method is straightforward.
- The generalization performance of the model trained on 3 dig countdown is impressive.
- The method matches or surpasses distillation from stronger models.
- The paper also has good ablations and baselines.

### Weaknesses
- The biggest weakness of the paper is that all experiments are with the qwen2.5-1.5B model. Adding a model from another family and a different model size would help show the generalizability of the method.
- I have some questions about circularity: How do you reconcile necessity of the method if behaviors are already present in the pretraining data? “skills surface less consistently.” If the model can generate correct solutions and reflections (required for silver traces), why can't RL alone elicit these behaviors?
- If the reasoning behaviors aren’t present in the pretraining data, how does the model produce those?
- The paper mentions silver traces "may contain errors" but provides no analysis of error rates, types, or impact on learning.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
