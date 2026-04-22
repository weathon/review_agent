# OneReward: Unified Mask-Guided Image Generation via Multi-Task Human Preference Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
In this paper, we introduce OneReward, a unified reinforcement learning framework that enhances the model's generative capabilities across multiple tasks under different evaluation criteria using only One Reward model. By employing a single vision-language model (VLM) as the generative reward model, which can distinguish the winner and loser for a given task and a given evaluation criterion, it can be effectively applied to multi-task generation models, particularly in contexts with varied data and diverse task objectives. We utilize OneReward for mask-guided image generation, which can be further divided into several sub-tasks such as image fill, image extend, object removal, and text rendering, involving a binary mask as the edit area. Although these domain-specific tasks share same conditioning paradigm, they differ significantly in underlying data distributions and evaluation metrics. Existing methods often rely on task-specific supervised fine-tuning (SFT), which limits generalization and training efficiency. Building on OneReward, we develop a mask-guided generation model trained via multi-task reinforcement learning directly on a pre-trained base model, eliminating the need for task-specific SFT. Experimental results demonstrate that our unified edit model consistently outperforms both commercial and open-source competitors, such as Ideogram, Adobe Photoshop, and FLUX Fill [Pro], across multiple evaluation dimensions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces OneReward, proposing a unified reward model that are fine-tuned with preference data to achieve accurate assessment for reinforcement learning. The main contributions are two-fold: (1) a novel pipeline that considers task ID as a condition to achieve reward unification. (2) reinforcement learning pipeline with the proposed reward model to achieve image editing on various tasks (image inpainitng, outpaining, etc).  Extensive experiments demonstrate their methods' efficacy.

### Strengths
(1) OneReward aims to unify reward based on the given task, which is beneficial to achieve compact image editing training.

(2) The overall writting is good and method is easy to understand.

### Weaknesses
(1) One of the main contributions for this work is training a unified reward model to achieve comprehensive evaluation for reinforcement learning. However, improving generation quality could also be achieved by jointly using multiple reward experts. The paper lacks comparison or discussion with the ``mixture-of-reward-experts'' variant in Tab.2, which is proven to be effective in reinforcement finetuning [a].

(2) In Tab.1, recent VLMs such as [b] is also capable of comparing two images based on the given instruction. It is necessary to compare your uni-reward model with them.

(3) I am curious about the method's  training efficiency. Specifically, the reward is assigned on the final generated results, which means a full-time sampling before obtaining a reward. The whole process is time-consuming, especially when being deployed on FLUX-based  models. 

[a] T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT. In NIPS'25.
[b] Q-Insight: Understanding Image Quality via Visual Reinforcement Learning. In NIPS'25.

### Questions
Please refer to "weaknesses"

### Soundness
2

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
The paper introduces "OneReward", a unified reinforcement learning framework for enhancing generative models, particularly in mask-guided image generation tasks. By employing a single VLM as the reward model, the framework allows the model to efficiently distinguish between the winner and loser in multi-task settings with varied data distributions and evaluation metrics. The proposed model aims to eliminate task-specific fine-tuning (SFT) and offers a unified approach for tasks such as image inpainting, object removal, image extension, and text rendering. Experimental results demonstrate that OneReward outperforms both commercial and open-source competitors across several metrics. The model offers improvements over existing generative models, enhancing flexibility, scalability, and performance.

### Strengths
The introduction of OneReward as a unified reward model for multiple image editing tasks is highly innovative, addressing a clear gap in current image generation models that rely on task-specific fine-tuning. The paper provides strong experimental evidence showing that OneReward outperforms several competitive methods across multiple tasks. The ability to apply OneReward to diverse tasks like image fill, object removal, and text rendering without needing task-specific fine-tuning is a significant strength. This work could significantly advance the field of multi-task generative models, serving as a new baseline for future research.

### Weaknesses
The multi-task reinforcement learning framework is complex and could be challenging to train and fine-tune, especially on large-scale datasets. The paper could have delved deeper into potential limitations of the framework, particularly with respect to edge cases where the model might struggle with certain tasks or where performance might degrade. The training time and computational cost could also be discussed further, as multi-task reinforcement learning can be computationally expensive. While the experimental results are convincing, a more detailed analysis of the failure cases or scenarios where OneReward may not perform as expected would be useful.

### Questions
1.Can the OneReward framework be adapted to other domains beyond image editing, such as video generation or more complex multimodal tasks?
2.What are the specific computational costs associated with training the OneReward model, and how does this compare to task-specific models in terms of efficiency?
3.In multi-task reinforcement learning, how do you ensure the model doesn't overfit to one task at the expense of others? More details on the regularization methods used to balance task performance would be helpful.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes a unified reinforcement learning framework that uses a single vision-language model as a generative reward model to supervise multi-task image generation under varied evaluation criteria. Applied to mask-guided editing (image fill, extension, object removal, text rendering), it replaces task-specific SFT with multitask RL on a pretrained base model, despite differing data distributions and metrics. Experiments show the unified edit model consistently outperforms commercial and open-source systems across multiple evaluation dimensions.

### Strengths
- The idea of leveraging powerful VLMs as reward models is promising.

- Internal model shows promising results, even outperforming commercial models.

- The authors propose a high-quality preference model and dataset, which will be open-sourced.

### Weaknesses
Thanks to the authors’ efforts in this manuscript and their commitment to open-sourcing the data and model to benefit the community. However, I have several concerns about the work.

- Because VLMs such as Qwen-2.5-VL are heavily pre-trained, they should be treated as general reward models. The substantial jump from baselines to the *no-finetune* setting in Table 2 supports this. However, the proposed fine-tuning yields only marginal gains over *no-finetune*, which substantially weakens its contribution to the preference dataset.

- In the open-source setting, key fair comparisons with RLHF baselines (e.g., DanceGRPO, MixGRPO, Pref-GRPO) are missing. 

- Because the main qualitative results omit text rendering, some claims appear overstated (e.g., in the Abstract). The logical organization should be revised.

- Lacks comparison with general-purpose image editing models, such as MagicBrush, Qwen-Image-Edit, and Gemini-Banana.

- The ablation studies are insufficient. For example, the impact of probability-based data sampling is not analyzed, and the per-task contributions are unclear. Given the authors’ statement that "different sub-tasks differ significantly in underlying data distributions and evaluation metrics," this raises concerns about potential task conflicts, which may misguide the community’s research direction.

### Questions
The high-level motivation is to leverage VLMs as reward models to guide generation. Why not compare against unified understanding-and-generation models?

### Soundness
1

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
3

### Summary
This paper introduces OneReward, a smart framework that uses a single, powerful VLM as a flexible "judge." By telling this one reward model what task and what metric to evaluate, it can be used to train a single generator for multiple, conflicting tasks. This unified model ends up beating specialized SOTA competitors like Adobe Photoshop and FLUX Fill.

### Strengths
- Using a single, promptable VLM as a multi-task, multi-metric judge is a brilliant, scalable idea.

- They successfully trained one model to do conflicting tasks and win.

- It outperforms strong commercial and open-source competitors in head-to-head comparisons.

### Weaknesses
- The entire system relies 100% on the VLM (Qwen2.5-VL) being an accurate and unbiased judge.

- It's unclear how a single generator learns to be good at opposite goals simultaneously.

- They had to build a massive, 120k-pair multi-dimensional preference dataset, which is very hard to reproduce.

### Questions
How does the single generator model handle the conflicting objectives? Does getting better at "object removal" ever make it worse at "image fill"?

### Soundness
3

### Presentation
3

### Contribution
3
