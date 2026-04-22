# Task Tokens: A Flexible Approach to Adapting Behavior Foundation Models

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Recent advancements in imitation learning for robotic control have led to transformer-based behavior foundation models (BFMs) that enable multi-modal, human-like control for humanoid agents. These models generate solutions when conditioned on high-level goals or prompts, for example, walking to a coordinate when conditioned on the position of the robot's pelvis. While excelling at zero-shot generation of robust behaviors, BFMs often require meticulous prompt engineering for specific tasks, potentially yielding suboptimal results. In this work, we introduce ``Task Tokens'' - a method to effectively tailor BFMs to specific tasks while preserving their flexibility. Our approach integrates naturally within the transformer architecture of BFMs. Task Tokens trains a task-specific encoder (tokenizer), with the original BFM remaining untouched. Our method reduces trainable parameters per task by up to $\times 125$ and converges up to $\times 6$ faster compared to standard baselines. In addition, by keeping the original BFM unchanged, Task Tokens enables utilizing the pre-existing encoders. This allows incorporating user-defined priors, balancing reward design and prompt engineering.
We demonstrate Task Tokens' efficacy across various tasks, including out-of-distribution scenarios, and show their compatibility with other prompting modalities. Our results suggest that Task Tokens offer a promising approach for adapting BFMs to specific control tasks while retaining their generalization capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Task Tokens, a parameter‑efficient adaptation mechanism for goal‑conditioned behavior foundation models (GC‑BFMs) such as MaskedMimic. A lightweight task encoder maps task observations to a learned token that is appended to the BFM’s input sequence alongside optional human priors; the BFM is kept frozen, and PPO updates only the task encoder via gradients flowing through the frozen model. Across several humanoid control tasks, Task Tokens achieve competitive success rates to full fine‑tuning with far fewer trainable parameters, improved robustness to friction/gravity perturbations, and preserved multi‑modal prompting. A human study indicates motions are generally more human‑like than PPO/AMP and fine‑tuned MaskedMimic.

### Strengths
The paper is well‑motivated. It clearly articulates the gap between prompt‑only GC‑BFMs and reward‑driven task optimization, and proposes a neat integration that leverages the transformer token interface without touching the backbone. Freezing the foundation model to avoid collapse/forgetting while steering behavior through a small encoder is a sound and practical design choice to preserve the motion manifold. Empirically, the method is consistently competitive with full fine‑tuning while using ~200k parameters per task, and it outperforms standard RL and AMP/PULSE on several tasks. The study on combining learned tokens with human priors (orientation and text prompts) convincingly demonstrates directability that full fine‑tuning appears to erode. The experimental section is generally decent with five seeds, ablations on the task‑encoder architecture, OOD tests, and a user study. I especially appreciate the explicit discussion of limitations around human‑likeness and reliance on the pretrained BFM compared to PULSE.

### Weaknesses
My main issue is the overstatement of some contributions. The method is a logical adoption of ideas from NLP to BFMs, RL training for these tasks is a sensible way to tune the performance of the model, and the results are decent. There is no need to exaggerate other parts of the paper.

Statements on the criticality of using up to “×125 fewer parameters” seem misplaced when the baseline methods train only approximately 25M parameters. This is especially true when training on humanoid simulations, where the sim is also likely a major bottleneck. If the efficiency gains are truly massive, it would be great to have a comparison of GPU‑hours, memory, and inference latency. In addition, it seems like the baseline fine-tuning does not employ standard parameter‑efficient fine‑tuning methods inside the BFM (e.g., LoRA/adapters), which would be the real baseline when talking about efficiency.

The robustness claims are also inflated. “Order of magnitude” gains at very low friction values are only caused by the baseline going slightly faster towards zero. At that point, any small increase in success rates (i.e. going from 0.01% to 1%) is going to be orders of magnitude larger. Given the huge standard deviations, it is even questionable if the result is statistically significant.

Finally, scalability remains unclear: training one encoder per task is simple but offers no multi‑task, compositional, or continual learning path, and the claim that fine‑tuning catastrophically forgets multi‑modal prompting is largely anecdotal without a quantitative retention test on the original prompts.

### Questions
- Figure 3 shows several saw-tooth-like drop‑offs for MaskedMimic fine‑tune and Task Token. What causes these instabilities?
- AMP often trails PPO and even fails on Direction/Steering. Why is that when it’s specifically designed for these scenarios? Did you retune AMP per task?
- Efficiency: please report wall‑clock training time (GPU‑hours) and inference throughput for Task Tokens vs. full fine‑tune and PPO/AMP. Does the 6× speedup in steps translate to real time?
- Could you compare against PEFT baselines applied inside the BFM (LoRA, adapters) with comparable parameter budgets? Where does Task Tokens win/lose relative to these?
- Section 4.4 mentions two phases (direction prior then semantics “kick”). Are you using multiple task token adapters here or just multiple prior tokens with a single task token? If multiple task tokens are used, how are they scheduled/combined?
- Token analysis: do task tokens cluster across runs/rollouts and across tasks? Visualizing the resulting token embeddings across training runs for different tasks could reveal whether clustering suggests scalability.

Overall, this is a well‑motivated and seemingly effective adaptation mechanism that preserves the strengths of GC‑BFMs. Addressing the missing PEFT baselines, clarifying efficiency in wall‑clock terms, explaining AMP’s underperformance, and adjusting the framing of some sections would strengthen it further.

### Soundness
3

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
2

### Summary
The paper proposes an approach to adapt pre-trained Behavior Foundation Models (BFMs) to downstream tasks by training "task tokens" -- special tokens which are added to the prompt of BFM, which represent the task-specific information. These tokens are trained via RL from environment rewards. They are modeled as neural networks which map task goals to a fixed tokens dimension.

The authors conduct multiple experiments to demonstrate the efficiency of their approach compared to baselines -- finetuning BFMs directly, pure RL without BFMs and other approaches which make use of motion capture data.

Overall, the proposed approach is simple, concise and efficient.

### Strengths
* The approach is sound, makes sense, and simple
* The experimental evidence suggests that it works well in practice, despite being much cheaper than alternatives
* The approach essentially makes use of "prompt engineering" by making it automatic and training "tokens" from data

### Weaknesses
* I think the technical presentation of the paper could be improved:

** The "task tokens" is the major contribution of the paper, but it does not spend enough time discussing different design choices regarding "Task encoder" architecture. It feels that authors just tried a few simple things and it "just worked". However, I think the research community will benefit greatly from a proper analysis of the different impacts of design choices in such architecture. I.e., do they use some form of normalization ? What are activation functions? Are there residual connections and if not, why not? I think also trying to conduct a more careful study here will end up in a method which eventually performs even better than what the authors have

** Table 1 looks a bit suspicious to me. The results of the baselines are inconsistent. You see that PPO performs well on Strike, Reach and Direction but achieves significantly lower performance on Steering and Long Jump. Similar can be said about other methods. Why is that? What it looks to me could be that the authors did not spend enough effort trying to make sure that the baselines are optimized carefully.

Minor: Since the proposed method is much cheaper than the baselines, it would be very helpful to have a figure which shows total FLOPs spent compared to baselines.

### Questions
* What are the design choices which are important for the task encoder? Can you do a more careful scientific study here so that readers understand what things matter?

* Can you explain discrepancies in Table 1? Why baselines numbers are inconsistent?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents "Task Tokens," a novel framework for adapting large-scale Behavior Foundation Models (BFMs), such as MaskedMimic, to specific downstream tasks. They propose to use a parameter-efficient approach that trains a new, lightweight "Task Encoder" for each specific task using reinforcement learning (PPO) while freezing the pretrained network to avoid catastrophic forgetting. This small encoder maps current task-specific goals (e.g., target coordinates) to a "Task Token." This token is then concatenated with the BFM's standard inputs (like state tokens and user-defined "prior tokens") to adapt the BFM's behavior

The proposed method allows the PPO gradient to flow through the frozen BFM, updating only the small Task Encoder. This encoder learns to "steer" the general-purpose BFM to solve the specific RL task while keeping the BFM's learned prior of natural human motion. The method is up to 125x more parameter-efficient and converges 6x faster compared to finetuning the whole BFM network. It also preserves the BFM's robustness in out-of-distribution (OOD) scenarios where standard fine-tuning fails.

### Strengths
**Elegant and Novel Solution to Important Problem** The main contribution is the application of a light-weight task encoder to steer large-scale behavior models. The method, by design, prevents catastrophic forgetting by freezing the core BFM. This is an elegant and, to my knowledge, novel solution (at least in the field of RL and robot learning) to address the balance of general policy and specific task performance. The effectiveness of such an approach opens up the opportunity to better control the behavior of robot foundation models.

**Solid Empirical Validation** The experiments show the proposed method is highly parameter-efficient, and it also preserves the foundation model's robustness in some cases compared to finetuning (see Figure 4). The human study (Table 2) confirms that the resulting motions are perceived as more human-like than those from baselines like PPO, AMP, or naive fine-tuning, demonstrating that the BFM's prior is indeed preserved. The framework (Sec 4.4, Fig 5/6) also allows for a seamless combination of two control modes: user-defined "Prior Tokens" (like text or joint conditions) provide high-level style guidance, while the learned "Task Token" optimizes for the low-level, reward-specific objective. This hybrid approach is highly flexible and a step forward in usability.

### Weaknesses
**Limited Task Complexity:** This is the main weakness. The tasks evaluated (Direction, Steering, Reach, Strike, Long Jump) are primarily reactive motor skills or goal-reaching tasks. While diverse, they do not appear to require complex, long-horizon composition of tasks. It is unclear if the "Task Token" approach, which optimizes a single, continuous token, is sufficient to adapt the BFM to tasks requiring multi-stage sequential logic (e.g., "find the block, pick it up, and place it in the correct box").

**Reward and Human-likeness Trade-off:** The human study (Table 2) reveals that Task Tokens were consistently rated as less human-like than PULSE. The authors suggest PULSE "constrains the high-level representation to stay closer to the prior." This implies that the RL-driven Task Token, in its pursuit of maximizing the reward, may be finding solutions that are on the edge of the BFM's natural motion manifold—less human-like, but more optimal for the task.

**Scalability:** The current approach trains one separate Task Encoder for each new task. The paper's scalability claim (Contribution #2) refers to *per-task* efficiency (a small ~200k encoder vs. a 25M model). It also highly depends on the reward definition to adapt to certain tasks.

### Questions
Besides some questions raised in the weakness section, I have some additional questions:

Q1. It's not very clear to me what exactly the inputs to the task encoder are. Are they the same observation as the BFM? Is some goal information concatenated to the task encoder?

Q2. Is the BFM sensitive to the exact position of the task tokens in the input sequence?

Q3. Could there be some task tokens that represent some more "generation" modifications to the model? This reminds me of a paper altering the behavior of VLA models via some "activation steering" [1]. For example, one might train a task encoder that makes the humanoid move faster or move with lower torso height, and these encoders may be useful to alter the model's behavior in different tasks, along with more task-specific encoders.

Reference: 
[1] Häon, B., Stocking, K.C., Chuang, I., & Tomlin, C.J. (2025). Mechanistic interpretability for steering vision-language-action models. ArXiv, abs/2509.00328.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this work, the authors propose Task Tokens, where a pre-trained behavior foundation model is then adapted for a new task by using RL to learn a conditioning latent. In this method, the authors first train a behavior foundation model that takes a "goal" through some task encoder as a conditioning. Then, given a new task specified through a reward function, the task token itself is optimized with RL. The authors evaluate this method through task performance and efficiency. In addition, they analyze the stability and the human preference of the results. In these performance analyses, the authors show favorable results for their method.

### Strengths
1. The demonstrated performance of the model in the videos is quite impressive.
2. The idea itself is simple and leads to impressive results.
3. The authors evaluate it on a diverse set of tasks and on good set of robustness evaluation.

### Weaknesses
1. The method is only evaluated in one setup with one behavior foundation model, MaskedMimic. Evaluation in other settings, such as manipulation, would go a long way to establish the generality of this method.
2. It is hard to evaluate how "new" each of the elicited behaviors are. I.e. how well can this method elicit behaviors that were and were not in the training set of the original training dataset of the BFM? Especially tasks like directions should be in the dataset, and yet the Joint Conditioning method fails to elicit this behavior, which makes it seem the JC method is not great at eliciting pre-trained behavior.
3. The authors make the "human-like" behavior of the prompting method a big part of the study, but devote very little space to understanding the difference between this and PULSE which outperforms task tokens. A further understanding of this difference would be appreciated.
4. In figure 3, the Y axis label is in % but the labels are between (0-1).

### Questions
1. What are the major challenges in adapting this method to a manipulation setup?
2. How well can you teach this method tasks that were not in the original dataset? I imagine full fine-tuning can learn any new tasks, vs. without fine-tuning you will not be able to learn something sufficiently off-distribution. But how can we evaluate it?
3. What's the reason behind this method taking equivalent amount of environment steps to reach success as full-fine tuning? My intuition says you should be able to reach comparable performance much faster since this is RL in a much smaller space.

### Soundness
2

### Presentation
3

### Contribution
2
