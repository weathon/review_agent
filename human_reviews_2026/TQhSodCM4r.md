# SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Vision-Language-Action (VLA) models have emerged as a powerful paradigm for robotic manipulation. Despite substantial progress enabled by large-scale pretraining and supervised fine-tuning (SFT), these models face two fundamental challenges:
(i) the scarcity and high cost of large-scale robotic trajectories required for SFT scaling,
and (ii) limited generalization to tasks under distribution shift.
To overcome these limitations, we explore reinforcement learning (RL) as a pathway to scaling VLA training beyond limited datasets.
Inspired by LLM breakthroughs where RL with outcome rewards enhances step-by-step reasoning, we ask: Can outcome-driven RL improve long-horizon step-by-step action planning of VLA?
In this work, we introduce SimpleVLA-RL, an efficient RL framework tailored for VLA models.
Building upon veRL, we introduce VLA-specific trajectory sampling, scalable parallelization, multi-environment rendering, and optimized loss computation.
Applied to OpenVLA-OFT, SimpleVLA-RL achieves 99\% of SoTA performance on LIBERO and 80\% relative
improvement on RoboTwin 1.0\&2.0, outperforming $\pi_0$ with our proposed exploration-enhancing strategies.
SimpleVLA-RL reduces dependence on large-scale data, enables robust generalization, and remarkably surpasses SFT in real-world tasks.
Moreover, we identify a novel phenomenon "pushcut'' during RL training, wherein the policy discovers unseen patterns beyond those seen in previous training process.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes SimpleVLA-RL, a practical online RL framework for Vision-Language-Action (VLA) models that scales beyond limited demonstration data. Building on veRL, the authors implement VLA-specific interactive rollout (closed-loop env stepping with token-based action distributions), simple outcome-only (0/1) trajectory rewards, and GRPO training with three exploration enhancers: dynamic sampling (discard all-success/all-failure groups), higher clipping range (0.8–1.28), and higher rollout temperature (1.6). They also drop the KL penalty (no reference model) to reduce memory and encourage exploration. Experiments fine-tune OpenVLA-OFT and then apply online RL across LIBERO, RoboTwin-1.0, RoboTwin-2.0, plus real-robot sim2real tests. Results show ~99% avg SR on LIBERO, large gains on RoboTwin (e.g., 38.3→68.8% overall on RT-2.0; 39.8→70.4% on RT-1.0), strong data-efficiency (with one demonstration per task: 48.9→96.9% on LIBERO), improved generalization to unseen spatial/object/goal tasks, and notable sim2real improvements.

### Strengths
1. Clear, simple RL recipe for VLAs (outcome-only rewards + GRPO) that removes reward-engineering and reference-model overhead; the method details (interactive rollout, reward assignment, GRPO objective) are explicit and easy to reproduce.

2. Exploration enhancements are well-motivated and ablated (dynamic sampling / higher clip / higher temperature), each contributing sizable gains. 

3. Strong, broad empirical gains: SOTA-level LIBERO (~99% avg), large improvements across short/medium/long/extra-long horizons on RoboTwin-2.0, and substantial jumps on RoboTwin-1.0. 

4. Data efficiency: with one traj per task, RL lifts LIBERO-Long from 17.3%→91.7% and overall 48.9%→96.9%, nearly closing the gap to full-data training. 

5. Generalization: RL improves OOD tasks while SFT overfits/forgets—documented across spatial/object/goal suites with learning-progress plots.

### Weaknesses
(1) pic 5.1 is a bit confusing for me. I am not sure why the upper half has the different background to the bottom half. I think the best way to show emergent behavior is comparing behaviors under same conditions.

(2) Compute/throughput transparency: The paper lacks precise reporting of wall-clock training time, GPU type.

(3) Ablations indicate RL fails when the base success rate is ~0% and remains weak <5%; this limits plug-and-play use in very hard tasks without warm-starts.

### Questions
please see the weakness and try to answer them, especially weakness 3. I am curious about if you explored how to deal with it.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes **SimpleVLA-RL**, a framework designed to overcome the dependence of Vision-Language-Action (VLA) models on large-scale demonstration data during supervised fine-tuning. Building upon the veRL framework, the authors introduce tailored optimizations for reinforcement learning on autoregressive VLA models and validate their approach through both simulation benchmarks and real-world robotic experiments.

### Strengths
The paper is well-written, clearly structured, and presents a concise yet precise problem formulation. The experimental design is robust and the results are solid, demonstrating substantial improvements over existing state-of-the-art models across multiple benchmarks. Notably, the experiment using only a single demonstration during the RL phase provides compelling evidence that reinforcement learning can effectively mitigate the challenge of demonstration data scarcity.

### Weaknesses
Although the experimental results are comprehensive and detailed, the methodological innovation appears somewhat limited. The proposed parameter adjustments effectively improve performance, yet the paper does not sufficiently justify the suitability of directly transferring RL algorithms from large language models to VLA settings. 

Moreover, the removal of the KL regularization term lacks supporting ablation studies, and the sparsity of the outcome-based reward may weaken the gradient signal, potentially affecting the stability of policy optimization. Furthermore, while the “Pushcut” phenomenon is intriguing and conceptually inspiring, it is demonstrated only in simulation, without quantitative validation in real-world settings—thus its generality and robustness remain uncertain.

### Questions
1. The current RL framework is validated primarily on autoregressive discrete-action VLAs. Have the authors considered extending SimpleVLA-RL to continuous-action VLA models?
2. Since the proposed method primarily relies on training in simulation and transferring to the real world, have the authors examined the potential sim-to-real gap? 
3. The current generalization experiments focus primarily on task and scene transfer. Have the authors considered evaluating language generalization—for example, testing the model’s robustness to paraphrased or diverse natural-language commands within the same scene?
4. The authors remove the KL regularization term and adopt a sparse outcome-based reward. While this design may encourage greater exploration, it could also weaken policy constraints and optimization signals. Have the authors observed any instability or performance oscillation during training as a result?
5. The current RL training primarily focuses on in-domain tasks that share similar distributions with the supervised fine-tuning data. Have the authors considered performing RL training on out-of-domain tasks to further evaluate the model’s generalization ability and policy stability under distributional shifts?

### Soundness
3

### Presentation
3

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
The paper demonstrates how advances in LLM post-training can be ported over to robotics domains by finetuning VLA models in simulation with simple modifications to GRPO. Specifically, the paper introduces best practices to encourage exploration and ensure stable training, such as dynamic sampling of batches to ensure only informative gradients are propagated, best practices for clipping policy updates, and high-temperature sampling. The approach only uses sparse rewards. The paper demonstrates impressive improvements over SFT, and then demonstrates these improvements hold when transferring back to the real world.

### Strengths
$\textbf{Simplicity + Scalability}$: The approach is extremely simple and scalable, and I expect the community to adopt and build on the work. Specifically, the approach is much simpler than recently proposed techniques for training diffusion-based policy heads, and opens the door for porting over new insights from LLM finetuning to robotics. 

$\textbf{Strong Results}$:  To my knowledge, this is the first work to demonstrate reliable improvement with RL for VLAs in simulated domains. The paper demonstrates the ability of the method to achieve substantial improvements in success rates across a range of tasks, demonstrates some ability to generalize to out of distribution tasks, and demonstrates these improvements actually transfer to the real world.

$\textbf{Thorough Evaluation}$: The paper has a strong evaluation and clearly conveys the importance of each of the design decisions. 

$\textbf{Timliness:}$ In short, this is a solid, timley submission and an easy acceptance for me.

### Weaknesses
$\textbf{Additional Related Work}$: The discussion of related work focuses primarily on other approaches for post-training VLAs. However, the paper would be strengthened by additional discussion one recent works which finetune diffusion policy heads, such as Diffusion Policy Policy Optimization. We know that modifications to GRPO are effective at post-training LLMs, but to me the most surprising thing about this paper is that it was able to stably optimize through a diffusion head. Some additional intuition on why this simple approach works so well would be nice.

### Questions
Can the authors provide some intuition for why they were able to effectively optimize through diffusion-based policies with such a simple approach?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SimpleVLA-RL, an RL framework built on top of VeRL to train Vision-Language-Action (VLA) models for robotic manipulation tasks. The work addresses two fundamental challenges in current VLA training: (1) the scarcity and high cost of large-scale robotic trajectory data required for supervised fine-tuning (SFT), and (2) limited generalization to tasks under distribution shift. SimpleVLA-RL leverages all recent findings from LLM RL finetuning literature such as GRPO with dynamic sampling, variable clipping threshold, high temperature rollout sampling and no KL penalty to build a stable RL finetuning setup for VLA training. The paper itself does not introduce any new ideas or algorithms to enable RL training for VLA but shows that findings from LLM community generalize to robotics without any modifications.

### Strengths
1. Application of RL for training low-level control policies for manipulation to scale VLA training without relying on  large amounts of demonstration data.
2. The experimental results clearly show performance improvements achievable through RL training are significant
3. The analysis section is interesting and studies valuable questions with well thought out experiments

### Weaknesses
1. The experiments section doesn’t mention details about the training datasets and benchmarks. From the current description it is a bit unclear to me on what task, episodes, and bechmarks was the RL finetuning for the VLA policies done? My understanding is OpenVLA-OFT was first trained on real world robotics datasets and then for benchmarking on libero it was finetuned with small amounts of simulation only trajectories. I would like to understand for experiments presented in tables 1, 2 and 3 did the authors use the SFT finetuned OpenVLA-OFT model as initial checkpoint for RL or RL is performed directly on base model (aka not finetuned on sim data with SFT).
2. Another concern I have is the SFT finetuned results for full libero dataset experiment used by OpenVLA-OFT in the paper does not match the results presented in the original paper table 2 here https://arxiv.org/pdf/2502.19645. I would like to ask authors to clarify why there is such a big gap (ex: Spatial SR is 96.2 in original paper vs 91.6 in table 3 and 4). 
3. The experiments presented in analysis section 4.2 are definitely interesting and valuable. However, the details about how the amount of compute/data used for both SFT and RL is a confounding variable that is unaccounted for. Can authors please describe the exact amount of model updates, trajectories used for this comparison? I believe such a comparison only makes sense when done under a fixed budget of training updates to get a clear idea.

### Questions
On L177 authors claim that critic free RL algorithms suffer from vanishing gradients problem. Can authors add a supporting citation for this? I am unsure how true this claim is. Critic free RL algorithms suffer from high-variance but there is no inherent issue in these algorithms that causes vanishing gradients. If there are no supporting citations for this statement I would suggest the authors to remove this line.

This paper does not propose any novel ideas to enable RL for VLAs. It can be seen as more of application of recent techniques used in RL finetuning for LLMs applied directly to VLA finetuning. However, it does have valuable experiments and insights that most of the techniques used in making RL training stable and effective for LLMs generalize to VLA finetuning in the current setup. That is why I recommend the paper to be accepted conditioned that authors address the above mentioned weaknesses and minor issues in some claims made.

### Soundness
3

### Presentation
3

### Contribution
3
