# Toward Evaluative Thinking: Meta Policy Optimization with Evolving Reward Models

- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Reward-based alignment methods for large language models (LLMs) face two key limitations: vulnerability to reward hacking, where models exploit flaws in the reward signal; and reliance on brittle, labor-intensive prompt engineering when LLMs are used as reward models. We introduce \textbf{Meta Policy Optimization (MPO)}, a framework that addresses these challenges by integrating a meta-reward model that dynamically refines the reward model's prompt throughout training. In MPO, the meta-reward model monitors the evolving training context and continuously adjusts the reward model's prompt to maintain high alignment, providing an adaptive reward signal that resists exploitation by the policy. This meta-learning approach promotes a more stable policy optimization, and greatly reduces the need for manual reward prompt design. It yields performance on par with or better than models guided by extensively hand-crafted reward prompts. Furthermore, we show that MPO maintains its effectiveness across diverse tasks, from essay writing to mathematical reasoning, without requiring specialized reward designs. Beyond standard RLAIF, MPO's meta-learning formulation is readily extensible to higher-level alignment frameworks. Overall, this method addresses theoretical and practical challenges in reward-based RL alignment for LLMs, paving the way for more robust and adaptable alignment strategies. The anonymized code repository and data can be found at: https://anonymous.4open.science/r/mpo-CD3B

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Meta Policy Optimization (MPO), which augments standard RLAIF by adding a meta‑reward model that periodically refines the reward model’s evaluation prompt/rubric using recent training context. This evolving rubric aims to reduce reward hacking, lessen manual prompt engineering, and provide more discriminative rewards over time. The method is instantiated with small policy models and larger RM/MRM pairs, and evaluated across essay writing, summarization, ethical reasoning, and math reasoning. Results show consistent gains over fixed‑prompt PPO baselines and robustness against typical reward‑hacking failures; on essay writing, MPO surpasses even heavily hand‑engineered “oracle” prompts under similar compute.

### Strengths
Practicality: Turning reward design into an evolving rubric reduces brittle prompt engineering and mitigates fixed‑prompt reward hacking.

Generality: The framework is task‑agnostic and plugs into PPO‑style RLHF/RLAIF pipelines; evaluations span writing, summarization, ethics, and math.

Evidence of robustness: Qualitative/quantitative analyses show MPO detecting and correcting gaming behaviors (e.g., title‑only outputs), while fixed‑prompt PPO can collapse.

Analysis: Tracks rubric growth and stricter scoring over iterations, explaining how finer evaluation criteria can yield more informative gradients.

### Weaknesses
1. Model scaling: It is unclear why experiments focus on Qwen2‑1.5B and Llama‑3.1‑8B only; a systematic sweep across Qwen2.5 scales (e.g., 0.5B→7B) under matched setups would better reveal scaling trends and improvement curves.

2. Benchmarks: Important, contemporary alignment/reasoning suites are missing; adding AlpacaEval 2.0 (length‑controlled) and Arena‑Hard variants would strengthen generalization and robustness claims.

3. Role mapping: The conceptual mapping between reward models and LLM‑as‑judge is muddled. Recent work often finds LLM‑as‑judge competitive or superior to trained reward models; the “student/junior/senior instructor” analogy would be clearer if the policy is the student, the LLM‑as‑judge the instructor/judge, and the reward model a distilled proxy of that judgment. Clarify roles and justify terminology.

4. Objective choice: The training objective is under‑motivated. Given a focus on reward shaping and rubric evolution, comparisons with DPO or GRPO would isolate whether gains come from MPO itself or from PPO specifics.

5. Baselines: Please add strong recent verifiable‑reward pipelines (e.g., RLVR‑style systems) and widely used public suites for instruction‑following/reasoning. Self‑configured evaluations are valuable but less convincing without head‑to‑head comparisons against recognized baselines.

### Questions
See weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Meta Policy Optimization (MPO) for the RLAIF setting, inspired by the psychological concept of metacognition. In this framework, a meta-reward model (MRM) dynamically refines the reward model’s prompt throughout training. The MRM monitors the evolving training landscape by processing the prompt instructions, reference solutions (if available), policy generations, the reward model’s evaluation rubric, and the scores assigned to those generations. Using this information, the MRM identifies weaknesses in the current rubric that the policy may be exploiting (or is likely to exploit) and modifies the rubric to make it more targeted and fine-grained. This helps the reward model resist reward hacking by the policy and promotes more stable policy optimization. MPO reduces the need for manual prompt design and proves effective across diverse tasks without requiring specialized reward designs. Experimentally, the authors show that MPO outperforms PPO with expert or auto prompting across four different domains.

### Strengths
The paper proposes Meta Policy Optimization (MPO) for the RLAIF setting, inspired by the psychological concept of metacognition. MPO addresses reward hacking in RLAIF by introducing a meta-reward model (MRM) that periodically refines the reward model’s prompt during training. This ensures that the evaluation rubric remains granular, targeted, and resistant to exploitation by the policy. MPO is a timely contribution toward mitigating reward hacking in RLAIF and promotes more stable policy optimization. Furthermore, the prompts used for the MRM are general and task-agnostic, enabling their usage across diverse domains.

MPO demonstrates strong effectiveness compared to approaches that rely on static, hand-crafted prompts, even those designed by domain experts, across diverse tasks such as essay writing, summarization, ethical reasoning, and mathematical reasoning, showcasing its versatility. Additionally, MPO reduces the burden of prompt engineering by automatically refining the reward model’s prompt throughout training based on the observed training context. 

Finally, the paper is clearly written and well-organized, making it easy to follow.

### Weaknesses
The sample selection process for rubric refinement is completely random. Samples drawn from much earlier stages of training may not be informative, as the training context and policy behavior could have evolved significantly. Given this, it may be more effective to prioritize recent samples or those with higher estimated informativeness when updating the rubric. Such an approach could make the refinement process more adaptive to the model’s current failure modes.

As the MRM continuously evolves the reward model’s rubric, the rubric appears to become increasingly complex and fine-grained over the course of training. This process resembles inferring highly detailed reward functions that fit the observed training context but may not generalize well to unseen or downstream tasks. In light of this, it might be useful to regularize the inferred rubric, for instance, by penalizing excessive complexity or enforcing smoothness constraints, to improve generalization and stability.

Another concern is that the scoring scale of the rubric can change dynamically during training. At one point, the maximum score might be 30, whereas at a later stage it could increase to 50. This variability may lead to inconsistent reward magnitudes for the policy. To address this, it would make sense to use a normalized reward score, $s\in [0,1]$, ensuring a consistent and comparable reward scale across training iterations.

Finally, the experiments are primarily conducted on smaller models (e.g., Qwen2-1.5B-Instruct), with limited evaluation on larger LLMs due to resource constraints. This leaves open questions about scalability, in particular, whether MPO remains effective and stable as model size increases and training dynamics become more complex.

### Questions
1) Does the rubric becoming more complex over the course of training, as it is evolved by the MRM, affect the generalization performance of the LLM aligned via MPO? Wouldn't regularizing the rubric help improve generalization without sacrificing MPO's ability to curtail reward hacking?

2) Since the rubric scoring scale can change over the course of training, wouldn't it be better to use a normalized score as the training signal for the RL algorithm?

3) Do you have experimental results for other model scales (3B, 7B, 13B, etc.) and potentially other models (e.g., LLaMA) for the policy?

4) In Section 3.3.1, PPO-aligned 32B_AP receives the highest rating in pairwise Elo evaluations. The hypothesis was that the GPT-4o judge favors outputs from models aligned using evaluation rubrics it helped generate. Why was this not observed in the results for the essay writing task?

5) Why were 72B RM and MRM sizes used only for the essay writing task and not for the other three domains?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Meta Policy Optimization (MPO), a framework that dynamically refines the evaluation rubric used by a reward model (RM) during reinforcement learning from AI feedback (RLAIF). The core idea is to introduce a meta-reward model (MRM) that periodically analyzes the RM’s scoring behavior and updates its rubric to mitigate reward hacking and reduce reliance on manual prompt engineering. The authors evaluate MPO on four tasks—essay writing, summarization, ethical reasoning, and mathematical reasoning—reporting improvements over fixed-prompt baselines, including hand-crafted and AutoPrompt-generated rubrics.

### Strengths
- The motivation—addressing reward hacking and prompt brittleness in RLAIF—is well-articulated and practically relevant.  
- The MPO framework is conceptually clean and integrates naturally into existing PPO-based pipelines.

### Weaknesses
1. **Limited experimental scope and reliability**: The evaluation is conducted exclusively on a single small policy model (Qwen2-1.5B) and only with PPO. This raises concerns about the generalizability of the findings. In the current RLAIF literature, standard benchmarks such as **Arena-Hard-v2** or **Alpaca-Eval** are expected for alignment claims, yet these are entirely absent. Without results on more representative models (e.g., 7B+ scale) or alternative RL algorithms (e.g., GRPO), it is unclear whether MPO’s benefits are robust or merely artifacts of a narrow setup.

2. **Rubric design appears misaligned with task heterogeneity**: The paper implies that a single evolving rubric is shared across all queries within a task (e.g., all essay prompts use one rubric).  However, it is natural that different queries may require distinct evaluation criteria (e.g., creativity vs. factuality in essays). The current design risks oversimplifying the complexity of human preferences.

3. **Oracle rubric and evaluation protocol lack rigor**: The “oracle” rubric is derived from 60+ PPO runs on the same small model—an ad hoc and non-standard baseline. More convincingly, the paper could compare MPO’s evolved RM against a much stronger fixed judge (e.g., Qwen-2507-235B or GPT-4o) to assess whether dynamic rubric refinement truly closes the gap with top-tier static evaluators. As it stands, the claim that MPO “surpasses human-engineered prompts” is overstated given the weak oracle baseline and reliance on GPT-4o as the sole judge (which may favor its own prompt styles).

### Questions
seed weakness

### Soundness
2

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
This paper introduces Meta Policy Optimization (MPO), a novel framework that tackles two persistent challenges in Reinforcement Learning from AI Feedback (RLAIF) for large language models: vulnerability to reward hacking, where models exploit flaws in the reward signal, and the heavy reliance on brittle, manual prompt engineering for reward models. Inspired by metacognition and evaluative thinking, MPO augments the standard RLAIF pipeline with a meta-reward model (MRM) that dynamically refines the reward model's prompt throughout training. Empirically, MPO demonstrates significant advantages across diverse tasks spanning the depth-breadth spectrum of evaluative thinking, such as essay writing, summarization, ethical reasoning, and mathematics. It achieves performance on par with or superior to models using extensively hand-crafted prompts, while crucially preventing policy collapse due to reward hacking, as observed in fixed-prompt setups.

### Strengths
- MPO directly addresses two of the most significant pain points in RLAIF: reward hacking and the immense burden of manual prompt engineering.
-  A major strength is the extensive empirical validation across four distinct tasks, each representing different challenges on the spectrum of evaluative thinking.
- The paper goes beyond simply reporting results to analyze how MPO works. The discussion on the evolution of the rubric's linguistic structure provides valuable insights into the framework's inner workings.

### Weaknesses
- The entire MPO process hinges on the quality of the MRM's analysis and refinements. The paper does not deeply explore what happens if the MRM itself is flawed, generates poor rubrics, or introduces new biases. 
- While the paper reports an 11% compute overhead and argues it is modest, this is a critical factor for adoption.

### Questions
- The paper demonstrates strong results on single-turn generation tasks. How would you envision and potentially adapt the MPO framework for multi-turn interactive tasks, such as dialogue or long-horizon instruction following?
- Your results show that performance improves with the size of both the RM and MRM. Could you discuss the interplay between the policy model size and the effectiveness of MPO? What are the optimal scaling relationships between these three components?

### Soundness
3

### Presentation
2

### Contribution
3
