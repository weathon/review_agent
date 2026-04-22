# Your Models Have Thought Enough: Training Large Reasoning Models to Stop Overthinking

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Large Reasoning Models (LRMs) have achieved impressive performance on challenging tasks, yet their deep reasoning often incurs substantial computational costs. To achieve efficient reasoning, existing reinforcement learning methods still struggle to construct short reasoning path during the rollout stage, limiting effective learning. Inspired by Evidence Accumulation Models, we find that LRMs have accumulated sufficient information early in reasoning, making further reasoning steps redundant. Based on this insight, we propose Just-Enough Thinking (JET), which trains models to proactively terminate unnecessary reasoning. JET performs trajectory truncation during rollout to expose the model to short, distributionally consistent reasoning paths. Besides, it uses a quality-controlled length reward to better encourage concise reasoning while maintaining correctness. Extensive experiments demonstrate that JET significantly improves reasoning efficiency without sacrificing accuracy. In particular, JET delivers a 4.6% accuracy improvement while reducing the output length by 46.3% on the Olympiad benchmark using DeepSeek-R1-Distill-Qwen-1.5B. Our code is available in the GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Just-Enough Thinking (JET) tries to address the well-known **overthinking** problem in Large Reasoning Models (LRMs) through introducing (1) a trajectory truncation mechanism that helps early terminate unnecessary reasoning and increases effective rollout sizes, and (2) a quality-controlled length reward used in their DAPO-based RL policy optimization. Through experiments on `DeepSeek-Distill-Qwen` models (training through math datasets), the paper demonstrates considerable reduction in reasoning length on both in-domain and out-of-domain datasets without compromising accuracies.

### Strengths
- Overall the paper's writing is clear and easy-to-follow.
- The piloting experiment provides a good intuition and justification on the huge compressibility within LRM's reasoning traces (given that the full reasoning trace leads to correct answer in the first place).
- The main experiments seem comprehensive, including a range of in-domain and out-of-domain tasks and multiple alternative efficient reasoning baselines.
- The paper also includes concrete case studies in the appendix.

### Weaknesses
The weaknesses below are **ranked from high priority to low** 

_(priority in terms of how they would influence my final rating to this manuscript):_

**Weakness 1: Limited evaluation beyond `Deepseek-Distilled-Qwen`**

Recent RLVR research (not limited to efficient reasoning) often relies on the `Qwen` + `Math` setup. However, several studies -- such as *“RL with One Example”* [[1]](https://arxiv.org/abs/2504.20571) and *“RL with Random Rewards”* [[2]](https://www.interconnects.ai/p/reinforcement-learning-with-random) -- have raised concerns that performance gains might stem from factors like mid-training on math problems or potential data leakage, rather than the RL signal itself. While I am not asserting that such confounding effects necessarily apply here, it would strengthen the paper’s robustness to demonstrate that the reported improvements generalize to other model families beyond Qwen (e.g., the Llama series).

**Weakness 2: Incomplete treatment of overthinking phenomena**

While the paper’s pilot experiment provides an interesting motivation for truncation -- showing that correct reasoning traces are often compressible -- it only captures one side of the broader “overthinking” issue in LLM reasoning. The analysis (particularly the `ARR` metric) is conditioned on problems that are *already solved correctly* with full reasoning traces. This design primarily reflects case (i): verbose reasoning where the correct answer appears early but is followed by unnecessary continuation.  

However, overthinking also includes case (ii): overly long or convoluted reasoning that ultimately leads to *incorrect* answers due to confusion or drift. The current truncation approach and pilot study seem to overlook this dimension. Similarly, in the reward design, the use of a constant **0** length reward for all incorrect rollouts fails to distinguish between long and short reasoning traces that are both incorrect.

**Weakness 3: Missing ablations and unclear design justifications**

Although the paper includes numerous additional experiments in the appendix, several (e.g., the curriculum learning experiment) appear unrelated to the main narrative and do not directly clarify the proposed method’s key design choices. More targeted experiments are needed, especially in two areas:

1. **Ablation: Effect of PES without the length reward.**  
   It remains unclear how much of the observed improvement comes from the proposed length reward versus PES. Since PES-based truncation effectively constructs “low-likelihood” reasoning trajectories, GRPO/DAPO training might already encourage shorter reasoning paths by increasing the likelihood of these constructed rollouts. An ablation removing the length reward would help isolate its true contribution.

2. **Design Choice: Comparison with established length reward baselines.**  
   While the paper proposes alternative reward designs and demonstrates why the chosen one performs better, prior works such as [*Training Efficient*](https://arxiv.org/abs/2502.04463) and [*ShorterBetter*](https://arxiv.org/abs/2504.21370) have already explored efficient reasoning using length-based rewards. Comparing directly against these established formulations -- rather than self-introduced variants -- would provide a more meaningful and standardized evaluation of the proposed approach.

### Questions
In addition to questions mentioned in the `Weaknesses` section:

1. What's the rollout size for baseline methods like DAPO/AdaThink?

2. Why finetune SFT/DPO with LoRA in baseline implementation (instead of full finetuning to be consistent with the RL training)?

3. It seems that some important citations (esp. for the baseline methods) are missing, e.g. the Laser paper.

### Soundness
3

### Presentation
2

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
Language Reasoning Models (LRMs) generally achieve higher accuracy as their reasoning depth increases; however, this improvement often comes at the cost of generating excessive tokens, which significantly slows down inference. Even when an LRM has already gathered sufficient information to reach the correct answer, it frequently continues producing unnecessary reasoning steps — a phenomenon referred to as overthinking.  
  
This paper proposes a reinforcement learning framework called Just-Enough Thinking (JET), which enables LRMs to learn efficiently while suppressing overthinking. The core of JET lies in two main components: (1) a two-stage rollout construction, and (2) a reward design based on the principles of Correctness First, Conciseness Preference, and Per-Question Normalization.  
  
In the two-stage rollout construction, JET uniformly truncates the model’s full reasoning trajectories at several intervals using a strategy called Progressive Early-Stopping (PES). Because PES utilizes trajectories generated directly by the model itself, it maintains the model’s natural consistency during training. Both full trajectories and truncated trajectories are then employed in reinforcement learning, allowing the model to learn when to stop reasoning without losing accuracy.  
  
The reward function prioritizes correctness above all. Among responses that are correct, shorter reasoning paths are assigned higher rewards, thereby encouraging concise and efficient reasoning.  
  
JET is applied to DeepSeek-Distill-Qwen-1.5B and 7B models and evaluated across five mathematical reasoning benchmarks — GSM8K, MATH500, AIME24, AMC, and Olympiad — as well as three out-of-domain datasets: CSQA, GPQA, and MMLU.  
  
Across both in-domain and out-of-domain tasks, JET consistently improves accuracy while substantially reducing output token length, leading to faster inference compared with the base model.  
  
The analysis further examines the effectiveness of the PES strategy and compares token length variations under different length reward designs, demonstrating the efficiency and robustness of the proposed method.

### Strengths
### 1. Overall Assessment
The proposed method, JET, is both simple and effective. Because it leverages trajectories directly generated by the model, JET preserves the model’s natural consistency during training. The approach is also computationally efficient, achieving faster training speeds without sacrificing accuracy.  
  
### 2. Originality
The work presents a well-motivated application of Evidence Accumulation Theory to computational reasoning, highlighting meaningful parallels between human cognitive efficiency and model reasoning.
The design of the Progressive Early-Stopping (PES) mechanism and length-aware reinforcement learning is particularly well-executed. The reward function is carefully structured into three components — format reward, accuracy reward, and length reward — with the principles of Correctness First, Conciseness Preference, and Per-Question Normalization clearly reflected in the mathematical formulation.  
Unlike previous approaches that rely on fixed token budgets or handcrafted truncation, JET autonomously learns when to stop reasoning in a self-consistent and context-sensitive manner.  
  
### 3. Clarity
The paper is clearly written and well-organized, providing a strong motivation, detailed mathematical formulation, and a comprehensive algorithmic description (as presented in the Appendix). Figures such as Figure 1 (token length distribution) and Figure 2 (two-stage rollout diagram) effectively convey the core ideas and experimental insights.

### Weaknesses
### 1. Need for Further Discussion
Several aspects of the paper would benefit from deeper discussion. In the mathematical reasoning tasks, the Laser methods generally perform better on average and often outperform JET, achieving either higher accuracy or shorter token length. Moreover, Laser shows strong performance on out-of-domain datasets such as CSQA, GPQA, and MMLU. However, the paper provides almost no comparative discussion of these results, and notably, it does not cite the paper “Learn to Reason Efficiently with Adaptive Length-based Reward Shaping,” which originally proposed Laser-D/DE.  
In the PES (Progressive Early-Stopping) strategy, trajectories are truncated arbitrarily at 25%, 50%, and 75% of their original length. The paper does not discuss the potential side effects of this truncation. While the authors state that JET preserves the model’s natural generation distribution, one could question whether cutting reasoning trajectories at arbitrary points might itself introduce unnatural discontinuities.  
Finally, the paper lacks a discussion of its limitations — for instance, in scalability or potential failure cases.   
  
### 2. Overstated Claims
Some of the claims in the paper appear somewhat overstated.  
At line 400, the statement “the harder, the stronger” seems based on dataset-level observations and may be an overgeneralization. Since the study includes only eight datasets, this conclusion about performance trends appears insufficiently supported. An analysis at the sample-level difficulty would provide a stronger empirical basis for such a claim.  
Additionally, while the paper states that JET’s generation process remains consistent with the model’s natural distribution, there is no experiment directly validating this claim. The intuition that JET should be “natural” is understandable. However, the paper does not include comparative experiments showing how other methods—for example, those using explicit length control or artificially shortened data—may lead to unnatural generation, nor how such unnaturalness affects performance, while JET maintains naturalness. Including such comparisons would strengthen the discussion.  
In the 7B model, JET actually leads to accuracy drops on GSM8K and MATH500, which requires further analysis. Although the reward function is designed to favor correctness over length, these results suggest that this objective may not always hold in practice.
  
### 3. Experimental Scope  
The study primarily evaluates JET on DeepSeek-Distill-Qwen models. To strengthen claims of generality, additional experiments on other architectures (e.g., Phi-4) would be desirable.   
  
### 4. Minor Issues  
Typographical errors: e.g., line 303, “we teste”; and in Equation (3), redundant parentheses “))”.

### Questions
Why did you not cite the Laser methods paper? The Laser methods generally show better average performance than the proposed JET approach on the target task (mathematical reasoning), and also perform strongly on out-of-domain benchmarks. In what aspects can JET be considered to have clear advantages over the Laser methods?  
  
You argue that JET preserves the model’s natural reasoning distribution. Could you provide empirical validation for this claim? The Progressive Early-Stopping (PES) uses fixed truncation ratios (25%, 50%, 75%). Did you observe any cases where these arbitrary cutoffs caused the model to stop reasoning unnaturally — even if the truncated reasoning still led to correct answers?  
  
Experiments are limited to the DeepSeek-Distill-Qwen models. Have you tested or considered applying JET to other models (e.g., Phi-4)?  
  
How does the training time of JET compare to that of the Laser methods?  
  
Finally, could you elaborate on potential scenarios where JET underperforms or fails?

### Soundness
2

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
The paper proposes a method called Just-Enough Thinking (JET) that trains LRMs to proactively terminate unnecessary reasoning. Specifically, the paper first performs pilot experiments to reveal that LRMs have accumulated sufficient information early in their reasoning. Based on this phenomenon, JET adopts a Progress Early-Stopping (PES) strategy to sample reasoning trajectories with varying length and introduces a linearly normalized length reward to encourage more efficient reasoning. Experimental results show that JFT significantly improves reasoning efficiency without sacrificing accuracy on 5 math benchmarks and some other reasoning benchmarks.

### Strengths
1.The paper is mostly well written and presents the topic clearly.

2.The pilot studies are interesting and directly supports the design of the JET method.

3.The proposed method (JET) is described in sufficient detail and appears technically sound.

### Weaknesses
Major Concerns

1.Potential distributional inconsistency. The paper introduces a truncation strategy that also involves inserting an additional prompt. There is insufficient evidence to confirm that this modification does not introduce a harmful distributional inconsistency problem.

2.Sub-optimal performance on math tasks. The results on mathematical reasoning tasks do not clearly demonstrate the superiority of the proposed method. As shown in Table 1, the JFT-DeepSeek-Distill-Qwen-7B model underperforms the Laser-DE model in both accuracy (-0.5%) and efficiency (using more tokens).

3.Insufficient experiments. The empirical evaluation is constrained to a specific set of models (DeepSeek-Distill-Qwen-1.5B and DeepSeek-Distill-Qwen-7B). The experiments should be extended to include more recent model families (e.g., the Qwen3 series) and potentially more model sizes.

### Questions
My main questions for the authors directly relate to the major concerns identified above. Addressing these points during the rebuttal period would be crucial for re-evaluating my rating.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes JET, a reinforcement learning method based on DAPO to reduce unnecessary reasoning steps in Large Reasoning Models. The approach uses Progressive Early-Stopping to create truncated reasoning trajectories during training and employs a length-aware reward to encourage concise outputs. Experiments on DeepSeek-Distill-Qwen models (1.5B and 7B) show token reductions while maintaining accuracy on math benchmarks.

### Strengths
1. The paper demonstrates that LRMs accumulate sufficient information early in reasoning chains, providing good empirical justification for the approach.

2. JET generates shortened trajectories from the model's sampled outputs, reducing distribution mismatch issues that arise from using external short answers.

3. The paper includes  evaluations across multiple benchmarks and model sizes.

### Weaknesses
1. In Section 1, line 87, I think that assigning length penalties to longer correct trajectories is problematic because the model cannot guarantee that longer sequences always contain redundant information. Although the authors use a "Correctness first" rule where shorter responses receive better reward signals, penalizing longer correct answers does not make much sense.

2. The values of non-negative coefficients w_f, w_acc, and w_ℓ are shown in the appendix as w_acc = 0.9, w_f = 0.1, and w_ℓ = 1. More diverse strategies should be explored since these coefficients significantly influence performance.

3. Since JET is implemented based on DAPO, the accuracy of DAPO should theoretically represent the upper bound for JET. However, I am confused why GSM8K improves by 3.8 points when length decreases from 826 to 605, AIME24 improves by 6 points when length decreases from 8000 to 6000, and Olympiad improves by 5 points when length decreases from 5000 to 4000.

4. When the model scales from 1.5B to 7B, most baseline models decrease or slightly increase sequence length on AIME24, AMC, and Olympiad. However, JET increases sequence length from 6641 to 7981 on AIME24, 3872 to 4301 on AMC, and 4121 to 5083 on Olympiad. What explains this counterintuitive behavior?

5. In Table 2 regarding language reasoning tasks, JET's sequence length is 1013, which is comparable to other baseline models, indicating that JET does not achieve expected length reduction on these tasks.

6. In Table 2, why do baseline models decrease length when scaling from 1.5B to 7B, while JET increases from 800 to 1000? More analysis should be provided here.

7. The paper presents performance on many benchmarks but lacks analysis of model training dynamics such as KL, policy entropy, gradient norm, etc.

8. In Section 3.2, there is ambiguity: does "ℓ_min and ℓ_max are the shortest and longest correct responses" refer to the responses themselves or their lengths?

9. In Formula 6, what would happen if the "+ δ" term were removed?

### Questions
See the Weaknesses described above.

### Soundness
2

### Presentation
3

### Contribution
2
