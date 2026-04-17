# AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints

- Decision: Accept (Poster)
- Scores: 8, 6, 2, 6

## Abstract
Tool use represents a critical capability for AI agents, with recent advances focusing on leveraging reinforcement learning (RL) for test-time scaling to achieve better performance through more deliberate reasoning. 
However, there are some key challenges in current RL-based scaling approaches: 
(a) direct RL training often struggles to scale up thinking length sufficiently to solve complex problems, 
and (b) scaled-up models tend to overthink simpler problems, resulting in substantial token inefficiency.
To address these challenges, we propose a novel training paradigm that first employs warm-up supervised fine-tuning to help models distinguish between simple and complex problems, followed by RL that enable models to automatically determine appropriate reasoning trajectories. 
Furthermore, to tackle the issue of automatic thinking-length scaling, we discover that entropy-based optimization objectives effectively maintain model diversity while successfully unlocking the model's scaling capabilities.
Based on this insight, we introduce an entropy-based long-short reasoning fusion RL strategy. 
Our experiments on three benchmarks demonstrate that model successfully achieves auto-scaling for efficient tool use, achieving significant 9.8\% accuracy improvements while reducing computational overhead by ~81\%.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduced an adaptive entropy constraint during RL training phase to address the reasoning collapse issue in the hard tool-use task and overthink in the easy cases.

* It provides an new entropy constraint to adaptively prompt long or short reasoning trajectories based on the task difficulty.
* It proposes to use a dataset with emphasis on the difficult task to improve the model performance.
* It provides a thorough analyze on the approach's performance against different approaches proposed to improve tool-use performance and an ablation study on the contribution from different parts of the proposed approach.

### Strengths
* Overall. the paper is well-written, it provides an clear illustration on the proposed approach. The experiments setup and results are presented clearly.
* The proposed approach is easy to follow and implement. It provides detailed information on replicating the results.
* The proposed approach improves the model performance on the multi-turn scenario significantly, proving the promoting long reasoning strategy is beneficial to solving complex multi-turn task.
* The ablation study demonstrates the importance of all 3 parts of the proposed approach.

### Weaknesses
* While the author puts testing the approach in different architecture series in the future work, but I personally would prefer includes it in the experiments to prove the proposed methods works not just on the Qwen2.5-7B-Instruct.
* Missing how $H_l = 0.2$ and $H_s = 0.1$ are defined, such as a experiment on the performance with different choices of $H_l$ and $H_s$.
* The ablation experiment w/o data refinement does not describe what's the data used in the experiment.
* Fig 4, the loss for short trajectories uses the long trajectory's target entropy.
* It's unclear how are the short and long path/trajectory defined?

### Questions
* Line 075, what's the in-depth experiments.
* Line 133, the citation is on the open-source code which does not provide useful information on the finding.
* Table 2, why is the # of data in ToolACE and Hermes Function-Calling remain the same after downsampling.
* Sec 4.3.2, if the ACU includes the accuracy and # of tokens, shouldn't the ACU score be the overall performance metric?
* Line 752, it states `Results indicate data refinement increases accuracy reward score by +15%`, but the figure 9 shows the other way around. The blue line with data refinement does not have much accuracy score increment.
* Line 911, PPO citation incorrect?
* While the # of turns might correlate to the task difficulty, but it would be nice to have an analyze on the performance (accuracy, # of tokens) in different task difficulty.

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
The paper proposes AutoTool, an RLVR-style training recipe for tool use that decouples entropy regularization between short (no-think) and long (think) trajectories and adapts the long-path entropy coefficient toward a target entropy. The pipeline warms up with SFT to expose both modes, then runs GRPO with: (i) a mode tag ([think]/[no_think]) and a format reward, (ii) an answer reward, and (iii) an entropy loss that activates only when entropy drops below a target (adaptive β for think, fixed β for no-think). On BFCL (non-live, live, multi-turn), API-Bank, and ACEBench, a ~7B model improves accuracy vs. SFT/GRPO baselines while cutting inference tokens, and the model auto-scales its "think rate".

### Strengths
1. Clear motivation & mechanism. The paper links tool-use failures under RL to entropy collapse and proposes a targeted fix—decoupled, adaptive entropy—implemented with minimal changes to GRPO. 
2. Solid empirical suite. Evaluations span BFCL (non-live/live/multi-turn), API-Bank, and ACEBench, with consistent gains over SFT/distillation/RLVR-like baselines; ablations remove each component (data refine, decouple, adapt-coef) and show clear drops. 
3. Efficiency analysis. The paper reports token-cost reductions, defines ACU (Accuracy per Params×Tokens), and visualizes training dynamics (entropy, think-rate, lengths).

### Weaknesses
1. Data construction bias and potential leakage. The curated training set mixes downsampled public data with RL-refined, low-variance samples, which likely skews toward cleaner/easier cases; possible overlaps with evaluation sets are not audited. 
2. Hyperparameter sensitivity. In multi-turn settings, performance appears sensitive to target entropies and the choice of β; broader sweeps or learned schedules would strengthen the claims.

### Questions
1. Sensitivity & ablations. Will you provide sweeps over target entropies for think/no-think and over β (fixed vs. adaptive), and include a single unified entropy-penalty baseline to isolate the benefit of decoupling?
2. Format reliance. Can you evaluate without the [think]/[no_think] tags (no format reward), or on prompts that prohibit tags, to test format-agnostic generalization? 
3. Data & overlap checks. Would you release the curated training data (or a representative subset) and report an overlap analysis with evaluation sets, and add results where any dataset family used for training is excluded from the evaluation domains?

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
This paper aims to improve Tool-use LLMs' performance through the control of thinking length. It proposes applying distinct entropy constraints to long and short trajectories during RL.

### Strengths
* This paper proposes a new loss function designed to apply an entropy penalty that is conditional on the trajectory length within the RL process.

### Weaknesses
* The paper suffers from a critically confusing narrative and fundamental conceptual ambiguity. The core concept of TTS introduced in line 40 is unsupported in the literature and largely diverges from the definitions provided in the very papers cited. This narrative choice, which mixes TTS terminology with RL problem, is highly confusing and obscures the paper's actual contribution.
* For the RL component, the idea of activating or deactivating thinking via special tokens is a well-established and standard paradigm. The paper's contribution is reduced to proposing a new training loss within this existing framework, making the technical novelty incremental. The authors fail to properly contextualize their work within the relevant RL literature, which is a major omission.
* The experiments also lack rigor.
  * The paper largely ignores previous work on critical related issues, such as thinking length and entropy constraints in RLVR.
  * The evaluation is demonstrably unfair due to a vast disparity in the inference budget. Figure 1-c explicitly shows the proposed method uses an inference budget greater than 1k, while the baselines are restricted to less than 100. This makes the reported performance gains unconvincing, especially since many *actual* TTS methods can achieve similar gains without requiring any additional RL training.

### Questions
See above.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel reinforcement learning training paradigm to address two key challenges in tool-based AI agents: (1) direct RL training struggles to scale up thinking length sufficiently to solve complex problems, and (2) scaled-up models tend to overthink simpler problems, resulting in substantial token inefficiency.

Main contributions include:
1. Proposed a decoupled adaptive entropy constraint strategy, which separates entropy constraints for short and long reasoning paths, enabling the model to automatically adjust its reasoning scale based on problem difficulty.
2. Designed an automatic test-time scaling reward module, which uses an asymmetric reward mechanism to encourage short reasoning for simple problems and long reasoning for complex ones.
3. Constructed the PubTool dataset and optimized RL training effectiveness through a data quality refinement strategy.
4. Validated the method's effectiveness on three benchmarks, achieving a significant 9.8% accuracy improvement while reducing computational overhead by approximately 81%.

### Strengths
- The decoupled entropy constraint mechanism is novel. By applying different entropy constraint strengths to short and long reasoning paths, it effectively solves the reasoning collapse problem prevalent in traditional RL training.
- The paper is well-structured, with logical coherence from problem analysis and method design to experimental validation. The figures are well-designed, particularly the training dynamics visualization, which effectively demonstrates the auto-scaling effects.
- The experimental design is comprehensive, featuring systematic evaluation on three benchmarks: BFCL, API-Bank, and ACEBench.

### Weaknesses
- Entropy constraint hyperparameter selection: Although an adaptive mechanism is proposed, the initial choices for H_l and B_s lack sufficient theoretical justification or ablation analysis.
- While sample filtering based on reward variance is mentioned, the specific threshold settings and selection criteria are not described in detail.

### Questions
- The legend for Figure 2 could be added to each of the three subplots or provided as a common legend separately.
- What is the specific theoretical rationale behind the asymmetric reward in Formula (5) (only +0.5 for correct answers in 'think' mode versus +1.0 in 'no-think' mode)? Could this design potentially bias the model towards preferring the 'no-think' mode?

### Soundness
3

### Presentation
4

### Contribution
3
