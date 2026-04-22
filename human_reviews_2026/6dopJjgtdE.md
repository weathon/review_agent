# Progra: Progress-Aware Reinforcement Learning for Multi-Turn Function Calling

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Large language models (LLMs) have achieved impressive success in single-turn function calling, yet real-world applications such as travel planning or multi-stage data analysis typically unfold across multi-turn conversations. In these settings, models must not only issue accurate function calls at each step but also maintain progress awareness, the ability to summarize past interactions and plan future actions to ensure coherent, long-horizon task execution. Existing approaches, however, either reduce multi-turn training to isolated single-turn samples, which neglects task-level planning, or employ end-to-end reinforcement learning (RL) that struggles with redundancy and lacks explicit integration of progress awareness. To overcome these limitations, we introduce Progra, a framework that explicitly incorporates progress awareness into LLM training for multi-turn function calling. Progra combines (i) a Progress Awareness Generation (PAG) pipeline, which automatically constructs datasets coupling conversation summaries with future task planning, and (ii) a Progress Awareness-Guided Reinforcement Learning (PAG-RL) algorithm, which integrates progress awareness into RL training to reduce contextual redundancy and improve alignment between local actions and global task completion. Empirical results on two public benchmarks demonstrate that Progra significantly outperforms existing methods, highlighting the effectiveness of progress awareness in enabling robust and efficient multi-turn function calling. Our code can be available at https://anonymous.4open.science/r/Progra_ICLR26-57F0

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PROGRA, a two-phase training framework designed to improve LLMs' multi-turn function calling capabilities by explicitly incorporating "progress awareness"—the ability to summarize conversation history and plan future actions. Phase 1 (PAG) automatically constructs a dataset where each conversation turn is augmented with generated summaries and plans, verified through model-aware validation. Phase 2 (PAG-RL) integrates this progress awareness into GRPO-based reinforcement learning, where the model first generates awareness text before producing function calls. Experiments on BFCL and τ-Bench with three backbone models (Qwen2.5-7B, xLAM-2-3B/8B) show improvements over baselines, with gains ranging from modest (1.5% on xLAM-2-3B BFCL) to substantial (44% on Qwen2.5-7B τ-Bench).

### Strengths
- Well-motivated problem: The observation that existing multi-turn training methods degrade to single-turn samples and neglect task-level planning is valid and clearly articulated. Figure 1 effectively illustrates the delayed feedback issue in multi-turn conversations.
- Comprehensive pipeline design: The PAG data synthesis pipeline with model-aware verification and diversity-preserving augmentation demonstrates solid engineering. The two-phase approach (PAG for data generation + PAG-RL for training) is systematic.
- Systematic evaluation: The paper tests on multiple models (3 backbone LLMs) and benchmarks (BFCL, τ-Bench), includes ablation studies (Table 2), and provides awareness capability analysis (Figure 4). The experimental design covers different model scales.

### Weaknesses
1. Questionable conceptual novelty: The core concept of 'progress awareness' appears to be a repackaging of existing ideas (history summarization + future planning). The paper mentions SPA-RL's 'stepwise progress attribution' in related work but fails to clearly differentiate PROGRA's contributions. Section 2.2 lists several recent multi-turn RL works without adequate comparison.
2. Model-Aware Verification creates circular dependency: Using the target policy πθ to verify generated awareness quality (Section 3.2) is fundamentally problematic. If the model is weak initially, verification will approve low-quality data, perpetuating poor training. The paper provides no analysis of verification failure rates, quality distribution, or strategies to break this cycle.
3. Inconsistent and limited experimental gains: Table 1 shows highly variable improvements ranging from 1.5% (xLAM-2-3B BFCL Overall) to 44% (Qwen2.5-7B τ-Bench), suggesting the method lacks robustness. More critically, Table 2 Row 'PAG + MT-GRPO' for xLAM-2-8B shows **negative** results (-1.4% compared to baseline), directly contradicting the hypothesis that PAG improves awareness. Only 200 training samples are extremely small. It is unclear if results generalize beyond these specific benchmarks.
4. Missing computational cost analysis: PAG-RL requires generating awareness text before each action, at least doubling inference cost. Section A.1 acknowledges 'additional time costs' but provides zero quantitative analysis (no training time, wall-clock hours, FLOPs, or token throughput comparisons). This is critical for practical deployment and represents a major gap.
5. Weak baselines and missing comparisons: The paper only compares against vanilla MT-GRPO as the RL baseline, despite mentioning ARTIST, RLFactory, SPA-RL, and other recent methods in Section 2.2. The 'Reasoning' baseline is merely CoT prompting. No comparison with simpler alternatives like longer context windows or retrieval-augmented generation.
6. Limited evaluation scope: Only evaluation on two benchmarks (BFCL, τ-Bench). Case study (Section 4.6) shows only 1 cherry-picked example from 50 samples, detailed error analysis is missing.

### Questions
1.	How do you address the circular dependency in Model-Aware Verification? If the target model $\pi_{\theta}$ is initially weak, won't it approve low-quality awareness data, leading to poor SFT and subsequent RL training? Can you provide verification accuracy curves or inter-annotator agreement with human labels?
2.	Table 2 shows 'PAG + MT-GRPO' performs worse than baseline on xLAM-2-8B (-1.4%). This directly contradicts your central claim that PAG improves awareness. Is this a statistical artifact, or does PAG harm certain model architectures? Please explain.
3.	Please provide quantitative analysis: (a) Total training time in GPU-hours (PROGRA vs MT-GRPO), (b) Inference latency per action with/without awareness generation, (c) Total tokens generated and associated API costs. Without this data, the practicality of your method cannot be assessed.
4.	Why only compare against vanilla MT-GRPO for RL? Can you add comparisons with ARTIST, RLFactory, or SPA-RL? Or at minimum, demonstrate that MT-GRPO is a strong baseline by comparing it against these methods?
5.	With only 200 training samples, how do you ensure results aren't overfitting to the small benchmarks?

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
This work introduces Progra, a framework that explicitly incorporates progress awareness into LLM training for multi-turn function calling. Progra combines (i) a Progress Awareness Generation (PAG) pipeline, which automatically constructs datasets coupling conversation summaries with future task planning, and (ii) a Progress Awareness-Guided Reinforcement Learning (PAG-RL) algorithm, which integrates progress awareness into RL training to reduce contextual redundancy and improve alignment between local actions and global task completion. Empirical results on two public benchmarks demonstrate that Progra significantly outperforms existing methods, highlighting the effectiveness of progress awareness in enabling robust and efficient multi-turn function calling.

### Strengths
This work makes substantive contributions to multi-turn function calling through the explicit formalization and integration of progress awareness into LLM training. The originality lies in identifying progress awareness as a compositional capability encompassing both retrospective summarization and prospective planning, operationalized through a principled two-phase framework that couples synthetic data generation with awareness-guided reinforcement learning. The model-aware verification mechanism represents a methodologically sound approach to quality control, leveraging the target policy itself as an oracle for data validation. The technical execution demonstrates rigor through formal MDP-based problem formulation, comprehensive empirical evaluation spanning multiple benchmarks and models, and systematic ablation studies that establish the necessity of both framework components. The experimental design effectively isolates the contribution of progress awareness through controlled comparisons against baselines. The work exhibits strong clarity in its presentation, with coherent mathematical notation.

### Weaknesses
1. Only evaluation on two benchmarks (BFCL, τ-Bench). Can you provide more results on other benchmarks?

2. PAG-RL requires generating awareness text before each action, at least doubling inference cost. Section A.1 acknowledges additional time costs but provides zero quantitative analysis. This is critical for practical deployment and represents a major gap.

3. Only compare against vanilla MT-GRPO for RL. Can you add comparisons with SPA-RL or other methods? Or demonstrate that MT-GRPO is a strong baseline by comparing it against these methods?

4. Based on the ablation study in Table 2, there exists a substantial performance discrepancy between Qwen2.5-7B and xLAM models suggesting the method maybe lacks robustness.

### Questions
Please  refer to the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
PROGRA is a two-phase training framework that enhances large language models’ performance on multi-turn function calling by explicitly incorporating progress awareness---the ability to track task progress, summarize relevant history, and plan subsequent actions.

Phase 1 (Progress Awareness Generation, PAG) constructs synthetic training data that couples conversation histories with automatically generated progress summaries and future plans. A strong generator LLM produces these summaries, which are verified by testing whether they contain sufficient information to reconstruct the correct function calls, augmented for diversity, and then used to warm up the target model through supervised fine-tuning.

Phase 2 (Progress Awareness–Guided Reinforcement Learning, PAG-RL) introduces progress awareness into the rollout process: before each function call, the model generates a concise awareness summary and conditions its next action on it. The combined behavior--summary and action---is optimized end-to-end with the GRPO (Group Relative Policy Optimization) algorithm using a composite reward that balances structural validity, functional correctness, task success, and efficiency.

Evaluated on BFCL-V3 and τ-Bench benchmarks using Qwen2.5-7B, xLAM-2-3B, and xLAM-2-8B backbones (200 examples per benchmark), PROGRA consistently outperforms baselines including supervised fine-tuning and vanilla multi-turn GRPO. Reported relative gains range from roughly 5 to 40% depending on model and dataset, with the best overall accuracy reaching ≈52%. The results demonstrate that explicitly modeling progress awareness improves coherence, reduces redundant calls, and strengthens long-horizon task execution, though at the cost of additional computation during training.

### Strengths
### 1. **Well-Designed Data Synthesis Pipeline with Quality Verification**

The Progress Awareness Generation (PAG) pipeline includes a genuinely clever verification step: after generating progress summaries using a strong LLM, it tests whether these summaries contain sufficient information by attempting to reconstruct the correct function call from the summary alone, keeping only summaries that pass this test. This "model-aware verification" ensures the synthetic training data is actually useful for the target model rather than just plausible-sounding text. Combined with diversity-preserving augmentation strategies (paraphrasing, schema perturbations, word masking), this creates a principled approach to synthetic data generation that could be adapted to other domains. The pipeline is fully automated and appears reproducible based on the provided implementation details.

### 2. **Consistent Improvements with Proper Ablations**

Despite varying magnitude, PROGRA shows improvements over baselines across all tested configurations (three model sizes, two benchmarks, multiple evaluation subsets), with the full method consistently outperforming partial implementations. The ablation studies systematically demonstrate that both phases contribute to performance, and the paper provides sufficient implementation details (training procedures, hyperparameters, reward functions) for reproduction. The gains on τ-Bench are substantial for smaller models (44% relative improvement for Qwen2.5-7B), and even the more modest improvements on other settings suggest the approach provides real value rather than being entirely spurious. The case study showing both higher accuracy (26%→40%) and fewer steps (27.08→23.30) provides supporting evidence that the method produces more efficient and coherent trajectories.

### Weaknesses
### 1. **Insufficient Evaluation of Efficiency Trade-offs**

The method generates substantially more tokens per turn (estimated ~320 tokens vs ~20 tokens for direct function calling due to `<summary>` and `<think>` sections), likely resulting in ~15x higher output token costs and proportionally increased latency. While the paper shows fewer function calls (23.30 vs 27.08 average steps), which could save on API costs, it provides no analysis of the overall cost-benefit trade-off. Without reporting token counts, latency measurements, or cost modeling, it's impossible to assess whether the accuracy improvements (which range from 1-44% relative gain) justify the computational overhead, or to determine under what conditions (expensive APIs vs. cheap APIs, latency-sensitive vs. batch processing) this approach would be practical to deploy.

### 2. **Weak Baselines and Limited Generalization Testing**

The evaluation compares only against basic baselines (base model, simple prompting, standard SFT, vanilla GRPO) and lacks comparison to stronger alternatives like other structured reasoning approaches, advanced prompt engineering, or state-of-the-art commercial models. More critically, all models are trained and tested on the same domain (BFCL→BFCL, τ-Bench→τ-Bench) with small datasets (200 examples each), raising concerns about overfitting. The paper provides no cross-domain evaluation (train on airline, test on retail), no out-of-distribution testing, and no analysis of how performance scales with more training data. Combined with inconsistent results across settings (1% gains on some benchmarks, 44% on others, with ablations showing contradictory effects across models) and lack of statistical significance testing (only 3 trials), it's unclear whether the improvements represent genuine capability gains or artifacts of the specific evaluation setup.

### Questions
Q1. The ablations show interesting variance - Phase 1 alone hurts performance on Qwen/τ-Bench but helps on xLAM-2-8B. Have you tested cross-domain generalization (e.g., train on airline, test on retail, or train on BFCL, test on τ-Bench)? What do you think explains the different component contributions across model sizes?

Q2. Since PROGRA introduces explicit <summary> and <think> generations at each turn, have you measured the additional token and latency overhead relative to vanilla GRPO? It would be helpful to understand whether the observed accuracy gains translate into overall efficiency improvements, and in what deployment settings you think this trade-off is worthwhile.

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
Most large language model training with reinforcement learning typically solves problems using single-step reinforcement learning, ignoring the fact that models interact with people over multiple turns. Specifically, the dialogue leading up to the final turn is treated as context, and the model is only trained to predict the last turn in the dialogue. The authors propose to address this issue by introducing a framework called Progra, which explicitly incorporates progress awareness into LLM training for multi-turn function calling. Progra combines two key components: Progress Awareness Generation (PAG) and Progress Awareness Guided Reinforcement Learning (PAG-RL). PAG is designed to generate high-quality datasets consisting of metadata that includes tool descriptions and conversation contexts—essentially, the history of the conversation. Meanwhile, PAG-RL learns how to sample responses based on summaries of this PAG history. The authors conduct several ablation studies, comparing their framework against models that use either PAG, PAG-RL, or both. They demonstrate that their proposed framework outperforms existing methods across two benchmarks, BFCL and \tau-Bench.

### Strengths
- The ability to train LLMs for multi-turn conversations is a crucial problem.
- The authors proposed a novel strategy to enable LLMs not only to learn in a multi-turn setting but also to capture information in a way that allows them to build on previous utterances.
- The description of the proposed method was clear, and I was able to understand the significance of the proposed approach.

### Weaknesses
- The baseline algorithms, such as GRPO, are not being compared fairly. Specifically, PAG is essentially another form of supervised fine-tuning, whereas MT-GRPO is not fine-tuned in a supervised manner at all. Combining PAG with GRPO does not count because PAG is a different format that includes information and metadata, which GRPO cannot utilize effectively.
- The results do not present standard deviations, making it unclear how significant the findings are.

### Questions
- If you supervised fine-tuning of the baseline with in-domain data, how well does it perform? Additionally, if you applied GRPO on top of this newly supervised fine-tuned baseline, what results did you achieve?

- What does "w/o PAG" mean in the plots? Does it indicate that you are running PAG-RL on top of the base model? If so, how does the baseline, which does not understand how to emit compatible progress awareness information (meta), achieve this without training first?

- GRPO is known to have subtle issues in optimization. Did you consider other variants, such as Dr.GRPO, which address some of the shortcomings of GRPO?

- Did you perform any qualitative analysis on why the proposed multi-turn framework performs better? Specifically, is there anything in the utterances generated by the model that suggests why it is performing better?

- Did you try training with different prompts using MT-GRPO on the benchmark tasks evaluated in the paper?

### Soundness
2

### Presentation
2

### Contribution
2
