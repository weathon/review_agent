# CodeRule-RL: Standard-Guided RL with Per-Rule Reward Scheduling for Code LLMs

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Large language models for code often pass unit tests yet remain brittle in practice: they may overfit to a test suite, rely on undefined semantics, or fail under small perturbations. Existing RL-based code generation methods optimize rewards from unit test execution, tightly coupling the training signal to a specific test suite. In contrast, we focus on coding standards as the primary source of feedback. We use rules (e.g., MISRA C) as machine-checkable outcomes, converting them into per-rule reward components with a frequency-aware schedule.
During reinforcement learning, the model maximizes this rule-based proxy reward. We hypothesize that enforcing well-established coding rules provides a generalizable training signal that improves both adherence to standards and pass@1 (single-attempt functional success). Motivated by these concerns, we present CodeRule-RL, a reinforcement learning approach that integrates coding rules directly as reward signals for code generation. A frequency-aware curriculum prioritizes frequently violated rules and downweights them as compliance improves. The model, optimizer, data, and prompts remain fixed, with training adjusting only reward weights. Unit tests may appear in prompts for specifications, but they are not executed during training. On the public CodeContests+ C subset, CodeRule-RL achieves higher pass@1 while reducing training wall clock time by more than an order of magnitude compared with RL that executes tests during training. Across 1.5B–7B backbones, it consistently improves both coding-standard compliance and functional success, delivering an 87% relative pass@1 gain.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a reinforcement learning framework that enhances code generation by utilising coding standards as structured, machine-checkable guidance, rather than relying on costly unit-test execution during training. The method converts each coding-rule violation into a separate, bounded reward component and introduces a frequency-aware curriculum that dynamically prioritizes frequently violated rules, gradually reducing their influence as compliance improves. Using GRPO for policy optimization, CodeRule-RL focuses solely on maximizing pass@1 at evaluation time. Experiences show improvement in pass@1 with faster training compared to text-executing RL baselines.

### Strengths
- Fine-grained reward design: Converts heterogeneous rule violations into per-rule reward components, allowing more interpretable and stable credit assignment than scalar execution rewards.
- Frequency-aware scheduling: A Curriculum mechanism that adaptively emphasises frequently violated rules, reducing interference and improving sample efficiency.
- Efficiency gains: Removes test execution from the RL loop, achieving over 10× faster training while maintaining or improving performance.

### Weaknesses
- Lack of related works and baselines: There are few related works reported in the paper on rule-guided code generation. The following papers seem to relate:
    - Dolcetti, Greta, et al. "Helping LLMs improve code generation using feedback from testing and static analysis." arXiv preprint arXiv:2412.14841 (2024).
    - Agrawal, Lakshya A., et al. "Monitor-guided decoding of code lms with static analysis of repository context." Advances in Neural Information Processing Systems 36 (2023): 32270-32298.
    - Yao, Feng, et al. "Training Language Models to Generate Quality Code with Program Analysis Feedback." arXiv preprint arXiv:2505.22704 (2025).
- Limited scope of evaluation: Experiments focus mainly on C language and MISRA C:2012; unclear generality to other languages or coding standards.
- Functional correctness dependence: While tests are excluded from training, pass@1 remains the only optimization target—may overlook broader program semantics or long-horizon correctness. How do other metrics change as the training goes on?
- No ablation on rule-set granularity: The paper doesn’t quantify how different rule subsets (Mandatory vs. Advisory) or rule complexity affect learning dynamics.
- Lack of benchmarks: The paper only evaluated on CodeContests+. Evaluation on other benchmarks is missing.
- Writing format: Some equation numbers are missing.

### Questions
- How is your framework compared with other RL-based frameworks with rule-guided reward?
- How does your framework generalize to other languages or coding standards?
- How do other coding metrics (eg. number of passing tests, static-rule-violation count, code robustness, recall@k) change as the training goes on?
- How does your framework perform on different rule subsets?
- How does your framework perform on different benchmarks?

### Soundness
2

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
This paper proposes CodeRule-RL, a reinforcement-learning scheme for code LLMs that never executes unit tests during training. Instead, it treats a coding standard (MISRA C:2012) as a source of per-rule, machine-checkable signals and builds a frequency-aware curriculum that adjusts only the weights of rules over time while keeping data, prompts, and optimizer fixed. Experiments on a C subset of CodeContests+ show sizeable pass@1 gains across Qwen2.5-Coder and DeepSeek-Coder backbones and substantial efficiency gains versus an execution‑based RL baseline. The paper also reports a VFR via Infer and finds little degradation on HumanEval/MBPP.

### Strengths
1. Compelling efficiency: Training is roughly 13× faster and 9× less latency-heavy than execution-based RL. This could make RL for code far more practical in large-scale systems.
2. Simple&clean design: The per-rule reward shaping and the Top-K EMA schedule are straightforward yet effective. The idea that the curriculum exists within the reward function is clean and easy to replicate.
3. Performance gains: consistent pass@1 improvements across model families/sizes

### Weaknesses
1. Objective framing is overstated: Saying the model “optimizes pass@1 as the sole objective” is inaccurate and potentially misleading. The method optimizes a proxy reward based on static rule outcomes; pass@1 is simply how performance is measured. The authors should clarify this distinction and, ideally, provide a correlation analysis to demonstrate alignment between the two.
2. Single-language scope: The entire study is in C / MISRA C:2012. The claim that the approach is “standard-agnostic” would be much stronger with even one more domain. For example, Python with PEP 8 or JavaScript with ESLint.
3. Limited hyperparameter analysis: Many knobs, like clip bounds, EMA rates, gating thresholds, could affect outcomes, but only warm-up length is explored. More systematic sensitivity tests would increase confidence.
4. Possible reward-hacking behavior: Since rewards are static-rule-based, models might learn superficial tricks (like adding redundant casts) to satisfy the rules without improving semantics. The paper briefly mentions this risk but doesn’t examine it empirically.

### Questions
1. Can you clarify the Table 1 vs. Table 2 discrepancy for Qwen-3B base results? Are these different prompt settings or evaluation slices?
2. Have you computed a correlation between rule reward and pass@1 improvement across training steps?
3. Do you have any cross-language or cross-standard experiments (e.g., PEP 8, ESLint)? Even small-scale results would strengthen the “standard-agnostic” claim.
4. How sensitive are results to reward-clipping bounds and schedule hyperparameters ($\tau, W, \lambda, \epsilon_p$)?
5. When VFR decreases (e.g., for smaller models), which rule categories worsen?
6. Could you include a fixed-weights per-rule shaping ablation to separate the effects of per-rule decomposition from scheduling?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces CodeRule-RL, a reinforcement learning framework designed to improve the functional correctness (pass@1) of code-generating large language models. The core idea is to use feedback from a static code analyzer, specifically violations of a coding standard like MISRA C:2012, as the reward signal during RL training. This approach deliberately avoids executing unit tests in the training loop, leading to gains in training efficiency. The authors demonstrate that across various model sizes (1.3B to 7B), their method consistently improves pass@1 on the CodeContests+ benchmark while reducing training time compared to RL methods that rely on unit test execution.

### Strengths
*  The method is evaluated across two different model families (Qwen and DeepSeek) and multiple model sizes, showing consistent improvements in `pass@1` (Table 1). This demonstrates the general applicability of the approach beyond a single model architecture. The training appears stable.
*  Figure 2 provides a helpful overview of the system architecture.

### Weaknesses
*   **Misleading Phrasing of the Optimization Objective:** The paper repeatedly states that it "optimizes `pass@1` as the sole objective" (e.g., Abstract, Lines 9-10). This is misleading. The actual reward signal being maximized during RL training is a function of coding standard violations, not the `pass@1` metric. The underlying hypothesis is that maximizing this proxy reward will *indirectly* lead to better `pass@1`, which is the *evaluation metric*. The current phrasing conflates the training objective with the evaluation goal and should be clarified for accuracy. 
*   **Insufficient Ablation to Justify the Reward Signal's Superiority:** The central claim is that coding standards provide effective *guidance* for functional correctness. However, the experiments in Table 2 do not fully support this. The comparison is between CodeRule-RL and CURE (a unit test-based RL method). While this shows CodeRule-RL is much more *efficient*, it does not prove that the rule-based signal is a better or even comparable *guide* for `pass@1`. The comparison conflates the reward source (rules vs. tests) with different RL implementations. A more convincing experiment would be to add a baseline within the authors' own framework: **`CodeRule-RL + unit tests`**, where the GRPO algorithm is used but the reward is derived from both executing unit tests and the rules. If the current `CodeRule-RL` (using only rule feedback) outperforms this new baseline, it would strongly support the claim that rule-based feedback is a superior training signal. Without this, one might conclude that the method is simply a faster, but potentially less effective, proxy for the true signal of functional correctness.
*   **Limited Discussion on Novelty** The idea of using automated feedback from tools like compilers or linters as a reward signal for RL has been explored in prior work. For instance, Dou et al. (2024, "StepCoder") explicitly use RL with feedback from compiler errors as part of their reward function. While the authors' approach of using *only* per-rule static analysis feedback, the novelty seems to be very incremental.

### Questions
The reported training efficiency of 1.6 hours is a major claimed advantage. However, the hyperparameters section reveals this is for only 80 optimization steps, which is an unusually short duration for a GRPO-based fine-tuning experiment. This raises questions about the scale of the experiment and the robustness of the findings.  Furthermore, could you provide more details on the size of the training dataset (e.g., number of unique prompts) to help contextualize whether the observed gains are the result of a comprehensive training process or short-term tuning on a limited set of problems?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
CodeRule-RL proposes standard guided reinforcement learning for code models where pass@1 is the only optimization target and MISRA C:2012 rule outcomes provide auxiliary, per rule reward signals. The paper defines a spec to reward mapping that converts static analyzer findings into bounded per rule penalties, aggregates them with a clipped reward, and schedules rule weights by a simple frequency driven curriculum that focuses on the most violated rules and reduces their weight as violation rates drop. Unit tests may appear as text in prompts but are not executed during training. Experiments on a frozen CodeContests+ C subset and several backbones show higher pass@1 and much lower reward latency and wall clock time than RL that executes tests during training.

### Strengths
The method is explicit and testable. The paper defines the per rule signals and the squashing map, the exponential penalty, and the clipped aggregate reward with clear bounds and rationale, which makes reuse straightforward. The curriculum definition uses an EMA of rule frequencies, a Top K active frontier, warmup and cool down, and a weight mask, again with equations and typical values, which supports replication. The training objective and optimizer are standard and fully specified. The evaluation is careful about decontamination, seed reporting, decoding settings, and toolchains. Gains in pass@1 are consistent across two model families and multiple sizes, with a marked reduction in training wall clock time and reward latency relative to execution based RL. The qualitative example illustrates that the policy learns rule aligned edits while preserving task logic, matching the aggregate improvements.

### Weaknesses
Scope is limited to single translation unit C and MISRA C:2012 with one static analyzer. The paper claims a standard agnostic design yet does not test a second analyzer or a different rule family. The curriculum has several hyperparameters $ (\lambda, \tau, W, T_{\mathrm{warm}}, T_{\mathrm{cool}}, K(0)) $. Defaults are given, but sensitivity analysis is limited, so brittleness under other data or analyzers is unclear. Gains in $ \mathrm{pass@1} $ without running tests may depend on correlation between static compliance and runtime success, and the paper notes association rather than causation. The VFR study with Infer is useful and suggests that smaller models may trade security for functionality after training; checking false positives and false negatives across analyzers would help. The appendix reports general coding benchmarks with little or no drop, yet results are brief and deserve clearer placement in the main text. Priority masking and gating are described, yet the ablation contrasts the curriculum against all rules without isolating the effect of each gate or the chosen thresholds.

### Questions
Can you isolate the effect of priority masking and within sample precedence by disabling each gate in turn or varying the gate threshold, and add sensitivity curves for $ \lambda $, $ \tau $, $ W $, $ T_{warm} $, $ T_{cool} $, and the Top $ K $ schedule to show how $ pass@1 $ changes

Can you quantify the link between static compliance and runtime success by executing held-out tests after training under the same prompt setting, reporting the correlation between violation rate and $ pass@1 $ across bins, and giving examples where compliance rises while $ pass@1 $ drops

Can you add statistical support for Table 1 and the CURE comparison by reporting confidence intervals or a randomization test, listing per-task variance, and extend compiler checks with extra flags and sanitizers beyond GCC 13 and Clang 17 to show stability across toolchains

### Soundness
3

### Presentation
3

### Contribution
3
