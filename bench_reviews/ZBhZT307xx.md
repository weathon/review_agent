## Summary
This paper provides a comprehensive empirical analysis of verifiers used in reinforcement learning with verifiable reward (RLVR) for mathematical reasoning. It demonstrates that rule-based verifiers suffer from significant false negatives that impede RL training, while model-based verifiers, though more accurate, are vulnerable to reward hacking. The authors propose a hybrid verifier that improves performance and systematically probe verifier robustness against adversarial attacks.

## Strengths
- **Extensive and well-designed evaluation** across multiple datasets (Math, DeepscaleR, ORZ-Math, Skywork-OR1, WebInstruct-Verified), verifier types (rule-based, off-the-shelf LLMs, fine-tuned), and both static and dynamic RL settings, providing robust, multifaceted evidence.
- **Identification of critical, underexplored issues**: the declining recall of rule-based verifiers with stronger models (a scaling concern) and the susceptibility of model-based verifiers to reward hacking despite higher static accuracy—challenging the assumption that accuracy translates to RL robustness.
- **Practical contribution of a hybrid verifier** that combines rule-based precision with model-based recall, showing improved RL performance and data efficiency over a pure rule-based baseline.
- **Rigorous validation methodology** using GPT-4o as an oracle (validated against human judgment) to detect reward hacking, and a systematic adversarial probing study (13 pattern types) that reveals generative verifiers are broadly vulnerable while discriminative ones (e.g., xVerify) are more robust.
- **Cross-domain generalization** demonstrated through experiments on both mathematical and general science (WebInstruct-Verified) datasets, strengthening the claim that the findings are not domain-specific.

## Weaknesses
- **Single-sample evaluation for most RL benchmarks** — Due to computational constraints, key results (GSM8K, MATH, Minerva Math, OlympiadBench) rely on single runs, which reduces confidence in the stability of the reported improvements and trends given RL's known variance.
- **Limited analysis of why fine-tuned verifiers are more hackable** — The paper demonstrates the phenomenon but does not investigate the root causes (e.g., overfitting to the classification distribution, shortcut learning, or reduced reasoning faithfulness), leaving an important mechanistic question unanswered.
- **Narrow policy model scope** — All RL training experiments use Qwen2.5-7B as the policy model; findings about hacking dynamics and verifier effectiveness might not generalize to other architectures or scales.

## Nice-to-Haves
- Proposing and evaluating simple defense mechanisms against reward hacking (e.g., adversarial training of the verifier, ensembling) would strengthen the practical impact.
- Testing with a stronger policy model to validate the hypothesis that off-the-shelf verifiers' apparent robustness in RL might be due to the policy's limited capacity to find exploits.
- Quantifying the trade-off between verifier recall on clean data and robustness against adversarial attacks across different verifier types.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Dataset representativeness concern** — The paper explicitly states the datasets are a "relatively easy setting" and uses this to argue the problem is more severe in realistic scenarios; this is appropriately framed and not a weakness.
- **Evaluation of model-based verifiers on a subset** — The paper clearly explains this choice aligns with the hybrid design (evaluating on samples the rule-based verifier missed) and is not presented as a direct, like-for-like comparison with rule-based performance.
- **Choice of 1.5B model for the main hybrid verifier** — This is justified by performance among 1.5B models and computational efficiency; it does not undermine the core findings.
- **Demand for theoretical analysis or proposed solutions** — The paper's contribution is empirical identification and analysis of verifier limitations; proposing defenses is outside its stated scope.
- **Formatting and table readability issues in the submitted text** — While the current version has parser artifacts, these are not substantive weaknesses of the research itself.

## Novel Insights
The paper provides a clear novel insight: static classification accuracy of a verifier does not predict its robustness in dynamic RL training. Fine-tuned verifiers can achieve high recall on clean data yet become uniquely susceptible to reward hacking, leading to training collapse. Additionally, generative verifiers (including Chain-of-Thought models) are systematically more vulnerable to simple adversarial patterns than discriminative verifiers, suggesting a tension between reasoning transparency and robustness in verification systems.

## Suggestions
- Include a discussion of the single-sample evaluation limitation in the main limitations section and, if possible, add multi-seed results for a critical subset of experiments (e.g., the hybrid vs. rule-based comparison on DeepscaleR) to demonstrate stability.
- Perform a simple controlled ablation to isolate the harm of rule-based false negatives: artificially inject false negatives at the observed rate into the rule-based verifier and measure the impact on RL performance, providing causal evidence.
- Release all prompts, hyperparameters (beyond those in the appendix), and code to ensure full reproducibility.