# CALM Before the STORM: Unlocking Native Reasoning for Optimization Modeling

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Large Reasoning Models (LRMs) have demonstrated strong capabilities in complex multi-step reasoning, opening new opportunities for automating optimization modeling. However, existing domain adaptation methods, originally designed for earlier instruction-tuned models, often fail to exploit the advanced reasoning patterns of modern LRMs --- In particular, we show that direct fine-tuning on traditional non-reflective datasets leads to limited gains. To fully leverage LRMs’ inherent reasoning abilities, we propose **CALM** (Corrective Adaptation with Lightweight Modification), a framework that progressively refines LRMs within their native reasoning modes for optimization modeling tasks. In CALM, an expert intervener identifies reasoning flaws and provides concise corrective hints, which the LRM incorporates to produce improved reasoning trajectories. These interventions modify fewer than 2.6% of generated tokens, but generate high-quality data for soft adaptation through supervised fine-tuning. The adapted model is then further improved through reinforcement learning. Building on CALM, we develop **STORM** (Smart Thinking Optimization Reasoning Model), a 4B-parameter LRM that achieves a new state-of-the-art average accuracy of 68.9% across five popular optimization modeling benchmarks, matching the performance of a 671B LRM. These results demonstrate that dynamic, hint-based data synthesis both preserves and amplifies the native reasoning patterns of modern LRMs, offering a more effective and scalable path towards expert-level performance on challenging optimization modeling tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes CALM (Corrective Adaptation with Lightweight Modification), a framework that refines large reasoning models (LRMs) for optimization modeling (OR) tasks through an automated Reasoner–Intervener collaboration loop. By detecting and correcting reasoning flaws via lightweight interventions, CALM generates high-quality expert trajectories to train STORM, achieving near–DeepSeek-R1 performance with a 4B model on multiple OR benchmarks.

### Strengths
- The paper introduces a novel self-refining adaptation framework (CALM) that enhances LRM reasoning without relying on large teacher models.
- The proposed Reasoner–Intervener loop effectively identifies and corrects reasoning flaws, producing high-quality training trajectories with minimal intervention.
- Experiments on multiple OR benchmarks demonstrate strong performance—STORM (4B) nearly matches the 671B DeepSeek-R1, showing impressive parameter efficiency.

### Weaknesses
- The paper does not include a comparison with the typical teacher-generated data paradigm, where large LRMs (e.g., DeepSeek-R1) generate reasoning trajectories to train smaller models. This omission makes it unclear whether CALM’s self-refining pipeline achieves comparable data quality and efficiency to large-model supervision.
- The experiments use only the Qwen3-4B model, without testing other architectures (e.g., Llama, Phi) or larger scales (8B, 14B). This limits the understanding of CALM’s generality and scalability across different model families and sizes.
- The paper does not evaluate whether CALM/STORM training affects the model’s original general abilities (e.g., math reasoning, coding, or general instruction following). It remains unclear if domain specialization leads to performance degradation on non-OR tasks.

### Questions
- Can the authors provide or discuss a baseline where a large reasoning model (e.g., DeepSeek-R1) generates the training data, to compare against CALM’s self-refinement approach in terms of data quality, performance, and cost?
- Have the authors tested CALM on larger or different LRM architectures to verify whether its self-refining mechanism generalizes beyond Qwen3-4B?
- Have the authors examined the impact of CALM training on the base model’s general capabilities (e.g., math, code, GPQA)? Does specialization in optimization modeling cause catastrophic forgetting or trade-offs in other domains?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper targets optimization modeling (OR tasks like LP/ILP) as a reflective reasoning problem for large reasoning models (LRMs). The authors observe that standard SFT on non-reflective QA pairs often hurts complex problem performance by discouraging multi-step, code-driven reasoning.

They propose CALM (Corrective Adaptation with Lightweight Modification): within an executable environment (natural language + Python + solver), an Intervener LLM inserts very short, pinpoint hints into the model’s ongoing reasoning, correcting two pervasive errors: (1) Code/solver underuse or distrust and (2) OR modeling gaps (e.g., LP vs ILP, missing/incorrect constraints, implementation drift). The corrected “gold” traces supervise SFT to calibrate behavior while preserving reflective style. Then RL (GRPO) continues training in the same executable loop, rewarding correct/optimal solutions to reinforce computation-driven reasoning.

### Strengths
Treats optimization modeling explicitly as a reflective reasoning problem, avoiding the way standard QA-style SFT suppresses multi-step thought and the code–solver loop.

Uses very small, pinpoint hints (few-token edits) to convert flawed chains into “gold” traces—preserving the model’s native reasoning style while correcting errors.

Keeps a full trail from error → micro-hint → fix → solver-verified result, which supports failure analysis and produces instructive, teaching-style exemplars.

Validated across NL4Opt, MAMO-Easy/Complex, IndustryOR, and OptMath; gains are especially strong on the harder subsets, making the claims more persuasive.

### Weaknesses
All CALM data synthesis and the two-stage training pipeline start from one base LRM—Qwen3-4B-Thinking-2507—used as the Reasoner, with Gemini-2.5-Pro as the Intervener. The paper does not report adapting multiple base LRMs to test generality of the pipeline.

In the reproducibility statement, the authors say they plan to release code and models, implying they are not available yet, which limits independent verification and adoption.

The overall “reflect–feedback–revise with executable code/solver” paradigm has been explored in prior art such as Reflexion and Self-Refine for iterative self-correction, PAL for code-centered reasoning, and the NL4Opt line for NL-to-OR modeling; several MILP auto-formulation works also adopt prompt-/template-based pipelines. Hence, while the paper integrates these ideas neatly, the framework itself is conceptually close to existing paradigms.

### Questions
See Weakness

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
4

### Summary
This paper explores how to adapt Large Reasoning Models (LRMs) for automated optimization modeling.

 The authors identify that conventional instruction-tuning methods - based on static, non-reflective datasets - undermine the multi-step reasoning patterns inherent in LRMs.

They propose CALM (Corrective Adaptation with Lightweight Modification), a hint-based intervention framework that corrects reasoning flaws while preserving the model’s reflective reasoning flow. Through CALM, the authors curate high-quality reasoning trajectories and use them to fine-tune an LRM, followed by reinforcement learning to produce STORM, a 4B-parameter model that achieves 68.9% average accuracy across five optimization benchmarks - matching the performance of a 671B model.

The work includes a taxonomy of reasoning flaws, an interpretable correction protocol, and detailed ablation and behavioral analyses showing that CALM’s hint-based adaptation leads to more sample-efficient reinforcement learning.

### Strengths
Original idea of aligning adaptation with native reasoning instead of retraining from scratch.

Methodological soundness: clear formalization of the reasoning loop and rigorous evaluation.

Practical impact: dramatic performance/parameter efficiency gains (4B → 671B equivalence).

High interpretability: the taxonomy and hint examples make the improvement mechanism transparent.

Thorough analysis: ablations and behavioral plots (code-usage ratio, response length, flaw frequency) offer unusually strong introspection for this domain.

Reproducibility: detailed appendices and explicit commitment to code release.

### Weaknesses
Limited dataset size in the CALM curation phase (112 “golden” trajectories) may raise questions about scalability; a brief discussion on automating large-scale generation would strengthen the paper.

Domain scope: experiments are confined to optimization modeling; it would be valuable to show transfer to another structured-reasoning domain (e.g., planning or theorem proving).

Comparative baselines: although strong, the study lacks direct comparison to contemporaneous reflective-alignment frameworks like START or CoRT under identical setups.

Intervener dependence: using Gemini-2.5-Pro as the expert may raise reproducibility concerns if closed-source models differ in reasoning behavior.

Some claims of “expert-level performance” might benefit from human expert evaluation rather than relying solely on pass@1 accuracy.

### Questions
How sensitive is CALM to the choice of Intervener model? Could a smaller or open-source Intervener reproduce similar quality in the curated trajectories?

The current pipeline uses a discrete number (≤ 5) of interventions. Have you explored adaptive stopping criteria or uncertainty-driven intervention scheduling?

Since CALM focuses on hint-based reasoning correction, could its framework generalize to non-code-based reasoning tasks (e.g., logical proofs, multi-hop QA)?

In reinforcement learning, the reward function is binary on final correctness. Have you considered incorporating intermediate reflection-quality rewards to further stabilize training?

Finally, how does STORM perform under zero-shot generalization to unseen industrial problem types not represented in the benchmarks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents CALM (Corrective Adaptation with Lightweight Modification), a framework for aligning large reasoning models (LRMs) with optimization modeling tasks. The key idea is that simply fine-tuning LRMs on non-reflective datasets — i.e., problem–solution pairs without intermediate reasoning — harms their performance on complex problems by suppressing their native multi-step reasoning ability. To address this, CALM introduces a “Reasoner–Intervener” collaboration pattern where an expert model identifies reasoning flaws and provides minimal corrective hints (fewer than 3% of tokens). The resulting corrected trajectories are used to fine-tune the base LRM, followed by reinforcement learning to achieve full adaptation. The final model, named STORM (Smart Thinking Optimization Reasoning Model), achieves 68.9% average accuracy across five optimization benchmarks, matching the performance of a much larger 671B model while using only 4B parameters. The paper combines an insightful problem diagnosis, a clean methodological design, and strong empirical validation.

### Strengths
The main strengths lie in the depth and coherence of the study. The authors don’t just propose a new method—they build a full story around understanding why current adaptation methods fail, and how to fix that in a principled way. The CALM framework is well-motivated and the “Reasoner–Intervener” pattern is intuitive and scalable. The two-stage training pipeline (SFT then RL) is validated thoroughly, with strong ablation studies that isolate each component’s contribution.

### Weaknesses
The weaknesses are relatively minor. The evaluation, while comprehensive, depends heavily on Qwen and Gemini models; testing on a different family (e.g., DeepSeek or Llama) would help show generality. Some parts of the taxonomy (especially the seven triggers) could be streamlined — the paper spends many pages on classification detail that could be summarized. There is also limited theoretical discussion about why minimal interventions produce such a large effect; the explanation is intuitive but mostly empirical. Lastly, while the results are impressive, it’s not entirely clear how much of the performance gain comes from the data filtering versus the interventions themselves.

### Questions
How sensitive is the performance of STORM to the 2.6% intervention ratio? Would increasing or decreasing the intervention intensity change the outcome significantly?

Could CALM be generalized to domains beyond optimization modeling, or does it rely heavily on the availability of an executable solver for immediate feedback?

Have the authors verified whether Gemini-2.5 as the Intervener introduces stylistic bias into the reasoning traces?

How do you ensure that the reinforcement learning stage does not overfit to the benchmarks used for calibration?

Is there a qualitative difference between trajectories produced after CALM fine-tuning and those after RL, beyond shorter length and higher correctness?

### Soundness
3

### Presentation
2

### Contribution
3
