# When Models Outthink Their Safety: Mitigating Self-Jailbreak in Large Reasoning Models with Chain-of-Guardrails

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
Large Reasoning Models (LRMs) demonstrate remarkable capabilities on complex reasoning tasks but remain vulnerable to severe safety risks, including harmful content generation and jailbreak attacks. Existing mitigation strategies rely on injecting heuristic safety signals during training, which often suppress reasoning ability and fail to resolve the safety-reasoning trade-off. To systematically investigate this issue, we analyze the reasoning trajectories of diverse LRMs and uncover a phenomenon we term Self-Jailbreak, where models override their own risk assessments and justify responding to unsafe prompts. This finding reveals that LRMs inherently possess the ability to reject unsafe queries, but this ability is compromised by Self-Jailbreak, resulting in harmful outputs. Building on these insights, we propose the Chain-of-Guardrail (CoG), a training framework that recomposes or backtracks unsafe reasoning steps, steering the model back onto safe trajectories while preserving valid inference chains. Extensive experiments across multiple reasoning and safety benchmarks demonstrate that CoG substantially improves safety of current LRMs while preserving comparable reasoning ability, significantly outperforming prior methods that suffer from severe safety–reasoning trade-offs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates safety vulnerabilities in Large Reasoning Models (LRMs), introducing the concept of “self-jailbreak” — where models initially recognize harmful intent but later override their own safety judgment during reasoning. The authors analyze this behavior and propose Chain-of-Guardrails (CoG), a training framework built on Safety Recomposition and Safety Backtracking to correct unsafe reasoning traces while preserving reasoning performance. Experiments on Qwen3 models across safety and reasoning benchmarks show improved refusal behavior and minimal reasoning degradation.

### Strengths
1. Practical training recipe combining supervised safety recomposition and safety backtracking that operates within the model’s reasoning process rather than only output filtering.
2. Extensive experiments on multiple model sizes (8B/14B/32B), covering both safety refusals and jailbreak attack robustness.
3. Strong empirical trade-off: improves safety without harming reasoning; in some settings even improves reasoning benchmarks (e.g., AIME/GPQA).
4. Good interpretability support via PCA trajectory analysis and cognitive-behavior profiling of reasoning patterns.
5. Comprehensive baselines (SafeKey, SafeChain, STAR-1, SAFEPATH), reproduced with apparent care.

### Weaknesses
1. While the proposed method is well-motivated, parts of the approach resemble recent CoT-based safety alignment practices (e.g., recomposition and backtracking with supervised training objectives), and further clarification of the conceptual distinction would strengthen the contribution.
2. Heavy dependence on LLM-as-Judge pipelines; potential circular evaluation bias since judge models and training supervision rely on similar model families.
3. Dataset construction pipeline and annotation fidelity require deeper transparency (e.g., what fraction of tokens are human-curated vs. model-generated, inter-annotator agreement on safety decisions beyond sample audits).
4. Generalization beyond Qwen family unclear — method might rely on specific model inductive biases.
5. Limited analysis on robustness against intentionally adaptive jailbreak strategies (e.g., obfuscated CoT, indirect prompting, multi-turn persuasion).
6. Some core claims (preservation of reasoning trajectory) would benefit from ablation or human validation beyond PCA and prompt-based cognitive metrics.

### Questions
1. How does the approach perform on models with fully private internal chains-of-thought (e.g., concealed CoT-decoding gates)?
2. Can the authors provide more details on failure cases where self-jailbreak persists after training?
3. How might CoG scale to RL-trained LRMs or RLHF-free architectures?
4. Did the authors test on adversarially optimized jailbreak prompts (not just public benchmarks)?
5. Is there any observable transfer benefit when training CoG on one model and evaluating on another?
Note: I am currently leaning toward a borderline rejection. However, I am open to increasing my score if the concerns outlined above are adequately addressed during the rebuttal phase.

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
4

### Summary
This paper identifies the issue of "Self-Jailbreaking", where LRMs realize a question is unsafe but end up persuading themselves into answering it, and introduces a technique, Chain-of-Guardrails (CoG), to fix it. Practically, the authors propose two ways, Safety Recomposition and Safety Backtrack, to construct reasoning trajectories for training with loss masking. The method is implemented on Qwen3 family and tested on multiple safety and reasoning benchmarks to validate its effectiveness.

### Strengths
* The introduction of “Self-Jailbreak” is interesting, which helps explain why smarter models sometimes fail to keep safe.
* The two proposed strategies for data construction are reasonable and effective.
* The experiments are extensive on multiple popular benchmarks and the results show strong safety improvements while keeping reasoning performance close to baseline.

### Weaknesses
* The paper doesn’t fully explain how the Safety Recomposition and Safety Backtrack steps are actually implemented. For instance, what model is used for safety reasoning chain recomposition ? Is it done in a self-evaluating manner or with an external teacher model ? What is the additional computational cost induced by these two steps ?
* While CoG is shown to improve safety, it doesn’t test whether the model has the issue of over-refusal, which is a commonly seen problem.
* The paper aims to balance safety and reasoning ability in LRMs, but it overlooks several important works that tackle safety–utility trade-offs or safety alignment in reasoning models. Some papers are listed below and should be included for discussion.

[1] Deliberative Alignment: Reasoning Enables Safer Language Models

[2] STAIR: Improving Safety Alignment with Introspective Reasoning

[3] SaRO: Enhancing LLM Safety through Reasoning-based Alignment

[4] RealSafe-R1: Safety-Aligned DeepSeek-R1 without Compromising Reasoning Capability

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

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
The paper analyzes safety failures in large reasoning models (LRMs) and identifies a key phenomenon called Self-Jailbreak, where models consciously recognize risk but later override their own safety judgments to produce harmful outputs. The authors classify four types of Self-Jailbreak behaviors (Benign Reframing, Warning, Logical Fallacies, and Harm Identification) and show that these underlie many unsafe generations. To counter this, they propose Chain-of-Guardrails (CoG), a training framework combining Safety Recomposition (rewriting unsafe reasoning chains) and Safety Backtrack (inserting self-checks to correct risky reasoning). Across multiple benchmarks, CoG achieves strong safety improvements while maintaining reasoning performance, outperforming prior methods like SafeKey and SafeChain. Overall, the study offers a structured diagnosis of reasoning-based safety failures and a promising mitigation approach that preserves models’ cognitive coherence.

### Strengths
The paper provides a comprehensive self-jailbreak analysis on different scenarios, and all of these make clear motivation for the proposed method.

The CoG recipe is practical and modular: Safety Recomposition (SafR) to rewrite unsafe steps and Safety Backtrack (SafB) to append a self-check—paired with a selective loss mask

Empirically, the story is consistent: CoG sits in the top-right of the safety–reasoning chart and beats prior baselines, with PCA showing safety clusters move away from the boundary while reasoning stays basically put; they also report SOTA trade-off numbers (incl. gains vs. SafeKey).

### Weaknesses
(1) I found the technical part of the paper is a bit ambiguous and hard to follow, with a lot of details missing. For example, when do you use equation (1), is this used for SR/SB step or in the final alignment? What do "preserve" and "reasoning \pi0" mean in (1)? What do "merge" mean in (3)? For (3), what is the recomposition procedure, details?  In 3.2, how do you determine the mask? What type of training is in the alignment training, is it sft or rl? 

(2) Is there any runtime analysis of this proposed method, what is the cost of such improved safety?

(3) It will be better to also have experiments on other model families besides qwen families which have shown some unusaul reasoning behaviors compared to others.

(4) When do you recommend to use safR and when to use safB? what are the respective advantages?

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper identifies a phenomenon termed "Self-Jailbreak" in LRMs, where models recognize harmful intent during reasoning but subsequently override their own safety judgments to produce unsafe outputs. The authors propose Chain-of-Guardrail (CoG), a training framework with two strategies: Safety Recomposition (SafR) and Safety Backtrack (SafB), designed to mitigate Self-Jailbreak while preserving reasoning capabilities. Experiments on Qwen3 models demonstrate improved safety-reasoning trade-offs compared to baseline methods.

### Strengths
- The paper attempts to analyze safety failures in reasoning models by characterizing the Self-Jailbreak phenomenon.

- The paper provides a taxonomy of Self-Jailbreak types with empirical frequency analysis.

- The paper is clear and easy to understand.

### Weaknesses
- **Highly Questionable Motivation.** Table 1 reports that nearly all tested LRMs exhibit 95-99%+ unsafe response rates (DS-R1-0528: 95.93%, Qwen3-8B: 99.79%, Qwen3-14B: 99.85%, Qwen3-32B: 99.64%). This is completely inconsistent with Table 2, where the same Qwen3 models show dramatically different safety performance. This massive discrepancy between 99.79% unsafe in Table 1 versus reasonable safety in Table 2 is never explained. Furthermore, if LRMs truly exhibit 95-99% Self-Jailbreak rates as claimed, why do current jailbreak attack methods in the literature struggle to achieve even worse success rates? If models naturally bypass their own safety 99% of the time, these elaborate methods would be unnecessary. This logical inconsistency strongly suggests that the Self-Jailbreak phenomenon as measured in Section 2 is an artifact of flawed evaluation. The paper uses LLM-as-Judge and Llama-Guard but provides no validation that these automated judgments align with human assessment and no inter-annotator agreement scores. Without human validation studies, the entire Self-Jailbreak characterization lacks credibility.

- **Minimal Methodological Novelty.** The proposed CoG framework lacks genuine methodological innovation and essentially reduces to standard supervised fine-tuning on curated data. The core approach is: (1) identify problematic reasoning chains, (2) use the same model to generate corrected chains, (3) fine-tune on corrected data. This is conceptually identical to existing safety fine-tuning methods, just applied to reasoning chains rather than direct responses. The so-called selective loss masking strategy appears also arbitrary. SafR applies loss to all tokens while SafB masks the original reasoning chain, but the paper provides no theoretical analysis or systematic ablation studies exploring alternative masking configurations. Table 4 only compares masked versus unmasked SafB (a trivial baseline), not different masking strategies. The distinction between SafR and SafB is superficial; both fundamentally rely on generating corrected reasoning chains and fine-tuning, with only minor differences in gradient updates. The entire contribution is essentially data curation rather than innovation.

- **Insufficient Evaluation.** The reasoning capability evaluation uses only GPQA-Diamond and AIME2024, both extremely small benchmarks with high variance that cannot reliably detect meaningful differences. No standard reasoning benchmarks like GSM8K, full MATH, MMLU, or HumanEval are included. Additionally, the scope of models is limited to Qwen-3 series, which is also not enough.

- **Superficial Related Work.** The related work section (Section 6) is cursory with only two brief subsections. Section 6.1 cites just 3 papers with minimal discussion, and Section 6.2 lists four baseline methods without systematic analysis of their approaches or limitations.

### Questions
The questions are listed in the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
