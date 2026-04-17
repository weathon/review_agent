# The Realignment Problem: When Right becomes Wrong in LLMs

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
The alignment of Large Language Models (LLMs) with human values is central to their safe deployment, yet current practice produces static, brittle, and costly-to-maintain models that fail to keep pace with evolving norms and policies. This misalignment, which we term the Alignment-Reality Gap, poses a growing challenge for reliable long-term use. Existing remedies are inadequate: large-scale re-annotation is economically prohibitive, and standard unlearning methods act as blunt instruments that erode utility rather than enable precise policy updates. We introduce TRACE (Triage and Re-align by Alignment Conflict Evaluation), a framework for principled unlearning that reconceives re-alignment as a programmatic policy application problem. TRACE programmatically triages existing preference data against a new policy, identifies high-impact conflicts via a alignment impact score, and applies a hybrid optimization that cleanly inverts, discards, or preserves preferences while safeguarding model performance. Empirical results show that TRACE achieves robust re-alignment across diverse model families (Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B). On both synthetic benchmarks and the PKU-SafeRLHF dataset under complex policy shift, TRACE enforces new principles without degrading general capabilities. Our work establishes a scalable, dynamic, and cost-effective paradigm for maintaining LLM alignment, providing a foundation for sustainable and responsible AI deployment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses a critical and timely challenge in the deployment of Large Language Models (LLMs): the Alignment-Reality Gap. This gap arises because current alignment methods produce static models that cannot adapt to evolving societal norms, corporate policies, or regulations. The authors argue that existing solutions—costly full re-annotation or blunt machine unlearning techniques—are inadequate. To solve this, they propose TRACE (Triage and Re-align by Alignment Conflict Evaluation), a novel framework that reframes re-alignment as a programmatic policy application problem over existing preference data. Experiments on both a synthetic benchmark (SynthValueBench) and the real-world PKU-SafeRLHF dataset demonstrate that TRACE achieves re-alignment performance close to a “gold standard” DPO re-training (which uses full new annotations), while preserving general capabilities and showing strong robustness under adversarial attacks.

### Strengths
1. The paper clearly articulates the “Alignment-Reality Gap” as a fundamental limitation of current LLM deployment practices. This framing is insightful and highly relevant to real-world AI governance.

2. The core idea—reinterpreting old data through a new policy lens—is conceptually elegant and represents a significant paradigm shift from data-centric to policy-centric re-alignment.

3.  The three-stage TRACE pipeline is well-designed. The triage mechanism is a crucial innovation that overcomes a key flaw in naive unlearning. The hybrid loss and alignment impact weighting together enable precise, utility-preserving updates.

4.  The experiments are thorough. The use of a synthetic benchmark with ground-truth policy labels provides a controlled validation, while the complex, multi-axis policy shift on PKU-SafeRLHF demonstrates real-world applicability. Evaluation across three diverse model families (Qwen, Gemma, Llama) further strengthens the claims. The inclusion of human preference studies, general capability benchmarks, and adversarial robustness tests provides a holistic view of performance.

### Weaknesses
### 1. The Core Assumption is Unrealistic and Untenable in Practice

The entire framework hinges on the existence of a perfect, programmable policy oracle $\pi_{new}$ that can reliably classify any model response as “compliant” or “non-compliant.” This is a fantasy.

- In the real world, policies are ambiguous, contextual, and often contradictory. The paper’s own $\pi_{\text{new}}$ for PKU-SafeRLHF (Fig. 5) is a 5-page legalistic document with tiered priorities and contextual exceptions. Encoding this into a reliable, zero-error program is a research problem of its own magnitude—far beyond the scope of this paper.
- The paper provides no evidence that such an oracle can be built robustly. It simply assumes it as a black box. This renders the entire contribution theoretically sound but practically inapplicable. It solves a problem that doesn’t exist (perfect policy specification) while ignoring the real problem (imperfect, evolving policy specification).


### 2. The Method is Fundamentally Unsound for “Invert” Samples

The False Dichotomy Problem is correctly identified, but the proposed solution for Type I (Invert) samples is naive and dangerous.

- The method assumes that if $y_w$ is non-compliant and $y_l$ is compliant, then $(y_l, y_w)$ is a valid new preference pair. This is a massive leap of faith.
- The original $y_l$ was generated as a deliberately bad response to contrast with $y_w$. It may be compliant with $\pi_{{new}}$ only by being vacuous, evasive, or unhelpful. Promoting such a response to the “preferred” status actively degrades model helpfulness and quality.
- The paper provides no analysis of the quality of these inverted $y_l$ responses. It’s optimizing for policy compliance at the direct expense of response quality, which is a core tenet of alignment (helpfulness + harmlessness).

### 3. The Alignment Impact Score is a Hand-Wavy Justification for a Simple Heuristic

The derivation of the alignment impact score (Eq. 5) is presented as a rigorous theoretical contribution, but it’s built on unrealistic and unjustified approximations.

- The assumption that the Hessian $H \approx \gamma I$ is standard but unjustified in this context. For a complex, non-convex loss landscape of an LLM, this approximation can be wildly inaccurate, making the dot product $\langle g_J, g_{L_i} \rangle$ a poor proxy for the true marginal gain.
- The paper provides no ablation study to show that this weighting scheme is necessary or even beneficial. It’s entirely possible that uniform weighting or a simpler heuristic (e.g., based on policy violation severity) would perform just as well, rendering this “theoretical” contribution superfluous.

### 4. The Experiments Lack Critical Baselines and Robustness Checks

The experimental validation is superficial and cherry-picked.

- Missing Baseline: The most obvious and practical baseline is full fine-tuning on a small, newly annotated dataset (e.g., 1k samples). This is far more realistic than full re-annotation and would test if TRACE’s complexity is worth it. Its absence is a glaring omission.
- Oracle Sensitivity: There is zero analysis of how TRACE’s performance degrades with a noisy or imperfect $\pi_{{new}}$. A single experiment with a corrupted oracle would reveal the method’s brittleness, but the authors avoid this critical test.
- Generalization: The adversarial attacks (Table 3) are simplistic. The paper does not test generalization to out-of-distribution prompts or novel value conflicts not seen in the original dataset. The claim of “robust re-alignment” is therefore unsubstantiated.

### 5. The Broader Impact and Ethical Implications are Glossed Over

The paper promotes a method for automatically editing a model’s values. This is a profoundly powerful and dangerous capability that is treated with shocking casualness.

- The framework could be trivially used to instill harmful or biased policies just as easily as beneficial ones. The paper offers no safeguards, no discussion of misuse potential, and no consideration of who controls the $\pi_{{new}}$ oracle.
- By framing re-alignment as a purely technical, programmatic problem, the paper depoliticizes a deeply political act—changing an AI’s moral compass. This is a serious ethical oversight.

### Questions
1. The paper’s success hinges on $\pi_{{new}}$ being programmable. How would TRACE handle a policy shift that is qualitative or based on subjective judgment (e.g., “be more empathetic”)? Could the framework be combined with methods like Inverse Constitutional AI (ICAI), mentioned in the appendix, to *learn* $\pi_{\text{new}}$ from a few examples?

2.  What is the sensitivity of TRACE to errors in the $\pi_{\text{new}}$ oracle? If $\pi_{\text{new}}$ misclassifies a Retain sample as Punish, could the KL regularization term be strong enough to prevent degradation, or would this lead to unnecessary utility loss?

3. The experiments involve well-defined policy shifts. How would TRACE scale to a scenario where $\pi_{\text{new}}$ introduces entirely new value dimensions not present in the original data ($\pi_{\text{old}}$)? The current triage mechanism might not capture such “out-of-support” requirements.

4.  If an LLM undergoes multiple sequential re-alignments ($\pi_1 \rightarrow \pi_2 \rightarrow \pi_3 \dots$), should the reference model ($M_{\text{ref}}$) be updated after each step, or should it always remain the original base model? The paper uses the original model as the anchor, but this could accumulate drift over many iterations.

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
5

### Summary
This paper addresses the "Alignment-Reality Gap" problem where LLMs become misaligned as societal values evolve. The authors propose TRACE (Triage and Re-align by Alignment Conflict Evaluation), a framework that enables dynamic re-alignment without expensive re-annotation. TRACE programmatically identifies conflicts between existing preference data and new policies using an "alignment impact score," then applies a hybrid optimization strategy (inverting, discarding, or preserving preferences) to update model behavior. Experiments across three model families (Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B) demonstrate that TRACE successfully enforces new alignment principles on both synthetic benchmarks and PKU-SafeRLHF dataset while maintaining general capabilities.

### Strengths
**Originality**: The paper introduces a novel and important problem formulation—the Alignment-Reality Gap—which addresses a practical challenge in LLM deployment that has received limited attention. The TRACE framework is conceptually innovative in treating re-alignment as a programmatic policy application problem rather than a simple unlearning task.

**Quality**: The experimental design is comprehensive, testing across three distinct model families (Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B) and multiple scenarios (synthetic benchmarks and PKU-SafeRLHF). The alignment impact score is a principled metric for identifying conflicting preferences, and the hybrid optimization strategy (invert/discard/preserve) is well-motivated.

**Clarity**: The paper is well-written with clear motivation and problem statement. The methodology is explained systematically, and figures effectively illustrate the framework. The progression from problem to solution is logical and easy to follow.

**Significance**: This work addresses a critical real-world problem for maintaining aligned LLMs in production. The ability to update alignment principles without expensive re-annotation is highly valuable for practitioners. The results demonstrate practical applicability across multiple model families, suggesting broad utility.

### Weaknesses
**Scalability Analysis**: The paper does not thoroughly analyze computational costs or scalability to larger models (e.g., 70B+ parameter models) or larger preference datasets. Given that the method involves computing alignment impact scores for all preference pairs, understanding the computational overhead is critical for practical deployment.

**Threshold Selection**: The paper uses fixed thresholds (α=0.6, β=0.4) for the triage process but does not provide sufficient guidance on how to select these thresholds for new scenarios or justify their robustness across different policy shifts and model families.

**Long-term Alignment Drift**: The experiments focus on single policy shifts, but do not investigate whether repeated applications of TRACE over multiple policy updates lead to cumulative degradation in model quality or alignment coherence.

**Evaluation Metrics**: The evaluation relies heavily on win-rate comparisons and categorical accuracy. More nuanced metrics examining alignment quality, preference consistency, and potential unintended behavioral changes would provide deeper insights.

### Questions
1. **Computational Complexity**: What is the computational overhead of TRACE compared to standard DPO training? Specifically, how does the alignment impact score computation scale with dataset size and model parameters? Can you provide wall-clock training times and memory usage for the experiments?

2. **Multiple Policy Shifts**: Have you tested TRACE on sequential policy updates? For instance, if a model undergoes 3-4 policy shifts over time, does performance degrade? How does TRACE compare to re-training from scratch in such scenarios?

3. **Preference Consistency**: After applying TRACE, how do you ensure the updated preference data remains internally consistent? Could the triage process introduce contradictory preferences that confuse the model?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the ‘alignment reality gap’, whereby aligned LLLMs become misaligned over time as human values evolve. They propose TRACE as an approach for re-aligning LLMs to new value frameworks without full data re-annotation and retraining. TRACE uses programmatic triage to classify existing data preference pairs based on their correctness under the new policy, followed by a hybrid optimisation that matches the training loss to the data conflict category (invert/punish /retain).

The method is tested on 2 pairs of old and new policies and 3 different open-weights LLMs (7B-9B parameters). It shows improvements in human preference evaluations compared to a U2A baseline, along with minimal degradation in general capabilities.

This work makes useful conceptual contributions, proposing the programmatic triage and hybrid optimisation objectives, and offers some good analysis as a limited proof of concept. However, there are a number of methodological limitations and missing details. For example, only 2 policy pairs are tested, the work lacks a proper validation of its efficiency claims, and the details of the adversarial robustness tests and results are unclear.

### Strengths
The reframing of the data re-annotation problem as a re-interpretation problem is useful and addresses the important problem of maintaining alignment as values drift.

The hybrid objective design is an elegant way to match the loss function to the policy conflict type.

The experimental validation is generally thorough: evaluating the retrained models on different benchmarks to ensure performance is not affected is a useful validation, benchmarking against full retraining and U2A as a baseline and testing on 3 models.

The paper is generally well written and easy to follow, though some important experimental details are lacking (see below).

### Weaknesses
Only two policy pairs are tested (one per dataset). To validate this approach it would be important to include multiple different, more variable policy pairs per dataset e.g. different numbers of transformations (invert/retain/punish). I appreciate this adds a lot of experimental overhead, but it would be useful to see, even for a single model.

There does not seem to be strong evidence for the robustness of the Hessian approximation in this context, or the benefit of using the alignment impact score. Can you run an ablation study which evaluates this method without the alignment impact weight - i.e. with fixed hyperparameters instead?

A core claim of the work is cost effectiveness, but there is no theoretical or empirical analysis of the efficiency gains. For example, what overhead does the KL regularisation and calculation of the alignment impact weighting add to training?

The reliance on LLMs to label data based on desired human values raises a number of concerns that should be better discussed, also in relation to related work which addresses similar challenges, e.g. constitutional AI, and works assessing the reliability of LLMs used as judges.

**Minor comments**

Check your citation formatting! (e.g. lines 50 and 52-53 should be in brackets)

The framing feels exaggerated in several places, for example describing the work as a ‘paradigm shift’ (line 81). Toning this down would improve the readability and credibility of the work.

The related work section seems limited. For example, it misses several recent works which apply a KL penalty in training to restrict behavioural changes [1-3]. However, I am not very familiar with the other areas of related work.

[1] Azarbal et. al. Selective Generalization: Improving Capabilities While Maintaining Alignment. (2025) https://www.lesswrong.com/posts/ZXxY2tccLapdjLbKm/selective-generalization-improving-capabilities-while
[2] Turner et. al. Narrow Misalignment is Hard, Emergent Misalignment is Easy. (2025) https://www.lesswrong.com/posts/gLDSqQm8pwNiq7qst/narrow-misalignment-is-hard-emergent-misalignment-is-easy 
[3] Kaczer et. al. In-Training Defenses against Emergent Misalignment in Language Models. (2025) https://arxiv.org/abs/2508.06249

### Questions
Table 3 claims that the attack resilience is ‘nearly indistinguishable’ from the gold-standard full retraining. However, the numbers appear to show that it in fact performs similarly to the U2A method - can you explain this?

How were the adversarial stress tests conducted? Please include the prompting and evaluation details in the Appendices.

See also the questions in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper identifies the "Alignment-Reality Gap", where a model's alignment remains static while real-world policies evolve rapidly. The authors highlight limitations of existing re-alignment approaches, including costly re-annotations for re-training and inappropriate use of machine unlearning methods such as U2A. To address this, they propose TRACE, the first programatic framework for principled unlearning. Given an existing preference dataset and an updated preference policy, TRACE uses a programmatic oracle to categorize the data pairs into three types: (1) conflicting pairs; (2) invalid pairs; and (3) unchanged pairs. It applies a hybrid optimization approach that employs DPO, NPO, and KL-divergence to each type, guided by alignment-impact weighting. The authors demonstrate the effectiveness of the proposed methods on two datasets (PKU-SafeRLHF and a synthetically constructed data SynthValueBench) across three models (Qwen2.5 7B, Llama3.1 8B, Gemma2 9B). They also empirically show that the method does not hurt general performance or robustness.

### Strengths
1. The paper presents a clear and well-motivated framework TRACE, highlighting the limitations of existing work and explaining why TRACE is deigned in its particular 3-category way. The motivation and methodology are communicated clearly and logically.
2. The authors provide comprehensive empirical results across three models (Qwen2.5-7B, Gemma2-9B, Llama3.1-8B) to demonstrate the effectiveness and generalizability of TRACE for realignment. They also evaluate on utility and adversarial attack datasets to assess its impact on general performance and robustness.

### Weaknesses
1. TRACE categorizes the existing dataset into three types, Invert, Punish, and Retain, and employs three different losses to each. However, the paper lacks sufficient baselines to demonstrate the necessity of using different losses instead of applying DPO alone which is simpler and commonly used. 
   - For Retain, KL divergence is used to penalize deviation from correctly aligned behaviors. However, applying DPO loss to the retained pairs might achieve a similar effect. An empirical comparison demonstrating the advantage of KL divergence in terms of quality or efficiency (e.g. FLOPs/speed) would strengthen the claim.
   - For Punish, the D_{II} subset is corrected using an LLM oracle to construct DPO pairs. It is unclear whether applying NPO to the negative counterparts is still necessary, as the DPO might steer the model away from the negative counterparts. An ablation study would clarify the contribution of NPO.
2. The paper considers DPO-Gold as an upper bound, but it is not clear why it is not a baseline. While full re-training with human re-annotation is costly, it is common to use LLM-based re-annotation to reduce cost, and the authors already use an LLM as a programmatic triage in 4.2 to judge compliance. A re-annotated dataset (Invert, Retain, and optional corrected Punish) could be trained with DPO as a baseline.
3. The result tables show the average scores across all three models without model-level breakdown. It lacks sufficient detail to know if the method generalizes well across three models or the improvement comes from a specific model. Adding per-model results to the main tables or appendix would improve clarity.
4. Some reported numbers appear inconsistent with statements:
   - The table 2 says TRACE has consistently minimal degradations, but on HellaSwag, TRACE shows clear regression (81.4 vs 78.2; 81.4 vs 77.3)
   - The table 3 says "TRACE achieves a resilience to attacks that is nearly indistinguishable from DPO-Gold", but the numbers suggest the opposite (11.3 vs 27.3; 12.8 vs 19.7).

### Questions
See weaknesses.

1. Out of curiosity, how would the proportion of Type II (Punish) samples vary with different policy shifts? I expect higher rates when the existing preference data lacks examples of the newly desired behaviors.

Minor: There are several citation format issues. For example, line 53 and 655 should be \citep.

### Soundness
1

### Presentation
2

### Contribution
2
