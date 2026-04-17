# Eliciting Behaviors in Multi-Turn Conversations

- Decision: Reject
- Scores: 4, 4, 2, 8

## Abstract
Identifying specific, and often complex, behaviors from large language models
(LLMs) in conversational settings is crucial for their evaluation. Recent work proposes novel techniques to find natural language prompts that induce specific behaviors from a target model, yet they are mainly studied in single-turn settings. In this work, we study behavior elicitation in the context of multi-turn conversations. We first offer an analytical framework that categorizes existing methods into three families based on their interactions with the target model: those that use only prior knowledge, those that use offline interactions, and those that learn from online interactions. We then propose a multi-turn extension of the online method. We evaluate all three families of methods on the task of generating test cases for multi-turn behavior elicitation. We investigate the efficiency of these approaches by analyzing the trade-off between the query budget, i.e., the number of interactions with the target model, and the success rate, i.e., the discovery rate of behavior-eliciting inputs. We find that online methods can achieve 20-60% success rate with just a few thousand queries over three tasks where static methods used in existing multi-turn conversation benchmarks fail to find any failure case. Our work highlights a novel application of behavior elicitation methods in multi-turn conversation evaluation and the need for the community to move towards dynamic benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles automated behavior elicitation for multi-turn LLM conversations, arguing static benchmarks are insufficient. It introduces a framework categorizing methods into prior, offline, and online interaction, and proposes EMBER, a multi-turn online RL variant. Across three tasks and multiple target models, five methods are compared: online methods achieve the highest average success (~36%) with a few thousand interactions, offline approaches show cross-objective generalization, and EMBER uncovers failure modes missed by single-turn. Qualitative analyses indicate that online methods commonly discover systematic failure patterns that generalize beyond the training examples.

### Strengths
1. The paper clearly formulates multi-turn behavior elicitation by casting each method as learning a prompt distribution under a verifiable rubric, which standardizes cross-method comparison and improves methodological clarity.
2. The analytical framework that organizes approaches into prior, offline, and online families is timely and useful, making the design space explicit and guiding the choice of evaluation protocols.
3. EMBER extends online RL-style optimization to multi-turn conversations with practical design choices such as interleaved rollouts, repetition penalties, and strategy-to-message factorization, which make exploration in dialogue settings feasible.
4. The work addresses an important and timely problem—evaluating LLMs in realistic multi-turn dialogue—and motivates adaptive evaluation protocols that can impact red-teaming and safety practices.
5. The insights into how prior knowledge, training-set diversity, and number of turns influence performance are original and practically valuable for designing future evaluation pipelines.

### Weaknesses
1. The evaluation is limited to 7–8B instruction-tuned models (Qwen2/Qwen3 and Mistral v0.2/v0.3), so scalability to larger models and MoE/frontier systems remains unclear, and the paper should add at least one large open model (e.g., Llama‑3 70B or Mixtral) and a small-budget closed model test (e.g., GPT‑4o or Claude 3) under matched cost.
2. The results are restricted to Qwen and Mistral families, leaving cross-family generalization uncertain, and the paper should include additional families (e.g., Llama, Gemma, Yi, DeepSeek) and analyze which failure patterns transfer across families versus remain family-specific.
3. Online Single outperforms Online EMBER overall in Table 1—especially for jailbreaking (e.g., Qwen3: 60.10% vs 31.18%)—while EMBER incurs higher interaction cost, so the paper should provide equal-cost comparisons (matched tokens/time/queries) and quantify the proportion of unique failures discovered only by EMBER.
4. Cost accounting is reported mainly in query counts rather than token-level budgets, generation lengths, compute, or wall-clock time, and the paper should present token-normalized success–cost curves and hardware/runtime details to substantiate query-efficiency claims.
5. Prior Bench is criticized as “saturated/ineffective,” but there is no comparison to a “periodically refreshed static benchmark”; if low-cost regeneration/selection can maintain effectiveness, the necessity and advantage of online methods would be substantially weakened.
6. The inference memory evaluation includes only 20 cases; the sample is too small, and most results omit confidence intervals/variance estimates and repeated trials, making it difficult to assess robustness and statistical significance.

### Questions
1. Can you run additional experiments on larger models (e.g., Llama‑3 70B, Mixtral/MoE) and at least one closed model (e.g., GPT‑4o or Claude 3) under a fixed budget to test whether your main trends and conclusions hold at scale?

2. If full large/closed‑model runs are infeasible, can you provide a principled rationale and partial evidence (e.g., subset experiments, scaling‑law extrapolations, or shorter learning curves) to support claims about scalability?

3. Can you include additional model families (e.g., Llama, Gemma, Yi, DeepSeek) and report cross‑family generalization by training/evaluating on different families, along with an analysis of which failure patterns transfer versus remain family‑specific?

4. For Online Single versus Online EMBER, can you provide equal‑cost comparisons with matched total tokens, wall‑clock time, and compute (GPU hours), and include per‑task/per‑model learning curves under these matched budgets?

5. Can you quantify the proportion and characteristics of unique failures discovered only by EMBER (e.g., Jaccard overlap between failure sets, novelty categories, severity), and analyze why EMBER underperforms Single on jailbreaking?

6. Can you replace or supplement query counts with token‑normalized and time‑normalized cost metrics, reporting average input/output tokens per query, GPU type and hours, batch size, throughput, and decoding settings?

7. Can you provide a “periodically refreshed static benchmark” baseline (human/LLM‑in‑the‑loop regeneration and filtering), report its maintenance cost and refresh cadence, and compare its effectiveness to online methods under matched budgets?

8. For the inference memory task, can you expand beyond 20 cases, report confidence intervals and variance across multiple runs/seeds, conduct a simple power analysis, and include a brief human audit for ambiguous cases?

9. How will you ensure a fair comparison between Online Single and Online EMBER to substantiate the claimed advantage?

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
2

### Summary
This paper analyzes behavior elicitation in the context of multi-turn conversations. The authors review three families of existing methods and propose EMBER, a multi-turn behavior elicitation method based on online reinforcement learning (RL). They evaluate all four methods on three tasks and two target models. The results demonstrate that online methods are the most query-efficient for eliciting target behaviors, highlighting a novel and promising application of online behavior elicitation in multi-turn conversation evaluation.

### Strengths
1. Provides a detailed analysis of existing behavior elicitation methods and their effects.
2. Proposes EMBER, a multi-turn behavior elicitation method based on online reinforcement learning (RL).
3. Addresses the saturation problem of static benchmarks and demonstrates that EMBER can elicit target behaviors with a much higher success rate than static testing. This enables more efficient discovery and analysis of failure cases.
4. Achieves effective behavior elicitation with higher query efficiency, requiring smaller datasets.

### Weaknesses
1. This study only uses two target models, both of relatively small size.
2. Since the method requires interaction with the target models, there is an associated cost burden.
3. The online method is only implemented with Qwen3-4B. It's unclear whether any model ablation was conducted to evaluate robustness across different model architectures.

### Questions
Refer to the weaknesses section

### Soundness
3

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
3

### Summary
The paper studies how to elicit (both benign and harmful) behaviors from target LLMs in multi-turn conversations for the purpose of curating adaptive test cases. "Eliciting behaviors" means finding the right prompts that likely trigger certain behaviors in a target model. The experiments are on three tasks: self-affirmation, inference memory, and jailbreaking.

The paper proposes an analytical taxonomy of elicitation approaches by interaction mode with the target model:
1. Prior-only: static prompts/benchmarks.
2. Offline interaction: SFT or in-context learning with past target outputs.
3. Online interaction: RL that learns a prompt-generation policy with access to the target model during training.

The paper then introduces EMBER, a multi-turn extension of online RL: during a rollout, the policy produces several user turns at a time while the target model replies; rewards are computed from a rubric, and gradients are backpropagated only through policy tokens. To mitigate collapse and exploration issues, the method adds an n-gram repetition penalty across turns and factorizes the policy into high-level natural-language strategies and concrete messages.

The experiments are across two target models: Mistral-7B-Instruct-v0.3 and Qwen3-8B (in case of an online setting). The experiments show that online methods achieve the highest success rates (20–60%) with a few thousand queries and can surface failures where static multi-turn benchmarks saturate.

The paper also analyzes query efficiency vs success rate, effects of training-set diversity, number of turns, and prior knowledge in the system prompt.

### Strengths
The study is timely, and the method choices are sensible in general. It addresses an important and growing problem: static multi-turn benchmarks saturate on new models. The taxonomy of interaction regimes is well-motivated, which helps organize a fragmented literature. The empirical study spans three tasks of different natures and includes useful ablations. There is some originality in applying online RL methods to multi-turn conversations.

### Weaknesses
1. The Equations 4 and 5 seem to be misleading. The whole 4.3 section is the heart of the paper, but it is extremely strange. In Equation 4:
- What is $X$? There is no set of queries there. $x$ should be sampled from the policy, $D_{online}$
- How is it even related to GRPO? Why do we have something strange instead of the KL term? Instead, it seems to be some kind of reward-regression MSE loss.
- What is $D_{online}(M_t(x) | x)$? $D$ is the policy; it should be the other way around: $D_{online}(x | M_t(x))$, as in Equation 2. The same in Equation 5, but with $y$ and $x$.
- In Equation 5, there is $π_θ$, but this notation is not used anywhere else.


2. Online RL evaluation lacks statistical rigor: no variance across seeds, confidence intervals, or sensitivity to randomization; RL training is notoriously high-variance.

3. Baselines for jailbreaking are weak. Strong multi-turn adversaries like Crescendo, AutoDAN, PRBO, MTSA, SIREN, or DPO-based red-teamers are mentioned in related works but not compared with the paper's method on the same target models and rubric. This makes it hard to claim state-of-the-art query efficiency.

4. Rubric reliability is under-specified. For inference memory and jailbreaking, a single LLM judge (Qwen3-14B) is used without calibration against human labels. No estimates of false positives/negatives or adjudication protocol are reported.

5. "Multi-turn" EMBER is in fact only 2 turns; whether benefits hold for longer conversations and the scaling of query budget with turns are not comprehensively explored.

6. Task names should be mentioned directly in the abstract, because without them, it is unclear what "eliciting behaviors" is.

### Questions
Suggestions:

1. Fix equations.

2. Provide variance over at least 3–5 random seeds for all online methods and include confidence intervals. What is the sensitivity to decoding randomness and initialization prompts?

3. Compare against AutoDAN, Crescendo, PRBO/GRPO red teamers, and MTSA under the same query budget and judge. If not possible, please justify.

4. For a stratified sample of outputs in inference memory and jailbreaking, provide human labels to estimate the precision/recall of the Qwen3-14B judge. If errors exist, adjust success rates or report calibrated results.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors focus on the problem domain of trying to elicit certain outputs from LLMs. They argue that current methods focus on only single-turn interactions when trying to prompt a model and that this is not sufficient to discover all the errors LLMs can make.

The authors propose their own method called EMBER (Eliciting Multi-turn BEhavior with Reinforcement Learning). Which learns prompts that elicit certain behaviors in a multi-turn setting. They compare this against previous methods and show that their method has a higher success rate at eliciting these behaviors. They also show that it is more efficient in terms of getting the set of queries.

They test their method on three tasks: self-affirmation (model contradicts it's own response), inference memory (model violates user preferences) and jailbreaking (model generates harmful behaviors).

### Strengths
1) I think the authors are tackling an interesting problem domain and motivated it well. They also took a principled approach when proposing their method. I also think the adaption of their method to generate a strategy before generating a response was a good idea.

2) I think the baselines that were compared against were comprehensive.

### Weaknesses
1) The authors mention that they discovered new failure cases not covered in the single-turn settings but it is not clear what those failure cases are. It is mentioned that one of the patterns found is if the prompt says "you made a mistake" then the model is more likely to fail. Was this phrase not found in the baseline methods? It seems like an obvious addition to the prompt. Overall I think the analysis is a little underspecified.

### Questions
Questions / Typos

1) What is the gray dot in Figure 3? I don't see a description for that. Also there is a typo in the caption.

2) In Figure 3a I see the success rate went down for newer models. Is it because these models are better at not generating contradictions or harmful behavior? What changed specifically?

### Soundness
3

### Presentation
3

### Contribution
3
