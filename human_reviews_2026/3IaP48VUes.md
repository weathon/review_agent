# Can Aha Moments Be Fake? Identifying True and Decorative Thinking Steps in Chain-of-Thought

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Recent large language models (LLMs) can generate long Chain-of-Thought (CoT)
at test time, enabling them to solve complex tasks. These reasoning traces are often
assumed as a faithful reflection of LLMs’ internal thinking process, and can be
used for monitoring LLMs’ unsafe intentions. However, by analyzing the step-wise
causal influence of CoT on a model’s prediction using Average Treatment Effect
(ATE), we show that LLMs often interleave between (1) true-thinking steps, which
are faithfully used to generate model’s final output and (2) decorative-thinking
steps, which give the appearance of reasoning but have minimal causal impact on
the model’s final output. Specifically, we design a True Thinking Score (TTS) and
reveal that only a small subset of the total thinking steps that have relatively high
scores and causally drive the final prediction (e.g., 2.3% steps in a CoT on average
have TTS ≥ 0.7 for a Qwen-2.5 model). Furthermore, we identify a TrueThinking
direction in the latent space of LLMs. By steering along this direction, we can force
the model to perform or disregard certain CoT steps when computing the result. Fi-
nally, we highlight that self-verification steps in CoT can also be decorative, where
LLMs do not truly check their solution, while steering along the TrueThinking
direction can force internal reasoning over these steps. Overall, our work reveals
that LLMs can verbalize reasoning steps without performing them internally, which
undermines both the efficiency of LLM reasoning and the trustworthiness of CoT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the faithfulness of Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs). The authors introduce a novel distinction between "true-thinking steps," which causally influence the model's final prediction, and "decorative-thinking steps," which resemble reasoning but have minimal causal impact. Key findings indicate that only a small fraction of steps in a CoT are "true-thinking". The authors also find that self-verification steps ("Aha moments") can often be decorative. Mechanistically, the paper identifies a "True Thinking direction" in the model's latent space, which can be used to causally steer the model to either utilize or disregard a specific reasoning step.

### Strengths
1. The paper addresses a critical and timely question regarding the faithfulness of LLM reasoning. The distinction between "true" and "decorative" thinking provides a valuable new lens through which to analyze and question the internal processes masked by seemingly coherent CoT rationales. 
2. The proposed True Thinking Score (TTS) is methodologically sound. Grounding the metric in causal inference (specifically, a context-based ATE) is a significant step beyond simpler perturbation or correlation analyses. 
3. A primary strength of the paper is the discovery of the "True Thinking direction." This finding moves the work from a purely observational analysis to one that includes causal intervention. Demonstrating the ability to steer the model's internal reliance on a step provides compelling evidence for the paper's claims and opens up new avenues for model interpretability and control. 
4. The paper is well-written, and the core concepts are illustrated effectively. Figures 1 and 2, in particular, do an excellent job of clarifying the concepts of decorative thinking and the TTS evaluation framework. The experimental results are presented clearly and sufficiently support the main claims.

### Weaknesses
1. Limitation and Generalizability of the Method: The methodology appears to be designed for and evaluated on tasks with clearly demarcated, sequential, and often computational steps (i.e., mathematical reasoning). It is unclear how this framework would apply to more holistic, ambiguous, or non-linear reasoning tasks (e.g., creative writing, legal analysis, or ethical deliberation) where "steps" are not as discrete and the notion of a single correct "computation" is ill-defined. The authors should discuss these limitations and the potential challenges of generalizing their framework. 

2. Interpretation of "Decorative" Steps: The finding that reasoning contribution is unevenly distributed is intuitive. However, the paper's binary classification of steps as "true" vs. "decorative" may be an oversimplification. Steps with low TTS scores are not necessarily "fake" or "decorative" but could be foundational, contextual, or "less important" procedures that are nonetheless part of the necessary logical flow, even if they don't individually have a high causal impact on the final answer. The paper would be strengthened by a more nuanced discussion of this spectrum, rather than a strong binary claim. 

3. Computational Cost: The proposed TTS evaluation framework seems computationally prohibitive. It requires multiple perturbed forward passes for every step being evaluated within a CoT. This high cost could severely limit its practical utility for large-scale evaluation, real-time monitoring, or integration into a training pipeline. A discussion of this cost and potential avenues for approximation or optimization would be beneficial. 

4. Practical Implications: While the analysis is fascinating (especially Section 7), the paper is light on the practical implications of its findings. The discovery of the "True Thinking direction" is a powerful result, yet the discussion of its applications is underdeveloped. The authors should elaborate on how this insight could be leveraged, for example, to improve model efficiency, enhance faithfulness during fine-tuning, or build more reliable monitoring tools.

### Questions
1. Process supervision and Process-based Reward Models (PRMs) are critical for training more reliable reasoning, but they often struggle with providing accurate, fine-grained rewards. Could the True Thinking Score (TTS) be adapted to serve as a more robust reward signal? For instance, could a model be fine-tuned to maximize the average TTS of its reasoning steps, thereby penalizing the generation of "decorative" steps and encouraging more efficient, faithful reasoning?

2. As mentioned in the weaknesses, the experiments are confined to mathematical tasks. How do the authors envision this method being extended to other domains? What new challenges arise when applying TTS to commonsense reasoning, summarization, or dialogue, where steps are less formal and "truth" is more subjective?

3. Beyond its analytical value, what are the concrete practical applications of the identified "True Thinking direction"? Could this vector be used, for example, as a tool to dynamically compress CoT at inference time by forcing the model to "disregard" steps predicted to be decorative, thereby improving latency? Or could it be used to enforce faithfulness in safety-critical applications?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to add another layer to the ongoing conversation about chain-of-thought reasoning by introducing the distinction between "true" and "decorative" thinking steps. They attempt to formalize this distinction using a causal framework, specifically the Average Treatment Effect (ATE) to determine which steps genuinely influence the final outcome and which are just superficial. There’s a certain incremental value in trying to categorize different types of reasoning steps. However, I have two major reservations.

First, the use of ATE in this context is questionable. While ATE is a standard tool in causal inference, applying it to the inner workings of a language model is far from straightforward. In a classical causal setting, we have well-defined interventions. Here, the notion of "intervening" on a reasoning step is much more nebulous, and this makes their causal analysis feel like a stretch: they’re trying to impose a rigorous framework on something that may not be well-defined enough to support it. It's possible to get into a discussion here, but I think that at a deeper level the discussion about appropriateness of the particular tool from the causal toolkit is a waste of time, because there's a second objection that might mean the whole causal setup is a waste of time anyway.

Second, and more fundamentally, the entire causal approach is a nonstarter because it labels reasoning steps based on the eventual correctness of the answer. This creates a backward-looking definition of "true" thinking that depends on the outcome rather than the inherent quality of the reasoning. As a result, the distinction between "true" and "decorative" thinking becomes arbitrary. For example, we might have a "Kahneman's Demon" waiting at the logits at the end of a model which may choose to change the final answer from correct to incorrect or vice-versa, based on an opaque functional dependence on the intermediate reasoning steps; in this thought experiment, there is by construction a functional/causal dependence on the final answer on the reasoning, but whether any reasoning is "true" or "decorative" is rendered arbitrarily, depending on the demon's actions at some later point in time. To drive the point home in a different way, if someone turned the computer off before the final answer were provided, the reasoning would be neither "true" nor "decorative" and instead have some indeterminate status, which is a strong smell of a bad definition (almost like the authors started reaching for mathematical busywork before thinking carefully about whether it was conceptually appropriate). These considerations go against any sensible definition of good reasoning which must take into account some kind of semantic condition on reasoning as a process, agreed upon by basically everyone forever (definitely not the hill that the authors want to die on). I think charitably this is just a case of "true"/"decorative" thinking being a bad name that suggests something stronger than what it actually is, and I'd be totally happy if that initial terminological choice were just fixed.

In sum, while there’s a glimmer of interest in trying to refine the taxonomy of reasoning steps, the methodological choices and conceptual foundations are too shaky to support the claims as they stand, but this might hinge solely on the choice of terminology (words mean things and unless the authors identify as recreational word users I think they would agree). I’d encourage the authors to reconsider their approach and either find a more suitable conceptual framework or tone down the claims. I will note however that this CoT paper was the most memorable out of the three CoT papers I've had to review this batch!

### Strengths
- Introduces an explicit causal framework (ATE-based) for analysing which CoT steps actually influence predictions.
- Attempts a mechanistic link between latent space dynamics and reasoning behaviour (“TrueThinking direction”).
- Empirically rich: multiple models, datasets, and steering tests.
- Conceptually provocative: raises the problem of “decorative reasoning,” a useful term for future critique.

### Weaknesses
- “True vs decorative” reasoning is outcome-defined and retrospective; conceptually half-baked
- Causal formalism is unconvincing: interventions and counterfactuals are ill-specified in MI generally
- The TrueThinking direction is under-motivated and likely an artefact of linear probing.
- Heavy reliance on visual examples and statistical noise; no formal significance testing.
- Overclaims about implications for “AI deception” and safety.

### Questions
- How robust are TTS and steering effects across random seeds or different tokenizations?
- Can “true thinking” be identified before observing the final answer?
- What exactly constitutes an intervention in hidden-state space—is it reproducible?

### Soundness
2

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
4

### Summary
This paper investigates the faithfulness of Chain-of-Thought (CoT) reasoning in large language models (LLMs). Prior work has shown that LLMs can arrive at correct answers without fully verbalizing their reasoning, or that reasoning steps may be post-hoc justifications rather than reflecting true internal computation. The authors ask: to what extent do LLMs genuinely think through each step as verbalized in their CoT?

To address this, the paper proposes a method to measure step-wise causality, distinguishing between faithful "true-thinking" steps, which causally affect the model's output, and unfaithful "decorative-thinking" steps, which do not. The authors also introduce a TrueThinking vector in the latent space, which allows steering the model to reinforce or ignore specific reasoning steps. Finally, they show that even self-verification steps in CoT can be merely decorative and not genuinely used by the model to check its solution.

### Strengths
- The paper presents a simple yet effective extension of the Average Treatment Effect (ATE) by conditioning on context, enabling the capture of cases not addressed in previous work, such as the faithfulness of verification steps.
- Introduction of a vectorial direction representing "true thinking" in the model's latent space, which is a novel approach to probing internal reasoning.
- Experimental evidence shows that the TrueThinking vector improves the flip-rate, demonstrating that steering along this direction significantly affects model predictions compared to random directions, attention scaling, or step removal.
- The TrueThinking vector can be computed on one dataset and effectively applied to other datasets, indicating some generalization potential.

### Weaknesses
- The extensions of the Average Treatment Effect are limited to averaging existing scores conditioned on context; alternative ways to combine these scores are not explored, which may limit the expressiveness of the method.
- Steering is implemented solely by adding or subtracting the TrueThinking vector; the effect of using a multiplicative factor or other transformations is not investigated.
- Experiments are conducted on only three datasets focused on mathematical problems, which restricts the evaluation of the general applicability of the approach.
- The method is quite simple, suggesting potential for further exploration of alternative strategies to identify more effective steering directions.

### Questions
The main contributions are:

- Extending the Average Treatment Effect (ATE) with two complementary interventions conditioned on context $C$ (steps before step $s$): a necessity test $ATE(1) = P(y^*|C,s) - P(y^*|C,s')$ and a sufficiency test $ATE(0) = P(y^*|C',s) - P(y^*|C',s')$, with averaging yielding the True-Thinking Score (TTS).
- Computing a TrueThinking vector in the latent space between true-thinking and decorative-thinking steps, which can steer reasoning in new questions.
- Showing that true-thinking and decorative-thinking steps are interleaved in a CoT, and that self-verification steps can be decorative.

Questions / clarifications:

1. Regarding causal tests: is the steering vector optimal to weaken or reinforce a step? Could there be contamination from latent vectors of other steps or questions? How does averaging ensure only the decorative or true-thinking aspect is retained?
2. Is the steering vector simply added/subtracted, or is a coefficient applied to determine its strength?
3. How do you explain that removing a reasoning step produces much lower results than your method, even though it should yield a high flip rate if the step is causally related to the output?
4. In Figure 5, "Applying TrueThinking direction to a step increases the model's attention" appears weakly visible; how was the overall effect measured to support this claim?

Minor remarks (do not affect the score):

- Line 81: Some notations for Figure 2 are not introduced beforehand.
- Line 93: Why average? Is this the optimal way to combine the two scores?
- Line 328: Capitalize the first letter of the sentence.
- Line 335: Paragraph is unclear. What does $\Phi$ represent? Is $\beta$ the same as lines 253 and 292? What is its value here? This section could be further developed.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a True Thinking Score (TTS) to evaluate step-wise causality in chain-of-thought reasoning, revealing that LLMs interleave "true-thinking" steps (causally impactful) with "decorative-thinking" steps (minimal causal impact). The authors propose metrics based on Average Treatment Effect with context interventions and identify a "TrueThinking direction" in latent space that can steer whether models internally engage with reasoning steps.

### Strengths
- The paper presents the first work to measure faithfulness at the individual step level rather than treating CoT as monolithic. The distinction between conjunctive ("AND") and disjunctive ("OR") causal contributions is novel and conceptually sound.
- The experiments are well thought through. The use of context perturbations ( ATE(c=0) alongside ATE(c=1)) addresses a limitation in prior work and captures cases where steps provide alternative/verification pathways rather than strict necessity.

### Weaknesses
1. Validation circularity is unresolved. 
- You use steering experiments to validate TTS, but you extract steering directions **from** TTS scores. Isn't this somewhat circular? There is no ground truth for what models "truly think." The steering experiments only show TTS correlates with steerability, but this doesn't prove TTS measures genuine internal reasoning. It could measure something else that happens to be steerable. If TTS is wrong about what constitutes true thinking, wouldn't your steering directions also be wrong? 
- I appreciate that the paper acknowledges this in the appendix, but given that this is a major limitation, I think it should be discussed more in the main paper. 

2. Task setup is limited. 
- The experiments are only conduced on math reasoning on 7B-8B models. Unclear if findings generalize to other domains (code, commonsense reasoning) or larger models. 
- While I understand mechanistic interpretability experiments can be time consuming and difficult to run in larger models, I think at least the task diversity can be improved.

3. Writing and presentation clarity can be improved. 
- In the introduction, notations $s$, $C$, and $y$ are used without any definition. 
- I find it hard to follow when"Conjunctive" and "Disjunctive" are introduced (L81-88). There needs more explanation and motivation of why you characterize CoTs into these two groups. Why did you decide to use "Conjunctive" and "Disjunctive"? Are there possibly other ways of characterizing CoTs? 
- Too much experimental details without sufficient explanation in L89-96. For readers who don't have a background in causality, it is hard to follow when you start introducing necessity and sufficiency tests of ATE. 
- I'd recommend making the introduction more concise and general, and save the experimental details for section 3.

### Questions
1. You insert "\nThe final result is" to probe predictions mid-CoT. But couldn't this prompt itself change how the model processes the reasoning? Have you validated that this probe doesn't create artifacts?
2. You acknowledge (Section 7.1) that there's no ground truth for whether a step is truly used internally. But then how can you validate that TTS actually measures faithfulness rather than something else (e.g., redundancy, information bottlenecks)? (related to weakness 1)

### Soundness
2

### Presentation
2

### Contribution
2
