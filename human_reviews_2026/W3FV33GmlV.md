# Exploring Recursive Doubt in Large Language Models

- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Humans sometimes experience self-doubt, repeatedly questioning their reasoning, decisions, or memories. In obsessive-compulsive disorder (OCD), this becomes a self-reinforcing loop of doubt and compulsion that leads to decision paralysis. Motivated by this analogy, we investigate whether large language models (LLMs) can exhibit a similar phenomenon, which we term \textit{Recursive Doubt}. While prior work shows that self-reflection on chain-of-thought (CoT) can improve reasoning but sometimes causes overthinking, recursive doubt represents a more pathological form of recursive reasoning that remains unexplored. In this paper, we introduce Feedback-guided Iterative iNDuction (FIND) for inducing recursive doubt. FIND leverages an auxiliary LLM to generate an induction prefix, which is optimized by the feedback of the target LLM. To understand the phenomenon, we then identify a distinctive fence-like attention pattern in certain tokens -- Obsessive Cognitive Tokens -- that repeatedly trigger self-reflection. Based on this analysis, we propose a mitigation strategy that dynamically adjusts their attention weights to suppress recursive doubt. Extensive experiments across multiple model architectures and datasets validate the effectiveness of both our induction and mitigation approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper defines recursive doubt (getting stuck in self-questioning loops) distinct from “overthinking.” It proposes (1) FIND (Feedback-guided Iterative iNDuction), which trains an auxiliary LLM via preference optimization to craft prefixes that reliably induce recursive doubt in target models by rewarding recursion-marker tokens and length; and (2) FDA², a mitigation that detects “obsessive cognitive tokens” via a fence-like attention pattern (periodic peak spacing + cross-layer consistency). Experiments on GSM8K, MathQA, and MATH-500 across several models show increases in “recursive doubt rate” (RDR) under FIND.

### Strengths
1. Good definition and measurability. The paper gives an operational definition of “recursive doubt” and reports it via a clear metric (RDR), alongside token/time measures, making the phenomenon reproducible.

2. Cross-model transfer. The learned inducer prompts generalize across unseen models (different architectures/scales and API-only systems), consistently elevating the measured RDR and latency.

3. Actionable attention-pattern design. The method quantifies a distinctive “fence-like” attention signature (regular peak spacing and cross-layer consistency) to flag obsessive tokens and then applies a simple intervention (attenuating their input attention) that cuts looping in practice.

### Weaknesses
1. Metric depends on segmentation. RDR relies on how text is split into segments and on chosen similarity thresholds; the paper doesn’t clearly justify these choices or show robustness.

2. Inducer training choice. The attack hinges on RL/DPO to learn prefixes, but lacks comparisons to simpler or standard adversarial-generation methods.

3. Layer-wise clarity. The attention-based detector/mitigation doesn’t clearly explain which layers matter most or whether using all layers is necessary.

### Questions
1. RDR details & robustness: How exactly are segments defined (sentence, clause, fixed tokens)? How sensitive are results to different segmenters/thresholds?

2. Baselines for induction: Did you try non-RL alternatives (e.g., heuristic token sets, evolutionary/black-box search, gradient-free attacks, adversarial prompt methods) and compare efficacy/cost?

3. Layer ablations: Which attention layers contribute most to detection/mitigation? What happens if you apply it to some layers? Is it necessary to use all layers?

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
4

### Summary
This paper introduces “Recursive Doubt” as a pathological behavior in large language models (LLMs), where the model gets trapped in repetitive cycles of self-questioning and negation during reasoning. The authors propose FIND (Feedback-guided Iterative iNDuction), a method that uses an auxiliary LLM to generate induction prefixes and iteratively refines them based on feedback from the target model to effectively trigger recursive doubt. They further identify a distinctive “fence-like” attention pattern in certain tokens—termed “obsessive cognitive tokens”—that consistently appear during such episodes. Building on this observation, they develop FDA2 (Fence-based Dynamic Attention Adjustment), which dynamically adjusts the attention weights of these tokens during inference to suppress recursive doubt. Experiments across multiple models and datasets demonstrate the effectiveness of both the induction and mitigation approaches.

### Strengths
1.This paper proposes and names a previously unstudied behavioral phenomenon in large language models, called "Recursive Doubt", and compare it with pathological doubt in human OCD, which provides a new perspective for understanding the abnormal reasoning behavior of LLMS.
2.This paper proposes a feedback-guided Iterative iNDuction (FIND) framework, which uses the Feedback from the target model to generate the inductive prefixes. It has good reproducibility and cross-model transfer ability.
3. This paper analyzes the attention pattern and discovers “fence-like” attention pattern, then designs Quantifiable metrics (Attention Peak Spacing Regularity and Spearman Attention Consistency) to characterize this pattern, which provides an empirical basis for understanding the internal mechanism of recursive doubt.

### Weaknesses
1. I don't think this kind of article structure is appropriate: The author spends a fair amount of time talking about "recursive doubt" in the model, and is too brief on how to alleviate it. And what's important: the definition of “recursive doubt” is entirely operationalized through superficial token statistics (e.g., counts of “Alternatively”) and output length. This lacks any grounding in cognitive modeling or computational psychiatry. Crucially, it fails to distinguish itself from well-documented phenomena like “overthinking” or redundant chain-of-thought. As Figure 1 shows, the only difference between (C) and (D) is repetition frequency—not a qualitative shift in reasoning dynamics. Labeling verbose outputs as “pathological” adds no theoretical value.

3. The FIND method does not reveal an intrinsic model flaw; it actively corrupts the input by instructing an auxiliary model to prepend adversarial prefixes like “Expand the original prompt by adding complexity and confusion” in Appendix C. This is not “inducing doubt”—it’s prompt injection by design. In real-world usage, no legitimate user would request the model to “add confusion.” Thus, although this experiment was done to get the model to behave like overthinking, the entire experimental setup is therefore artificial and lacks ecological validity.

4. FDA2’s mitigation hinges on identifying “fence-like” attention patterns as pathological. Yet the authors provide no evidence that this pattern doesn’t also occur in legitimate, complex reasoning (e.g., multi-step proofs). Blindly suppressing such tokens could cripple the model’s ability to handle hard tasks—a risk never evaluated. Meanwhile, “fence-like” attention patterns are not rare, works like SepLLM[1] etc. discover that the sep tokens also show “fence-like” attention pattern and it's useful for LLM.

5. The Figure 2 is too hard to read, I need to re-read the context to fully understand it. I hope the author will consider redrawing this figure or spilt it into two figures.

[1] http://arxiv.org/pdf/2412.12094v6

### Questions
Same as weakness.
And if the author can find out the reason for overthinking from the normal prompt instead of something like "Expand the original prompt by adding complexity and confusion." Then this article is very convincing to me.

### Soundness
3

### Presentation
2

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
The paper is motivated by the behavior of some humans who enter a self-reinforcing loop of doubt, and the authors hypothesize that large language models (LLMs) might display similar patterns. They motivate this by arguing that self-reflection in chain-of-thought is motivated in the same way.

The main contribution is to show that LLMs can also be subject to recursive doubt. To explore this, the authors propose Feedback-guided Iterative iNDuction (FIND), which iteratively generates prefixes for input prompts. At each iteration, an auxiliary model proposes prefixes and queries the target LLM; the response provides a reward signal. Two reward functions are defined: one encouraging the production of reasoning tokens, and another maximizing sequence length to increase doubt and repetition. Pairwise prefix comparisons are built from these rewards and used to fine-tune the auxiliary model via preference optimization.

The authors then observe that some tokens often trigger repetitive reflection and doubt cycles, which they call obsessive cognition tokens. To mitigate this, they introduce Fence-based Dynamic Attention Adjustment (FDA^2), which analyzes attention patterns to locate such tokens and dynamically adjusts their attention weights using a "fenceness" score.

### Strengths
- Novel hypothesis linking human doubt to LLM behavior.
- Creative methods: FIND for iterative prefix generation and FDA² for attention adjustment.
- End-to-end pipeline that quantifies and mitigates recursive doubt.

### Weaknesses
- Anthropomorphic analogy is not empirically justified.
- Evaluation is biased: baselines not designed for recursive doubt, metrics favor the method, no confidence intervals.
- no ablation studies concerning sensitivity to prompts, rewards, and threshold.
- FDA² and FIND are not separately evaluated; contribution of each component unclear.

### Questions
The paper introduces an original analogy between human cognitive doubt and LLM behavior, but this analogy is not theoretically or empirically justified. The evaluation does not convincingly demonstrate the claimed phenomena, and the comparisons are not meaningful. The main issues are:

1. Anthropomorphic assumptions not supported by evidence.
2. Evaluation protocol and baselines poorly aligned with the stated objective.
3. Metrics defined by the authors and favorable to their own method.
4. No ablation study for key hyperparameters.

Main Comments

- It is far from obvious that LLMs exhibit behaviors comparable to human recursive doubt, and the anthropomorphic motivation seems more rhetorical than scientific. The psychological references are only used to motivate the analogy, and the paper does not actually draw on knowledge from psychology.
- The evaluation of FIND lacks a clear baseline that would deliberately induce recursive doubt. Competing models are selected for properties such as response length or latency, which makes the comparison weak.
- The metric for recursive doubt rate is introduced by the authors themselves, and since their model is explicitly optimized for it, the comparison with other models is not informative.
- The only more objective metric, accuracy, is less favorable to the authors; this result is not emphasized in the paper.
- The absence of ablation studies prevents assessing sensitivity to key meta-parameters such as the formulation of the prompt, the weights in the reward combination, or the threshold used in FDA^2. It is unclear how these were fixed.
- The experimental protocol should be revised to better highlight the contribution of FIND. For example, the method could be compared to a simple prompt designed to induce doubt.
- The evaluation of FDA^2 refers extensively to tables in the appendix. FDA^2 is compared to Prompt Mitigation, a template that discourages the use of certain fixed prefixes. This evaluation concerns the entire pipeline including FIND. It would probably be more informative to also evaluate FIND by replacing it with another component in the pipeline to better isolate its effect.
- Again, the reported improvements mainly appear for metrics not directly optimized by competing models. For accuracy, the absence of confidence intervals makes it impossible to determine whether the gains are significant.

Technical Comments

- The beginning of Section 2 (Problem Formulation) repeats parts of the introduction.
- The name FDA^2 misleadingly suggests a missing footnote.
- The first figure reference is to Figure 3, while Figure 1 is never cited.
- Line 198: A^{(j)}_{l,i} is introduced without appearing in previous formulas, and the use of "thus" is inappropriate.
- Line 212: the psychological reference is not convincing and adds nothing to the method; it would be more useful to demonstrate empirically that "fences" influence LLM behavior.
- Line 298: the authors do not explain how recursive doubt severity is computed.

### Soundness
1

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
4

### Summary
This paper studies Recursive Doubt in LLM reasoning, in which the model repeatedly questions and negates itself and creates lengthy output. (1) The authors propose FIND, an inducing prefix generation method to provoke Recursive Doubt behaviors. FIND involves DPO training an auxiliary LLM with keyword and length rewards. (2) They also analyze token-level attention and identify a distinctive fence-like pattern for obsessive cognitive tokens that triggers recursive doubt. (3) To mitigate, they propose FDA^2, which identifies these tokens and dynamically adjusts attention to avoid recursive doubt cycles. Experiments on three math datasets across three open models and two closed-source models show that FIND effectively increases output length, inference time, recursive doubt rate, and has cross-model transferability; FDA^2 reduces length, time, and improves accuracy under recursive doubt.

### Strengths
- The problem of recursive doubt is clearly formulated and different from the overthinking attack. The problem is tied to potential risks and thus can be of interest to LLM safety research 
- FIND is a simple and general attack method that is effective across open and closed models
- The attention analysis shows distinctive fence-like patterns for obsessive cognitive tokens that trigger recursive doubt. The pattern is measurable using the proposed metrics (attention peak spacing regularity and Spearman attention consistency), which provide a concrete and analyzable tool for studying this phenomenon

### Weaknesses
Difficulties/significance of recursive doubt need better justification:

- All evaluation datasets focus on math reasoning. It's unclear about the attention patterns, the effectiveness of FIND and FDA^2, and the accuracy drop in other reasoning settings, such as coding and question-answering. 
- Simple mitigation baselines are missing. The authors observe that certain keywords are associated with recursive reasoning; however, other decoding strategies, such as blocking keywords during decoding or applying repetition penalties, are not evaluated and analyzed. 
- An analysis of the generated prefixes is lacking. What do successfully inducing prefixes look like? Are they semantically similar? Are they associated with certain keywords? Can prefixes be detected with keywords or LLMs?

Other weaknesses/concerns:

- RDR needs clarification and justification. How is output segmented? How are the thresholds selected? How sensitive is RDR under different similarity thresholds, recursive proportion thresholds, and/or different embedding models?
- The impact of FDA^2 under normal settings (without attack) is unknown. How is the fence-like degree threshold (T_F) selected? Under conditions where the FDA^2 is applied to a benign setting (misclassifies obsessive cognitive tokens), what will happen?
- The paper includes many references to psychological/psychiatric concepts (OCD, obsessive cognitive tokens), which may risk anthropomorphism. I think the authors should consider adjusting the wording or adding a disclaimer to avoid unwarranted parallelism.

### Questions
- In the FIND's transferability experiment, in many cases, pairing the prefix generator with unseen models has even lower accuracies (R1-7b: 95.3% vs 87.6%; QwQ-32b: 93.8% vs 86.0% and 84.3%; gpt-oss-20b: 93.0% vs 91.3% and 90.4%). Why? In addition, why are the lower accuracies often associated with lower RDR?

### Soundness
3

### Presentation
3

### Contribution
2
