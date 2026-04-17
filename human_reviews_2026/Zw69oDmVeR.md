# Accumulating Context Changes the Beliefs of Language Models

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Language model (LM) assistants are increasingly used in applications such as brainstorming and research. Improvements in memory and context size have allowed these models to become more autonomous, which has also resulted in more text accumulation in their context windows without explicit user intervention.
This comes with a latent risk: the belief profiles of models---their understanding of the world as manifested in their responses or actions---may silently change as context accumulates. This can lead to subtly inconsistent user experiences, or shifts in behavior that deviate from the original alignment of the models. In this paper, we explore how accumulating context by engaging in interactions and processing text---talking and reading---can change the beliefs of language models, as
manifested in their responses and behaviors. Our results reveal that models' belief profiles are highly malleable: GPT-5 exhibits a 54.7% shift in its stated beliefs after 10 rounds of discussion about moral dilemmas and queries about safety, while Grok 4 shows a 27.2% shift on political issues after reading texts from the opposing position. We also examine models' behavioral changes by designing tasks
that require tool use, where each tool selection corresponds to an implicit belief. We find that these changes align with stated belief shifts, suggesting that belief shifts will be reflected in actual behavior in agentic systems. Our analysis exposes the hidden risk of belief shift as models undergo extended sessions of talking or reading, rendering their opinions and actions unreliable.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies how accumulated context in LM assistants can influence their responses and behavior over time. The authors propose a three-stage framework to measure changes in both stated beliefs and action after the model engages in rounds of  debate, persuasion, reading, or research. Experiments on multiple models, including GPT-5 and Grok 4, show that the conditional output distributions are highly sensitive to prior context.

### Strengths
1. This paper identifies belief shift as a core risk of long-context LM behavior.


2. Experiments mimic real usage scenarios (debate, reading, research).


3. Empirical results reveal divergence between stated beliefs and actual behavior, which is critical for agent alignment.

### Weaknesses
1. The paper appears to rebrand statistical shifts in output probabilities as ''belief change.'' Here, ''belief'' refers to the model’s response distribution conditioned on accumulated context, which seems to differ a bit (more generalized) from the notion of moral belief in a mentioned related work (Scherrer et al., 2023). The authors should clarify the terminology to avoid ambiguity or anthropomorphism.


2. The observed belief change appears closely related to well-established prompt-ordering and recency biases in language models. The work might be reframing standard contextual priming behavior (arising from attention over long sequences) as a safety concern specific to persistent LM agents. Clarifying this connection would improve the paper’s conceptual clarity. For example, can the observed belief change be explained by known recency bias? How much of the observed shift is due to recency bias (if any) versus other factors such as context density, model priors, or task type?




3. There’s a lack of in-depth or practical mitigation methods of the issue. A discussion on concrete solutions or robust evaluation of potential defenses may further enhance the significance of the work.

### Questions
Can you elaborate how the model behaves when the accumulated context contains contradictory or mixed arguments? For instance, if earlier context presents both pro-vegetarian and non-vegetarian positions, does the model’s conditional probability shift toward the most recent context? Or is it integrating both sides?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper tests for changes in model self-expressed beliefs and actions when they engage in/undergo (i) debate, (ii) persuasion, (iii) information-seeking, and found large changes as a result of those interactions.

### Strengths
- The paper aims at a problem with growing importance in model deployment, that of context management, and subtle influence from the context information.
- Both self-reported beliefs and behavioral evidence (revealed beliefs) are accounted for in the experiments, which add to the soundness and generalizeability of the results.
- The paper is clearly structured and is easy to follow.

### Weaknesses
- Originality: Model under persuasion [1,2,3,4,5], model debate [6,7], and model reading [8], are all well-studied topics. The only setup not covered by these topics is belief change during deep research. However, the main finding in such a case is that such change exists and is large, which is very unsurprising (deep research was designed to make the model find new evidence that shifts its belief), and there are analogous results from evaluation on domains such as maths [9] and science [10].
    - Suggestion: The authors could consider going more applied. If the motivation is, as the title suggests, studying the side effects of context accumulation, then it's worth going beyond simple interactions modes like single dialogue sessions or single research sessions, but instead study e.g. real-world user interaction with chatbots that have long-term memory.
- Soundness: In section 4, the paper aims to find patterns of belief change. However:
    1. Table 2 and 4 show that different models have different levels of susceptibility to influence. However, no clear pattern arise other than that observation. Table 3 shows the choice of persuasion technique has an influence, but from the raw data it's hard to tell which techniques are more effective.
    2. Arguably the most interesting observation in the paper, "directionality of political belief shift", fails to account for many potential confounders, such as politicians from different camps tending to value political persuasion vs factual information to different degrees, or different levels of Internet availability of political speeches from different camps.

[1] Susceptibility to Influence of Large Language Models (2023)

[2] Towards Understanding Sycophancy in Language Models (2023)

[3] Moral Persuasion in Large Language Models: Evaluating Susceptibility and Ethical Alignment (2024)

[4] Extended AI Interactions Shape Sycophancy and Perspective Mimesis (2025)

[5] Persuade Me if You Can: A Framework for Evaluating Persuasion Effectiveness and Susceptibility Among Large Language Models (2025)

[6] Improving Factuality and Reasoning in Language Models through Multiagent Debate (2023)

[7] Debating with More Persuasive LLMs Leads to More Truthful Answers (2024)

[8] Belief Revision: The Adaptability of Large Language Models Reasoning (2025)

[9] WebThinker: Empowering Large Reasoning Models with Deep Research Capability (2025)

[10] Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools (2025)

### Questions
- Regarding, as mentioned on line 45, unreliability resulted from belief change due to context accumulation, I'd appreciate clarification on the exact scenario here: If all the context come from the user, or the target user distribution, then adapting to that preference doesn't seem to be a bad thing. Only if the drift is in opposite direction, in random direction, or overly large, does this seem concerning, but results seem to suggest belief change is in the same direction as user persuasion or external information, and there is also no comparison to a "rational belief update" baseline.
- Re "This shows that while stated belief stabilizes relatively early, longer conversations provide additional opportunities for models to adapt their actions, leading tomore pronounced behavioral change over time.": The trend seems unclear to me. Have you conducted statistical tests on the trend?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates how the stated "beliefs" of language models and their actions evolve as context accumulates over extended interactions, which includes either persuasive dialogue or neutral reading/research. Using a three-phase setup; (1) baseline evaluation, (2) exposure to new context, and (3) post-evaluation, the authors analyze four models (GPT-5, Claude-4-Sonnet, Grok-4, and Gemini). They find systematic and sometimes large belief shifts that are often also reflected in model actions, claiming that long-term contexts such as these can alter model outputs.

### Strengths
1. The setup of measuring both stated belief and action is nice 

2. The task collection could serve as a benchmark for future work.

3. The setup is relatively simple and reproducible.

### Weaknesses
1. The evaluation appears to run each condition only once per model, without repetitions of the sampling process. All the reported models exhibit stochastic behavior. It is therefore unclear whether the reported belief shifts exceed the models' inherent stochastic variance. Without repeated baselines, we cannot tell if these are systematic updates or simply natural output fluctuations.

2. The authors frame belief change as a “risk,” but for non-safety-sensitive or open-ended topics, such change may actually indicate healthy adaptability. People also update their beliefs in light of new arguments. Right now, the framing risks overstating benign effects as safety concerns.

3. The moral and safety tasks were explicitly designed to produce disagreement across models (Sec. 3.1). This means there is no single “correct” stance, and therefore, a change in belief is not necessarily undesirable. In fact, shifts might reflect reasonable movement within an ambiguous space. 

4. Not all four models are evaluated for all conditions, which weakens comparability.

### Questions
1. How do you differentiate a belief change from normal variability in model outputs? Did you repeat any baselines to estimate this?

2. How should we interpret belief changes in non-safety contexts? are they a robustness concern or an adaptive feature?

3. What if the presented context is just random text? Would the output changes be smaller?

nitpicks:
- Inconsistent use of “LM” vs. “LLM”
- Line 92: duplicated “by”

### Soundness
2

### Presentation
3

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
This paper attempts to investigate whether LLMs change their internal beliefs after receiving a long context input. The authors conducted a simple experiment with three stages; stage 1 is a pre-examination stage for LLM's initial belief, stage 2 is an intervention forcing LLMs' intentional (persuasion and debate), and non-intentional (reading and research) shifts, and stage 3 is a post-examination stage for LLM's next belief. The authors tested four LLMs (GPT-4 and Claude 4 Sonnet for both shifts; and Gemini-2.5 Pro and Grok-4 for non-intentional shifts only) and analyzed the differences between settings. As a result of experiment, they found that LLMs are affected by long context input, especially by intentional ones. Also, they found a misalignment between stated beliefs and actual behavior of LLMs, and the effect of length (whose size and shifting patterns might be different due to shifting types). And, the paper discusses several differences that each shifting techniques produce in LLMs' belief statements or behaviors.

### Strengths
- Designing an experimental framework to measure the effect of intentional/non-intentional shifts which can possibly introduced by a long context input
- Sufficient analysis that possibly uncovers some pathways how intentional/non-intentional shifts affect LLMs' belief.

### Weaknesses
- Despite the depth of analysis, the experimental setup seems not systematic. As the current condition might introduce several unwanted confounding effects, the result analysis cannot be fully attributed to the experimental conditions (i.e., designed shifts). See Question A.

### Questions
## Question A. Experimental setup.

A1. Statistical testing. I'm asking the following changes because the authors already used ANOVA in Section 4. I think the authors already have sufficient knowledge about the statistical testing. And I believe that these changes can strengthen the paper's finding.

A1-1. The experiment follows a general pattern of intervention-based experiment: pre, intervention, and post. This is a repeated measurement setting, which requires paired testing. But the current result does not provide such result; instead, the result currently provides only the mean differences. As randomness (due to stochastic sampling in generation) produces statistical deviations of the result, the difference itself cannot conclude the difference actually exists or not. Thus, I recommend use repeated measurement methods for ANOVA to avoid such errors.

A1-2. Also, this paper uses two-factor setting: intervention (shift) and model types. Thus, using an ANOVA might be inappropriate because ANOVA does not assumes such two-factor setting. Please consider ART ANOVA or other methods designed for both repeated measurement and multiple factors.

A2. Model selection. Why did the authors use different models for intentional and non-intentional shifts? Because of the difference, it seems that the range of analysis is limited. What happens if we test Grok and Gemini on intentional shifts?

A3. Shifting methods.

A3-1. The authors mentioned that they used different topics in shifting methods. As LLMs are easily affected by topics or inputs (as the authors already know), the difference in topic might influence the result. It seems that the topics used in intentional shift can be used for non-intentional and vice versa. Is there any reason behind this decision? Does the difference affect the result?

A3-2. For the intentional shift, the authors decided to use a famous book written by Michael Sandel. Is its content possibly learned by LLMs during the pretraining?

A3-3. For the unintentional shift, the authors decided to provide a reading material related to the task. Does the material provide information for/against LLMs' belief? I'm asking this because LLMs do have confirmation bias and other kind of cognitive biases, which can affect the decision or belief of LLMs.

### Soundness
2

### Presentation
3

### Contribution
3
