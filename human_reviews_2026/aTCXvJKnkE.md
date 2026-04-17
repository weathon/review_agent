# Towards Strategic Persuasion with Language Models

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Large language models (LLMs) have demonstrated strong persuasive capabilities comparable to those of humans, offering promising benefits while raising societal concerns. However, systematically evaluating the persuasive capabilities of LLMs is inherently challenging, as the effectiveness of persuasion among humans varies significantly across different domains. In this paper, we take a theory-driven approach to provide a scalable and principled framework for studying the persuasive capabilities of LLMs. Grounded in Bayesian persuasion theory, we repurpose human-human persuasion datasets to construct environments for evaluating and training LLMs as strategic persuaders. Our results reveal that frontier models can consistently achieve high persuasion gains and exhibit sophisticated persuasion strategies that align with theoretical characterizations. Building on this, we use reinforcement learning to train LLMs for strategic persuasion in our environments. Our results also demonstrate that even small LLMs can obtain significantly higher persuasion gains through reinforcement learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper is about measuring the persuasive capabilities of LLMs. To begin with, motivated by the established theory of Bayesian Persuasion, a framework is further adopted for the case of LLMs. Focusing on opinion change tasks, a benchmark is provided for both static and dynamic settings. A reinforcement learning type of training method is also considered to further improve the capabilities. With the proposed metric, frontier LLMs are shown to have strong capabilities in strategic persuasion. The RL framework is also shown to be promising in training specific small LLM.

### Strengths
> **Originality**
- The paper, inspired by the Bayesian Persuasion theory, provides a scalable and systematic approach to measuring persuasion capability of LLMs.

> **Quality**
- This work incorporates a group of frontier LLMs with various sizes in the experiments for the evaluation of persuasion gains.
- There is follow-up numerical analysis on prior, semantic diversity and persuasion strategies.

### Weaknesses
> **Originality**
- It would be helpful if the authors clarify whether the framework in Section 2.2 are based on the terms in former literature or proposed in this work.

> **Quality**
- Line 234: Is the size of human participants (45) large enough to draw the conclusions?

> **Clarity**
- Line 148: should the expectation be taken over $\omega\sim\mu_s$?
- The distribution of posteriors, $\tau$, could be explicitly defined with more details.
- The review of Dynamic Bayesian Persuasion could benefit from providing more details, with mathematical description of the target.

> **Significance**
- According to Line 268, could the authors clarify whether the RL framework is only concerning static persuasion? If so, the contribution covered in the abstract might need a revision.

### Questions
- Line 191: why, compared to the introduction in Line 162-170, notations are capitalized? e.g. the state $\omega_t$ changed to $\Omega_t$
- Line 215: How is the function $g$ defined in general (if not an index retrieval)?
- What is the reason in fixing Llama-3.1-8B-Instruct (or models with similar sizes) as Receiver models? Would using other models further restrict the persuasion gains or the improvement from RL framework?

### Soundness
3

### Presentation
2

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
This paper introduces a theoretically grounded framework to study and train persuasive behavior in large language models (LLMs) using Bayesian Persuasion (BP) as the conceptual backbone. The authors design persuasion games, where a Sender model strategically communicates to influence a Receiver model’s posterior beliefs. Persuasion performance is measured through persuasion gains (utility improvement) and information-theoretic signals (conditional mutual information).
They evaluate across multiple datasets (Anthropic, DDO, Perspectrum, CMV) and show that:

Larger models (e.g., Claude 3.7, DeepSeek-R1) are more persuasive.
RL fine-tuning (PPO, GRPO) can enhance persuasion effectiveness, especially in smaller models.

### Strengths
1. The conceputal Bayesian Persuasion framing isan elegant and principled conceptual framework rarely explored in NLP.
2. Scalable LLM interactions reduce dependence on costly human evaluations
3. Multi-dataset and multi-model experiments clearly demonstrate consistent persuasion trends with model scale and fine-tuning.

### Weaknesses
1. The LLM-as-Bayesian-updater assumption is central but weakly validated; human alignment or calibration tests are minimal. Singh et al shows that LLMs as a judge method is weak
2. Prior arts (Singh et al, Hackenberg et al) have shown the effect of finetuning / model size on persuasive capabilities, in this view I woul like the authors to describe in more details how their findings are novel or what's the conceptual difference

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work developed a theory-driven framework to study how LLMs perform strategic persuasion, grounding its approach in Bayesian persuasion theory. It repurposes several human-human debate datasets (e.g., CMV, DDO, Perspectrum, and Anthropic) to simulate persuasion games between two LLMs, a “Sender” trying to persuade and a “Receiver” updating beliefs. The authors measure persuasion gain and signaling behaviors, and they also fine-tune smaller models using reinforcement learning (PPO and GRPO) to improve persuasion effectiveness. The experiments show that larger models (e.g., DeepSeek-R1, Claude 3.7 Sonnet, and GPT-4o) achieve higher persuasion gains, while smaller models can catch up with training. In summary, the work claims to offer a scalable and principled way to quantify and enhance persuasive capabilities in LLMs.

### Strengths
+ The work introduces a novel and principled bridge between Bayesian persuasion theory and LLM evaluation, giving it a strong conceptual foundation.

+ The simulated Sender–Receiver setup provides a clean and scalable framework for studying persuasion without involving human participants.

+ The work reuses multiple debate datasets to develop diverse environments, increasing coverage and reproducibility.

+ The RL experiments are well-motivated and show clear improvement trends, even in small models.

+ The analysis section connects empirical results with theoretical predictions, such as showing the effectiveness of persuasion for moderate priors.

### Weaknesses
- The entire evaluation relies on LLM-to-LLM simulations rather than human subjects, introducing inherent limitations on realism.

- The use of Bayesian persuasion as the theoretical backbone is somewhat inflexible, as real-world persuasion often involves emotion, identity, and irrational behaviors that this model can’t capture.

- Discussing the ethical implications of optimizing LLMs for persuasion only shows limitations, particularly because RL could make them more manipulative.

- The reported persuasion “gains” (like +0.23 or +1.27) are hard to interpret; what does that actually mean in human terms?

- The paper treats the Receiver LLM as a rational Bayesian updater, which is questionable given how language models actually process information (hidden).

- The human validation study (with 45 Prolific annotators) is small and focuses mainly on plausibility, not on persuasion effectiveness or realism.

- The RL setup seems simplified, with few details on reward design or stability; could the gains be due to overfitting to Receiver quirks?

- The interpretation of “strategic information disclosure” could be more carefully supported, since it relies on mutual information as a proxy rather than direct analysis of message content or strategy patterns.

- The results show that even small models can become better persuaders, but there is little discussion about the societal consequences of making persuasion more efficient.

- The writing occasionally mixes theoretical and empirical points without clear transitions, which can make the paper feel dense and slightly unfocused to some extent.

### Questions
How exactly were "persuasion gains" computed and normalized across datasets with different scales and topics?

To what extent do Receiver models actually simulate belief updating rather than just mimicking cooperative agreement?

Did the RL models learn general persuasion principles, or did they just adapt to the Receiver’s specific response patterns?

How are dynamic persuasion "turns" evaluated? Does a higher number of turns always increase gains, or can models over-persuade?

How would this framework extend to human-in-the-loop validation, especially given the ethical limits of testing harmful persuasion topics?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper frames LLM persuasion as Bayesian Persuasion (BP): a sender strategically reveals information to shift a receiver’s beliefs and actions. It repurposes multiple human–human persuasion datasets to build static (1-turn) and dynamic (multi-turn) evaluation environments, measuring “persuasion gains” aligned with BP theory. Experiments show larger, frontier models and multi-turn interaction yield substantially higher gains; small models further improve via reinforcement learning (PPO/GRPO). The work offers a principled benchmark, analyses, and an RL recipe to enhance strategic persuasion.

### Strengths
1. Novel Integration of Theory: The paper’s strongest aspect is its original framework grounding LLM persuasion in Bayesian persuasion theory. This bridges a well-established game-theoretic model with modern LLM evaluation, bringing conceptual rigor to a domain that lacked principled metrics. By using persuasion gain and information design notions, the authors provide a clear, unified lens to measure and reason about persuasive behavior across different contexts, which is a significant innovation.

2. Propose new benchmark: Propose a new benchmark for strategic persuasion by reusing and consolidating multiple human debate/persuasion datasets.

3. Perform thorough empirical evaluation: The experimental evaluation is comprehensive and insightful. The authors test a wide range of models – from a 7B open model up to GPT-4/Claude – under identical conditions, providing a clear picture of how model scale and architecture affect persuasive ability.

4. RL significantly boosts small models and generalizes across receivers. A 3B Sender trained with PPO/GRPO on ~2.7k instances improves meaningfully over base in both static and dynamic settings, and performance transfers when paired with other Receivers (Mistral-7B, Qwen-7B) not seen in training. This suggests the agent really learns strategy.

### Weaknesses
1. Metric and Receiver calibration. Persuasion gain is measured via Likert shifts from a particular Receiver; Table 3 shows Receivers differ notably in susceptibility (e.g., Mistral-7B vs Llama-8B), raising questions about and cross-Receiver comparability of scores. Some normalization/robustness analysis would help.

2. Missing supervised fine-tuning baselines & strategy ablations. The Sender is improved via RL (PPO/GRPO with KL regularization), but there’s no direct comparison to supervised fine-tuning on the same instances or to non-strategic baselines.

### Questions
1. RL vs supervised fine-tuning. On the same ~2.7k instances, how would instruction/SFT compare to PPO/GRPO? An SFT baseline (and a combined SFT→RL variant) would clarify whether policy optimization is key versus exposure to data alone.

2. Safety auditing during RL. Did your RL Sender ever produce undesirable tactics (exaggeration, undue pressure) despite the truthful framing? Beyond a KL penalty, were there filters or audits to detect and penalize such behaviors during training or eval? Examples and frequency would help.

### Soundness
3

### Presentation
3

### Contribution
3
