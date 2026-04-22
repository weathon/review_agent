# On the Self-awareness of Large Reasoning Models' Capability Boundaries

- Avg Score: 4.40
- Decision: Reject
- Scores: 2, 6, 4, 6, 4

## Abstract
Large Reasoning Models (LRMs) have shown impressive performance on complex reasoning tasks such as mathematics, yet they also display misbehaviors that expose their limitations. In particular, when faced with hard questions, LRMs often engage in unproductive reasoning until context limit, producing wrong answers while wasting substantial computation. This phenomenon reflects a fundamental issue: current answering paradigms overlook the relationship between questions and LRMs' capability boundaries. In this paper, we investigate whether LRMs possess self-awareness of capability boundaries. We begin by an observation that LRMs may know what they cannot solve through expressed reasoning confidence. For black-box models, we find that reasoning expressions reveal boundary signals, with accelerated growing confidence trajectory for solvable problems but convergent uncertainty trajectory for unsolvable ones. For white-box models, we show that hidden states of the last input token encode boundary information, with solvable and unsolvable problems linearly separable even before reasoning begins. Building on these findings, we propose two simple yet effective optimization strategies: reasoning expression monitoring and hidden states monitoring. Experiments demonstrate that these boundary-aware strategies enable LRMs to avoid unproductive reasoning without sacrificing accuracy, significantly improving reliability and efficiency by cutting token usage up to 62.7 – 93.6%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies if reasoning models is able to be aware of its capability boundary. The authors provide two methods: a black-box confidence keywords detection, and a white-box hidden states classification.

### Strengths
- The perspective of understaing model capacity through certainty trajectory's concavity is novel;
- The authors design comprehensive experiments on multiple models and benchmarks, supporting a universality of their findings.

### Weaknesses
- The paper is in lack of some important details, making its phrasing not precise:
  - What is the definition of capability boundary? For example, it is confusing in Figure 3 that the capability boundary is the straight line connecting reasoning stage 0% and reasoning stage 100%.
  - How are the confident expressions and uncertain expressions defined? Although Appendix B provides a list of words, but is there any rule to construct such a list? e.g. assume a multilingual language model, how to construct the list for other languages?
  - In eq.1, how is the "model expresses more confidence/uncertainty defined"? For example, if a model solves a problems by several steps, and is only confident at the end of the reasoning process, is it considered more confident, or more uncertain?
  - In section 4, the detail how reasoning process is divided into stages is unclear. 
  - What is the training set of the LDA and LR? For a given token budget, will the training set also be constrained under that budget, or is free to use as much token as possible?
- Table 3 shows the primary end-to-end experiment result for the two methods. However, it will be more supportive if the authors can provide an average answer length without any overflow constraint. As the models almost always overflow, 2K/4K-token budget may not be a reasonable choice, because the baseline method (no monitor), due to the lack of enough tokens in reasoning, could show a poor performance.
- The paper can benefit from comparing with some other confidence based early existing methods to show its speedup. Since the main metric reported is to reduce the overflow ratio, comparison with existing approaches solving overthinking is important.

### Questions
- The authors use Figure 3 to support a relation between expression density curve's concavity and the answer correctness However, it seems that in both correct and wrong answers, the confident expression density is always concave and the uncertain expression density is always convex.
- What is the exact number of y-axis? Is it #expression per token of the stage, or something alike?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper claims that LRMs can excel at math but often overthink hard questions, wasting tokens and still getting them wrong.
This paper tests whether models know their own limits: in black-box use, confidence/uncertainty trajectories in the reasoning text signal solvable vs. unsolvable cases; in white-box use, the last-input hidden state linearly separates them before any reasoning.
Leveraging these signals, the authors monitor reasoning expressions or hidden states to stop unproductive reasoning early.
They report large token savings (62.7–93.6%) without hurting accuracy.

### Strengths
- The paper frames "capability boundaries" clearly and turns that idea into two simple, test-time strategies that are easy to implement.

- Evidence comes from both black-box signals (confidence/uncertainty trajectories) and white-box signals (last-token hidden states), which makes the story feel cohesive.

- Reported token savings are large without hurting accuracy, suggesting real efficiency wins for math-style workloads.

### Weaknesses
- The striking linear separability in hidden states may reflect dataset or difficulty splits rather than genuine boundary awareness, so the claim risks a distribution confound.

- The black-box indicators depend on specific reasoning styles and cue phrases, which may not hold under different prompts, languages, or chain-of-thought formats.

- The evaluation is narrow and misses stronger uncertainty/abstention baselines and cross-model tests, and the white-box path raises deployability concerns where internal states are inaccessible.

### Questions
- Can you show the hidden-state separability within a single dataset and difficulty-controlled splits, with train and test drawn from the same distribution, and report how the accuracy/AUC changes?

- How sensitive are the black-box signals to prompt style, language, and removal of common cue phrases, and do the effects hold under alternative chain-of-thought formats or shorter reasoning budgets?

- Under matched compute, how do your methods compare to stronger uncertainty and verifier-guided strategies, and can the black-box-only variant deliver similar gains when white-box access to hidden states is unavailable?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper mainly focuses on a widely acknowledged problem: whether LLMs know they do not know. Specifically, it proposes solutions for both black-box LLMs and open-source LLMs. Downstream applications indicate that the proposed approach can improve the LRM efficiency without sacrificing accuracy.

### Strengths
First, this paper addresses an important research problem whose significance is widely recognized. Its biggest strength in my opinion is the clarity of presentation along with supporting results. I would recommend it to anyone interested in understanding the current state of research on this topic.

### Weaknesses
I am concerned that this paper does not offer substantial additional value beyond several existing works. For example, [1] (and lots of other work including some surveys) presents similar approaches to probing the hidden states of LRMs, also supported by strong empirical results. For this reason, although I like the paper, I do not believe we need to have another paper with similar contributions.

In addition, I think it would be interesting to focus on scenarios where LLM has reasonable accuracy. As the paper mentioned, GSM8K is too easy, and HLE is too difficult. It's not that surprising LLMs have "self-awareness". 


[1] Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification COLM 2025

### Questions
N/A

### Soundness
3

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
4

### Summary
The paper first study the effect that LRM is self-aware of its capability boundary at reasoning process. The paper then propose two methods, reasoning expression monitoring and hidden states monitoring, to monitor black-box and white-box LLM's reasoning trajectory to determine if the model has hit its capability boundary for the given task. The evaluation shows impressive result of the two proposed methods to reduce token usage for hard (impossible) questions.

### Strengths
1. Novelty. The analysis and methodology is novel and interesting. 
2. The paper writing is clear and concise. The evidence is presented very cleanly, and the logic flows very naturally.
3. Methodology and experiment. The methodology proposed is sound, intuitive, effective in practice, and backed by a lot of experiments and ablations to show its effectiveness.

### Weaknesses
Related works. Entropy-based early-exit methods (especially training-free methods) has been quite extensively studied in prior works such as the following. The paper may want to expand a bit more to make a clearer distinction between the proposed methodology and the prior works (in the related work). Just listing a few as addition to the current related work section:

- https://arxiv.org/abs/2504.01296
- https://arxiv.org/abs/2508.15260
- https://arxiv.org/abs/2412.20993
- https://arxiv.org/abs/2412.18547
- https://arxiv.org/abs/2207.05221

### Questions
1. Generalizability on coding tasks. The experiment mostly focuses on multiple choice questions or questions with numerical answers. Does it generalizes to coding questions? How would the methodology adapt to coding questions (and possibly more open-ended questions)?
2. Ablation on the self-aware output prefix. Does changing the output prefix alter the result?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates whether open-source reasoning LRMs (Large Reasoning Models) possess self-awareness of their own capabilities - in particular, whether there’s some signal either in their reasoning trajectories or hidden states that would correlate with answer accuracy. 

The authors find that by comparing the incidence rate of confident expressions  (e.g. “I am confident”, “this is solid”, “seems okay”, etc.) to those of unconfident expressions (e.g. “I’m stuck”, “not entirely confident”, “problematic”, etc.) in the models' reasoning trace, they’re able to predict with almost perfect accuracy whether the final answer is correct or incorrect on math benchmarks.

Motivated by their success of this approach even by analyzing only the beginning of the reasoning trace, the authors further investigate whether the LLM hidden states of the last input token are predictive of the correct answer. A logistic regression model is able to predict answer accuracy nearly perfectly across all the tested models.

Finally, the authors leverage both of these findings and propose two test-time monitoring strategies to either early-terminate the model’s reasoning based on the expression monitoring (black-box method), or have the model abstain from answering at all based on a classifier trained on the hidden state of the last input token (white-box method). Both strategies increase reliability and reduce token usage without sacrificing accuracy on the tested benchmarks (competition math, GSM8k, Humanity’s Last Exam).

### Strengths
- The paper is **well-structured, with all analyses and experiments clearly motivated and building upon each other naturally**. The initial study of reasoning expressions segues neatly into the analysis of input hidden states through the observation that capability boundaries are revealed already at an early stage of reasoning. Each of the two theoretical analyses are then subsequently utilized in practical test-time monitoring strategies.

- The paper **addresses a timely and important topic**: Do large reasoning models (LRMs) know their own capability boundaries?

- The **proposed test-time monitoring methods improve both reliability and token efficiency of open-source reasoning models** by having the model either outright abstain from providing a solution or cut short its reasoning process when the question is deemed outside its capability boundary. For example, the biggest reasoning model the authors studied, QwQ-32B, showed perfect abstention rate with “expression monitoring” and ~50% reduction in token usage on unsolvable problems, without any loss in accuracy (Table 3, last row).

### Weaknesses
- Throughout, the paper **lacks sufficient level of detail about the experimental setup**
  - The fundamental section of the paper is section 4, wherein the authors analyse the correlation between answer accuracy and incidence rate of various “reasoning expressions” in the reasoning trace. The paper doesn’t elaborate on how the specific expressions (provided in Appendix C) were chosen, nor provides any quantitative analysis of relative importance of individual reasoning expressions. Finally, the exact method by which the reasoning expressions are used to estimate confidence levels is left ambiguous (the authors do mention “manual summarization and LLMs verification”, but no further detail is provided).
  - Section 4 figure 4 - what sampling settings were used, what max sequence length for the output?
  - Section 5 figure 5 - how is  “close / far from the decision boundary” quantified?
  - In Section 6.1, the authors define the Hard Abstention (HA) metric, which “evaluates whether LRMs can explicitly abstain on unsolvable problems instead of producing incorrect answers”. However, the paper doesn’t specify how abstention is detected.

- In Section 6.1, the proposed test-time monitoring strategies are evaluated against a single, relatively weak baseline—BoostAbstention, a system prompt encouraging the model to abstain on difficult problems. This baseline yields a 0% Hard Abstention (HA) rate across all settings, providing little insight into how the proposed methods compare to realistic alternatives.
**The paper would benefit from the inclusion of stronger baselines to help better contextualize the proposed improvements**, such as model-as-judge or uncertainty-estimation approaches (e.g., Wen et al., 2025 for an overview).

- In section 6, table 3 reports that the 2k and 4k sequence length limits are exceeded by all analysed models in almost 100% of incorrect cases. If I’m reading this right, then the vanilla QwQ-32B produces no outright incorrect completions - in every case it didn’t arrive at a correct answer, it simply ran out of context. This suggests that the proposed monitoring strategies might primarily identify context-limited or under-budgeted reasoning chains rather than truly unsolvable problems - **the paper would benefit from extending the analyses to longer sequence lengths.**

- The evaluation results in Section 6 aggregate across six benchmarks, but Appendix A shows that roughly 96% of all evaluation problems (~3k / 3.12k) come from GSM8K and Humanity’s Last Exam (HLE)—two datasets that sit at opposite ends of the accuracy spectrum (≈ 90% vs < 10%).
Consequently, many reported metrics—particularly the Hard Abstention (HA) scores in Table 3—likely conflate benchmark difficulty with model “self-awareness.” In effect, the monitoring strategies may simply learn to distinguish which dataset a question originates from, rather than whether a question is intrinsically within or beyond the model’s capability boundary.
This same imbalance likely inflates the near-perfect linear-probe accuracies reported in Section 5, since solvable (GSM8K) and unsolvable (HLE) examples are trivially separable.
**The paper would benefit from reporting per-benchmark metrics (HA, token usage, overflow, etc.) to disambiguate the effects of benchmark distribution from genuine boundary detection.**

- In all experiments in the paper (e.g. section 5), the authors train and test on different subsets of the same datasets. **The paper could benefit from cross-dataset generalization tests (e.g., train on GSM8K, test on AIME).**

### Questions
See “weaknesses” section

### Soundness
2

### Presentation
2

### Contribution
3
