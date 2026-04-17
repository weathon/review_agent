# Choices Speak Louder than Questions

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Recent findings raise concerns about whether the evaluation of Multiple-Choice Question Answering (MCQA) accurately reflects the comprehension abilities of large language models. This paper explores the concept of \textit{choice sensitivity}, which refers to the tendency for model decisions to be more influenced by the answer options than by a genuine understanding of the question. We introduce a new scoring method called **Normalized Probability Shift by the Question (NPSQ)**, designed to isolate the impact of the question itself and provide a more reliable assessment of comprehension. Through experiments involving various input formats, including cloze, symbols, and hybrid formats, we find that traditional scoring methods — such as those based on log-likelihood or its length-normalized variant — are vulnerable to superficial characteristics of the answer choices. In contrast, NPSQ remains stable even when modifications are made to the answer options.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, authors raise the issue of LLM answers to questions with option choice being driven by some intrinsic hints in the option texts, not by understanding of the question. Authors decompose the log-probability that model assign to each answer to the given question into choice- and question-driven components and develop a method to identify questions for which the model's answers are caused by choice texts. Authors apply this method to analyze several LLMs (base and instruction-tuned) on three common MCQA benchmarks. Finally, they propose a new method to select the answer option (NPSQ) that is less sensitive to superficial characteristics of the answer choices than log-probability and provide analysis to show that it is a more reliable tool to assess the model's comprehension of the question.

### Strengths
- This paper addresses an important problem; it proposes a novel approach to identify and mitigate such "choice-induced" biases; this method is easy to deploy and does not require training of a model or additional data.

- The proposed method can be used to improve existing (and future) MCQA benchmarks by removing "shortcuts" presented in the formulation of choice option texts.

- Experiments cover different task formulations (options only/cloze prompting/hybrid). A detailed analysis of the proposed method (NPSQ) and observed effects is provided.

- Paper is well-written and easy to follow.

### Weaknesses
- My main concern regarding the proposed method is the fact that it does not take into account effects caused by the order of the options (which is known from the previous studies in the field to be an influential factor). In several works it was shown that, given a list, LLMs often `focuses' more on the later entries from it than on earlier. It can (theoretically) affect the proposed sensitivity analysis method:

Choice sensitivity inequality for one question (line 140) can be reformulated into
$$
2 * (Score_{choice}(Q, C, x_1) - Score_{choice}(Q, C, x_2) ) > Score(Q, C, x_1) - Score(Q, C, x_2)
$$
if $x_1$ is the last option (e.g., D for 4-option setup in MMLU), and $x_2$ is one of the first, left side may be very high, because without question (empty "C" in calculation of $Score_{choice}$) mentioned above effect of list item recall may take place.

A similar issue arises in the formula for NPSQ (line 321).

 I would recommend to further analyze this issue and, maybe, (although it is just one of the potential ways to address it, if it is a real problem; I do not insist on using this suggestion) average probabilities from initial options order and after a certain permutation.

- The performed experiments cover only relatively small models up to 8B parameters. Presented results show that larger model has smaller (but still noticeable) sensitivity bias, and it would be very interesting to see the results for a larger model (at least -14B or -32B).

### Questions
- Would inclusion of irrelevant (i.e., ) "uncertainty sinks" options (like ``I don't know`` or ``None of the above``) affect the model's sensitivity to options?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates "choice sensitivity" in the MCQA evaluation of LLMs: the phenomenon where models exploit superficial features of the answer options rather than genuinely comprehending the question. The authors demonstrate that a significant portion of model decisions (20-60%) is driven by this sensitivity. To address this, they propose a new evaluation metric, Normalized Probability Shift by the Question (NPSQ), which is designed to isolate the impact of the question itself. Experiments, particularly those using adversarial choices, show that NPSQ is significantly more robust to option-based artifacts than traditional log-likelihood or length-normalized metrics.

### Strengths
The paper addresses a clear and increasingly important issue in LLM evaluation. As models achieve high scores on benchmarks, it is crucial to understand if this reflects true comprehension or artifact exploitation.

The proposed NPSQ metric is intuitive, well-motivated, and directly targets the identified problem by quantifying the "value" of the question.

The use of adversarial choices provides a very clear and convincing demonstration of the weaknesses of existing metrics and the robustness of NPSQ. The results in Figure 3 are particularly compelling.

### Weaknesses
While valuable, the contribution is an incremental improvement in evaluation methodology rather than a new task, model, and with no fundamental insight into model reasoning.

The analysis of why models exhibit this sensitivity is not deeply explored, though this is not the primary focus. The experiments are solid but could be extended to a wider range of model architectures and benchmarks.

### Questions
N.A.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper argues that standard MCQA metrics reward models for exploiting "option-surface cues" rather than understanding the question. It proposes NPSQ (Normalized Probability Shift by the Question), a new metric that isolates the question's contribution by comparing an option's log-likelihood with versus without the question, then normalizing this gain. Experiments on HellaSwag, ARC-Challenge, and MMLU show that while standard accuracy is fragile to adversarial or rephrased choices, NPSQ accuracy remains stable and can reorder model rankings. The study also quantifies this "choice sensitivity" and explores other factors like formats and prompts.

### Strengths
1. The paper identifies and formalizes a pervasive evaluation artifact, choice sensitivity, and gives a principled, testable metric to mitigate it.

2. The core construct (question-conditioned vs. question-ablated likelihood shift with normalization) is simple, auditable, and easy to slot into existing LM-eval pipelines.

### Weaknesses
1. The normalization in NPSQ is not stress-tested against plausible alternatives (e.g., z-scores, temperature scaling, ECE), leaving ranking stability under-substantiated.

2. Key stability claims (flip rates, adversarial drops) lack uncertainty quantification and significance testing, weakening the statistical support for the conclusions.

3. The metric relies on token-level probabilities and a hand-crafted “no-question” template whose wording or API backend may change outcomes, reducing reproducibility.

4. Computing scores with and without the question doubles evaluation cost, which is non-trivial for large suites.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper investigates LLM benchmarking methodology.  It begins by exploring the fact that some LLMs can answer multiple-choice benchmarking questions without actually looking at the question, implying that they are influenced directly by the content of the answers, as opposed to a true understanding of the question. This suggests some natural measures of the influence of the choice text versus the question text.  A probabilistic view on these measures further suggest the NPSQ (for "Normalized Probability Shift for the Question") which normalizes the vanilla measure to account for different baselines.

Empirically, the paper explores the choice sensitivity and NPSQ on a small set of models and benchmark tasks.  Results show that a surprisingly high percentage of performance is attributable to choice sensitivity, suggesting both improved measures are needed (and, I will add, improved benchmark questions, although this is not argued for in the paper).

### Strengths
I liked this paper. I think it's well-written, addresses an interesting issue, primes other researchers for future work in this area, and makes non-obvious contributions to the literature.

* Understanding choice sensitivity seems like an important issue in benchmark design.

* The method of calculating choice sensitivity is natural and intuitive.

* The empirical results convincingly show a wide variety of surprising behavior of LLMs wrt choice sensitivity.

* The authors (generally) do a great job of writing, and highlighting important conclusions.

Despite its weaknesses, I recommend acceptance.

### Weaknesses
I think the biggest weakness of the paper lies in the presentation.

I think Section 3 was beautiful.  It flowed naturally, the experiments were clean, and the authors did a great job of pulling out crisp conclusions.

Section 4 was fine; it introduces NPSQ. It would have been nice to connect the mathematical notation in Section 3 to that in Section 4 a bit more directly -- it was VERY unclear what the "score" function in Section 3 was -- and since it was mentioned that "log p(x|q,c)" was part of it, it seems like there are some unstated notational overlaps between the sections.

Where things get muddled is Section 5.  I supposed I expected to see a clean comparison of NPSQ vs. the vanilla Choice Sensitivity in Section 3, but instead, the authors *also* introduced the idea of adversarial prompting.  This came out of nowhere, and (to me, at least) derailed the narrative flow.  I kind of see how it was designed to really skew the choice information, and therefore demonstrate how NPSQ was more robust the regular CS, but it was pretty unclear to me why this idea was introduced in Section 5 -- it seems like it should have been introduced earlier, or not at all.

I of course wish that the authors had tested on a wider variety of LLMs.  I understand computational limitations and all, but still - it seems that, as a benchmarking paper that is entirely empirical, more could have been done.

### Questions
* Why wasn't adversarial prompting introduced earlier (in sec 3?) and evaluated as part of the basic CS experiments?

* I was interested to see how choice sensitivity decreases as a function of model size. It seems like you could have tested on 70/80b variants of several of your models; is there a reason you didn't?

### Soundness
4

### Presentation
3

### Contribution
3
