# Evidence for Limited Metacognition in LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
The possibility of LLM self-awareness and even sentience is gaining increasing public attention and has major safety and policy implications, but the science of measuring them is still in a nascent state. Here we introduce a novel methodology for quantitatively evaluating metacognitive abilities in LLMs. Taking inspiration from research on metacognition in nonhuman animals, our approach eschews model self-reports and instead tests to what degree models can strategically deploy knowledge of internal states. Using two experimental paradigms, we demonstrate that frontier LLMs introduced since early 2024 show increasingly strong evidence of certain metacognitive abilities, specifically the ability to assess and utilize their own confidence in their ability to answer factual and reasoning questions correctly and the ability to anticipate what answers they would give and utilize that information appropriately. We buttress these behavioral findings with an analysis of the token probabilities returned by the models, which suggests the presence of an upstream internal signal that could provide the basis for metacognition. We further find that these abilities 1) are limited in resolution, 2) emerge in context-dependent manners, and 3) seem to be qualitatively different from those of humans. We also report intriguing differences across models of similar capabilities, suggesting that LLM post-training may have a role in developing metacognitive abilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates metacognitive abilities in LLMs, taking inspiration from research on metacognition in nonhuman animals. The authors adopt two task paradigms that measure metacognition without explicit self-reports, and find somewhat mixed results: recent models show modest success, but there is no clear, consistent evidence that the tested models are showing strong metacognitive abilities.

### Strengths
Overall, I liked this paper. The contribution of evaluating metacognition without self-reports is a nice addition to the literature. The writing is clear, the analyses are well-motivated and sound, and the conclusions are appropriate given the findings. I appreciated that the authors included surface features in the regression model in Section 2.4. The expected outcomes and logic laid out in Section 3.2 are also great, and lacking in most AI papers these days.

### Weaknesses
I like that the authors eschew self-reports, but the paper could be strengthened by providing more motivation for using implicit measures of metacognition. The current motivation in the second paragraph of the Introduction feels a bit weak. For example, I don’t understand the argument laid out by the following sentence: “Because LLMs have vast memory capacities and are trained on a nontrivial fraction of everything humans have ever written with the singular goal of generating plausible and pleasing responses, they are almost preternaturally ill-suited to trustworthy self reports” (l. 045-048). Why does having a large memory capacity, or being trained on a large amount of data, make the model ill-suited for self-reports? Since implicit measures of metacognition might introduce additional task demands, I think a bit more work needs to be done to motivate them.

### Questions
A few minor suggestions/questions:
- Section 2.1 (“Models”) is quite hard to parse. This information could maybe be better presented in a table, with the main text just highlighting the high-level motivation for how these models were chosen, and any theoretically relevant differences between them.
- In the Discussion, the authors write: “Speculating, as with the lack of advantage for factual knowledge in metacognition, the relatively poor performance in the self-modeling task may relate to the fact that LLMs don’t have the equivalent of the hippocampus, which in mammals subserves both the explicit recollection of facts and the ability to simulate one’s own behavior” (l. 473-476). This claim feels like a big logical jump. I doubt the authors are implicitly claiming that models need brain-like functional architectures to achieve mammals’ abilities, but this claim hints at that. Also, to my knowledge, we just don’t know (without further mechanistic studies) whether LLMs have hippocampus-like “regions” or not.
- You might want to check out relevant papers by Song et al.: “Language Models Fail to Introspect About Their Knowledge of Language” (COLM 2025), and “Privileged Self-Access Matters for Introspection in AI” (preprint 2025).

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
2

### Summary
The paper introduces two behavioral paradigms—Delegate Game and Second Chance Game—to test metacognition in LLMs without relying on self-reports. In Delegate, a model must decide to answer a question or delegate to a teammate; metacognition is inferred if delegation correlates with model-internal confidence (proxied by baseline correctness or token-probability entropy) after controlling for surface difficulty cues. In Second Chance, a model is told its previous answer was wrong and must re-answer; metacognition/self-modeling is inferred if it changes appropriately beyond neutral-prompt change rates and beyond alternative strategies (random choice, implausible-option choice, added noise). Experiments span many frontier models released since 2024 and two datasets (GPQA and SimpleQA) plus format-converted variants (GPSA and SimpleMC). The main finding: limited, context-dependent evidence for metacognitive abilities—partial correlations typically ≤0.3 in Delegate (correctness) and up to ~0.5 using entropy; Second Chance shows modest, model-specific gains after ruling out alternatives for some OpenAI models. The authors argue post-training may shape these abilities and provide a public code release.

### Strengths
1. Careful design and statistical controls target internal confidence rather than surface difficulty or stylistic priors.

2. (confidence-use vs. self-modeling) provide complementary views; the figure suite clearly visualizes small but significant effects. 

3. Breadth of models and format-matched datasets (MC and short-answer) aid generality claims; reproducibility details and code availability help follow-on work.

### Weaknesses
1. Effect sizes are small and variable. Delegate partial correlations peak around 0.3 (correctness) / ~0.5 (entropy), and elicitation is inconsistent across formats and models; some provider-specific RLHF “personalities” (over-answering) confound interpretation.

2. Short-answer scoring uses LLM-as-judge. Although cross-provider panels are used, human adjudication or stricter exact-match normalization would strengthen claims.

3. Tasks are QA-centric and relatively short context; it’s unclear whether effects hold for planning-heavy, long-horizon, or tool-use settings.

### Questions
1. How sensitive are Delegate/Second-Chance effects to minor wording or teammate-history changes?

2. Can you harmonize decoding across models (e.g., deterministic MC with calibrated proxies) to reduce comparability confounds?

3. Could you run the same paradigms with humans (and report comparable partial correlations/change-lifts) to contextualize effect sizes?

4. Do findings persist on chain-of-thought-heavy reasoning or multi-step tool-use tasks where internal planning signals might be stronger?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores whether there is evidence that frontier LLMs have metacognitive abilities largely through behavioral evaluations. More specifically, the authors investigate self-modeling using two behavioral paradigms: the delegate game and the second chance game. They define self-modeling as the ability to monitor and control one's internal states and propose that their evaluations inspired by non-human animal studies are more reliable than self-reports. Their results suggest limited evidence of graded, metacognitive abilities in LLMs, which the authors suggest are qualitatively different from human metacognition.

### Strengths
The paper is original in its proposal of a novel non-linguistic methodology from studies in non-human animals for LLMs to evaluate self-modeling abilities. In particular, the fact that the method bypasses the need for self-reports from the LLMs makes it a significant contribution in the right direction for designing evaluations of this kind. Although the results are suggestive at best, the methodology is clear and the analyses sound.

### Weaknesses
The paper's central claim is finding "evidence for limited metacognition", but the supporting evidence is consistently described by the authors themselves as "limited in resolution". Given the modest effect, it remains highly plausible that other subtle, non-introspective strategies (e.g., learned heuristics that associate "your answer was incorrect" with "pick the next most likely token") could be responsible for the small performance lift. The paper does not provide a strong affirmative case for why self-modeling is a more parsimonious explanation than some unknown, simpler heuristic. Additionally, the authors don't address what seems like a fundamental question about whether it even makes sense to ask this question about self-modeling or awareness given the architecture that LLMs run on (i.e., lack of recursion, feedback connections, and so on). Finally, some more work on situating the behavioral results using mechanistic interpretability would solidify the findings if further support was found there.

### Questions
- What might be some other heuristics that you have considered might explain the behavior of the modest effect presented in the paper but are difficult to evaluate, and why? 
- What does it mean for an LLM to have a "partial" self-modeling ability? Your results and discussion highlight the differences with human metacognition, but it's not clear what it means if a model is not acting fully in accordance with what you would expect from an agent that has metacognitive abilities.
- What kinds of mechanistic interventions would provide stronger evidence that the models are indeed demonstrating metacognitive abilities?

### Soundness
3

### Presentation
4

### Contribution
3
