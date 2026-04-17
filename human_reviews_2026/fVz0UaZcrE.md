# Not All Code Helps: Disentangling the Impact of Code Data on Mathematical Reasoning in Large Language Models

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Incorporating code into training corpora has become a widely acknowledged practice in the development of modern foundation language models (LMs). Compared with a general Internet corpus, code offers high-quality, well-structured signals that substantially augment the coding proficiency of models. Beyond programming skills, prior research has suggested that code data may also contribute to non-coding capabilities. Nevertheless, through a series of rigorous controlled experiments, we demonstrate that the influence of code on other domains—particularly reasoning—remains limited.
Our principal findings are as follows: (1) Code corpus yields substantial gains in programming-related abilities but only marginal improvements in non-coding tasks. We further observe that code competed with knowledge-intensive tasks. (2) Not all code data enhances the mathematical reasoning ability. We identify core subset that functions as cognitive scaffolding for mathematical reasoning, especially for complex problem-solving scenarios. (3) Formal reasoning (e.g., code reasoning or program-of-thought approaches) provides more pronounced improvements in challenging mathematical reasoning tasks, while natural language–based reasoning proves more effective for simpler reasoning problems.
Finally, by probing the internal mechanisms of LMs, we reveal how training data modulates routing patterns, thereby shaping emergent model behavior. As a central driver of model capability, our findings disentangle domain-specific data into finer-grained, cross-domain ability dimensions and underscore promising directions for future data optimization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a systematic analysis of how code data influences mathematical reasoning in large language models. The authors perform extensive controlled ablation studies using a rigorously curated, 10T-token multi-domain corpus, explicitly separating pure code data, mathematical data, and cross-domain data. Their findings demonstrate that while pure code data enhances programming proficiency, it yields negligible or even negative transfer effects on non-programming reasoning tasks, with this trend being particularly pronounced for complex mathematical tasks. Beyond this, the work identifies core subsets of cognitive scaffolds (i.e., structured reasoning data) that effectively improve LLM mathematical reasoning, clarifies the competitive and cooperative dynamics between different data domains, and investigates how these domain-specific effects manifest at the level of MoE expert routing.

### Strengths
- **Rigorous controlled ablation design**: The work stands out for its systematic and large-scale controlled experiments, directly isolating the effects of code and math data. The methodology surpasses prior works in precision, especially in explicitly separating “pure code,” cross-domain, and math, avoiding previous confounds
- **Identification and operationalization of cognitive scaffolds**: Through targeted data selection and experiments, the paper concretely shows that structured reasoning (“cognitive scaffolds”) enhance complex mathematical reasoning without the same trade-offs as simply adding code or general math data
- **Mechanistic analysis**: The MoE expert routing analysis provides concrete, interpretable evidence for internal distributional shifts associated with different data compositions, lending mechanistic insight rarely addressed in prior work.

### Weaknesses
- **Poor generalization and transfer performance**: The experiments are limited to a single ≈10T-token corpus and one MoE architecture. It therefore remains unclear whether the findings generalize to other model families or scales (e.g., dense transformers or different MoE sizes) or to training regimes with different token budgets or data distributions. Additionally, the evaluation is heavily math-centred, so it is unknown whether the observed code–text interference also arises in other knowledge-intensive domains (for example, legal or scientific text).

- **Confounding by budget and token allocation is not ruled out**: The claim that “code causes degradation on complex math” is based on ablating domains while keeping total training schedule variable. This confounds domain content with token-budget allocation and capacity effects. The authors should run controlled comparisons that keep token budget and effective model capacity fixed (e.g., replace removed tokens with matched-quality neutral text or randomized tokens) to isolate content effects.

- **Insufficient theoretical grounding for certain claims**: While the MoE equations and routing mechanisms are clearly stated, some mechanistic claims (e.g., about expert collapse, cognitive scaffolding as meta-reasoning support) are asserted more on empirical observation than theoretical underpinning. Further analysis, possibly through formal modeling or causal inference, would substantiate these conclusions more rigorously.

- **Operational definition of “cognitive scaffolds” somewhat heuristic**: While the structured reasoning subset is constructed with clear intent, its precise operational window is not exhaustively defined or justified beyond the FastText-based selection pipeline. Also, the claimed “cognitive scaffold” set is judged by downstream effect rather than human assessment of whether it truly scaffolds reasoning.

- Errors in writing details: For instance, in the introduction section:

> models exposed to code exhibit superior performance in programming-related tasks Ma et al. (2024); Aryabumi et al. (2025).

This formatting is non-standard; the in-text citations should use the \citep command
Another example:

> code is generally of higher quality and more structurally coherent, often leading to lower training loss at convergence 6

Here, the number “6” refers to a figure included later in the text. However, the standalone numeral provides no explicit indication that it points to a figure. This lack of clarity creates confusion for readers.

### Questions
- The operationalization of “cognitive scaffolds” relies on a FastText-based selection and symbol-density heuristics. Can the authors clarify if other methods (graph/rule-based extraction, manual annotation, or logic graph mining) were considered and how their approach compares in error/signal separation?
- To what extent do your empirical findings generalize to standard dense transformers, or other architectures not using MoE, and to different corpus sizes (e.g., under 10T tokens)?
- What are the implications or proposed practical guidelines for including code/math in future LLM training, based on your competitive/cooperative findings?
- Can you provide 3–5 anonymized example items from each domain: pure code, code-NL, scaffold positive, and scaffold negative? This will help validate the data-split claims.
- What is the precision / recall of your automatic classifier for domain splitting (evaluated on a human-annotated sample)? Please report confusion matrix.
- How would you expect your recommendations to change under data-efficient regimes (e.g., for teams that cannot train on 10T tokens)? Is scaffold selection still sufficient at much smaller budgets?

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
4

### Summary
This paper examines the effect of code data on pretraining, ablating the effect of holding out/including code and math datasets when pretraining a MoE trained over 200B tokens. They find that, contrary to prior work, code data does not always help reasoning abilities, but there does exist a subset of ‘cognitive scaffold’ code data that yields large improvements in difficult reasoning tasks. They also find that math can help some coding tasks, code hurts knowledge-intensive tasks, and ablating domain-specific data can have a substantial effect on routing patterns of the MoE.

### Strengths
- Directly performing these data ablations at pretraining time is an interesting and useful contribution to the community, and the findings have clear insights into what sorts of data should be prioritised for what tasks.
- The findings being against prior work (likely due to the difference in classification) is interesting.
- The experiments around cognitive scaffolds (and more broadly identifying useful sub-parts of datasets) is interesting and useful for the community.

### Weaknesses
- The claim about the findings being different to prior work around code data would be useful to more thoroughly test: if we include code with code-NL data, do we then see results in line with prior work, or are there other aspects of the setup (such as the dataset being considered) that affect the result?
- Only one model type/size (MoE, 24 layers) is considered. While I appreciate that these experiments are expensive, it would be useful to show results across different model types and sizes to be more certain about the findings. Similarly, some notion of what differences are statistically significant would be nice to see/
- I found the description of how the data was made and filtered a bit unclear: How was the corpus split into the domains in section 3.2.2, exactly? More details on the synthetic data creation in section 3.2.1 would also be useful.

Overall, I think the paper is solid and has some interesting insights, but more information about the data creation and curation would help in making it more useful for the community.

### Questions
- The authors state they use dynamic sampling for data during training (line 257), could they provide more details on what this is?
- Could you show some examples of what the cognitive scaffold data looks like? Currently it is a little unclear to me what sort of data is getting picked out by the fast text classifier.

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
3

### Summary
This paper disentangles the impact of code data on large language model reasoning through controlled experiments, showing that while generic code mainly boosts programming skills, only structured reasoning data functions as cognitive scaffolds that enhance complex mathematical reasoning and cross-domain generalization.

### Strengths
- The paper tackles the problem of disentangling the impact of code data on large language model reasoning and conducts large-scale experiments with systematic ablations and analyses.
- The paper is clearly written and easy to follow.

### Weaknesses
- The study bases all findings on MoE architectures, yet MoE and dense models are not functionally equivalent in terms of reasoning behavior and representation dynamics. Since the competitive effects between code and math data may partly arise from MoE's expert routing and specialization mechanisms, it remains unclear whether the same patterns would hold for dense transformers commonly used in previous "code enhances reasoning" studies. Including dense baselines or discussing this architectural limitation would help validate the broader applicability of the conclusions.

- The paper redefines "code" as pure executable code while retaining Code-NL in ablations and then attributes "code competes with reasoning" to the code bucket. This boundary shift introduces a major construct confound relative to prior work and could invert conclusions unless quantitatively controlled.

- The y-axis meaning in Figures 2–4 is not explicitly defined. Although the context suggests that the y-axis likely represents relative performance change in percentage, the paper does not clearly specify the metric, unit or normalization procedure used.

- Some subplots in Figure 16 appear to be blank and it would be better for the authors to check whether this issue stems from missing results, or plotting errors.

### Questions
- Could the authors clarify whether the observed competitive effects between code and math data are specific to the MoE architecture? Have the authors conducted whether similar patterns hold for dense transformer models used in prior studies?

- Could the authors explain how redefining "code" as pure executable code while retaining Code-NL in ablations affects the interpretation of the finding that "code competes with reasoning"? How do the authors ensure that this redefinition does not introduce a construct confound compared with prior work?

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
4

### Summary
The paper re-examines the claim that “code helps reasoning” and finds it only partly true. Using MoE models trained from scratch on a 10T corpus cleanly split into six domains (Web, Code, Math, Wikipedia, Books, Multilingual), the authors run controlled ablations and observe negative coupling between domains: pure code predictably boosts programming, but competes with knowledge-intensive and especially complex math and slightly depresses commonsense. Conversely, math improves competitive programming while hurting code-reasoning and some common-sense tasks. A key design choice is to separate pure code from Code-NL to decouple earlier reports of broad code, leading to reasoning gains from cross-domain text signals. 
To mitigate the trade-off, the authors up-weight a curated subset of structured “cognitive scaffolds” while keeping the math budget fixed, yielding large lifts on hard math with minimal side-effects elsewhere. An MoE expert-routing analysis suggests these scaffolds strengthen complex reasoning while leaving routing distributions relatively stable.

### Strengths
**Important Question, Strong Empirical Stance:**
The paper rigorously re-tests the assumption that “code universally helps reasoning,” using large-scale, controlled domain ablations to reveal negative coupling between capability axes. The result is immediately actionable for pretraining mixture design and evaluation practice. 

**Operational Taxonomy of Code:**
By separating pure code from Code-NL (code interleaved with natural language), the study disentangles signals that prior work often conflated, enabling cleaner attribution of gains and setting a practical standard for dataset curation and reproducible ablations.

### Weaknesses
**Undercharacterized Per-benchmark Deltas:**
This paper highlights large deltas (e.g. -71.53%, -47.16%) which are sprinkled through the text, but provide no single view that aggregates all gains/losses across benchmarks. Without a consolidated table or plot, it’s hard to assess the true magnitude, variance, and pattern of negative coupling (where one domain’s gains coincide with another’s losses). Providing a consolidated view will materially strengthen the paper’s central claims about negative coupling.

**Under-Specified Scaffold Selection:**
Cognitive scaffolds are central to this paper’s claims, yet the selection pipeline lacks crucial detail. The FastText filter (~400k train, ~200k positives) is described, but there’s no labelling protocol, source list, heuristics/regex criteria, thresholding rationale, or contamination audit against eval sets. Report precision/recall, calibration curves, examples of common false positives/negatives, and an ablation vs. simpler baselines. Providing these details will substantively strengthen the causal interpretation and credibility of the scaffold-driven gains.

**Metrics Headlines:**
The authors cite two different headline degradations for math (-14.38% “on average” and -10.1% “overall”) without defining the aggregation behind each, leaving readers unsure which number reflects the main effect. The authors also consistently plot single numbers/bars with no error bars or confidence intervals, leaving readers unable to judge statistical reliability or run-to-run variance. Clarifying the aggregation and adding uncertainty will strengthen the quantitative claims and their reproducibility.

**Missing Axis:**
The authors present bars, Figures 2, 3, and 4, but the Y-axis is unlabeled, so readers can’t tell whether those bars are accuracy points, relative percent change deltas, or some other normalized score, nor what range they span. Clear axis labelling will strengthen the paper’s interpretability and make the negative-coupling trade-offs unambiguous.

### Questions
Main Comments are located in Weaknesses

Small Comment:
- β1 = 0.9, β1 = 0.95, Did the authors mean B2?

### Soundness
2

### Presentation
2

### Contribution
3
