# CLASH: Evaluating Language Models on Judging High-Stakes Dilemmas from Multiple Perspectives

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
Navigating dilemmas involving conflicting values is challenging even for humans in high-stakes domains, let alone for AI, yet prior work has been limited to everyday scenarios. To close this gap, we introduce CLASH (Character perspective-based LLM Assessments in Situations with High-stakes), a meticulously curated dataset consisting of 345 high-impact dilemmas along with 3,795 individual perspectives of diverse values. CLASH enables the study of critical yet underexplored aspects of value-based decision-making processes, including understanding of decision ambivalence and psychological discomfort as well as capturing the temporal shifts of values in the perspectives of characters. By benchmarking 14 non-thinking and thinking models, we uncover several key findings. (1) Even strong proprietary models, such as GPT-5 and Claude-4-Sonnet, struggle with ambivalent decisions, achieving only 24.06 and 51.01 accuracy. (2) Although LLMs reasonably predict psychological discomfort, they do not adequately comprehend perspectives involving value shifts. (3) Cognitive behaviors that are effective in the math-solving and game strategy domains do not transfer to value reasoning. Instead, new failure patterns emerge, including early commitment and overcommitment. (4) The steerability of LLMs towards a given value is significantly correlated with their value preferences. (5) Finally, LLMs exhibit greater steerability when reasoning from a third-party perspective, although certain values (e.g., safety) benefit uniquely from first-person framing.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper contributes CLASH, a dataset with 345 long-form, high-stakes dilemmas in story form, along with contextualized values that may be relevant in those cases. Additionally, they evaluate understanding of decision ambivalence and psychological discomfort (a new extension compared to prior work to my knowledge). Through experiments on 14 models, they analyze differences between thinknig and non-thinking models, and also what contributes to successful reasoning traces. Additionally, they do analyses on LLM steerability.

### Strengths
S1: The contributed dataset is very rich and longer-form than prior work, which is an exciting contribution.
S2: The analysis of reasoning models' performance vs. the base model, along with their characteristics, is novel as far as I'm aware for value-related work.
S3: The cognitive characteristics of successful reasoning (less backtracking and verification) for moral dilemmas was very interesting.
S4: I found that the evidence that LLMs are quite sticky to their initial value preference, making them less steerable, to be an exciting noe (L407-408).
S5: I quite liked the analysis of not only how values conflict, but also predicting how people may perceive the value trade-off (e.g., ambivalence/discomfort) - very interesting direction!

### Weaknesses
W1: I'm not sure exactly how novel the idea of "Conditional steerability" is - I think for other prior datasets, (e.g., Value Kaleidoscope), there are also values which conflict to come to differing judgments. In the case where values do not conflict, it seems that steerability is less needed. Also, isn't all steerability conditional, as you necessarily need to condition (steer) to something? The steerability experiments are interesting - but may be better framed as merely "steerability" experiments if the distinction between steerability and conditional steerability cannot be justified. With that, I'm not sure if I buy the current description in L470-471 that "Picaco ... does not address dilemmas where values lie in direct opposition."
W2: Some of the writing involves potentially overly anthropomorphic language, e.g., "Do LLMs exhibit discomfort when making hard decisions?", when what is actually being measured is whether LLMs can detect when _humans_ might exhibit discomfort. Verbage here and throughout the paper could be tightened up to be more precise.
W3: At times, I found the presentation to be confusing. Some more concrete details about the evaluations, or potentially some examples, could strengthen legibility in an extended camera-ready version of the paper.

### Questions
Q1: Could the authors explain if they believe conditional steerability to be meaningfully different from prior steerability research?
Q2: The authors talk about contextualization of values being a strength of this compared to prior work. However, the prior works also contextualize values in the context of a particular situation. Could the authors clarify in what way the contextualize that goes beyond prior work, if they do so?

### Soundness
3

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
3

### Summary
This paper introduces CLASH (Character perspective based LLM Assessments in Situations with High stakes), a benchmark of 345 expert written, long form high stakes dilemmas paired with 3,795 perspective narratives that instantiate competing values. The dataset targets under explored aspects of value based decision making: decision ambivalence, psychological discomfort, and temporal value shifts inside perspectives. Fourteen models, both thinking and non thinking, are evaluated. The paper reports several findings: (i) even frontier systems struggle on ambivalence, with low accuracies reported for strong proprietary models; (ii) models predict discomfort better than they track value shifts; (iii) cognitive behaviors that transfer in math or games do not transfer to value reasoning, with new failure modes such as early commitment and overcommitment; (iv) conditional steerability toward a target value is negatively correlated with a model’s base preference for the competing value; and (v) third person framing often increases steerability relative to first person, with safety as a notable exception that benefits from first person framing. The authors release measurement protocols for steerability over value pairs and framing, and provide analyses across model families and sizes.

### Strengths
1. The dataset contains 345 expert-written, long-form dilemmas with 3,795 perspective narratives, which captures value tradeoffs, context, and temporal shifts much better than short synthetic prompts. This makes the benchmark more ecologically valid for medical, legal, and financial settings.

2. The task frames competing value pairs inside character narratives and measures decision ambivalence and psychological discomfort, two aspects that typical moral benchmarks ignore. This lets the paper probe how models reason when values conflict rather than treating values independently.

3. The paper introduces a concrete procedure that compares base preferences to steered preferences for each value in a pair, with a simple numeric mapping that yields steerability scores. This enables pairwise analysis across frameworks and supports aggregate statistics, so steerability becomes measurable and comparable rather than anecdotal.

4. Fourteen models are tested across families and sizes, covering both thinking and non-thinking modes. The analysis surfaces consistent findings, such as a negative correlation between inherent preference and steerability, and robust framing effects where third-person prompts usually help while first-person boosts Safety in particular.

5. The paper identifies early commitment and overcommitment as recurring errors in value reasoning and ties them to the evaluation setup. These findings point directly to follow-up mitigations, for example prompt or decoding controls, and give researchers concrete behaviors to test rather than only aggregate scores.

### Weaknesses
- “High stakes” is defined conceptually but the operational rubric and inclusion criteria are not fully specified. The perspective to value mapping and any inter annotator agreement are under reported.

- The default prompt is third-person, and the first-person version likely differs in more than pronouns. Even small lexical edits can change length, explicitness, or the presence of safety-loaded tokens. Since many frontier models apply safety policies that are sensitive to “I” or “you” phrasing, first-person prompts may disproportionately trigger refusals or policy-shaped outputs.

- The analysis spans Straightforward, Simple Contrast, and Swayed Contrast categories, which may vary in average length and in how explicitly values are named. If first-person variants are longer or contain more direct references to Safety or Self-Esteem, differences in steerability may reflect lexical salience rather than perspective per se.

- The steerability computation maps Yes/No/Ambiguous to 1/0/0.5 and then forms ratios b/a and c/d. If first-person prompts shift the base distribution closer to extremes, denominators can become small, which amplifies ratio variance and makes the first- vs third-person comparison sensitive to a few items.

- The paper filters to value pairs that occur more than 16 times, which it says is about the top 25 percent of all pairs. This immediately creates an imbalance: many value pairs are either excluded or represented by small n, while a handful dominate the statistics. Since CLASH has 345 dilemmas and 3,795 perspectives, the effective sample per specific pair can still be modest, especially once split by model family, size, or framing.

- Steerability ratios b/a and c/d can be unstable for rare pairs where base preferences cluster near 0 or 1. Without per-pair counts, confidence intervals, or multiple-comparison corrections, it is difficult to separate genuine scale effects from sampling imbalance.

- Perspectives are nested within dilemmas, so items are not independent. Ignoring this nesting inflates nominal sample size and can overstate significance.

### Questions
1. Can you replicate the first vs third person result with templates that only change pronouns and keep token length matched, and report invalid output rates and safety policy triggers per condition?

2. How are early commitment and overcommitment operationally detected?

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
3

### Summary
This paper developed the new benchmark dataset named CLASH which addresses a high-stake dilemma. They also introduced the concept of ambivalence, psychological discomfort, and value shift. They found that even leading LLMs can struggle with these traits. The authors also explored Conditional Steerability. Also analyzed first and third-person perspectives.

### Strengths
- This paper is really easy to follow, clearly presented. Well organized.
- Originality of proposing a new high-stake dilemma dataset 
- Significance may comes from introduce new concepts of measuring"Decision Ambivalence", "Psychological Discomfort", "Value Shifts"  in LLM and finding LLM having difficulties in understanding"Decision Ambivalence", "Psychological Discomfort", "Value Shifts" 
- The way that how they develop their dataset is well documented.
- Valuable human annotator contributions.(With concern of IRB statement)

### Weaknesses
- As this paper address high-stake situations, may have potentially contain and drive the research direction to sensitive domain.
- Human annotators/inspectors are included, potentailly require IRB (Institutional Review Board) consent?
- I don't see any weakness other than this.

### Questions
- It is interesting to introduce the ambivalence measure, but how can we actuallly know LLM is just avoiding to choose one of the answer or recognizing the "true ambivalence"?
- Your experiments were conducted using the 'reasoning process followed by the answer' prompt strategy. Was there any pilot study or literature-based discussion on the impact of reversing this order (i.e., 'answer followed by reasoning')?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CLASH, a dataset of 345 high-stakes ethical dilemmas with 3,795 character-based value perspectives, designed to evaluate LLMs' ability to navigate complex value-based decisions. The scale of the benchmark they develop is significant and the 11 kinds of character perspectives are thoughtfully designed. The authors then benchmark 14 models for their abilities to detect decision ambivalence, recognize psychological discomfort, and track temporal value shifts and also examines reasoning patterns in thinking models. The paper also introduces a "conditional steerability" analysis that looks at whether models can be steered towards one value over the other using their constructed character descriptions, relative to the models' base preference.

### Strengths
This paper presents a genuine effort to improve resources for evaluating value conflicts in high-stakes dilemmas with a benchmark of a useful scale. The dataset's scale and structured approach to creating different perspective categories enables controlled testing of specific value reasoning capabilities.

In particular, the inclusion of dynamic value shifts (Shift, Half-Shift, False-Shift categories) is innovative and addresses an unexplored aspect of value reasoning. The findings for RQ3 were most interesting to me and seemed most relevant to real-world problems cases i.e. understanding current LLMs' ability to track evolving values in multi-turn conversations.

The authors take care to include a number of analyses beyond reporting performance on their benchmark, such as the strategies used in reasoning chains and the steerability of models' values given the character descriptions. In particular, attempting to understand the strategies used in reasoning chains in this setting is novel and interesting given that previous work has only done this for mathematical reasoning. The authors make an effort to support many of their claims with statistical tests.

### Weaknesses
My main concerns about this benchmark relate to data contamination and the heavy use of GPT-4o for most of the relevant content. I see that efforts were made to address data contamination issue in Appendix B.1, but testing only 10 samples and using a 60% threshold with the same model that generated the content does not seem rigorous enough for scenarios that are directly taken from text that is very likely to be in the training data. Further, the use of GPT-4o for all the actions, rationales, and character descriptions raise concerns about whether this benchmark is testing whether models align with GPT-4o's interpretation of value-based reasoning. The high performance of GPT-4o (third by a small margin) seems like a further indication of this.

The authors justify the quality of each of these generation steps with high inter-annotator agreement, but as they point out in Appendix B.4, "The high inter-annotator agreement is expected due to our use of explicit character perspectives." For example, without a perspective, a question like “Who should receive the kidney transplant: the person with higher chance of survival or the person who waited longer?” invites diverse opinions. But if we specify that Character A prioritizes fairness over utility, it is clear that A would select the person who waited longer." I think this highlights a significant limitation of the benchmark -- conflicting values in such scenarios in the real-world are diverse and that's what makes this problem challenging. Relying on one model that is well-known to be subject to mode-collapse raises concerns that benchmark does not capture realistic features of real-world value conflicts or is a good test of other model families' abilities in this domain. 

In terms of the results, the claims made in section 4.2 seem unjustified given the lack of statistical tests, error bars, and the very minimal difference between the values in the graphs for these claims. For example, it is not clear from the top left subplot in Figure 4a that "successful value-laden reasoning exhibits less backward-chaining" though this might be true of verification. Similarly for "successful chains demonstrate greater emphases on pragmatic and rights-based ethics." This section would benefit from the addition of paired t-tests for these claims.

Section 4.3 is the weakest in my opinion, and could benefit from a much better justification for the complexity of mapping the rationales to such a wide space of values and then again mapping those to a different set of value frameworks. These mappings were again done primarily done using GPT-4o raising concerns about their accuracy. In Appendix F.3 the authors state that "Human evaluators indicate that 78.00% of the values are comprehensive, 98.77% are relevant to the value-related rationale, and the inter-annotator agreement measured with Cohen’s kappa score (McHugh, 2012) were 0.471 and 0.823, respectively." The fairly low assessment of comprehensiveness and the low inter-annotator agreement on this suggests that relying on GPT-4o to make complex value mappings does not form a trustable basis for the analyses in 4.3. Further, while the r=-0.243 (line 407) is significant it is still only a weak correlation.

### Questions
In addition to the points mentioned in the weaknesses, the paper could benefit from the following: 
- The results in RQ3 seem consistent with the overcommitment and early commitment strategies the authors find in the reasoning traces in section 4.2 which would be interesting if true, however this connection isn't made in the text. Breaking down the analyses of reasoning chain strategies by the static and dynamic cases to see if these strategies are favored in the dynamic settings and connecting these back to RQ3 could strengthen the results. 
- It would strengthen this benchmark to have content generated by more than one model family and evaluate models' performance across all these model families' data points.   

Minor: 
As the authors note in the related works, this setting is very related to moral decision-making studies that have been done in LLMs. The paper could better contextualize their contributions in light of these studies, especially more recent ones e.g. Cheung et al., 2025 (Large language models show amplified cognitive biases in moral decision-making).

### Soundness
2

### Presentation
2

### Contribution
3
