# LINGOLY-TOO: Disentangling Reasoning from Knowledge with Templatised Orthographic Obfuscation

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Frontier language models demonstrate increasing ability at solving reasoning
problems, but their performance is often inflated by circumventing reasoning and
instead relying on their expanding knowledge and memorisation capacity. We
introduce LINGOLY-TOO, a challenging reasoning benchmark of 1,203 questions
and a total of 6,995 sub-questions that counters these shortcuts by applying expert-
designed obfuscations to Linguistics Olympiad problems. These obfuscations
preserve the underlying solution logic while reducing the likelihood problems
are solvable with via knowledge or memorisation. Our experiments show that
models exploit shortcuts on the original question as performance markedly drop
upon obfuscation. Even the best reasoning models remain highly sensitive, with
scores dropping from around 0.59 on original problems to 0.48 after obfuscation.
LINGOLY-TOO disentangles reasoning from knowledge, offering a clearer measure
of true reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces LINGOLY-TOO, a 6,995-QA benchmark built by applying expert, grapheme-level orthographic obfuscations to 82 UKLO problems, aiming to disentangle reasoning from knowledge/memorization. The authors define clear metrics (Mog vs. Mobf, plus robust variants), run broad model evaluations, and provide validation via auditors and a human RCT. Results show sizable drops under obfuscation (e.g., top model ≈0.60→0.48), correlations with language resourcedness, and modest gains from guided reasoning.

### Strengths
* Clear problem framing: measuring reasoning when shortcuts (knowledge, contamination) are minimized is timely and important.
* Methodological originality: the reasoning-equivariant, linguistically-informed permutations are thoughtful and non-trivial; the Turkish vowel harmony example nicely motivates rule design.
* Strong experimental design: multiple families of models, bootstrap analysis, no-context control, tokenization controls, and human study make the case persuasive.

### Weaknesses
1. **Insufficient Failure-Mode Analysis**

   The paper documents performance declines but does not deeply probe *why* models fail on obfuscated problems (e.g., difficulty inferring morphemic patterns, inconsistent multi-step reasoning, or fallback to guessing). Please add a qualitative analysis of model outputs—contrasting common errors on obfuscated vs. original items—to connect performance gaps to specific reasoning deficits. Building on this, apply statistical tools (e.g., clustering) to categorize and quantify linguistic reasoning failures.

2. **Limited Accessibility of the Permutation Ruleset**

   The permutation rules (Appendix B) are dense and lack a high-level summary in the main text. A concise overview—such as a table of key constraints and invariances, or a concept diagram illustrating how reasoning equivariance is preserved—would make the method more accessible, especially to readers without a linguistics background.

### Questions
1. **On Failure Attribution (related to Weakness 1)**

   Can the authors extend the analysis to *quantify* the causes of reasoning failure? For instance, by labeling dominant error types and reporting their prevalence across models and difficulty levels.

2. **On Ruleset Comprehensibility (related to Weakness 2)**

   While we appreciate the authors’ effort on the permutation design, could the paper include auxiliary tables or diagrams that summarize the rules and constraints at a glance, to help non-linguist researchers quickly understand how reasoning equivariance is enforced?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces LingOLY-TOO, a new benchmark for evaluating reasoning abilities in LLMs. It is built from Linguistics Olympiad problems. The key innovation is the use of "reasoning-equivariant permutations" to obfuscate the problem text. This process changes the orthography but is carefully designed to keep the underlying logical structure and solution steps unchanged. The authors show that while models perform reasonably well on the original problems, their performance drops significantly on the obfuscated versions. This suggests that models rely on prior knowledge and memorization rather than pure reasoning on the original tasks. The paper also includes detailed analysis on the effect of language resourcefulness and a human study.

### Strengths
The core idea of the paper is excellent. Using orthographic obfuscation to create a "knowledge-free" test for reasoning is a very clever and direct way to tackle the problem of data contamination. This is a timely and important contribution.

The process for creating the permutations is very rigorous. I am impressed by the careful, manual design of the rulesets by experts to ensure the problems remain solvable. The validation by IOL medallists adds strong credibility to the method.

The experimental section is very comprehensive. The authors evaluated a wide range of models, including the latest reasoning-specific ones. The analysis goes beyond just overall scores to include "no-context" tests, robustness checks, and the correlation with language resources. The human study is also a valuable addition.

### Weaknesses
The use of exact match is simple but might be too strict. Sometimes, a model might have the correct reasoning but make a small mistake in formatting the final answer. Using only exact match could penalize such cases.


The human study shows a small but noticeable performance drop (5.7%) for humans on obfuscated problems. This suggests that the obfuscation itself might add some cognitive load, making the problems slightly harder to parse, even for humans who don't rely on prior knowledge of the language.

The process of creating permutation rulesets relies heavily on manual expert work. This might make it difficult to scale the benchmark to a much larger size or to adapt it quickly to other domains.

### Questions
1. Have you considered any evaluation metrics other than exact match (e.g., edit distance or partial credit schemes) that could capture instances where the model's reasoning is mostly correct but the final output has a minor error? What are the potential challenges in implementing such metrics for this benchmark?

2. The human study shows a 5.7% performance drop due to obfuscation. Could you discuss a bit more how we should interpret the model's performance drop in light of this? Specifically, how much of the model's drop might be attributed to the increased difficulty of processing the unfamiliar orthography, versus the removal of knowledge-based shortcuts?

3. The permutation rulesets are designed by experts. Do you think this method could be applied to other domains like mathematical reasoning or code generation? What would be the main challenges in designing "reasoning-equivariant permutations" for those domains?

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
This paper introduces a challenging reasoning benchmark of 7k question answer pairs, built by applying grammeme-level obfuscations to Linguistic Olympiad problems.  The motivation for building a new dataset is that models rely on prior language knowledge learnt via pre-training. This dataset is built to test out models' reasoning capabilities and removes any cues that could trigger memorized translations.

### Strengths
1. Release of a large-sized (7k) benchmark dataset that clearly separates the reasoning abilities from memorized knowledge.
2. In-depth analysis: The authors conducted various analyses, such as the ability of the model to reason, the effect of tokenization on uncommon characters, and the various across different permutations. 
3. Release of dataset and code

### Weaknesses
1. In the no-context setting, the difference between the original and the obfuscated dataset is very less; how does the author come to a concrete conclusion?
2. To check the effect of performance drop with uncommon characters, why don't we replace the context with random but real tokens that are not part of the training set? Similar to the ProntoQA dataset, where entities are replaced with false ontology.

### Questions
The authors manually created rulesets for this dataset, which makes it harder to extend this work to other datasets. Did the authors try to use LLMs as annotators to see how feasible it is to extend them to other domains/ datasets?

### Soundness
3

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
5

### Summary
This paper introduces LingOly-TOO, a challenging reasoning benchmark that obfuscate Linguistics Olympiad Problems to avoid advantaging models that are using shortcuts such as memorisation and knowledge. The obfuscations preserve the underlying solution logic while removing orthographic clues that could trigger patterns from memorisation or knowledge. Without surprise, the performance of models drastically decrease.

The authors defined knowledge as information stored in model parameters after training, which captures linguistic, factual, and commonsense patterns useful for downstream tasks, and memorisation as when models exploit contaminated datasets, reporting answers previously seen in training

The authors adapted 82 problems from the UKLO, and obfuscate the problem to avoid models relying on linguistic patterns. More specifically, the authors manually created a ruleset for each problem to generate valid permutations of targeted tokens. They apply extra-care to keep names of people, sacred places, etc intact. Overall, the authors generate 6 valid permutations per problem and generate obfuscated versions by altering the problem text. Overall, there are 6995 question-answer pairs.

The authors propose two metrics: the average exact match score across all questions in all permutations and the average exact match across all questions in the unpermuted problem.

In terms of experiments, the authors evaluate 15 reasoning models. The performance varies between the original and obfuscated variants, with GPT and Claude being the most robust models. The authors do a detailed analysis to measure the gap between reasoning and knowledge. It would be great to add a case study to support the findings. In terms of metrics, it would be good to compare human performance on the task and include more details (e.g., how do they compare with LLMs) - I'm aware of Appendix L but would like to see more. Moreover, it would be fairer to assess the performance of models with pass@k.

Overall, I am lukewarm about this paper. Creating obfuscating variants of problems does not seem to relate to real-life tasks, even for measuring reasoning. I have the feeling that this benchmark is focusing on a unrealistic problem / not a real problem. I appreciate the analysis and experiments of the authors, and the paper is very well written and structured.

### Strengths
- Various models are used in the benchmark.
- Detailed analysis.

### Weaknesses
- I have the feeling that this benchmark focuses on an unrealistic problem / does not represent real life tasks. We cannot expect LLMs to necessarily perform good on those.
- Please add human evaluation + pass@k metrics

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2
