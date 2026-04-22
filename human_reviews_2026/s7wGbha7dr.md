# NUBench: A Benchmark for LLMs' Sentence-Level Negation Understanding

- Avg Score: 2.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2

## Abstract
Negation is a fundamental linguistic phenomenon that presents ongoing challenges for Large Language Models (LLMs), especially in tasks that require a deep understanding of semantics. Existing benchmarks often treat negation as a minor aspect within broader tasks, such as natural language inference. As a result, there is a lack of benchmarks specifically designed to evaluate negation comprehension. In this work, we introduce **NUBench**—a novel benchmark explicitly created to assess sentence-level understanding of negation in LLMs. NUBench goes beyond simply detecting surface-level cues by contrasting standard negation with structurally diverse alternatives, such as local negation, contradiction, and paraphrase. This benchmark includes manually curated sentence-negation pairs and a multiple-choice dataset, enabling a thorough evaluation of models' understanding of negation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces NUBench, a benchmark designed to evaluate LLMs understanding of sentence-level negation. The authors provide a formal definition of "standard negation" grounded in sentential logic, create a manually curated dataset with sentence-negation pairs (for finetuning) and multiple-choice questions, and evaluate various LLMs (2-3B to 8B and some proprietary) using both prompting and supervised fine-tuning approaches, and provides detailed error analysis, revealing that models frequently confuse standard negation with local negation.

### Strengths
- The dataset is constructed with strong linguistics grounding.

### Weaknesses
- The authors provide lots of details on the linguistics grounding and construction process, but failed to state what is the task (and the goal of the benchmark). Figure 1 shows an example. The instruction is simply just "negate the sentence", and the answer is A so does the benchmark only test for standard negation? Also, what if we only negate only one main verbs, e.g. "Batts aren't typically used in the walls"? In my opinion, this sentence is still negated. Please clarify more on what exactly is the benchmark intended to test.

- The claim that NUBench deal with "sentential logic, moving beyond the cue-based and often ambiguous accounts of prior work" is strange. Negation itself modifies sentence-level semantics, and the previous benchmarks are in fact sentence-level. The paper mentions NegNLI, MoNLI, and NaN-NLI but doesn't provide detailed comparisons. If the focus is only on the task framing (generative models), we can easily do that for previous benchmarks too.

- May not have impact beyond the CL/NLP community. The finding that models still struggle with different types of negation is not new.  
While the authors included an error analysis, there's insufficient analysis of why models make specific errors.

- Closed model selected are also small variants (mini) instead of the larger, more capable versions.

### Questions
- Models perform SFT for the option setting and some models (Gemma-1.1-2b-it) produce 1,239 format errors after SFT. This suggests that SFT is not properly done. Did the authors ensure that the finetuning setting is sound and working as intended? Also, why not SFT using the multiple-choice format?

- Why does option-selection evaluation yield such different patterns from completion-based?

- The fact that simple prompt works better than definition and detailed prompt is questionable. Is it because the samples are very long and contain many main verbs?

### Soundness
2

### Presentation
2

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
The paper created a new benchmark for evaluating sentence-level negation understanding in LLMs called "NUBench". The task is a simple multiple-choice question where given a sentence (without negation), the model has to pick the standard negation among the options (local, contradiction, and paraphrase). The paper formalizes standard negation with logical operations and provides a taxonomy on negation (relative clause, participle clause, etc). The results show that the performance on the benchmark increases with more few-shot examples and SFT.

### Strengths
* The discussion on operationalizing standard negation across conjunction/disjunction/conditionals is valuable towards understanding negation and consistency.
* Comprehensive evaluation experiments (zero-shot, few-shot, SFT) and inclusion of multiple models from different families, and an error analysis uncovering that more examples result in lower error rates and an unusually high error on selecting the paraphrased option (which doesn't even have negation)
* The creation process is robust with most of the creation being done by authors themselves (standard/local negation), which are challenging for LLMs, and the generated ones are refined by humans (authors).

### Weaknesses
1. The inter-annotator agreement and human performance on the benchmark are not reported. Adding the performance of an independent human on the test would help validate the benchmark creation process and better interpret model performances.
2. The related work is broad and doesn't explain how NUBench is different than previous negation benchmarks like CondaQA, LAMA-Neg, NLI-neg, etc. 
3. It is not exactly clear to me what this benchmark evaluates and whether there exists a single answer to the question of "negate this sentence". For example, "John went to the shop and bought an apple" can be negated into "John didn't go to the shop or didn't buy an apple" or "John went to the shop but didn't buy an apple". The latter sounds more natural to me, but my understanding is that the benchmarks is looking for the former one. Addressing W.1 should help understand this too. From example in Figure 1, I don't think there's a single way to negate the given sentence in a natural way without more context.
4. LLMs are known to avoid generating/picking unfactual statements because of their safety filters. Given that one of the sources is WikiPedia, there's a high chance that the negated statements are not factual anymore and this would explain the high error rate on picking paraphrases (which should be still factual). Have you verified the factuality of the negated sentences in the dataset? Adding a discussion on this would help understanding errors.

### Questions
1. Have you tried prompting LLMs to negate the sentences (without having access to any options) for a subset of NUBench and evaluating the performance and error types?
2. In L.352 you mention that an OpenAI API was used to generate contradiction and paraphrases. Which model was used to do this?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors introduce NUBench, a negation benchmark that goes beyond cue based negations. The benchmark is manually curated, consisting of 2 types of datasets: the sentence-negation pair dataset and the multiple choice dataset. It consists of both training and evaluation sets, with evaluation being done in a completeness setting (predicting log probs) as well as a multiple choice setting (which is needed for non open souce LLMs given lack of access to log probs). 

The authors report results using zero shot, few shot (demonstration examples) and supervised fine tuning (using the training set specified in the benchmark) on both the completion based as well as multiple choice setting across a variety of open as well as non-open source models.

### Strengths
* Negation is an important area to focus on. LLMs need to be tested more on negation since its a complex linguistic phenomena. The topic of the paper is well chosen.
* The paper is well written, and the theoretical bases for certain decisions (like types of negations and categorization) is well explained. These writing is especially important since a lot of readers aren't native to linguistics.
* In general, the approaches to collect data through humans (and failure to do properly through LLMs) is explained decently.

### Weaknesses
* I am worried about the soundness of either the dataset or the techniques given the fact that fine tuning barely moves the needle and in some cases has a negative effect. Its good they didn't observe this in the completion based setting. Is this because its harder to overfit on logits v.s a more direct text evaluation? I think some perplexity curves of training v/s dev and more analysis is absolutely needed here.
* For this paper to be considered ICLR level, we need a very thorough analysis of the results, gains and weaknesses of various models.This needs to take center stage, otherwise its hard to legitimize this work to the community.

### Questions
* Why the authors didn't make the decision to propose an evaluation only benchmark? What is the rationale here?

### Soundness
2

### Presentation
3

### Contribution
2
