# From Human-Level AI Tales to AI Levelling Human Scales

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
What does ``human-level'' mean when model scores come from heterogeneous benchmarks? Or when the human data comes from a W.E.I.R.D. distribution? 
Can we place AI on a more comprehensive human-referenced scale? 
To have insight and progress on this question, we first work with ratio scales using difficulty in $L$-units, from the probability of success of the whole world population $p_W$ on each item. 
Each level is defined by $L = -\log_{B} p_W$ (so $L=0$ $\approx\$ near-universal success, $L=1$ $\approx$ 1-in-$B$, $L=2$ $\approx$ 1-in-$B^2$, etc.). 
Then we compile publicly released test items spanning education and reasoning benchmarks (PISA, TIMSS, ICAR, PR, and ReliabilityBench), annotating each with capability scales (reasoning, attention, volume, etc.). 
The estimation of $B$ and location of anchor questions is done by extrapolating from a biased source sample (characterized by its demographics and other known information of how it was obtained) towards a larger target population (with a new demographic profile) using LLMs, with the hypothesis that they condense vast amounts of demographic data during their training. 
We explore different prompting mechanisms and ways to specify source and target distributions and evaluate their quality using group slicing on some of the datasets and post-stratification. 
The techniques introduced here allow for the definition of calibrated scales from which we can standardize AI measurements relative to the world population, and scalable `equating' of human populations in the social sciences.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes an ambitious framework for calibrating AI evaluation on human-referenced, population-anchored scales. Building on psychometrics and measurement theory, the authors argue that “human-level AI” comparisons are unreliable due to heterogeneous benchmarks and biased human baselines. They introduce an automated methodology using LLMs to extrapolate item difficulty and success rates from limited human samples (e.g., PISA, TIMSS, ICAR, UK Biobank) to the estimated world population, yielding unified, logarithmic capability scales for comparing AI and humans. The paper demonstrates feasibility through preliminary validation, suggesting that such “human-anchored” metrics could make AI evaluation more commensurate across domains.

### Strengths
1. The idea of constructing population-calibrated, commensurate capability scales for AI–human comparison is novel and intellectually stimulating.

2. The paper bridges psychometrics, measurement theory, and AI evaluation, offering a new lens for understanding “human-level” claims.

3. Using LLMs for demographic extrapolation and capability annotation is an interesting and technically clever direction that may inspire further research.

### Weaknesses
1. The methodology is described in broad, conceptual terms, with many unverifiable steps (e.g., LLM-based demographic extrapolation). The pipeline feels speculative rather than reproducible or theoretically grounded.

2. The experiments are limited to a few datasets and report only correlation-based metrics **without strong evidence that the proposed scales produce more meaningful or reliable comparisons than existing methods**.

3. The paper reads more like a position or conceptual essay (similar to a measurement-theory manifesto). It lacks formal models, quantitative ablations, and clear algorithmic contributions expected at ICLR.

### Questions
None

### Soundness
1

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
4

### Summary
This paper presents a framework for evaluating LLMs' capabilities on a unified, World-Wide Population (WWP)-anchored scale. It aims to move beyond comparisons to narrow human samples (e.g., specific age cohorts or WEIRD populations) or heterogeneous benchmarks. The core contribution is a five-stage pipeline combining a criterion-referenced capability framework (ADeLe) with LLM as an extrapolation tool for demographic adjustment. It takes the known success rate of a test item on a small and biased human sample as the source group, and predicts the success rate for the entire 2025 WWP. This predicted WWP success rate is then transformed into a single, commensurate difficulty scale. The authors validate the LLM's extrapolation ability on ICAR and TIMSS. They argue that such calibrated scales are standardized and hence more suitable for reporting "human-level" performance.

### Strengths
Originality: The paper addresses a very relevant and difficult problem in AI evaluation: the lack of a standardized, non-biased human baseline. The approach of using an SOTA LLM as a demographic extrapolation tool is quite original. 

Significance: If the proposed method were proven robust, it would be significantly impactful. It introduces the concept of commensurability to AI benchmarking, allowing a reviewer to meaningfully compare a model's performance on a high-school math test (TIMSS) with its performance on a fluid intelligence measure (ICAR), all using the same human-anchored ruler.

Clarity: The paper is generally well-written and clear. The authors do an excellent job of articulating the problem of "incommensurate" benchmarks and clearly detailing the five-stage pipeline. 

Quality: The paper demonstrates a good understanding of psychometrics. The experimentation is thorough, using multiple LLM models and two validation datasets.

### Weaknesses
1. The paper describes the ADeLe annotation process but does not provide any statistical distribution or visualization of the resulting 18-dimensional capability profiles. This makes it difficult to assess the quality or coverage of the initial scale-definition step.

2. The LLM's performance for extrapolation seems brittle. The validation on homogeneous data (ICAR) was successful, but the performance severely degrades on the more complex, heterogeneous TIMSS. Furthermore, the full "subgroup-to-WWP" calibration step failed for most models on TIMSS, indicating the method is not robust for general and cross-distribution extrapolation.

3. Using a WEIRD-trained LLM as the correction mechanism means the resulting WWP scale inherits the LLM's stereotyped model of global demographics and cognitive capability. The scale risks trading one form of human bias for another.

4. LLMs are tested in a simple, out-of-the-box fashion. There is a lack of interpretability regarding how the LLM reasons through the demographic adjustment. This makes the reported numerical results seem shallow.

### Questions
1. Given the failure on TIMSS, can the authors justify the use of the WWP-calibrated results derived from the other, non-validated benchmarks (PISA, UK Biobank), or should those results be treated as speculative?

2. What steps were taken to ensure high inter-rater reliability during the ADeLe annotation? Were multiple human annotators used, and what were the resulting Cohen's Kappa scores? Please also include a figure or table showing the distribution of the capability levels across the datasets. 

3. Could the authors include an appendix detailing the demographic information provided to the LLM for the WWP? This is necessary to understand the "input" that supposedly guides the LLM's prediction.

### Soundness
3

### Presentation
3

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
The submission proposes a methodology that attempts to obtain a multifaceted measure of AI versus human performance using large-scale datasets derived from human performance on certain standardized scholastic and psychometric tests. The goal of the paper is to obtain a standardized scale for each measure of intelligence that is derived from the fraction of humans that would solve a given task correctly. Since any given task may require different aspects of intelligence, it is also necessary to annotate every task with the demand it places on each aspect intelligence. This annotation is based on publicly available rubrics designed in prior work on measuring multifaceted aspects of intelligence. Finally, for any given dataset based on human performance on scholastic or psychometric tests, there is usually significant demographic bias. Hence, the submission proposes a method to extrapolate from demographically biased samples to the true demographic distribution of the whole world population. 

Both the annotation of tasks with the demands on intelligence and the extrapolation to the world population are done with LLMs. For annotation the LLMs are prompted to follow the publicly available rubrics. For extrapolation, LLMs are directly prompted to predict the world population performance on a task given the sample performance along with demographic information about the world population and the sample population. Experiments with subsamples of large datasets are used to measure how well these LLM assisted annotation and extrapolation steps work empirically.

### Strengths
Building large-scale human-normalized measurements of LLM cognitive performance is an important question, and the idea of using LLMs to assist in key aspects of building such a novel measurement is interesting.

### Weaknesses
1. The empirical results seem quite mixed, LLM demographic extrapolation did not seem to perform consistently well across different datasets.
2. The empirical setup lacks adequate baselines. Extrapolating from a biased sample population to a larger population with known demographics is a very well-studied problem in statistics. For instance, all effective political polling uses statistical methods to make such adjustments based on predicted voter turn-out rates. The use of a LLMs to make such predictions based on demographic information needs, at a minimum, to be compared to the standard techniques from statistics that are designed to accomplish the same task.
3. The writing and organization of the paper are a significant impediment to understanding. There are many convoluted sentences and undefined terms that the reader is forced to struggle with in the beginning of the paper. The related work section is also quite strange, with an entire paragraph about how measurements of physical quantities arose in much earlier historical context, but no mention of the (highly relevant) standard statistical methods used to extrapolate from a biased sample based on demographics.

### Questions
How would standard methods like importance weighting based on demographic data perform on the extrapolation task? 
Why would one expect LLMs to outperform rigorous statistical methods for what is fundamentally a problem of basic statistics?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To address misleading comparisons between AI and narrow human benchmarks, this paper introduces a framework to calibrate AI performance against the entire world population. The core contribution is a novel technique that uses Large Language Models (LLMs) to extrapolate data from existing, biased source samples (like PISA or UKBioBank) to this broader, global scale. This methodology allows for the definition of "calibrated scales," enabling AI evaluations to be meaningfully standardized relative to the global population rather than unrepresentative test groups.

### Strengths
1.The paper tackles a recognized and significant flaw in AI evaluation: current comparisons to "human level" are often misleading due to heterogeneous benchmarks and narrow, unrepresentative human baselines (e.g., W.E.I.R.D. populations). The motivation to correct this is well-justified and highly relevant.
2.The paper proposes an ambitious framework to move beyond fixing local benchmarks, aiming instead to calibrate AI evaluation onto a single, commensurate scale anchored to the "world population". The core novel method is the use of LLMs as "demographic extrapolators," leveraging their condensed knowledge to map biased source samples to a global distribution.
3.The study provides empirical validation of its approach. The results on ICAR dataset are particularly strong, showing that the LLM extrapolation from subgroups to the full sample achieves very low MAE (0.03-0.04) and extremely high correlation (Pearson r > 0.92).

### Weaknesses
1.You have pointed out the core methodological gap in the paper. The experiment's "validation" setting only proves that the method can extrapolate from a biased subgroup to that dataset's full population (e.g., from "males aged 20-30 in ICAR" to "all participants in ICAR"). However, that "full population" from the dataset is still a biased sample and is not representative of the "whole world population" (WWP).Therefore, the experiment cannot validate the final, most important step: the "calibration," which attempts to extrapolate from that biased dataset to the ideal WWP. This final step remains an unverifiable assumption because no ground truth exists for the WWP to measure against.
2. A significant weakness of this methodology is its fundamental reliance on the Large Language Model's intrinsic capability to perform demographic extrapolation . This introduces a critical and potentially circular flaw: the LLMs themselves are trained on datasets that are notoriously biased and in no way representative of the "world population." The vast majority of their training data is sourced from the internet and is heavily skewed toward English-language content and WEIRD (Western, Educated, Industrialized, Rich, and Democratic) populations. Therefore, when the method prompts the LLM to estimate the success rate for the "whole world population" by accounting for global factors , it is not drawing from a neutral, ground-truth understanding of humanity. Instead, it is highly likely that the LLM is projecting its own inherent training biases onto the extrapolation. This "bias-in, bias-out" problem undermines the core claim, as the method risks simply replacing the known bias of the source sample with the unknown, but significant, bias of the LLM itself.

### Questions
1.Could you provide a more in-depth diagnostic analysis explaining why the model performs poorly when extrapolating TIMSS data? Is this failure due to content heterogeneity, cross-cultural differences, or a lack of specific data in the LLM knowledge base? More importantly, if the method fails when dealing with complex, heterogeneous data, does this fundamentally undermine its core value as a general calibration tool to address current benchmark heterogeneity issues?
2.Can the authors provide additional theoretical or empirical evidence (e.g., using large datasets as simulations of “proxy worlds”) to demonstrate why we should believe that success on the “verification” task (as shown by ICAR) can translate into accuracy on the final, completely unverifiable “calibration” task?
3.When prompted to consider factors such as global age distribution, education level, and health status, how can we be sure that LLM is not "guessing" or "adjusting based on stereotypes," thereby introducing a new, systematic bias? Can you demonstrate that the results "calibrated" by LLM are closer to the true "global population" than the original biased sample, and not just "another form of bias"?

### Soundness
2

### Presentation
3

### Contribution
2
