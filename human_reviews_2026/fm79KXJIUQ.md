# EigenBench: A Comparative Behavioral Measure of Value Alignment

- Decision: Accept (Oral)
- Scores: 4, 4, 10, 6

## Abstract
Aligning AI with human values is a pressing unsolved problem. To address the lack of quantitative metrics for value alignment, we propose EigenBench: a black-box method for comparatively benchmarking language models’ values. Given an ensemble of models, a constitution describing a value system, and a dataset of scenarios, our method returns a vector of scores quantifying each model’s alignment to the given constitution. To produce these scores, each model judges the outputs of other models across many scenarios, and these judgments are aggregated with EigenTrust (Kamvar et al., 2003), yielding scores that reflect a weighted consensus judgment of the whole ensemble. EigenBench uses no ground truth labels, as it is designed to quantify subjective traits for which reasonable judges may disagree on the correct label. Hence, to validate our method, we collect human judgments on the same ensemble of models and show that EigenBench’s judgments align closely with those of human evaluators. We further demonstrate that EigenBench can recover model rankings on the GPQA benchmark without access to objective labels, supporting its viability as a framework for evaluating subjective values for which no ground truths exist. The code is available at https://github.com/jchang153/EigenBench.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents EigenBench, a novel method to quantitatively measure subjective value alignment in language models. The system uses an ensemble of models to judge each other's responses to scenarios against a given "constitution". By aggregating these peer judgments using the EigenTrust algorithm, EigenBench produces a consensus alignment score for each model without requiring any ground truth labels. The authors validate their method by showing its alignment with human judgments and its ability to successfully recover objective model rankings on the GPQA benchmark, suggesting its effectiveness for evaluating subjective values.

### Strengths
The core problem addressed in this paper is how to quantify subjective values, which I believe is a very important question.

The EigenTrust algorithm, which relies on mutual evaluation among a group of models, is a very interesting idea.

The paper also considers how to reduce bias during the data collection process and conducts an analysis of the consistency between evaluation models and human annotators.

### Weaknesses
The approach in this paper, which relies on models evaluating each other, seems highly dependent on the population; models with different capabilities from different groups appear difficult to compare horizontally. Moreover, for stronger models, being evaluated by weaker ones seems somewhat unfair, which may lead to underestimated evaluation results.

When introducing a new model, the evaluation cost of the proposed method seems to be much higher than that of direct evaluation.

The fact that only two human participants were recruited for human evaluation is indeed far from convincing.

### Questions
Why do we necessarily need this population-based evaluation approach? How much benefit does this method actually offer compared to direct evaluation, CoT evaluation, or finetuned Judge model? The paper lacks an analysis of the performance and differences among them. I’m also very curious whether using the strongest model with different prompts for evaluation would yield better results than the current evaluation method.

Does the ranking discrepancy between model self-evaluation and EigenBench evaluation directly indicate that self-evaluation is inferior to the proposed method (Section 5.1)? It seems that such a conclusion would require comparison against another ground-truth ranking, which might call for increasing the number and diversity of human annotators.

### Soundness
3

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
This paper introduces EigenBench, a novel black-box method for benchmarking language models' alignment with specified value systems using peer evaluation and EigenTrust aggregation. The core innovation lies in having models judge each other's responses to scenarios according to a constitution, then aggregating these judgments through the principal eigenvector of a trust matrix. The method is validated through human judgment comparison and a creative GPQA experiment. While the approach is intellectually interesting and addresses an important problem in AI alignment, the paper has significant methodological concerns regarding circularity, limited experimental scope, and unclear practical utility.

### Strengths
- Novel and Well-Motivated Methodology.
The paper introduces a creative solution to measuring subjective alignment by combining peer evaluation with EigenTrust aggregation, where alignment scores are computed as the left eigenvector of a trust matrix. The low-rank Bradley-Terry-Davidson model with vector embeddings elegantly captures multiple dimensions of subjective interpretation beyond scalar rankings. The double-blind design and judge scaffold with reflection steps reduce gaming potential and judgment biases.


- The GPQA experiment ingeniously validates the method by recovering objective rankings (Kendall τ ≈ 0.77) without ground truth labels, demonstrating viability for subjective domains. Robustness analysis thoroughly tests sensitivity across scenario distributions, constitution wording, and population changes. Extensive appendices with full prompts, model IDs , and implementation details demonstrate strong commitment to reproducibility.

### Weaknesses
- Technical Correctness: Equation 1 defines alignment scores, where higher-scoring judges (ti) weight trust more heavily. The premise in footnote 4 that "a model whose behavior aligns better with C is also a better judge of whether others' behavior aligns with C" is assumed but not validated. This creates a self-reinforcing loop that may amplify initial biases rather than reveal true alignment.

- Evaluation Scope: Only 8 models tested in main experiments, mostly frontier models from major labs (Table 4, lines 540-574). Missing: open-source models, fine-tuned variants, models from smaller organizations.

- Constitution design: Constitutions are LM-generated with minimal human iteration. The robustness test shows some stability, but only examines 5 variations of one constitution generated by different models.

- Human validation severely limited: Only 2 human judges with ~50 scenarios each. This is insufficient to establish reliability.

### Questions
- The core premise ("aligned models are better judges") is untested. Could you add an analysis where you correlate a model’s score \(t_i\) with the accuracy of its judgments (e.g., how often its evaluations match human judgments for the same value system)? This would validate whether the EigenTrust weighting is justified.


- The scalability analysis is limited. Could you add a plot of "dataset size vs. score stability" for 10–50 models? This would help users estimate the resources needed to apply EigenBench to larger model populations.

- Fine-tuning application. You mention "character training" as an application. Have you actually demonstrated that EigenBench scores improve when a model is fine-tuned for alignment? This would be a compelling validation.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper contributes EigenBench, a black-box method for comparing LLMs' adherence with a particular set of constitutional principles. Namely, with an ensemble of models, models rate eachother according to a constitution which is hidden from the models to be evaluated, and which fit the data to model-specific ELO scores and judge dispositions. The authors compare a set of frontier models across the three tested constitutions on three datasets. Finally, to test the robustness of the method, the authors test compared to human judgments, cases with ground truth labels (GPQA), and modified constituiton and population.

Overall, the work presents a compelling way to measure models' values, a problem that (in my opinion) has not yet been well tackled in the literature - a strong contribution.

### Strengths
S1: The problem of measuring principled model value trade-offs is (in my opinion) a very important problem that (in my opinion) no work has tackled in a completely satisfying way so far - until this paper. I found the method to be well-principled, and well stress-tested in sections 5 and 6. I think that this will be very impactful for the community.
S2 The experiments are quite extensive and use a variety of recent models.
S3: I quite like the estimated measure of the model disposition from the LM vs the persona prmopt in 4.2! I've been curious about an experiment like this for a while, but haven't seen a paper convincingly measure something in this direction. Very nice.
S4: The human evaluation experiment in 5.2 is a creative and useful for measuring variance / doing interrater reliability between models and humans.
S5: The GPQA experiment's strong results provide strong evidence for the model comparisons being meaningful.
S6: Overall, the paper was very well-written.
S7: Fig 1 was very well made and helped me build intuition on the method very quickly.

### Weaknesses
W1: While interesting, I found figure 2 a little bit confusing as to what exactly we are meant to take away from it in relation to the rest of the paper. If additional commentary connecting it to the main text were added, that would be very interesting.
W2: The entire suite of experimetns utilizes at least some proprietary models, making reproducibiltiy difficult over time. 
W3: The paper has little discussion of limitations and directions for future work. I would appreciate it if this were fleshed out in a future camera-ready version.
W4: The clarity could be moderately improved with additional figures / tables presenting some of the results, to augment the discussion of many results which are presented in the main text.

Typo/grammar nit:
- L343 should be "as fewer scenarios" instead of "as less scenarios"
- I believe that there's an extra parentheses at the end of L447 - "(Kirk, 1993)^10)"
- If the authors wish to save some space, Algorithm 1 could be trimmed to half-width to allow more room for additional discussion in the main text. (very optional).
- In figure 2, it might be more helpful if the legend were alphabetized by eg first name - I had to spend some time to try to map a data point to the legend as the order was not clear to me

### Questions
Q1: What do the authors think it means that such a small dimensionality (d=2) performs so well? Some intuition on this in the main-text would be very interesting

### Soundness
4

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
The paper presents EigenBench, a black-box method for comparatively benchmarking language models’ values. The study formulates a trust matrix from natural language critique to rank the alignment of the models. It uses a set of LLMs that act as both judge and candidate along with a judgement criteria called constitution and a set of prompted scenarios. The proposed method first obtains a trust matrix and then obtains a trust vector t to benchmark the models.

### Strengths
- The proposed method is sound to measure the alignment of LLM for a given value system
- The method is discussed both theoretically and empirically. The community aggregation method, utilizing the Bradley-Terry model, aggregates the results for all judges. Employing eigenvalue to rank LLMs is technically sound and draws inspiration from PageRank.
- The paper is clear and well written. The results are also discussed in depth, and the inclusion of the human trust vectors in the study is valuable
- EigenBench generates rankings that are aligned with ground truth without explicitly being provided with ground truth labels, demonstrating its potential to rank subjective traits or values.

### Weaknesses
- The EigenBench scores assume that the models are capable of judging a criterion $C_k$. The study presents no evidence supporting the premise in footnote 4. According to appendix section C, the authors added a reflection step for a judge before developing a preference.
- The EigenBench scores assume that the models possess the capability to assess a criterion $C_k$ discussed in footnote 4. However, the study fails to provide any evidence that substantiates this premise. As per appendix section C, the authors incorporated a reflection phase for a judge before developing a preference, indicating that additional consideration is required if LLMs are used as judges.
- Experiments are limited to closed-source LLMs with the exception of Deepseek v3. Evaluating the results for open-source would strengthen the contribution of the paper, but it is uncertain whether open-source models would be able to act as competent judges as compared to the closed-source LLMs employed in this study.
- The experimental evaluation is constrained by the absence of comparisons to established alignment benchmarks. For instance, if the method produces a ranking for models aligned to a constitution C, does that ranking correlate with rankings on other benchmarks? Without such cross-benchmark analyses, the paper’s contribution is limited, making its generalizability also limited.
- Quantifying the success of the fine-tuning process using EigenBench is an expensive process, as it requires sending multiple requests to multiple LLMs at predetermined intervals and subsequently fitting the BTD model.

### Questions
- What is the effect of the number of judges? Does a higher number of judges produce an accurate ranking, and can it be approximated with fewer judges?
- Why the method outlined in Appendix C is not employed, despite its potential to provide a larger dataset for comparative analysis?
- Many core traits valued in practice, including reliability, accuracy, safety, and trustworthiness, are objectively measurable. Please specify some of the "most highly-valued" traits you regard as subjective, as indicated in the introduction section, and how your method captures and measures those specific traits.
- Not a question, but it is difficult to read figure 2. Please consider using either different shapes or a color-blind-friendly color palette.

### Soundness
3

### Presentation
3

### Contribution
3
