# Benchmarking Large Language Model Benchmarks: Popular Benchmarks vs. Human Perception

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 2

## Abstract
Benchmarks play a critical role as a measure of large language model (LLM) capabilities. However, whether LLM performance on benchmarks is similar to their real-world performance, especially human perception of their outputs, remains questionable. This study specifically focuses on whether \textbf{LLM performance on benchmarks is similar to human perception}. The study investigates this gap by quantifying the similarity between LLM rankings derived from benchmarks and LLM rankings generated from human votes on the prominent LMArena platform. It systematically compares benchmark rankings against rankings in corresponding task-specific categories in LMArena for over 100 top-tier LLMs. The findings reveal that LLM performance on several popular benchmarks has low similarity with human perception, even though these benchmarks are more recent or challenging. The results highlight limitations in current benchmarking practices and underscore the need for evaluation frameworks that more accurately reflect the human perception and real-world performance of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper attempts to understand whether standard and popular benchmarks correlate with human perception, i.e., if the performance of these benchmarks is proportional to the "holistic goodness" and "overall" performance of an LLM. For this, the authors correlate the performance of an LLM on these benchmarks with the corresponding split from LMArena, which is an open leaderboard of LLMs as judged by humans via pairwise ranking (ELO score). The main comparisons are along QA, coding, math, and instruction-following. The main takeaway from this paper is that most of the popular and hard benchmarks reported with SOTA LLMs have low similarity with human perception.

### Strengths
- The paper addresses a very interesting problem of "whether standard benchmarks are enough for holistic evaluation of LLMs as perceieved  by humans". I personally appreciate this research direction, given the number of LLMs being produced with no solid and uniform holistic evaluation scheme that correlates with human perception and requirements.

### Weaknesses
- Overall, I find the paper quite poorly written, rather bland, and limited in both scope and analysis in its current form. It seems to contain a lot of filler content (e.g., lines 218–252), which provides redundant information more suitable for an appendix. I believe the authors could include more substantive core content if the paper were streamlined.

- Many of the hypotheses and claims in the paper rely heavily on the assumption of a "strong alignment" between LMArena and human perception across all tasks. While I acknowledge that the authors have mentioned this limitation and provided some justification, I believe that this assumption requires more corroboration. 

- I am also surprised that the authors did not discuss benchmark contamination in Section 4.3 ([Ahuja et al., 2024a](https://aclanthology.org/2023.emnlp-main.258/); [Ahuja et al., 2024b](https://arxiv.org/pdf/2410.16186)). Contamination in LM-Arena is less likely due to its dynamic nature—new questions and tasks are continuously added—whereas standard benchmarks are static and more prone to such issues.

- Much of the paper’s results read like table-to-text descriptions of the correlation metric, without deeper analytical insights. The whole analysis and findings are based on the correlation between rankings from LMArena and standard benchmarks, which makes it unidimensional at the moment. 

- I believe the paper would benefit from a more concrete analysis of contamination effects, inclusion of a broader range of models, comparisons across model sizes and families, and differentiation between “thinking” and “non-thinking” modes. Additional analysis supporting the conclusions of Section 4.1 would also be valuable, as prior studies have shown significant divergences between LLM-judge and human evaluations. In its current state, the paper feels more like a technical report than a comprehensive research study.

- Was there any analysis of overlap between LM-Arena tasks and standard benchmarks? If so, similar or intersecting problems should be excluded, as they could lead to inflated correlation scores.

- Finally, the paper lacks an analysis or visualization comparing the rankings of LLMs on standard benchmarks versus LM-Arena results (not with a simple correlation coefficient). Including such a comparison would strengthen the paper significantly.

### Questions
- It would be better to report the $R^2$ value for the regression lines in Figures 2 and 3 to provide a clearer measure of fit.

- The finding in Section 4.4 seems somewhat obvious, as math problems are objective and have a single fixed answer. Math problems in LMArena and any standard benchmark like GSM are very similar in terms of problem types. In this context, analyzing the intersection between LMArena and math benchmarks is crucial, as they may share some problems.

- Please provide the exact date on which LMArena was accessed to ensure reproducibility, given that the leaderboard and question bank are dynamic

- I am also unsure of the correlation comparison, as LMArena rankings are calculated via ELO (pairwise battles), while the paper just computes accuracies and ranks them.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper examines the correlation in model rankings between LMArena and 24 benchmarks. The authors compare the rank correlation of LMArena and (1) different benchmark categories (QA, math, code, and instruction following), (2) benchmark difficulty, and (3) benchmark release date.

### Strengths
The topic of the paper is interesting, the writing is reasonably clear, and, to the best of my knowledge this work is original in considering rank correlations for LMArena.

### Weaknesses
The main limitation of this work lies in the significance of its contributions and the soundness of its claims.
The authors show that the rankings of some popular benchmarks differ from those of LMArena. However, this is to be expected. As noted in line 127, the authors correctly point out that even benchmarks that appear very similar can exhibit substantial disagreement. If the authors had found that **most** popular benchmarks were uncorrelated with LMArena rankings, that would indeed be surprising and noteworthy. But the fact that **some** benchmarks differ is neither unexpected nor particularly significant. In fact, the strength of the correlations reported in Table 3 is, if anything, somewhat surprising.

Regarding Figure 2 (which I believe corresponds the missing Section 4.2), I think the authors have not identified the correct source of the low rank correlations. I hypothesize that HLE is uncorrelated not because it is difficult, but because all models are tightly clustered together (i.e., lack of resolution). The fact that this cluster happens to be centered around zero (due to the task’s difficulty) is less relevant than the clustering itself. The authors might obtain more insight by examining the mean performance difference between consecutive ranks as an alternative metric to regress.

Regarding Figure 3, I believe the findings are not robust, and the conclusions could change simply by excluding the very earliest benchmark. The trend for LMArena Code appears constant, but might appear increasing if HumanEval were removed. Conversely, the trend for IF appears increasing, but could appear constant if IFEval were excluded. The fact that removing a single point would alter the conclusion makes me worry of the soundness of the findings.

While the topic of the paper is very interesting, and I encourage the authors to pursue this line of work further, I believe that, at this stage, the significance and scope of the contributions do not warrant acceptance.

### Questions
Note: I disagree with the framing that LMArena reflects “human perception”. LMArena is also a benchmark, albeit one that arguably more closely resembles chat assistant-like use of models compared to most other benchmarks out there. But it certainly does not evaluate LLMs “in the wild” or human perception in broad sense, it is very much a controlled environment where users go for the explicit purpose of evaluating LLMs.

It would be interesting to compare the average correlation between LMArena and the different benchmark categories, to the average correlation between the benchmarks in each of the categories (e.g., mean correlation between LMArena math and Benchmarks math against mean correlation between Benchmarks math). 

Much of the content in pages 3, 4, and 5 would be better suited for the Appendix.

Typos:
MMLU in L32

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper attempts to "benchmark the benchmarks" by comparing LLM rankings on 24 popular academic benchmarks against the crowd-sourced rankings from the LMArena platform. The authors use LMArena as a 'ground truth' for 'human perception'. They find that many specialized benchmarks (e.g., FACTS Grounding, Humanity's Last Exam) show low or insignificant correlation with LMArena rankings. They conclude that many benchmarks are misaligned with human perception and that neither difficulty nor recency usually guarantees alignment.

### Strengths
The paper's motivation is valid. It targets an increasingly important issue, which is to investigate whether benchmark performance translates into user-perceived quality, as LLM research shifts from leaderboard optimization to practical usability. The inclusion of 24 benchmarks and 114 LLMs is also impressive.

### Weaknesses
1. The paper considers that LMArena represents Human Perception. Which is a flawed assumption. LMArena assesses how humans comparatively favor models when evaluated head-to-head in dialogue-based interactions. Therefore, it is completely unsurprising that a model's ability to pass a graduate-level exam (SuperGPQA) or solve complex reasoning problems (Humanity's Last Exam) does not correlate with its ability to be a more engaging or helpful chatbot for simple queries.

2. Moreover, the comparison methodology is fundamentally broken. 

i. FACTS Grounding (tests factual grounding in long documents) is compared to LMArena's overall ranking, which is dominated by general chat quality. Why would these correlate?

ii. Humanity's Last Exam (extremely difficult reasoning) is compared to LMArena, where users ask simple questions, which the authors also discuss in the Limitations section. 

3. No confidence intervals reported on correlations. 

4. The work seems quite similar to what Laskar et al. (https://aclanthology.org/2024.emnlp-main.764/) have done in Table 3. However, no citations have been provided. This also makes the novelty very limited.

### Questions
1. Justify how LLMArena represents human perception?
2. Justify how the comparison methodology is correct (e.g., how FACTS grounding or Humanity's last exam benchmarks could correlate with LMArena)
3. What correlation coefficient would you consider good enough? and Why?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper examines how correlated popular LLM benchmarks agree with human perception by computing the Spearman correlation between the leaderboard LLM rankings from 24 benchmarks and task-matched rankings on LMArena. Several QA/alignment benchmarks have low or non-significant rank correlation with LMArena. Also, benchmark difficulty and recency mostly don't predict higher correlation.

### Strengths
- A clear, focused research question of interest using a simple, reproducible method (Spearman's correlation on overlapping model sets)
- It broadly used 24 benchmarks across QA/Math/Code/Alignment areas and 4 LMArena categories

### Weaknesses
- The paper acknowledges several limitations of using only LMArena as a standalone human perception dataset. It is plausible but still incomplete. LMArena tends to consist of casual, English queries. In the LMArea leadeboards, top models receive disproportionate exposure, and prompt mix varies over time. The paper notes this, but it still interprets correlations as "resemblance of human perception" rather than 'LMArena-specific preferences." Some level of claim tempering is needed. 

- The paper reports 96 correlation scores (24 benchmarks x 4 LMArena categories) with significance, but it doesn't control for multiple comparisons or show any uncertainty on the correlation scores (e.g., no 95% CIs, no bootstrap variability, no sensitivity to model-set perturbations). The paper currently only provides statistics for significance and one p-value per pair. 

-  Only one rank-agreement metric. Alongside Spearman, report Kendall's tau, top-k overlap (k=5 or 10), and RBO to conduct a robustness check on your correlation results. These are important as some benchmarks have small Ns. I would recommend using RBO as it doesn't require two ordered lists of the same length. 

- The paper assembles benchmark scores and rankings from heterogeneous leaderboards and then computes correlation between rankings after intersecting model sets (common models in both ordered lists). Differences in scoring rules, prompt formats (if LLM-as-judge), and date of entries, etc can induce spurious rank differences. The authors did not provide details of how they employ fair rules in such cases.

### Questions
- Please address the above weaknesses. 
- Include other human preference datasets or curated ones in order to make the paper claim generalizable.

### Soundness
2

### Presentation
2

### Contribution
1
