# Cross-Lingual Data Scaling for Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Large language models (LLMs) achieve consistent performance gains through data scaling, yet low-resource languages remain limited by small and stagnant dataset sizes.
To address this limitation, we introduce cross-lingual data scaling, where performance in low-resource languages scales with the dataset size of high-resource languages.
We systematically investigate two potential approaches: (i) transforming high-resource language data into synthetic data for low-resource languages via translation or code-switching, and (ii) transferring the learned knowledge from high-resource languages to low-resource languages by adjusting language order and proportion during pretraining.
Experiments on English and Chinese show that data transformation fails to sustain cross-lingual data scaling, whereas knowledge transfer enables low-resource language performance to scale with the growth of high-resource language data.
Building on these findings, we propose ScaleX, a two-stage pretraining framework designed for effective cross-lingual data scaling.
In the first stage, LLMs are pretrained on high-resource language data under a constant learning rate schedule;
in the second stage, training continues on a mixture of high- and low-resource languages under a cosine learning rate schedule.
ScaleX outperforms existing approaches with progressively larger margins as high-resource data scales up, and further generalizes to both multilingual and large-scale bilingual pretraining.
Our analysis also reveals that learning rate scheduling and shared tokens across languages are critical to sustaining performance scaling in low-resource languages.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors analyze different approaches to scale low-resource language 
performance by scaling the dataset size of high-resource languages: (i) 
data transformation (creating low-resource synthetic data from the 
high-resource language data), and (ii) knowledge transfer. Motivated by 
the finding that knowledge transfer outperforms data transformation 
techniques, the authors propose ScaleX, a pretraining framework where 
models are first trained only on the high-resource language, and are 
then trained on a mixture of high and low-resource language data. ScaleX 
outperforms baseline approaches that rely on synthetic data.

### Strengths
- The authors find that knowledge transfer techniques can leverage 
high-resource language data to improve low-resource performance. Their 
approach outperforms methods that rely on synthetic data generation.
- The preliminary studies that motivate the design of ScaleX are 
interesting and well-structured.

### Weaknesses
- The authors choose Chinese, a class-5 language, to treat as a 
"low-resource" language across the majority of experiments. While I 
understand that higher availability of data facilitates data sampling 
and the design of the experimental setup, I believe the experiments in 
sections 3.3 and 3.4 would benefit from being repeated with truly 
low-resource languages. Specifically, the authors state that 
"translation alleviates performance degradation compared to baseline, 
but Chinese performance still declines as English data increases". Would 
this also be true for low-resource languages, across language families, 
and across translation models? To make the case for the usage of this 
framework on truly low-resource language performance improvement, I 
would suggest deepening these analyses.

- The authors evaluate ScaleX on Turkish, Hungarian, and Bengali. A 
closer look at the appendix shows that perplexity decreases very 
slightly across the size of English data, notably on Bengali. Are these 
results statistically significant?

### Questions
- The sections on the effects on learning rate schedule and shared 
tokens are extremely interesting. Are the results of these experiments 
similar across other language pairs?
- Typo: "Basline" (line 213)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes to train models in a targeted low resource language with the help of data from an auxiliary language. The experiments consider (i) transformations of the auxiliary data (translation, code mixing), (ii) varying the mixing proportion of aux/target language and (iii) pretraining on auxiliary data followed by joint training. The paper concludes that (iii) is the best strategy with experiments over Chinese (low resource) + English (auxiliary).

### Strengths
The paper reads well, it defines the problem clearly and proposes a sensible list of methods to benefit from auxiliary language data.

### Weaknesses
1. The paper does not compare with strategies established in recent papers (even when these papers are cited, e.g. for the first two).


He et al., 2024 (Findings of ACL) — Scaling Laws for Multilingual Language Models: derives scaling relations that include language-family sampling ratios; directly relevant to “how much” auxiliary language to mix. arXiv:2410.12883
-> why not determine the mixing weights with scaling laws as in this paper?


Li et al., 2024 (NAACL) — Upsample or Upweight? Balanced Training on Heavily Imbalanced Data Mixtures: analyzes temperature sampling vs. scalarization and proposes “Cooldown,” useful for tuning bilingual/multilingual mixtures. arXiv:2410.04579
-> why not try the mixing schedule from this paper?


Seto et al., 2024 (Findings of ACL) — Training Bilingual LMs with Data Constraints in the Targeted Language. Summarizes auxiliary-language vs. translate-then-train strategies and practical upsampling. arXiv:2411.12986
-> why not tune a mixing weight when introducing translation data?




2. The paper targets Chinese end-task performance but intermediate results are established with Chinese perplexity. It remains to be shown that a degradation in perplexity always translates into a degradation in end-task performance. It would be more sensible to split the tasks into validation and test tasks in order to drive data decisions based on end-task validation performance. E.g. I would not be surprised if machine translation improves end-task performance but degrades perplexity in line with [Seto et al., 2024]. If one is interested only in perplexity, Figure 2b. suggests training on 100% Chinese data, yet you pick 80% in practice.


3. The paper defines a search space that is only partially searched. The paper considers that (i) the auxiliary data could be introduced with 3 types of transformation (none, code switching, translation), (ii) the auxiliary data mixing proportion is important, (iii) the auxiliary data can be introduced before or after pretraining on auxiliary data. Yet the conclusions of the different parts are not connected together. It seems that Figure 1 suggests that translating the data is better than using the English data as-is, yet the next experiments (mixing, continued pretraining) do not use translation data. Similarly the mixing experiments do suggest to use only Chinese data in the second phase (Figure 2b) yet the next experiments consider different proportions.


4. The paper considers a single language pair, and only two model scales (1B, 2.5B). The conclusions of the paper cannot be applied to other language pairs and different model sizes. It would be helpful to consider more language and more (smaller) model scales to draw conclusions that can be more generic in terms of language proximity, translation performance and model sizes. Similarly only two sizes of Chinese set are considered (50B and 100B tokens) are considered, what happens if less tokens are available in the targeted language? How does the performance scale wrt to the number of target tokens? How to select the fraction of English data in the second phase when the amount of Chinese data changes?

### Questions
1. Why did you pick 80% as the mixing weight for the second phase? 
2. For the learning rate schedule, does the training schedule include a warmup phase? Is the warmup applied again at the beginning of the second phase? Is cos-cos stopping at 1/2 max lr for the first phase? Why?
3. Why not use translated data (En->Zh) for phase 1 and phase 2?
4. Why not include a baseline with 100% Chinese data with varying sizes? This would help evaluate the value of adding auxiliary (En) data (possibly translated) as opposed to genuine Chinese data.
5. What is the purpose of the English end-tasks? They are only used in Table 1 to say that better results are achieved when more English data is available. I would remove them. 
6. Do 7a and 7b (impact of loss weighting and repetition) behave differently for end-task performance?

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
This paper investigates the impact of cross-lingual data scaling on capability transfer in LLMs. By comparing data conversion and knowledge transfer paradigms, the authors find that knowledge transfer enables low-resource languages to benefit from the scaling of high-resource language data. Building on this insight, they propose ScaleX, a two-stage pretraining framework for efficient cross-lingual data scaling, and demonstrate its effectiveness through extensive experiments.

### Strengths
1. This paper provides a clear and insightful analysis of cross-lingual data scaling mechanisms, offering direct guidance for optimizing LLM training strategies.

2. The experimental design is systematic, covering multiple languages across different resource levels with thorough validation.

3. The methodology is clearly described, and the overall structure of the paper is logical and well-organized.

### Weaknesses
1. Using Chinese as an example of a “ow-resource language is questionable, as its data scale far exceeds that of genuinely low-resource languages. As shown in Table 5, Chinese data (3.99B tokens) is much larger than Bengali (0.17B tokens).

2. The translation experiments only evaluate open-source single-step models (the Qwen2.5-Instruct series, as noted in Appendix A.1), without comparison to stronger closed-source models such as GPT-4o or Claude.

3. The code-switching experiments employ fixed mixing ratios (20% sample-level, 30% token-level replacement). Appendix A.2 tests only static ratios from 10%–90%, without exploring dynamically adjusted schedules.

4. The choice of 80% low-resource ratio in the second stage lacks empirical or theoretical justification.

5. Although ScaleX avoids high-cost translation operations, the two-stage training procedure itself may introduce additional scheduling complexity. A comparison of training time and resource consumption would strengthen the practical analysis.

6. Details regarding ScaleX-DR repetitions are missing. For example, the paper mentions a “single repetition setting” but does not specify repetition curves, and the claim that “benefits peak between 16x and 24x” lacks an explanation for the subsequent performance drop.

### Questions
Refer  Weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies cross-language data scaling issues regarding how the performance of low resource language can scale with the high resource language data size. The primary claim is that creating synthetic data by translating from high resource languages into low resource languages can not achieve cross lingual data scaling; on the other hand, pretraining models solely in high resource languages followed by continuing training with balanced mixed of high and low resource languages can help to achieve cross-language data scaling.

### Strengths
• The topic of scaling performance of low resource languages by scaling the data size of high resource language is attractive.
	• The epxeriments indicate that pretraining models with high resource language data followed by continuing training with balanced mixture of high- and low- resource language data is effective.

### Weaknesses
• High resource language pretraining followed by low resource language continuing training with carefully designed balanced data mixing and learning rates is not new.

### Questions
1. Line 185, a typo: a period punctuation is needed after "..for pretraining".
	2. Section 3.4, please clarify why "order adjustment" and "proportion adjustment" is deemed as knowledge-transfer? What exactly is "knowledge" here? 
	3. Figure 2 (b) please label which plot is English and which is Chinese.
	4. Figure 4, how do the English tasks perform as the Chinese tasks performance increase? 
	5. Line 408-413, as learning rate schedule is experimented with, how about having longer training steps with small learning rates? How to rule out the possiblity of immatured training without enough steps?

### Soundness
2

### Presentation
2

### Contribution
2
