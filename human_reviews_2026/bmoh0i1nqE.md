# When Data is the Algorithm: A Systematic Study and Curation of Preference Optimization Datasets

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Aligning large language models (LLMs) is a central objective of post-training, often achieved through reward modeling and reinforcement learning methods. Among these, direct preference optimization (DPO) has emerged as a widely adopted technique that fine-tunes LLMs on preferred completions over less favorable ones. While most frontier LLMs do not disclose their curated preference pairs, the broader LLM community has released several open-source DPO datasets, including TuluDPO, ORPO, UltraFeedback, HelpSteer, and Code-Preference-Pairs. However, systematic comparisons remain scarce, largely due to the high computational cost and the lack of rich quality annotations, making it difficult to understand how preferences were selected, which task types they span, and how well they reflect human judgment on a per-sample level. In this work, we present the first comprehensive, data-centric analysis of popular open-source DPO corpora. We leverage the Magpie framework to annotate each sample for task category, input quality, and preference reward, a reward-model-based signal that validates the preference order without relying on human annotations. This enables a scalable, fine-grained inspection of preference quality across datasets, revealing structural and qualitative discrepancies in reward margins. Building on these insights, we systematically curate a new DPO mixture, **UltraMix**, that draws selectively from all five corpora while removing noisy or redundant samples. UltraMix is 30\% smaller than the best-performing individual dataset yet exceeds its performance across key benchmarks. We publicly release all annotations, metadata, and our curated mixture to facilitate future research in data-centric preference optimization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a systematic comparison of five widely used DPO post-training datasets: TuluDPO, ORPO, UltraFeedback, HelpSteer, and Code-Preference-Pairs. The authors analyze each dataset through task category, input quality, difficulty, and preference reward annotations, and propose a new DPO mixture, UltraMix, which is 30% smaller than TuluDPO but outperforms it on various benchmarks. The paper highlights the importance of systematic curation, task diversity, and the impact of data quality on the performance of DPO-based models. It provides a detailed evaluation across multiple models, demonstrating that the optimal composition of data mixtures is task-sensitive and requires careful consideration of quality and diversity, particularly for instruction-following tasks.

### Strengths
The paper’s detailed dataset analysis and creation of UltraMix reflect a high level of thoroughness. The inclusion of annotations for quality, difficulty, and preference rewards provides a nuanced and comprehensive look at the DPO process.

### Weaknesses
1. Comparison of different datasets: While the comparison of different datasets in Table 1 is useful, it does not account for size differences between datasets. For example, TuluDPO has 273k pairs, while ORPO only has 44k.
2. Preference Signal Accuracy: The authors mention that current preference signals have only about 70% accuracy, especially in datasets like UltraFeedback. Since this dataset is annotated by GPT-4o, it would be expected to have a higher level of accuracy. The paper should clarify why the preference signal in this dataset is weaker than anticipated.
3. Model Completion Correlation: The paper claims that model completion is correlated with prompt quality (line 289), but similar results have already been discussed in previous work, such as in [1].
4. Some related works have already discussed what contributes to the quality of a preference dataset; you can consider citing and comparing with them. [2,3,4]
5. Empirical Focus: While the study is solid, it is highly empirical. A more theoretical analysis could help explain the findings and connect them to broader research questions. Insights into the theoretical mechanisms behind dataset curation and the resulting performance gains would strengthen the paper's contributions.

[1] Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback
[2] AIR: A Systematic Analysis of Annotations, Instructions, and Response Pairs in Preference Dataset
[3] R.I.P.: Better Models by Survival of the Fittest Prompts
[4] Direct preference optimization with an offset

### Questions
See above.

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
4

### Summary
This paper presents an empirical study of existing open-source DPO datasets. The authors analyze benchmarks resulting from fine-tuning models across various public datasets, and conduct sample-level analysis using Magpie. Overall, the study offers some useful insights for developers applying DPO to train models, though some experiments may not be convincing enough. Additionally, the methodologies employed largely rely on existing pipelines or models, resulting in limited technical contribution.

### Strengths
- This paper presents a thorough and extensive analysis of five open-source DPO datasets, examining aspects such as task type and prompt quality. The study offers valuable insights—for instance, highlighting that poorly written instructions can lead to subpar preference alignment. 

- The experimental evaluation is comprehensive, employing different benchmarks to assess model performance. Furthermore, the authors validate their findings using different models, enhancing the reliability of the results.

### Weaknesses
- Unclear Motivation. The claim that "quality annotations are mostly missing" is not entirely accurate, as widely-used datasets like UltraFeedback and HelpSteer do provide fine-grained, human-annotated preference scores. Improving these details will help readers understand the contribution of this paper better.

- The current taxonomy requires refinement. Categorizing UltraFeedback and HelpSteer (lines 100-101) as "instruction-following" datasets is imprecise, as they are primarily preference datasets designed to enhance model helpfulness, not strictly instruction adherence.

- The benchmarking results are not fully convincing. They state that they "fix the training hyperparameters for each model," but optimal hyperparameters would vary across datasets. There are also factors (e.g., dataset size) that could significantly impact model performance. 

- The analysis employing the MAGPIE pipeline, while somewhat useful for understanding type distribution, prompt quality, and difficulty across datasets, lacks methodological novelty, as it relies on existing tools and reward models.

- The heuristic selection of thresholds (e.g., 25th and 80th percentiles) appears arbitrary. Since this is an empirical study, a more principled or experimentally justified approach is necessary.

- Figure 3 is not convincing enough. Reward models are known to have biases and may not align perfectly with human judgment, while datasets like Helpsteer are annotated by human and should be more reliable.

### Questions
See weaknessness.

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
4

### Summary
This paper conducts a comparative study on common DPO preference datasets (Ultrafeedback, Helpsteer, TuluDPO…) using the Magpie framework. Using their findings, they construct a new mixture UltraMix, which draws high quality data from all 5 datasets with coherent preferences. Training on UltraMix consistently improves performance across diverse downstream tasks.

### Strengths
1. Data Engineering for post-training is often overlooked but important. The paper presents a nice study and comparison of how to curate data for DPO and demonstrates that their recipe indeed improves performance. 

2. The paper is well-written, and clear. Comprehensive analysis and experiments. Solid results.

3. It would be a good contribution if the authors can release their UltraMix dataset for the community to use and study.

### Weaknesses
1. My main concern is about the novelty of this paper. The main takeaway of this paper is data filtering is important, we need to select high quality and difficult data. Such findings are not new to the field and I don't know how much people would learn from reading this paper. The resulting dataset (UltraMix) itself might be more useful.

2. The quality filtering is conducted by an "independent reward model", in this case, FsfairX. I don't know how much of the gains comes from using this particular reward model v.s. using the same reward model to filter examples from all datasets. In other words, the gains from UltraMix might simply originate from FsfairX is a really really good reward model, which undermines the strength of this paper.

3. The authors' recipe has many "ingredients" in it: data quality filtering, difficulty filtering, consistency filtering ... It is hard to tell which part of the recipe makes it good. It could be because of some of them, it also could be because each one individually is not enough but together they are good. There are some missing ablations here.

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
3
