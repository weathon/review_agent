# Less Data Less Tokens: Multilingual Unification Learning for Efficient Test-Time Reasoning in LLMs

- Decision: Reject
- Scores: 4, 4, 8, 6

## Abstract
This paper explores the challenges of test-time scaling of large language models (LLMs), regarding both the data and inference efficiency. We highlight the diversity of multi-lingual reasoning based on our pilot studies, and then introduce a novel approach, $L^2$ multi-lingual unification learning with a decoding intervention strategy for further investigation. The basic idea of $L^2$ is that the reasoning process varies across different languages, which may be mutually beneficial to enhance both model performance and efficiency. In specific, there are two types of multi-lingual data: the entire long chain-of-thought annotations in different languages and the step-wise mixture of languages. By further tuning based on them, we show that even small amounts of data can significantly improve reasoning capabilities. Our findings suggest that multilingual learning reduces both the required data and the number of inference tokens while maintaining a comparable performance. Furthermore, $L^2$ is orthogonal to other data efficient methods. Thus, we also emphasize the importance of diverse data selection. The $L^2$ method offers a promising solution to the challenges of data collection and test-time compute efficiency in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates how multilingual learning can improve long-form reasoning in large language models with less data and fewer inference tokens. The authors discover that different languages yield distinct reasoning patterns, affecting both accuracy and token efficiency. Based on this insight, they propose a multilingual unification learning framework that augments a small amount of high-quality CoT data across languages and applies language-aware decoding. Experiments show that the proposed framwork significantly boosts reasoning performance while reducing test-time compute, offering a more efficient pathway to strong general reasoning.

### Strengths
1. The results are strong, showing that the proposed approach can significantly enhance long-form reasoning with very limited data.

2. Leveraging multilingual diversity to improve test-time efficiency is a sensible idea, supported by clear empirical evidence.

3. The L2 framework is orthogonal to existing data-efficient methods and provides a promising direction for reducing inference token usage without sacrificing performance.

### Weaknesses
1.The evaluation is somewhat limited in scope. Including results on multilingual benchmarks such as MGSM and PolyMath would further strengthen the validation of the proposed approach. 

2.The naming and organization of datasets in the paper can lead to confusion when reading the methodology and experimental setups.

3.Considering practical applications, users generally expect the model’s reasoning and responses to be in a single preferred language. While multilingual mixing may improve efficiency, it could be less practical or even undesirable in real-world usage scenarios where language consistency is essential.

### Questions
1.How does the model perform when the test data are presented in different languages?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explored using multilingual long CoT data, combining entire long chain-of-thought annotations in different languages and the step-wise mixture of languages can boost both performance and efficiency. It shows small amounts of data can improve reasoning capabilities.  Furthermore, it propose language-based logit interventions during inference to switch language.

### Strengths
1. Multilingual thinking as a resource for efficient reasoning is compelling. Using different language to make reasoning more efficient is a novel way to explore. The idea of different languages induce distinct reasoning compression patterns is interesting.

### Weaknesses
1. There are some previous works already discussed using multilingual data to boost reasoning performance [1], maybe you should consider to compare your method with theirs and tell more differences. 

2. Only using tokens number as the metric of measuring efficiency, need also consider using other metrics like the time of inference, FLOP cost, and memory usages.

3. When using tokens in other languages, the factor of compression rate of the tokenizers in that language should also be considered, which wasn't discussed in the paper.

3. Some of experiments are use model training on extremely small multilingual sets, I doubt whether this will make model only memorize input templates. 

[1] Could Thinking Multilingually Empower LLM Reasoning?

### Questions
Please refer to weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a novel technique for data augmentation for LLM finetuning using multiple languages for generation.

Overall, I like the idea, but I have some comments provided below.

### Strengths
The novel data augmentation method.

### Weaknesses
The authors compare their method to some unnamed but presumably simplistic technique for data augmentation, I think that it is better to compare to several existing techniques, e.g. starting with classic backtranslation. I suppose that the results could be explained by data augmentation itself, not the language diversity.

### Questions
Please give more details with examples on your baseline data augmentation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper describes a new method for improving the reasoning efficiency of LLMs at test time via multilingual data. Unlike prior work which focused on single-language CoT annotations and large amounts of fine-tuning data, the authors propose L_2 a multilingual unification learning approach that augments a small number of high-quality CoT examples into multiple languages and mixes language-step reasoning to exploit diverse reasoning patterns across languages. They design experiments by fine-tuning a base model (e.g. Qwen2.5-32B) on only a handful 6 up to ~1000 multilingual annotated samples, evaluating on datasets like AIME24, GPQA-Diamond and MATH500. Experiments show that multilingual augmentation significantly improves reasoning accuracy and also reduces inference token usage compared to monolingual baselines.

### Strengths
1. The idea of leveraging multilingual reasoning diversity (rather than simply more data) to increase reasoning efficiency is novel and sound.
2. The authors showed strong empirical gains even with extremely small annotated sample sizes when augmented via multilingual CoT.
3. Addresses both data efficiency and inference efficiency which is increasingly important in practice for LLM deployment.

### Weaknesses
1. The experiments use relatively small and controlled benchmark sizes e.g., AIME24 with only 30 problems. It's not very clear how much it would scale to broader diverse reasoning tasks.
2. While the reduction in inference tokens is claimed, detailed breakdowns of token savings vs accuracy trade-offs (e.g. across languages and varying lengths) are not sufficiently discussed.

### Questions
1. How sensitive is the method to the quality of the multilingual translations or CoT annotations in non-audited languages?
2. Does the multilingual mixing approach generalize to non-math reasoning tasks?

### Soundness
3

### Presentation
2

### Contribution
3
