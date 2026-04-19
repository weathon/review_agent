# To Code or Not To Code? Exploring Impact of Code in Pre-training

- Decision: Accept (Poster)
- Scores: 8, 3, 6, 6

## Abstract
Including code in the pre-training data mixture, even for models not specifically designed for code, has become a common practice in LLMs pre-training. While there has been anecdotal consensus among practitioners that code data plays a vital role in general LLMs' performance, there is only limited work analyzing the precise impact of code on non-code tasks. In this work, we systematically investigate the impact of code data on general performance. We ask “what is the impact of code data used in pre-training on a large variety of downstream tasks beyond code generation”. We conduct extensive ablations and evaluate across a broad range of natural language reasoning tasks, world knowledge tasks, code benchmarks, and LLM-as-a-judge win-rates for models with sizes ranging from 470M to 2.8B parameters. Across settings, we find a consistent results that code is a critical building block for generalization far beyond coding tasks and improvements to code quality have an outsized impact across all tasks. In particular, compared to text-only pre-training, the addition of code results in up to relative increase of 8.2% in natural language (NL) reasoning, 4.2% in world knowledge, 6.6% improvement in generative win-rates, and a 12x boost in code performance respectively. Our work suggests investments in code quality and preserving code during pre-training have positive impacts.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
## Summary:
The authors conduct a systematic investigation into the role of code data in the pre-training phase of LLMs, assessing the impact on three task categories: natural language reasoning, world knowledge, and code generation. They analyze three pre-training stages, including initial pre-training, continual pre-training, and a cooldown phase; evaluate the effect of code data’s proportion, quality, and properties on the model’s performance across the three task categories. Their results demonstrate that incorporating code data enhances model performance across all the pre-training stages, suggesting a balanced approach with optimized code data quality can boost the model’s ability in reasoning, general knowledge, and coding.

### Strengths
- _Novelty_: This work is a first-of-its-kind, comprehensive study on how code data in pre-training affects LLMs across various natural language and coding tasks, offering new insights into data requirements for LLM pre-training.

- _Comprehensiveness_: The authors carry out a broad set of experiments, exploring various scales of model size, data quantity, and training stages. Their work provides actionable guidance for future LLM development.

### Weaknesses
1. **Limited Details on High-Quality Code Data**: A significant portion of the paper’s conclusions, particularly in Section 3.4, emphasize the positive impact of incorporating high-quality code data on LLM performance across different task categories. However, the description of the high-quality code data is limited to that it is proprietary and synthetically generated. The lack of transparency about the dataset's characteristics, such as the source, data format, or complexity raises questions about what specifically constitutes "high quality" and how these qualities directly influence performance. More details on criteria for data quality would strengthen the paper and enable the research community to apply similar quality standards in future work.

2. **Potential Bias in Generative Ability Evaluation**: The evaluation of generative ability relies on the LLM-as-a-Judge approach, specifically through win-rate scores on the Dolly-200-English dataset, which is designed to reflect open-ended, non-code tasks. The authors attribute improvements in generative quality to the inclusion of code data in pre-training, arguing that code data enriches the model’s ability to generate preferred outputs. However, using LLMs to evaluate other LLM outputs has known limitations[1], including potential biases toward verbosity, structured formatting, and positional advantage. Moreover, the inclusion of markup-style code may enhance readability and formatting, making responses more appealing to the LLM evaluator. To mitigate ambiguity, qualitative examples of output differences would be helpful in illustrating how the inclusion of code data contributes to generative improvements beyond format and readability alone.

3. **Minor Issues**: 
   - In Section 3.2, Figure 3 is referenced incorrectly as Figure 8, which could confuse readers. 
   - The legend for Figure 8 lacks a label for the gray bar, which reduces clarity in interpreting the data presented.

[1] Zheng, L., Chiang, W. L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., ... & Stoica, I. (2023). Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36, 46595-46623.

### Questions
- Could the authors clarify whether the Balanced LM (50% text and 50% code) incorporates markup-style data in its training mix? The paper specifies that the Code-initialized model includes 80% code data and 20% markup-style data, but it remains unclear whether this type of data is also included in the Balanced LM.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper investigates the impact of code data in pre-training for NLP tasks. The authors pre-train Transformer models with sizes ranging from 470M to 2.8B parameters with ablations of code and non-code data. The pre-trained models are evaluated on downstream tasks such as NL reasoning, world knowledge, and code benchmarks. Experimental results show that incorporating code into pre-training has a positive impact on non-coding tasks. In addition, improvements to code quality have an outsized impact across all tasks. In particular, compared to text-only pre-training, the addition of code results in up to relative increase of 8.2% in natural language (NL) reasoning, 4.2% in world knowledge, 6.6% improvement in generative win-rates, and a 12x boost in code performance respectively. The results suggest future investments in code quality and preserving code during pre-training.

### Strengths
• The paper provides insights into the effects of code data on pre-training across a range of NLP tasks.

• Experiments are conducted with multiple configurations, such as balancing code and text data, adding depth to the analysis.

### Weaknesses
• The findings are somewhat unsurprising, as incorporating diverse data sources (like code) in pre-training is known to improve model performance—a principle demonstrated by previous studies, including PaLM (Chowdhery et al., 2022), Gopher (Rae et al., 2022), and Bloom (Workshop et al., 2023). This paper extends these findings by analyzing code quality and proportion but does not significantly depart from established conclusions, such as the importance of data quality in pre-training (https://arxiv.org/abs/2207.05579).

• Figure 4 suggests that including code data may actually reduce performance on some natural language tasks, which appears contradictory to the paper’s main claim. This inconsistency weakens the overall reliability of the conclusions.

• The pre-training models used in this study are relatively small (470M and 2.8B parameters), which limits the generalizability of the findings. Larger models may be necessary to substantiate conclusions regarding the effects of code data on NLP pre-training.

### Questions
1. Given the observed performance drop on certain NL tasks, how does the study reconcile this finding with the broader claim of code data’s benefits for NLP tasks?

2. Have the authors considered evaluating the pre-trained models on additional natural language tasks to better understand the generalization effects of code data?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors examine the role of code data in the pre-training of large language models (LLMs), a practice that has become increasingly common, even for models not specifically designed for code. Despite anecdotal consensus among practitioners regarding the importance of code in enhancing the performance of general LLMs, there has been limited research analyzing its precise impact on non-code tasks. This study systematically investigates the influence of code data in pre-training on a diverse array of downstream tasks beyond code generation. The authors pose the question: “What is the impact of code data used in pre-training on a large variety of downstream tasks?” Through extensive ablation studies, they evaluate models ranging from 470M to 2.8B parameters across natural language reasoning tasks, world knowledge tasks, code benchmarks, and LLM-as-a-judge win rates. The results consistently demonstrate that code serves as a critical building block for generalization well beyond coding tasks. Improvements in code quality significantly enhance overall task performance, with findings indicating up to an 8.2% relative increase in natural language reasoning, a 4.2% boost in world knowledge, a 6.6% improvement in generative win rates, and a 12x enhancement in code performance when compared to text-only pre-training. The authors suggest that investing in code quality and retaining code during pre-training can yield substantial benefits across various tasks.

### Strengths
+ Important Area.
+ Interesting Finding.
+ Analysis of the Code Quality is Good.

### Weaknesses
- Technical Contribution is Weak.
- Limited Scale.

### Questions
The investigation into the role of code in LLM pre-training addresses a crucial aspect of model performance with good implications for both research and practical applications. The exploration of the impact of code data and strategies for leveraging it provides valuable insights into enhancing model effectiveness, contributing to a deeper understanding of LLM capabilities.

I have two concerns:

1. Limited Technical Contribution: The study primarily consists of empirical analyses without introducing new methodologies or frameworks. While the findings are good, the technical contribution could be perceived as limited due to the absence of novel approaches or theoretical advancements.

2. Limited Scale: The analysis is restricted to models with 470M and 2.8B parameters, which may not capture the performance dynamics across a broader range of model sizes and architectures.

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
4

### Summary
The paper investigates the impact of incorporating code data into pre-training for large language models (LLMs).
It explores how code data influences various tasks, including natural language reasoning, world knowledge, and open-ended generation. 
Through extensive ablation studies, the authors reveal that adding code data yields substantial benefits across tasks beyond coding. 
The study emphasizes code quality and finds that synthetic, high-quality code data further enhances general performance. 
The results imply that code data should be included in the pre-training process as it significantly enhances generalization across tasks.

### Strengths
- The paper addresses a crucial question: does code data improve performance on non-code tasks? This question has been relatively unexplored, and the paper offers a fresh perspective on the potential benefits of code data for models primarily intended for natural language tasks.

- The methodology is rigorous, employing controlled experiments that examine various factors, such as code quality, proportions, and stages of pre-training. The large-scale experiments (up to 2.8B parameters) provide an in-depth understanding of how code data impacts model performance.

- This work has significant implications for pre-training dataset design, suggesting that code data improves generalization and that high-quality synthetic code can be particularly impactful. This insight can guide future LLM development, especially in choosing and refining pre-training data.

### Weaknesses
- For the scaling experiments, some key information could be explored further. For instance, does the trend observed in Figure 4 hold consistently across models of different scales? Additionally, how can the trends identified be extended or generalized to larger models?

- I am also curious about the model's performance on mathematical tasks, as these, like coding, are highly logical and reasoning-intensive. Including a benchmark evaluation on mathematical tasks could provide valuable insight into whether similar gains extend to other logic-focused domains.

- In terms of writing, certain critical details could be clarified. For instance, it may not be immediately clear in the text which model size (e.g., 470M or 2.8B) is being evaluated in figures such as Figures 4 and 5. Adding such context in the captions or main text would enhance clarity for readers.

**Given the close relationship between scaling laws and research on dataset composition, there is still room to refine and expand upon some of the paper's exploratory findings.**

### Questions
- In the sentence "Figure 8 shows the results of xxx" in Section 3.2, it seems that "Figure 8" should actually be "Figure 3." May be a typo?

### Soundness
3

### Presentation
3

### Contribution
3
