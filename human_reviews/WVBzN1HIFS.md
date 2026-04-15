# PolyMATH: A Challenging Multi-Modal Mathematical Reasoning Benchmark

- Decision: Reject
- Scores: 5, 5, 6, 6

## Abstract
Multi-modal Large Language Models (MLLMs) exhibit impressive problem solving abilities in various domains, but their visual comprehension and abstract reasoning skills remain under-evaluated. To this end, we present POLYMATH, a challenging benchmark aimed at evaluating the general cognitive reasoning abilities of MLLMs. POLYMATH comprises 5,000 manually collected high-quality images
of cognitive textual and visual challenges across 10 distinct categories, including pattern recognition, spatial reasoning, and relative reasoning. We conducted a comprehensive, and quantitative evaluation of 15 MLLMs using four diverse prompting strategies, including Chain-of-Thought and Step-Back. The best scores achieved on POLYMATH are ∼ 41%, ∼ 36%, and ∼ 27%, obtained by Claude-3.5 Sonnet, GPT-4o and Gemini-1.5 Pro respectively highlighting the logical and visual complexity of these questions. A further fine-grained error analysis reveals that these models struggle to understand spatial relations and perform drawn-out, high-level reasoning. This is further strengthened by our ablation study estimating MLLM performance when given textual descriptions in place of diagrams. As evidenced by ∼ 4% improvement over textual descriptions as opposed to actual images, we discover that models do not truly comprehend visual diagrams and the spatial information therein, and are thus prone to logical errors. Finally, we evaluate the OpenAI o1 models and find that their performance only matches the human baseline, highlighting the difficulty of the benchmark. The results on POLYMATH highlight the room for improvement in multi-modal reasoning and provide unique insights to guide the development of future MLLMs

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper present PolyMath, a benchmark evaluating visual abstraction reasoning consists of 5000 mannually collected images acrosss 10 categories. The author also extract testmini (1k) and test-image (with diagram) subset of the benchmark for diverse usage. A comprehensive experiment on testmini evaluating 15 MLLMs with four diverse prompting strategies. The overall results demonstrate a gap between current MLLMs and humans. A fine-grained error analysis reveals model's weak capability on spatial relaition and high-level reasoning. Further ablation study on providing aditional textual description shows that the main difficulty may be comprehensing visual diagrams.  In the end, the author also evaluated the newly introduced GPT-o1 and find that its performance align with human baseline. The author release their benchmark for public usage.

### Strengths
1. A extensive mannually created benchmark: PolyMATH consists 5k high quality images covering 10 patterns, allowing for a more thorough assessment of MLLMs' abilities across various abstract reasoning concepts. The benchmark can be further splited into mini subset for quick evaluation and diagram-only subset for visual-centric abstract evaluation.
2. A comprehensive evaluation: The author conduct a comprehensive evaluation over 15 MLLMs with four prompting approaches, demonstrating current MLLMs visual abstraction capabilities. Further ablation study provide insight on weakness of MLLMs (diagram understanding)
3. The paper already release the dataset for public use.

### Weaknesses
1. Benchmark Positioning:  The comparison between previous VQA or read-world dataset is not enough. I think there is a bunch of benchmarks related with visual abstraction reasoning (https://arxiv.org/abs/2202.10284). Is PolyMATH different than previous dataset or If it is just a combination of previous one. Can the evaluation on PolyMATH bring new insight compared to previous ones?
2. Pattern Design:  The paper mentions the cognitive reasoning and abstract reasoning. But it seems lack a sufficient elaboration on the why do we need to evaulate these patterns on MLLMs. It will be helpful to show practical application of these pattern in real world scenraio or relavent research from human coginition literature.
3. Value Justification: From the ablation study on diagram-only subset, the result showing the major issue blocking models abstract visual reasoning ability is the diagream comprehension. Can we understand the model have difficulties on percepting diagrams? If so, it may make the evaluation less insightful if the major bottleneck is in model's perception abilities instead of abstract reasoning ability. An experiment showing how much detail and vision information MLLMs can percept from diagream will be helpful.

### Questions
1. Based on the examples in Figures 1 and 3, it appears that the format of the questions and answer choices varies. For instance, in the first question of Figure 1, choices are labeled with letters (A, B, C, D) at the bottom, while in the second question, choices are labeled with numbers (1, 2, 3, 4) on the right side. I believe reorganizing all questions in a consistent format could improve the validity of the evaluation.
2. The dataset is unevenly distributed, with three out of ten patterns accounting for more than 40% of the baseline data, while some patterns represent only 3% of the data. Could the imbalanced data distribution lead to misleading conclusions in Section 4.2?
3. In Lines 462 to 466, simply comparing test-img and testmini makes the argument that "decreased accuracy on test-img is largely due to the presence of diagram-based problems" problematic. It's possible that test-img has a different distribution compared to testmini. For example, the questions in test-img may be more challenging, prompting designers to include diagrams for better understanding.
4. The authors adopt four prompting strategies for evaluating the MLLMs but only mention performance differences in one sentence (Lines 418-421). Could more insights or explanations be provided regarding these four strategies?
5. As the paper presents the complete PolyMATH dataset with 5,000 images, it seems unusual to report results on only 20% of them. Could the authors also provide results for the entire dataset? This would be beneficial for comparing new MLLMs more comprehensively in future research.


Overall, I appreicate the effort author and team paid on construct such a comprehensive datasets. I am willing to adjust my score if questions metioned previously can be solved and clarified.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces POLYMATH, a multimodal mathematical reasoning benchmark designed to assess the performance of large multimodal language models (MLLMs) in terms of visual understanding and abstract reasoning abilities. The benchmark covers 10 different categories, including cognitive text and visual challenges such as pattern recognition, spatial reasoning, and relative reasoning. The researchers conducted a comprehensive and quantitative evaluation of 15 MLLMs using four different prompting strategies, including Chain-of-Thought and Step-Back.

### Strengths
1. A new multimodal mathematical reasoning benchmark, POLYMATH, is proposed, covering 10 different categories and contributing to the advancement of mathematics reasoning tasks.
2. The paper conducted a comprehensive and quantitative evaluation of 15 MLLMs using four different prompting strategies.

### Weaknesses
1. The paper does not provide open-source examples from the POLYMATH dataset. Without such examples, the unique challenges and value of the dataset are not clearly conveyed, weakening the paper’s effectiveness.
2. The analysis provided is superficial and covers mostly well-known conclusions without exploring the performance differences across models in different reasoning tasks or the underlying reasons. This shallow analysis fails to offer new insights or robust support for the conclusions, limiting the paper's contribution to the field.
3. Although the paper mentions the issue of large models’ mathematical reasoning capabilities, it lacks further exploration of this topic. The absence of an in-depth discussion or future directions weakens the paper's forward-looking impact and completeness.

### Questions
1. Till the deadline of the review process, the link to the benchmark is not open-source. I hope the authors would organize the benchmark website properly during the rebuttal so that the benchmark’s composition and evaluation content could be inspected.
2. Experiment analysis is a little superficial. I hope the authors may provide some detailed analysis of the experimental results to yield more insightful conclusions.
3. The paper proposes many problems of current LLM models, but does not provide possible solutions. It will make this paper more meaningful if the authors would do some explorations on it.

### Soundness
2

### Presentation
2

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
This paper introduce PolyMath, a multimodal math benchmark curated from India National Talent Search Examination.
PloyMath is carefully curated following 5 steps on collection, categorization and quality assurance.
Extensive experiemnts are conducted, showing that the best MLLM only achieve at most 41.9% performance on PloyMath.
Further analysis find that MLLM rely more on textual descriptions than visual information, and the logical flaw to be the most common error.
Study also finds that openai-o1 model performs closely with human baseline on textual only version of the dataset.

### Strengths
- The writing is clear and easy to follow.
- The proposed dataset is challenging, which is useful for future development of LLMs and MLLMs.
- The ayalysis showing that MLLMs rely more on text descriptions and the detailed error analysis are insightful.

### Weaknesses
- The findings in the paper are not new. Firstly, for 'MLLMs rely more on text than image', MathVerse[1] specifically create 6 versions of their dataset with incremental reliance on visual information, showing the same trend.
- The error analysis suggest logical flaw to be the main bottle neck of MLLM's performance. But Table3 and 4 suggests that different inference stretagy does not help much, comparing with zero-shot. (only about 2%-6%) So it's unclear how we can improve our models on this dataset.
- Insufficient discussion on related works. Since PolyMath is scoped as abstract visual reasoning benchmark, the author should discuss how PloyMath is different from and better than existing benchmarks in this field, like Puzzle-VQA[2], Marvel[3].

[1] Zhang, R., Jiang, D., Zhang, Y., Lin, H., Guo, Z., Qiu, P., ... & Li, H. (2025). Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems?. In European Conference on Computer Vision (pp. 169-186). Springer, Cham.

[2] Chia, Y. K., Han, V. T. Y., Ghosal, D., Bing, L., & Poria, S. (2024). PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns. arXiv preprint arXiv:2403.13315.

[3] Jiang, Y., Zhang, J., Sun, K., Sourati, Z., Ahrabian, K., Ma, K., ... & Pujara, J. (2024). MARVEL: Multidimensional Abstraction and Reasoning through Visual Evaluation and Learning. arXiv preprint arXiv:2404.13591.

### Questions
- Table 5 and table 3's performance are not consistant, the reviewer understands they are on different subsets, but the big difference is not expected. Could the author explain why?
- For the second weakness above, could the author give hypothesis on why inference scaling doesn't help the model much? And what is expected to improve our model on PolyMath?
- Since the error analysis suggest logical flaw to be the main bottle neck of MLLM's performance (which is irrelavant to image information), and all data can be converted to textual format. What make it necessary for this dataset to be made multimodal?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces POLYMATH, a novel benchmark designed to evaluate the mathematical and visual reasoning capabilities of multi-modal large language models (MLLMs). The authors conducted extensive evaluations on 15 state-of-the-art MLLMs, analyzing their performance through multiple prompting strategies, such as zero-shot and Chain-of-Thought prompting. Results reveal significant gaps in MLLM performance compared to human baselines, particularly in spatial reasoning and diagram interpretation.

### Strengths
S1: The authors provide a detailed categorization schema, covering ten reasoning types, which ensures a thorough assessment of MLLM capabilities in cognitive tasks.

S2: The authors implemented a rigorous evaluation using multiple inference strategies, including zero-shot, few-shot, Chain-of-Thought, and Step Back prompting.

S3: A fine-grained analysis highlights common pitfalls in model reasoning, such as logical flaws and spatial misunderstandings, providing insights into areas that need improvement.

### Weaknesses
W1: While the paper presents error analysis for major models like Claude-3.5 Sonnet and GPT-4o, extending this analysis to more open-source models could yield deeper insights into shared weaknesses.

W2: The reported improvements when replacing diagrams with text descriptions raise questions about MLLM comprehension of visual information. A more in-depth discussion or qualitative examples would enhance understanding.

### Questions
The paper notes a performance improvement when diagrams were replaced by textual descriptions, indicating that MLLMs may not be effectively processing visual information. Could the authors provide more qualitative examples or specific insights into how the textual descriptions differed from the diagrams and why these led to improved performance?

### Soundness
2

### Presentation
3

### Contribution
2
