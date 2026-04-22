# CharLuMA: Efficient Multi-Language Chart-to-Code Generation with Low-Rank Subspace Adaptation

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 4

## Abstract
Chart-to-code generation involves translating a chart image into an executable plotting script. However, prior work has largely focused on Python-only solutions, limiting real-world applicability and leaving the learning signals inherent in cross-language equivalences untapped. We argue that aligned multi-language scripts serve as complementary “views” of the same chart, providing mutual guidance to regularize the visual-to-code mapping. 
As an instantiation of this idea, we introduce CharLuMA – a multimodal large language model (MLLM) that integrates a language-guided mixture of low-rank subspaces into its multimodal projector. This architecture enables parameter-efficient adaptation via dynamic routing to language-specific subspaces, while preserving shared visual-semantic representations of charts. 
To facilitate training and evaluation at scale, we present Chart2NCode, a dataset of 176k Chart–Python–R–LaTeX quadruples that maintain consistent visual equivalence across languages. Experiments on multiple benchmarks demonstrate that CharLuMA achieves state-of-the-art performance among open-source MLLMs and even surpasses some proprietary systems.
Critically, training with more diverse and balanced language sets yields consistent and substantial improvements across all languages by leveraging the rich supervisory signals embedded in cross-language equivalences.
Subspace activation analysis further reveals a hybrid allocation pattern, with compact shared cores complemented by broader language-specific zones, while stronger models exhibit smoother and more balanced allocations.
Taken together, these results establish multi-language alignment as an effective supervision paradigm for achieving universal chart-to-code generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents an auto-generated large-scale multilingual chart-to-code dataset Chart2NCode, and fine-tunes a multimodal LLM based on Deepseek-Coder for chart-to-code generation across multiple languages (Python, R, LaTeX) based on this dataset naming CharLuMA.  The authors claim that CharLuMA achieves state-of-the-art performance among open-source MLLMs and even surpasses some proprietary systems on multiple chart-to-code datasets. The authors also conducts various ablation studies for model choice and subspace activations of different coding languages.

The paper suffers severe disadvantages as the evaluation metrics are mostly similarity metrics and do not include an accuracy metric that evaluates how well the model can solve the problems, and it does not include most recent open-sourced VLMs in evaluations but claims to be SOTA.  I suggest to reject the paper if the authors do not offer sufficient clarifications.

### Strengths
1: The paper creates a large multi-lingual chart to code dataset Chart2NCode. The task of generating codes from image-form charts is very helpful at works and highly-valuable.

2: The paper trains a model CharLuMA based on the generated dataset Chart2NCode, the model performs better than a lot of open-source models with similar scales on the chart to code task. 

3: The paper conducts comprehensive ablation studies on the model choice and cross-lingual sub-space activation patterns of CharLuMA.

### Weaknesses
1: The paper claims that CharLuMA achieves state-of-the-art performance among open-source VLMs, this is not proper given its submission time of 2025.9, as it only compares with open-source VLMs mostly in 2024 and misses more advanced open-source VLM releases like GLM-4.5V and Qwen3-VL. 

2: All of the evaluation metrics, ER, CB, DS, F1 and TED, are not real accuracy metrics, they just compare similarities between the generation and reference. None of metrics can tell if the generated chart contains tiny but critical error. The evaluation lacks a true accuracy metric that tells whether the generated chart is fully correct or not. 
Moreover, CB, F1 and TED are rewarding codes that are similar to reference codes instead of codes that are correct, and will automatically gives higher score(introduce bias) to models that are trained on Chart2NCode datasets due to learning the unique patterns introduced in the generation process of the Chart2NCode dataset, especially given the dataset is not fully accurate and contains sufficient errors.(See table B.3)

3: The prompt used to evaluate VLMs (shown figure 11) claims the figure is from "a STEM paper", while figures in CharLuMA dataset(as shown in the anonymous link in paper) are mostly in domains of economics and investments, this inaccuracy may cause confusion for other VLMs that are not trained on this pattern.

### Questions
1: Is it possible to evaluate based on the real accuracy of the generated figure that whether the figure is fully correct or not instead of only presenting similarity metrics? If this process is hard to be automated you may sample a smaller subset(100 pictures for example) and evaluate them manually via human volunteers and only compare CharLuMA with 2-4 major baselines. 

2: What's the performance of more advanced open-source VLMs like GLM-4.5V, or Qwen3-VL? Is CharLuMA still better?

3: According to table B.3, the generation process of Chart2NCode still contains around 5-10% of errors. Have you introduced, or do you plan to introduce any further processes to reduce the errors in the dataset?

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
This paper presents CharLuMA, a multimodal large language model for multi-language chart-to-code generation. It introduces a language-guided low-rank subspace adapter that enables efficient and adaptive alignment across Python, R, and LaTeX. The authors also build Chart2NCode, a 176k chart–code dataset supporting balanced multilingual training. Experiments show that CharLuMA achieves state-of-the-art performance among open-source models and narrows the gap with proprietary systems, demonstrating robust cross-language generalization and parameter efficiency.

### Strengths
1. The paper introduces Chart2NCode, the first large-scale multi-language chart-to-code dataset covering Python, R, and LaTeX. It fills an essential gap in the community and is constructed through automatic annotation, LLM-assisted debugging, and human validation to ensure data quality and cross-language consistency.

2. CharLuMA achieves state-of-the-art results among open-source models across multiple benchmarks (Chart2NCode, ChartMimic, Plot2Code), demonstrating robust cross-language generalization and even approaching or surpassing closed-source systems like GPT-4o-mini and Claude-Haiku-3.5.

3. The model exhibits excellent parameter efficiency, enabling seamless adaptation across languages without retraining. This makes CharLuMA highly practical and scalable for real-world multimodal and multilingual applications.

### Weaknesses
1. Although Figure 5 claims to control for training steps across one-, two-, and three-language settings, the experimental design remains questionable. In the single-language setup, the authors randomly resample and duplicate the same data to match the total number of steps used in the multi-language setting. This repetition increases the model’s exposure to identical samples, potentially causing overfitting and reduced representation diversity. Consequently, the observed improvement under the three-language configuration may partly result from richer and more varied training signals rather than the proposed multilingual routing mechanism itself.

2. The core idea that parallel code snippets serve as complementary "views" to regularize learning is intuitive and effective. However, the paper could be strengthened by explicitly connecting this concept to the extensive body of work in multi-task learning, multi-view learning, and cross-lingual transfer learning. For decades, research in fields like machine translation has shown that training on multiple languages simultaneously can foster more robust and generalized intermediate representations that benefit all constituent tasks.

3. Since the authors have already considered a multilingual setting, in reality there are many more plotting languages beyond the three used in the paper. This training approach can only ensure good performance on those three languages.
If a new language needs to be added, it would still require joint multilingual training to update the subspace structure, rather than being able to adapt effectively through training on the new language alone.

4. The author does not give a base LLM and ViT, nor does he explore the impact of different LLMs.

5. Ablation experiments are inadequate and comparisons may be unfair. The authors used an additional dataset to complete the task, but this dataset may be potentially the most important factor

### Questions
see above.

### Soundness
3

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
This paper introduces **CharLuMA**, a multimodal large language model for **universal chart-to-code generation** across Python, R, and LaTeX. Unlike prior Python-only approaches, CharLuMA leverages **cross-language alignment** to improve visual-to-code mapping through a **language-guided mixture of low-rank subspaces**. The authors also release **Chart2NCode**, a dataset of 176k Chart–Python–R–LaTeX quadruples ensuring visual consistency. Experiments show that CharLuMA achieves **state-of-the-art performance**, demonstrating that multi-language supervision significantly enhances generalization and code quality.

### Strengths
* Studies an essential and timely area of cross-lingual learning in the context of chart-to-code generation.
* Proposes a new multimodal architecture (CharLuMA) supporting chart-to-code generation in three languages: Python, R, and LaTeX.
* Demonstrates strong performance, outperforming existing baselines and even some proprietary models.
* Releases models and a large, well-structured dataset (Chart2NCode) containing 176k visually consistent Chart–Code quadruples.
* The paper is well written, clearly presented, and thoroughly executed, with comprehensive experiments and analyses.

### Weaknesses
* The contribution novelty is somewhat limited, as it mainly extends LLaVA to a new application rather than introducing a fundamentally new modeling approach.
* In Figure 5, the imbalance setting lacks reported results for the LaTeX language.
* The metadata construction process requires more clarification and illustration. A concise figure (adapted from the appendix) showing how metadata are extracted from source scripts—perhaps with annotations or visual markings—would improve clarity. It is also unclear whether metadata are extracted via execution, parsing, or the use of LLMs. Although the authors mention an automatic pipeline, the released codebase does not appear to include these components.
* The choice of the vision encoder (SigLIP) is not well justified or compared against alternative encoders.
* Only three languages are considered (Python, R, LaTeX), which limits the generality of the “multi-language” claim and raises questions about scalability.

### Questions
Q: How do u really get the meta data

### Soundness
3

### Presentation
3

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
This paper focuses on the chart-to-code field, extending the previous chart-to-Python approach to LaTeX R and Python. The paper presents 176k training data sets and trains charLUMA, a chart-to-code model based on the MOE architecture.

### Strengths
1. Chart-to-Ncode is a novel field. Unlike previous work that used only single Python code, extending the task to multiple code types better reflects real-world needs.

2. ChartMULA achieved state-of-the-art (SOTA) results on various benchmarks, demonstrating the effectiveness of the model training strategy and training data.

### Weaknesses
1. There appears to be some key information missing, such as the statistical breakdown of different code types in the training set. I am also unsure how the distribution mentioned in Section 6.2 (76.6% Python, 19.2% R, and 4.2% LaTeX) corresponds to the results in Figure 5 and how this ratio was sampled. To my knowledge, the DaTikZ dataset does not contain a large amount of chart-to-LaTeX code, and as stated in Section 3.1, the chart-to-R data is limited to 40k samples at most. Does this imply that the vast majority of the data is chart-to-Python and originates from ChartCoder? If so, the contribution of the proposed dataset might be weakened. My brief review of the images in the anonymous link seems to support this assumption, as most of the data appears to have been rendered using Python. If my understanding is incorrect, I welcome any corrections.

2. The model architecture and training strategy appear to be a combination of ChartMoE and ChartCoder. The former utilizes a Mixture-of-Experts  architecture with different routes for various chart understanding tasks, while the latter employs DeepSeek-Coder as the LLM to improve code generation performance. The overall model structure and training approach of CharLUMA are very similar to these two works, which may indicate a lack of novelty.

3. The evaluation data for Chart2NCode is sampled from the 176k dataset. This could result in the training and test data being drawn from the same distribution, which may inflate the reported performance metrics. I could not find any validation of this test set, analysis of the train/test distributions, or statistics on the quantity of different code types within the test set. This raises questions about the model's out-of-domain generalization capabilities.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
2
