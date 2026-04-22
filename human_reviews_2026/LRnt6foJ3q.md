# HalluText: Towards Benchmarking and Mitigating OCR Hallucination for LVLMs

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Optical Character Recognition (OCR) serves as a critical bridge connecting vision and language, attracting increasing attention in the community of Large Vision-Language Models (LVLMs). However, due to the prevalent encode-then-decode architecture, LVLMs tend to over-rely on language priors, leading to frequent failures in following basic visual-text instructions. We term this issue OCR hallucination. 
To systematically mitigate it and facilitate reliable OCR perception in LVLMs, we conduct the first large-scale empirical analysis based on OCRBench v2. Our findings reveal that current LVLMs frequently misinterpret or ignore textual visual content, particularly across two orthogonal dimensions, including perception task and hallucination taxonomy. Building on these insights, we introduce HalluText, a benchmark specifically designed to comprehensively evaluate OCR hallucination in LVLMs across nine subclasses. Alongside this benchmark, we propose OCRAssistor, a lightweight plug-and-play method pioneering large-small model collaboration. By integrating compact OCR model outputs into the LVLM decoding process, it achieves a 9.6\% improvement on HalluText with only marginal computational cost. When applied to OCRBench v2, this method also improves the performance of the top-performing open-source model Qwen2.5-VL-7B, achieving a 3\% gain and highlighting the importance of addressing OCR hallucination in LVLMs. Through our benchmark and proposed solution, we hope to shed light on the challenges and potential pathways for improving visual text perception in LVLMs. The organized benchmark and the relevant code will be released soon.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper addresses a relevant problem and provides both a benchmark and a practical solution. The empirical results are promising, showing consistent improvements across multiple models and scales. However, the limited baseline coverage, missing comparisons with state-of-the-art OCR systems, and shallow exploration of the method's design space prevent a stronger recommendation. The work feels somewhat incomplete in its current form—it identifies an important problem and proposes a reasonable solution, but does not sufficiently validate the solution against alternatives or provide deep understanding of when and why it works. With revisions addressing the comparison gaps and providing more thorough analysis of the method's strengths and limitations, this could become a solid contribution to the OCR and LVLM literature. 
Additionally, I believe that as a benchmark paper, it is necessary to provide more fundamental insights similar to works like M3COT. Currently, various works and solutions appear to be more engineering-oriented rather than addressing fundamental issues. The authors need to carefully revise the paper based on reviewer feedback. I personally recommend discussing the relationship between visual compression rate and hallucinations mentioned in DeepSeek OCR, which could help this work make a more substantial contribution.

### Strengths
**Comprehensive benchmark construction**: HalluText is thoughtfully designed with nine subsets covering different failure modes, from existence and position to more subtle issues like the Stroop effect and frequency counting. The use of existing datasets (Total-Text, ICDAR2013, Union14M) as sources provides some grounding, while the synthetic components (Stroop, Typo) address specific phenomena that might be underrepresented in natural images. The multiple-choice QA format with up to four options provides standardized evaluation.

**Practical method with minimal overhead**: OCRAssistor demonstrates that integrating external OCR signals can improve performance without fine-tuning the base model. The consistent improvements across different model scales (Table 7) suggest reasonable generalization.

### Weaknesses
**Questionable problem motivation and scope**: While the paper identifies OCR hallucinations in LVLMs as a problem, the motivation for focusing exclusively on single-LVLM solutions is unclear. The benchmark design assumes that OCR tasks must be completed by a single LVLM, but this constraint is not well justified. Pipeline-based approaches (e.g., MinerU [1,2]) that combine specialized detection and recognition modules also exhibit hallucination behaviors, yet are excluded from consideration. Furthermore, given that several recent works have already begun addressing LVLM hallucinations in OCR contexts (e.g., consensus-based methods, calibration approaches), the paper does not adequately position itself relative to these existing mitigation strategies. The three failure cases in Figure 1 illustrate problems, but it remains unclear whether these are unique to end-to-end LVLMs or represent broader OCR system failures that warrant a more comprehensive benchmark encompassing both single-model and pipeline approaches.


### 1. Limited Baseline Coverage for Hallucination Mitigation

While Table 2 provides a thorough comparison of base LVLMs, it conspicuously lacks comparisons with other hallucination mitigation methods beyond VDGD. The paper positions OCRAssistor as a solution to OCR hallucination but does not compare against:
- Calibration methods that could provide uncertainty estimates for OCR outputs
- Consensus-based approaches that aggregate predictions from multiple models or multiple forward passes [4]
- Other contrastive decoding variants beyond VDGD that have been proposed for reducing hallucinations in VLMs
- Pipeline-based OCR systems that combine specialized detection and recognition models, which might not suffer from the same hallucination patterns as end-to-end LVLMs

This omission makes it difficult to assess whether the gains from OCRAssistor are competitive with alternative approaches to the same problem. The fact that only two methods (Baseline + CoT, Baseline + OCR, and OCRAssistor) are compared in Table 4 suggests a somewhat narrow exploration of the solution space.

### 2. Missing Comparisons with Document Parsing Systems

The paper claims to address OCR tasks but does not engage with recent advances in document parsing and OCR systems. Specifically:
- MinerU [1] and MinerU2.5 [2] represent current state-of-the-art in document content extraction with precise handling of complex layouts and text recognition
- OmniDocBench [3] provides a comprehensive benchmark for diverse document types that would complement the focused HalluText evaluation
- The paper does not discuss how OCRAssistor compares with or could be combined with these specialized systems

Given that the proposed method integrates an external OCR model (PaddleOCR), it is essential to understand how it compares with other OCR systems. The difference in performance between PaddleOCR and EasyOCR in Table 5 hints at this sensitivity, but the paper does not explore whether using more advanced OCR engines like those in MinerU would yield further improvements. This leaves open the question of whether the gains are primarily from the integration mechanism or from the specific OCR model chosen.

### 3. Lack of Computational Cost Analysis

While Table 6 provides timing comparisons, the paper lacks a comprehensive analysis of the computational overhead introduced by OCRAssistor. The authors should provide:
- A performance-cost trade-off figure (similar to what DeepSeek-OCR provides in their Figure 1) showing how different methods balance accuracy gains against computational overhead
- Detailed breakdown of where the additional 0.6s per image is spent (OCR inference vs. distribution modification)
- Analysis of how the overhead scales with image resolution, text density, and model size

The current presentation mentions that "the relative overhead introduced by the OCR module" would be smaller in complex generation scenarios, but this is only briefly discussed without empirical evidence. For a method positioned as plug-and-play and practical, a thorough cost-benefit analysis is essential.

### 4. Shallow Exploration of Design Choices

Several aspects of the method feel underexplored. The ablation studies in Section 5.2.3 show that OCRAssistor outperforms simply appending OCR results, but the paper does not investigate:
- Why the KL-divergence-based guidance is specifically beneficial for OCR tasks compared to other distribution modification strategies
- Whether the temperature parameter T and regularization factor λ require tuning for different types of OCR tasks (the paper only tests λ in Table 10 for Qwen2.5-VL-3B)
- How the method handles cases where the OCR model is completely wrong versus when it is partially correct
- Whether different OCR hallucination types (e.g., position vs. typo) benefit differently from the guidance mechanism

The decision to include both OCR outputs and CoT in the OCRAssistor prompt (Table 8) versus using only OCR outputs in the baseline + OCR setting makes it difficult to isolate the contribution of the decoding guidance mechanism itself.

### 5. Benchmark Validity Concerns

While HalluText provides structured evaluation, some design choices raise questions about ecological validity:
- The Stroop Effect subset uses synthetic images with color/shape words, which may not reflect natural OCR scenarios
- The Typo subset uses a general typo corpus rather than actual OCR errors, conflating typographical errors with visual recognition errors
- The Incompletion subset's construction process (manually verified and cleaned from OCRBench v2) is only briefly described, making reproducibility difficult
- The paper does not provide inter-annotator agreement statistics or validation that the failure cases identified in the empirical analysis (Section 3.1) actually represent hallucinations rather than ambiguous or incorrectly annotated samples

The fact that only 1,034 out of 3,006 failure cases (34%) were deemed analyzable hallucinations suggests that many errors in OCRBench v2 stem from other issues. This raises the question of whether HalluText's nine categories capture the most important failure modes or just those that are easiest to categorize.

### 6. Limited Analysis of Failure Modes

The paper identifies that LVLMs struggle with certain subsets (Index, Slice, Relative Position, Counting) but provides limited insight into why OCRAssistor helps. For instance:
- Table 2 shows that OCRAssistor dramatically improves Index (from 50.3 to 72.9) and Slice (from 49.8 to 82.0) for Qwen2.5-VL-7B, but these gains are not explained mechanistically
- The paper does not analyze whether the improvements come from better attention to text regions, better character-level recognition, or simply anchoring to the OCR outputs
- There is no discussion of cases where OCRAssistor fails or makes performance worse (the method appears to uniformly improve all subsets, which seems surprising)

The visualizations in Appendix E provide qualitative examples but do not systematically analyze error patterns or explain the mechanism behind the improvements.

### Questions
1. **Comparisons with hallucination mitigation methods**: Can the authors include comparisons with other calibration and consensus-based methods for reducing hallucinations in VLMs? How does OCRAssistor compare with ensemble methods or other uncertainty quantification approaches that have been proposed for VLMs?

2. **Integration with advanced OCR systems**: How would OCRAssistor perform if integrated with more sophisticated OCR engines like MinerU or MinerU2.5? Is the gain primarily from the integration mechanism or from the specific OCR model quality? Could the method be applied to document-specific benchmarks like OmniDocBench?

3. **Cost-benefit analysis**: Can the authors provide a comprehensive performance-cost trade-off figure showing how OCRAssistor compares with other methods in terms of accuracy gains versus computational overhead? How does the overhead scale with different input characteristics?


[1] Bin Wang, Chao Xu, Xiaomeng Zhao, et al. MinerU: An Open-Source Solution for Precise Document Content Extraction. arXiv:2409.18839, 2024.

[2] Junbo Niu, Zheng Liu, Zhuangcheng Gu, et al. MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing. arXiv:2509.22186, 2025.

[3] Linke Ouyang, et al. OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations. In CVPR, pp. 24838-24848, 2025.

[4] Consensus Entropy: Harnessing Multi-VLM Agreement for Self-Verifying and Self-Improving OCR
4. **Design choice justification**: Why is KL-divergence-based guidance specifically appropriate for OCR tasks? Have the authors experimented with other distribution modification strategies? How sensitive is the method to the hyperparameters T and λ across different OCR hallucination types?

5. **Benchmark validation**: What is the inter-annotator agreement for identifying hallucination types in the empirical analysis? How were the 1,034 analyzable samples selected from the 3,006 failures, and could this selection introduce bias? Are the synthetic components (Stroop, Typo) validated against real-world OCR errors?

6. **Failure case analysis**: Are there cases where OCRAssistor fails or makes performance worse? Can the authors provide systematic analysis of when the method helps versus when it does not? What is the mechanism behind the dramatic improvements on Index and slice tasks specifically?

7. **Prompt design ablation**: The OCRAssistor setup includes both CoT and OCR information, while the baseline + OCR uses only OCR. Can the authors provide fair comparison where both methods use the same prompt structure to isolate the contribution of the decoding guidance?

8. **Generalization beyond multiple-choice**: All HalluText evaluations use multiple-choice QA format. How does OCRAssistor perform on open-ended OCR tasks where the model must generate free-form text transcriptions?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates a critical weakness in Large Vision-Language Models (LVLMs)—their tendency to hallucinate text in visual scenes, termed OCR hallucination. These hallucinations occur when models misinterpret or fabricate text during OCR-related tasks. The authors perform a large-scale empirical study on OCRBench v2, identifying consistent hallucination patterns along two axes: (1) perceptual stage (localization vs. recognition) and (2) hallucination type (category, relation, attribute).
Based on this taxonomy, they propose HalluText, a new benchmark containing 4,678 image–question–answer triplets covering nine hallucination types (e.g., typo, counting, Stroop effect). Experiments show that existing state-of-the-art LVLMs—including Qwen2.5-VL, InternVL3, and GPT-4o—perform poorly (≤80% accuracy) on HalluText, indicating widespread OCR hallucination.
To address this, they introduce OCRAssistor, a training-free, plug-and-play framework where a small OCR model guides the LVLM’s decoding via a KL-divergence regularization between OCR-derived and model-generated token distributions. OCRAssistor boosts accuracy by +9.6% on HalluText and up to +3.7% on OCRBench v2, with minimal computational cost (<1 s overhead per image).
Overall, the paper contributes: A systematic taxonomy and benchmark (HalluText) for diagnosing OCR hallucination. A lightweight mitigation method (OCRAssistor) demonstrating effective large-small model collaboration. Empirical validation showing consistent gains across models and languages.

### Strengths
1. The authors build HalluText, a fine-grained benchmark that systematically diagnoses OCR hallucination across nine subtypes (e.g., typo, Stroop effect, counting, position). It provides structured evaluation missing in existing datasets like OCRBench, allowing researchers to pinpoint failure modes more precisely.

2. The paper conducts a large-scale empirical analysis across several major LVLMs. The proposed dual-perspective taxonomy (perception stage × hallucination type) offers conceptual clarity and diagnostic interpretability.

3. Compared to prior decoding-based hallucination control methods (e.g., VDGD), OCRAssistor is 10× faster while maintaining strong performance, making it suitable for deployment.

### Weaknesses
1. The approach’s success relies heavily on the OCR model’s accuracy. If the external OCR module fails or misreads (e.g., under extreme blur or multilingual scripts), the guidance may propagate incorrect cues to the LVLM.

2. Although the authors test many LVLMs, experiments are limited to multiple-choice QA settings. Real-world scenarios (e.g., open-ended generation, document captioning, long-context OCR) are not explored, leaving questions about robustness and generalization.

3. Can OCRAssistor be applied to other benchmarks like ChartQA or DocQA, which also need OCR ability to understanding the image but need reasoning skill? Can the authors explains this? When I check the examples in your benchmark, I found most of them are recognition task without reasoning.  I am really curious how the OCRAssistor performs in the reasoning task.

### Questions
My questions are mentioned in the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This study proposed a method to reduce the hallucination in OCR tasks with large vision language models (LVLMs). The authors observed that LVLM biases on the language knowledge they've learned, so they will ignore some special occasions when the texts presented in images are manipulated, imcompleted, or uncommon.

To deal with this problem, the authors designed a method to help LVLMs to make more factual predictions using with the assistance of compact OCR models that better align to the actual texts they detect from the image and don't have much text knowledge to bias on. with the output of OCR models, the LVLMs can better predict the actual texts shown in an image.

Experiments showed that the proposed method achieves significant improvement on recognition tasks. this validates authors motivation that manipulated texts are hard to be correctly recognized by LVLMs and the compact OCR models, without heavy prior language models to bias to, can help language models to fix this problem.

### Strengths
- The problem is well motivated and he experiments proves that learning the actual spelling / presence of the texts can significantly improve the OCR model accuracies significantly. As far as I'm concerned here, the major claim of this paper is mostly supported by experiments.
- The dataset is interesting too. If contained examples that none of the selected LVLMs can correctly recognize, and the authors took the effort to remove test cases that are not relevant to the OCR hallucination problem.

### Weaknesses
## Some over clamings
  - In line 215, it says the development of this hallucination benchmark has "theoretical" grounding. I think it's too much. a better term here is "conceptual" grounding.
  - I feel that main take away of experiments is that when the words are not spelled in the way they usually are, LVLM's cannot recognize them accurately. However in the sections before experiments, the authors presented much more categories than this case but haven't explained them enough.

## Lack of simple baselines
there are two simple baselines that can be tested before proposing an over-complicated method
1. Just give the LVLM the plain text output by the compact ocr models in the prompt and let the LVLMs predict. I guess baseline+ocr in table 4 stands for this test, but I'd love to see the numbers in the main table across models and tasks.
2. Given image and question, score the logits of all text snippets discovered by the OCR model and select the top 1 as the answer. This method is widely used in single-choice tasks.

### Questions
n/a

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces the HalluText benchmark for evaluating OCR hallucination in VLMs. It systematically analyzed OCR failure cases and categorized the errors along different dimensions. Based on the error analysis, HalluText is constructed targeting the OCR hallucinations. The authors also propose OCRAssistor, which uses a lightweight OCR model to guide a large model during decoding via KL divergence, boosting probabilities for OCR-consistent tokens. This plug-and-play method requires no retraining, and shows improved accuracy across various benchmarks and conditions.

### Strengths
- Paper presentation is clear.

- Comprehensive analysis of VLM OCR hallucination errors and categorization of the error types with different axes.

- Construction of a challenging benchmark for VLM OCR that is specifically designed to target the OCR hallucinations.

- A mitigation method, OCRAssistor, in a training-free setup, that relies on an off-the-shelf open-source OCR to improve the performance.

### Weaknesses
- The paper presents a valuable dual contribution of a new benchmark and a novel method. However, the description of the HalluText benchmark is somewhat brief, which undermines its potential as a foundational tool for the community. While Appendix B outlines the sources for each subset, the overall description remains high-level. Key details regarding the benchmark's construction are missing:
What were the specific criteria for data cleaning and filtering?
How were the question-option pairs generated? 
Were they based on manual templates, or automatically generated using LLMs and then verified? 
How was the challenge and fairness of the distractors ensured?
For instance, in constructing the 'Position' subset, how was the ambiguity between adjacent categories (e.g., 'top-left' vs. 'top') precisely defined and resolved? How many samples were filtered out as a result?
What was the quality control protocol? For example, was there a multi-annotator process with inter-annotator agreement metrics?

I recommend that the authors enhance the main text, perhaps by restructuring, to include more details on the benchmark construction, such as: More detailed data cleaning and annotation protocols; An analysis of the benchmark's internal validity (e.g., inter-subset variability); and its ability to discriminate between models of different capabilities.


- The core of the OCRAssistor method relies on an external, lightweight OCR model to provide 'grounded' visual text cues. However, this OCR model is not infallible and is prone to errors itself (e.g., missed detection, misrecognition). A critical potential impact is that if the OCR model provides incorrect text, the guidance mechanism of OCRAssistor could amplify these errors, leading the LVLM to generate new hallucinations based on faulty OCR results. Although Table 5 briefly mentions the impact of OCR quality, the paper does not deeply investigate the method's robustness in cases of severe OCR failure, such as completely missing a crucial text block in the image.


- The error analysis is not sufficiently detailed. The paper demonstrates overall performance improvements but does not provide a deep dive into the remaining failure cases after applying OCRAssistor. Analyzing these cases is crucial for understanding the method's failure modes and guiding future improvements. For example, when OCRAssistor is applied but the model still fails, is it primarily due to: An error in the OCR output that misled the LVLM? The LVLM ignoring or overriding the OCR guidance? A qualitative analysis of these persistent errors would significantly strengthen the paper's insights.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2
