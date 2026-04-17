# Interpretable Oracle Bone Script Decipherment through Radical and Pictographic Analysis with LVLMs

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
As the oldest mature writing system, Oracle Bone Script (OBS) has long posed significant challenges for archaeological decipherment due to its rarity, abstractness, and pictographic diversity. Recently, deep learning-based methods have made exciting progress on the OBS decipherment task. However, they often ignore the intricate connections between the glyphs and meanings of OBS, resulting in limited generalization and interpretability. 
To this end, we propose an OBS decipherment method based on Large Vision-Language Models, which attempts to bridge the gap between glyphs and meanings and to interpret the deciphering process. Specifically, we propose a progressive training strategy that guides the model from radical analysis to pictographic analysis and then to mutual analysis, enabling it to comprehend the rich semantic information embedded within OBS glyphs. These analysis contents are used to obtain decipherment results (i.e., the corresponding modern Chinese characters), retrieved from a dictionary via our proposed Radical-Pictographic Dual Matching mechanism, thereby allowing the decipherment process to be interpretable. To facilitate model training, we also propose a Pictographic Decipherment OBS Dataset, which comprises 3,173 OBS classes and 47,157 Chinese characters from different dynasties, which is a well-organized dataset containing detailed glyph analysis. Experiments on public benchmarks demonstrate that our method achieves competitive OBS decipherment capabilities and interpretability. Additionally, the interpretability enables our method to provide possible applicable reference content for undeciphered OBS, and thus has potential applications in historical research. 
The dataset and code repository will be released in camera-ready.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes an VLM-based Oracle Bone Script decipherment. A progressive training strategy is introduced to guide the model from radical analysis to pictographic analysis, enabling rich semantic understanding. A Radical-Pictographic Dual Matching mechanism is designed to connect the decipherment results with modern Chinese characters. The authors also introduce a Pictographic Decipherment OBS Dataset, consisting of 3,173 OBS classes and 47,157 Chinese characters with glyph analysis. Experiments show that the proposed method achieves superior performance in OBS decipherment.

### Strengths
1. A VLM-based OBS decipherment model that combines both radical and pictographic information to generate more comprehensive interpretations.
2. The PD-OBS dataset, with its comprehensive radical and pictographic annotations, provides an enduring resource for future research in ancient Chinese character analysis and serves as a benchmark for developing interpretable AI systems in digital humanities applications.

### Weaknesses
1. Using pictographic structures such as radicals to interpret the meanings of oracle bone script is a common approach, and there have been many related studies.
2. PD-OBS, though large, is derived from modern Chinese and known decipherments. This may bias the model toward modern semantics rather than authentic ancient meaning reconstruction. In particular, many undeciphered oracle bone inscriptions have lost their evolutionary path and cannot correspond to modern Chinese characters.
3. Lack of visualization of PD-OBS dataset.
4. As acknowledged by the authors, LoRA fine-tuning reduces base model reasoning ability and can lead to overfitting or reliance on seen glyphs, weakening zero-shot interpretability.
5. While the approach is effective, it primarily involves clever integration and fine-tuning of existing components (LVLM, LoRA, dictionary matching) rather than introducing a fundamentally new modeling theory.
6. Since the PD-OBS dataset uses GPT-4.1 for annotation and expansion, how is annotation quality and potential LLM hallucination controlled or audited? How does the self-checking mechanism of GPT4.1 operate? Will it introduce biases?
7. Lack of cross-font decipherment effect evaluation. The current experiments mainly focus on the handprinted OBS, while most oracle bone inscriptions exist in rubbing form and contain various types of noise.
8. The performance of the overall decipherment framework is limited by the diversity of the constructed PD-OBS dictionary. 
9. What is the specific construction of spatial patch merger in line: 243? Fully connected layer?
10. What mechanism is used to detect the error in A2 (Fig. 4)?
11. The logical expression in Figure 4 needs improvement. Currently, the sequence of the block diagram is difficult to understand.
12. Lack of comparison with the latest visual large models (e.g., Gemini 2.5 Pro, Claude 4.1, GPT-5) with reasoning abilities.
13. Incorrect annotation. For the HUST-OBC column in Table 1, the underlined and bolded parts at Top-10 accuracy.
14. As shown in Tab. 1, the performance improvement on validation set is not significant and even there is a decline, compared with the existing methods.
15. Table 1: What are the evaluation settings for commercial VLMs? Their rationality will significantly impact model output quality.
16. Lack of evaluation for those characters themselves are components. We call these components those with the ability to independently form words.
17. Does the model perform well when confronted with ancient characters that share *no radicals* with known ones? How does it handle rare or unique radicals?
18. The study relies mainly on HUST-OBC and EV-OBC benchmarks. Further human-expert evaluation of interpretability would strengthen the archaeological credibility. 
19. Although BERTScore can capture semantic similarity, existing studies have shown that it is more suitable for evaluating the fluency of generated text rather than the accuracy of content. Further human evaluation is needed to validate its alignment with experts.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper utilizes LVLMs to establish the relationship between the radicals, pictographs, and meanings in Oracle Bone Script and explains the deciphering process. The paper designs a progressive training strategy that guides the model from radical analysis to pictograph analysis, ultimately achieving an interpretable deciphering process. Furthermore, the paper introduces the PD-OBS dataset, which includes Chinese character radicals and detailed glyph analysis. Experiments on publicly available benchmark tests demonstrate that the proposed method performs well in terms of deciphering ability and interpretability.

### Strengths
1.	The use of LVLMs, combined with radical and pictograph analysis, provides excellent interpretability for deciphering Oracle Bone Script.
2.	A progressive training strategy is proposed, gradually transitioning from the radical features of Oracle Bone Script to pictograph analysis, ultimately obtaining a more comprehensive understanding of glyph semantics.
3.	The PD-OBS dataset is introduced, containing detailed pictograph analysis annotations for 3,173 types of Oracle Bone Script and 47,157 types of Chinese characters.
4.	The experiments in the paper are thorough, and the visualization analysis is well done.

### Weaknesses
1.	The paper claims to be the first method to explain the deciphering process through pictograph analysis. However, in reality, both OracleSage and OracleFusion have used Oracle Bone Script components and pictograph information for deciphering. I believe the main contribution of this paper is the construction of a new pipeline that integrates radical and pictograph information for deciphering, rather than being the first to propose it. The term "first" may be misleading and could be off-putting.
2.	Similarly, PD-OBS is not the first dataset annotated with detailed radical and pictograph analysis. Both the OracleSem dataset introduced in OracleSage and the RMOBS dataset proposed in OracleFusion contain similar annotations. The PD-OBS dataset proposed in this paper can be said to be more detailed and larger in scale, but it cannot be called the "first."
3.	The paper claims to achieve state-of-the-art (SOTA) deciphering accuracy, robust zero-shot capability, and interpretability. However, based on Table 1, the method proposed in this paper is not SOTA in Oracle Bone Script recognition and zero-shot deciphering. It is comparable to OBSD in the Top-1 zero-shot task. Therefore, the paper should state that it strikes a good balance across these three aspects, rather than claiming it to be state-of-the-art in all areas.

4.	The components of Oracle Bone Script and the radicals of modern Chinese characters do not necessarily correspond one-to-one. For some Oracle Bone Script characters, the proposed method may not solve the problem effectively. In lines 156-157, the authors mention that the OBSD method has the drawback of producing unpredictable outputs. However, Oracle Bone Script itself may not have a direct correspondence with modern Chinese characters for every glyph. This is actually an advantage of OBSD, as it is not limited to a strict one-to-one correspondence between components and radicals. Therefore, I suggest that the paper should only emphasize OBSD’s output instability and lack of interpretability.
5.	In the OBSD method, the OCR engine provided can recognize nearly 90,000 types of Chinese characters. However, I noticed that the PD-OBS dataset in this paper contains 47,157 types of modern Chinese characters, and in the supplementary materials, I found that some characters lack "reference" annotations. This suggests that the deciphering scope of the proposed method may be limited to these 47,157 characters, which is narrower than the OBSD method. This is a weakness of the paper.
6.	The radical analysis method proposed in the paper seems to only recognize a single radical in Oracle Bone Script, whereas both Oracle Bone Script and modern Chinese characters may consist of multiple radicals. OracleFusion analyzes each component of Oracle Bone Script, and in comparison, the method proposed in this paper has limitations.
7.	There is an error in Figure 1, where two (b) labels appear and no (C) is shown. The reference to the EVOBC dataset is also incorrect; the author should be Guan et al., not Wang et al.

### Questions
I am inclined to accept your paper and increase the score, but I would need you to address each point raised in the Weaknesses section with explanations and clarifications, and remove any absolute or inappropriate wording. Additionally, you should pay attention to the various references and figure details in the paper, as these adjustments will significantly improve the overall quality of the article.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a framework for Oracle Bone Script (OBS) decipherment based on Large Vision-Language Models (LVLMs). The core of the method is a progressive training strategy that moves from radical analysis to pictographic analysis and then to mutual analysis. Instead of direct classification, the framework retrieves the corresponding modern Chinese character from a dictionary using a proposed "Radical-Pictographic Dual Matching" mechanism. To support this approach, the authors have also constructed the PD-OBS dataset. Experiments conducted on the HUST-OBC and EVOBC benchmarks demonstrate that the method achieves high Top-10 accuracy and shows good performance in zero-shot settings, along with enhanced interpretability.

### Strengths
1) An Interpretable Pipeline Based on OBS Structure: The paper introduces a well-structured and interpretable pipeline for decipherment. The methodology is built upon a progressive, three-stage analysis that begins with radical analysis, proceeds to pictographic analysis, and culminates in a mutual analysis stage. This approach is explicitly defined and logically motivated, improving the semantic alignment between ancient glyphs and their modern meanings. The proposed methodology is straightforward and logically presented.
2) Construction of the PD-OBS Dataset with Analysis Annotations: The paper contributes the PD-OBS dataset, which includes detailed analytical annotations. The composition of this dataset is clearly described. The "data engine" employed to generate these annotations integrates authoritative sources with GPT-4.1, followed by both automated self-checking and manual correction steps.

### Weaknesses
1) Limitations in Performance and Incomprehensive Comparisons:
(1.1) The reported performance, while notable in certain metrics, reveals some limitations. For instance, in the validation setting on the HUST-OBC and EVOBC dataset, the model's Top-1 accuracy is below that of the PyGT baseline. And in the zero-shot setting on the HUST-OBC dataset, the model's Top-1 accuracy is slightly below that of the OBSD baseline, with its primary advantage appearing in the Top-10 results (Table 1). This suggests that while the method is effective at generating a list of plausible candidates, its precision in identifying the single best answer at the top rank could be improved.
(1.2) The comparison with commercial Large Vision-Language Models (LVLMs) is not sufficiently comprehensive. The authors note that the tested models (GPT-4.1 and Qwen-VL-Max) yield poor results (<6% Top-1 accuracy). However, the comparison does not include several of the most recent and powerful state-of-the-art multimodal models, such as GPT5, Gemini 2.5 Pro, claude-opus-4-1, grok-4, or others that are considered current leaders in the field. Benchmarking against these would provide a more convincing measure of the proposed method's capabilities.
(1.3) The exclusion of several recent methods on the grounds of "dataset inconsistency" weakens the experimental evaluation. To properly situate the contributions of this work, it is important to include direct comparisons with these relevant baselines on a consistent experimental setup. The authors should therefore re-evaluate these methods or provide a more compelling justification for their omission.
2) Lack of Robustness Analysis for the Multi-Stage Pipeline: The paper proposes a multi-stage pipeline, which inherently raises concerns about error propagation. The performance of later stages likely depends heavily on the accuracy of earlier ones. For instance, it is unclear how the final decipherment would be affected if the initial "Radical Recognition" stage yields an incorrect prediction. The paper does not include ablation studies or a sensitivity analysis to quantify the impact of such upstream errors. This omission makes it difficult to assess the model's overall robustness, especially when dealing with ambiguous or noisy inputs that could lead to failures in the initial stages.
3) Unsubstantiated Qualitative Comparisons in Figure 1: The comparative chart on the right side of Figure 1, which rates different paradigms as "Poor," "Medium," or "Good," lacks rigor. The paper fails to provide any objective criteria, quantitative thresholds, or a clear rubric for what constitutes each of these qualitative labels. Without a defined standard, this assessment appears subjective and potentially misleading. For such a comparison to be meaningful, it should be supported by either direct numerical evidence or a detailed set of criteria that justifies the assigned ratings.

### Questions
Please provide an analysis and explanation of "Weaknesses".

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
The paper addresses the issue of the gap between glyphs and semantics in the field of oracle bone script decipherment and the lack of interpretability. It proposes an interpretable framework for deciphering oracle bone script based on Large Vision-Language Models (LVLMs). Through a progressive training strategy, the model is guided to gradually understand oracle bone script, enabling its decipherment. Additionally, an ideographic decipherment oracle bone script dataset is constructed, which includes 3,173 oracle bone script glyph classes and 47,157 characters from different dynasties. The method is validated on two benchmark datasets, HUST-OBC and EVOBC, demonstrating the effectiveness of the approach.

### Strengths
1.	The progressive training design proposed in the paper aligns with the cognitive logic of oracle bone script, following a training sequence from radicals to pictograms and then to interactions. This sequence is consistent with the evolutionary rule of oracle bone script, which progresses from shape construction to semantic expression, thus avoiding semantic confusion caused by directly analyzing complete characters.

2.	The constructed PD-OBS dataset includes detailed radical and pictogram analysis, providing high-quality benchmark data for future AI methods for oracle bone script.

3.	The overall writing of the paper is clear, the method is well-explained, and the illustrations clearly demonstrate the overall process of the method, making it highly understandable.

### Weaknesses
1.	The paper points out that supervised fine-tuning restricts the model's generalization and inference capabilities to some extent. However, in the task of oracle bone script decipherment, there are many characters with similar shapes or meanings. In zero-shot testing scenarios, the model might rely on information from similar characters in the training labels, rather than performing a complete analysis of radicals and pictograms.

2.	The method proposed in the paper is primarily based on large models and uses progressive training to guide the model in establishing associations between shapes and semantics in stages, addressing the issue of insufficient domain knowledge for oracle bone script in large models. However, this mechanism seems more like a customization for the oracle bone script decipherment task, with overall innovation being somewhat limited.

3.	Current methods for undeciphered oracle bone script mainly provide potential reference content, relying on human experts for further verification. The model's Top-k outputs offer possible references, but the result depends on dictionary matching, which still requires manual intervention in practical archaeological applications.

4.	The details of the method need further refinement. Some formulas lack proper punctuation, key variables are not defined, and some hyperparameters are not explained with their specific values or settings.

### Questions
1.	The paper highlights some hyperparameters, such as how the loss function weights are set. Was hyperparameter sensitivity analysis conducted?

2.	In the radical-pictogram dual matching mechanism, semantic similarity mainly relies on BERT-Score. However, the pre-training data for BERT mainly consists of modern text, with limited semantic information related to oracle bone script. How can it be ensured that semantic differences between ancient and modern texts do not cause biases in the similarity judgment?

3.	In Section 4.2, the paper mentions the design space patch merger as a visual adapter, which is used to downsample visual embeddings and obtain representative feature vectors suitable for classification tasks. What is the specific process here? How can it be ensured that the obtained feature vectors fully consider the structural features of the oracle bone script, especially in complex script structures, and how can the downsampling process avoid losing detailed and structural information?

### Soundness
2

### Presentation
3

### Contribution
2
