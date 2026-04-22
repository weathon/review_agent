# OIG-Bench: A Multi-Agent Annotated Benchmark for Multimodal One-Image Guides Understanding

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Recent advances in Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities. However, evaluating their capacity for human-like understanding in **One‑Image Guides** remains insufficiently explored.  One‑Image Guides are a visual format combining text, imagery, and symbols to present reorganized and structured information for easier comprehension, which are specifically designed for human viewing and inherently embody the characteristics of human perception and understanding. Here, we present **OIG‑Bench**, a comprehensive benchmark focused on One-Image Guide understanding across diverse domains. To reduce the cost of manual annotation, we developed a semi-automated annotation pipeline in which multiple intelligent agents collaborate to generate preliminary image descriptions, assisting humans in constructing image–text pairs. With OIG-Bench, we have conducted comprehensive evaluation of 29 state-of-the-art MLLMs, including both proprietary and open-source models. The results show that Qwen2.5-VL-72B performs the best among the evaluated models, with an overall accuracy of 77%. Nevertheless, all models exhibit notable weaknesses in semantic understanding and logical reasoning, indicating that current MLLMs still struggle to accurately interpret complex visual-text relationships. In addition, we also demonstrate that the proposed multi-agent annotation system outperforms all MLLMs in image captioning, highlighting its potential as both a high-quality image description generator and a valuable tool for future dataset construction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents OIG-Bench, the first benchmark specifically designed to evaluate Multimodal Large Language Models (MLLMs) in understanding "One-Image Guides"—a visual information format that integrates text, images, and symbols to present content in a human-readable, structured manner. The authors develop a semi-automatic annotation pipeline using a multi-agent system to generate high-quality image descriptions, reducing manual labeling costs while ensuring accuracy. They conduct a comprehensive evaluation of 29 state-of-the-art MLLMs (including both commercial and open-source models) on two tasks: description generation and visual question answering (VQA), assessing five key capabilities: semantic consistency, detail coverage, hallucination control, reasoning ability, and text recognition. Results indicate that Qwen2.5-VL-72B performs best overall, but all models exhibit significant shortcomings in semantic understanding and logical reasoning. The proposed multi-agent system outperforms individual MLLMs in description generation, demonstrating its potential as a tool for high-quality dataset construction.

### Strengths
Originality: The introduction and formalization of the "One-Image Guide" (OIG) as a distinct and challenging domain for MLLM evaluation is a key contribution. The proposed semi-automatic, multi-agent annotation pipeline can obtain high-quality, detailed annotations

Significance: This paper provides a more realistic and demanding testbed than many existing benchmarks, by focusing on a format that is ubiquitous in real-life (guides, infographics, tutorials) yet challenging for models. The multi-agent annotation pipeline offers a scalable method for generating high-quality data.

Clarity: The paper is well-structured and clear.

### Weaknesses
1. In the introduction, the authors mention that “To achieve a complete and accurate understanding of one-image guides, MLLMs must possess fine-grained visual perception (e.g., text and details recognition) and spatial structure parsing capabilities (e.g., interpreting layouts, arrows, and segmented regions).” However, there is no detailed evaluation dimension for spatial parsing ability, such as arrows or segmentation, and it seems that the evaluation could be more comprehensive.

2. The evaluation of cross-modal reasoning capabilities could be more comprehensive. The authors claim that “Experimental results show that current MLLMs still exhibit limitations in semantic understanding and logical reasoning.” However, semantic understanding and logical reasoning seem to have a causal relationship. If a model achieves good semantic understanding of an image (i.e., converting the image into an equivalent textual representation), is it possible that its logical reasoning ability could actually be strong? Which specific dimensions of semantic understanding hinder logical reasoning? Which dimensions still lead to reasoning errors even when semantic understanding is relatively good? The evaluation could be made more fine-grained.

3. The paper does not provide an ablation study to dissect the contribution of each agent. Report the performance (on the same description generation metrics) of: a. A single best model (e.g., GPT-4o alone). b. Multiple models without OCR correction. c. Multiple models with OCR correction but without the final summarization agent (e.g., just picking the "best" description). d. The full pipeline.

### Questions
1. Figure 5(b) is unclear, so it is not possible to tell how the ground-truth answer was obtained.

2. Lacks key analysis of OIG-Bench, such as the average/max number of words in the description generated by the model, the average/max number of words in the ground-truth description, and the numbers of single-choice questions and multiple-choice questions.

3. It seems that there is only a reasoning prompt for single-choice questions (“Prompt to answer the question with a given image”), while a reasoning prompt for multiple-choice questions is missing. What is the performance difference between single-choice and multiple-choice questions? What conclusions can be drawn?

4. What is the exact procedure for training and conducting manual annotation? Who performed the manual annotation? What was the specific cost of the manual annotation?

5. Lacks experiments on performance comparison between English and Chinese.

### Soundness
2

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
Aiming at the problem that the understanding ability of multimodal large language models (MLLMs) in the "One-Image Guide" is insufficient, this paper proposes the first benchmark dataset OIG-Bench focusing on this scenario, and designs a multi-agent semi-automatic labeling process to reduce the construction cost. At the same time, 29 mainstream MLLMs are systematically evaluated. The research motivation is clear, and the paper fills the evaluation gap of "human-like cognitive level single-graph guide understanding". Data screening combines MLLM cross-validation and manual review to evaluate the core capabilities of dimension coverage and human verification.

### Strengths
1. Existing multimodal benchmarks mostly focus on diagrams/documents of layout rules, while "single-graph guide", as a human-friendly visual format that integrates text, images, and symbols (such as travel route maps, game guide maps), is more close to the complex information presentation of real scenes, but there is a lack of specific evaluation benchmarks before. This paper takes this scene as the research object for the first time, and clarifies its "human-like cognitive characteristics" and the three major challenges faced by MLLMs: fine grain visual details, complex logical structures, and cross-modal reasoning. The research scene is innovative and practical.
2. The evaluation plan is well designed. The image description generation task covers "text recognition, semantic consistency, detail coverage, illusion control", and the VQA task focuses on "reasoning ability", which are the core capabilities of single-image guide understanding.
3. Evaluate 29 MLLMs (including 5 closed-source and 24 open-source ones, covering scales from 6B to 78B), with unified prompt templates, temperature=0, and other experimental settings, to conduct a fair comparison of the performance of different models, across domains (games/travel/medical/food), and graph types (layout diagrams/logic diagrams).

### Weaknesses
1. The document mentions that the data is collected from Xiaohongshu, Baidu, and Google, but only states that "anonymization ensures privacy," and does not mention whether it is authorized by the platform or uses "publicly available and copyright-free content," which poses a risk of infringement. It is recommended to supplement the "Data Copyright Compliance Statement" to clarify that the data source is publicly available for non-commercial use, or has been authorized by the platform for research.
2. Existing experiments only demonstrate the overall performance of multi-agents (e.g., 86.4% semantic consistency, 96.8% text recognition), and do not validate the necessity of each component of "description-generated agent, OCR-generated agent, OCR-corrected agent, summary agent". It is recommended to supplement ablation experiments, such as the change in semantic consistency after removing OCR-corrected agents, and the performance difference between agents and multi-agents generated using only a single description, to strengthen the persuasiveness of the advantages of multi-agent collaboration.
3. The existing analysis only compares "image entropy" and fails to compare the performance of OIG-Bench with existing benchmarks on the same model, making it difficult to intuitively demonstrate the unique value of OIG-Bench. It is recommended to select multiple representative models, reproduce and evaluate them on the "description generation" and "VQA" tasks of InfographicVQA, compare the score differences between OIG-Bench and InfographicVQA, quantify the challenges of OIG-Bench, and strengthen the argument of "filling the evaluation gap".

### Questions
The OIG-Bench benchmark and multi-agent labeling method proposed in this paper provide a high-quality tool for the evaluation of "single-graph guide understanding" of MLLMs, and the experimental results have important guiding significance for model optimization and practical application. If the above ablation experiments and compliance instructions can be supplemented, the scientificity, integrity and persuasiveness of the paper will be further enhanced.

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
4

### Summary
This paper introduces an evaluation benchmark for MLLMs called OIG-Bench, which focuses on one-image guide understanding. To build this benchmark, the authors employ a multi-agent-based semi-automated construction method that effectively generates image descriptions and reduces manual annotation costs. Evaluation of 29 mainstream MLLMs across five dimensions of one-image guide understanding shows that existing models still have substantial room for improvement in semantic understanding and logical reasoning.

### Strengths
- The paper is well-written and easy to follow, with clear visualizations.
- A more complex benchmark on infographics is constructed, whose data contain richer visual logical and textual content.
- An effective multi-agent architecture is proposed for dataset construction, which achieves strong results across all metrics, and can benefit future works.

### Weaknesses
- Multi-option bias: The multi-agent system generates options based on descriptions. Could this introduce bias in MLLMs if the target option distribution is not uniform, especially without post-processing such as shuffling?
- Complexity metrics: Image entropy may not sufficiently reflect structural complexity. For example, a full-text image could also yield high image entropy.
- Need more examples: In line 465, "One possible explanation is that…" would be more convincing with additional failure cases.
- Some typos: For instance, in line 300, "log" -> "\log".

### Questions
- Could the multi-agent system introduce bias in the target option distribution during the construction of benchmark, if no post-processing (such as shuffling) is applied?
- Although overly simple images are filtered out, the complexity of the remaining data is not clearly explained. How is the difficulty of the generated questions or options evaluated?
- Image entropy may not be a convincing metric for structural complexity. Are there potentially better evaluation metrics?

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
This paper addresses the gap in evaluating MLLMs’ human-like understanding of **One-Image Guides (OIGs)**—text-imagery-symbol composites optimized for human cognition—by proposing **OIG-Bench**, the first dedicated benchmark. Key contributions include: (1) OIG-Bench itself: 808 bilingual (56.6% Chinese, 43.4% English) OIGs across 4 domains (game, travel, food, medicine) with 2800 reasoning questions, featuring higher visual complexity (average entropy 5.8) than existing benchmarks (e.g., InfographicVQA: 4.8). (2) A **semi-automated multi-agent annotation pipeline**: specialized agents (description generation via GPT-4o/Claude-4, OCR via PaddleOCR, correction/summarization via GPT-4.1) collaborate to generate high-quality labels, reducing manual effort and outperforming single models. (3) Comprehensive evaluation of 29 MLLMs (6B–78B, proprietary/open-source): Qwen2.5-VL-72B achieves the best overall accuracy (77%), but all models struggle with semantic consistency and logical reasoning. The paper also reveals domain/type disparities (game/travel harder than food/medicine; logic diagrams harder than layout) and prompt trade-offs (CoT aids VQA but harms descriptions).

### Strengths
* The paper identifies a clear and valuable gap in MLLM evaluation. "One-Image Guides" are a distinct, commonly used format that is more complex and less rigidly structured than standard charts or documents. The quantitative analysis showing OIG-Bench images have a higher average image entropy than related benchmarks (InfographicVQA, SEED-Bench-2-Plus) supports the claim of higher visual complexity.
- The paper thoroughly evaluates its framework against relevant and strong baselines, including "Strong-to-Weak Distillation" and "Multi-Agent Collaboration". The method is tested across a wide array of 11+ benchmarks covering comprehensive understanding, hallucination, chart/table understanding, and knowledge-oriented tasks, demonstrating broad improvements.
- The paper provides a thorough evaluation of 29 state-of-the-art MLLMs, including both open-source and proprietary models. The findings are insightful, pinpointing a clear weakness across all models: they perform well on "Text Recognition" but "struggle to accurately interpret complex visual-text relationships".

### Weaknesses
* The evaluation for both tasks (Description Generation and VQA) relies heavily on GPT-4.1 as the "Judge Model". While the authors provide a human correlation study (Table 5, Fig. 8) for two models , this reliance is a potential source of bias. The judge model's own limitations or stylistic preferences could unfairly penalize or reward certain models, and its ability to robustly evaluate 27 other models for complex reasoning is not fully established.
* While OIG-Bench is bilingual, there is no analysis of model performance disparities between Chinese and English OIGs.
* The benchmark contains 808 images. While the filtering process is rigorous, this is a relatively small dataset. Furthermore, the domain distribution is skewed, with "Game" (31.2%) and "Travel" (30%) accounting for over 60% of the data. The paper notes that models perform worst in these two domains, which means the overall performance scores are heavily weighted by these challenging, but potentially niche, domains.
* The authors need to provide more reasoning MLLMs to support experimental conclusions, such as Gemini-2.5-pro and o3.
* The data and code not be publicly released.

### Questions
* The multi-agent annotation system (Section 3.1.2) uses GPT-4.1 for both the "OCR Correction Agent" and the "Description Summarize Agent". The evaluation (Section 3.2) also uses GPT-4.1 as the primary judge model. How do you account for the potential bias where the evaluation judge (GPT-4.1) might favor the stylistic, structural, or reasoning patterns of the same model family used in the data annotation pipeline?
* Do models perform differently on Chinese vs. English OIGs? If yes, is the gap due to OCR accuracy (e.g., worse English OCR), language model proficiency, or annotation differences?

I look forward to an active discussion with the authors during the rebuttal phase and will revise my score accordingly.

### Soundness
3

### Presentation
3

### Contribution
3
