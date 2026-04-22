# MARVIS: Modality Adaptive Reasoning over VISualizations

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 2

## Abstract
Predictive applications of machine learning often rely on small (sub 1 Bn parameter)  specialized models tuned to particular domains or modalities. Such models often achieve excellent performance, but lack flexibility. LLMs and VLMs offer versatility, but typically underperform specialized predictors, especially on non-traditional modalities and long-tail domains, and introduce risks of data exposure. We propose MARVIS (Modality Adaptive Reasoning over VISualizations), a training-free method that enables small vision-language models to solve predictive tasks on any data modality with high accuracy, and without exposing private data to the VLM. MARVIS transforms latent embedding spaces into visual representations and then leverages the spatial and fine-grained reasoning skills of VLMs to interpret the visualizations and utilize them for predictions successfully. MARVIS achieves competitive performance on vision, audio, biological, and tabular domains using a single 3B parameter model, achieving results that beat Gemini 2.0 by 16\% on average. MARVIS drastically reduces the gap between LLM/VLMs approaches and specialized domain-specific methods, without exposing sensitive data or requiring any domain-specific training. We open source our code and datasets at https://anonymous.4open.science/r/marvis-6F54

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper is motivated by the observation that although existing LLMs and VLMs demonstrate strong flexibility and generalization capabilities, their performance on specialized tasks still lags far behind that of dedicated predictors. To address this gap, the authors propose MARVIS, a framework that enriches VLMs’ understanding of task-specific contexts by embedding data and enabling the models to analyze various statistical representations—such as t-SNE plots, k-NN graphs, and related forms. In experiments, MARVIS significantly outperforms baseline LLMs and VLMs, achieves performance comparable to specialized predictors, and even surpasses them on certain tasks.

### Strengths
1. The motivation is well-founded. Exploring the gap between general intelligence and specialized intelligence is both an interesting and valuable research direction.

2. MARVIS demonstrates strong effectiveness: as shown in Figure 1 and Table 1, VLMs equipped with MARVIS achieve performance comparable to specialized predictors, effectively bridging this gap.

3. The methodology is diverse and thorough. To identify the optimal statistical representation, the authors experimented with 25 different configurations, reflecting substantial exploration and rigor in design.

### Weaknesses
1. Some aspects of the background description are not entirely appropriate. To my understanding, the authors’ motivation has two main components: 1) the gap between general and specialized predictors, and 2) the issue of data leakage. The authors emphasize their goal of "combining the reasoning capabilities of LLMs without requiring modality-specific fine-tuning or exposing sensitive data". However, citing data leakage as a justification for avoiding fine-tuning may not be fully convincing, since the tasks explored in this paper mainly involve common, traditional benchmarks. That said, I do recognize the value of exploring zero-shot methods to enhance existing VLMs; I simply believe the authors should provide a stronger and more compelling rationale for their position.

2. The method description is too brief and lacks clarity. In Section 2, the authors present their proposed method, MARVIS, and divide the pipeline into four stages. However, each stage is described only superficially. In the Embedding Generation stage, it remains unclear which embedding models are used for different tasks. In the Dimensionality Reduction stage, what exactly serves as the input to t-SNE — all training data points or only a subset? This section lacks a formal formulation or mathematical definition. In the Visual Reasoning stage, the authors rely solely on Qwen 2.5-VL 3B as the base VLM. Why was this particular model chosen, and why were larger or alternative VLM series not considered? Is MARVIS specifically tailored to Qwen 2.5-VL 3B, or is it intended to generalize across other architectures? Furthermore, besides the t-SNE and k-NN visual inputs, do the VLMs also take the original images as input? Overall, while the pipeline appears simple in structure, its presentation is unclear and potentially confusing due to the lack of detailed methodological explanation.

Overall, the paper reads more like a technical report than an academic paper. Almost every paragraph includes a link to the appendix, suggesting that the main text merely provides a high-level overview while most of the essential content resides in the supplementary materials. Many critical methodological details, such as the operations of t-SNE, k-NN, and perturbation, are barely discussed in the main body. Moreover, the presentation and layout are problematic: several figures with sparse information occupy nearly an entire page, leaving little room for substantive explanation. As a result, readers must constantly refer to the extensive appendix to understand even the basic methodology. It is unclear whether this structure aligns with standard academic publication practices.

### Questions
All of my questions fall under the weaknesses section.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces MARVIS, a training-free method that enables VLMs to perform predictive tasks on any data modality. The core idea is to:
1. Use a specialized model to create latent embeddings of the data.
2. Plot these embeddings (and a query point) onto a 2D visualization using t-SNE.
3. Feed this image to a standard VLM, which then reasons visually about the query point's cluster and neighbors to make a prediction.

This approach achieves performance competitive with specialized models while significantly outperforming other LLM baselines. Its key advantages are data privacy and universality.

### Strengths
1. The method is training-free yet achieves results close to specialist models across diverse modalities with a single 3B VLM.
2. Solves a key adoption blocker by ensuring the VLM never ingests the sensitive raw data, only an anonymized plot.
3. Requires no domain-specific finetuning, making it far cheaper and easier to deploy than training specialist models.

### Weaknesses
1. Performance is strictly capped by the quality of the upstream embedding model. MARVIS cannot outperform its embedding source.
2. The method relies on t-SNE, which is computationally slow and becomes visually illegible on datasets with millions of data points, limiting its use on web-scale data.
3. Introduces a new set of visualization hyperparameters (e.g., t-SNE perplexity) that must be tuned for optimal VLM performance.

### Questions
1. How does the method handle tasks with thousands of classes? A t-SNE plot with 1000+ colors seems visually unmanageable for a VLM.
2. For the tsne_knn method, how much performance comes from true spatial reasoning versus just parsing the explicit KNN text/pie chart provided in the image?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MARVIS, a training-free framework that enables VLMs to perform predictive tasks across a wide range of data modalities. The main idea is to first generate latent embeddings of the data using a specialized model, then create a 2D visualization of this embedding space using dimensionality reduction techniques. A VLM is then prompted to reason over this visualization to make a prediction. The authors demonstrate that this approach achieves performance comparable to specialized models and yields better results than other LLM/VLM-based methods.

### Strengths
- The paper recasts diverse modalities as a common visual interface to a VLM, using a simple embed→visualize→infer pipeline that avoids modality‐specific fusion or fine-tuning. 
- This work facilitates reproducibility by releasing code and evaluation scripts and by introducing two semantically annotated benchmarks—CC18-Semantic and Regression2025-Semantic—which address the gap in semantic labels for the evaluation of tabular data.

### Weaknesses
- Unconvincing example for problem-specific reasoning capabilities
The authors claim that MARVIS enables transparent, interactive reasoning beyond black-box predictions. However, their main evidence, the dialogue in Fig. 4, is internally inconsistent and unconvincing. In that dialogue, the VLM says, “While I cannot directly visualize the data here,” indicating it is not reasoning over the provided image. This contradicts MARVIS’s core basis of image-grounded reasoning. Consequently, its subsequent "insights" are only generic textbook definitions of t-SNE and universally applicable advice, such as "Increase Training Data." These responses could have been generated without any visual input, creating a notable gap between the paper's claims and its evidence. As a result, this claimed benefit of MARVIS remains unsubstantiated, and its practical utility for generating genuine, data-driven insights is questionable.

- Overstated claim on “training-free”
The "training-free" claim is overstated because the method's effectiveness depends on a complex, sensitive "visualization engineering" stage that requires dataset-specific tuning. According to the ablation study in Figure 3, the choice of visualization strategy alone accounts for a range of ~25% to over 50% accuracy. This reveals that the method is not robust to these choices. The burden of optimization is simply shifted from model fine-tuning to a manual, heuristic-driven search over a new set of hyperparameters in the visualization pipeline. These include not only the choice of context strategy (tsne_knn, tsne_semantic_axes, etc.) but also t-SNE parameters and visual settings, such as the "zoom factor."
Moreover, the paper offers no clear principles for selecting these parameters a priori for a new task. This means achieving the reported performance would likely require a costly, trial-and-error process, weakening the "plug-and-play" utility of the proposed framework.

- Lack  of justification for the default experimental setting
The experiments in Table 1 use the tsne_knn configuration, yet the ablation in Figure 3 indicates that other configurations, such as tsne_perturbation_axes, achieve higher accuracy. The justification for this choice is that the configuration "exposes less information... and therefore better reflects real-world use," which is unconvincing and requires further elaboration.

- Inference latency
The paper reports an inference time of 0.5-2.0 seconds per sample. While acceptable for some interactive use cases, this is much slower than a typical specialized model, which performs a single, fast forward pass. This latency could be a bottleneck for applications requiring large-scale batch processing or real-time responses. A more direct comparison of inference speed against the baselines would be beneficial.

Minor Issues
- Line 102: Github -> GitHub
- Line 187: T-SNe -> t-SNE
- Line 319: TSNe -> t-SNE
- In Figure 4, the text within the dialogue bubbles is difficult to read due to its small size and has been truncated with ellipses despite ample empty space.

### Questions
- Computational Cost: Could you please elaborate on the inference latency? How does the 0.5-2.0s per sample compare to the specialized model baselines? Are there opportunities to optimize this, for example, by batching the generation of visualizations or the VLM inference step?

- Choice of Visualization Method: The tsne_knn method was used for the main experiments, despite tsne_perturbation_axes and tsne_semantic_axes showing higher mean accuracy in the ablation study. Could you further justify this choice? In particular, could you elaborate on the statement that the chosen method “exposes less information” and “better reflects real-world use”? Is the performance gap between these top configurations statistically significant?

- Robustness to Embedding Quality: The experiments use state-of-the-art embedding models. How does MARVIS perform if a weaker or less suitable embedding model is used for a given modality? Does the performance degrade gracefully, or does the approach fail if the embedding space is not well-structured?

- Baseline Performance on Regression: In the tabular regression results (Table 1), the LLM/VLM baseline (JOLT) has an R2 score of just 5.1, while MARVIS achieves 66.0. This gap is exceptionally large compared to other tasks. Could you confirm this baseline result is correct and, if so, offer any insights into why existing LLM-based tabular methods fail so dramatically on this benchmark while MARVIS succeeds?

### Soundness
2

### Presentation
3

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
This paper aims to investigate the reasoning capabilities of large language models (LLMs) and the representational power of predictive models without requiring modality-specific fine-tuning or access to sensitive data. The core idea is to project data from any modality into a meaningful embedding space, where visualization of these embeddings enables vision-language models (VLMs) to perform accurate reasoning without modality-specific training.

### Strengths
1. The idea of performing embedding visualization and using it as a shared representation space across different data modalities is interesting.

2. Without additional fine-tuning, the combination of feature visualization and VLM-based reasoning demonstrates strong performance on several predictive tasks across diverse modalities.

### Weaknesses
1. Although the proposed solution is clearly presented, the connection between the visualized features and how VLMs perform accurate reasoning based on them is not well established. In addition, it remains unclear whether the visualized features introduce potential out-of-distribution (OOD) issues.

2. The overall solution lacks intuitiveness. While the authors make considerable effort to explain their approach, the rationale behind why it works is not sufficiently analyzed. Further investigation into key components, such as feature selection, would strengthen the paper.

### Questions
1. The main contribution of this paper lies in exploring feature visualization combined with VLM-based reasoning for zero-shot prediction. It is strongly recommended to provide a clearer visualization of the extracted features.

2. When feeding the feature visualizations into VLMs for reasoning, are there any out-of-distribution (OOD) issues? How robust is the model to such distributional shifts?

3. Given the central role of feature visualization, it is also suggested to include further analysis on the processing of features extracted from foundation models. For example, how would feature selection or dimensionality reduction affect performance? How robust is the model to different feature processing techniques?

4. The reported performance on multi-modal data is promising; however, it remains unclear whether the comparisons are fair. Additional analysis and justification are needed.

### Soundness
3

### Presentation
1

### Contribution
2
