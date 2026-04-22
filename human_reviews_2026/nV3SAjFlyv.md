# Rethinking Radiology Report Generation: From Narrative Flow to Topic-Guided Findings

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Vision-Language Models (VLMs) for radiology report generation are typically trained to mimic the narrative flow of human experts. However, we identify a potential limitation in this conventional paradigm. We hypothesize that optimizing for narrative coherence encourages models to rely on linguistic priors and inter-sentence correlations, which can weaken their grounding in direct visual evidence and lead to factual inaccuracies. To investigate this, we design a controlled experiment demonstrating that as textual context increases, a model's reliance on the input image systematically decays. We propose LLaVA-TA (Topic-guided and Anatomy-aware), a new fine-tuning framework that directly addresses this challenge by re-engineering the generation process. Instead of producing a linear narrative, LLaVA-TA decomposes the report into a set of independent, clinically-relevant topics. By training the model to generate a discrete finding for each topic conditioned on both the full image and its corresponding anatomical region, we reduce the model's reliance on narrative flow and enforce stricter visual grounding. Our experiments show that LLaVA-TA sets a new state of the art on the MIMIC-CXR dataset, significantly improving clinical accuracy on metrics like RadGraph F1 (from 29.4 to 44.0) and CheXpert F1-14 (from 39.5 to 71.5) over strong baselines. Our work demonstrates that dismantling a report's narrative structure to enforce independent, visually-grounded observations is a crucial and effective step toward building more accurate and reliable medical VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a method for radiology report generation. It it based on an earlier method (LLavaRad), where the major extension is that it performs a segmentation of the radiology images and that the system is trained using these segmented images, instead of the image a a whole. The segments obtained are then linked to textual topics. A significant improvement over the existing method is obtained although less dramatic for the improvement over LlavaRad.

### Strengths
- Very well written paper where the method is first motivated by experimentally showing that using longer contextual information on the text side leads to less importance of the image information.
- There is a clear method defined.
- The experiments are all answering a specific question for the methodology. 
- The results are significantly better than the baseline method

### Weaknesses
- Somewhat limited in terms of innovation over the existing method
- The literature could be more recent (limited number of papers from 2025/2024) although this is a hot topic in many venues. Also many refs are basic elements used and not really related work. 
- Ignores uncertain information in the caption while they might be highly relevant. for a clinician / doctor 
- Focus on one family of models (LLava)
- The rad graph F1 measure is core. Although it is not a contribution of the paper as coming from another reference, it needs more explanation and embedding in the paper. No detailed description needed but at least enough to relate it to the method and the experiments. 

Small typo:
P4: should be <image> now it has strange symbols

### Questions
- what are exactly the changes compared to the method in the reference i.e. LlavaRad
- you indicate that you are throwing away uncertain statement like "there might be ...". But aren't these important for the doctor who wrote that report? Aren't you throwing away elements that are uncertain but relevant? 
- By modelling the data in terms of topic there seems a clear connection to medical knowledge graphs. Is that indeed the case (you do mention an ontology but it remains a bit vague what that is).

### Soundness
3

### Presentation
4

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
This paper addresses a fundamental limitation in current Vision-Language Models (VLMs) for radiology report generation (RRG): their tendency to prioritize narrative coherence over visual grounding. The authors argue that conventional sequential, narrative-based training encourages models to rely excessively on linguistic priors and inter-sentence correlations, which can undermine factual accuracy and lead to clinical hallucinations.

### Strengths
1. The method is novel, combining with anatomy.
2. Strong Empirical Evidence for the Hypothesis

### Weaknesses
1. Dependence on External Tools and Heuristics:
The framework relies heavily on external systems — DeepSeek-V3 for report decomposition and CXAS for anatomical segmentation. This raises questions about scalability, potential propagation of upstream errors, and domain transferability.
2. Theoretical grounding is limited:
The paper lacks a formal definition of “narrative bias” and a clear explanation of how topic-guided supervision improves representation learning.
3. Lack of Theoretical Justification for Topic Independence Assumption:
The core idea of decomposing radiology reports into independent topics presumes that findings from different anatomical regions are conditionally independent given the image. In reality, many radiological pathologies exhibit inter-regional correlations (e.g., cardiomegaly co-occurring with pulmonary edema).  It may conduct ablation experiments on different partitioned parts to demonstrate the impact of combined inputs and single inputs on the results.

### Questions
see weakness

### Soundness
3

### Presentation
3

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
This paper addresses a critical issue in vision–language models for radiology report generation: the over-reliance on narrative flow, which leads to factual inaccuracies and weak visual grounding. The authors empirically demonstrate that as textual context increases, models such as LLaVA-Rad rely less on the image itself. To mitigate this, they propose LLaVA-TA, a topic-guided and anatomy-aware fine-tuning framework that decomposes radiology reports into discrete, topic-specific findings (e.g., lungs, heart, pleura) rather than generating a sequential narrative. The framework uses DeepSeek-V3 to split reports by topic, a segmentation model (CXAS) to provide anatomical masks, and trains Vicuna-7B-based models to generate topic-specific sentences conditioned on both the global and local images. Experiments on MIMIC-CXR and IU-Xray show strong improvements in RadGraph F1 and CheXpert metrics, significantly outperforming LLaVA-Rad and other medical VLMs. The authors further analyze model interpretability through attention maps and discuss limitations around static topic ontologies and reliance on external segmentation models.

### Strengths
The paper presents a clever and well-motivated reformulation of the radiology report generation problem. By decomposing reports into independent topics, it identifies a key failure mode—narrative bias—that has not been explicitly quantified before. The authors provide clear empirical evidence that textual coherence can override visual grounding, demonstrated through the “blank image” experiment. The proposed LLaVA-TA framework is conceptually simple yet effective, improving factual accuracy without resorting to larger model scaling. The paper’s experimental rigor is commendable, including ablations on topic disentanglement and anatomy-aware guidance, parameter-efficient fine-tuning, and out-of-distribution generalization. Visualizations of attention maps lend interpretability and practical relevance for clinical use. Overall, the work offers a valuable diagnostic insight into a core weakness of narrative-based report generation.

### Weaknesses
While the empirical findings are interesting, the novelty and technical depth of the proposed method are somewhat limited. The “topic-guided generation” mainly relies on pre-processing via an external LLM (DeepSeek-V3) and a segmentation model, combined with a straightforward adaptation of LLaVA-Rad; the contribution lies more in data structuring than algorithmic innovation. The reliance on hand-crafted topic ontologies and static segmentation mappings reduces generalizability, especially for unseen pathologies or imaging modalities. Moreover, while the study provides large quantitative gains, it is unclear whether these improvements persist in realistic clinical report generation, where topics and findings are interdependent and contextually nuanced. The “report-level” evaluation remains partially artificial, as the model is prompted with ground-truth topic sets rather than autonomously selecting them, making the benchmark setup favorable to the proposed method.

In addition, although the attention visualizations are appealing, they are qualitative and anecdotal, lacking rigorous evaluation of interpretability or human expert validation. The paper also misses a more critical discussion of semantic completeness—whether topic-wise independence may lead to omission of cross-topic findings (e.g., cardiopulmonary interactions). Finally, despite the strong experimental section, the writing can feel overly verbose, with extensive metrics and tables that could be summarized more effectively, while the theoretical grounding of the “narrative flow” hypothesis remains intuitive rather than formalized.

### Questions
Can the authors quantify how topic decomposition affects semantic completeness or diagnostic recall compared to full-narrative generation?

How sensitive is the model to the choice of ontology or to errors in DeepSeek-V3’s report splitting?

Could this approach generalize to multimodal datasets (CT, MRI) or longitudinal exams where temporal context is key?

Is there any evidence that radiologists prefer topic-wise generation in practice, or that this improves interpretability during review?

The experiments assume ground-truth disease labels at inference; how would the model perform without this assumption?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper identifies a key flaw in current radiology report generation models: their narrative flow causes them to rely increasingly on language context rather than visual evidence, leading to hallucinations and factual errors. To address this, the authors propose LLaVA-TA (Topic-guided and Anatomy-aware), which restructures report generation into independent, topic-specific sentences aligned with corresponding anatomical regions. By guiding the model to generate findings per topic instead of following a continuous narrative, LLaVA-TA significantly improves factual accuracy, visual grounding, and interpretability across multiple radiology benchmarks.

### Strengths
The paper starts from a very insightful and underexplored motivation: rethinking why current radiology report generation models tend to overproduce normal descriptions and underrepresent abnormalities. Instead of merely improving architectures or datasets, the authors identify a fundamental bias in the narrative flow paradigm, where sequential language modeling causes models to ignore visual cues and rely excessively on linguistic priors.

The proposed topic-guided and anatomy-aware framework is a clear and elegant solution: by decomposing reports into topic-level units and generating findings region by region, the model enforces stronger visual grounding and factual consistency. This design directly addresses the identified failure mode rather than treating it as noise or data imbalance.

The experiments are comprehensive and convincing, showing substantial gains across factuality and grounding metrics on multiple benchmarks. The presentation is also clear and well-structured, with strong empirical evidence and visual explanations supporting the claims.

### Weaknesses
While the paper presents a clear and well-motivated framework, there are a few important limitations that should be addressed to strengthen the work.

First, the method design could be made more comprehensive. Unlike prior work such as RGRG, which explicitly classifies each anatomical region as normal, abnormal, or not described, this paper only generates topic-specific sentences without explicitly distinguishing whether a region is normal or contains pathology. Such classification could make the model more interpretable and closer to real radiology reasoning, where identifying both the presence and absence of findings is equally important. Integrating this step could also prevent redundant or missing descriptions during generation.

Second, the evaluation setup raises potential fairness concerns. The proposed approach simplifies the report structure by rewriting reports into short, topic-level factual statements, removing stylistic and narrative variations. This preprocessing makes the output text inherently easier to match with reference reports under metrics like BLEU or RadGraph F1. In contrast, existing baselines such as R2Gen or RGRG are evaluated on the original, noisier report format, where stylistic diversity and longer narratives make the task harder. To ensure fair comparison, the authors should retrain or evaluate prior models on the same topic-decomposed version of the dataset, or at least quantify how much of the performance gain comes from dataset simplification versus model improvements.

Overall, while the paper introduces a meaningful direction and shows strong results, its methodological completeness and experimental fairness could be improved. A more balanced evaluation, both in terms of report structure and the explicit modeling of normal/abnormal reasonin, would make the contribution more convincing.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
2
