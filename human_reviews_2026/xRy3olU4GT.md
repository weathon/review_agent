# FUAS-Agents: Autonomous Multi-Modal LLM Agents for Treatment Planning in Focused Ultrasound Ablation Surgery

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 8, 4

## Abstract
Focused Ultrasound Ablation Surgery (FUAS) has emerged as a promising non-invasive therapeutic modality, valued for its safety and precision. Nevertheless, its clinical implementation entails intricate tasks such as multimodal image interpretation, personalized dose planning, and real-time intraoperative decision-making processes that demand intelligent assistance to improve efficiency and reliability.
We introduce FUAS-Agents, an autonomous agent system that leverages the multimodal understanding and tool-using capabilities of large language models (LLMs). By integrating patient profiles and MRI data, FUAS-Agents orchestrates a suite of specialized medical AI tools, including segmentation, treatment dose prediction, and clinical guideline retrieval, to generate personalized treatment plans comprising MRI image, dose parameters, and therapeutic strategies. The system also incorporates an internal quality control and reflection mechanism, ensuring consistency and robustness of the outputs.
We evaluate the system in a uterine fibroid treatment scenario. Human assessment by four senior FUAS experts indicates that 82.5\%, 82.5\%, 87.5\%, and 97.5\% of the generated plans were rated 4 or above (on a 5-point scale) in terms of completeness, accuracy, fluency, and clinical compliance, respectively. In addition, we have conducted ablation studies to systematically examine the contribution of each component to the overall performance. These results demonstrate the potential of LLM-driven agents in enhancing decision-making across complex clinical workflows, and exemplify a translational paradigm that combines general-purpose models with specialized expert systems to solve practical challenges in vertical healthcare domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents FUAS-Agents, an autonomous multi-agent system leveraging multimodal LLMs for treatment planning in Focused Ultrasound Ablation Surgery (FUAS). By integrating patient profiles and MRI data, the system orchestrates specialized medical AI tools, such as segmentation, dose prediction, and guideline retrieval, to generate personalized surgical plans. The framework combines general-purpose foundation models with domain-specific expert systems, incorporates internal quality control, and demonstrates strong performance in uterine fibroid treatment scenarios, as validated by expert assessment. The approach highlights the potential of agent-based AI for complex, personalized clinical decision-making in FUAS and broader healthcare applications.

### Strengths
•	The paper tackles a highly relevant clinical problem and proposes a modular, multi-agent framework for FUAS treatment planning that is well-aligned with real clinical workflows. 
•	The integration of multimodal data sources and the use of expert-annotated clinical data enhance the potential impact of the work.

### Weaknesses
•	Unclear or missing description of the internals of individual agent. Undefined notations and semantics of inputs, outputs, and intermediate variables. These make it impossible to understand the contributions of this work.
•	Insufficient benchmarking (single and small dataset, single task) against baselines, also potentially unfair benchmarking.

### Questions
+ The agents are described abstractly, with no concrete details, pseudocode, or examples describing the internal workflow.
++ Planned agent does not describe how clinicians’ instructions are interpreted or patient data is organized to decide and invoke the subtasks.
++ Strategy agent does not describe how segmentation and dose results are combined with patient records, MRI reports and domain knowledge.
++ FUAS model fine-tuning:
+++ Undefined notations:
++++ Key notations (M2F, T2F, F2I, M, T, B, F, G, I) are not defined. Also, the data types, encoding process, dimensions etc. of inputs and outputs are not specified. 
++ Optimizer agent:
+++ In the Optimizer Agent’s objective, θ, L_task, L_constraint, and G are not formally defined. There is no explanation of how clinical guidelines (G) are encoded, how 〖L_task,L〗_constraint are computed.
+++ What is the format of a “generated plan” that is input to L_task?
++ Memory module: 
+++ The semantic meaning and embedding method for queries (q) and documents (d) are not specified. What are the queries and documents?
+++ The synthesis of reasoning path S from evidence (E) is not detailed. What algorithms or heuristics are used?
+++ How does the quality verification done in the Optimizer agent?
+++ These undefined notations make it impossible to understand the FUAS model design, its fine-tuning process, and the Optimizer agent.
	
+ Segmentation module:
++ The term “our method” is used without specification. Is it a fine-tuned MedSAM-2, an entirely new architecture, or a MedSAM-2 variant with additional modules? If a MedSAM-2 variant, what are the changes: architecture, training strategies, or post-processing? Ablation studies etc.?
++ The dataset split is described, but there is no information on patient diversity, lesion types, imaging protocols, or annotation variability. This limits assessment of generalizability. Were the improvements consistent across all patient subgroups and lesion types?
++ Can the authors provide qualitative segmentation examples and error analysis?
	
+ Dose prediction module:
++ The dataset is small (n=93 after exclusions) and filtered, which risks overfitting and limits generalizability.
++ No baselines shown for simpler models or clinical heuristics, making it unclear if the proposed approach is truly superior.
++ The clinical significance of prediction errors and the consequences of underestimation in high-dose regions are not discussed.
++ Key implementation details (model parameters, hyperparameters) are missing.

+ Model comparison issues:
++ Only benchmarked on report generation task. 
++ Insufficient evaluation. Evaluated only on a single dataset.
+++ Small set, n=200
+++ Randomly sampled test data without considering underlying data distribution, e.g., patient diversity, lesion types etc.
++ Not clear if the MRI scans are used during FUAS.
+++ If FUAS uses MRI, then this is an unfair comparison against the competing baselines that use only text input.
+++ If FUAS does not use MRI, then there must be some benchmarking on a relevant task and against relevant multimodal baselines.  

+ Human evaluation issues:
++ The competing baselines (or a subset of them) from Table 1 must be evaluated by the human experts to understand the added value of the proposed framework.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents **FUAS-Agents**, an *autonomous multimodal large language model (LLM)–based agent system* for **personalized treatment planning in Focused Ultrasound Ablation Surgery (FUAS)**.  
The system integrates **multimodal reasoning, autonomous tool use, and reflection-based optimization** to simulate a clinician’s decision-making workflow.  

The proposed system employs a **modular multi-agent architecture** consisting of:  
- **Planner Agent** – orchestrates the workflow and decomposes clinician instructions into subtasks.  
- **Executor Agent** – handles MRI segmentation (via MedSAM2) and dose prediction using radiomics and XGBoost regression.  
- **Strategy Agent** – integrates multimodal data (MRI, patient reports, clinical guidelines) to generate personalized treatment plans.  
- **Optimizer Agent** – validates and refines generated plans through reflection and consistency checks.  
- **Memory Module** – serves as a retrieval-augmented knowledge base containing clinical guidelines and prior cases.

A large real-world dataset (2,000+ uterine fibroid cases) was used for fine-tuning, ensuring clinical relevance. The model, built upon Qwen3-14B with LoRA fine-tuning, demonstrates superior performance in segmentation (Dice up to 0.85), dose prediction (AUC = 0.91), and treatment strategy generation (ROUGE-1 = 0.55, BLEU-4 = 0.13).  
Human evaluation by four senior FUAS clinicians shows **over 80% satisfaction** in completeness and accuracy, and **97.5% compliance** with clinical guidelines.  

This work exemplifies how LLM-based agents can bridge general-purpose reasoning with domain-specific medical intelligence, forming a concrete step toward **autonomous, explainable clinical decision support**.

### Strengths
1. **Innovative multi-agent design:** Each agent mimics a medical professional’s function, leading to explainable and structured autonomy.  
2. **High empirical performance:** FUAS-Agents substantially outperforms both closed-source (GPT-4, Claude) and open-source baselines (GLM-4, DeepSeek) across all text-generation metrics (Table 2).  
3. **Comprehensive evaluation:** Includes segmentation (Dice/IoU), dose modeling (AUC/KS tests), text-based treatment plan generation (ROUGE/BLEU), and human expert validation.  
4. **Interpretability and reproducibility:** Figures in Appendix B demonstrate qualitative comparisons with other LLMs, showing FUAS’s superior personalization and accuracy.  
5. **Strong real-world motivation:** FUAS is a high-impact medical application; the work clearly bridges research with translational relevance.  
6. **Ethical awareness:** The authors emphasize human-in-the-loop supervision, privacy protection, and federated learning in future deployment.

### Weaknesses
1. **Limited theoretical depth:** The system design and reflection optimization lack formal guarantees (e.g., stability, safety bounds). The authors could include a convergence or reliability analysis for multi-agent coordination.  
2. **Single-center dataset limitation:** The clinical dataset is from one medical institution, which may bias fine-tuning and limit generalization.  
3. **Potential reproducibility barrier:** Though code is released, the model relies on private medical data that cannot be publicly shared. Synthetic data is mentioned but may not capture full variability.  
4. **Insufficient ablation granularity:** The ablation study removes entire agents but does not isolate finer-grained mechanisms (e.g., memory retrieval, LoRA rank, reflection depth).  
5. **Scalability concerns:** The experiments are limited to uterine fibroids (≈3.5 cm lesions). The paper does not evaluate multi-organ or multi-modal expansions.  

These weaknesses are not fatal but should be explicitly discussed or mitigated in future work.

### Questions
1. How does the reflection mechanism scale in multi-turn reasoning — is there a risk of overfitting to local optima during optimization?  
2. Could the system integrate *real-time intraoperative feedback* (e.g., thermal maps) for adaptive dose control?  
3. How would FUAS-Agents perform in *cross-institution* or *cross-device* scenarios where imaging quality and guidelines differ?  
4. What measures are in place to ensure **patient safety** if the system generates suboptimal or inconsistent treatment plans?  
5. Could this framework generalize to other precision therapies, such as radiotherapy or focused microwave ablation?  
6. Have you considered integrating a *causal reasoning* component to improve interpretability of dose–response predictions?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work presents FUAS-Agents, an autonomous agent system that integrates large language models (LLMs) with specialized medical AI tools to assist in Focused Ultrasound Ablation Surgery (FUAS). By combining patient profiles and MRI data, FUAS-Agents performs multimodal interpretation, dose prediction, and guideline retrieval to generate personalized treatment plans with built-in quality control and self-reflection. In uterine fibroid treatment experiments, expert evaluations rated over 82–97% of generated plans in completeness, accuracy, fluency, and clinical compliance. Ablation studies further validate the contribution of each component, highlighting FUAS-Agents as a promising paradigm for LLM-driven clinical decision-making and intelligent surgical planning.

### Strengths
1. The system effectively combines the reasoning and tool-using abilities of large language models with domain-specific modules, enabling end-to-end clinical workflow automation.
2. FUAS-Agents produces individualized treatment plans that align with real clinical needs rather than generic templates.
3. Ratings above 82–97% across completeness, accuracy, fluency, and compliance demonstrate strong practical relevance and clinical acceptance.

### Weaknesses
1. There is no clear clarification of difference against existing AI agent generating medical planning.
2. The proposed Memory module adopted the topK strategy, which is naïve and lack specialized design.
3. The author should compare the proposed optimizer strategy against the beam/tree search.
4. While this work investigates medical AI, there is no comparison against existing medical LLMs, such as Baichuan-M2, HuatuoGPT.
5. The author should clarify the difference and comparison against recent treatment planning work, e.g., medical world model[1].
[1] Yang Y, Wang Z Y, Liu Q, et al. Medical world model. ICCV, 2025.
6. There is no ablation in dose prediction module. Why select such a dose prediction module?
7. The presentation of references in the main text is in the wrong format.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
