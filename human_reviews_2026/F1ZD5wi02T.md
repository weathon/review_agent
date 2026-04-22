# REMSA: An LLM Agent for Foundation Model Selection in Remote Sensing

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Foundation Models (FMs) are increasingly integrated into remote sensing (RS) pipelines for applications such as environmental monitoring, disaster assessment, and land-use mapping. These models include unimodal vision encoders trained in a single data modality and multimodal architectures trained in multiple sensor modalities, such as synthetic aperture radar (SAR), multispectral, and hyperspectral imagery, or jointly in image-text pairs in vision-language settings. FMs are adapted to diverse tasks, such as semantic segmentation, image classification, change detection, and visual question answering, depending on their pretraining objectives and architectural design. However, selecting the most suitable remote sensing foundation model (RSFM) for a specific task remains challenging due to scattered documentation, heterogeneous formats, and complex deployment constraints. To address this, we first introduce the RSFM Database (**RS-FMD**), the first structured and schema-guided resource covering over 150 RSFMs trained using various data modalities, associated with different spatial, spectral, and temporal resolutions, considering different learning paradigms. Built on top of RS-FMD, we further present **REMSA** (**Re**mote-sensing **M**odel **S**election **A**gent), the first LLM agent for automated RSFM selection from natural language queries. REMSA combines structured FM metadata retrieval with a task-driven agentic workflow. In detail, it interprets user input, clarifies missing constraints, ranks models via in-context learning, and provides transparent justifications. Our system supports various RS tasks and data modalities, enabling personalized, reproducible, and efficient FM selection. To evaluate REMSA, we introduce a benchmark of 75 expert-verified RS query scenarios, resulting in 900 task-system-model configurations under a novel expert-centered evaluation protocol. REMSA outperforms multiple baselines, including naive agent, dense retrieval, and unstructured retrieval augmented generation based LLMs, showing its utility in real decision-making applications. REMSA operates entirely on publicly available metadata of open source RSFMs, without accessing private or sensitive data. Our code and data will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose REMSA, the first LLM agent that automates remote-sensing foundation-model selection from free-form queries. Powered by the schema-guided Foundation Model Database (FMD) catalog of 150+ models, it retrieves candidates, clarifies missing constraints, ranks choices via in-context learning and delivers transparent justifications. Evaluated on 75 expert-curated scenarios totaling 900 task-system-model ratings, REMSA surpasses dense-retrieval and RAG baselines, offering reproducible, user-tailored recommendations without accessing private data.

### Strengths
The work introduces the first domain-specific agent for remote sensing model selection, allowing natural language queries to yield ranked recommendations with explanations. 
It builds FMD, a large, fully structured, and machine-readable database of remote sensing models, openly maintained for the community. 
The authors propose a comprehensive evaluation protocol based on multiple real-world scenarios and expert scoring, showing that their agent-based approach outperforms retrieval-only and RAG baselines.

### Weaknesses
Evaluation is limited, using only a few experiments, without broad, multidimensional validation; code is currently unavailable.

### Questions
1.The framework relies on a closed-source commercial LLM (e.g., GPT-4.1), which raises concerns about reproducibility and accessibility. How can the authors demonstrate that REMSA remains effective when using open-source LLMs? Adding experiments with other LLMs would strengthen the evidence.

2.The baselines are entirely self-designed and do not include comparisons with established AutoML frameworks or human expert manual selection. How do the authors justify that these baselines are sufficiently strong, and what prevents the reported performance gains from being overstated?

### Soundness
2

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
4

### Summary
This paper introduces REMSA (Remote-sensing Model Selection Agent), a large language model (LLM)-based agent designed for automated selection of remote sensing foundation models (RSFMs). To address issues of scattered documentation, heterogeneous formats, and complex model selection in remote sensing, the authors constructed the RS Foundation Model Database (FMD)—the first structured database covering over 150 RSFMs, systematically organizing information on architectures, data modalities, training strategies, and benchmark performance. Building upon FMD, REMSA integrates structured retrieval, contextual ranking, multi-round clarification, and memory-augmented reasoning to achieve automated model recommendation from natural language task descriptions. The authors further established a benchmark with 75 expert-validated tasks and 900 configurations, and compared REMSA against baselines such as DB-Retrieval, Unstructured RAG, and Naive Agent. Experimental results demonstrate that REMSA significantly outperforms all baselines in Top-1 accuracy, MRR, and overall evaluation scores.

### Strengths
The main advantage of this paper lies in proposing the model selection problem in the remote sensing field, constructing a benchmark, and introducing the REMSA method.  REMSA is the first domain-specific LLM agent for RSFM selection, combining structured knowledge with reasoning capabilities to enable transparent and reproducible model recommendations. The FMD serves as the first structured RSFM resource, featuring an extensible schema and ensuring data quality through automated extraction and human verification. REMSA’s multi-round clarification and task-aware ranking mechanisms substantially enhance recommendation accuracy and interpretability.

### Weaknesses
Although the paper presents an innovative framework and system design, its experimental and evaluation components exhibit several limitations. 
1. The experiments are clearly insufficient. The paper includes only a main experiment and a single ablation study, without conducting more comprehensive analyses or comparisons with existing retrieval-augmented or tool-scheduling approaches (e.g., ToolRerank [1], COLT [2]). As a result, the empirical evidence supporting REMSA’s relative performance and methodological advantages remains limited.
2. The proposed benchmark consists of only 75 query samples. While the authors explain that this decision was made to maintain the feasibility of expert evaluations, this scale is too small to capture the diversity and complexity of remote sensing tasks in foundation model selection. Moreover, both the benchmark and evaluation protocols suffer from limited extensibility. Although FMD can be updated with new foundation models, the current benchmark cannot directly incorporate them for unified evaluation. The evaluation process also depends heavily on expert judgments and does not cover all models in FMD, making it nearly impossible for other researchers to reuse the benchmark.
3. In addition, the paper does not provide any examples of expert scoring, which makes it difficult to assess the consistency and validity of the evaluation criteria. Including illustrative examples—such as a query, recommended models, and corresponding expert ratings—would greatly improve transparency and credibility. Finally, although the paper claims that REMSA is LLM-agnostic, the experiments are conducted solely with GPT-4.1, without testing other major models such as LLaMA-3 or DeepSeek-R1. This weakens the argument for the generalizability of the proposed framework.

[1] Toolrerank: Adaptive and hierarchy-aware reranking for tool retrieval. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING), 2024 124. 
[2] Towards completeness-oriented tool retrieval for large language models. In: Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 2024, 1930–1940

### Questions
1. The experimental section appears somewhat limited, containing only a main experiment and a single ablation study, without comparisons to other retrieval-augmented tool selection approaches such as ToolRerank [1] or COLT [2]. Could the authors include additional comparative experiments to better demonstrate the empirical advantages of REMSA over existing methods?
2. Given that task templates can be readily expanded, it is unclear why the proposed benchmark includes only 75 queries. This number seems insufficient to capture the diversity and complexity of real-world remote sensing tasks. Could the authors explain the reasoning behind this limited scale?
3. The benchmark and evaluation framework appear to lack extensibility. New foundation models cannot be seamlessly integrated, and the expert-generated evaluation scores do not cover all models in the FMD, which limits reproducibility and external usability. How should other methods be evaluated when their results correspond to models not present in the current scoring set?
4. No examples of expert scoring are provided, making it difficult to assess the validity and consistency of the evaluation criteria. Could the authors provide representative examples of expert annotations—such as task queries, recommended models, and corresponding scores—in the appendix to enhance transparency? 
5. Although the paper claims that the framework is LLM-agnostic, all experiments are conducted using GPT-4.1 without testing alternative models such as LLaMA-3 or DeepSeek-R1. Could the authors include additional experiments to verify the generalizability of the framework across different LLMs?

[1] Toolrerank: Adaptive and hierarchy-aware reranking for tool retrieval. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING), 2024 124. 
[2] Towards completeness-oriented tool retrieval for large language models. In: Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 2024, 1930–1940

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a Foundation Model Database (FMD) and schema for documenting remote sensing foundation models and their different settings/designs. The paper also introduces REMSA, a Remote sensing Model Selection Agent, which uses the FMD to provide recommendations to users on which remote sensing foundation model is best suited for their needs. The paper provides a benchmark of 75 expert-verified remote sensing query scenarios for evaluating REMSA’s selections. Experiments show that REMSA outperforms a database retrieval baseline (using only the similarity search) and a unstructured RAG baseline.

### Strengths
- The FMD would provide a useful resource for documenting the many design choices of remote sensing foundation models.
- Application end-users often feel overwhelmed at the number of remote sensing foundation models available to choose from. FMD would provide a helpful resource to sort through the different options, and REMSA could provide a helpful user interface to do that sorting.

### Weaknesses
- The FMD and REMSA seem like more of an engineering or product contribution than a research contribution. There are no significant research insights presented.
- The approach relies on the assumption that end-users’ requirements are mostly limited to design choices like which modalities are used. In my experience, the biggest factor determining which FM is a good fit for a user (beyond modality compatibility) is how easy the model is to use and how well it fits into their current workflow (i.e., how much energy it will take to swap in the FM for whatever they are already using). This doesn’t seem to be captured by the proposed framework.
- Concerns about the FMD
    - There is not sufficient data provided to assess the quality and content of the FMD. The paper says there is an example record in the Appendix section A, but I only saw the schema.
    - I feel that the QA process, which only reviews low confidence records, could miss records that are confident but wrong. It seems that 150 records could all be reviewed, but at least a sample of the higher confidence records could be checked.
    - The paper says the database will be open and updated regularly. How will it be updated? Who will do the updating? This seems important for the utility of the model since new papers are coming out frequently.
    - The paper says the FMD includes “all existing RSFMs we could find”. There is no information about how the authors searched for RSFMs and what sources were provided to the automatic population model.
- Concerns about the query benchmark
    - The paper says that the 75 queries in the benchmark dataset are expert verified, but there is no information about what the expertise of those reviewers was or what the review process entailed. It is also not clear how the label of the “best” FM to use for each query is assigned.
    - The paper says “Each selected FM is manually reviewed and rated by multiple domain experts … we compare the top-3 FMs from 4 selection systems.” Does this mean they used the 4 systems in Table 2 to make selections, then expert reviewers assessed which one made the best choice? Where is the ground truth? This seems very prone to confirmation bias, especially because it seems like there might not be any ground truth to the queries in the example table in the Appendix, since many models could meet the query criteria.
- Concerns about evaluation protocol
    - The evaluation criteria seem ill-defined and subjective. How do you measure the diversity of pretraining data? Does Popularity mean that a model is a good choice? What constitutes “recent developments”?
    - It is very difficult to see the green text in Table 3.
    - REMSA allows clarifications to be made. How was this handled in evaluations? Who ran REMSA to provide the clarifications? It seems that this content could bias the evaluation if it is not run by an independent user.

### Questions
- How will the FMD be updated in the future?
- How did the authors search for models for the FMD and what sources were provided?
- The paper says “Each selected FM is manually reviewed and rated by multiple domain experts … we compare the top-3 FMs from 4 selection systems.” Does this mean they used the 4 systems in Table 2 to make selections, then expert reviewers assessed which one made the best choice? Where is the ground truth?  How does this avoid confirmation bias?
- The evaluation criteria seem ill-defined and subjective. How do you measure the diversity of pretraining data? Does Popularity mean that a model is a good choice? What constitutes “recent developments”?
- REMSA allows clarifications to be made. How was this handled in evaluations? Who ran REMSA to provide the clarifications? It seems that this content could bias the evaluation if it is not run by an independent user.
- The paper says that the 75 queries in the benchmark dataset are expert verified. What is the expertise of those reviewers and what did the review process entail? How did you choose the label of the “best” FM to use for each query?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
- The paper introduces REMSA (Remote-sensing Model Selection Agent), the first LLM-based agent designed to automate the selection of Remote Sensing Foundation Models (RSFMs).
- Built upon a structured Foundation Model Database (FMD) of over 150 models, REMSA uses structured retrieval, in-context ranking, memory, and clarification loops to match user queries with optimal models.
- It is evaluated through a benchmark of 75 expert-verified query scenarios (900 configurations) and consistently outperforms retrieval-only and unstructured RAG baselines

### Strengths
- REMSA is the first agentic framework tailored for model selection in remote sensing, integrating structured metadata retrieval, in-context reasoning, and user interaction for interpretability and reproducibility
- The paper establishes a rigorous expert-driven benchmark with diverse, realistic query scenarios and clear evaluation criteria (e.g., task compatibility, modality match, efficiency), providing a strong foundation for reproducible assessment
- Experimental results show REMSA outperforms all baselines in model relevance and reasoning quality, highlighting the benefits of agentic orchestration and structured knowledge grounding in real-world remote sensing workflows

### Weaknesses
- **Major Limitation:** Throughout the paper, the authors frequently mention that the dataset and benchmark are expert-verified, but they do not provide any concrete information about these experts. It remains unclear who they are, what their qualifications are, and what roles they played in data creation and validation. Specifically:
    - Are they from a computer science background with expertise in remote sensing, or are they domain specialists purely in remote sensing / Earth observation?
    - How many experts were involved in the verification process?
    - What were the specific procedures or criteria followed to ensure quality and consistency?
    - What references or prior datasets were used as a foundation during the curation process?

    Without this information, it is difficult to assess the credibility, reproducibility, and domain validity of the so-called expert-verified dataset and benchmark.

- While the paper introduces the Foundation Model Database (FMD) with over 150 remote sensing foundation models, no data link or access information is provided. Even the list of models mentioned in the paper is not made available in the supplementary materials. As a result, it is not possible to verify the coverage, completeness, or quality of the dataset. Given that transparency and reproducibility are key aspects of modern benchmark development, this omission is a significant weakness.

- The intended end-users of REMSA are not clearly defined. It is uncertain whether the tool is aimed at remote sensing scientists, machine learning researchers, or industry practitioners. Specifying the user group and application context would make the scope and usability of the framework much clearer.

- From a remote-sensing and Earth-observation perspective, the literature discussion appears shallow. For instance, when mentioning geographic coverage as a factor influencing model performance, the authors do not cite prior works that explicitly study this relationship. Geographic diversity and sensor coverage are critical elements in model generalization and selection. The lack of discussion or citation of such works (e.g., [1, 2, 3]) weakens the connection of the paper to the broader EO research landscape.

- It would be helpful to include an ablation study showing the contribution of each component in the multi-agent system. For example, evaluating performance when the LLM-based Ranking Module is removed (using only the Retrieval Module) would clearly show how much each part contributes. Quantifying the performance drop in such cases would make the architecture’s design choices easier to understand.

- The paper could also be improved by adding a latency–performance trade-off analysis, comparing the proposed multi-agent setup with a single LLM-based model selection approach. This would help clarify whether the added complexity and computational cost of the multi-agent system are justified by the performance gains.

- It would be good to discuss how the system handles failure cases, especially when the model selection is incorrect. Adding or even discussing a feedback mechanism like LLM-as-a-Judge could make the system more robust and strengthen the overall contribution.
----------

[1] Roscher, R., Russwurm, M., Gevaert, C., Kampffmeyer, M., Dos Santos, J.A., Vakalopoulou, M., Hänsch, R., Hansen, S., Nogueira, K., Prexl, J. and Tuia, D., 2024. Better, not just more: Data-centric machine learning for earth observation. IEEE Geoscience and Remote Sensing Magazine.

[2] Purohit, M., Muhawenayo, G., Rolf, E. and Kerner, H., 2025. How Does the Spatial Distribution of Pre-training Data Affect Geospatial Foundation Models? In Workshop on Preparing Good Data for Generative AI: Challenges and Approaches.

[3] Plekhanova, Elena, Damien Robert, Johannes Dollinger, Emilia Arens, Philipp Brun, Jan Dirk Wegner, and Niklaus E. Zimmermann. "SSL4Eco: A Global Seasonal Dataset for Geospatial Foundation Models in Ecology." In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2403-2414. 2025.

### Questions
Suggestion:

- Table 4 only describes the schema attributes of FMD but does not show what actual entries look like. The authors should include an example column illustrating typical field values (e.g., model name, data modality, spatial resolution, evaluation dataset, performance score). Including such examples would improve readability and help readers understand the structure and semantics of the database more effectively.

### Soundness
3

### Presentation
2

### Contribution
3
