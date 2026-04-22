# Modeling Multi-Scale Scientific Impact via Heterogeneous Networks and LLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Accurately modeling scientific impact is essential for understanding research dynamics and supporting evidence-based decisions in funding, hiring, and policy. However, despite substantial interest, three core challenges remain unresolved: (i) the heterogeneous and multi-scale nature of scientific impact, encompassing short-term citations to long-term disciplinary influence; (ii) the underexplored potential of large language models (LLMs) to capture the rich semantics embedded in scientific texts; and (iii) the absence of standardized benchmarks, which impedes rigorous comparison and evaluation of predictive methods. In this work, motivated by the need to capture both paper content and research trends, we propose a unified framework that integrates heterogeneous graph neural networks with pretrained LLMs to model scientific impact across temporal and structural scales. To balance effectiveness and efficiency, we freeze the backbone LLM parameters and train only a small set of task-specific parameters using prefix-tuning, which enables scalable training while preserving strong semantic representations. To enable systematic training and evaluation, we construct a large-scale and multi-grained benchmark dataset that combines diverse metadata and multiple impact indicators from real-world scientific corpora. Extensive experiments demonstrate that our approach substantially outperforms both traditional baselines and recent LLM-based methods. All our used datasets and code will be released on GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles the challenge of accurately predicting the multi-scale scientific impact (specifically future citation counts at yearly and monthly horizons) of research papers, a task complicated by heterogeneous factors and the limitations of existing methods in leveraging domain knowledge. Current approaches either focus on graph-based structural features (using GNNs on citation/collaboration networks) or semantic content (using LLMs) in isolation. To address this, the authors propose LLM4Impact, a unified framework that integrates heterogeneous graph neural networks (GNNs) with pre-trained large language models (LLMs). Instead of fine-tuning the entire LLM, which the authors found can introduce bias and lead to poorer performance, their method uses a lightweight prefix-tuning strategy . Graph-derived structural and temporal features are encoded and injected as prefix tokens into a frozen LLM, conditioning its powerful semantic representations on the network context . To facilitate research, the authors also constructed and released a new large-scale, multi-temporal benchmark dataset with rich metadata across ten disciplines. Experiments demonstrate that LLM4Impact consistently outperforms both traditional graph-based and LLM-only baselines, significantly reducing prediction errors for both yearly and monthly citation counts while showing robustness to temporal distribution shifts.

### Strengths
1. The paper investigates the highly practical and important problem of predicting the future scientific impact of research papers, crucial for guiding research dynamics and informing decisions in funding, hiring, and policy.
2. The proposed LLM4Impact framework rationally integrates heterogeneous graph neural networks (GNNs) for structural context with frozen Large Language Models (LLMs) via lightweight prefix-tuning, effectively combining network information and rich semantic representations.
3. The study includes rich experimental analysis, comparing the proposed method against various baselines, conducting ablation studies on feature importance, examining cross-domain generalization, and evaluating different strategies for LLM integration.

### Weaknesses
1. The comparison baselines are somewhat limited, notably excluding established scientific document embedding models like SPECTER, which was mentioned in the related work but not included in the experimental evaluation.
2. While citation count is a primary focus, the study does not incorporate other widely used scientometric indicators from the science of science field, such as disruption scores, which could provide a more holistic measure of a paper's impact beyond mere citations.
3. The experiments involving LLMs are restricted to models with fewer than 10 billion parameters (specifically up to 8B), leaving the potential performance gains or different behaviors of larger foundation models unexplored.

### Questions
Please see paper weaknesses.

### Soundness
3

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
This paper builds a unified framework that fuses heterogeneous academic networks with a frozen large language model. It uses prefix tuning to connect graph structure and textual semantics within the LLM, and evaluates the system on the newly constructed million-scale benchmark that tracks citation growth across multiple time scales. The method consistently outperforms both traditional baselines and LLM-based methods.

### Strengths
• The idea of predicting scientific impact is really meaningful. It could benefit a lot of emerging areas such as automatic surveying, automated peer review, AI scientists, and even serve as a reward signal in reinforcement learning, although the authors didn’t fully discuss this in the paper.

• The motivation is clear and reasonable. The authors noticed that previous LLM-based methods mainly focus on semantic information, while graph-based ones focus on relational structure. They designed a clever way to combine the strengths of both LLMs and graphs.

• The dataset construction process is solid and scalable. The dataset is quite large, containing about 1.08 million samples, which makes it a valuable contribution to the community.

• The reported performance is impressive. The proposed method clearly outperforms previous LLM-based approaches on both the “previous” and “fresh” test splits, setting a new state of the art.

### Weaknesses
• The writing quality needs improvement. For example, the motivation part in the introduction could be organized more clearly. The dataset comparison section might fit better under related work. The subsection “From Words to Worth” should be consistently placed either under Predicting Scientific Impact or LLMs for Scientific Understanding. There are also citation problems. The same paper appears three times in the references, and the authors should keep only the AAAI version while removing the two redundant arXiv entries. In line 280, the period at the end of the equation should be a comma, and in line 322, the comma should be a period. The figures are not vector-based, and some text in Figure 4 (a, b) is too small to read. In general, many details need careful polishing, and the current version does not yet meet the formatting and presentation standards expected at ICLR.

• As the authors mentioned, works such as “From Words to Worth” and De Winter et al. did not include author or affiliation information, but that was part of their design choice. Their goal was to make predictions without relying on such metadata. Therefore, some of the claims made in this paper are not entirely appropriate.

• The performance tables should include graph-based models for comparison, although this work is mainly based on an LLM with a regressor. In addition, models such as NAIP and Qwen-ft should also be provided with author and affiliation information to ensure a fairer comparison.

• While the combination of LLM and graph information is interesting, the overall contribution still resembles an “A plus B equals C” type of innovation. At ICLR, such a combination, although effective, may not be particularly attractive or groundbreaking.

### Questions
• What is the distribution of the dataset? Are the citation counts uniformly distributed, or do they follow a more natural long-tailed or Pareto-like pattern (for example, an 80–20 distribution)?

• What is the conceptual difference between stage-wise impact prediction (for example, predicting after one month, two months, one year, two years, etc.) and predicting the TNCSI metric? It would be helpful to clarify how these two tasks differ in goal and modeling setup.

• In line 211, LLM4Impact predictor appears in italics. Is this the proposed method’s official name? I would suggest adopting a shorter and more specific name, similar to HINT or NAIP, rather than a broad label such as LLM4xxx.

• Since arXiv does not provide author affiliation information, how did the authors extract the author and institution data used in this work? Please clarify the data source or the extraction process.

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
2

### Summary
The paper proposes LLM4Impact, a unified framework that combines heterogeneous GNNs and LLMs to predict scientific impact at multiple temporal scales. The method injects structural knowledge into LLMs using prefix-tuning, avoiding full fine-tuning while retaining efficiency. The authors also present a large-scale, multi-temporal benchmark covering ten disciplines, with cleaned metadata, citation graphs, and public tools for data processing. Experiments show that this method outperforms both classical GNN models and pure LLM-based predictors, reducing error by over 25% for yearly and 18% for monthly predictions.

### Strengths
1. The paper introduces a novel GNN–LLM integration via prefix-tuning, effectively combining structural, semantic, and temporal features.
2. A million-scale, multi-domain benchmark with cleaned metadata and citation graphs greatly supports reproducible research.
3. Extensive experiments across multiple baselines and time horizons demonstrate strong accuracy and robustness.

### Weaknesses
1. Fusion mechanism limitation (Sec. 4.4):
The prefix-tuning strategy injects graph features into LLM prompts, yet this approach may only provide shallow conditioning. It might miss deeper structural interactions or high-order dependencies between semantic and relational information.
2. Temporal modeling inadequacy (Sec. 4.3):
The temporal embedding based on FastText treats time as static tokens. It does not explicitly model temporal evolution, which could limit the model’s ability to capture changing citation dynamics.
3. Robustness evaluation is not comprehensive (Sec. 5.2):
While the authors test on “Previous” and “Fresh” splits, it remains unclear how the model performs under stronger temporal drift or across unseen research fields.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper focuses on the important yet challenging task of predicting the future impact of academic papers. This problem holds practical significance for research evaluation and resource allocation and has attracted increasing attention from researchers in recent years. The authors point out that existing methods typically suffer from two main limitations: one class of models relies solely on the structural features of academic networks (such as authors, institutions, journals, and citation relationships), making it difficult to capture the semantic content of papers; the other class depends only on textual information (such as titles and abstracts), neglecting the structural dependencies within the academic ecosystem.

To address these issues, the paper proposes a unified multimodal prediction framework that integrates heterogeneous graph neural networks (GNNs), large language models (LLMs), and other shallow meta-information features. The core innovation lies in the use of a prefix-tuning mechanism, which transforms the structural graph features extracted by the GNN into prefix tokens for the LLM. This design enables the dynamic injection of academic network structure into the semantic modeling process, thereby achieving an effective fusion of semantic and structural information.

In the experimental section, the authors construct a large-scale, multidisciplinary dataset with both monthly and yearly temporal granularity and compare their framework against a variety of representative models, including traditional GNN-based approaches and recent LLM-based impact prediction models. The results demonstrate consistent improvements across different time scales and test sets, confirming the relative effectiveness of the proposed framework for scientific impact prediction tasks.

### Strengths
1. This work achieves an effective fusion of network structure and textual semantics in the task of scientific impact prediction.

2. The model has been systematically validated on a large-scale, multidisciplinary dataset, demonstrating a degree of robustness and generalizability.

3. The authors commit to releasing the dataset and code, facilitating reproducibility and further research.

### Weaknesses
Major issues:

1. Although the paper proposes a unified framework integrating heterogeneous graph neural networks (GNNs) and large language models (LLMs), aspects such as heterogeneous graph modeling, lightweight LLM fine-tuning, and the idea of combining structural and textual information are largely an integration of existing methods. The work lacks deep, task-specific mechanistic innovation or new theoretical insights for scientific impact prediction. Therefore, its level of novelty may be insufficient to meet the standards of top-tier conferences.

2. While the related work section mentions several recent studies, the experiments compare only a small number of baselines (particularly including only one GNN model). This does not cover the representative methods from 2023–2024, nor does the paper explain the rationale for model selection or exclusion.

3. The paper does not systematically report the contributions of each module to overall performance. For example, the importance of each feature source could be verified by removing one at a time. Additionally, the effectiveness of the FastText temporal embedding component should be evaluated separately. Such ablation analyses are crucial for demonstrating the necessity of the model design.

4. The authors do not provide significance testing for the results, nor do they compare different models in terms of computational latency or resource usage. Moreover, the performance of larger-scale LLMs is not discussed, and no comparison with other multimodal fusion methods is provided, which limits the generalizability and comparability of the conclusions.

Other minor writing issues:

1. The paper "From Words to Worth: Newborn Article Impact Prediction with LLM" appears three times in the reference list; only the officially published version should be retained.

2. The use of "domain-specific" in the abstract is inaccurate. For scientific impact prediction, "domain" more naturally refers to academic disciplines rather than "scientific vs. other fields," and the current phrasing may cause semantic confusion.

3. The repeated use of "multi-scale, heterogeneous, and dynamic" in the abstract and introduction lacks precise definition or quantification, making the research problem description somewhat vague.

4. The division of factors into "scientific factors" and "non-scientific factors" is not rigorous. Both are related to research activities; a more accurate description would be "intrinsic factors" (within the paper) and "extrinsic factors" (external environment).

5. The phrase "multiple impact indicators" in the introduction is not specific; clarifying which indicators are included would improve the clarity of the abstract.

### Questions
1. In your experiments, only a single graph neural network (GNN) model was compared. Could you explain why more recent graph learning methods were not included? Compared to these latest models, what do you consider to be the main advantages of your framework?

2. Could you further clarify how your method differs from other recent approaches that integrate structural and textual information, and what improvements it brings in terms of mechanism or performance?

3. Would it be possible to provide more detailed ablation studies to verify the individual contributions of each component?

4. Could you include inference latency results for different models? Additionally, please consider providing clearer definitions for several concepts in the paper, such as "multi-scale" and "domain-specific."

Overall, I believe the first three points reflect serious deficiencies in the rigor of the original submission. Any additions made during the rebuttal may not be sufficient to enhance the rigor and scientific validity.

### Soundness
2

### Presentation
2

### Contribution
2
