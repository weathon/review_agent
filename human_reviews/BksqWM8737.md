# ProteinBench: A Holistic Evaluation of Protein Foundation Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Recent years have witnessed a surge in the development of protein foundation models, significantly improving performance in protein prediction and generative tasks ranging from 3D structure prediction and protein design to conformational dynamics. However, the capabilities and limitations associated with these models remain poorly understood due to the absence of a unified evaluation framework. To fill this gap, we introduce ProteinBench, a holistic evaluation framework designed to enhance the transparency of protein foundation models. Our approach consists of three key components: (i) A taxonomic classification of tasks that broadly encompass the main challenges in the protein domain, based on the relationships between different protein modalities; (ii) A multi-metric evaluation approach that assesses performance across four key dimensions: quality, novelty, diversity, and robustness; and (iii) In-depth analyses from various user objectives, providing a holistic view of model performance. Our comprehensive evaluation of protein foundation models reveals several key findings that shed light on their current capabilities and limitations. To promote transparency and facilitate further research, we release the evaluation dataset, code, and a public leaderboard publicly for further analysis and a general modular toolkit. We intend for ProteinBench to be a living benchmark for establishing a standardized, in-depth evaluation framework for protein foundation models, driving their development and application while fostering collaboration within the field.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a unified protein benchmark to evaluate various methods for different tasks such as protein structure prediction, sequence design, structure design, sequence-structure design, and molecular dynamics. For each task, results are presented for various state-of-the-art methods and a discussion is followed on the strengths/weaknesses of these methods. The evaluation is performed using multiple metrics commonly used for these tasks, and, that sometimes incorporate different objectives in protein design. The benchmark is planned to be shared in an open-source manner for the community to compare methods for the presented tasks.

### Strengths
1.	A benchmark with multiple metrics is proposed for protein-related tasks. The metrics incorporate different objectives for different problems encountered in protein design and molecular dynamics tasks.
2.	The paper tackles an important and complex problem in the protein community which is the creation of benchmarks and datasets for training AI methods.
3.	The benchmark incorporates recent challenges like evaluating sequence-structure compatibility in recent co-design methods.

### Weaknesses
1.	The evaluation of methods for protein-related tasks is challenging and, usually, the decision of dataset curation and splits, metrics used for evaluation, and other small details are very important so the results can be trusted by the community. The reviewer thinks that the manuscript could have more information on all these details, instead of presenting results and discussing the performance of the methods. Some of these details are presented in the Appendix while others are missing.
2.	Some metrics used for the benchmark, even though used before by previous references, might need more discussion as they can be misleading for checking the quality of designs, e.g. scRMSD in antibody design.

### Questions
Comments:

1.	My main concern is related to how the paper is structured and the information it contains. From Page 3 the task results are presented with a small definition and comparison between methods. I see that the authors presented the creation of unified datasets as a current limitation and future work by the community, but the reviewer thinks that the manuscript should contain more information about the thought process about creating the benchmark, with the thought process about the metrics, the impact of methods using different datasets, tasks in which this is critical, etc. These are crucial for the community to adopt the benchmark and trust the results that are being presented and compared. In its current form, it is hard to understand these intrinsic details that are important for protein-related tasks.
2.	The authors define the methods as “protein foundation models” and add their own explanation of how this definition is being used. From the reviewer's understanding, usually, foundation models are defined for methods that can be applied, e.g. using their latent space, for many different tasks.  Any additional reasoning for using this new definition of protein foundation models?
3.	Motif Scaffolding: For the Motif Scaffolding task, which evaluation metrics are being used? The reviewer is confused if RMSD metrics are being used or also designability metrics are being used.
4.	Antibody Design: Antibody metrics are usually very challenging to trust. As a benchmark, some of the metrics such as the scRMSD from structure prediction networks like IgFold can be misleading, as we are more interested in antigen-antibody complex structures. I understand current structure prediction networks accuracy for antibodies is limited, but it would be interesting to discuss and choose reliable metrics, even at a reduced number. As the evaluation metrics evolved, they could be added to the benchmark.

Minor Comments:

1.	Line 81: “we aims”
2.	Lines 193-195: “We have noticed…” and Lines 214-215: “We will soon…”: These sentences can be re-written to mention or list just current state-of-the-art methods that are currently not evaluated.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper  introduces ProteinBench, an evaluation framework aimed at standardizing and broadening the assessment of protein foundation models. These models have gained prominence due to advancements in protein prediction and design, covering tasks from structural prediction to conformational dynamics. ProteinBench aims to address gaps in current evaluation practices by introducing a comprehensive, multi-dimensional approach that evaluates models based on quality, novelty, diversity, and robustness. The authors aim for ProteinBench to become a continually updated benchmark to guide research and collaboration in protein modeling.

### Strengths
Novel Evaluation Framework: The paper proposes a well-structured framework that standardizes evaluation for protein foundation models, addressing a significant need in the field. By evaluating on multiple fronts—quality, novelty, diversity, and robustness—ProteinBench gives a well-rounded assessment of model performance.

Task Diversity and Practical Relevance: ProteinBench is inclusive of various protein modeling tasks, including antibody design and multi-state prediction, which are highly relevant to real-world applications in pharmaceuticals and bioengineering.

User-centered Analysis: The framework is flexible, accommodating different user needs (e.g., evolutionary fidelity vs. novelty), which makes the tool versatile for diverse research goals. This feature improves the applicability of model results to specific scientific or engineering contexts.

### Weaknesses
Lack of Standardized Training Data: Differences in training datasets among models hinder direct comparison. Standardizing datasets would improve the ability to compare model architectures and may be essential for achieving fairer assessments within ProteinBench.

### Questions
1. I am not very familiar with AI for Protein. Could you provide with the reason why you separate whole protein tasks to these 8 parts?

2. Could you give some additional metric about tasks? While ProteinBench is a strong foundation, additional tasks and metrics (especially regarding dynamics and multi-modal integrations) could improve its scope, making it more universally applicable in protein science.

3. Could you plan to implement controls to account for differences in training data?

### Soundness
3

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
2

### Summary
I am not sure how to review this manuscript. I have been in the field for over 20 years, so I am an experienced ML/Deep learning researcher, but I do not know much about deep learning for protein design/generation beyond having read blog posts about AlphaFold, ESM, and similar advances. I do know a fair bit about biology (probably more than most ML researchers), but I am still far from an expert in this area. I thus have some high-level opinions about the work, but cannot evaluate the vast majority of the content. I would like my high-level thoughts to be considered, but I cannot evaluate any of the details in the paper, so I have to defer to other expert reviewers to do that. Here are examples of the questions I cannot evaluate: Are the challenges in the benchmark the right ones? Are they comprehensive enough? Are the models evaluated the right ones? Are major ones missing? Were the evals done fairly and correctly? All of that I’ll have to leave for subject matter experts. 

That said, here are my high-level thoughts:

The field benefits from benchmarks, and we should reward people who take the time to make them. I thus support publication because from what I can tell, this is a new, needed, helpful benchmark. However, I would not know if other similar benchmarks already exist. 
It is also helpful to have a set of leading models tested on such benchmarks. Assuming the set of models is a good one and the evals are well done, that is another contribution worthy of sharing with the community via a publication. That said, I do not know how novel such a comparison is. 
The paper does a poor job of explaining almost anything in a way that a non-domain expert could understand. The worst example of this is the challenges in the benchmark. They are extremely superficially described, with a reference to the Appendix presumably doing all the work of explaining what these challenges really are, how they are evaluated, why they matter, why they were chosen, etc. I think more of that belongs in the paper. I’d rather see a paper arguing for and introducing a benchmark do MUCH more of that motivation and explanation than a slog through the performance on the benchmark of a bunch of models. After all, why do we care how these models perform on problem x before we know what problem x is and why it matters? That said, perhaps domain experts already know these challenges so well that the current level of text is sufficient? I am not in a position to judge, but I think the paper would likely be higher-impact and better if it was more readable by non-insiders. I was hoping to learn a LOT about this area by reading this manuscript, and instead, sadly, I learned basically nothing (except there is a new benchmark). The reason I am still voting for publication despite that is I imagine this paper is quite valuable for insiders, but great papers do better at offering *something* to non-insiders. 

Minor: 
You say ProteinBench provides a holistic view. That is too strong. It is virtually impossible to provide a holistic view of such a complex topic. Please switch such language to “more holistic” or similar. 
You say benchmarks are crucial or critical for progress. This is a strong claim. I think they are helpful, but progress can be made without them, and they can actually hurt progress too (see Why Greatness Cannot Be Planned), so I recommend a more nuanced statement. 
Line 181: You say this finding has significant implications for the field, but it is drawn from 2 data points only! Please properly soften the claims. 
Line 192: “We have noticed that….” That’s a weird way to describe your own work/choices. You sound surprised you didn’t include more methods? I suggest finding clearer, less confusing language. 

Note: I am rating this a 6 and not an 8 simply because I do not know enough to evaluate the vast majority of the work. If the other reviewers think it is good, then this 6 should support publication. But I do not want to set a super high score that would override experts with doubts. If all experts agree the technical work and novelty are solid, I'd be happy with an 8.

### Strengths
See main review.

### Weaknesses
See main review.

### Questions
See main review.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors introduce a standardized benchmarking framework to evaluate the performance of protein foundation models. The framework includes 1) task taxonomy, categorizing key tasks in protein modeling (protein design, confirmation prediction);  2) multi-metric evaluation across quality, novelty, diversity, and robustness dimensions. 3) In-depth analyses based on different user objectives. The study finds that thorough evaluation metrics are crucial for adequately validating protein models and that no single model is optimal across all protein design tasks, which underlines the need to match models to specific applications. The authors intend the framework to serve as a collaboratively developed comprehensive and transparent benchmark for evaluating protein foundation models.

### Strengths
-	The framework’s taxonomy of tasks within the domain of protein foundation models is insightful.  It makes it easier to evaluate where each model excels or falls short.
-	The multi-dimensional metrics aims to capture various aspects of model performance which is appropriate given the complexity of the protein modeling. 
-	The authors conduct a large number of experiments, demonstrating the breadth of the evaluation and ensuring the results' validity across various models and tasks. 
-	Leaderboard and open-source code can potentially facilitate more fair comparison and promote transparency.

### Weaknesses
-	Given that the authors have made an extensive amount of experimental study, some reorganization of the paper could strengthen the delivery of the contributions of the paper. Including clear and complete definitions, explanations, and relevance of the metrics would be helpful. The relevance and insights of the results could replace the explanations of the results. For example, Section 2.2.6 Antibody Design, instead of listing the outperforming models for evaluation, which is provided in Table 6, authors could discuss the relevance of these metrics along with the insights gained from the results similar to the one that they provided in the last paragraph.  
-	Lack of consistency in the training data across models is a limitation that undercuts the one of the main promises of the proposed framework which is standardization of the evaluation of protein foundation evaluation. This may not be an issue in the future as the framework is further developed and more mature. 
-	The authors mention that no model performs optimally across all metrics. It does not, however, fully explore the causes and potential trade-offs. Insights into why certain models perform better for certain tasks would provide better guidance on choosing task-appropriate models.
Minor Issues:
-	The description of Table 12 does not align with the data presented in the table
-	Items (3) and (4) in the conclusion are the same.
-	Figure 2 is too small.

### Questions
- The paper highlights four key dimensions (quality, novelty, diversity, and robustness), but the result tables, including Table 1, emphasize only the first three dimensions. Why are robustness metrics omitted from the tables? Additionally, some dimensions in Table 1, such as those in antibody design, do not align with these four key categories.
-	Given that no model excels across all metrics, how should the models be ranked on the leaderboard, given the trade-offs across different metrics?

### Soundness
2

### Presentation
3

### Contribution
3
