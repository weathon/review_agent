# OptimSyn: Influence-Guided Rubrics Optimization for Synthetic Data Generation

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Large language models (LLMs) achieve strong downstream performance largely due to abundant supervised fine-tuning (SFT) data that imparts problem-solving capabilities. However, as applications expand, high-quality SFT data in knowledge-intensive verticals (e.g., humanities and social sciences, medicine, law, finance) is exceedingly scarce: expert curation is costly, privacy constraints are strict, and label consistency is hard to guarantee. Recent work turns to synthetic data, typically prompting a teacher model over domain documents and filtering with handcrafted rubrics. Yet, rubric design is expert-dependent and rarely transfers across domains; moreover, prevalent heuristic optimization follows a brittle loop (write rubric $\rightarrow$ synthesize $\rightarrow$ train $\rightarrow$ inspect $\rightarrow$ guess tweaks) that lacks reliable, quantitative feedback about a rubric's true contribution to downstream performance.
We argue for assessing synthetic data quality through its causal impact on the target model, using this feedback to guide data generation. Inspired by classic influence functions, we repurpose an optimizer-aware estimator that uses gradient information to quantify each synthetic sample's contribution to the objective of a given target model on specific tasks. Our analysis reveals a gap: although synthetic and real samples may be close in embedding space, their influence on learning can differ substantially. Building on this insight, we propose an optimization-based synthetic data framework that adapts rubrics with target-model feedback. Instead of manually engineering domain rubrics, we supply lightweight guiding text and delegate rubric generation to a rubric-specialized model conditioned on the task; crucially, rubric (and data) selection is supervised by estimated downstream impact rather than proxy formality. Empirically, the framework yields consistent gains across domains (HSS and health), target models (e.g., Qwen and Llama families), and data generators, demonstrating broad generalization and engineering portability without task-specific tuning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method to generate specialized rubrics to condition synthetic data generation. The authors optimize these rubrics via an RL pipeline based on GRPO, where the reward is defined using validity checks and an influenced based score. The intuition is that a rubric is good if the generated synthetic data conditioned on that rubric maximize the influence on evaluation samples.

### Strengths
Relevance of the problem: the problem tackled in this paper is an important one, as high quality SFT data is scarce in some domains.

Novelty: the use of Adam-aware influence estimation in the context of synthetic data seems to be novel.

Performance: the reported downstream performance is competitive with the baselines, as it often beats both the base and instruct Qwen model on HSS tasks, and narrows the gap with medically-oriented baselines on medical tasks.

### Weaknesses
Details about the validation dataset: since rewards are computed using a validation set from the same topics as the benchmarks (HSS/medical), the authors should provide more details about how this validation dataset is constructed. It is important that there is no leakage between the validation and the test dataset.

Seeds and test datasets: seeds come from Goodreads books and Meditron/PubMed. It is important to check for overlap vs. MMLU-pro, SuperGPQA, PubMedQA, etc. to verify that performance gains just do not come from data contamination.

Comparison with the baselines: the paper does not report matched data volumes, number of fine-tuning steps, or compute budgets per baseline. This means improvements could come from getting more tuning tokens rather than better data quality.

### Questions
How do you ensure no contamination between the validation sets used for influence scoring and the benchmarks you report?
Did you run any overlap analysis between the seeds and the eval benchmarks?

Can you provide token/step parity vs each baseline in Table 1?

How does the method scale beyond 20k samples ?

### Soundness
2

### Presentation
3

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
The paper introduces OptimSyn, a method that optimizes rubrics for synthetic SFT data by directly maximizing each rubric’s training impact on a chosen target model. The approach replaces heuristic or embedding-similarity–based rubric design with an optimizer-aware influence objective aligned to Adam, ensuring the signal reflects actual learning dynamics. Experiments span multiple domains and model families, showing consistent gains over strong SFT and heuristic-rubric baselines.

### Strengths
* The paper identifies a core mismatch between embedding proximity and true training utility, motivating an influence-aligned objective that targets what actually updates the model.
* The empirical coverage is broad, including HSS/medical domains and cross-family targets/generators, which strengthens external validity.

### Weaknesses
* The approach requires a dedicated validation set to estimate influence, which some tasks may not have; when available, the estimator’s reliability can be highly sensitive to how that set is constructed, risking leakage and brittleness.
* The paper defines the reward function with hyper-parameter $\lambda$. Sensitivity analysis for $\lambda$ is needed to assess the robustness of the reward choice.

### Questions
* Could you explain how validation data is selected? How selection of validation data impacts the downstream target model training performance?
* When computing influence score as in equation (1), are gradients taken with respect to the full parameter set used for training? Are there any analysis or experiments on runtime and memory complexity for scaling the model size from 8B to larger models, what gradient-feature compression method could be used to reduce the computation and storage?

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
This paper proposes a framework to unify the influence-based reward into model training to guide the synthetic data generation.
The motivation stems from the observation that current synthetic data pipelines rely heavily on handcrafted rubrics and heuristic feedback loops, which are expensive, brittle, and poorly correlated with true downstream model performance. The authors introduce an influence-function–based estimator that quantifies each synthetic sample’s contribution to the fine-tuned model’s objective. This influence signal is then used to guide rubric optimization and synthetic data selection via a lightweight optimization loop, replacing manual rubric engineering with model-informed adaptation.

### Strengths
1. Motivation is clear. Synthetic data generation requires a lot rubric engineering and hard to scaling.
2. Frameworks is good. Put influence functions into RL framework to make the whole procedure automatic. 
3. Good performance. The improvements are consistent and statistically meaningful.

### Weaknesses
1. Limited Novelty and Conceptual Depth. While the paper presents a plausible framework, the underlying technical innovation appears incremental. The idea of integrating quantitative signals to guide synthetic data generation is intuitive and has been explored under different names (e.g., data valuation, reward-guided synthesis, active selection). The contribution primarily involves adopting influence-based rewards for rubric optimization, which builds directly on prior influence-function research rather than introducing a fundamentally new mechanism. As a result, the work feels more like a thoughtful application of existing tools than a conceptual breakthrough.
2. Scalability and Efficiency Unclear. Traditional influence functions are notoriously computationally expensive. The paper claims to deploy an “optimizer-aware estimator” that scales efficiently, but does not provide sufficient empirical evidence or runtime comparisons to support this claim. Since the method is applied per-sample, a clear analysis of runtime overhead and system-level feasibility is crucial to evaluate its practicality for real-world synthetic data pipelines.
3. Questionable Necessity of the Influence Signal. The paper argues that influence estimation provides a better feedback signal for synthetic data quality, but the empirical justification is not fully convincing. The improvement margins over simpler proxy metrics (e.g., downstream reward, embedding-similarity heuristics, or small proxy validation sets) are modest. Without ablation or sensitivity studies, it remains unclear whether the influence-based signal is truly superior or merely another heuristic choice. A comparison against alternative feedback mechanisms would strengthen the paper’s argument.

### Questions
1. Can the authors provide quantitative results about efficiency?
2. Have the authors compared their influence-based signal with alternative feedback mechanisms?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Previous studies on synthetic data generation typically rely on expert-designed, domain-specific rubrics, which often suffer from limited transferability and scalability. This paper proposes a novel paradigm: instead of handcrafting rubrics, it advocates evaluating synthetic data quality based on its actual training utility on the target model. This feedback is then used to iteratively refine the rubrics themselves, forming a closed loop between data synthesis and downstream performance. 

The resulting QA pairs are optimized to maximize estimated downstream benefit under these learned rubrics. Experiments in two low-resource domains—Humanities & Social Sciences (HSS) and Medical & Health (MH)—demonstrate that the learned rubrics are both model- and task-conditioned, reducing reliance on domain expertise.

### Strengths
1.	The key insight—replacing heuristic rubric design with a model-impact-driven objective—is interesting. Training a rubric generator to maximize downstream utility offers a principled and adaptive alternative to manual engineering.
2.	Evaluation in HSS and MH domains, where high-quality SFT data is scarce, showcases the method’s practical value and generalizability beyond standard benchmarks.

### Weaknesses
1.	How do the number and length of rubrics evolve throughout reinforcement learning? Could the observed improvements be primarily driven by the generator's capacity or the diversity of synthesized QA pairs, rather than the rubrics themselves? What mechanism controls the ratio between rubric guidance and QA pairs? Additionally, have you analyzed the diversity of the generated data and its correlation with specific rubrics?
2.	Could the authors provide more details about the lightweight verification rules (e.g., formatting correctness, non-triviality, safety filters)? 
3.	Are there notable differences in data statistics (e.g., size, average length, complexity) between prior datasets and the proposed one? Such differences may confound the observed gains; a comparative analysis would strengthen the claims.
4.	In the 14B model results, the performance gain in the MH domain appears less pronounced than in HSS. Have the authors explored whether increasing the volume of synthetic data improves outcomes in MH, suggesting a data-hungry regime?

### Questions
please refer to "Weaknesses"

### Soundness
3

### Presentation
3

### Contribution
3
