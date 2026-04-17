# MARS: Harmonizing Multimodal Convergence via Adaptive Rank Search

- Decision: Reject
- Scores: 2, 6, 6, 6, 4

## Abstract
Fine-tuning Multimodal Large Language Models (MLLMs) with parameter-efficient methods like Low-Rank Adaptation (LoRA) is crucial for task adaptation. However, imbalanced training dynamics across modalities often lead to suboptimal accuracy due to negative interference, a challenge typically addressed with inefficient, heuristic methods like manually tuning separate learning rates. 
To overcome this, we introduce **MARS** (**M**ultimodal **A**daptive **R**ank **S**earch), an approach to discover optimal rank pairs that balance training dynamics while maximizing performance.
Our key innovation, a proposed framework of dual scaling laws, enables this search: one law models module-specific convergence time to prune the search space to candidates with aligned dynamics, while the other predicts final task performance to select the optimal pair from the pruned set.
By re-purposing LoRA rank as a controller for modality-specific convergence speed, MARS achieves superior performance over baseline methods and offers a robust, automated strategy for optimizing MLLM fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes MARS (Multimodal Adaptive Rank Search), a novel and efficient framework for harmonizing the imbalanced training dynamics in multimodal large language model (MLLM) fine-tuning. Recognizing that uniform LoRA ranks or heuristic learning rate tuning often lead to suboptimal performance due to mismatched convergence speeds between the modality encoder (ME) and the LLM backbone, MARS introduces a dual scaling law approach: Scaling Law-C models module-specific convergence time to prune the search space to rank pairs with aligned training dynamics, while Scaling Law-P predicts final task performance to select the optimal pair from this pruned set. By treating LoRA rank as a direct controller of modality-specific adaptation capacity and convergence speed, MARS automates the discovery of optimal, differential rank configurations. Experiments show that MARS consistently outperforms strong baselines including differential learning rates, fixed rank pairs, and AdaLoRA, improving ScienceQA accuracy by up to 12.0% and reducing LLaVA Bench perplexity by up to 13.2%, while cutting total search and fine-tuning time by over 11.5× compared to naive search.

### Strengths
1. The problem defined in this paper is highly valuable and carries significant practical relevance to the fine-tuning of Multimodal Large Language Models (MLLMs).
2. The two proposed scaling laws are conceptually insightful and offer meaningful inspiration for the community.
3. The paper is generally well-written, with clear and thorough explanations of the problem formulation, motivation, and methodology.

### Weaknesses
1. The motivation lacks strong empirical evidence to substantiate the core claims.
2. The experimental evaluation in Section 3 is relatively limited; additional results across more models, datasets, and benchmarks are needed.
3. Several technical details are insufficiently explained, reducing the reproducibility and clarity of the method.

### Questions
1. Regarding the motivation and empirical evidence:
a) Why do the authors believe that aligning the convergence times of the modality encoder (ME) and LLM leads to better final performance? This conclusion is not evident from Figures 2(a) and (b). A direct study correlating the gap in convergence times with final task performance would be more convincing.
b) Convergence time is highly dependent on experimental setup factors such as hardware, training framework, and distribution configurations, and may not be a stable or generalizable metric. Using aligned convergence time as a search criterion may therefore lack robustness across different deployment scenarios.
c) Figures 2(a) and (b) only report results on LLaVA-OV-0.5B. This is insufficient to support the general claim. The authors should include results from additional architectures and a broader set of rank pairs.

2. Regarding the evaluation:
a) The paper evaluates only on LLaVA-Bench and ScienceQA. Broader benchmark coverage (e.g., MME, MM-Vet, POPE, or domain-specific tasks) is necessary to demonstrate the generality of MARS.
b) In Table 1, performance consistently improves as the learning rate increases. The authors should explore higher learning rates to better characterize the performance boundary of LoRA.
c) The baselines appear somewhat outdated (e.g., AdaLoRA from 2023). Comparisons with more recent adaptive methods would strengthen the evaluation.

3. Regarding writing and technical details:
a) Beyond the ambiguity of “convergence time” the definition of convergence itself is unclear. What specific criterion or threshold is used to determine that a module has converged? This needs explicit clarification.
b) In Figure 3(a), the bar for “FT w/ MARS (final FT)” is missing for Qwen-3B, and similarly in Figure 3(b) for LLaVA-0.5B. What is the reason for this omission?
c) The ablation study states: “increasing the data sampling range up to 2^11 leads to a clear improvement,” yet the x-axis in Figure 4 only goes up to 8 (2^8 ?). This discrepancy should be resolved, and the figure should be updated to reflect the claimed range.

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
3

### Summary
The paper proposes MARS, a novel adaptive rank search method for parameter-efficient fine-tuning of MLLMs. MARS addresses the imbalance in training dynamics across different modalities by searching for optimal modality-specific LoRA ranks to harmonize convergence speeds. The method employs dual scaling laws—one predicting convergence speed and the other predicting final task performance—to efficiently prune and select rank pairs without exhaustive fine-tuning. Evaluation demonstrates that MARS outperforms baseline methods including differential learning rates and unimodal adaptive rank tuning, achieving superior performance and significantly reduced search time.

### Strengths
- The paper addresses a compelling and practical issue in MLLM fine-tuning: imbalanced training dynamics across modalities leading to suboptimal performance.

- MARS innovatively leverages dual scaling laws to guide rank search, avoiding costly exhaustive hyperparameter tuning.

- Experimental results are thorough and convincing, demonstrating consistent performance gains across multiple MLLM architectures and benchmarks including from-scratch and domain-specialized settings.

### Weaknesses
- The dual scaling laws are empirically derived from limited datasets and modeling scenarios; their generalizability to broader or more diverse MLLM setups remains to be established.

- The method is demonstrated on models with primarily two modality types (modality encoder and LLM), limiting insights on scalability to more complex multimodal architectures involving multiple modalities simultaneously.

### Questions
- Does MARS assume or require fixed LoRA configuration points for calibration? How would it perform with different or continuous rank values outside the discretized set?

- How sensitive is MARS to the initial model state, e.g., pre-trained vs. from-scratch? Are there differences in optimal rank configurations or scaling law parameters?

- Would the approach extend naturally to fine-tuning MLLMs with more than two modalities, e.g., including audio, video, or text-vision-audio tri-modal models?

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
This paper introduces MARS, a method to address imbalanced convergence speeds in fine-tuning multimodal large language models (MLLMs). By leveraging dual scaling laws, MARS predicts module-specific convergence and prunes the LoRA rank search space, efficiently identifying optimal rank pairs prior to full fine-tuning. Experiments demonstrate that MARS achieves superior performance over baseline heuristic strategies, such as differential learning rates or naive rank selection, across multiple tasks, including ScienceQA and LLaVA Bench.

### Strengths
The proposed method is experimentally validated, showing consistent improvements in task performance across different LLMs. It clearly outperforms naive and heuristic approaches. Mostly easy to follow.

### Weaknesses
The scaling laws are empirically derived from a limited set of tasks (ScienceQA, LLaVA Bench); it remains unclear how well they generalize to other multimodal tasks or broader modality combinations.

### Questions
- Is there an impact of MARS on model robustness or generalization to out-of-distribution multimodal inputs?
- Can the method scale effectively for MLLMs with more than two modalities, or in scenarios with limited pretraining alignment across modalities?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces MARS a two-stage procedure that automates modality-specific LoRA-rank selection for MLLM fine-tuning by combining two compact scaling laws. A closed-form solution finds the balanced ME-rank for a given LLM-rank. The laws are calibrated with a few short training runs. On models like LLaVA and Qwen2.5-VL, MARS outperforms baselines (tuned LRs, fixed ranks, AdaLoRA) on LLaVA-Bench and ScienceQA. It also cuts the total search and fine-tuning time by 11.5-15x compared to a 4x4 grid search.

### Strengths
* Replaces a brute-force grid search with an efficient "prune-then-predict" strategy and a closed-form solution (Eq. 3) for balancing ranks.

* The log-log plots (Fig. 2b, 6) show near-parallel lines, supporting the formula's assumption that rank and data size effects are separable.

* Achieves better perplexity and accuracy than baselines on LLaVA-OV-7B and other models.

* Cuts search and training time by over 11.5x versus a simple 4x4 grid search.

* Still shows gains even when tested on "from-scratch" models (Table 3).

### Weaknesses
* Results focus on "search time" savings, but higher ranks (which MARS might pick) cost more per-step in FLOPs/memory. The total end-to-end wall-clock/energy cost isn't compared.

* Why only LoRA on q/k/v? The projector is tuned but not part of the rank search. Ranks are per-module, not per-layer. Rounding the continuous rank from Eq. 3 to the nearest discrete one seems crude; a local sweep wasn't tested.

* The scaling exponents are fit on "from-scratch" models but then used for pre-trained MLLMs. How stable are these exponents across different models, seeds, or domains?

### Questions
* Is "convergence" based on the validation set or test set? How do you ensure the calibration phase doesn't peek at the test set?

* Could this same two-law framework be used to optimize the projector rank or to find layer-specific ranks?

* Do the scaling exponents generalize to other tasks (like OCR or chart VQA), or must they be re-fit for every new domain?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MARS (Multimodal Adaptive Rank Search), an approach for automatically discovering optimal differential LoRA rank pairs for fine-tuning MLLMs. The authors posit that imbalanced training dynamics between modality encoder and LLM modules, caused by fixed rank assignments, result in suboptimal convergence and performance. The proposed MARS utilizes dual scaling laws—one for module-specific convergence speed (Scaling Law-C) and another for final task performance (Scaling Law-P)—to efficiently prune and search the candidate space of LoRA ranks. The method is evaluated under various MLLM configurations and benchmark datasets, showing improved accuracy and efficiency.

### Strengths
1.	The idea of using dual scaling laws to guide rank selection for multimodal fine-tuning is innovative and gains a clear reduction in computational burden for hyperparameter search.
2.	The proposed dual scaling laws are empirically calibrated and validated with extensive experiments. 
3.	Experimental results over multiple baselines across two standard multimodal benchmarks demonstrate consistent improvements.

### Weaknesses
1.	The related work section does not address several directly pertinent recent efforts on adaptive rank selection in LoRA and multimodal search. It's better to take more relevant baselines and benchmarks for comparison.
2.	The scaling laws are empirically fitted without strong theoretical justification, which may limit interpretability and generalizability.

### Questions
1.	How does MARS perform in low-data regimes or on tasks with very large domain shifts not seen during calibration?
2.	Can the authors provide additional evidence or theoretical insights supporting the generalizability of the proposed scaling laws to models with more than two modalities?

### Soundness
3

### Presentation
3

### Contribution
3
