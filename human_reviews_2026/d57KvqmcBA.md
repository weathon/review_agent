# CPQS-Tuning: A Model Self-Perception-Based Data Filtering Algorithm for Efficient Instruction Fine-Tuning

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 10

## Abstract
Instruction fine-tuning is a key technique for enhancing the performance of large language models (LLMs), but low-quality and redundant data often hinder its effectiveness. Recent studies suggest that filtering a small amount of high-quality data for instruction fine-tuning can achieve faster and more efficient training performance. However, existing data filtering approaches predominantly depend on predefined evaluation models or manually designed metrics, without leveraging information from the target LLM itself. This limitation may result in a mismatch between the filtering criteria and the actual requirements of the LLM being fine-tuned, thereby reducing the effectiveness of the fine-tuning process.  To address these issues, we propose a novel perspective: the hidden states of LLMs implicitly reflect the quality of the training data. Based on this insight, we propose a novel data filtering method that extracts the hidden states that reflect the target LLM’s perception of the data as representative features, and builds a data classification model upon them, which outputs the Contrastive Perception Quality Score (CPQS) for dataset filtering. Our experiments are conducted in both general and downstream domains.
(1) In the general domain, our experiments show that training on under 10\% of the data from both the Alpaca\_GPT4 and DeepSeek-R1 synthesized reasoning datasets enables our method to outperform models trained on the complete datasets. Moreover, it surpasses the performance of current state-of-the-art data-selection techniques.
(2) In downstream tasks, our approach delivers an average performance gain exceeding 3.6\% over leading data-selection algorithms across multiple benchmarks, including GSM8K, HumanEval, and HumanEval-Plus.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel method for filtering instruction-tuning data for LLMs. The authors identify that many available instruction-fine-tuning datasets contain low quality or redundant examples, and that existing filtering approaches often rely on external evaluation models or manually designed metrics, which may mis‐align with the target LLM’s actual needs. 
To address this gap, they posit that the hidden states of the target LLM implicitly encode its perception of data quality. They extract hidden‐state embeddings from the LLM when processing candidate instruction examples, train a contrastive classification model to distinguish high vs low quality samples, and define CPQS to rank/filter the dataset. 
Empirically, they show that using under 10 % of the data from large synthetic instruction dataset, their method outperforms models trained on full dataset; in downstream tasks, the CPQS-filtered subset achieves on average +3.6 % improvement over leading selection algorithms.

### Strengths
Introduces an innovative strategy: leveraging the target LLM’s own hidden‐state representations to capture its perception of data quality — this internalizes the filter criterion to the fine-tune target rather than relying on external proxies.

Demonstrates significant empirical efficiency: using under 10% of data and still achieving (or exceeding) full‐data performance is compelling for resource‐constrained tuning scenarios.

The pipeline is reasonably practical — hidden‐state extraction + classifier + ranking is implementable and does not require extensive external hardware (relative to training giant filter models).

Addresses a timely problem: as LLM fine‐tuning scales up, data redundancy and dataset cost are real bottlenecks; effective filtering helps reduce compute/data cost.

### Weaknesses
The training data for classifier (high/low quality) comes from synthetic sources (GPT-4 vs weaker models) rather than human annotated “instruction usefulness,” which may bias the filter toward model‐generated signal rather than actual instruction utility for users.

The evaluation, while covering several benchmarks, is limited in scope: primarily to specific datasets (Alpaca_GPT4, DeepSeek-R1) and certain LLM sizes; less evidence for very large scale (>30 B) or for domain‐specific instruction sets.

The cost/efficiency claims are somewhat under‐detailed: while “under 10% data” is quoted, actual compute/GPU‐hours, hyper‐param parity, training seeds, variation across runs are less fully described, which may reduce replicability confidence.

The method introduces potential bias in data selection: by relying on hidden-state classification, it may inadvertently favour samples that “look good” to the model’s internal representation (which may favour certain styles or tasks) and down-select others (e.g., rare domains, under‐represented instruction types). The paper does not deeply analyse coverage/diversity of the selected subset or risk of bias.

While the hidden‐state classifier is shown to separate high vs low quality (AUC=1.00 in logistic regressor example), that may reflect a somewhat contrived two‐class setup (strong model vs weak model generation) rather than realistic open filtering scenarios; the real generalization beyond this setting may be less strong.

### Questions
The paper defines “high‐quality” vs “low‐quality” data via outputs from stronger vs weaker models (e.g., GPT-4 vs Llama-3.2-1B-Instruct). Could the authors clarify how this quality definition correlates with user‐instruction usefulness or downstream task gain? Have you measured human‐annotated instruction usefulness to validate this proxy?

How robust is the CPQS classifier to architecture shifts? For example, if the target fine-tune model is of a significantly different architecture (or size) from the one used to extract hidden‐states, how does performance vary?

Could the authors provide more detailed compute cost comparisons (selection + fine-tuning) versus baseline filtering algorithms (e.g., IFD, Superfiltering) under identical hardware settings, including GPU-hours, memory, latency?

Have you analysed the selected subset in terms of domain/task coverage, instruction style diversity, difficulty distribution, or edge‐cases (e.g., dialogue vs code vs reasoning)? Specifically, is there a risk that CPQS selects only “safe/easy” instructions that align with the model’s hidden-state biases?

In downstream tasks (GSM8K, HumanEval, HumanEval-Plus), the average +3.6% improvement is promising — could you provide per‐benchmark breakdowns (wins vs losses) and variance across random seeds to help assess stability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a method to select data that is most informative for a specific target model, rather than data that is "high quality" in a general sense. The approach uses the hidden states of LLMs to train a two-way classifier on a synthetic dataset labeled with quality scores, and this framework is validated through experimental analysis.

### Strengths
The paper’s core motivation and its corresponding experimental validation are internally consistent and logically presented.

### Weaknesses
1. The paper proposes a pipeline with data synthesis, CNN training and data evaluation, while completely omitting an analysis of computational cost, which is a critical metric for any work in the data efficiency domain.

2. The comparative study is weak. For example, but not limited to:
- What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning
- DataMan: Data Manager for Pre-training Large Language Models 
- IMPROVING DATA EFFICIENCY VIA CURATING LLM-DRIVEN RATING SYSTEMS
- SelectIT: Selective Instruction Tuning for LLMs via Uncertainty-Aware Self-Reflection
- LESS: Selecting Influential Data for Targeted Instruction Tuning

Besides, It notably lacks a comparison against random sampling, which is the simplest and one of the most essential baselines for data selection.

3. The motivation is not novel. The idea of selecting `actual requirements of the LLM` is the fundamental premise behind all uncertainty-based and influence-based data selection methods.

4. The proposed pipeline appears overly complex and is not clearly justified. See question.

### Questions
The synthetic "quality" dataset (and its good/bad labels) is shared across all experiments, even though a separate CNN is trained for each target LLM. This design implies that "quality" is a pre-defined, model-agnostic concept, determined by the generative source. If all model-specific CNNs are being aligned to this single, shared quality space, why do the final results show that different LLMs have different judgments of "quality"? This seems contradictory.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper trains a CNN to classify low-quality data and high-quality data with the hidden state of a model, before tuning the model on the selected high-quality data. It achieves state-of-art performance on multiple tasks.

### Strengths
1. The paper has a novel idea
2. The experiments can demonstrate the effectiveness of the proposed method
3. The paper is written clearly

### Weaknesses
1. The paper lacks in-depth analysis of the method's improvement over baselines. For example, Alpagasus is the method to provide ground truth labels, but the paper's method performs even better than that. The paper provides no explanation to this.
2. The paper lacks a justification of the CNN structure: It would be good to compare with a more natural transformer structure.

### Questions
1. I see in the appendix A.4.2, you demonstrated that for similar sized models, it's better to use the model's own hidden states to do the classification. What if you use models of different size? For example, will a 13B model's hidden state be better than a 1B model's own hidden state?
2. See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper introduces CPQS-Tuning (Contrastive Perception Quality Score), a data filtering algorithm for instruction fine-tuning based on model self-perception. The core idea is to leverage the hidden states of large language models (LLMs) as implicit indicators of data quality. A CNN classifier is then trained on these hidden-state features to output a contrastive perception quality score (CPQS) for each training example, enabling the selection of high-quality samples. This work is particularly promising in the area of evaluating the quality of synthetic instruction data.

### Strengths
1.The paper takes an innovative perspective by using the model’s own hidden states as a signal of data quality, removing dependence on external evaluation models or manually crafted metrics. It would be interesting to see future work analyzing how these hidden-state signals correlate with human evaluation metrics to improve interpretability.

2.Experimental results demonstrate that the proposed approach can achieve better performance with less than 10% of the data compared to full-dataset training, which substantially reduces training costs.

3.The linear separability and layer-wise comparison analyses nicely support the claim that hidden states indeed encode discriminative semantic features of high- vs. low-quality data — this is a very interesting finding.

4.The appendix includes additional experiments such as full-parameter fine-tuning, iterative filtering, inter-layer comparison, and cross-model preference studies, all of which strengthen the robustness of the method. I look forward to the authors releasing the project code after acceptance so that I can further explore this work in detail.

### Weaknesses
1.The method is relatively complex, as it requires extracting multi-layer hidden states and training an external CNN model, which demands significant GPU memory and computation.

2.The cross-model experiment shows that using “high-quality data” selected by Qwen2.5–7B-Instruct to fine-tune LLaMA2–7B–Chat actually led to degraded performance. This indicates that each LLM has its own internal definition of “high-quality data,” implying that CPQS must be trained separately for each target model.

### Questions
1.It remains unclear whether CPQS can generalize well across different model architectures or parameter scales. Can the method be effectively applied to other open-source LLMs?

2.Although the results show linear separability between high- and low-quality samples, the underlying mechanism of why hidden states reflect data quality is still largely empirical. I encourage the authors to explore this aspect more theoretically in future work.

3.Since hidden layers contribute differently to performance (see Appendix A.4.1), it would be valuable to develop an automatic strategy for determining the optimal layer combination for CPQS extraction.

### Soundness
4

### Presentation
3

### Contribution
4
