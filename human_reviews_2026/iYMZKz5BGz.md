# Demystifying Supervision Data Generalization in Multimodal LMs

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Conventional wisdom in selecting supervision data for multimodal large language models (MLLMs) is to prioritize datasets that are intuitively similar to the target task (e.g. text-rich v.s. vision-centric). However, it remains unclear how reliably such similarity translates into improved performance on the test benchmarks. 
In this paper, we take the first step to study the problem in MLLMs: can we predict a training data's influence on a target benchmark even before any training takes place?
To answer this question, we first conduct an in-depth analysis using 14 vision-language datasets covering 7 diverse tasks. Our analysis shows that intuitive task
similarity is unreliable in predicting task generalizability, and that transfer depends on the specific dataset rather than the broader task category. 
We propose DATAPROPHET, a training-free, simple yet effective metric based on multimodal perplexity, similarity, and data diversity. Our experiments demonstrate that the influence rankings for different supervision datasets derived from DATAPROPHET is strongly-correlated with rankings based on the actual performance increase after training, with a Kendall’s $\tau$ correlation coefficient of 86.0\%. Moreover, we show that DATAPROPHET can help select better supervision data, achieving up to 6.9\% improvement in average over uniform selection, 1.4\% over SoTA training-based baseline, and 0.2\% higher than oracle experiment performance-based selection. Our code and data will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the fundamental question of predicting training data influence on target benchmarks in multimodal large language models (MLLMs) before any training takes place. The authors conduct a comprehensive empirical analysis using 14 vision-language datasets across 7 diverse tasks and make several counter-intuitive discoveries that challenge conventional wisdom about data selection.
Key Contributions
1. Counter-intuitive Empirical Findings: The paper argues that intuitive task similarity is an unreliable predictor of cross-task generalization. 
2. Dataset-level vs. Task-level Influence: The authors demonstrate that data influence operates at the individual dataset level rather than broad task categories.
3. DATAPROPHET Metric: The paper introduces a training-free, interpretable metric that combines three components:
    - Multimodal Perplexity: Measures source data difficulty relative to target
    - Cross-dataset Similarity: Captures alignment in questions, answers, and images using MLLM embeddings
    - Source Dataset Diversity: Quantifies question coverage using clustering-based silhouette coefficient and entropy

### Strengths
**Originality**

- The paper introduces a research question - predicting cross-dataset influence in MLLMs before any training occurs. While data selection has been extensively studied, the specific focus on training-free prediction of multimodal data influence represents a clear departure from existing gradient-based or proxy-model approaches.

- Systematic Empirical Investigation: The comprehensive 14×14 influence matrix analysis is unprecedented in scope for multimodal settings. This systematic approach to mapping cross-task transfer patterns fills an important gap in understanding MLLM behavior.

**Quality**
- The paper demonstrates experimental breadth by selecting 14 diverse vision-language datasets spanning 7 task families (OCR, chart understanding, document understanding, general VQA, spatial reasoning, counting, and map reasoning). This coverage, with 2 datasets per task type, enables robust cross-task transfer analysis. The systematic 14×14 experimental matrix (196 training-evaluation combinations) provides empirical evidence for understanding how different source datasets influence performance across various target benchmarks.
- Rigorous Experimental Design: The controlled experimental setup is well-designed with consistent hyperparameters, fixed compute budgets (20K samples each), and standardized evaluation protocols across all 14 datasets. 

**Clarity**
- Well-Structured Presentation: The paper follows a logical progression from motivation → empirical analysis → method development → validation. The three-part takeaway in Figure 1 effectively communicates the main idea.

- Effective Visualization: Figure 2's heatmap clearly illustrates the nature of cross-dataset influence. The color-coding and task groupings make patterns easily interpretable.

**Significance**
The paper challenges the widespread assumption that "similar tasks help similar tasks more." The systematic influence analysis framework and the 14-dataset benchmark provide a reference for comparative studies.

### Weaknesses
**Single Model Architecture**: All experiments are conducted exclusively on InternVL3-2B, which severely limits the generalizability of findings. Different MLLM architectures (e.g., GPT-4V, LLaVA, BLIP families) may exhibit fundamentally different cross-task transfer patterns due to varying pretraining objectives, data distributions, and architectural choices. The authors provide no evidence that DATAPROPHET's effectiveness extends beyond this single model family, making it unclear whether the discovered "counter-intuitive" patterns are universal phenomena or model-specific artifacts.

**Ill-Defined Problem Formulation and Claims**
- Vague "Counter-Intuitive" Claims: The paper's central claims about counter-intuitive findings are problematic due to poorly defined baselines. In the Introduction, the authors state that "humans may intuitively assume that training the model on OCR task data will help its performance on chart tasks more than spatial reasoning tasks, since OCR and chart both require extracting text and numbers in an image." However, "intuitive" is not a well-defined, measurable concept. The authors provide no systematic survey of expert opinions, formal definition of intuitive similarity, or principled baseline for what constitutes "expected" transfer patterns. This makes their counter-intuitive claims essentially unfalsifiable and scientifically questionable.

- Undefined Task Categories: The paper repeatedly refers to findings being "dependent on individual datasets" rather than "task category," but "task category" itself lacks clear definition. The authors don't explain what constitutes a task boundary, how fine-grained the categorization should be, or what criteria distinguish one task category from another. This conceptual ambiguity undermines the central thesis about dataset-level vs. task-level influence. This subjective categorization makes it impossible to assess whether the reported transfer patterns reflect genuine task relationships or merely artifacts of the chosen taxonomy.

**Limited Long-term Training Analysis**: All experiments use single-epoch training, but production MLLM training typically involves multiple epochs and complex scheduling. The influence patterns observed in short training runs may not persist during extended training.

### Questions
**Contradictory Evidence in Task Category Claims**
There appears to be a contradiction between your claim that "datasets from the same task category do not necessarily help each other the most" (Figure 1b) and the actual results in Figure 2. Taking ChartQA as an example from the Chart understanding task family: the highest improvement comes from Chart2Text (+16.04%), which is indeed from the same task category as defined in Section 2.1. Spatial reasoning tasks like Open-Spatial (+12.46%) and CLEVR(R) (+9.19%) show lower improvements than the same-category Chart2Text. This pattern appears to support intuitive same-task-category transfer rather than contradicting it. Could you:
  - Clarify how you define "task category" boundaries for this specific analysis?
  - Explain why the ChartQA example doesn't contradict your main claims about counter-intuitive transfer patterns?
  - Provide a more systematic analysis of when same-category transfer does vs. doesn't dominate?

**Fundamental Flaws in Cross-Task Transfer Analysis**

Your Observations 2 and 3 in Section 2.2 are based on comparing relative improvements across different target datasets from the same source dataset (same row, different columns in Figure 2). However, this analytical approach suffers from critical confounding factors that invalidate your conclusions:

Core Problem: You compare relative improvements across tasks with fundamentally different baseline difficulties and improvement potentials. The tasks themselves have different intrinsic characteristics that affect their "improvability," making cross-task comparisons of relative gains meaningless.

 Specific Evidence from Your Data: For Observation 2, you claim: "after training on OCR-VQA data, the relative performance gain achieved on ScreenQA (OCR, 17.88%) is lower than that achieved on GeomVerse (map understanding,21.74%)."
   However, examining Figure 2's data reveals confounding factors:
  - GeomVerse shows consistently higher average relative gains (16.89%) compared to ScreenQA (11.14%) across ALL source datasets
  - GeomVerse's self-improvement ($\Delta_{s→s}$) is 55.71% vs ScreenQA's 30.38%
  - This suggests GeomVerse is simply more "improvable" as a benchmark, regardless of source dataset

 Questions:

  1. Confounding Control: How do you distinguish between genuine cross-task transfer effects and task-intrinsic improvability differences? Your current analysis cannot separate these factors.
  2. Baseline Normalization: Have you considered normalizing improvements by task-specific baselines or maximum achievable gains? Without such normalization, comparing raw relative
  improvements across different tasks is scientifically invalid.
  3. Alternative Explanations: How do you rule out that the observed patterns are due to:
    - Different evaluation metric sensitivities
    - Varying task saturation points
    - Benchmark design artifacts
    - Different training dynamics needed by task types
  4. Causal Claims: Your claim that transfer depends on "individual datasets" rather than "task categories" requires showing that dataset-specific factors (beyond task characteristics) drive
  the observed patterns. How do you establish this causal relationship?

  The Same Issue Applies to Observation 3: Your examples of "text-rich tasks influencing vision-centric ones more than text-rich ones" likely reflect the same confounding - vision-centric
  tasks may simply have more room for improvement rather than indicating genuine cross-modal transfer superiority.

  Impact on Paper's Validity: This analytical flaw undermines your central claims about dataset-specific vs. task-specific influence. Without proper controls for task-intrinsic factors, your
  conclusions about "counter-intuitive" transfer patterns may be statistical artifacts rather than genuine insights.

  Suggested Resolution: Re-analyze your data using improvement metrics that account for task-specific characteristics, or restrict comparisons to tasks with matched baseline difficulties and
  improvement potentials.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the challenge of predicting a multimodal large language model’s (MLLM’s) generalization benefit from a given supervision dataset, even before training. The authors develop DataProphet, a training-free metric combining multimodal perplexity, cross-modality similarity, and dataset diversity to estimate the influence of candidate training data on downstream benchmarks. They show that traditional intuition about task similarity fails to predict which training data is most helpful, and demonstrates that its influence ranking correlates with performance gains.

### Strengths
- The paper presents a systematic, large-scale analysis of data generalization in MLLMs, involving 14 datasets over diverse VQA-related vision-language tasks. This scope (detailed in Section 2 and Figure 2) provides a valuable resource and a robust empirical foundation for the claims made.
- By quantitatively demonstrating (Figure 1, Figure 2) that surface-level or even task-category similarity does not reliably predict data transfer utility, the work provides a corrective to widespread "intuitive data selection" assumptions in the field.
- DataProphet leverages a mix of complexity (perplexity), alignment (similarity), and diversity factors, all easily computable without requiring costly pretraining or proxy tasks. This increases applicability in compute-limited scenarios.
- The authors use strong evaluation protocols (employing Kendall's \tau between predicted influence rankings and the ground truth) and ablation studies to show the contribution of each metric component.
- They have shown DataProphet is competitive or superior to state-of-the-art, more expensive approaches, boosting both real and synthetic data selection, and even outperforming performance-based oracle mixtures in average (Table 3).

### Weaknesses
- All experiments are performed using InternVL3-2B. The generality of findings to other competitive models (e.g., Gemini, GPT-4V, Qwen-VL) is not established. Since DataProphet calculations depend on the base model’s embedding space and perplexity, there is a risk of model-specific artifacts, especially when transferred to different backbone architectures.

- Critical equations such as the definition of dataset diversity (Page 6), silhouette score, and their combination with entropy lack detailed motivation for the additivity, choice of $K=10$, cluster methodology, and scaling. The rationale for normalizing perplexities when transferring across datasets is not justified (e.g., differing answer lengths, vocabulary distributions, or answer spaces could bias comparisons. There are also minor ambiguities in notation (e.g., use of $\Delta_{s \rightarrow t}$ notation on Page 3, 4) should clarify whether this strictly measures test accuracy delta or considers area under curve for training budget). Additionally, the metric for synthetic data selection drops the diversity term (Page 8) without justification or discussion of why this is valid—an omission that may have side effects for coverage in less-redundant pools.

- While the procedure is generally clear, some details are underspecified. For instance, random seeds, batch sizes, number of epochs, hardware, and exact computation of metrics like Kendall’s \tau and confidence intervals are missing. There is no assessment of the variability of DataProphet scores under resampling, nor is there a discussion of selection stability over different random seeds.

- While the DataProphet metric is intuitive and empirically effective, the paper advances no rigorous theoretical analysis of why the product formulation (Equation 3), or the selection of factors, is optimal or robust. What are the theoretical regimes or failure cases? Why use a product rather than, e.g., weighted sum, normalized version, etc.? The absence of any formal support or negative result is especially glaring given the field’s growing interest in principled data mixture theory (see, e.g., [1]).

- Minor: suggestion include some recent related works:
  - [2] related for benchmarking MLLMs under data ambiguity and generalization; should be referenced in Section 5 (Related Works) and discussed in terms of differences in experimental scope and generalization perspective.
  - [3] gives metrics for bias/generalization in model evaluation; should be mentioned when discussing selection criteria and effects of selection (Section 4, Section 5).
  - [4] is related for ongoing tuning/data mixture effects; discuss in Related Works and comparison section.

[1] Jiasheng, Ye et al. Data mixing laws: Optimizing data mixtures by predicting language modeling performance. In International Conference on Learning Representations (ICLR), 2025.

[2] Wang, Ru; Song, Selena; Ding, Liang (2025): MMA: Benchmarking Multi-Modal Large Language Model in Ambiguity Contexts

[3] Vo, An; Taesiri, Mohammad Reza; Kim, Daeyoung (2025): B-score: Detecting biases in large language models using response history

[4] Chen, Cheng; Zhu, Junchen; Luo, Xu (2024): CoIN: A Benchmark of Continual Instruction Tuning for Multimodal Large Language Models

### Questions
Around Table3, the discussion lacks qualitative analysis or concrete case studies of failures: When does DataProphet select harmful or unhelpful examples? How robust is the metric to data imbalance, annotation quality, or label noise in the source datasets?

And also see my embedded question in "weaknesses".

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the critical, yet poorly understood, problem of data selection for training Multimodal Large Language Models (MLLMs). The authors first conduct a large-scale experiment (14 datasets, 7 tasks) to "demystify" data influence, finding that human intuition about "task similarity" is an unreliable predictor of which training datasets will improve performance on a target benchmark.Based on these counter-intuitive findings, the paper proposes DataProphet, a simple, interpretable, and training-free metric to predict a source dataset's influence on a target benchmark before any training occurs. The metric is a product of three components:Multimodal Perplexity: How "difficult" the source and target data are for a base model.Cross-dataset Similarity: Cosine similarity of text (question, answer) and visual (image) embeddings.Data Diversity: The coverage and balance of questions in the source data.The paper shows that DataProphet's predicted influence rankings are strongly correlated (86.0% Kendall's $\tau$) with the actual performance rankings obtained after SFT. Finally, it demonstrates that this metric can be used for data selection, creating training mixtures that outperform both uniform sampling and a state-of-the-art training-based selection method (ICONS).

### Strengths
- The paper addresses one of the most important and practical problems in modern ML: data selection.

- The fact that the DataProphet metric requires no training is its greatest strength, making it universally applicable, cheap, and fast.

- An 86.0% Kendall's $\tau$ correlation with ground-truth training outcomes is extremely high and validates the metric's effectiveness.

- The paper successfully translates the predictive metric into a practical data selection algorithm (Sec 4) and demonstrates its superiority. It outperforms uniform selection by a large margin (e.g., +6.9% on synthetic data) and, impressively, beats the computationally expensive training-based SoTA (ICONS).

- The initial analysis (Sec 2) is a valuable contribution in itself, providing concrete evidence that "intuitive task similarity" is not a reliable guide for data selection.

### Weaknesses
- The paper reports in Table 3 that DataProphet-guided selection (D.P.) outperforms the "Oracle" baseline on average (e.g., 71.0% vs 70.8% on real data). The "Oracle" is defined as reweighting based on the observed single-dataset improvements from Figure 2. This result is confusing and potentially counter-intuitive. How can a predictive metric beat an oracle based on the ground-truth outcomes? This implies that the optimal mixture is non-linear and that the single-dataset oracle is not the "true" oracle for a mixed-data SFT. This is a fascinating result but is not explained at all, leaving the reader to wonder if it's a profound insight or a methodological quirk. This must be clarified.

- The metric (Eq 3) is a simple product of its components. While the ablation (Table 2) shows all components are important, it's not clear why a product is the optimal combination. This feels heuristic. The paper's strength is its empirical results, but it's lighter on the theory of why this specific combination works so well.

### Questions
Major: 
- Please explain the result in Table 3 where DataProphet (D.P.) selection outperforms the "Oracle" baseline. The Oracle is defined as using the ground-truth relative improvements from the single-dataset experiments (Fig 2). How is it possible for a predictive metric to beat the ground-truth oracle? Does this suggest that the optimal data mixture is non-linear, and that the single-dataset-training oracle is not, in fact, the true "oracle" for this task? This is a key point that needs clarification.

Minor
- How sensitive are the DataProphet rankings to the choice of the base model? The experiments use InternVL3-2B. If all components (perplexity, embeddings for similarity) were calculated using a different, popular MLLM (e.g., LLaVA-1.5), would the resulting Kendall's $\tau$ correlation still be as high?

- For the synthetic data selection (Sec 4.2), the diversity term ($Sil+H$) was removed from the metric (Eq 4). What was the reason for this? Was it computationally too expensive to calculate at the instance level, or did it simply not improve the results in this setting?

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
2

### Summary
The paper explores the effectiveness of supervised data selection for training multimodal large language models (MLLMs). It introduces a novel metric, DATAPROPHET, which predicts the impact of training datasets on benchmark performance without prior training. Through empirical research on 14 diverse visual-language datasets, the authors find that intuitive task similarity does not correlate with model generalization. Instead, dataset characteristics significantly influence performance. The findings highlight the importance of careful data selection in enhancing MLLM efficacy.

### Strengths
Strength:

Novel Perspective on Data Selection: The article presents a new viewpoint on data selection, highlighting factors beyond mere similarity that influence model performance.

Effective Task Selection Method: The introduction of the DATAPROPHET metric has shown excellent experimental results.

Comprehensive Empirical Validation: The study includes extensive empirical validation across diverse visual-language datasets, providing robust evidence for the proposed method's effectiveness.

### Weaknesses
Insufficient Innovation: The proposed method lacks substantial innovation, as it appears to be merely a combination of existing metrics rather than offering a fundamentally new approach to data selection. This may limit its perceived contribution to the field.

Limited Exploration of Alternatives: The focus on the DATAPROPHET metric may overshadow other potentially valuable data selection strategies. The study doesn’t thoroughly explore how different combinations of metrics or alternative methodologies might yield even better results.

The experiment relies solely on one evaluation metric, which may lead to an incomplete assessment of the model's performance.

### Questions
How is the target task selected? I don't fully understand data selection. Is the ultimate goal of data selection simply to improve accuracy on the target task? How does it contribute to enhancing the model's generalization ability on new datasets?

The evaluation metric used is too singular.  We also need to consider efficiency, the sensitivity of the task data, and the sensitivity of different components of the new methods.

### Soundness
3

### Presentation
3

### Contribution
2
