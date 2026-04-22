# Benchmarking ECG FMs: A Reality Check Across Clinical Tasks

- Avg Score: 3.50
- Decision: Accept (Poster)
- Scores: 4, 2, 8, 0

## Abstract
The 12-lead electrocardiogram (ECG) is a long-standing diagnostic tool. Yet machine learning for ECG interpretation remains fragmented, often limited to narrow tasks or datasets. FMs promise broader adaptability, but fundamental questions remain: Which architectures generalize best? How do models scale with limited labels? What explains performance differences across model families? We benchmarked eight ECG FMs on 26 clinically relevant tasks using 12 public datasets comprising 1,650 regression and classification targets. Models were evaluated under fine-tuning and frozen settings, with scaling analyses across dataset sizes. Results show heterogeneous performance across domains: in adult ECG interpretation, three FMs consistently outperformed strong supervised baselines. In contrast, ECG-CPC, a compact structured state-space model, dominated 5 of 7 task categories, demonstrating that architecture matters more than scale. FMs improved label efficiency 3.3-9× over supervised baselines, though scaling behaviors varied across architectures. Representation analysis reveals that models with similar performance learn markedly different internal structures, suggesting multiple viable paths to effective ECG representation. Overall, while FMs show promise for adult ECG analysis, substantial gaps remain in cardiac structure, outcome prediction, and patient characterization. ECG-CPC's strong performance despite being orders of magnitude smaller challenges the assumption that FM quality requires massive scale, highlighting architectural inductive biases as an untapped opportunity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a large-scale benchmark of electrocardiogram (ECG) foundation models (FMs) across 12 datasets and 26 downstream tasks, grouped into seven clinical categories (adult/pediatric interpretation, cardiac structure & function, cardiac/non-cardiac outcomes, acute care, and patient characteristics).
The authors compare eight public ECG FMs and two supervised baselines under three evaluation modes (finetuning, frozen, and linear evaluation), analyze statistical significance via bootstrapping, and report label-efficiency scaling curves.

The paper’s stated goal is to offer a unified benchmark for ECG foundation models and to assess how pretraining methodologies, model architectures, and dataset scales influence transfer performance.

### Strengths
1. Timely topic. Benchmarking ECG foundation models is both clinically and methodologically relevant as the field moves toward general physiological representation learning.
2. Broad coverage. The benchmark spans a diverse set of tasks beyond pure diagnostic classification (e.g., echo structure and clinical outcomes), which is rare in existing works.
3. Consistent evaluation setup. Use of three evaluation protocols and bootstrapped significance tests provides a reasonable basis for cross-model comparison.
4. Meaningful observations. It identifies model-dependent task preferences (e.g., ECG-CPC performs best on non-interpretation tasks while ECG-JEPA dominates pediatric tasks).
5. Practical insight. Findings on layer-wise learning rates and 2.5 s crops, are useful for the community and highlight training sensitivity in ECG models.

### Weaknesses
1.Writing quality / presentation.
	1) Background and Introduction repeat the same opening sentence and ECG definition .
	2) Numerous typographical and notation issues (“5 s” vs “5s”, "z-normalization" vs "$z$-normalization", two different MAE meanings, non-standard math syntax).
	3) Figures and tables lack consistent captions and unit notations. Overall readability is below ICLR standards.
2.Reproducibility and implementation details.
	1) Key preprocessing steps (lead selection, resampling, normalization, filtering) and training hyper-parameters (batch size, epochs, scheduler, random seed) are missing.
	2) The “query attention head” used for frozen evaluation is not fully described and thus not reproducible.
	3) No code release or data manifest for pretraining datasets (especially the pretraining dataset d (9.1 M) and dataset e (2.5 M) claims is not convincible in Table 1) .
3.Statistical rigor.
	1) Although bootstrapping is applied, effect sizes and confidence intervals are not consistently reported; rank-based summary figures hide variance and task-specific uncertainty.
4.	No consideration of multi-modal or text-supervised ECG representations.
5.	Interpretation and insight depth.
	1) The discussion section remains descriptive; there is no mechanistic analysis of why certain models excel on specific task types.
	2) Visualization of representational spaces or layer-wise probing would strengthen claims about foundation knowledge .

### Questions
Distinguish between MAE-pretrain and MAE-metric throughout.
Verify the validity of pretraining sample counts (9 M / 2.5 M); add explicit deduplication rules.
Ensure consistent naming (e.g., ECG-FM vs ECGFM) and unit spacing (“mV”, “s”).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a benchmark study that compares several ECG foundation models and supervised baselines (e.g., Net-1D, S4) across multiple public datasets and diverse tasks. The scope is broad, and the implementation effort is good. The benchmark itself has potential value for the *medical* community. That said, the contribution is primarily *engineering-oriented*, resembling a large-scale code integration effort rather than a scientific advance. I do not see any novel **modeling** ideas or **methodological** innovations. As a result, the conclusions boil down to *each model has its strengths*, which feels somewhat descriptive rather than insightful. 

Overall, while the benchmark could be useful as an implementation resource, the current version does not fully substantiate its claim as a comprehensive benchmark for ECG foundation models.

### Strengths
- Unified code framework (code factory) is a promising contribution.
- Task coverage is broad.
- The small-sample scaling experiment (Sec. 4.2) adds engineering and evaluative value.

### Weaknesses
- The paper mainly reports AUC, which is insufficient for assessing clinical utility. More comprehensive performance metrics should be included.

- S4 is quite old among state-space models.

- Regarding **ssm**, however, the paper only reports the number of parameters, without including FLOPs, latency, or GPU memory usage.

- I do not see any novel **modeling** ideas or **methodological** innovations.

### Questions
- table 1 reports results that deviate from original papers (parameter), without explanation.

- While Sec 4.2 provides engineering insights, its conclusion seems inconsistent with the main narrative of the paper, unless the authors merely wish to highlight that *each model/training framework can exhibit its own advantages*.

- The benchmark compares models trained with completely different datasets, architectures, and strategies, making it impossible to tell which factors drive performance differences. To ensure fairness and interpretability, at least one unified training dataset (e.g., MIMIC-IV or HEEDB) should be used across models.

- The authors mention that they propose the ECG-CPC model as an ECG foundation model, but in reality, the main innovation of this model lies in the training strategy, specifically using CPC. On the other hand, the S4 model is fundamentally a supervised learning model that does not use contrastive learning. The issue here is that the core difference between the two seems to be limited to the training strategy rather than the model architecture or task adaptability. This difference is insufficient to distinguish a foundation model from a pre-trained model or a supervised model. What exactly is the definition of an ECG foundation model in this paper? We recommend that the authors clarify the definition of an ECG foundation model in the paper and choose the evaluation baselines based on this standard. Generally, a foundation model should have zero-shot inference capabilities, broad task adaptability, and generalizability across tasks, rather than simply being a pre-trained model fine-tuned for specific tasks.

-  The core value of foundation models lies in their emergent abilities obtained through large-scale pre-training, especially in terms of zero-shot and few-shot generalization in unknown scenarios. However, the evaluation framework in this paper mainly focuses on fine-tuning and frozen/linear probe. While the authors mention the limitation of the lack of out-of-distribution evaluation, if the main value of a model is realized through fine-tuning, it would be more appropriate to call it a pre-trained model. I suggest that the authors supplement their evaluation tasks with zero-shot and few-shot evaluations in unknown scenarios.

- Several implementation details (layer-wise learning rate, 2.5 s cropping, sliding-window averaging) are described as *crucial*, yet their contribution is not quantified. Ablations are needed here.

- Beyond dataset integration, the paper does not articulate specific research questions or hypotheses. As a result, it feels more like a large-scale reproduction project than a scientific study.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This manuscript presents a comprehensive benchmark evaluation of 8 ECG foundation models across 26 clinically relevant tasks which span seven categories, using 12 public datasets. The authors systematically compare these models under fine-tuning, frozen, and linear evaluation settings, and also analyze their scaling behavior with dataset size. Another key contribution is the introduction of ECG-CPC, a lightweight structured state-space model pretrained with contrastive predictive coding, which demonstrates strong performance despite its small size. The study highlights the heterogeneous performance of existing foundation models across different ECG tasks and provides insights into their label efficiency and representational quality.

### Strengths
1. This work addresses a significant gap in the literature by providing a large-scale, systematic benchmark of ECG foundation models across a diverse set of clinical tasks. The breadth of tasks and datasets evaluated is impressive and adds substantial value to the field.
2. The experimental design is rigorous, with careful attention to evaluation protocols, statistical testing, and comparison against strong supervised baselines. The use of bootstrapping for significance testing and the inclusion of multiple evaluation modes (fine-tuning, frozen, linear) enhance the reliability of the results
3. The introduction of ECG-CPC as a lightweight yet effective model is a notable contribution. Its strong performance despite a smaller parameter count suggests that model efficiency can be achieved without sacrificing accuracy, which is particularly relevant for clinical applications where computational resources may be limited.

### Weaknesses
1. While parameter counts are reported, practical clinical deployment hinges on more than just parameter size. Metrics such as inference speed (latency), GPU memory footprint during inference, and energy consumption are critical for real-world applications, especially in resource-constrained environments (e.g., bedside monitors, mobile apps). Including these metrics would greatly enhance the benchmark's utility for clinicians and developers aiming to deploy these models.
2. Why do some large models (e.g., ST-MEM) underperform despite extensive pretraining? Is it an architecture-pretraining mismatch, overfitting, or other factors? A more in-depth analysis (e.g., probing experiments, representation similarity analysis, or ablations on pretraining components) would transform the benchmark from a performance leaderboard into a source of actionable design principles for future ECG foundation models.

### Questions
See the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper conducts a systematic benchmark study evaluating multiple ECG foundation models under consistent experimental conditions.
By analyzing model performance across diverse datasets and task types, it provides quantitative insights into model behavior and practical guidance for model selection in clinical applications.
The work focuses on empirical benchmarking and analytical insights rather than proposing a new algorithmic contribution.

### Strengths
Comprehensive benchmarking: The study compares a wide range of state-of-the-art ECG models in a controlled setting, offering a fair and transparent evaluation of their relative strengths and weaknesses.

Data-driven analysis: The authors carefully distinguish model performance according to data characteristics and clinical task types, providing actionable insights for practical deployment.

Reproducibility and transparency: Experimental design and results are clearly documented, facilitating reproducibility and potential extensions by future researchers.

### Weaknesses
Limited theoretical novelty: The paper lacks a fundamental methodological or architectural innovation in representation learning, focusing instead on empirical evaluation.

Misalignment with ICLR's thematic focus: While the study is rigorous, its emphasis on benchmarking and empirical analysis does not align closely with ICLR's focus on advances in representation learning and theory-driven contributions.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
1
