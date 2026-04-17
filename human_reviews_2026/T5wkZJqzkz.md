# How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining

- Decision: Accept (Oral)
- Scores: 6, 6, 6

## Abstract
Due to the scarcity of high-quality data, large language models (LLMs) are often trained on mixtures of data with varying quality levels, even after sophisticated data curation. A natural approach to better leverage high-quality data is curriculum-based pretraining, where the model is trained on data sorted in ascending order of quality as determined by a quality metric. However, prior studies have reported limited improvements from such curriculum-based pretraining strategies. This work identifies a critical factor constraining these methods: the incompatibility between the ascending data quality order and the decaying learning rate (LR) schedule. We find that while curriculum-based training substantially outperforms random shuffling when using a constant LR, its advantage diminishes under standard LR decay schedules. Our experiments show this incompatibility can be mitigated by two simple strategies: (1) employing a more moderate LR decay schedule, where the final LR is only moderately smaller than the peak LR, and (2) replacing LR decay with model averaging, i.e., computing a weighted average of the final few checkpoints. By combining these strategies, we improve the average score on a suite of standard benchmarks by 1.64% over random shuffling, without additional data refinement. Validated on 1.5B-parameter models trained over 30B tokens with various data-quality metrics, our findings call for a re-evaluation of curriculum-based LLM pretraining and underscore the potential of co-designing data curricula with optimization methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper argues that **curriculum-based pretraining** (train low→high quality) quietly clashes with the usual **decaying learning-rate (LR)** schedules: when the best data finally arrives (late in training), the LR is tiny, so the model barely learns from it. Experiments with a **1.5B** model on **30B** tokens show that curricula look strong under a **constant LR**, but their gains **shrink under cosine/WSD decay** (Figure 1–2). The authors propose **Curriculum Model Averaging (CMA)**: keep a high/constant LR during training, then do **model averaging** over the last checkpoints (SMA/EMA/WMA) to stabilize. They further recommend **co-designing** a *moderate* LR decay with model averaging (**CDMA**), which outperforms standard decay with uniform order and also beats “decay + curriculum” alone, with especially clear gains in **mid-training** when only some high-quality data is available (Table 1–2; Figure 5). A simple theory sketch supports why curriculum + averaging can keep strong updates from high-quality data while still reducing noise.

### Strengths
- The paper **clearly diagnoses** an intuitive but under-discussed coupling: high-quality data arrives when LR is tiny, dulling its impact. The **constant-LR** comparisons make this visible. 
- **CMA** is **simple and actionable**: keep LR high, then **EMA/SMA** the last checkpoints; implementation details are explicit.  
- **CDMA** (moderate decay + averaging) finds an **under-explored optimum** that beats standard decay+uniform and decay+curriculum, especially in **mid-training**.

### Weaknesses
- The study focuses on **one model size (1.5B) and a 30B-token corpus**; it’s unclear whether the same sweet spots hold for much larger models or different corpora/metrics. 
- **Baselines are limited** for some comparisons (e.g., stronger **learned curricula** or **adaptive/variance-aware** data-selection methods aren’t included), so it’s hard to judge competitiveness against the latest dynamic strategies. 
- The **mid-training** setting is promising but still tied to the specific phasing and quality signals here; more domains or public corpora would help establish external validity.

### Questions
Do the CMA/CDMA gains persist for **larger models** (e.g., 7B/13B) and for other **quality metrics** (beyond DCLM/PreSelect)? Please include at least one bigger-model run.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates a critical and often-overlooked issue in curriculum-based LLM pretraining: the detrimental coupling between the data curriculum and the learning rate (LR) schedule. The authors diagnose that standard LR decay schedules, which are designed to reduce noise and stabilize training, conflict with the goal of curriculum learning (CL), which places the most valuable, high-quality data at the end of training. They frame this conflict through the lens of the LR's "dual role," acting simultaneously as an update step size and an implicit importance weight for data. By decaying the LR, standard methods effectively "waste" the best data by learning from it with minimal update steps.

To resolve this, the paper proposes to decouple these two roles. Their primary solution, Curriculum Model Averaging (CMA), replaces aggressive LR decay with a constant high LR during training, relying on model averaging (e.g., EMA or SMA) over the final checkpoints to ensure stability. They further propose a co-designed strategy, CDMA, which combines a moderate LR decay with model averaging. Through extensive experiments on a 1.5B parameter LLM, the authors demonstrate that their proposed methods outperform standard pretraining baselines, especially when compared to the widely-used cosine schedule. The work identifies an "optimal area" of moderate decay where co-designing the curriculum, LR schedule, and model averaging yields the best results, highlighting a previously underexplored optimization regime.

### Strengths
1.  The paper's primary strength lies in its clear and intuitive diagnosis of a fundamental conflict in modern pretraining. 

2.  The authors provide strong empirical validation for their central thesis. 

3.  The proposed solutions, CMA and CDMA, are not ad-hoc but are direct, logical consequences of the initial diagnosis.

### Weaknesses
1.  While the paper's framing is compelling, the core ideas—the conflict between CL and LR decay, and the use of model averaging for stabilization—are not entirely new. The provided literature survey highlights several precedents. For instance, Weinshall & Amir (2020) theoretically showed that optimal CL requires non-decaying LRs, and Jiang et al. (2021) empirically demonstrated that LR decay undervalues late-stage data. Similarly, model averaging techniques like SWA (Izmailov et al., 2018) have long been proposed as alternatives to LR decay. The paper could strengthen its contribution by more explicitly positioning its work against these specific precedents.

2.  The experiments are missing a key and highly relevant baseline: training exclusively on a high-quality data subset for the same computational budget (e.g., by filtering out the bottom 80% of data and repeating the top 20%). Without this comparison, it is difficult to disentangle the benefits of the curriculum's data ordering from the benefits of simply focusing compute on high-quality data. If this simple filtering baseline performs comparably to CMA/CDMA, it would challenge the necessity of the curriculum itself.

3.  The paper observes that a descending (high-to-low quality) curriculum performs poorly but misses an opportunity for deeper analysis. This ordering, when paired with a standard LR decay, seems intuitively synergistic (high LR on high-quality data, low LR on low-quality data).

---

[1] Weinshall, D., & Amir, G. (2020). On the Theory of Curriculum Learning. *Advances in Neural Information Processing Systems*.

[2] Jiang, L., et al. (2021). Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt. *International Conference on Machine Learning*.

[3] Izmailov, P., et al. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. *Uncertainty in Artificial Intelligence*.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the overlooked interaction between data curriculum strategies and learning rate decay in LLM pretraining. The authors argue that when high-quality data are placed at the end of a curriculum schedule, the model fails to fully learn from these data since the LR has already decayed. To address this issue, the paper proposes Curriculum Model Averaging (CMA), which combines a quality-based curriculum with model averaging, replacing LR decay with CMA to focus the final model on high-quality signals. The authors also proposed CDMA, which combines CMA with LR decay. Experiments on a 1.5B-parameter model trained on 30B tokens show that CMA/CDMA consistently improve both validation loss and downstream task performance, especially on Core benchmarks (max.+2.50%).

### Strengths
The paper identifies a realistic but previously underexplored issue -- the negative coupling between LR decay and quality-based curricula. The authors proposed a simple yet effective method, using checkpoint averaging to counteract LR over-decay, without additional retraining or architecture changes. The approach can be easily integrated into standard pipelines, and is computationally inexpensive. Theoretical modeling and gradient trajectory visualization illustrate how the proposed method alleviates the loss of learning signal caused by LR decay in curriculum learning.

### Weaknesses
Results are mostly at 1.5B model parameters and 30B tokens. It remains uncertain whether the same trends hold for much larger models (e.g., 7B–70B) or longer training runs. The paper also lacks exploration of hyperparameters for checkpoint averaging (decay factor, checkpointing interval, number of checkpoints) or justification for the selected hyperparameter values.

### Questions
1. How does CMA/CDMA perform on larger models (e.g., >=7B) or longer training (e.g. 100B+ tokens)? It would be better if you provide at least one larger-scale validation experiment to improve the generality of your claims.
2. How did you tune the hyperparameters or select their specific values?

### Soundness
3

### Presentation
3

### Contribution
3
