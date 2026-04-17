# Cross-Cancer Knowledge Transfer in WSI-based Prognosis Prediction

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Whole-Slide Image (WSI) is an important tool for estimating cancer prognosis. Current studies generally follow a conventional cancer-specific paradigm in which each cancer corresponds to a single model. However, this paradigm naturally struggles to scale to rare tumors and cannot leverage knowledge from other cancers. While multi-task learning frameworks have been explored recently, they often place high demands on computational resources and require extensive training on ultra-large, multi-cancer WSI datasets. To this end, this paper shifts the paradigm to *knowledge transfer* and presents the first preliminary yet systematic study on cross-cancer prognosis knowledge transfer in WSIs, called CROPKT. It comprises three major parts. (1) We curate a large dataset (UNI2-h-DSS) with 26 cancers and use it to measure the transferability of WSI-based prognostic knowledge across different cancers (including rare tumors). (2) Beyond a simple evaluation merely for benchmarking, we design a range of experiments to gain deeper insights into the underlying mechanism behind transferability. (3) We further show the utility of cross-cancer knowledge transfer, by proposing a routing-based baseline approach (ROUPKT) that could often efficiently utilize the knowledge transferred from off-the-shelf models of other cancers. ROUPKT could serve as an inception that lays the foundation for this nascent paradigm, *i.e.*, WSI-based prognosis prediction with cross-cancer knowledge transfer.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the problem that survival prediction from gigapixel histopathology WSIs typically follows a cancer-specific training paradigm, which does not scale to rare tumors and cannot leverage knowledge across cancers. Specifically, the authors curated a pan-cancer dataset UNI2‑h‑DSS and proposed a routing-based mixture-of-experts that adaptively combines frozen, off‑the‑shelf cancer-specific models to improve target survival prediction.

### Strengths
1.	Problem framing is timely and relevant for computational pathology: leveraging cross-cancer information is important for rare tumors and for improving generalization.
2.	Systematic transferability evaluation across many cancer pairs is informative. Observing both positive and negative transfer is clinically and methodologically relevant.
3.	ROUPKT is a pragmatic, compute-efficient architecture that reuses frozen per-cancer encoders and learns a small router + per-expert adapters on the target. Training and inference efficiency considerations are appropriate for WSI-scale workloads.
4.	The paper is generally clearly written and scoped.

### Weaknesses
1.	OLS appropriateness: The response variable (C-index) is bounded in [0,1] and fold-averaged. The design likely violates OLS assumptions (normality, homoskedasticity, independence).
2.	Few-shot adaptation baselines: Since rare tumors may have very small labeled sets, compare ROUPKT against simple few-shot fine-tuning or adapters on the target, with and without source experts, to contextualize zero-shot vs few-shot practicality.
3.	Claims of efficiency vs MTL should be supported with actual training time, memory, and FLOPs for ROUPKT vs representative MTL and stacking baselines.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work explores the transferability of models across organs and tasks for survival analysis. The authors use the state-of-the-art UNI2-h patch encoder to curate an extensive dataset to thoroughly study this question. They show the mechanism behind this transfer through attention heatmaps, and examining tumor-specific factors that are predictive of a task’s transferability. Lastly, the authors propose a mixture of experts approach to utilize each pretrained model as an expert, demonstrating substantially improved performance from this pan-cancer model. While the breadth of experiments is extensive and the MoE method is interesting, the paper is unfortunately limited by somewhat poor writing quality. I would be willing to increase my score if my concerns are addressed.

### Strengths
- Explainability with attention heatmaps and correlation analysis are interesting and insightful. 
- Evaluation is extensive, performed across 26 cancer types. 
- The routing method for fusing models is novel and interesting. Benchmarking against other ensemble approaches is also extensive. 
- The authors will release their UNI-2-h pan-cancer feature dataset, reducing the barrier for additional works on this topic.

### Weaknesses
- The writing quality is quite poor, in terms of both grammar and flow. I strongly recommend the authors work closely with an LLM to improve the writing quality while maintaining the desired meaning.
- Line 203: Positive transfer should be compared against random initialization and/or mean pooling. With good feature encoders, even these simple baselines can achieve competitive performance.
- Line 210: The existence of positive transfer for a single task is not necessarily indicative of transferability without accounting for multiple hypotheses. For instance, the likelihood that 10 random classifiers to all have transfer performance <0.5 is 0.5^10
- Line 215: Why is C-index of 0.6 the cutoff? 
- More detail is needed on Figure 2 caption. How is classification performed? KNN? Is the classification head from the previous task used? 
- A benchmark majority vote of ensembled models should also be included in Table 3. 
- While Shao et al 2025 only performed classification evaluation, their published slide foundation model (Feather) should be benchmarked against. 

Textual edits:
Line 46: insufficient detail
Line 51: What does “undesirable performance” mean?
Line 54: needs citation
Introduction needs discussion of slide foundation models as a means of model transfer.
Line 66-67: Please define knowldege transfer clearly. What is the “switch in focus?”
The heatmap for Figure 2 should be task-specific (i.e row-wise)
Line 307-308: Describe how distance is measured
Clarify R and C in the Figure 1 caption.

### Questions
In Figure 3, I would be interested to see a quantitative report of attention overlap between the transferred model and the target model for a task, and report whether performance correlates with this degree of overlap.

### Soundness
3

### Presentation
2

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
The paper studies cross-cancer knowledge transfer for WSI-based prognosis. To mitigate limited sample sizes, it curates a benchmark, analyzes what drives successful transfer across cancer types, and proposes an MoE-based method to exploit transfer. Results support the main claim but would be more convincing with stronger baselines, richer metrics, and deeper analysis.

### Strengths
1.	The paper is quite well-written and easy to follow.

2.	The paper proposes using knowledge transfer methods to address the challenge of rare tumors, which is a natural and direct solution.

3.	The experimental results demonstrate the main points of this paper (transfer learning does helps when samples are limited).

### Weaknesses
1.	Using only the ABMIL model is questionable. Survival prediction is quite hard; even with gene information, multimodal survival prediction performance is still not fully satisfactory, unlike WSI classification (C-index around 0.5-0.7, few 0.8). The results in Figure 2 are obtained with only the simplest WSI model (ABMIL), which makes it less convincing. To prove the transferability across organs, it would be better if the authors could demonstrate multiple models all converge to similar patterns (not have to be very complicated, TransMIL and MambaMIL would be enough, since they cover major MIL families: Transformer, Mamba, and ABMIL)

2.	The criterion for the experiments in Figure 2 is not fully convincing. There are four available criteria for this: 1) a fully trained model on target, 2) a randomly initialized, untrained model on target, 3) random scores on the target generated by a chosen strategy (for example uniform or Gaussian), and 4) 0.5. The authors chose the fourth choice, but in my view, 2) and 3) would be more rigorous. 

3.	Building on the previous point, the authors have not trained a target model initialized with the source weights, and if they do so, it would be fairer to compare those results with the first choice to confirm positive or negative transfer.

4.	The authors should provide more intuitive explanations for the first row of Figure 3, rather than simply saying ‘they are similar’. In addition, there are no such conclusions or guidelines on which cancer types could benefit from transfer (though from the experiments, most of them benefit from transfer). 

5.	In Table 2, I suggest reporting baselines in both frozen and fine-tuned settings. ROUPKT freezes experts by design, but the comparison methods should be shown in both modes.

6.	The number of baseline methods is limited. There are plenty of recent works for MIL, and the authors should consider adding some of them to Table 3 for more convincing comparison.

### Questions
Besides the questions in weaknesses, I have one additional question for invasiveness.

1.	Why choose invasiveness, why can RMST fully represent invasiveness, and are there references that support this?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper conducts the first systematic study on cross-cancer knowledge transfer for WSI-based prognosis prediction, curating a large 26-cancer dataset to demonstrate the feasibility of transferring prognostic knowledge, especially to rare tumors. The authors provide crucial insights into the transfer mechanism by visualizing how transferred models identify novel prognostic regions and statistically analyzing the factors that govern transferability. Finally, the practical utility of this paradigm is validated through a routing-based baseline model, ROUPKT, which effectively leverages multi-cancer knowledge to improve prognostic performance on target tasks.

### Strengths
A key strength of the paper is its clear, hypothesis-driven approach. The authors propose that prognostic knowledge can be transferred across different cancer types and that this process is governed by identifiable factors. They then systematically test these hypotheses through a series of well-designed experiments. This structured investigation, which progresses from demonstrating feasibility to exploring underlying mechanisms, provides solid support for the paper's conclusions.

Another significant strength is the clarity of the writing and presentation. The paper is well-structured, logically progressing from problem formulation to experimental results and analysis. The authors effectively use figures and tables to convey complex results in an accessible manner, which makes the paper's novel contributions easy to follow and understand.

### Weaknesses
1.  **Limited Generalizability Due to Reliance on a Single Foundation Model:** The study's conclusions are entirely contingent on the feature space of a single foundation model, UNI2-h. The observed transferability patterns (Figure 2) and the predictive power of inter-task factors might be specific artifacts of UNI2-h's architecture and training data. This limits the generalizability of the core findings.
    *   **Actionable Suggestion:** To strengthen the claims, the authors should validate their key findings (e.g., the transferability for a few cancer pairs) using features from at least one other SOTA foundation model (e.g., CTransPath). Minimally, this limitation must be thoroughly discussed.

2.  **Weak Baselines Obscure the True Benefit of Knowledge Transfer:** The main comparison in Table 2 is against simple baselines: a target-specific model ($M_T$) and a fine-tuned version ($E_T$ + MLP). These models do not represent the current state-of-the-art for WSI-based survival analysis, many of which now employ more sophisticated architectures (e.g., graph-based or transformer-based MIL). By comparing `ROUPKT` only to these simple baselines, the paper fails to demonstrate whether cross-cancer knowledge provides a genuine advantage over a more powerful, SOTA target-specific model. The reported 3.1% improvement might diminish or disappear when compared against a stronger baseline.
    *   **Actionable Suggestion:** The authors should implement and compare `ROUPKT` against at least one recent, high-performing SOTA model for WSI survival prediction (e.g., HVT-Surv or a GCN-based model). This would provide a much more convincing measure of the "true" performance gain attributable to knowledge transfer.

3.  **Insufficient Investigation into Negative Transfer:** The paper identifies negative transfer but stops short of providing a meaningful explanation, attributing it simply to an "intrinsic generalization gap." Understanding *why* transfer fails is critical for future work on multi-task learning or task grouping.
    *   **Actionable Suggestion:** The authors should expand their analysis to hypothesize *why* these specific transfers fail, perhaps by analyzing the histopathological uniqueness of tumors like SARC or by visualizing their feature space to show they are significant outliers.

4.  **Missed Opportunity to Evaluate in Few-Shot Scenarios:** The paper's motivation heavily relies on the challenge of modeling rare tumors, which is inherently a low-data problem. However, the experiments are conducted on target datasets with hundreds of samples (e.g., BLCA has N=372). While this demonstrates general applicability, it fails to evaluate the paradigm in the most critical, few-shot learning scenarios where knowledge transfer is expected to be most impactful.
    *   **Actionable Suggestion:** The authors should add experiments that simulate a few-shot setting. This could be done by creating subsets of the existing target datasets with a small number of training samples (e.g., N=20, 50) and evaluating how much knowledge transfer (via `ROUPKT`) improves performance compared to a target-specific model trained on the same limited data. This would directly validate one of the paper's core motivations.

5.  **Methodological Concern in Baseline Selection:** In Table 2, the baselines $M_{S \to T}$ and $E_{S \to T}$ + MLP are based on an "oracle" selection of the best-performing source model on the test set. This inflates the baseline's performance and is not a realistic scenario.
    *   **Actionable Suggestion:** The authors must clarify this oracle setup. A more rigorous evaluation should report a baseline where the source model is selected using a dedicated validation set.

### Questions
1.  **On the Generalizability of Findings:** Your study's conclusions are based entirely on features from the UNI2-h model. Could you please clarify how confident you are that the observed transferability patterns (e.g., which cancers are good sources/targets) are fundamental to cancer histology, rather than being specific to the UNI2-h feature space? Would you expect similar results with other foundation models like CTransPath? A response here could address the concern about the generalizability of your core claims.

2.  **On the Strength of Baselines and True Performance Gain:** In Table 2, the performance gain of ROUPKT is measured against relatively simple baselines (ABMIL). How does ROUPKT's performance compare against a more recent, state-of-the-art survival prediction model trained specifically on the target cancer? Clarifying this would help assess the "true" practical benefit of adding cross-cancer knowledge over simply using a more powerful single-task architecture.

3.  **On the Practicality for Rare Tumors (Few-Shot Performance):** A primary motivation for your work is its application to rare tumors with scarce data. However, the experiments were not conducted in a true few-shot setting. Could you provide results or discuss how ROUPKT performs when the target task has a very limited number of training samples (e.g., N < 50)? This would directly test the utility of your approach for its intended key application.

4.  **On the "Oracle" Selection of the Best Source Model:** For the single-source transfer baselines in Table 2, you mention selecting the source with the best performance. Was this selection based on the test set? If so, this is an "oracle" setting. Could you please clarify this and perhaps provide results for a more realistic baseline where the source is selected via a validation set? This would ensure a fairer comparison for ROUPKT.

### Soundness
3

### Presentation
3

### Contribution
2
