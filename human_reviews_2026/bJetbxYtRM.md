# Bridging Temporal and Semantic Gaps: Prompt Learning on Temporal Interaction Graphs

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Temporal Interaction Graphs (TIGs) are widely utilized to represent real-world systems like e-commerce and social networks. While various TIG models have been proposed for representation learning, they face two critical gaps in their "pre-train, predict" training paradigm: a temporal gap limiting timely predictions and a semantic gap reducing adaptability to diverse downstream tasks. 
A potential solution is applying the ``pre-train, prompt'' paradigm, yet existing static graph prompting methods fail to address the time-sensitive dynamics of TIGs and have a deficiency in expressive power. 
To tackle these issues, we propose **Temporal Interaction Graph Prompting (TIGPrompt)**, a versatile framework that bridges the temporal and semantic gaps by integrating with existing TIG models. Specifically, we propose a "pre-train, prompt" training paradigm for TIGs, with a temporal prompt generator to offer temporally-aware prompts for different tasks. To cater to varying computational resource demands, we propose an extended "pre-train, prompt-based fine-tune" paradigm, offering greater flexibility. 
Through extensive experiments involving multiple benchmarks, representative TIG models, and downstream tasks, our TIGPrompt demonstrates the SOTA performance and remarkable efficiency advantages. The codes are available at an [Anonymous Repository](https://anonymous.4open.science/r/TIGPrompt).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this work, the authors introduced TIGPrompt, a novel framework that applies prompt learning to TIGs to bridge temporal and semantic gaps in current models. TIGPrompt focus on the pre-train and prompt paradigm, different from standard training. The Temporal
Prompt Generator generates personalised temporal prompts for each node to adapt to downstream task. The authors evaluated the proposed methods on link prediction and node classification tasks and shows improvement to the base model with TIGPrompt.

### Strengths
- **originally: idea of temporal prompt generator** the idea of using a prompt generator to adapt to downstream task is interesting. The authors introduced three prompt variants: Vanilla TProG, Transformer TProG, Projection TProG
- **clarity: easy to follow** the paper is easy to follow, the authors presented the ideas well.
- **extensive evaluation** the authors evaluated across four benchmark datasets (Wikipedia, Reddit, MOOC, LastFM) and seven TIG backbones (e.g., TGN, DyRep, TIGER).
- **task improvements**. The authors show that the TIGPrompt can improve baseline performances across a variety of TGNNs on both link and node tasks.

### Weaknesses
I believe the current work has the following weakness:
- **limited evaluation metrics**: the link prediction experiments rely primarily on Average Precision (AP), whereas more robust ranking-based metrics such as Mean Reciprocal Rank (MRR) have been extensively adapted in prior work such as TGB[1] and ROLAND[2]
would provide a fairer and more direct reflection of the improvement of TIGPrompt which leads me to the next weakness. 

- **performance saturation**: the main problem of the AP / binary classification evaluation lies in its over-inflated performances. 
This saturation makes it difficult to assess whether TIGPrompt provides substantial practical improvements or merely marginal gains within an already near-perfect range. For example,  on Wikipedia, DyGformer improvement in baseline is 99.03 and the improvement Projection TProG is 99.80, this is hardly convincing as the evaluation is simply too easy and the task with this evaluation is already solved. This is even worse when considering Table 2 where only 20% of data is used for training and most models can solve the tasks with > 95% AP on two out of the four datasets. 

- **unclear dataset transferability**: the main appeal of prompt learning is its ability to adapt to new datasets. From the provided results, it seems TIGPrompt mainly focus on improving task transferability on the same dataset yet it is still required that the model is trained and tested on the same dataset. It is unclear how TIGPrompt might be used to facilitate transfer to new datasets, for example, pre-train on Wikipedia and then transfer to Reddit.  

Suggestion: my main suggestion for the author is to provide new results with MRR or other ranking metrics that at least require the model to rank across many potential negative destinations. This will enable the demonstration of potentially more significant empirical gain of TIGPrompt and strengthen its significance. The near perfect performance of AP is not a good indicator for good performance as the task is too easy. 

[1] Huang S, Poursafaei F, Danovitch J, Fey M, Hu W, Rossi E, Leskovec J, Bronstein M, Rabusseau G, Rabbany R. Temporal graph benchmark for machine learning on temporal graphs. Advances in Neural Information Processing Systems. 2023 Dec 15;36:2056-73.

[2] You J, Du T, Leskovec J. ROLAND: graph learning framework for dynamic graphs. InProceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining 2022 Aug 14 (pp. 2358-2366).

### Questions
- it seems that TIGPrompt achieves good training efficiency with using only a small amounts of data. The four datasets benchmarked are all on the smaller side with only a few million edges, would it be possible to run TIGPrompt on large temporal graph datasets such as those in [TGB](https://tgb.complexdatalab.com/), i.e. with tens of millions of edges. Maybe with TIGPrompt it is possible to only use 10% of data in training, thus enabling existing models to scale to datasets where they would normally not be able work. 

- the Transformer TProG only considers one hop neighborhood, would more hops help?

- what if we don't even need the base TGNN and just use the prompt model like TProG for prediction?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper works on prompt learning on temporal interaction graphs (TIGs), i.e. a task, where one has a pre-trained graph model, which should be used for another task. Instead of changing the entire model, one keeps the pre-trained GNN frozen and learns only small prompt modules that adapt it to the new task. 
The work adresses this task by introducing Temporal Interaction
Graph Prompting (TIGPrompt), a framework that is supposed to bridge temporal and semantic gaps by integrating with existing TIG models.
They conduct multiple experiments, and the code is publicly available.

### Strengths
* clear and good writing
* good motivation, with good examples in the introduction, and a figure highlighting the temporal and semantic gaps
* methodology (TProG) is conceptually simple and integrates easily with existing TIG backbones
* the paper is well written and structured, with clear problem framing around the temporal and semantic gaps.
* There are many experiments across several temporal graph models and tasks demonstrate that prompt-based adaptation can be both effective and parameter-efficient.

### Weaknesses
## 1. strong overlap with 1.5 year old arxiv preprint (march 2024)
This work was initially released as an arXiv preprint in early 2024 ("Prompt Learning on Temporal Interaction Graphs") and has not been substantially updated since. Given that the community has already built upon and compared against this method (e.g., DygPrompt, ICLR 2025), in my opinion the contribution is no longer timely for ICLR 2026, despite being well executed. This would not be so much of an issue, if the experiment and related work section was updated, and compared to the works that have been introduced since then.

## 2. missing discussion and comparison to related work
* Since the arXiv release of Prompt Learning on Temporal Interaction Graphs (March 2024), other research has already extended this work: The most important one is Node-Time Conditional Prompt Learning in Dynamic Graphs (DygPrompt, ICLR 2025). 
* DygPrompt explicitly positions itself as an improvement over TIGPrompt, arguing that 
>While [TIGPrompt] employs time-aware prompts, it lacks fine-grained node-time characterization and is thus unable to capture
complex node-time patterns, where nodes and time mutually influence each other.  DygPrompt explicitly conditions prompts on both node identity and temporal context.

* In their paper and review discussions, the DygPrompt authors benchmarked their model against TIGPrompt. Because TIGPrompt’s code was not publicly available at the time, they reimplemented it and evaluated both methods on the same datasets. Their results show consistent improvements for DygPrompt, and they also introduced a more challenging low-resource evaluation protocol (see below).
* Given that DygPrompt explicitly positions itself as an improvement over TIGPrompt and has been publicly peer-reviewed at ICLR 2025, it now represents the de-facto state of the art in prompt learning for temporal graphs. 
* The present submission does not mention DygPrompt, reproduce its evaluation setup, or provide any updated comparison, which substantially weakens its novelty and relevance for ICLR 2026.

## 3. potentially outdated evaluation
* The authors in DygPrompt state in their ICLR 2025 rebuttal the following:
>TIGPrompt [4] uses "50% of the data for pre-training and 20% for prompt tuning or fine-tuning, with the remaining 30% equally divided for validation and testing." (see Section 4.2 of TIGPrompt). Note that pre-training data do not require any labeled examples, while prompt-tuning/fine-tuning data require labels for node classification. Hence, TIGPrompt requires 20% labeled data for node classification. In our experiments, we use 80% of the data for pre-training (which does not contain any labels for node classification), but only 1% of the data serves as the training pool for prompt tuning, with each task leveraging only 30 events (about 0.01% of the entire dataset) for prompt tuning (where only the starting nodes in these events are labeled for node classification). Therefore, our setting focuses on the more challenging low-resource scenario with very few labeled data, as labeled data are generally difficult or costly to obtain in real-world applications [1,3,5]. Hence, our setting is more practical and challenging than TIGPrompt's.```
* I agree with this critique. Using only 1% of labeled data for prompt tuning is indeed a more realistic and demanding setting than TIGPrompt’s 20%.
* Therefore, I believe TIGPrompt’s current evaluation protocol is outdated.
* It would be valuable to hear the authors’ thoughts on whether they have tested TIGPrompt under such low-data conditions, and what their opinion on this setup is.

## Overall 
Overall, the paper is well structured and clearly written, but it is mostly identical to its 2024 arxiv version without incorporating developments that have occurred since then. 
Because the authors have not updated or engaged with newer work, especially DygPrompt, the contribution feels dated and this leads to limited relevance for ICLR 2026. Thus I recommend rejection.

### Questions
1. Could you please clarify why DygPrompt (ICLR 2025) was not cited, discussed, or compared against in your submission?
2. Have you reproduced DygPrompt’s evaluation setup or considered running TIGPrompt under the same conditions?
3. The original TIGPrompt uses 20% of labeled data for prompt tuning, whereas DygPrompt uses only 1%, arguing it’s more realistic. Do you have any thoughts or experiments on whether your method still performs well under these stricter conditions? Could you update your evaluation or provide additional experiments?
4. Since TIGPrompt was released in March 2024 and DygPrompt builds on it, how would you position TIGPrompt’s contribution today relative to the current state of the art? Are there aspects of TIGPrompt that remain novel or useful even after DygPrompt’s improvements?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Temporal Interaction Graph Prompting (TIGPrompt), a framework addressing two critical gaps in existing TIG models: the temporal gap (performance degradation on future data) and semantic gap (misalignment between pretext and downstream tasks). The authors propose a "pre-train, prompt" paradigm with a Temporal Prompt Generator (TProG) that creates personalized, time-aware prompts for nodes. Three TProG variants are presented: Vanilla (learnable vectors), Transformer (encoding recent neighbors), and Projection (combining personalized vectors with time encoding). Experiments on four datasets with seven TIG models demonstrate performance improvements.

### Strengths
1. This work is based on a well-motivated problem formulation. Figure 1(a) clearly shows the temporal gap through performance degradation on temporally distant data, and Figure 1(b) provides quantitative evidence of the semantic gap between link prediction and node classification. These gaps are effectively demonstrated.
2. The proposed TProG has a flexible framework design. The paper introduces three complementary TProG variants for different use cases, and the approach can be extended to a "pre-train, prompt-based fine-tune" paradigm in resource-rich settings.
3. The results are supported by strong experimental validation. Extensive experiments are conducted across 4 datasets, 7 models, and 2 downstream tasks, and the performance improvements are evident in several settings.

### Weaknesses
1. The paper claims to be the "first attempt" at prompting for TIGs, but without direct empirical comparison to the concurrent DyGPrompt (whose code is unavailable), the claim of "first" is hard to verify, and the method's positioning relative to related work remains somewhat unclear. In addition, there are existing studies on snapshot-based dynamic graphs [1], which should be discussed more thoroughly.
2. The main novelty of this paper is applying prompting to TIGs to address temporal gaps in distant inference data and semantic gaps in multi-task learning. However, in real-world applications, it is worth questioning whether the extra effort of using prompting is truly more efficient than retraining or incrementally updating the model with new data. Prompting offers an incremental gain, while regular and necessary model updates may lead to more substantial improvements.
3. The proposed TProG framework relies on standard components such as learnable vectors, Transformers, and MLPs, which limits its algorithmic novelty and makes the contribution mostly incremental. Moreover, beyond empirical results, there is no theoretical explanation for why prompt fusion helps reduce the gaps, which somewhat weakens the depth of the analysis.
4. Although Section 3.2 explains the experimental setup, the difference in total training data used (50%+20% in this work vs 70% in prior related work) still raises concerns about potential unfairness in the comparison.
5. As noted in Appendix J, "we may need to conduct additional experiments to determine which TProG variant works better." It would be helpful if the authors could provide clearer guidance on how to choose among TProG variants for new datasets or models, along with more analysis of the tradeoffs between computational cost and performance across the variants.

Reference

[1] ProST: Prompt Future Snapshot on Dynamic Graphs for Spatio-Temporal Prediction https://dl.acm.org/doi/pdf/10.1145/3690624.3709273

### Questions
Please see weaknesses.

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
The paper introduces a novel paradigm for temporal graph learning, namely the "pre-train, prompt" framework, to systematically address two challenges of existing temporal graph models: temporal graph gap, where model performance degrades on temporally distant inference data; and semantic gap, where the model performs worse on a downstream task that is different from the task used in training. The authors propose three model-agnostic designs for the Temporal Prompt Generator (TProG): vanilla TProG, Transformer TProG and Projection TProG. Empirical experiment has demonstrated the substantial performance improvements by the new paradigm and TProGs in tasks: transductive link Prediction, inductive link Prediction and node classification.

### Strengths
- The paper formally defines and empirically quantifies the "temporal gap" and "semantic gap," the authors provide clear diagnostic tools and theoretically motivated objectives
- The paper proposed a novel "pre-train, prompt" paradigm and three approaches for the Temporal Prompt Generator (TProG)
- The paper conducts different experiment settings, including transductive/inductive link prediction, node classification, limited training and prompt data, performance w.r.t. the proportion of prompting data.to rigorously showcase the substantial improvements provided by the "pre-train, prompt" paradigm and TProGs
- The model-agnostic applicability of both the "pre-train, prompt" paradigm and TProG is robustly demonstrated, with clear performance gains reported on both memory-based and non-memory-based temporal graph models
- The new paradigm and TProGs are highly efficient: only the prompt generator is updated during adaptation, which significantly reduces computational cost and data requirements, enabling weakly supervised learning in resource-constrained scenarios

### Weaknesses
**W1: Reproducibility.**: The Anonymous Repository link is currently inaccessible. I recommend that the authors make sure to double-check the link and repository accessibility for the camera-ready version.

**W2: Notation and problem definition.** There are notations used without definition (e.g. line 160: $G$, $\mathcal{V}$ and $\mathcal{E}$). There is no mathematical definition of the temporal interaction graph (especially clarity on whether the paper focuses on the discrete-time (DTDG) or continuous-time dynamic graph (CTDG) setting), link and node prediction task formulation. I recommend that the authors add a dedicated section before the Proposed Methods to define key concepts, notation, and task settings to enhance clarity. 

**W3: Metric.** The results in the paper are mainly reported using Average Precision (AP). I recommended that the authors consider including additional evaluation metrics such as Area Under the Curve (AUC) and, in particular, Mean Reciprocal Rank (MRR) [1]. This would provide a more comprehensive and comparative assessment of model performance on temporal graph learning tasks.

**W4: Task diversity.**  The paper currently focuses on link property prediction and node property prediction tasks, without addressing graph property prediction. However, this limitation is acknowledged by the authors in Appendix Section J.

**Minor**

**W5. Paper representation.** I recommend that the authors use \citep to introduce parentheses between method names and corresponding author names

--- 
[1] Huang, Shenyang, et al. "Temporal graph benchmark for machine learning on temporal graphs." Advances in Neural Information Processing Systems 36 (2023): 2056-2073.

[2] Shamsi, Kiarash, et al. "GraphPulse: Topological representations for temporal graph property prediction." The Twelfth International Conference on Learning Representations. 2024.

### Questions
- Transformer TProG uses a transformer to generate temporal prompts $p_v$. How computationally expensive is fine-tuning this compared to baseline and baseline + other variants of TProG? Could the authors provide a computational analysis showing the increase in inference time for baseline without TProG and baseline with TProG?

- In Table 1, could the authors explain why Vanilla TProG sometimes outperforms the Transformer and Projection variants?

- In the setting where baseline+TProG is trained on link prediction, and then only the prompt (TProG) is fine-tuned for node classification, how does the performance of baseline+TProG  compare with models trained directly on node classification tasks?

- In Section D.2.1, why did the authors choose to evaluate only the first strategy for experiments? Studying all strategies and reporting their performance differences would enhance understanding of TProG's applicability.

- Regarding the domain gap, can any of the three proposed TProGs address transferability when a TIG model trained on one temporal graph domain is evaluated on a different domain? Can the authors suggest potential directions for designing a new variant TProG that improves transferability across different temporal graph domains?

- In Figures 7 and 8, could the authors explain why increasing the amount of prompt tuning data sometimes leads to a decrease in performance?

### Soundness
3

### Presentation
3

### Contribution
3
