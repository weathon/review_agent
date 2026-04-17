# Are EEG Foundation Models Worth It? Comparative Evaluation with Traditional Decoders in Diverse BCI Tasks

- Decision: Accept (Poster)
- Scores: 4, 2, 8, 8

## Abstract
Foundation models have recently emerged as a promising approach for learning generalizable EEG representations for brain–computer interfaces (BCIs). Yet, their true advantages over traditional methods—particularly classical non-neural approaches—remain unclear. In this work, we present a comprehensive benchmark of state-of-the-art EEG foundation models, evaluated across diverse datasets, decoding tasks, and six evaluation protocols, with rigorous statistical testing. We introduce spatiotemporal EEGFormer (ST-EEGFormer), a simple yet effective Vision Transformer (ViT)-based baseline, pre-trained solely with masked autoencoding (MAE) on over 8M EEG segments. Our results show that while fine-tuned foundation models perform well in data-rich, population-level settings, they often fail to significantly outperform compact neural networks or even classical non-neural decoders in data-scarce scenarios. Furthermore, linear probing remains consistently weak, and performance varies greatly across downstream tasks, with no clear scaling law observed among neural network decoders. These findings expose a substantial gap between pre-training and downstream fine-tuning, often diminishing the benefits of complex pre-training tasks. We further identify hidden architectural factors that affect performance and emphasize the need for transparent, statistically rigorous evaluation. Overall, this study calls for community-wide efforts to construct large-scale EEG datasets and for fair, reproducible benchmarks to advance EEG foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces a benchmark to compare EEG foundation models and baseline deep neural networks. It also introduces a ViT based model that is used to compare the other foundation models with.

### Strengths
The paper is very well organised. The authors provide a plethora of experiments, comparisons as well as ablation studies. These experiments range from various models, experimental setups (e.g. how the various ways one can treat training subjects) and classification / regression tasks. All of these make the analysis meticulously thorough.

### Weaknesses
1. Although very thorough, I fail to see the novelty behind the paper. For example, a recent work [1], although it covers less in experimental results, it comes across the exact same observations / conclusions. One could argue that this newest work extends [1] but its insights do not seem novel. 
2. The main message of the paper is also complicated. Is it a benchmarking paper or a new model (in this case ST-EEGFormer) introduction ?
3. The authors challenge LaBraM’s claim that direct MAE in EEG signals is ineffective but they fail to provide any evidence against it. For example, a comparison between reconstruction capabilities of LaBraM’s tokenizer and ST-EEGFormer would be useful here.

Writing:
The paper is well-written.

Overall:
The paper is clear and with insightful observations / results but the main message or its novelty need to be clearly decided.

[1]: Na Lee, Konstantinos Barmpas, Yannis Panagakis, Dimitrios Adamos, Nikolaos Laskaris, & Stefanos Zafeiriou (2025). Are Large Brainwave Foundation Models Capable Yet ? Insights from Fine-Tuning. In Forty-second International Conference on Machine Learning.

### Questions
1. What is the main purpose of the paper and if it’s just benchmarking what’s the novelty compared to [1] ?
2. How about the reconstruction ? Any examples here. 
3. An interesting addition to distinguish yourself from [1] would be an analysis of the number of training samples during deep baselines training and foundation model fine tuning. For instance, if currently 100 samples (all) are used for training of both models and the performance is similar how about if you use 10-20-40-60-80 random samples (over multiple sampling) for deep baselines training and foundation model fine tuning. Do the foundation models have any advantage there ?

[1]: Na Lee, Konstantinos Barmpas, Yannis Panagakis, Dimitrios Adamos, Nikolaos Laskaris, & Stefanos Zafeiriou (2025). Are Large Brainwave Foundation Models Capable Yet ? Insights from Fine-Tuning. In Forty-second International Conference on Machine Learning.

### Soundness
3

### Presentation
4

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
This paper offers a systematic, large-scale comparison of EEG foundation models against classic neural and non-neural decoders across multiple datasets, tasks, and six evaluation protocols. This paper also introduces a baseline, ST-EEGFormer (a ViT trained with MAE on ~8M raw EEG segments). This work systematically summarizes how different types of models perform under various evaluation protocols and draws a series of performance-related conclusions.

### Strengths
The paper delivers a systematic, comprehensive analysis spanning EEG foundation models, neural network–based approaches, and classical baselines. The empirical workload is substantial and reflects significant effort. It also provides a clear synthesis with thorough descriptions of existing models, and the manuscript is well written and highly readable.Its calls to the community are measured and well-judged.

### Weaknesses
Despite the substantial effort and extensive scope, the work lacks originality. As a benchmarking study, it largely aggregates and uniformly evaluates existing open source methods rather than introducing genuinely new ideas. Although the paper proposes ST-EEGFormer, the architecture amounts to a straightforward Transformer combined with MAE-style pretraining—a paradigm already widely used in EEG self-supervised learning (e.g., EEGPT [1], CBraMod [2], EEG2Rep [3]). Similarly, the spatio-temporal Transformer concept has ample precedent (e.g., CBraMod [2] , Brant-2 [4]). In addition, the manuscript contains several inappropriate or insufficiently rigorous elements (see Questions for details). Taken together, I do not consider this submission a good fit for ICLR; it reads more like an engineering report than a research contribution. Moreover, comparable benchmarking studies already exist such as Adabrain-bench [5], which is open source , so this is not the first of its kind.

[1] Wang, Guangyu, et al. "Eegpt: Pretrained transformer for universal and reliable representation of eeg signals." Advances in Neural Information Processing Systems 37 (2024): 39249-39280.

[2] Wang, Jiquan, et al. "Cbramod: A criss-cross brain foundation model for eeg decoding." arXiv preprint arXiv:2412.07236 (2024).

[3] Mohammadi Foumani, Navid, et al. "Eeg2rep: enhancing self-supervised eeg representation through informative masked inputs." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.

[4] Yuan, Zhizhang, et al. "Brant-2: Foundation model for brain signals." CoRR (2024).

[5] Wu, Jiamin, et al. "Adabrain-bench: Benchmarking brain foundation models for brain-computer interface applications." arXiv preprint arXiv:2507.09882 (2025).

### Questions
1. The six “evaluation protocols” described in this work do not always align with commonly used paradigms or terminology in EEG research—for example, the notions of “population” and the specific zero-shot configurations. To my knowledge, widely adopted paradigms include subject-dependent, subject-independent, cross-subject, cross-session, and cross-dataset. Approximately, your (2) within-subject seems to correspond to subject-dependent; your (4) leave-one-subject-out zero-shot to subject-independent; and your (3) per-subject zero-shot to cross-subject. However, other mainstream settings I mentioned appear to be missing, and some of your configurations—such as (5) and (6)—do not have clear counterparts in established experimental practice. Moreover, the terminology for all protocols is presented without supporting citations. I recommend clarifying the provenance and rationale of each protocol, aligning your naming with standard conventions (or explicitly motivating any deviations), and adding appropriate references.

2. The claim that this work is the “first comprehensive benchmark” does not seem justified; comparable efforts already exist—for example, Adabrain-bench [5].

3. The pretraining paradigm and architecture of ST-EEGFormer are not novel, as closely related designs already exist, thus this model lacks originality. 

4. Moreover, while ST-EEGFormer-s outperforms works like LaBraM and BENDR overall—and ST-EEGFormer-b and -l achieve further gains—I suspect these improvements appear largely attributable to substantially larger model sizes. For example, ST-EEGFormer-s has 32.7M parameters compared with only 5.8M for LaBraM-base, and ST-EEGFormer-b and -l scale up to 110.9M and 328.4M parameters, respectively. This also raises my next question.

5. In Section 4.5 the authors write: “although there is a slight upward trend in normalized accuracy with increasing model size, the poor logarithmic fit suggests that a clear scaling law does not exist for downstream EEG classification tasks.” I believe this analysis is problematic. The comparison mixes models of different sizes and different architectures rather than holding the architecture fixed, yet architectural choices are known to have a large impact on EEG performance; prior classic neural-network baselines for EEG have clearly demonstrated the sensitivity to model design. As a result, the paper’s conclusion about the absence of a scaling law is insufficiently rigorous. Moreover, within the authors’ own ST-EEGFormer family, a clearer scaling trend appears: moving from the small model (32.7M) to the base (110.9M) and then to the large (328.4M) yields consistent performance gains. This, if anything, suggests the presence of a scaling law and contradicts the stated conclusion.

6. The figures are difficult to read—especially the radar charts, which include too many elements to distinguish clearly. While I appreciate the substantial scope of the work, the visual presentation should be improved for clarity.

7. If positioned as a benchmarking suite, the current coverage of models and datasets is relatively limited and leaves ample room for improvement. Mainstream EEG tasks—such as seizure detection, emotion recognition, and mental workload detection—should be included, and additional foundation models (e.g., EEG2Rep [3]) as well as classic deep-learning baselines (e.g., LGGNet [6]) should be incorporated.

In sum, I agree the authors’ call to build a more cohesive and standardized EEG community; however, this advocacy cannot substitute for substantive novelty and technical contributions in the present work.

[6] Ding, Yi, et al. "LGGNet: Learning from local-global-graph representations for brain–computer interface." IEEE Transactions on Neural Networks and Learning Systems 35.7 (2023): 9773-9786.

### Soundness
2

### Presentation
3

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
This paper presents a comprehensive benchmark of EEG foundation models against traditional neural and non-neural decoders across diverse BCI tasks and evaluation protocols. The authors introduce ST-EEGFormer, a Vision Transformer-based model pre-trained with masked autoencoding on over 8 million EEG segments. The study finds that while fine-tuned foundation models perform well in data-rich, population-level settings, they often do not significantly outperform simpler neural networks or classical methods in data-scarce or subject-specific scenarios. Linear probing consistently underperforms, and no clear scaling law is observed. The work emphasizes the need for large-scale datasets and rigorous, reproducible benchmarking.

### Strengths
*   **Comprehensive and Rigorous Benchmarking:** The paper provides an exceptionally thorough evaluation, comparing a wide range of EEG foundation models against both classic neural networks and non-neural baselines across six diverse tasks and six distinct evaluation protocols, supported by extensive statistical testing.
*   **Introduction of a Strong, Transparent Baseline:** The proposed ST-EEGFormer model serves as a valuable and reproducible baseline, effectively demonstrating that a simple MAE pre-training strategy on raw EEG can achieve competitive performance, challenging assumptions from prior work.
*   **Critical and Actionable Findings:** The study delivers nuanced, evidence-based conclusions that question the universal superiority of foundation models, highlighting their limitations in data-scarce settings and the weak performance of linear probing, which provides crucial guidance for future research directions in the field.

### Weaknesses
**Limited Scope of Downstream Task Evaluation**: The selection of seven downstream datasets, while diverse, may not be fully representative of the broad spectrum of EEG applications. The benchmark omits several common and challenging tasks such as emotion recognition (e.g., FACED), sleep staging, and seizure or depression detection, which are frequently used to evaluate foundation models. Furthermore, the chosen tasks like ERP and SSVEP often have simple, stereotypical patterns that can be decoded effectively by classical non-neural methods, potentially skewing the conclusion that foundation models offer limited advantages. The low overlap with the evaluation benchmarks from existing foundation model papers also raises concerns about the generalizability of the findings across the field's common evaluation practices.

**Concerns over Data Transparency for Community Reference**: The heavy reliance on distribution plots (e.g., violin plots, box plots) to present aggregated results, while useful for visualizing statistical distributions, limits the transparency of the raw results. This makes it difficult for the community to directly reference specific performance numbers (e.g., exact accuracy on a specific dataset) or to perform alternative meta-analyses. Providing detailed numerical results in supplementary tables for each model-dataset-protocol combination would enhance reproducibility and utility for future comparative studies.

### Questions
1.  **Data Transparency:** The extensive use of distribution plots provides a excellent overview of model performance, but makes it difficult to extract precise numerical results for community reference and future benchmarking. Could the authors provide a supplementary file or table with the detailed, per-dataset numerical results (e.g., accuracy, MSE, Pearson R) for each model under the different evaluation protocols? This would greatly enhance the reproducibility and long-term utility of this valuable benchmark.

2.  **Ablation on Token Fusion Strategy:** The paper identifies the token fusion strategy (e.g., flatten-all vs. average) as a critical hidden implementation factor. Given its potential impact, have the authors conducted a systematic comparison of how different fusion strategies affect the performance of various foundation models (like EEGPT and CBraMod) across different types of downstream tasks (e.g., motor imagery vs. emotion recognition)? Such an ablation study would be highly insightful for the community to understand the interaction between model architecture, fusion strategy, and task characteristics.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper highlights the importance of comprehensive evaluation—an aspect often overlooked in previous studies—when assessing EEG foundation models. The authors emphasise that fair comparison with classical methods, diverse train–test splitting strategies, and statistical testing of performance differences are essential to truly assess progress in this field. They systematically evaluated five popular EEG foundation models and uncovered meaningful relationships between model performance, downstream tasks, and evaluation protocols, revealing that the performance gains over classical approaches remain limited. Furthermore, they proposed a robust EEG foundation model that serves as a strong baseline and underscored the need for more diverse and large-scale EEG datasets to enable more reliable evaluations.

### Strengths
This paper is the first study in EEG Foundation literature with systematic comparisons and standardised the evaluation protocols. It addresses the fundamental question of whether pre-training large-scale foundation models on extensive EEG corpora truly benefits EEG signal decoding. It also serves as an important reminder to researchers in the field not to blindly upscale models—an approach that often leads to wasted resources—and encourages future studies to pursue more reliable and reproducible results without selective reporting.

Quality and Clarity: 
- The comparative analysis is comprehensive, with evaluation protocols that cover most considerations relevant to real-world BCI applications. Fair comparisons are conducted using matched metrics, training strategies, and model sizes.
- Rationale of most strategic selections are clearly stated in the body or appendices. (e.g., selection of evaluation protocols and pre-training corpus of ST-EEGFormer)
- The Result section (Section 4) is well-organised, addressing important questions related to foundation models.

### Weaknesses
1.	A brief description of the train–validation–test splitting strategy (e.g., whether data were randomly shuffled or split by leaving entire trial sessions out) could be added to Section 3.1, as this choice may influence the performance of models.
2.	The scaling law of EEG foundation models has so far been examined only with respect to model size (i.e., the number of parameters). However, the volume of pre-training data is also a key factor contributing to the computational footprint and could influence the overall performance. Including an investigation of scaling behaviour with respect to dataset size would provide valuable insights and better inform the design of future EEG foundation models.
3.	This paper encourages the community to develop better EEG datasets, but it relies on vague descriptions such as “large-scale,” “diverse,” and “standardized.” These terms are not clearly defined, and the paper lacks elaboration on how greater diversity and standardisation in EEG datasets could specifically address the bottlenecks of EEG foundation models identified in the comparative analysis.
4.	Line 939 of Section C.9 highlights the importance of regression tasks; however, only one regression task is included in the evaluation.
5.	The joint visualisations of all six protocols in Figures 3 and 4 appear somewhat visually cluttered. Since each protocol or pair of protocols evaluates different aspects of the EEG decoding models, selectively presenting the most relevant ones would enhance clarity and allow readers to interpret the results more easily.

### Questions
1.	Only 5 representative EEG foundation models are selected. Would the conclusions generalise to the large body of EEG foundation models not tested in this evaluation study?
2.	Line 89 (section 2) mentions that many studies on EEG foundation model lack comparison to “compact neural network decoders.”  However, the well-cited foundation models -- for example, the one listed in Table C.1 --  are benchmarked against classical neural network models, including both general time-series models and EEG-specific models. Could you please clarify which types of baseline NN models are considered under-explored in existing EEG foundation model research?
3.	The difference between variants of STEEGformer -- STEEGformer-s, STEEGformer-b, and STEEGformer-l, is not clearly stated in the paper. Could you please explain the differences?
4.	The use of the phrases “significant” and “statistically significant” may be slightly misleading to readers in Section 4.3. This could be improved by replacing the non-statistical “significant” with other expressions such as “notable,” or “considerable.”
5.	How are the model sizes for EEG Foundation models in Figure 5 (a) calculated? Are the parameters of classification / regression heads included?

### Soundness
3

### Presentation
2

### Contribution
4
