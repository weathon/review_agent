# LLM4EHR: Aligning Clinical Time Series with Medical Event Sequences via Large Language Models

- Decision: Reject
- Scores: 8, 4, 6, 4, 2, 2, 4

## Abstract
Recent research in clinical machine learning, focusing on the intensive care unit (ICU), has shifted from bespoke supervised models to foundation models, utilising Large Language Models (LLMs). Here, LLMs are fine-tuned on mixtures of complex clinical data modalities, useful for various downstream tasks. However, existing methods do not sufficiently explore the shared temporal structure between the events on Electronic Health Records (EHRs) and clinical Time Series (TS) observations. This limitation potentially leads to less robust and adaptive clinical foundation models, resulting in reduced performance on downstream tasks. To fully exploit this temporal structure, we propose LLM4EHR, a new clinical foundation model trained on ICU data.
Combining pre-trained LLMs with additional trainable layers, we fine-tune our model to temporally align the EHR and TS modalities. For this, we propose a regularised contrastive objective to jointly learn representations of EHRs and clinical TS. 
Supported by an ablation study, we find that embeddings from LLM4EHR improve performance on various downstream clinical tasks with competitive performance in a few-shot setting. Further, we empirically demonstrate that LLM4EHR learns transferable clinical TS embeddings that can be deployed to new cohorts with minimal performance loss. These findings provide a step towards building more generalisable and performant clinical foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
​​LLM4EHR is a clinical foundation model that temporally aligns ICU time-series measurements with EHR event sequences using a frozen LLM encoder, patch-wise pooling, and a regularized contrastive objective to learn shared cross-modal representations. In experiments on mimic-iii and eICU, these embeddings improved downstream task performance and transferred to new cohorts with minimal performance loss.

### Strengths
- Clear, principled cross-modal alignment using an EHR-similarity-weighted contrastive objective that mitigates class collision.
- The problem being solved matters, EHR event sequences and physiologic time series are usually modeled separately, losing crucial temporal context that impacts diagnosis, risk prediction, and treatment timing, aligning them can boost accuracy and generalization with fewer labels.
- Practical temporal handling via non-overlapping time patches that bridge sparse EHR events and dense time-series data.
- Stable use of a frozen LLM with only new clinical token embeddings trained, leveraging general semantics without drift.
- Auxiliary autoregressive reconstruction preserves numeric fidelity of physiologic signals, with ablations showing its value.
- Strong, consistent improvements across multiple ICU prediction tasks and solid cross-dataset transferability.

### Weaknesses
- The few-shot claim is weakly supported; the experiments fine-tune on relatively large labeled cohorts and don’t show behavior at truly low-label regimes.
- Length-of-stay performance lags specialized baselines, and the paper offers limited concrete strategies to mitigate this gap.
- Heavy reliance on frozen LLM semantics for many new clinical tokens is unvalidated, there's no check that these learned token embeddings are clinically coherent.
- The choice to normalize target-domain data with source statistics is not standard and could bias transfer results, no ablation compares against target-stats normalization.

### Questions
- Why normalize target cohorts with source means/variance, can you provide an ablation with target stats normalization?
- How are patches with no EHR events handled in the alignment losses and what are the gradient implications?
-  How did you or would you validate the semantic quality of new clinical token embeddings, and would light adapter-tuning help?
- I wonder how sensitive are results to patch size and do you support off-diagonal alignment to capture realistic delays between orders, administrations, and physiologic response?
- Can you show learning curves for 1–5–10% labeled data or k-shot per phenotype to substantiate the few-shot claim?
- Could you report calibration metrics (brier) and whether the EHR-weighted term improves or harms calibration versus the baseline objective?
Minor:
- in line 67, duplicated word: calculating
- line 169-170, word predictive overused
- line 346, missing rationale, why the sequences were truncated at 200h?
- l. 486 should be ETHICS
- Please fix table headers with mirco instead of macro, and marco -> macro

### Soundness
4

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
4

### Summary
The paper introduces LLM4EHR, a multimodal clinical foundation model that jointly learns from electronic health record event sequences and clinical time series data. The authors argue that prior approaches fail to capture the temporal dependencies between these two modalities and propose a contrastive alignment objective that aligns TS observations with EHR event embeddings in a shared latent space. The model leverages pre-trained language model embeddings to regularize the contrastive loss and reduce class collision during training. LLM4EHR is trained on MIMIC-III and eICU datasets and evaluated across several downstream prediction tasks such as mortality, phenotyping, decompensation, and length of stay. Results show improved few-shot and cross-dataset generalization compared to baseline models. The paper concludes that the approach improves interpretability and transferability but faces scalability limitations due to memory demands of large LLMs.

### Strengths
Below are the strengths of the paper in my opinion:
1. Proposes a clear methodological contribution for temporal contrastive alignment using LLM embeddings.
2. Demonstrates consistent performance improvements across multiple clinical prediction benchmarks (mortality, phenotyping, decompensation, and length of stay) under few-shot and cross-dataset settings.
3. Includes interpretability analysis showing improved consistency of learned embeddings.
4. Evaluates on diverse datasets (MIMIC-III, eICU, Physio2012, and a private PICU dataset) with transparent experimental setup.

### Weaknesses
The major weaknesses of the paper in my opinion follows:

1. Offers an incremental contribution that primarily combines established contrastive and LLM-based methods rather than introducing a new paradigm.
2. The "foundation model" claim is overstated given the limited dataset scale and scope of downstream tasks.
3. Experimental analysis lacks depth (few ablations, no statistical significance reporting, limited robustness discussion).
4. Scalability remains a limitation due to high computational requirements for larger LLMs. (Although it is already noted in the paper)
5. Minimal qualitative or clinical validation beyond numerical benchmarks.
6. There is no clear comparison against recent multimodal architectures that jointly model structured and unstructured EHR data (e.g., transformer-based fusion models).
7. The current interpretability for such a clinical task lacks rigour and can be substantially improved.

### Questions
1. Please provide precise LLM configuration and compute profile. Please specify the exact sequence length, tokenizer choices, and any truncation rules for EHR events or TS tokens. Also report pretraining steps, batch sizes, device count, total GPU hours, and peak memory. This will help assess scalability, which you list as a key limitation.
2. You reformulate the commonly used contrastive objective to temporally align TS observations with EHR event sequences. Please formalize the positive and negative pair construction in time, the windowing or lag structure, and how you handle irregular sampling or missing TS. I think adding a figure or pseudo-code that shows how pairs are built over time would greatly improve this part.
3. Tables show means with parentheses that are std. Please add significance tests for key claims (few-shot gains, cross-dataset transfer.
(minor comment: For all tables please specify what the numbers in parentheses are (i.e., std).)
4. You repartition to 70% self-supervised pretraining, 20% fine-tuning, 10% testing. I suspect you did this already but can you please confirm splits are at the patient level and that no patient overlap exists across partitions or datasets (especially since for these datasets a given patient might have multiple visits so the visit id defers but patient id is the same). Describe any harmonization across MIMIC-III, eICU, and PhysioNet to avoid label or feature leakage, especially for remaining length of stay.
5. You state that, inspired by prior work, the model can make dynamic downstream predictions, such as an hourly mortality forecast. Yet the evaluation emphasizes classification. Can you please provide a forecasting setup with proper rolling origin evaluation, and reconcile this with the later statement that the method is less suitable for regression.
6. Please enumerate baseline implementations and hyperparameter search spaces in the main text or appendix. Clarify whether recent multimodal fusion baselines were included, not just generic TS or EHR models, since your contribution is cross-modality alignment.
7. I think adding calibration and other clinically meaningful decision metrics where applicable improves the paper. This will make the cross-dataset claims stronger, especially for mortality risk.
8. For reproducibility, how will the PICU-dependent steps be handled. (I assume that private data is not going to be released).
9. Can you please define how continuous TS vectors map to tokens, the discretization or projection used, and how variable-length episodes are handled?
10. You note poorer regression performance. Please provide a short analysis of failure modes and whether alternative reconstruction losses, discretized targets, or ordinal objectives improved results.
11. I think the description for the exact few-shot protocol can be improved by providing much more detail in the appendix such as shots per class, selection strategy, number of repeats, and how hyperparameters were tuned without peeking. This is important for interpreting "few-shot" gains.
Evidence: “Few-shot evaluation on MIMIC-III” in the outline.

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
5

### Summary
The paper proposes LLM4EHR, a clinical foundation model that temporally aligns ICU EHR event sequences with numerical clinical time series (TS) to learn cross-modal patient representations. The method freezes a pre-trained LLM backbone and builds patch-level embeddings for EHR and TS, then trains with a regularized contrastive objective to align modalities, plus a TS reconstruction loss. Experiments on MIMIC-III and eICU, with transfer to PhysioNet 2012 and a paediatric PICU cohort, show consistent gains on classification tasks (phenotyping, decompensation, mortality) and competitive but not SOTA performance on remaining length of stay (LoS) regression. Ablations cover temperature τ, patch size (five-hour windows), and LLM backbone choices.

### Strengths
1.Originality: Introduces semantic-weighted contrastive alignment between EHR and TS at the temporal patch level, which is a meaningful extension of multi-modal contrastive learning in clinical settings. 

2.Quality: Broad evaluation (few-shot hints, in-domain and cross-dataset) and ablations (τ, patch size/backbone). Cross-dataset mortality results show robust transfer. 

3.Clarity: The training objective and data flow are well presented (overview figure, patching diagram, tables).

### Weaknesses
1.Regression performance / numerical fidelity: Remaining LoS performance is only competitive; the paper itself hypothesizes TS embedding distortion. Consider adding variable-level numeric reconstruction, distribution/quantile losses, or hierarchical multi-tasking to improve numerical fidelity and report the impact on LoS. 

2.Robustness of semantic weighting (ω): If EHR coding is sparse/noisy or mismatched across sites/ages, ω could mislead alignment. Please simulate label/semantic noise, compare against unweighted or asymmetric weighting schemes, and quantify degradation. 

3.Temporal alignment granularity: Fixed, non-overlapping five-hour patches may miss asynchronous or delayed effects common in ICU. Explore adaptive/learned patching, overlapping windows, or soft DTW-like temporal weights.

### Questions
Q1 (critical): How robust is ω under coding-system changes or age-group shifts (adult ↔ paediatric)? Please report cross-site/cross-coding ablations or controlled noise experiments (e.g., token description perturbation, increased OOV rate).
Q2: For LoS, does adding variable-level numeric reconstruction or quantile losses improve RMSE/R² without hurting classification? A small ablation in the appendix would help.

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
4

### Summary
The author proposes LLM4EHR on general ICU data. LLM4EHR is built on pretrained LLMs to embed unstructured EHR text data, as well as autoregressively recover time series values from their latent representation. The major contribution in LLM4EHR is the temporally aligned embedding of EHR text records and time series records, as well as an additional regularization loss term to address the issue of class collision. Empirically, the author demonstrates that the embeddings from LLM4EHR improve various downstream tasks.

### Strengths
* The goal of this work, which is to improve the analysis of ICU data using both time series data and EHR data, has a significant impact and is beneficial to healthcare research.
 * The major novelty in the proposed work, aligning the embeddings of two modalities for a feasible contrastive loss, and an additional regularization loss utilizing the feature of LLM, is reasonable.
 * The experiments cover a variety of downstream tasks, showing both strengths and potential drawbacks of the proposed model.

### Weaknesses
* The major concern is that the benchmark multimodal models are not state-of-the-art. For example, some more recent works also study EHR / clinical note + time series representation learning ([1] Ma, Yingbo, et al. "Global contrastive training for multimodal electronic health records with language supervision." arXiv preprint arXiv:2404.06723 (2024).  [2] Wang, Fuying, et al. "CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis." arXiv preprint arXiv:2411.00696 (2024). [3] Cui, Hejie, et al. "Multimodal fusion of ehr in structures and semantics: Integrating clinical records and notes with hypergraph and llm." arXiv preprint arXiv:2403.08818 (2024).). A comparison between the proposed model and more recent multimodal EHR works will strengthen the work significantly. 
 * There is no explanation on how the learned embeddings of time series data are used to perform the downstream tasks studied in section 5.2

### Questions
* In Figure 3b, the legend on the top right says  "0 < w <= "; something is missing there.
* Equation 1 confuses me. If v and z are not aligned, then how can avgpool of the same kernel size & stride make aligned patches of v and z? For example, if v has a time length of 12 and z of 9, then a kernel of size 3 will give 4 patches of v and 3 patches of z. How are those patches further aligned?
 * In section 5.2, the paragraph says "Decompensation and remaining LoS predictions were made hourly, and we evaluated the remaining LoS predictions in days, as in Sheikhalishahi et al. (2020)." It is very confusing to read, and it will be clearer if the author adds the scale (hourly or daily) of remaining LoS predictions as Decompensation (hourly) in Table 3.
 * The model uses LLM to further embed the time series embedding (Figure 2). Is there any justification for this model design, other than that the AR generation of the next token can be naturally used to recover the time series data?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes LLM4EHR, a framework designed to learn joint representations of two modalities in : structured EHR sequences  and numerical clinical time series. The core method is to use the LLM to extract embeddings for  EHR events and to then align them with the time-series data by optimizing a contrastive learning objective. Another innovation is to use the semantic similarity of EHR events to weight the contrastive loss for time series, aiming to mitigate "class collision." The model is evaluated on downstream tasks on MIMIC, eICU, etc and shows superior performance compared to baselines.

### Strengths
The core idea of aligning EHR events and time series is a reasonable research direction. Moving beyond instance-wise alignment can be conceptually interesting. Additionally, the problem of better EHR data use is highly relevant. Developing strong foundation models for EHR data has great value for clinical AI. Additionally, the paper is generally well-structured, and the figures pretty illustrative. Finally, I like the few-shot and cross-dataset evaluation of the model.

### Weaknesses
Please see questions. Additionally, I am an emergency reviewer, so I have not had the chance to read the paper in detail. If I have misunderstood or missed anything, please bring it to my notice.

### Questions
How is the temporal alignment between EHR and the time series achieved/ how are the two modalities reconciled for patch creation. 
If a patch contains 6 hours of time series but only 2 EHR events, what is the 'alignment'? What about 1 EHR event? 

The model underperforms on the LoS task, which is supposed to be due to 'distortion in embeddings'  Why does a model designed for temporal understanding underperform on temporal regression problem?

Why LLM? If i understand correctly, the LLM is a feature embedder.  Comparing the A.8 results, it seems that the more powerful newer LLMs (like llama) do similar to old models like BERT/RobertA. This does not seem to be a framework which at core rely on a LLM knowledge or ability.  
Additionally for models like LLama, how were the embeddings obtained? Is it pooling tokens, using a pretrained MLP, etc. please give details.

Can you add comparisons with a direct baseline that combines the embeddings from a pre-trained EHR LLM (like the one you used) and a pre-trained time series model. Is there other experiments that show this compute-heavy training method is better than such a simpler, more interpretable approach?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 6

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a model for numeric EHR data that integrates text annotations for some observed variables. It proposes a contrastive task to train this model using various learned layers attached to a frozen language model backbone.

### Strengths
- Integrating language embeddings into clinical TS models is relevant to study and could boost performance.
- The use of contrastive training between reasonably well-aligned data is promising, given that other multimodal clinical data is challenging to use constrastive training on, due to its poor alignment (e.g., clinical notes versus vitals).

### Weaknesses
- The paper presents a separation between clinical time series and EHR entries that's unclear and doesn't reflect the nature of the data. These are not inherently distinct modalities: most EHR entries in the datasets being described are irregular samples of clinical time series, and are used as such in previous work (e.g., EBCL). For another example, the time series features in the Harutyunyan et al. MIMIC-III baseline are derived from chart events and lab events. While it's not clear how this paper understands chart events, it describes lab events as EHR entries even though they contain the same data as in an irregular time series representation. The practical relevance of this model's multimodality is therefore limited.
- I couldn't find basic information about the architecture in the text, including what the "Timeseries embedding" and "Timeseries decoder" blocks in Figure 2 are and how predictions are generated for fine-tuning and inference.
- Hyperparameter tuning for baseline models seems to be missing.
- The main evaluation is limited, being a few-shot prediction with only one fraction of labelled training data evaluated. Full-shot results are not given. Since the tasks being evaluated were generated from EHRs without manual labelling, they aren't the kind of medical modelling tasks where few-shot capabilities are particularly relevant.

### Questions
- While using a large language model for text embedding is standard, using one to embed time series with no text information seems awkward, and especially a frozen one. Why not use a time series embedding model instead, or at least train the transformer weights?
- Can you explain the missing elements in Table 3? It's not apparent to me why instances couldn't be constructed for the corresponding models on an hourly basis. While the appendix indicates that EBCL was intended for sequence classification, I would note that their paper does include length-of-stay regression forecasting results.
- Are there cases in the datasets of multiple episodes corresponding to the same patient, and if so, do you ensure that these remain in the same partition?
- How are EHR entries used during fine-tuning and evaluation, for your method and for the other methods?
	- For one, EBCL is designed to use lab events and other features that are discussed as EHR entries in this paper. Were they provided to your EBCL implementation when training and evaluating it?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 7

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Apologize !
I was assigned as an additional reviewer after the other reviews had already been completed, so I am submitting this review later than the official deadline.
I apologize for this; please consider my comments with the understanding that the authors don't have enough time to fully address them during the rebuttal phase.

The paper proposes LLM4EHR, a multimodal ICU model that uses a shared frozen GPT-2 backbone to encode both structured EHR events and physiologic time series. 
Time series are projected into the LLM embedding space, temporally patched, and trained with a combination of InfoNCE, an EHR-guided ω-regularised contrastive loss, and a next-step reconstruction loss. 
The model is evaluated via linear probing on multiple ICU prediction tasks and cross-dataset transfer, showing strong gains over TS-only and prior multimodal baselines, especially for classification and domain shift.

### Strengths
Conceptually clean architecture: A single frozen LLM shared by EHR and TS with only lightweight projection and head layers makes the approach simple, reusable, and computationally realistic.

Novel EHR-guided contrastive objective: The ω-regularised loss uses EHR semantic similarity to shape TS–TS relations, reducing class collision in InfoNCE and yielding more clinically meaningful TS representations.

Strong robustness and transfer: LLM4EHR consistently outperforms supervised TS models, TS self-supervised methods, and other multimodal baselines on phenotyping, decompensation, and cross-dataset mortality (including adult→pediatric transfer).

Solid ablations: Loss component and backbone ablations clearly support design choices, showing that L_ω mainly drives classification gains while L_recon is important for LOS and preserving numeric TS information.

### Weaknesses
1. Unclear whether LLM “knowledge” is truly leveraged.

Although the method is framed as an LLM-based clinical foundation model, in practice the LLM backbone is completely frozen and used purely as a encoder. There is no direct evidence that linguistic/medical knowledge from GPT-2 meaningfully drives the improvements (e.g., no comparison against a similarly sized transformer trained from scratch - SAND may have much smaller embedding dimension size, no analysis of whether EHR token semantics matter beyond providing any transformer encoder).
As a result, it is not entirely convincing that this is really “leveraging an LLM” rather than just using a convenient off-the-shelf backbone.

2. Only linear probing is considered; no end-to-end fine-tuning of GPT-2.

All downstream results are reported under a frozen-encoder, linear-head regime.
Given that GPT-2 at this scale (hidden size 768) is not extremely large by modern standards, it seems feasible to explore at least partial end-to-end fine-tuning (e.g., last few layers, adapters/LoRA), which may have better performance.
Without such experiments, it is hard to judge whether the proposed alignment and loss design remain beneficial once the backbone is allowed to adapt, or whether the gains are specific to the somewhat artificial “representation-only / linear probe” setting.

3. Multimodal and alignment claims remain indirect.

EHR is only used during pretraining and not used at inference (I understand, the number of EHR code used in this work are somewhat limited), and there is no direct embedding-level evidence (e.g., cross-modal retrieval, visualization, similarity analysis) that the model achieves genuine TS–EHR alignment. The observed gains could plausibly be explained by generic TS representation regularization rather than by strong multimodal/LLM effects.

### Questions
1. Have you tried partially unfreezing the backbone (last block, adapters, or LoRA) during pretraining, and if so, how does this affect downstream performance, LOS regression, and the stability of the EHR semantic space? 

2. In the backbone and baseline comparisons, GPT-2 small uses a hidden size of 768, but the paper does not report the embedding/hidden dimensions or parameter counts for baselines such as SAnD, EBCL, or the TS self-supervised models.
Could you clarify:
(a) the hidden size and total parameter count for each baseline model, and
(b) whether your gains might be partly explained by capacity differences rather than architectural advantages?

3. You may also want to discuss your method in relation to GenHPF (Hur et al., 2022; arXiv:2207.09858), which similarly encodes EHR using a text-based representation, but goes further by converting time-series values into text and using them directly without an explicit alignment stage. 
I do not expect a comparison given time and resource constraints, but a short conceptual comparison in the related work or discussion section would help clarify how LLM4EHR differs from this line of text-based EHR/TS modeling.

### Soundness
3

### Presentation
3

### Contribution
3
