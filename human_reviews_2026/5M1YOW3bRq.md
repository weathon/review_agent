# A foundation model with multi-variate parallel attention to generate neuronal activity

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Learning from multi-variate time-series with heterogeneous channel configurations remains a fundamental challenge for deep neural networks, particularly in clinical domains such as intracranial electroencephalography (iEEG), where channel setups vary widely across subjects. In this work, we introduce multi-variate parallel attention (MVPA), a novel self-attention mechanism that disentangles content, temporal, and spatial attention, enabling flexible, generalizable, and efficient modeling of time-series data with varying channel counts and configurations. We use MVPA to build MVPFormer, a generative foundation model for human electrophysiology, trained to predict the evolution of iEEG signals across diverse subjects. To support this and future efforts by the community, we release the SWEC iEEG dataset, the largest publicly available iEEG dataset to date, comprising nearly 10,000 hours of recordings from heterogeneous clinical sources. MVPFormer leverages MVPA to achieve strong generalization across subjects, demonstrating expert-level performance in several iEEG tasks. MVPFormer surpasses state-of-the-art (SOTA) Transformer baselines in seizure detection across the SWEC, the MAYO, and the FNUSA datasets, while also achieving SOTA performance on four Brain TreeBank iEEG decoding tasks (volume, pitch, onset, and speech). We further validate MVPA on standard time-series forecasting and classification tasks, where it matches or exceeds the performance of existing attention-based models. Together, our contributions establish MVPA as a general-purpose attention mechanism for heterogeneous time-series and MVPFormer as the first open-source, open-weights, and open-data iEEG foundation model with SOTA clinical performance. The code and weights are available at https://github.com/IBM/multi-variate-parallel-transformer. The SWEC iEEG dataset is available at https://huggingface.co/datasets/NeuroTec/SWEC_iEEG_Dataset.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Multi-Variate Parallel Attention (MVPA), a novel self-attention mechanism that addresses the challenges of multi-variate time-series data with heterogeneous channel configurations. The MVPA mechanism is designed to handle time-series signals that vary across different subjects, particularly in clinical domains like intracranial electroencephalography (iEEG). The model efficiently separates attention into three components: content-based, time-based, and channel-based attention, enabling flexible processing of data without relying on fixed channel positions or global positional encodings. The authors apply MVPA to develop MVPFormer, a foundation model for human electrophysiology, which is trained on the Long-term iEEG dataset, the largest publicly available iEEG corpus. MVPFormer achieves superior generalization across subjects and outperforms state-of-the-art (SOTA) models in several clinical tasks such as seizure detection, while also excelling in general time-series forecasting and classification tasks.

### Strengths
1. The introduction of the Multi-Variate Parallel Attention (MVPA) mechanism is innovative and addresses the challenge of heterogeneous channel configurations in multi-variate time-series data. The way MVPA separates content, temporal, and spatial attention is novel and can be generalized to other time-series domains beyond iEEG.

2. The MVPFormer model, powered by MVPA, shows impressive performance in several iEEG-related tasks, including seizure detection, outperforming existing models. The model demonstrates expert-level performance on the Long-term iEEG dataset and outperforms SOTA methods across various clinical benchmarks.

3. The paper releases the Long-term iEEG dataset, the largest publicly available iEEG dataset to date, containing nearly 10,000 hours of recordings. This is a significant contribution to the research community, addressing the issue of data scarcity in iEEG research.

4. The commitment to open-source the dataset, code, and weights is a major advantage for reproducibility and allows other researchers to build upon this work.

5. The use of MVPA and MVPFormer is not limited to iEEG but is shown to generalize well to classical time-series forecasting and classification tasks, offering a broader impact in the field of time-series modeling.

### Weaknesses
1. While the results on seizure detection are impressive, the paper lacks a detailed real-world scenario evaluation, particularly regarding how the model would perform in a clinical setting with real-time data or noisy recordings. The authors should consider adding practical deployment considerations and edge-case performance, such as handling low-quality signals or data interruptions.

2. MVPA introduces a level of computational complexity, especially in the time- and channel-based terms. While the paper provides solutions to mitigate this, it would benefit from more in-depth comparisons with simpler models that might achieve similar performance with less computational overhead.

3. Although MVPFormer performs well in seizure detection and on Brain TreeBank tasks, the paper could benefit from broader evaluation across more clinical tasks (e.g., epilepsy classification or other cognitive tasks), especially those commonly encountered in clinical settings.

4.While the Long-term iEEG dataset is a valuable contribution, it has limitations, such as lack of electrode location information, which may limit its utility in some clinical contexts. The paper mentions this but does not fully address how future versions of the dataset might overcome this limitation.

5. While MVPFormer outperforms existing methods like Brant-2, BrainBERT, and others, a more detailed analysis of how these models compare in terms of generalization across different subjects and datasets would strengthen the argument for MVPA’s superiority. Specific examples of failure modes in the comparison would be helpful.

### Questions
1. How does MVPA perform in scenarios with significantly different time series compared to iEEG, particularly in domains like financial time series or sensor data? Could the model’s flexibility be leveraged for other domains?

2. While the results show strong generalization across subjects, could the model handle extremely varied electrode setups (e.g., patients with unusual electrode configurations)? How does MVPA cope with potential signal distortions caused by non-standard setups?

3. With the Long-term iEEG dataset being very large (10,000 hours), what are the limitations in terms of processing and inference time when scaling to even larger datasets, especially in real-time clinical settings?

4. Could MVPFormer's real-world clinical application be affected by variability in electrode placement (e.g., anatomical differences across patients)? Have you tested MVPFormer on data from patients who have had non-standard electrode placements due to medical conditions?

5.While MVPFormer is open-source, how are you ensuring patient privacy and safety when providing access to the dataset and model weights? Are there any restrictions on data sharing due to privacy concerns that might hinder the broader clinical adoption of this model?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a new method for learning representations of multi-channel intracranial activity. The input is one embedding vector per segment of activity, augmented with position vectors for time and for channel. Gains in efficiency can be had by reusing attention computations for channels that share the same time. During pretraining, a discriminative loss is used. Downstream evaluation is done on a seizure detection task and audio-linguistic tasks. Additionally, the paper releases 10K hours of iEEG recordings.

### Strengths
●	Overall, this paper represents a very strong contribution to the community.
●	Releases a large amount of data. I am personally not aware of a larger publicly available iEEG dataset. This is a big boon to the community. Especially so, since many other foundation models for intracranial signal train on private data, e.g., BrainWave.
●	Evaluation is thorough: the authors use their own seizure detection task as well as the Brain Treebank tasks.
●	Performance on the epilepsy detection task exceeds the current state of the art

### Weaknesses
●	Am I misunderstanding something? The paper refers to a "generative" objective, but the loss seems to be discriminative, i.e., an InfoNCE loss? The output of the model is in the embedding space, not neural activity, correct?
●	Line 364: The claim is that the choice of objective is justified by an ablation. But if I read appendix G.14 correctly, it seems that there is only justification for doing pretraining, not the specific type of pretraining, i.e., some other choice of pre-training objective. This is not a major weakness, but more precise wording is probably needed.
●	In the related works section, it would be good to discuss various factored approaches that are proposed for spatio-temporal data. For example, for processing movies: https://arxiv.org/pdf/2106.05968.

### Questions
●	For comparison, you can cite BrainWave: A Brain Signal Foundation Model for Clinical Applications https://arxiv.org/pdf/2402.10251. They have 35K hours of data. But most of it is EEG and most of it is private.
●	Figure 2: In the legend, what is "two-step"? It looks like this is defined in Fig 7 of the appendix, but it would be good to have that in the main text somewhere.
●	Line 329: "We must also consider that our evaluation setup involves many more subjects and ictal events than are reported for human experts," — does this mean that the expert annotations contain false negatives? I.e., periods of activity that are not marked as seizures, but should be?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Multi-Variate Parallel Attention (MVPA), an attention mechanism that splits attention into content, temporal, and channel components to better handle heterogeneous multi-channel time series. Building on MVPA, the authors train MVPFormer, a generative pre-trained foundation model for iEEG on a newly released “Long-term iEEG” corpus of about 10k recording hours. MVPFormer generalizes zero-shot across patients for seizure detection and performs competitively on four Brain TreeBank decoding tasks, while MVPA also matches or exceeds strong baselines on standard time-series forecasting and classification benchmarks. 

Main contributions include:
1 MVPA: a self-attention variant that separately models content, time, and channel structure to support generalization under variable, heterogeneous channels.

2 MVPFormer: a generative, MVPA-powered foundation model for electrophysiology that outperforms vanilla-attention baselines for seizure detection and improves over a matched discriminative model.

3 Long-term iEEG dataset: release of a large public corpus, roughly 9,300 to 10,000 hours across 68 subjects, enabling open data, code, and weights for the community.

### Strengths
1 MVPA factorizes self-attention into content, time, and channel to handle heterogeneous, variable-channel signals that standard attention struggles with. MVPFormer’s generative pretraining and the Long-term iEEG corpus add fresh angles on both model and data.

2 The method is solid  and scalable, with a coherent pretraining recipe followed by light adaptation.

3. The three MVPA components and their roles in the logits are well-explained with figures. Pretraining and fine-tuning protocols are modular, and the dataset description is specific enough to judge external validity.

4  Addressing variable, heterogeneous channels is a core deployment blocker, and MVPA tackles it directly. The approach likely transfers beyond iEEG, while the large open corpus and a usable pretrained model can accelerate community progress.

### Weaknesses
1 Zero-shot tests use manual channel selection at inference, and preprocessing, post-processing, and thresholds are not harmonized across baselines. The reported gains may stem from pipeline differences rather than the core method.


2 Overstated “expert-level” claim: A single Kappa threshold from prior work is used instead of a same-dataset, same-protocol human comparison. No per-subject confidence intervals or significance tests are reported.

3 The MVPA decomposition lacks a rigorous derivation and error bounds. How summed logits are scaled or weighted is unclear, and notation/dimensions are inconsistent in places.

4: The stated complexity does not square with a local-window content term, and there are no reproducible runtime or memory curves versus sequence length, channel count, or window size. Missing direct comparisons with other efficient attention variants.

5: Pretraining choices (negative sampling strategy, temperature, hard negatives, sample count) are not systematically ablated. Cross-dataset protocols differ, and strong baselines are not adapted for channel heterogeneity, weakening SOTA claims.

### Questions
1 Can you provide a fully automatic end-to-end inference pipeline with no manual channel selection, and report side-by-side results on the same test set? Use identical preprocessing, post-processing, and thresholding across all models, select thresholds on a shared validation split, and include per-subject distributions with 95% CIs plus clear leakage controls (window overlap, session boundaries). 

2 Can you give a rigorous derivation from dual-encoding attention to the sum of content, time, and channel terms, stating the conditions under which cross terms are dropped and providing error bounds? Please clarify how the summed logits are scaled or weighted (and whether weights are learnable). Show ablations that remove each term, swap relative for absolute encodings, and vary channel count; add a synthetic study to demonstrate when each component is required.

3 Can you release reproducible runtime and memory curves versus sequence length T, channel count C, and local window L, reflecting the expected O(T·C·L) behavior for the content term? Provide controlled throughput comparisons on the same hardware and batch size against strong efficient-attention baselines (axial or factorized attention, linear-time variants, FlashAttention with GQA), with scripts and fixed seeds.

4 Can you supply a same-dataset, same-protocol head-to-head with human experts, including inter-rater agreement, event-level error breakdowns, and statistical tests? Please adapt strong baselines for channel heterogeneity, enforce a unified evaluation protocol across datasets, and add few-shot curves to separate pretraining benefits from protocol artifacts. If the method still leads under these stricter comparisons, the “expert-level” statement becomes defensible.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Multi-variate Parallel Attention (MVPA), which disentangles content, temporal, and spatial attention to handle heterogeneous time-series with varying channel configurations, and applies it to build MVPFormer, an iEEG foundation model achieving SOTA results on seizure detection and decoding tasks.

### Strengths
- The paper is well-written with a clear motivation for the problem and an architectural approach.

- The strong architecture design enables practical foundation models for clinical iEEG

- MVPA's decomposition of attention into content, temporal, and spatial components is novel and specifically addresses the real-world challenge of heterogeneous channel configurations in clinical data.

- The model demonstrates superior performance on seizure detection across three datasets and competitive results on Brain TreeBank decoding tasks and standard time-series benchmarks.

### Weaknesses
- The model is trained only on iEEG data, whereas foundation models typically leverage diverse datasets across multiple domains and modalities.
- The model is fine-tuned on target tasks, so calling evaluation on "unseen subjects" zero-shot is inaccurate. Normally, a true zero-shot would require no task-specific training data. Should be called fewshots? 
- Section 5.3 abruptly shifts to generic time-series forecasting and classification tasks, creating a disjointed narrative that dilutes the paper's clinical focus.
- The paper is missing an ablation study of the proposed attention and the tree components in the main body.

### Questions
What specific criteria define a foundation model in your view, and how does MVPFormer satisfy these conditions, given that it's trained on a single modality (iEEG) from one domain and requires fine-tuning for downstream tasks rather than demonstrating broad zero-shot generalization?

### Soundness
2

### Presentation
3

### Contribution
2
