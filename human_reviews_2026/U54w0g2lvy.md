# Large EEG-U-Transformer for Time-Step-Level Detection Without Pre-Training

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4, 4

## Abstract
Electroencephalography (EEG) reflects the brain's functional state, making it a crucial tool for diverse detection applications, including event-centric analysis like seizure detection and status-centric analysis like pathological detection. While deep learning-based approaches have recently shown promise for automated detection, traditional models are often constrained by limited learnable parameters and only achieve modest performance. In contrast, large foundation models showed improved capabilities by scaling up the model size, but required extensive time-consuming pre-training. Moreover, both types of existing methods focus on window-level classification, which requires redundant post-processing pipelines for event-centric tasks. In this work, based on the multi-scale nature of EEG events, we propose a simple U-shaped model to efficiently learn representations by capturing both local and global features using convolution and self-attentive modules for sequence-to-sequence modeling. Compared to other window-level classification models, our method directly outputs predictions at the time-step level, eliminating redundant overlapping inferences. Beyond sequence-to-sequence modeling, the architecture naturally extends to window-level classification by incorporating an attention-pooling layer. Such a paradigm shift and model design demonstrated promising efficiency improvement, cross-subject generalization, and state-of-the-art performance in various time-step and window-level classification tasks in the experiment. More impressively, our model showed the capability to be scaled up to the same level as existing large foundation models that have been extensively pre-trained over diverse datasets and outperforms them by solely using the downstream fine-tuning dataset.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes EEG-U-Transformer, a U-shaped architecture that combines convolutional encoders, ResCNN stacks, and transformer modules for time-step-level EEG analysis. The model aims to eliminate sliding window-based post-processing and pre-training requirements, achieving competitive results on seizure detection, sleep staging, and pathological detection tasks.

### Strengths
The sequence-to-sequence design effectively bypasses redundant window-level inference, simplifying pipelines for event-centric tasks like seizure detection.

The model scales competitively with large pre-trained foundation models while relying solely on downstream data, reducing computational costs.

### Weaknesses
The emphasis on "no pre-training" as a virtue seems misleading; large-scale pre-training learns generalizable representations that often enhance performance and robustness when fine-tuned to specific tasks. By forgoing this, the model may underutilize broad EEG patterns, limiting its adaptability to diverse domains or low-data scenarios. Please note that the data output by different EEG devices can vary significantly, and strong general knowledge is a necessary tool to bridge this gap in clinical practice.

Cross-dataset results (Table 6) reveal significant performance drops (e.g., ~34% F1-score decrease on SeizeIT1), indicating sensitivity to domain shifts in electrodes, demographics, or devices.

The post-processing pipeline (thresholding, morphological operations) remains heuristic and dataset-dependent, undermining the end-to-end promise.

Hyperparameter sensitivity (e.g., kernel sizes, transformer layers) is underexplored, raising reproducibility concerns without meticulous tuning.

Computational analysis focuses on inference time but omits training costs and memory footprint, especially for long sequences (e.g., T=15360).

Global-local interaction and fusion have been extensively explored in numerous previous works, encompassing both the general visual recognition and EEG analysis domains, e.g, [1-4]. Consequently, this cannot be considered a significant contribution of this paper. Furthermore, I did not observe any relevant discussions about these works.

Refs:   
[1] On the integration of self-attention and convolution. Arxiv '23.  
[2] TransXNet: Learning Both Global and Local Dynamics with a Dual Dynamic Token Mixer for Visual Recognition. Arxiv '23.  
[3] Multiscale global prompt transformer for EEG-based driver fatigue recognition. TASE'24.  
[4] Learning robust global-local representation from EEG for neural epilepsy detection. TAI '24.

### Questions
My questions have been listed in the weaknesses part.

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
4

### Summary
The paper introduces a new architecture for an EEG-based application. This model uses a U-Net architecture with Transformer layers. This allows to have more generalizable networks. The model is then tested on three different modalities: seizure detection, sleep staging, and abnormality detection.

### Strengths
- The authors propose a new architecture that leverages the power of U-Net with the generalization power of the transformer between the encoder and the decoder.
- This model offers versatility in the application for both Windows-based event and segmentation tasks.
- This new model is applied to three different modalities for which the model outperforms other models, except for the TUAB dataset, where the results are worse with their model.
- In addition, the model is faster to run compared to other models.

### Weaknesses
If the paper is proposing a new model that works well on several modalities, I think that the paper lacks clarity in some points:
- In Figure 1, it is not so clear that the arrows represent layers, especially when scaling embeddings, and Figure 3 represents layers by blocks.
- In the same Figure, it will be clearer to use a letter instead of a number (example: 15360 -> T and then T/2 ...)
- In Figure 2, the windowing of the signal is exactly the same between the two examples. Since the strength of the method is to be able to cut the windows as we want, it could be interesting to see the possible difference.
- The organization of the paper is hard to follow sometimes. Part 2.3 gives information on the pre-processing of the dataset, but Part 2 focuses on the architecture of the model. I would suggest moving it to part 3.1 or at least to the end of part 2.
- This claim: "Such experimental settings and
evaluation metrics do not fit with real-world requirements and often limit the model design, as
different model architectures might benefit from different sequence lengths," is not true for all tasks. In sleep staging, for example, all the datasets are annotated every 30 seconds. Adding a reference can give more strength to the claim that is central to the motivation of the paper.

Minor: 
- add number of subjects in Table 1
- For the TUAB dataset, the claim that you are in the top tier of the AUROC score and "marginally" lower than BIOT is too strong. Your method is losing 3% compared to Biot, which is the improvement that you have on sleed-EDFx.
- Several typos were seen in the paper.

In my opinion, this paper has good propositions and results, but the lack of clarity makes it hard to follow.

### Questions
- You are categorizing the datasets into no-activity, full-activity, and partial-activity. Is it something usual for one of the modalities? Did you do that on every dataset? 
- On which dataset was the ablation study done?
- In sleep staging, the models are usually using sequences of windows. For example, in the U-Sleep paper, they are using 35 windows as input. This is giving more context to the models. Is it something doable with your method? For sleep, for example, could we pass a longer time length than 30 seconds to get multiple outputs at the end?
- For the time comparison (Table 3), why are only 3 competitors given? Knowing that you run several models, it would be easy to give every running time.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the Large EEG-U-Transformer (EEG-U-T), a sequence-level model designed to perform end-to-end EEG time-step detection without pre-training. The authors argue that existing EEG foundation models (e.g., BIOT, LaBraM, EEGPT) rely on fixed window segmentation and heavy pre-training, leading to redundant processing and computational overhead. To address this, they introduce a U-shaped CNN–Transformer encoder–decoder that directly models long EEG sequences in a sequence-to-sequence manner, aiming to capture both local and global temporal dependencies. The model is evaluated on three datasets (TUSZ, TUAB, Sleep-EDF) across seizure detection, pathological EEG classification, and sleep staging tasks.

### Strengths
The paper tackles an important and timely problem: improving EEG temporal modeling beyond single-epoch representations. The paper is easy to follow with a clearly described architecture.

### Weaknesses
IMO the paper has a mispositioned motivation and runs the risk of overclaiming. It mischaracterizes the limitations of existing foundation models such as BIOT, LaBraM, and EEGPT. Those models are not designed for sequence-to-sequence detection but focus on learning single-epoch representations and channel adaptation, which are foundational rather than temporal tasks. The proposed model instead targets continuous sequence labeling—a fundamentally different objective. Moreover, the claim that existing foundation models suffer from “window segmentation limitations” is inaccurate and overstates their weaknesses. Windowing is a task-dependent design choice, not a methodological flaw.

Moreover, I am afraid the novelty is limited in that the proposed method is essentially a sequence wrapper over existing epoch-based models. The proposed EEG-U-Transformer can be viewed as stacking a CNN–Transformer encoder-decoder to model long-range temporal dependencies. Conceptually, this is a sequence-level wrapper built upon the same features that existing single-epoch models already extract effectively. It would be straightforward to use features from published foundation models (BIOT, LaBraM, EEGPT) and train a small seq-to-seq model for fine-tuning. Hence, the contribution lies more in architectural repackaging than in a novel modeling principle or learning paradigm.


The authors claim that existing foundation models require “diverse datasets and tremendous computation resources.” However, the proposed EEG-U-T introduces much larger model size (Table 5) and a heavier training process, contradicting the stated motivation of efficiency. While parameter counts increase significantly, the AUROC actually drops below BIOT on TUAB. Table 3 compares the proposed model’s runtime against task-specific baselines. However, the proposed method is presented as a foundation-style model, which in realistic usage would employ only the final-layer embeddings for inference. Thus, comparing full training runtime across architectures with different usage paradigms is not meaningful. A fair comparison would include inference-only latency and pre-training vs. fine-tuning cost under identical settings.

Regarding experiments, the setup lacks uniformity and rigor:
- Different baselines are used across datasets (TUSZ vs. TUAB vs. Sleep-EDF), making comparisons non-standardized.
- EDF is a very small dataset—training a large model on such a dataset raises questions about overfitting and necessity.
- TUAB results differ from reported metrics in prior papers, yet no explanation or setting alignment is provided.
- No parameter sensitivity, ablation, or standard deviation analysis is presented.
Overall, the empirical evidence is not strong enough to support the claimed generalization or efficiency improvements.

I should also point out that the contribution is incremental: recent research on EEG foundation models have already explored representation learning, cross-channel dynamics, and scaling. This paper does not introduce a fundamentally new modeling principle—it mainly combines known CNN–Transformer and U-Net design patterns for sequence labeling. The contribution feels incremental, and the conceptual advancement over existing frameworks remains limited.

### Questions
please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the "Large EEG-U-Transformer," a hybrid U-Net and Transformer architecture. It is designed for efficient, time-step level  event detection. The paper's central and most important claim is that this model, trained only on downstream datasets, can outperform large, pre-trained foundation models (LFMs) like EEGPT and BIOT.

### Strengths
1/ The core result—that a 6.1M parameter model trained from scratch can beat a 25M pre-trained LFM (EEGPT) on Sleep-EDFx.
Strong Empirical Performance: The model achieves convincing SOTA results on the TUSZ seizure task (beating DeepSOZ-HEM)。and the Sleep-EDFx task. And the algorithm is of high efficiency compared with prior arts.

### Weaknesses
The claim in Appendix A that combining U-Nets and Transformers is novel for biomedical signals is false. This concept is foundational in medical imaging (e.g., UNETR, Swin-Unet)  and exists in time-series (Yformer). The authors also failed to cite highly relevant prior work using attention-gated U-Nets for the exact same task (Chatzichristos et al. 2020).

The paper's scaling study (Tables 9 & 10) is a weakness, not a strength. Performance peaks at 59.9M parameters and then drops significantly at 80.9M . This contradicts the scaling laws that underpin LFMs and is poorly explained.

### Questions
See the weakness part.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper couples a U-Net style temporal convolutional backbone with a Transformer encoder to build a sequence-to-sequence model for per-time-step EEG labeling, and then uses attention pooling to convert those features into window-level predictions for tasks like seizure detection and sleep/abnormal screening. On TUSZ it reports solid event-level F1 and fast inference, but the method and evaluation largely combine standard components, so originality is limited.

The major contributions include: 
1. A unified EEG framework that performs time-step segmentation and window-level classification in one model, using U-Net temporal features, Transformer encoding, and attention pooling to bridge granular and aggregate predictions.
2 An efficient inference and post-processing pipeline that achieves competitive event-level performance on TUSZ while keeping runtime low, demonstrating practical viability for long recordings.

### Strengths
Originality: One backbone handles both per-time-step labeling and window classification via attention pooling.
Quality: Clinically aligned time-step and event-level metrics plus simple post-processing reduce false alarms.
Clarity: Figures and equations clearly explain the architecture and attention pooling.
Significance: Competitive TUSZ event-level F1 with fast inference shows practical value on long recordings.

### Weaknesses
1. Limitted novelty: U-Net–style temporal conv plus a Transformer encoder is already common in time-series/biomed (see recent EEG/ECG segmentation hybrids); the paper does not show why this variant is fundamentally better than a strong pure-Transformer or pure-UNet baseline under identical settings. 

2 I am worry that authors made several Unfair or under-specified comparisons. Several baselines use different window lengths, channels, or preprocessing, so current tables cannot isolate gains from the proposed model rather than from setup differences; a “same data, same window, same channels, same hardware” table is missing. 

3 Mathematical/formulation glitches. The time-step loss mixes indices and does not clearly sum over dataset/time; positional encoding uses a nonstandard denominator likely to be a typo, which hurts clarity and reproducibility. 

4 Event-level evaluation is too forgiving. Results rely on tolerant matching and post-processing (morphological ops, min-duration) without reporting FP/h, onset/offset error, or sensitivity to threshold, so it is unclear whether the method is robust in stricter clinical regimes. 

Additionally, figures/tables not presentation-ready. Table 1 mixes dataset stats with model configs, and several figures lack legend/abbreviation expansion, making it hard to verify the pipeline or reproduce it.

### Questions
1 You use 10000^(2i/Td) rather than the standard 10000^(2i/dmodel). Is this intentional? Please justify the choice and provide an ablation/sensitivity study versus the standard form.
2 Can you re-train and re-evaluate all baselines and your model under identical settings (same channels, window length, preprocessing, hardware, and batch size), reporting mean ± 95% CI over multiple seeds?
3 Current results rely on tolerant matching and post-processing. Please report FP/h, onset/offset error distributions, and FROC, include tolerance/threshold sweeps, and ablate the morphological filtering and minimum-duration pruning to quantify their contribution.
4 could you provide Leakage control with overlapping windows? It would be fair if authors could precisely document train/val/test split policy (file/patient level) and show that overlapping windows do not cross splits. Provide a sensitivity analysis to different overlap ratios to rule out temporal leakage.

### Soundness
3

### Presentation
2

### Contribution
2
