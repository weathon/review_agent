# Tokenizing Single-Channel EEG with Time-Frequency Motif Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Foundation models are reshaping EEG analysis, yet an important problem of EEG tokenization remains a challenge. 
This paper presents TFM-Tokenizer, a novel tokenization framework that learns a vocabulary of time-frequency motifs from *single-channel* EEG signals and encodes them into discrete tokens. 
We propose a dual-path architecture with time–frequency masking to capture robust motif representations, and it is model-agnostic, supporting both lightweight transformers and existing foundation models for downstream tasks. 
Our study demonstrates three key benefits:
*Accuracy:* Experiments on four diverse EEG benchmarks demonstrate consistent performance gains across both single- and multi-dataset pretraining settings, achieving up to $11\%$ improvement in Cohen’s Kappa over strong baselines.
*Generalization:* Moreover, as a plug-and-play component, it consistently boosts the performance of diverse foundation models, including BIOT and LaBraM. 
*Scalability:* By operating at the single-channel level rather than relying on the strict 10–20 EEG system, our method has the potential to be device-agnostic.
Experiments on ear-EEG sleep staging, which differs from the pretraining data in signal format, channel configuration, recording device, and task, show that our tokenizer outperforms baselines by $14\%$.
A comprehensive token analysis reveals strong class-discriminative, frequency-aware, and consistent structure, enabling improved representation quality and interpretability.
Code is available at https://github.com/Jathurshan0330/TFM-Tokenizer.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TFM-Tokenizer, a model-agnostic tokenization framework that learns discrete time–frequency motifs from single-channel EEG signals. The tokenizer produces interpretable and compact tokens via dual-path time–frequency masking, enabling integration with various EEG foundation models such as BIOT and LaBraM. Extensive experiments on four EEG datasets show consistent improvements in both single- and multi-dataset pretraining settings, as well as strong scalability to ear-EEG data. The authors also provide comprehensive analyses of token quality, distinctiveness, and interpretability.

### Strengths
1. The proposed TFM-Tokenizer is a model-agnostic and reusable component that can enhance a wide range of EEG foundation models.
2. The authors conduct detailed analyses of token quality (e.g., class-specificity, frequency awareness, consistency, and utilization), lending strong support to the claim that the tokens are both informative and interpretable.
3. The paper is technically sound and presents a well-motivated formulation of single-channel EEG tokenization, which addresses an underexplored yet important problem in EEG representation learning.

### Weaknesses
1. In Section 4.3, the authors only test replacing the neural tokenizer in LaBraM with TFM-Tokenizer. It would strengthen the claim of generalizability if the authors also tested using TFM-Tokenizer's token embeddings as direct inputs for masked EEG modeling.
2. The token utilization score decreases with larger vocabulary size (Appendix C.8). Could the authors explore some ways to improve utilization?
3. The embedding dimension of tokens is fixed in experiments. A discussion or ablation on how this dimension affects performance would improve clarity.
4. The paper could better articulate computational costs — for example, how much training overhead or inference latency is introduced by TFM-Tokenizer compared to standard segment-based tokenization.
5. There is a typo "Abnoral" in Table 4.

### Questions
How sensitive is the model to the choice of STFT parameters (e.g., window length, hop size)? Is the tokenizer robust to different preprocessing pipelines?

### Soundness
3

### Presentation
3

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
This submission proposes TFM-Tokenizer, a single-channel EEG tokenization framework that learns a discrete vocabulary of time-frequency motifs via a dual-path encoder. The tokenizer is used to produce per-channel token sequences which are then fed to a lightweight transformer (or plugged into existing foundation models) for  downstream tasks. Experiments over four dataset report performance gains; an ear-EEG sleep-staging dataset is used to argue scalability.

### Strengths
1/ This paper is related to a very timely topic. EEG tokenization for heterogeneous devices and non-stationary signals is an important and current problem.

2/ This paper provides rich token analyses with motif visualizations. It contains multiple analyses (such as classtoken uniqueness and class-wise token consistency) and visual examples of learned motifs in section 4.7, which enhance the insight.

### Weaknesses
1/ The novelty may be overclaimed, especially the third, tokenization learning objective, arguing that “Relying solely on capturing time-based motifs into discrete tokens risks losing important spectral structure” in L81-82. However, most existing EEG tokenization methods are already frequency or time-frequency-oriented: e.g., LaBraM reconstructs the frequency domain (with a TimeConv module on raw signals), and NeuroLM includes both time and frequency domain reconstruction. The proposed change mainly shifts from FFT to STFT, which feels incremental and weakens the claim of being “the first to encode diverse time-frequency motifs” in L142-143.

2/ The single-channel design fully discarding inter-channel topology, which seems questionable. Many EEG tasks, especially localization or differential montages in epilepsy or sleep staging, depend critically on spatial relationships and cross-channel synchrony. The relatively lower performance on CHB-MIT, compared to BIOT, may partly reflect this limitation. Moreover, such a setup implicitly assumes that the downstream backbone needs reintroduce spatial structure (as EEGPT or LaBraM do with hard-coded topographic embeddings), so the claim of being model-agnostic is also overstated.

3/ The baselines are outdated and inconsistent. Recent models such as CBraMod (ICLR’25)[1], EEG2Rep(KDD’24)[2] are not compared, and NeuroLM is included for only two of four datasets. This selective evaluation raises fairness concerns and weakens empirical credibility.

4/ This paper shows partial data leakage in its experimental setup, which weakens the claim of dataset generalization. Both the single- and multi-dataset experiments pretrain and evaluate on the same set of datasets. This design still leads to partial data leakage, as the tokenizer indirectly sees the target data distribution.

5/The writing could be more structured. The term motifs, central to the paper and appearing in the title, is not defined until the Related Work section. As this term is uncommon in EEG representation learning, it should be briefly introduced earlier to avoid confusion.

[1] Wang J, Zhao S, Luo Z, et al. CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding[C]//The Thirteenth International Conference on Learning Representations.

[2] Mohammadi Foumani N, Mackellar G, Ghane S, et al. Eeg2rep: enhancing self-supervised eeg representation through informative masked inputs[C]//Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024: 5544-5555.

### Questions
1/ Please include more recent baselines such as CBraMod (ICLR’25) and evaluate the proposed single-channel design on unseen EEG datasets or channel configurations that were not used during pretraining. This would help verify the claimed channel-invariant generalization and rule out potential data leakage.

2/ Could you clarify your fine-tuning strategy?

3/ This submission fixes the patch size at 1 s with 0.5 s overlap, but the reason for this choice is unclear. How does this fixed window align with the paper’s motivation of token resolution?

4/ The paper claims interpretability and perform motif case studies. Could you offer a quantitative measure of token-motif correspondence, for instance, the proportion of tokens aligning with known EEG events?

5/ As the model is based on VQ-VAE, which can suffer from instability or codebook collapse, please provide evidence that training remains stable, e.g., by reporting gradient norms, especially given the small datasets and large codebook size (8192).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes TFM-Tokenizer, a framework for tokenizing single-channel EEG signals by learning a vocabulary of time-frequency motifs using a dual-path architecture with frequency and temporal masking. The tokenizer is designed to be model-agnostic, integrating with existing EEG foundation models like BIOT and LaBraM, and claims improvements in accuracy (up to 17% in Cohen’s Kappa), generalization, and scalability (e.g., to ear-EEG). Experiments are conducted on four EEG datasets under single- and multi-dataset pretraining settings, with additional analysis on token quality.

### Strengths
The paper addresses an interesting problem in EEG foundation models: effective tokenization of signals to improve representation learning, which is underexplored compared to NLP or vision domains. The single-channel approach is a reasonable design choice for device-agnostic scalability, and the integration with existing models (e.g., BIOT and LaBraM) as a plug-and-play component shows practical potential. The experiments include a range of datasets (TUEV, TUAB, CHB-MIT, IIIC-Seizure) and settings (single- vs. multi-dataset pretraining), with some ablation studies on token quality (e.g., class-discriminative analysis). The inclusion of a scalability test on ear-EEG is a nice touch for real-world applicability. Overall, the work is clearly motivated by challenges like motif capturing and frequency entanglement in EEG signals.

### Weaknesses
The core contribution lacks sufficient novelty: the proposed TFM-Tokenizer heavily builds on existing VQ-based tokenization (e.g., from LaBraM) and time-frequency representations common in EEG analysis (e.g., spectrograms with masking, as in BIOT or related works like Yang et al., 2024). The "time-frequency motif learning" is essentially a combination of spectral patching, transformers, and VQ quantization, but it doesn't introduce fundamentally new mechanisms—e.g., the localized spectral window encoder is similar to patch-based processing in vision transformers, and the masking strategy mirrors BERT-like objectives without EEG-specific innovations. Claims of up to 17% improvement in Cohen’s Kappa are overstated, as they are relative to baselines on specific datasets (e.g., TUEV), but absolute gains are modest (e.g., 0.5273 to 0.6189), and statistical significance is only reported sporadically (e.g., p=1.5e-4 on IIIC-Seizure). Experiments are limited: no comparisons to more recent EEG models (e.g., BRANT or MMM beyond superficial mentions), insufficient ablation on key hyperparameters (e.g., codebook size K, masking ratios), and the multi-dataset setting uses only four datasets, which may not capture broader diversity in EEG corpora. Scalability to ear-EEG is promising but under-evaluated—only a 14% gain is claimed without details on transfer learning adaptations or failure cases. Interpretability analysis (e.g., token consistency) is superficial and lacks quantitative metrics like mutual information or visualization of failure modes.

### Questions
1. Could the authors provide more ablation studies on the vocabulary size K and masking strategies (e.g., frequency band size δf)? How sensitive is performance to these, and do they generalize across datasets?

2. The paper claims the tokenizer is "model-agnostic," but integration details with BIOT/LaBraM are brief—e.g., how exactly are token embeddings fused, and what modifications were needed? A response with pseudocode or specifics could clarify.

3. For the ear-EEG scalability experiment, what adaptations (if any) were made for differences in sampling rate or noise profiles? Baseline comparisons here seem weak; adding results from non-tokenized transfers could strengthen the claim.

I will consider raising my score if all my concerns are solved or clarified.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces TFM-Tokenizer, a novel framework that learns discrete tokens from single-channel EEG signals by capturing time-frequency motifs. Unlike existing methods that use raw or continuous embeddings, TFM-Tokenizer builds a learnable vocabulary of meaningful EEG patterns, enabling plug-and-play integration with any foundation model. It uses a dual-path encoder to jointly model time and frequency domains and is trained with a mask-and-reconstruct strategy. Evaluated on four EEG datasets and an ear-EEG sleep staging task, TFM-Tokenizer consistently outperforms strong baselines, improves existing models like BIOT and LaBraM, and offers cross-device generalization with fewer parameters and better interpretability.

### Strengths
1. First to introduce single-channel EEG tokenization using time-frequency motifs, filling a critical gap.

2. Outperforms SOTA by up to 17% with fewer parameters.

3.  Plug-and-play enhancement for existing foundation models like BIOT and LaBraM.

4.  Cross-device generalization (e.g., ear-EEG) demonstrates robust transferability.

5. Learned tokens show clear class discriminability and frequency awareness, aiding clinical understanding.

### Weaknesses
Downstream tasks focus on seizure and hospital datasets. The diversity of task types is relatively weak.

### Questions
How long does it take to train the tokenizer?

### Soundness
4

### Presentation
4

### Contribution
3
