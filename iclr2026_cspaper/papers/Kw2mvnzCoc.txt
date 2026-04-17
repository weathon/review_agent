# Tspulse: Tiny Pre-Trained Models With Disen- Tangled Representations For Rapid Time-Series Analysis

Vijay Ekambaram, Subodh Kumar*, Arindam Jati***, Sumanta Mukherjee**†,
Tomoya Sakai†, **Pankaj Dayama, Wesley M. Gifford, Jayant Kalagnanam**
IBM Research vijaye12@in.ibm.com

## Abstract

Time-series tasks often benefit from signals expressed across multiple representation spaces (e.g., time vs. frequency) and at varying abstraction levels (e.g., local patterns vs. global semantics). However, existing pre-trained time-series models entangle these heterogeneous signals into a single large embedding, limiting transferability and direct zero-shot usability. To address this, we propose TSPulse, family of ultra-light pre-trained models (1M parameters) with disentanglement properties, specialized for various time-series diagnostic tasks. TSPulse introduces a novel pre-training framework that augments masked reconstruction with explicit disentanglement across spaces and abstractions, learning three complementary embedding views (temporal, spectral, and semantic) to effectively enable zero-shot transfer. In-addition, we introduce various lightweight post-hoc fusers that selectively attend and fuse these disentangled views based on task type, enabling simple but effective task specializations. To further improve robustness and mitigate mask-induced bias prevalent in existing approaches, we propose a simple yet effective hybrid masking strategy that enhances missing diversity during pre-training. Despite its compact size, TSPulse achieves strong and consistent gains across four TS diagnostic tasks:
+20% on the TSB-AD anomaly detection leaderboard, +25% on similarity search, +50% on imputation, and +5–16% on multivariate classification, outperforming models that are 10–100× larger on over 75 datasets. TSPulse delivers state-of-theart zero-shot performance, efficient fine-tuning, and supports GPU-free deployment.

Models and source code are publicly available at https://huggingface. co/ibm-granite/granite-timeseries-tspulse-r1.

## 1 Introduction

Time-series (TS) analysis encompasses a broad class of problems that aim to extract meaningful insights and semantics from observed sequences. Among these, **time-series diagnostic tasks**—such as anomaly detection, imputation, classification, and similarity search—operate on observed data and focus on retrospective understanding, i.e., analyzing existing sequences to characterize behavior, identify irregularities, recover missing information, or compare patterns across time. These tasks are central to many real-world applications in observability, manufacturing, and industrial monitoring.

Inspired by the success of large language models (LLMs), time-series pre-trained models aim to learn reusable representations from large-scale public data for effective transfer learning. While time-series pre-trained models have seen rapid progress in forecasting (Ansari et al., 2024; Das et al., 2023; Ekambaram et al., 2024), their development for time-series diagnostic tasks remains relatively limited. A few pre-trained models—such as Moment (Goswami et al., 2024), UniTS (Gao et al., 2024), VQShape (Wen et al., 2024), and GPT4TS (Zhou et al., 2023)—support subsets of diagnostic tasks. However, their performance on time-series diagnostic tasks still leaves substantial 1 room for improvement, while their large model sizes further hinder real-time, low-latency deployment—especially in lightweight and CPU-only settings. Specifically, a key limitation underlying these approaches lies in how representations are learned during pre-training. Most existing diagnostic models rely on self-supervised objectives, with masked reconstruction emerging as one of the most widely adopted strategies (Goswami et al., 2024; Gao et al., 2024). While effective in capturing local & global structure, masked reconstruction alone is insufficient to model the full complexity of time-series data. Meaningful insights in time-series often arises, when signals are examined across different representation spaces(e.g. time vs spectral) and abstraction levels (e.g. local patterns vs global semantics). For example, abrupt spikes and local irregularities are most apparent in the time domain, whereas periodic patterns emerge more clearly in the frequency domain. Likewise, some structures are visible only at fine temporal resolutions, while others manifest at higher semantic levels. To be broadly useful, pre-trained embeddings must therefore capture these complementary cues across both spaces and abstraction levels. However, simply learning them jointly within a single embedding often leads to entanglement, making it difficult for downstream tasks to selectively access the information they require. For broad utility—particularly in zero-shot transfer—representations must explicitly expose these insights in a disentangled form, enabling temporal, spectral, and semantic signals to be accessed as needed.

To address these challenges, TSPulse proposes a

![1_image_0.png](1_image_0.png) novel pretraining framework to enable disentangled masked reconstruction across multiple spaces and abstraction levels, explicitly producing three distinct types of embeddings during pre-training: (i) detailed temporal embeddings for fine-grained time analysis, (ii) detailed spectral embeddings for frequency-aware fidelity, and (iii) semantic embeddings for high-level task understanding. The model formulates semantic and full reconstruction objectives across multiple spaces, employing multi-output heads that operate on distinct segments of the embedding to yield disentangled representations across spaces and abstraction levels (Figure 1). While prior works in traditional time-series modelling have explored time–frequency fusion (Zhang et al., 2022; 2023) or investigated disentanglement in isolation (Chang et al., 2024), TSPulse advances beyond these approaches by jointly learning disentangled representations across spaces and abstraction levels within a unified pre-training framework. Optimized together, this design yields substantial gains in both performance and transferability across diverse downstream tasks. Through extensive sensitivity analyses, we demonstrate that different segments of the learned embeddings indeed capture distinct and complementary properties that substantially enhance transfer learning. In particular, the semantic embedding exhibits strong robustness to various distortions—such as time shifts, magnitude variations, and noise—which is especially important for reliable semantic analysis. Moreover, since different tasks benefit from different combinations of these disentangled views, we introduce a set of lightweight post-hoc fusers that selectively combine these views based on the task type, providing a simple yet effective mechanism to exploit their complementary strengths for effective task specialization. Specifically, we propose two post-hoc fusers: (i) Multi-Head Triangulation (MHT) for anomaly detection and (ii) TSLens for classification, each demonstrating strong effectiveness for its respective task. In addition, TSPulse improves pre-training robustness through a simple but impactful refinement of masking strategies. Unlike existing approaches (Goswami et al., 2024) that rely on fixed masking types and span lengths, TSPulse adopts a hybrid masking scheme that randomizes both, better Figure 1: **TSPulse Overview.** % and X represent the accuracy and size improvements of TSPulse over SOTA pre-trained models across different benchmarks.

![2_image_0.png](2_image_0.png)

reflecting real-world missing patterns. Despite its simplicity, this increased corruption diversity reduces overfitting and consistently boosts downstream performance. Finally, to ensure efficiency, TSPulse replaces conventional Transformer backbones (Vaswani et al., 2017) with light-weight TSMixers (Ekambaram et al., 2023), further enhanced with improved initialization strategies. This substitution, when combined with disentangled hybrid reconstruction, yields small and fast pre-trained models, while still delivering state-of-the-art representational power. Contributions: (1) **Compact & Versatile.** We introduce a family of ultra-light time-series pre-trained models (1M parameters) with rapid zero-shot and fast multivariate fine-tuning support, specialized for 4 diagnostic tasks: classification, imputation, anomaly detection (AD), and semantic search. (2) **Architectural Novelties.** TSPulse introduces (i) disentangled masked reconstruction across multiple representation spaces and abstraction levels, yielding temporal, spectral, and semantic embeddings; (ii) simple yet effective post-hoc fusers (MHT, TSLens) built on these disentangled views for task-specialization; and (iii) a hybrid masking scheme to mitigate pre-training bias prevalent in existing approaches. (3) **Emergent Representation Properties.** Through extensive sensitivity analyses, we show that the learned disentangled embeddings capture complementary properties and exhibit varied level of robustness to common perturbations, including time shifts, magnitude variations, missingness, spectral perturbations, and noise. (4) **Benchmark Performance.** TSPulse achieves (i) robust anomaly detection with +20% gains on the TSB-AD leaderboard, ranking first in both uni- and multivariate settings; (ii) semantic search improvements of +25% via robust semantic embeddings; (iii) +50% gains in zero-shot imputation under diverse missing patterns; and (iv) +5–16% improvements in multivariate classification on UEA benchmarks. (5) **GPU-Free Deployment.** Despite its compact size, TSPulse consistently matches or outperforms models that are 10–100× larger across a broad range of benchmark datasets, while providing near-instant, CPU-only inference suitable for real-time applications.

## 2 Tspulse Architecture

Let X ∈ R
S×C be a multivariate time series with length S and C channels. We first project and mask X in both the time and frequency domains. The backbone and decoder then process these representations, mixing information across dimensions in both spaces. To guide learning, we use multi-output heads for semantic and full reconstruction on different parts of the embeddings, which encourages disentangled representation learning. These embeddings can then be directly used across downstream tasks. Figure 2 gives an overview of the framework. Masking & RevIN [Fig 2-⃝1 ]: TSPulse begins with a *masking block* that hides portions of the input sequence to enable self-supervised reconstruction. Given an input X ∈ R
S×C , we divide it into N non-overlapping patches of length pl and apply masking to obtain Xˆ ∈ R
S×C . TSPulse supports two masking strategies: **block masking** and **hybrid masking**. In block masking, entire patches are randomly replaced with a learnable mask token M ∈ R
1×pl, as commonly done in prior work (Goswami et al., 2024; Gao et al., 2024). While effective for robust feature learning, this approach is inadequate for real-world imputation tasks, where missing values occur irregularly at both patch and point levels. To address this, we introduce a more realistic **hybrid masking** pre-train strategy that masks both full and partial patches within each sample using variable masking ratios, preventing overfitting to fixed patterns. A key design choice in TSPulse is to define the mask token M ∈ R
1×pl at the *raw patch level*, unlike prior approaches that insert mask tokens in the embedding space (Goswami et al., 2024; Vaswani et al., 2017). This enables individual time-points to be masked by selecting the appropriate value from M based on their relative index within a patch (Figure 3(a)), allowing a single token to flexibly support both full and partial masking. After masking, a learnable RevIN block (Kim et al., 2022) is applied to normalize the input, yielding Xm ∈ R
S×C .

FFT Extraction [Fig 2-⃝1 ]: As TSPulse reconstructs in both time and frequency domains, this block extracts masked FFT features for backbone processing and also prepares the corresponding ground-truth for loss computation. Instead of explicitly masking the frequency space, we feed the scaled and masked time-series Xm directly into the Fast Fourier Transform (rfft), propagating the mask and ensuring the same data is consistently hidden in both spaces, preventing leaks. The real and imaginary FFT outputs from Xm are then packaged, scaled, and processed into tensor Xfm ∈ R
S×C
for further backbone processing. Refer to Appendix. A.10 for more details. Simultaneously, two ground-truths are computed from the frequency representation. First, the unmasked scaled time-series is transformed using the same approach described above to get Xf ∈ R
S×C , a clean, unaltered frequency representation of the time-series, which is used to guide the model's reconstruction. In addition, we also compute the log-magnitude spectrum of the unmasked time-series and apply softmax to obtain X
f sign 
∈ R
S/2×C , a normalized global frequency signature (Appendix. A.10 for more details). This global signature serves as an auxiliary reconstruction target, helping the model capture high-level semantic patterns and improving generalization to downstream tasks. The log transformation reduces the dynamic range and stabilizes training, while softmax emphasizes dominant spectral components, mapping the output to a probability-like distribution. Encoding [Fig 2-⃝2 ]: This block projects the input to an embedding space. The masked timedomain input Xm ∈ R
S×C is transposed and then divided into N non-overlapping patches of length pl, resulting in a tensor of shape R
C×N×pl. Now, each patch is projected via a linear layer from R 
pl → R
D to obtain time-encoded features TimeE ∈ R
C×N×D. Similarly, the masked frequencydomain input Xfm ∈ R
S×C is transposed, patched and projected to produce frequency-encoded features FFTE ∈ R
C×N×D. Motivated by recent advances in vision transformers (Darcet et al.,
2024), where adding learnable register tokens stabilizes training and improves transfer learning, we introduce R such tokens shared across channels: RegE ∈ R
C×R×D. The full input to the backbone is constructed by concatenating time, frequency, and register tokens along the patch axis and layer normalized: InputE = [TimeE; FFTE; RegE
] ∈ R
C×(2N+R)×D = R
C×K×D.

TSPulse Backbone [Fig 2 - ⃝3 ,⃝4 ]: The TSPulse backbone receives InputE ∈ R
C×K×D, a unified sequence of masked patches from both time and frequency domains, along with learnable register tokens. Its goal is to transform this input into semantically rich, task-robust representations. To maintain efficiency, we use the *TSMixer* backbone (Ekambaram et al., 2023), an MLP-Mixer based alternative to Transformers that performs strongly with reduced compute. TSMixer has stacked Mixer blocks interleaved with lightweight gated attention, enabling flexible feature mixing across three dimensions: within-patch, across-patch, and across channels. Since InputEalready integrates both time and frequency information, TSMixer effectively fuses these views, learning *dual-space* representations that capture temporal and spectral correlations. Gated attentions in TSMixer further prioritizes informative regions, enhancing the model's ability to generalize across downstream tasks.

TSPulse Mini-Decoder [Fig 2-⃝4 ]: The backbone output (BackboneE ∈ R
C×K×D) is passed through a lightweight *mini-decoder*, which mirrors the backbone but is only 10–20% of its size to output DecoderE ∈ R
C×K×D, where K = 2N + R. This compact decoder adapts representations during fine-tuning, enabling fast & efficient data-specific adaptation Multi-Objective Heads [Fig 2-⃝5 ]: The decoder output DecoderE, which consists of 3 segments
[TimeE; FFTE; RegE] is disentangled by optimizing each segment with a distinct head objective:
- **Full Reconstruction Heads [Fig 2-**⃝6 ]: The first N patch embeddings (TimeE) from the decoder pass through a linear layer (Time Head) and inverse RevIN to obtain the full reconstruction Y of the input time-series. The next N embeddings (FFTE) from the decoder are projected (via the FFT Head) to reconstruct the input frequency spectrum Yf, which is further reshaped, passed through torch.fft.irfft, and inverse RevIN to yield Y′, an alternate reconstruction of the input time-series from FFT-space. Losses are computed as Mean Squared Errors (MSE):
Ltime1 = MSE(X, Y), Ltime2 = MSE(X, Y′), and Lfft = MSE(Xf, Yf), where the first two losses are computed only on the masked time-points. This disentangled losses enables TimeE and FFTE to capture fine-grained temporal and spectral insights.

- **Semantic Heads [Fig 2-**⃝6 ]: The final R register embeddings (a.k.a semantic embeddings)
from DecoderE are primarily trained through a signature head that predicts Y
f sign, a softmax distribution over the log-magnitude frequency spectrum (i.e. semantic signature). This objective, optimized with cross-entropy loss Lsign = CE(X
f sign, Y
f sign), captures global spectral semantics and forms the core of semantic reconstruction. Optionally, a lightweight next-point prediction head Ypred can also be added, trained with Lpred = MSE(Xpred, Ypred), to inject temporal cues into the semantic embeddings. Xpred denotes the ground-truth future points. Importantly, this auxiliary head is limited to only a few points and is not designed for full-fledged forecasting; its sole purpose is to enrich the semantic representation. Together, these objectives ensure that the register tokens are converted into semantic embeddings for high-level understanding.

Finally, a weighted sum of all the above losses across heads is jointly minimized during pre-training.

## 3 Tspulse Workflows 3.1 Pre-Training:

TSPulse is pre-trained on diverse ∼1B TS samples as detailed in Appendix A.8. Inspired by the success of small, task-specialized pre-trained models in the language/vision domain (Schick & Schütze, 2020; Nguyen et al., 2024; Fu et al., 2023; Ling et al., 2024)—which achieve strong performance through minimal task-specific adaptations—we extend this strategy to time-series. Specifically, we specialize the pre-training for every task through reweighting loss objectives to prioritize heads most relevant to the target task. This enables TSPulse to refine task-specific representations while maintaining its lightweight design, facilitating efficient transfer learning across any datasets for the specified downstream task. Refer Appendix A.9 for more details. Pre-training on 1B samples takes just one day with 8×A100 GPUs, thus there are no practical challenges in pre-training task-specific models.

In addition, given the heterogeneous channel counts in pre-training datasets, TSPulse is pre-trained in a univariate mode (c = 1), treating each channel independently. Cross-channel modeling is deferred to fine-tuning, where channel-mixing is selectively activated based on the target dataset (Ekambaram et al., 2024).

## 3.2 Target Data Fine-Tuning:

During fine-tuning, the pre-trained model—already strong in zero-shot settings—is further adapted to the target data by updating the decoder and task-specific heads. For multivariate inputs, we enable *channel mixing* in the decoder to capture inter-channel correlations, which are absent in the univariate pre-training setup. Our design draws inspiration from TSMixer and TTM (Ekambaram et al., 2023; 2024), where channel mixer blocks are interleaved between patch and feature mixers within each TSMixer layer. A key limitation in the original design is the *random initialization* of these mixers, which introduces untrained parameters between already pre-trained layers. This can disrupt information flow and create sharp activation shifts, leading to unstable gradient propagation, especially during the early stages of fine-tuning. To address this, we initialize channel mixers with identity weights, which enable smooth gradient flow between pre-trained weights. These layers gradually learn inter-channel dependencies without interfering with earlier knowledge, leading to a significantly more stable fine-tuning process, as confirmed by our experiments.

![5_image_0.png](5_image_0.png)

## 3.3 Downstream Tasks & Post-Hoc Fusers:

TSPulse delivers strong zero-shot imputation benefitting from hybrid masked pre-training (Fig. 3-A),
which exposes the model to diverse corruption patterns. For semantic similarity search, the register token embeddings provide invariant representations that remain resilient to time shifts, magnitude changes, and noise (Fig. 3-B). For classification and anomaly detection, TSPulse further incorporates task-specific post-hoc fusers, detailed below. Classification via TSLens: TSPulse supports multivariate classification through a lightweight fine-tuning module, **TSLens** (Fig. 3-C), which replaces standard pooling across channels with a learned mechanism that adaptively extracts relevant features from disentangled embeddings. Unlike conventional methods, which average or max-pool all patch-level embeddings across channels followed by a linear head, TSLens selectively attends to and weights features across fine-grained patch embeddings and high-level register tokens.

TSLens takes the backbone output BackboneE ∈ R
C×(2N+R)×D, passes it through the mini decoder (initialized with pre-trained weights and channel-mixing enabled), and learns cross-channel dependencies via identity-initialized channel mixers as explained in Section 3.2. The resulting representation H ∈ R
C×(2N+R)×D is projected to a lower-dimensional space H′ ∈ R
C×(2N+R)×D′,
flattened into Hflat ∈ R
C·(2N+R)·D′, and passed through a linear layer to produce class logits ypred ∈ R
num_classes. The model is optimized with cross-entropy loss. This design allows TSPulse to dynamically focus on the most informative features across local and global representations, improving classification accuracy across diverse datasets.

## Robust Anomaly Detection Via Multi-Head Triangulation:

In anomaly detection, certain anomalies manifest in the time domain (sudden spikes), others in the frequency domain (periodicity breaks), and others in predictive space (missing trends). TSPulse leverages multi-output heads—Headtime, Headfft, and Headpred—to reconstruct or predict from complementary views, capturing signal continuity, spectral consistency, and temporal dynamics. This unified design enables detection of diverse anomaly types.

During inference, anomaly scores are computed from each head based on the deviations between the original and predicted signals(Fig. 3-D). Once the deviations are obtained from all heads, two approaches are possible. **Approach 1** (Headensemble) is to fuse the normalized scores using statistics, such as the maximum, to generate a unified score. In **Approach 2** (Headtriang.), when a small labeled validation set is available, it can be used to select the most effective head in zero-shot from the above four heads (including Headensemble). This allows the model to adapt to the anomaly type and structure specific to each application. Notably, TSPulse is the first pre-trained model to unify and triangulate multi-space outputs in a single lightweight framework, enabling robust anomaly detection in both zero-shot and fine-tuned settings.

## 4 Experiments

We evaluate TSPulse across 4 TS diagnostic tasks: classification, anomaly detection, imputation, and similarity search. Details of the pre-trained model configurations are in Appendix A.9. Pre-training datasets are listed in Table 10, and they do not overlap with any of the evaluation datasets.

![6_image_0.png](6_image_0.png) 
 

## 4.1 Anomaly Detection (Ad)

Setup: We evaluate TSPulse on the TSB-AD benchmark (Liu & Paparrizos, 2024) (recent comprehensive leaderboard for AD), which comprises 40 eval datasets, covering both univariate (TSB-AD-U) and multivariate (TSB-AD-M) anomaly detection. The benchmark includes results from 40 SOTA methods and establishes VUS-PR (Paparrizos et al., 2022) as the primary and robust evaluation metric. A small labeled official *tuning-set* is provided for hyperparameter selection, consistently used across all leaderboard methods. We adopt this tuning set for multi-head triangulation to select the best-performing head and report scores on the test set for both zero-shot (TSPulse-ZS) and fine-tuned (TSPulse-FT) variants. In the zero-shot case, TSPulse is evaluated directly without training on the target data; in the fine-tuned case, it is self-supervised using the official training split without access to any anomaly labels. Note that all non-pretrained neural network models are also trained using the same training split. Figure 4 summarizes VUS-PR results. Full details in Appendix A.11.

Results: As illustrated in Figure 4, TSPulse (ZS) outperforms all existing SOTA methods on both the uni and multi-variate AD benchmarks. Specifically, TSPulse (ZS) achieves 14% and 16% higher VUS- PR scores compared to the best-performing baselines—SubPCA for the univariate setting and CNN
for the multivariate setting, respectively. Notably, TSPulse, without any training on the target data, outperforms all models trained on it, underscoring its strong transfer learning.TSPulse also outperforms all the pre-trained models by +30% by using just a fraction of their model size. The fine-tuned variant, TSPulse (FT), further improves results, achieving 24% and 26% gains over SOTA on uni and multivariate benchmarks. These results underscore the effectiveness of TSPulse for diverse AD tasks.

```
5%
    9% 16% 
            5% 5% 
                    12% 10% 10% 

```

Pre-trained models Data-specific models 1M 37 M 340 M **10 M**
Figure 5: Classification Mean Accuracy results (higher is better); IMP(%)—the percentage improvement of TSPulse over baselines.

## 4.2 Classification

Setup: We evaluate TSPulse results on 29 datasets from the UEA Multivariate Time Series Classification Archive (Bagnall et al., 2018). Dataset and hyperparameter details are provided in Appendix A.12. We compare against recent pre-trained models—VQShape (Wen et al., 2024), Moment (Goswami et al., 2024), and UniTS (Gao et al., 2024)—as well as strong data-specific baselines including T- Rep (Fraikin et al., 2024), TS2Vec (Yue et al., 2022), T-Loss (Franceschi et al., 2019), TS-TCC (Eldele et al., 2021) and TNC (Tonekaboni et al., 2021). Results Classification results fine-tuned/trained on the labeled data are reported in Figure 5. TSPulse achieves state-of-the-art accuracy, surpassing VQShape, UniTS, and Moment by 5–16%, while being drastically smaller (1M vs. 10–340M parameters). It also outperforms contrastive and supervised baselines by 5–12%, highlighting the effectiveness of TSPulse fine-tuning with TSLens.

## 4.3 Imputation

Setup: We evaluate TSPulse on 6 LTSF benchmark datasets (Wu et al., 2021): ETTh1, ETTh2, ETTm1, ETTm2, Weather, and Electricity, under 4 mask ratios (12.5%, 25%, 37.5%, 50%) using irregular hybrid masking (a mix of block and point masks) to simulate real-world missingness. Results: We first evaluate TSPulse in a fully Zero-Shot (ZS) setup, requiring no data-specific tuning. Among pre-trained baselines, MOMENT supports native zero-shot imputation, for UniTS (Gao et al., 2024) the pretrained model is prompt-tuned (PMT) with 10% data in multi-task setup. We also compare against statistical baselines in ZS setup. As shown in Figure 6, TSPulse (ZS) outperforms MOMENT by over 70% and UniTS by 50% despite its prompt-tuning. Compared to statistical interpolation methods, TSPulse shows 50%+ gains, highlighting the effectiveness of TSPulse in hybrid masking setup and robust zero-shot generalization. More details in Appendix A.13 When Fine-Tuning (FT) is desired, TSPulse can be

![7_image_0.png](7_image_0.png) extended with a channel-mixing decoder to capture inter-variable dependencies, outperforming strong supervised models like TimesNet (Wu et al., 2022), FedFormer (Zhou et al., 2022), and Non-Stationary Transformers (Non-Stat.) (Liu et al., 2022) by over 40%. Remarkably, TSPulse's zero-shot performance already exceeds many of the fine-tuned benchmarks, demonstrating its strong generalization and transferability. We further evaluated TSPulse under the full block masking strategy in both ZS and FT settings, as illustrated in Appendix Figure 13. TSPulse continues to outperform all baselines by a significant margin in this setting as well.

## 4.4 Time-Series Similarity Search

windows for efficient storage, the same pattern can appear in different positions, making distortioninvariant embeddings essential for similarity search. We use real and synthetic data for indexing, and generate query samples by applying complex augmentations (time shifts, magnitude changes, and noise distortions) from indexed samples. This setup tests the embeddings' robustness in retrieving distorted similar patterns and simplifies evaluation, as the correct matches for each query are already known. Two tasks are defined: *Family Match* for high-level pattern retrieval and *Fine-Grained Match* for precise pattern matching and evaluated using PREC@k and MRR@k (Valcarce et al., 2020). We construct a synthetic dataset and a real dataset based on the UCR dataset (Chen et al., 2015), and report the average score across both for each task. See Appendix A.14 for full details.

![7_image_1.png](7_image_1.png)

Setup: We evaluate TSPulse's similarity search using its zero-shot semantic embeddings to retrieve time-series segments with similar patterns, even under real-world distortions like time shifts, magnitude changes, and noise. Since time-series are typically indexed via high-stride sliding

![8_image_1.png](8_image_1.png)

![8_image_2.png](8_image_2.png)

![8_image_3.png](8_image_3.png)

![8_image_0.png](8_image_0.png)

![8_image_4.png](8_image_4.png)

Table 1: Ablation results across tasks. **[VUS-PR, Accuracy, MRR]**: Higher is better; **[MSE]**: Lower is better. IMP(%) indicates the percentage improvement of the bold variant over the compared variant.

Results: Figure 7 compares TSPulse's similarity search performance against zero-shot embeddings from MOMENT and Chronos. We use their smallest variants to closely match TSPulse's embedding size and enable faster indexing for a fair comparison. As shown, TSPulse outperforms MOMENT by over 25% in family-level and 40% in fine-grained match accuracy, and surpasses Chronos by 100%. Notably, TSPulse's zero-shot embeddings are 2X smaller and enable 10–100X faster CPU inference, 9–15X faster GPU inference, and come from a model that is 40× smaller than the baselines. Further discussion in Appendix A.14.

## 5 Ablation Studies

- **Anomaly Detection (AD)** We evaluate the performance of individual TSPulse heads for anomaly detection. Table 1(a) reports average VUS-PR scores for each head used independently. The proposed multi-head triangulation outperforms all single-head variants on both TSB-AD-U and TSB-AD-M, demonstrating the strength of multi-head TSPulse. See Appendix A.11 for details.

- **Classification** We evaluate the impact of key design components in Table 1(b), using a representative subset of 17 UEA datasets for faster analysis. Removing either the short or long embedding from the *disentanglement* design reduces mean accuracy by 8–10%, confirming the importance of capturing both semantic and fine-grained features. Disabling *masking* during fine-tuning leads to an 8% drop—especially on smaller datasets—highlighting its role as a regularizer. Replacing TSLens with simple pooling causes an 11–16% drop, emphasizing the value of feature-attention.

Randomly initializing the *channel-mixing (CM)* blocks instead of using identity weights leads to a 9% drop, reflecting the need for stable gradient flow. Removing *dual-space learning* (i.e., reconstructing only in time domain) lowers accuracy by 7%, and omitting *virtual channel expansion*,
critical for low-channel datasets—causes a further 2% drop. More details in Appendix 18.

- **Imputation** Table 1(c) shows that removing dual-space learning leads to an 8% drop in zeroshot accuracy. When pre-training (PT) is done with only block masking (i.e., w/o Hybrid PT), performance drops by 79% under hybrid-mask eval settings, underscoring the importance of hybrid masking in pre-training for robust, generalizable imputation, where missingness is irregular and more reflective of real-world scenarios.

- **Similarity Search** Table 1(d) shows that TSPulse and baselines perform similarly without distortion. As augmentation distortion increases, all models degrade, but TSPulse remains notably more robust to time shifts, magnitude changes, and noise—highlighting the resilience of its embeddings in retrieving distorted yet similar patterns. Also, use of hybrid masking & semantic embedding boosts search performance by over 20% (Appendix A.14)
- **Efficiency** Appendix A.2 shows TSPulse is significantly faster, smaller and CPU-friendly.

## 6 Sensitivity Analysis Of Embedding Disentanglement

To validate that TSPulse learns genuinely disentangled temporal, spectral, and semantic representations, we conduct controlled experiments on synthetic signals under three perturbation settings: missing data, additive noise, and phase/time shifts. These perturbations allow us to isolate how each embedding type responds to missing data in the input (masking), stochastic changes (noise), and temporal misalignment (phase shift). We quantify embedding stability using a distortion metric that measures how much each embedding changes under controlled perturbations (formal definitions in Appendix A.3). Representative results are summarised in Table 2. Lower values indicate greater robustness.

| Experiment          | Time (d = 1536)   | FFT (d = 1536)   | Semantic (d = 256)   |
|---------------------|-------------------|------------------|----------------------|
| 30% Missing Data    | 8.3%              | 27.4%            | 4.6%                 |
| Noise Level η = 0.5 | 2.7%              | 6.8%             | 2.5%                 |
| Phase / Time Shift  | 130%              | 21%              | 12%                  |

The results exhibit clear and expected disentanglement patterns.

Temporal embeddings are highly sensitive to phase/time shifts (130% distortion), confirming preservation of fine-grained temporal alignment. This property is critical for tasks that depend on precise timing cues.

FFT embeddings demonstrate substantially lower phase sensitivity, reflecting invariance to temporal alignment while retaining spectral characteristics. Semantic embeddings (a.k.a Register embeddings) are the most robust to missing data and noise and the least sensitive to phase shifts, consistent with their role as high-level structural abstractions rather than fine-grained signal encoders. These complementary behaviours directly translate to downstream utility. Tasks such as anomaly detection and imputation benefit from time and FFT embeddings for high-fidelity reconstruction, while retrieval tasks primarily leverage semantic embeddings for compact summarisation. Moreover, disentangled reconstruction across spaces enables triangulation-based mechanisms that detect anomaly types missed by single-view models. For more details, refer to Appendix A.3. We have also conducted a sensitivity deep-dive analysis focused on register embeddings as explained in Appendix A.4. Together, these results confirm that TSPulse achieves effective disentanglement across both representational spaces (time vs. FFT) and abstraction levels (fine-grained vs. semantic), forming a core foundation for its strong zero-shot transfer performance.

## 7 Conclusion

TSPulse sets a new benchmark for ultra-compact time-series pre-trained models, achieving stateof-the-art performance in classification, imputation, anomaly detection, and similarity search—all with under 1M parameters. Powered by innovations like disentangled masked reconstruction across spaces and abstractions, TSLens, hybrid masking, and multi-head triangulation, TSPulse enables robust zero-shot and fine-tuned performance. Despite its small size, it outperforms models 10–100X larger and runs efficiently on CPUs, making it both powerful and deployment-ready. Appendix A.17 outlines limitations and future directions, including opportunities to expand to additional downstream tasks, enable incremental learning and reduce supervision requirements. We believe this work will inspire more advanced research and innovation in the field of lightweight time-series modeling.

## Reproducibility Statement

Models and source code are publicly available at https://huggingface.co/ ibm-granite/granite-timeseries-tspulse-r1. Detailed model parameters are included in Appendix A.9. All pretraining datasets are publicly available and referenced in Appendix A.8, and all evaluation datasets are likewise publicly accessible.

## References

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. Chronos: Learning the language of time series. *arXiv preprint arXiv:2403.07815*, 2024.

Anthony Bagnall, Hoang Anh Dau, Jason Lines, Michael Flynn, James Large, Aaron Bostrom, Paul Southam, and Eamonn Keogh. The UEA multivariate time series classification archive, 2018, 2018.

URL https://arxiv.org/abs/1811.00075.

Ching Chang, Chiao-Tung Chan, Wei-Yao Wang, Wen-Chih Peng, and Tien-Fu Chen. TimeDRL:
Disentangled representation learning for multivariate time-series, 2024. URL https://arxiv. org/abs/2312.04142.

Yanping Chen, Eamonn Keogh, Bing Hu, Nurjahan Begum, Anthony Bagnall, Abdullah Mueen, and Gustavo Batista. The UCR time series classification archive, July 2015. www.cs.ucr.edu/
~eamonn/time_series_data/.

Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers, 2024. URL https://arxiv.org/abs/2309.16588.

Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. *International Conference on Machine Learning (ICML)*, 2023.

Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. The Faiss library. *arXiv*, 2024.

Vijay Ekambaram, Arindam Jati, Nam Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam.

Tsmixer: Lightweight mlp-mixer model for multivariate time series forecasting. In *Proceedings* of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD '23, pp. 459–469, New York, NY, USA, 2023. ISBN 9798400701030. doi: 10.1145/3580305.3599533.

URL https://doi.org/10.1145/3580305.3599533.

Vijay Ekambaram, Arindam Jati, Pankaj Dayama, Sumanta Mukherjee, Nam H Nguyen, Wesley M. Gifford, Chandra Reddy, and Jayant Kalagnanam. Tiny time mixers (TTMs): Fast pre-trained models for enhanced zero/few-shot forecasting of multivariate time series. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL
https://openreview.net/forum?id=3O5YCEWETq.

Emadeldeen Eldele, Mohamed Ragab, Zhenghua Chen, Min Wu, Chee Keong Kwoh, Xiaoli Li, and Cuntai Guan. Time-series representation learning via temporal and contextual contrasting. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, pp. 2352–2359, 2021.

Archibald Fraikin, Adrien Bennetot, and Stéphanie Allassonnière. T-Rep: Representation learning for time series using time-embeddings, 2024. URL https://arxiv.org/abs/2310.04486.

Jean-Yves Franceschi, Aymeric Dieuleveut, and Martin Jaggi. Unsupervised scalable representation learning for multivariate time series. *Advances in neural information processing systems*, 32, 2019.

Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. Specializing smaller language models towards multi-step reasoning. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of *Proceedings of Machine Learning Research*, pp.

10421–10430. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/
fu23d.html.

Shanghua Gao, Teddy Koker, Owen Queen, Thomas Hartvigsen, Theodoros Tsiligkaridis, and Marinka Zitnik. UniTS: A unified multi-task time series model. In *The Thirty-eighth Annual* Conference on Neural Information Processing Systems, 2024. URL https://openreview. net/forum?id=nBOdYBptWW.

Rakshitha Godahewa, Christoph Bergmeir, Geoffrey I. Webb, Rob J. Hyndman, and Pablo Montero-
Manso. Monash time series forecasting archive. In Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski.

MOMENT: A family of open time-series foundation models. *International Conference on Machine* Learning (ICML), 2024.

Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations, 2022. URL https://openreview.net/ forum?id=cGDAkQo1C0p.

Chen Ling, Xujiang Zhao, Jiaying Lu, Chengyuan Deng, Can Zheng, Junxiang Wang, Tanmoy Chowdhury, Yun Li, Hejie Cui, Xuchao Zhang, Tianjiao Zhao, Amit Panalkar, Dhagash Mehta, Stefano Pasquali, Wei Cheng, Haoyu Wang, Yanchi Liu, Zhengzhang Chen, Haifeng Chen, Chris White, Quanquan Gu, Jian Pei, Carl Yang, and Liang Zhao. Domain specialization as the key to make large language models disruptive: A comprehensive survey, 2024. URL https:
//arxiv.org/abs/2305.18703.

Qinghua Liu and John Paparrizos. The elephant in the room: Towards a reliable timeseries anomaly detection benchmark. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 108231–108261. Curran Associates, Inc., 2024. URL
https://proceedings.neurips.cc/paper_files/paper/2024/file/ c3f3c690b7a99fba16d0efd35cb83b2c-Paper-Datasets_and_Benchmarks_
Track.pdf.

Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Non-stationary transformers: Exploring the stationarity in time series forecasting. *Advances in neural information processing systems*, 35: 9881–9893, 2022.

Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long.

iTransformer: Inverted transformers are effective for time series forecasting, 2024.

Chien Van Nguyen, Xuan Shen, Ryan Aponte, Yu Xia, Samyadeep Basu, Zhengmian Hu, Jian Chen, Mihir Parmar, Sasidhar Kunapuli, Joe Barrow, Junda Wu, Ashish Singh, Yu Wang, Jiuxiang Gu, Franck Dernoncourt, Nesreen K. Ahmed, Nedim Lipka, Ruiyi Zhang, Xiang Chen, Tong Yu, Sungchul Kim, Hanieh Deilamsalehy, Namyong Park, Mike Rimer, Zhehao Zhang, Huanrui Yang, Ryan A. Rossi, and Thien Huu Nguyen. A survey of small language models, 2024. URL
https://arxiv.org/abs/2410.20011.

Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In *ICLR*, 2023. URL https://arxiv.org/ abs/2211.14730.

John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S Tsay, Aaron Elmore, and Michael J Franklin.

Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection.

Proceedings of the VLDB Endowment, 15(11):2774–2787, 2022.

PyTorch Contributors. torch.fft.rfft - pytorch documentation https://docs.pytorch.org/
docs/stable/, 2025. URL https://docs.pytorch.org/docs/stable/. Accessed:
2025-04-23.

Kashif Rasul, Arjun Ashok, Andrew Robert Williams, Arian Khorasani, George Adamopoulos, Rishika Bhagwatkar, Marin Biloš, Hena Ghonia, Nadhir Vincent Hassen, Anderson Schneider, et al. Lag-Llama: Towards foundation models for time series forecasting. arXiv preprint arXiv:2310.08278, 2023.

Donald B. Rubin. Inference and missing data. *Biometrika*, 63(3):581–592, 1976. ISSN 00063444, 14643510. URL http://www.jstor.org/stable/2335739.