# Learnable Fractional Superlets with a Spectro-Temporal Emotion Encoder for Speech Emotion Recognition

- Decision: Accept (Poster)
- Scores: 2, 6, 4

## Abstract
Speech emotion recognition (SER) hinges on front-ends that expose informative time-frequency (TF) structure from raw speech. Classical short-time Fourier and wavelet transforms impose fixed resolution trade-offs, while prior ”superlet” variants rely on integer orders and hand-tuned hyperparameters. We revisit TF analysis from first principles and formulate a learnable continuum of superlet transforms. Starting from DC-corrected analytic Morlet wavelets, we define superlets as multiplicative ensembles of wavelet responses and realize learnable fractional orders via softmax-normalized weights over discrete orders, computed as a logdomain geometric mean. We establish admissibility (zero mean) and continuity in order and frequency, and characterize approximate analyticity by bounding negative-frequency leakage as a function of an effective cycle parameter. Building on these results, we introduce the Learnable Fractional Superlet Transform (LFST), a fully differentiable front-end that jointly optimizes (i) a monotone, logspaced frequency grid, (ii) frequency-dependent base cycles, and (iii) learnable fractional-order weights, all trained end-to-end. LFST further includes a learnable asymmetric hard-thresholding (LAHT) module that promotes sparse, denoised TF activations while preserving transients; we provide sufficient conditions for boundedness and stability under mild cycle and grid constraints. To exploit LFST for SER, we design a compact Spectro-Temporal Emotion Encoder (STEE), achieving strong performance with a parameter budget that is orders of magnitude smaller than large self-supervised models, at the cost of additional frontend computation compared to STFT- or LEAF-based baselines. STEE consumes two-channel TF maps, magnitude S and phase-congruency κ, through a compact multi-scale stack with residual temporal and depthwise-frequency blocks, Adaptive FiLM gating, axial (time-axis) self-attention, global attentive pooling, and a lightweight classifier. The full LFST+STEE system is trained in a standard train-validate-test regime using focal loss with optional class rebalancing, and is validated on IEMOCAP, EMO-DB, and the private NSPL-CRISE dataset under standard protocols. By unifying a principled, learnable TF transform with a compact encoder, LFST+STEE replaces ad hoc front-ends with a mathematically grounded alternative that is differentiable, stable, and adaptable to data, enabling systematic ablations over frequency grids, cycle schedules, and fractional orders within a single end-to-end model. The source code of this paper is shared on the GitHub repository: https://github.com/alaaNfissi/LFST-for-SER.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Learnable Fractional Superlet Transform (LFST), a novel, fully differentiable time-frequency (TF) front-end for speech emotion recognition (SER). LFST generalizes traditional superlets by learning fractional orders via softmax-weighted geometric means across multi-scale Morlet wavelets, enabling data-adaptive TF resolution. It jointly optimizes a log-spaced frequency grid, frequency-dependent base cycles, and fractional-order weights end-to-end. A learnable asymmetric hard-thresholding (LAHT) module promotes sparse, denoised TF activations while preserving transients. Coupled with a lightweight Spectro-Temporal Emotion Encoder (STEE) that leverages magnitude and phase-congruency maps, LFST+STEE achieves state-of-the-art results on IEMOCAP, EMO-DB, and NSPL-CRISE datasets, outperforming fixed front-ends (STFT, CWT, LEAF) and prior superlet variants.

### Strengths
LFST’s core strength is that it turns the classical “one-window-fits-all” spectrogram into a continuous, end-to-end-learnable surface: by simply back-propagating through a softmax-weighted geometric mean of Morlet responses, the network can allocate razor-thin time resolution to explosive anger bursts while giving melancholy vowels the long cycles they need for pitch precision.

### Weaknesses
Although the paper brands LFST as “lightweight,” it is anything but: for every one of its 96 frequency bands it performs eight separate 1 024-sample complex convolutions, giving 8–10× the FLOPs of an FFT-based STFT and a peak GPU memory that grows linearly with order O—yet no wall-clock, multiply-add, or mobile-device latency is ever quoted, so readers are trapped in an “accuracy-only” narrative that unfairly penalises genuinely cheap front-ends such as LEAF or SincNet. Inside its own ablation the authors keep the same heavy STEE trunk (128-D, residual TF-blocks, axial attention, FiLM) for LFST while forcing STFT/CWT/LEAF to feed that same over-parameterised backbone, turning the comparison into “big network versus big network” rather than an isolated probe of the learnable superlet layer. Meanwhile the strongest community baselines—wav2vec 2.0 and HuBERT—are dismissed as “compute-intensive” without a single fine-tune, distill, or plug-in experiment; consequently the paper leaves the field with an appealing but unvalidated feature extractor instead of proving that the extra computational cost still pays off once LFST is grafted onto the SOTA pipelines that actually matter.

### Questions
1. Prohibitive Computational Cost, Yet Marketed as "Lightweight"
Although the paper repeatedly uses the word “lightweight,” LFST is anything but. For every one of the F = 96 frequency bands it performs O = 8 separate convolutions with analytic Morlet kernels of length L = 1 024 samples. At 16 kHz this corresponds to 64 ms of context per kernel; when the batch size, number of bands and temporal steps are taken into account the total operation count is 8–10 times higher than a single FFT-based STFT of the same frame. The authors sidestep this issue by writing that they “stream over orders,” yet the released PyTorch code still instantiates O complex kernels and keeps their feature maps in memory until the geometric-mean reduction, so peak GPU memory grows linearly with O. Nowhere in the paper do we find wall-clock time, number of multiply-adds, or on-device latency, so the reader has no way of knowing whether the front-end can even run in real time on a mobile CPU. When accuracy is compared with genuinely cheap front-ends such as LEAF or SincNet the absence of any compute figure creates an “accuracy-only” narrative that unfairly disadvantages the baselines.
2. Baseline Selection Is a Straw-Man Comparison
The strongest published numbers on IEMOCAP and EMO-DB come from self-supervised models like wav2vec 2.0 and HuBERT, but these are waved away with the phrase “compute-intensive and opaque.” No attempt is made to fine-tune, distill, or even reduce the dimension of such models; they are simply excluded from the comparison table. Inside the paper’s own ablation study the same large STEE encoder (128-D, residual TF-blocks, axial attention, FiLM gating) is kept for LFST, while STFT, CWT and LEAF are forced to feed the very same heavyweight trunk. This gives LFST an implicit capacity boost: the baselines are not allowed to use a smaller, faster backbone that might have been chosen had the authors started from a conventional spectrogram. Consequently the ablation measures “front-end + big network” versus “front-end + big network,” not the isolated contribution of the learnable superlet layer. The resulting delta in accuracy is therefore as much a consequence of parameter count as of representational power, something a fair comparison would have controlled by equalising FLOPs or memory.
3. Perhaps the most pressing issue is that LFST has so far been evaluated only as a stand-alone front-end paired with a purposely lightweight backend. In reality it is nothing more than a differentiable time-frequency layer, and its true value can only be gauged when it is grafted onto the current heavy-weights of the field—namely wav2vec 2.0, HuBERT, or other large self-supervised speech models. These systems already encode rich paralinguistic information, but they still rely on conventional mel-spectrograms or raw-waveform convolutions for their initial acoustic impressions. Injecting LFST’s adaptive, fractional-order super-resolution representations into the very first layers (or using them as an auxiliary input) could reveal whether the extra computational cost actually yields gains on top of the SOTA baselines that matter. Without such an integration experiment, the paper leaves the community with an attractive but untested feature extractor rather than a proven upgrade to the best performing pipelines.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes the Learnable Fractional Superlet Transform (LFST) combined with a Spectro-Temporal Emotion Encoder (STEE) for end-to-end speech emotion recognition (SER). LFST generalizes superlet theory to learnable fractional orders, enabling continuous trade-offs between time and frequency resolution. It also introduces a learnable asymmetric hard-thresholding (LAHT) module for sparse denoising. Extensive experiments on IEMOCAP, EMO-DB, and NSPL-CRISE demonstrate strong performance, surpassing state-of-the-art baselines with high interpretability and stability.

### Strengths
• The paper introduces an innovative Learnable Fractional Superlet Transform (LFST) that enables a continuous and learnable trade-off between time and frequency resolution, extending conventional wavelet and STFT formulations.
• The integration with the Spectro-Temporal Emotion Encoder (STEE) results in a coherent and interpretable end-to-end framework for speech emotion recognition (SER).
• Theoretical derivations are solid, including proofs of differentiability, stability, and Lipschitz boundedness.
• The model provides meaningful interpretability—fractional order adaptation and frequency visualization reveal how LFST captures emotional cues in a physically interpretable manner.

### Weaknesses
1. The proposed approach, while theoretically elegant, may incur higher computational cost due to multi-order fractional convolutions. No runtime or FLOPs comparison is provided.
2. The Learnable Asymmetric Hard Thresholding (LAHT) module lacks a detailed ablation study to demonstrate its independent impact.
3. The datasets used are relatively limited in scale; generalization to multilingual or real-world noisy speech remains unclear.
4. Comparisons with large self-supervised models (e.g., wav2vec2.0, HuBERT) are qualitative rather than quantitative, weakening the positioning of the contribution.

### Questions
1. How does the LAHT behave under low-SNR or noisy conditions? Could over-sparsification harm important emotional features?
2. Would LFST work as a plug-in front-end for pretrained models such as wav2vec2.0 or HuBERT?
3. Can the authors provide quantitative comparisons in terms of FLOPs or inference time against STFT/LEAF front-ends?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a Learnable Fractional Superlet Transform (LFST) for speech emotion recognition, extending conventional superlets into a continuous fractional-order form. By applying a softmax-weighted geometric mean across multiple Morlet wavelet responses, LFST enables differentiable and data-driven time–frequency resolution adaptation. The authors jointly learn the log-frequency grid, base cycles, and order weights, combined with a phase-consistency measure and a Learnable Asymmetric Hard Threshold (LAHT) for denoising. A Spectro-Temporal Efficient Encoder (STEE) further integrates depthwise-separable convolutions, TF-hybrid residuals, Adaptive FiLM frequency gating, axial attention, and adaptive pooling. Experiments on IEMOCAP, EMO-DB, and NSPL-CRISE show consistent gains in accuracy and F1-score over prior handcrafted and learnable front-ends such as STFT, LEAF, and fixed superlets.

### Strengths
1. The fractional-order superlet design generalizes discrete-scale superlets into a learnable, continuous formulation with theoretically grounded differentiability and numerical stability.
2. Joint optimization of frequency grid, base cycles, and order weights, coupled with Adaptive FiLM gating, leads to a lightweight yet expressive front-end that achieves strong results with modest computation.
3. The work contributes a new class of differentiable TF front-ends that retain physical meaning, offering a bridge between signal processing and deep learning for audio emotion understanding.

### Weaknesses
1. While the paper claims interpretability, there is no visualization of learned order weights across frequency bands.
2. Although lightweight in design, there are no quantitative FLOPs or latency comparisons with LEAF, SincNet, or wav2vec2.
3. Comparing against a fixed fractional-order (non-learnable) baseline could isolate the effect of learnability versus the fractional formulation itself.
4. The datasets are small and homogeneous. Additional results under domain shifts (e.g., cross-language or different SNR levels) would test the claimed generalization.
5. Adding comparative TF heatmaps (STFT vs. fixed Superlet vs. LFST) for the same utterance would intuitively show the sharper band selectivity and phase consistency gained.

### Questions
1. How much does phase-consistency κ contribute independently of LAHT? Have the authors tested models without κ or LAHT separately to quantify their individual benefits?
2. Are the learned thresholds stable across sampling rates and speakers, or do they overfit particular acoustic domains?
3. The appendix mentions a softplus constraint to ensure fmax>fmin. Did early training ever show instability in frequency spacing, and how was it mitigated?
4. Any evidence that the model adapts meaningfully rather than overfitting individual timbre patterns?

### Soundness
3

### Presentation
2

### Contribution
2
