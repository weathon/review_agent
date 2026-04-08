## Human Reviewer 1

### Summary
The paper presents USR 2.0, an improved version of the Unified Speech Recognition (USR) framework that integrates ASR, VSR, and AVSR into a single model. The original USR is limited by the high cost and error accumulation of autoregressive decoding for attention-based pseudo-labels. USR 2.0 addresses this with CTC-driven teacher forcing, which uses CTC outputs to generate attention pseudo-labels in a single forward pass. A mixed sampling strategy balances the expressiveness of autoregressive training with the robustness of CTC. As a result, USR 2.0 achieves faster training, improved robustness to distribution shifts, and state-of-the-art performance on LRS2, LRS3, and WildVSR using a unified model.

### Strengths
* The authors' overall intuitions are reasonable and well-supported by design choices. USR 2.0 introduces a CTC-driven pseudo-labelling approach that effectively removes the autoregressive bottleneck in attention-based decoding, resulting in significantly faster training and improved inference efficiency.

* The model unifies ASR, VSR, and AVSR within a single architecture, and demonstrates strong robustness to long inputs, noise, and domain shifts, while maintaining competitive performance on ID benchmarks.

* The paper is further strengthened by a clear training strategy, a simple yet effective mixed sampling method, and a comprehensive set of experiments including ablations, OOD tests, and qualitative analysis.

### Weaknesses
* Since the attention decoder must operate autoregressively during inference, the benefit of test-time parallelism is limited, and the speedup primarily applies to the training phase.

* Although attention pseudo-labels generated from CTC-driven decoding may lack global coherence, the authors convincingly argue that this does not hinder learning during self-training, as both teacher and student are conditioned on the same token sequence. However, this could limit reuse of such pseudo-labels in non-self-training settings, where inference coherence is required.

* Minor weakness :
    * In Figure 2, the line representing teacher forcing looks quite similar to the other lines. Please change it to a different color for better visibility.

### Questions
* Mixed sampling uses AR decoding in only half of the training steps, yet inference is fully AR. Have the authors compared this to a model trained entirely with AR decoding? It would clarify how much performance is being traded for efficiency.

* Is there any potential to use CTC-driven attention decoding at inference time? While coherence may be limited, it might still be useful in latency-sensitive scenarios.

### Soundness
4

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper proposes a framework, for semi-supervised universal speech recognition, USR 2.0 building upon USR. Hybrid CTC–attention training in USR suffers from inefficiency and instability: autoregressive (AR) decoding is slow and error-prone when generating pseudo-labels for unlabeled data, and accumulated prediction errors can degrade model quality. The authors identify that the CTC branch—though less expressive—is inherently more stable and monotonic, making it a good candidate for guiding the attention branch.

To exploit this, the paper introduces CTC-driven teacher forcing, where the teacher model’s CTC predictions are first greedily decoded and merge-collapsed and then used as fixed token prefixes to condition the attention decoder. This approach removes the need for sequential AR decoding, allowing attention-based pseudo-labels to be generated in parallel in one forward pass. Although such CTC-conditioned sequences may not be globally coherent, they still provide consistent and aligned supervision for training, since both teacher and student operate under identical conditioning.

However, this introduces a mismatch between training and inference: during training, the decoder conditions on CTC tokens, while at inference it uses its own AR outputs. To reduce this exposure bias, the paper adds a mixed-sampling strategy, randomly alternating between CTC-driven mode and standard AR mode. The CTC-driven mode emphasizes efficiency and robust alignment, while the AR mode preserves linguistic coherence and matches inference conditions. Together, these yield faster training, improved long-form stability, and better robustness to out-of-distribution data, while retaining the modeling flexibility of attention decoding.

### Strengths
1. CTC-driven teacher forcing is an elegant idea that leverages the stability and monotonic alignment properties of CTC to guide the attention decoder. It enables parallel pseudo-label generation without slow autoregressive decoding, thereby reducing computational cost and eliminating cascading AR errors.

2. Improved training efficiency: The paper shows a 2× reduction in training time, which is significant for multimodal setups (audio, visual, audiovisual). This is achieved without compromising accuracy, highlighting the practicality of the approach.

3. Strong robustness across conditions: The mixed-sampling strategy strikes a balance between efficiency and exposure-bias mitigation. The model demonstrates improved out-of-distribution (OOD) robustness, particularly for long-form or noisy inputs, compared to both the baseline USR and modality-specific self-supervised systems.

4. Thorough experimentation: The experiments are well-structured, spanning ASR, VSR, and AVSR tasks and several benchmarks. The ablations clearly quantify the contribution of each component.

5. Conceptual clarity and reproducibility: The paper gives a clear conceptual link between the CTC and attention paradigms, and explains the motivation for bridging them. The code and implementation details are provided, making the approach accessible to replicate.

### Weaknesses
The proposed approach relies heavily on the quality of CTC-generated pseudo-labels. While the paper acknowledges that the student decoder is trained to predict the teacher’s outputs under the same CTC-driven inputs, the method still assumes that these pseudo-labels are sufficiently coherent to guide decoder learning. For challenging or noisy segments, however, CTC errors can propagate through teacher forcing since the decoder conditions directly on these imperfect sequences. Consequently, the learning signal may not always be optimal, potentially limiting the broader applicability of the approach beyond the specific experimental settings demonstrated in the paper.

### Questions
See Weaknesses above.

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces USR 2.0, an enhanced version of the Unified Speech Recognition (USR) framework. It effectively addresses its predecessor's limitations (high training cost, poor robustness, etc.) to deliver a more efficient and reliable semi-supervised model for ASR, VSR, and AVSR. The core innovation is CTC-driven teacher forcing, which speeds up pseudo-label generation approximately 40x and couples the two supervision signals, enabling the model to achieve new state-of-the-art results while halving the total training time.

### Strengths
1.	The paper itself is clearly written and easy to follow.
2.	The paper directly addresses the problems and limitations of USR.
3.	The results are robust compared to baselines.

### Weaknesses
There are no obvious flaws in this paper. Only a few points require clarification (see Questions).

### Questions
1.	Are there any failure cases for USR 2.0?
2.	Why weren’t other methods, especially USR, trained with the highest-resource setting (Huge)?
3.	Beyond speech recognition, could this CTC-driven teacher forcing paradigm be applied to other sequence-to-sequence tasks?

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper introduces USR 2.0, a new semi-supervised framework for unified speech recognition (ASR, VSR, AVSR). It directly addresses two issues in the original USR: the high training cost of autoregressive (AR) pseudo-labeling and the model's brittleness on out-of-distribution (OOD) data. The core contribution is a mechanism called CTC-driven teacher forcing, where fast and robust pseudo-labels from the teacher's CTC branch are used as input prefixes to the teacher's decoder to generate attention-based targets in a single, non-autoregressive pass. This is combined with a "mixed sampling" strategy to mitigate train-test mismatch. The authors claim this approach halves training time, dramatically improves OOD robustness, and achieves state-of-the-art results with a single model.

### Strengths
- The paper targets a clear and significant problem: the slowness and unreliability of AR pseudo-labeling in a strong existing framework.
- The "CTC-driven teacher forcing" method is an elegant solution. It cleverly uses the CTC branch's speed and robustness to guide the attention decoder, effectively solving the speed and stability issues simultaneously.
- The claims are not just supported by standard in-distribution (ID) benchmarks but by a comprehensive suite of OOD tests, including:
   - Demonstration of clear superiority over USR on long sequences, where USR's performance collapses.
   - Shows strong performance on data with additive noise.
- The 2x speedup claim is validated with wall-clock time comparisons, showing the method is both faster per step and converges in fewer epochs .

### Weaknesses
- The ablations show that the 50% mixed sampling is a compromise . The pure CTC-driven mode is best for OOD robustness, while the pure AR mode is best for ID accuracy. The final method balances these, but it's a trade-off, not a solution that maximizes both simultaneously.

### Questions
- Could you elaborate on the mechanism that allows the decoder to learn from "incoherent" attention targets (as shown in Fig 7 ) and still produce coherent AR outputs at inference? Is it fair to characterize this as the decoder learning a mapping from a coherent CTC prefix to the teacher's (conditionally valid) next-token prediction, and that this mapping task itself is what transfers the CTC's robustness, regardless of the global coherence of the resulting target sequence?

- Figure 5 shows ID and OOD performance are optimized at opposite ends of the AR sampling probability spectrum. This suggests a simple 50/50 mix is a compromise. Did you consider a staged training approach? For example, (1) training only in the fast and robust CTC-driven mode (p=0.0) to get a robust model, and then (2) fine-tuning only in AR mode (p=1.0) to recover the last bit of in-distribution performance? This might be more effective or faster overall than mixing at every step.

- You state the 2x training speedup comes from (i) faster training steps (due to non-AR pseudo-labeling) and (ii) faster convergence (requiring 50 vs. 75 epochs). Could you provide a rough breakdown of this? How much of the 2x gain is from (i) vs. (ii)? For instance, how does the wall-clock time of a 75-epoch USR 2.0 run compare to the 75-epoch USR run?

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4