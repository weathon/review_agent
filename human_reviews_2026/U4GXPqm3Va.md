# Toward Complex-Valued Neural Networks for Waveform Generation

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Neural vocoders have recently advanced waveform generation, yielding natural and expressive audio. Among these approaches, iSTFT-based vocoders have recently gained attention. They predict a complex-valued spectrogram and then synthesize the waveform via iSTFT, thereby avoiding learned upsampling stages that can increase computational cost. However, current approaches use real-valued networks that process the real and imaginary parts independently. This separation limits their ability to capture the inherent structure of complex spectrograms. We present ComVo, a Complex-valued neural Vocoder whose generator and discriminator use native complex arithmetic. This enables an adversarial training framework that provides structured feedback in complex-valued representations. To guide phase transformations in a structured manner, we introduce phase quantization, which discretizes phase values and regularizes the training process. Finally, we propose a block-matrix computation scheme to improve training efficiency by reducing redundant operations. Experiments demonstrate that ComVo achieves higher synthesis quality than comparable real-valued baselines, and that its block-matrix scheme reduces training time by 25%. Audio samples and code are available at https://hs-oh-prml.github.io/ComVo/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ComVo, a complex-valued neural vocoder designed to generate high-fidelity audio within the iSTFT-based vocoder framework. ComVo leverages complex-valued neural networks (CVNNs) for both the generator and discriminators, enabling better modeling of signal characteristics and thereby enhancing audio reconstruction quality. In addition, two practical techniques are proposed: (1) Phase quantization - stabilizes training by mitigating abrupt phase drift, and (2) Block-matrix computation - improves computational efficiency by performing block-wise operations on real and imaginary components of complex-valued tensors. Experimental results show that ComVo outperforms real-valued vocoders with comparable parameter counts across multiple evaluation metrics (e.g., UTMOS, PESQ, MR-STFT). Further ablation studies demonstrate the effect of applying the phase quantization and block-matrix computation.

### Strengths
- The use of CVNNs demonstrates clear performance gains over real-valued architectures.
- The paper presents various ablation studies, providing insight into the role of each proposed component in the overall system. The architectural design is well-motivated and supported by empirical evidence.
- ComVo scales effectively, showing competitive performance across both lightweight and large model configurations.
- The advantage of ComVo is maintained even when integrated into a TTS pipeline, highlighting its versatility and robustness.

### Weaknesses
- For me, the proposed block-matrix computation seems more like an implementation-level optimization rather than a novel research contribution.
- The ablation results for phase quantization appear relatively weak, offering limited evidence to justify its effectiveness.
- Although Table 8 indicates that ComVo and the baselines have similar parameter counts, ComVo uses a different parameter datatype (e.g., ComVo’s parameters are stored in complex64 format, which effectively corresponds to two float32 tensors for real and imaginary parts). Therefore, for a fair comparison of model capacity and memory usage, the baseline parameter counts should be doubled.

Overall, the ComVo model itself is competitive and valuable, and its performance improvements are noteworthy. However, given that ‘phase quantization’ and ‘block-matrix computation’ are addressed as major contributions in the main text, I think their novelty and impact should be more clarified and empirically supported to substantiate their significance as research contributions.

### Questions
- The phase quantization layer is applied after the first complex Conv1D in the generator. If the first Conv1D was not complex-valued, the phase quantization layer would not be required. What would happen if this sequence (complex Conv1D followed by phase quantization) was replaced with a single real-valued Conv1D layer? Such a comparison would help clarify the necessity and contribution of the phase quantization layer.
- Why is the phase quantization layer not commonly employed in other complex-valued CNNs?
- Could the authors provide a theoretical justification for why the phase quantization layer is beneficial or necessary for improved performance?
- Why the inference latency of ComVo is noticeably slower than Vocos, despite adopting a similar architectural foundation?
- A comparison of models with equivalent memory usage would be informative, since ComVo uses complex-valued tensors for its parameters, which actually consist of two float tensors. For example, it would be interesting to see whether a ComVo model with half the number of parameters still outperforms the real-valued baselines.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces ComVo, a complex-valued neural vocoder that performs waveform generation entirely in the complex domain using a GAN-based architecture. The authors argue that existing iSTFT-based vocoders rely on real-valued networks that process real and imaginary parts independently, limiting their ability to capture the inherent structure of complex spectrograms. The proposed model employs complex-valued neural networks (CVNNs) to jointly model the real and imaginary components of spectrograms. The paper also introduces phase quantization as a regularization method and a block-matrix computation scheme to improve training efficiency. The experimental results show that ComVo outperforms existing real-valued vocoders in terms of synthesis quality and training time.

### Strengths
Originality: The introduction of complex-valued neural networks (CVNNs) for waveform generation is an interesting and novel approach that is not widely explored in the context of vocoders. The proposed method shows potential in capturing the structure of complex spectrograms by treating them as unified complex entities.

Quality: The paper is well-written, and the experimental setup is clearly described. The proposed method shows promising results in terms of both objective and subjective evaluations.

Clarity: The paper is presented in a clear and structured manner. The explanations of the method, including the details of the generator, discriminator, phase quantization, and block-matrix computation scheme, are well-explained.

### Weaknesses
A major weakness of this paper is that it compares the proposed method only to real-valued vocoders and iSTFT-based methods. The paper does not include a comparison with vocoders that predict both amplitude and phase spectrograms (such as APNet and FreeV). These methods already integrate both real and imaginary parts in their amplitude and phase spectrogram predictions, which might address the issue the authors claim with real-valued networks. Without this comparison, it is difficult to conclusively prove that ComVo offers a significant advantage over existing methods. The authors should include such comparisons to strengthen the argument for their method's effectiveness.

### Questions
See the above Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ComVo, a neural vocoder for waveform generation that operates within the complex domain. The core idea is to leverage complex-valued neural networks (CVNNs) for both the generator and a multi-resolution discriminator (cMRD), arguing that this allows the model to better capture the intrinsic structure of complex spectrograms compared to conventional real-valued networks that process real and imaginary components independently. The authors also introduce two technical refinements: a phase quantization layer to act as a regularizer, and a block-matrix computation scheme to improve training efficiency. The paper presents a series of experiments and ablation studies showing that ComVo achieves competitive or superior performance on objective and subjective metrics against several strong real-valued vocoder baselines.

### Strengths
The exploration of complex-valued networks for generative audio tasks is a compelling research direction, and the authors present a well-executed implementation. The paper is technically solid; the proposed phase quantization is an interesting inductive bias for stabilizing phase prediction, and the block-matrix formulation for accelerating training is a valuable engineering contribution that demonstrably reduces training time by 25%. The experimental evaluation is thorough, with comparisons against strong, widely-used baselines like HiFi-GAN and Vocos on standard datasets. The ablation studies in Table 4 are particularly useful for dissecting the contributions of the complex-valued generator and discriminator.

### Weaknesses
Despite the positive results, I have fundamental reservations about the paper's central motivation. The primary claim is that CVNNs are superior because they "capture the intrinsic dependencies between the real and imaginary components." However, this central hypothesis is asserted rather than rigorously validated. The performance gains, while present, do not in themselves prove that this specific mechanism is the cause. My main conceptual issue is that for a spectrogram to be perfectly invertible back to a real-valued signal, it must satisfy time-frequency consistency. This means the space of valid spectrograms for real audio is a highly structured subspace within the broader domain of all possible complex spectrograms. By moving all computations into an unconstrained complex domain, the model may actually face the additional burden of learning to stay within this physically valid subspace, which might not be an advantage. The paper does not address this potential conflict.
Furthermore, the overall architecture still relies on a standard Multi-Period Discriminator (MPD) operating on the real-valued waveform. The MPD is known to be a very powerful component in modern vocoders. Its presence makes it difficult to ascertain whether the observed quality improvements truly stem from the benefits of complex-domain feedback via the cMRD, or if the MPD is still doing the majority of the perceptual heavy lifting. Finally, from a structural standpoint, the detailed background in Section 2.1 on the fundamentals of CVNNs feels more appropriate for an appendix, as it disrupts the main narrative of the paper.

### Questions
I hope the authors can address the following points to strengthen their claims and clarify the contributions of their work:
1. The core premise of the paper needs a stronger defense. Given that the ultimate target is a real-valued signal, which implies its spectrogram must satisfy time-frequency consistency, why is it advantageous for the hidden layers to operate under general complex arithmetic rather than in a way that respects this physical constraint from the outset? Could the unconstrained complex modeling actually be a less efficient path to the desired solution space?
2. The central claim is that the model better "captures intrinsic dependencies" between real and imaginary parts. Beyond the final performance metrics, is there any more direct analysis you can provide to support this? For instance, have you analyzed the learned phase-magnitude relationships or the structure of the internal representations to show they are more coherent than in real-valued models?
3. The ablation study in Table 4 is insightful, but the powerful real-valued MPD remains a constant across all configurations. How can we be confident that the gains attributed to the complex-valued cMRD are not simply an artifact of having two strong discriminators, with the MPD still being the primary driver of quality? Have you considered an ablation where only the cMRD is used, to truly isolate its effectiveness?
4. Other iSTFT-based models like Vocos and iSTFTNet also operate on complex spectrograms, but their motivation seems more direct—it's a computationally efficient and direct target for the network. Your paper claims a more fundamental modeling advantage. Could you elaborate on why your motivation leads to a better vocoder than one motivated purely by computational efficiency?

### Soundness
2

### Presentation
2

### Contribution
2
