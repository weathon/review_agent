# Self-Guidance: Training VQ-VAE Decoders to be Robust to Quantization Artifacts for High-Fidelity Neural Speech Codec

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Neural speech codecs, predominantly based on Vector-Quantized Variational Autoencoders (VQ-VAEs), serve as fundamental audio tokenizers for speech large language models (SLLMs). However, their reconstruction fidelity is limited by quantization errors introduced during latent space discretization. Existing solutions typically increase model complexity through larger codebooks or hierarchical quantization, which subsequently intensify the modeling challenge for downstream SLLMs. Inspired by the key insight that the codec decoder produces superior output from continuous pre-quantize embeddings, we propose a novel self-guided training mechanism that addresses this problem by enhancing decoder robustness rather than modifying the quantization process. Our method introduces an additional training objective that aligns the decoder's intermediate features when processing both quantized tokens and continuous pre-quantized embeddings through a feature-mapping loss. Extensive experiments on XCodec2 demonstrate that self-guidance consistently improves reconstruction quality across various codebook sizes and quantization techniques (FSQ, SimVQ, multi-codebook VQ), achieving state-of-the-art performance for low-bitrate speech codecs. The method requires minimal additional training cost and no inference-time modifications, offering an efficient solution for high-fidelity neural audio coding. Remarkably, our approach enables a 4× reduction in codebook size while maintaining comparable fidelity. Downstream text-to-speech experiments confirm that this reduction significantly improves LLM-based synthesis performance by simplifying the token modeling space.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a new self-guidance loss to improve the training of neural speech codecs. Motivated by the observation that decoders can produce better reconstructions when using pre-quantized encoder outputs, the authors introduce a feature mapping loss that aligns the decoder’s intermediate features with those produced from the pre-quantized encoder outputs. The proposed loss is evaluated on the XCodec-2 baseline (Ye et al., 2025b), showing improved low-bitrate reconstruction performance on the LibriSpeech dataset. In addition, results indicate that with the self-guidance loss, XCodec-2 can maintain reconstruction fidelity when the codebook size is reduced by 4x, and the downstream TTS performance is also improved.

### Strengths
1. The motivation for the self-guidance loss is clear, i.e., decoders reconstruct better when using the pre-quantized encoder features. 
2. The proposed loss is simple to implement and introduces negligible computational overhead. 
3. Empirical results demonstrate that the self-guidance loss improves reconstruction quality within the XCodec-2 framework.

### Weaknesses
1. The improvements in reconstruction quality appears marginal, e.g., gains of only about 0.1 in PESQ and 0.04 in UTMOS (Table 2). 
2. The experiments are not sufficient. The proposed loss is only evaluated on the XCodec-2 framework. Given the simplicity of the idea, it should be tested on additional single-codebook codecs to better demonstrate its general effectiveness.
3. In Table 4, the WER slightly increases after applying the self-guidance loss, suggesting semantic information loss. The authors should provide a stronger justification or analysis for this observation.

### Questions
1. Is the feature mapping loss in Equation 2 applied at each decoder resolution level? The current equation seems ambiguous. 
2. As in Weakness 2, does the self-guidance loss depend specifically on XCodec-like architectures, or is it generally applicable to single-codebook neural speech codecs? 
3. Since the reported improvements are relatively small, have the authors considered simpler ablation studies? For example, could you report the gain from the self-guidance loss alone, without the semantic and adversarial losses, to isolate its contribution? 
4. The authors claim that the self-guidance loss enables a smaller codebook, benefiting downstream speech LLMs. However, large codebooks are not necessarily problems for LLMs, sometimes larger vocabularies can enhance performance. I wonder why the authors do not frame this claim from a compression perspective, emphasizing that a 4x reduction in codebook size implies a 4x increase in compression rate.

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
4

### Summary
This paper introduces "self-guidance," a novel and elegant training mechanism for VQ-VAE-based neural speech codecs. The core idea is to improve the decoder's robustness to quantization artifacts by introducing an auxiliary feature-mapping loss that encourages the decoder to produce similar intermediate representations for both quantized tokens and their continuous pre-quantization counterparts. The method is simple to implement, adds negligible computational overhead during training, and requires no changes at inference time. Through extensive experiments on the state-of-the-art XCodec2 model, the authors demonstrate significant and consistent improvements in reconstruction quality across various metrics, codebook sizes, and quantization methods. A key finding is that self-guidance enables a 4x reduction in codebook size while maintaining comparable fidelity, which is shown to directly benefit downstream autoregressive TTS tasks by simplifying the token modeling space.

### Strengths
1. The proposed self-guidance mechanism is a simple, intuitive, and novel training objective. It addresses the core problem of quantization error by enhancing the decoder directly, rather than adding complexity to the quantizer or architecture. Its minimal overhead (a single extra forward pass during training with no gradient computation) makes it highly efficient and practical.

2. The method is applied to XCodec2 and evaluated on a standard benchmark (LibriSpeech), demonstrating consistent state-of-the-art performance. The ablation studies convincingly show the method's effectiveness across different codebook sizes and vector quantizer types (FSQ, SimVQ), proving its generalizability.

3. The paper provides a crucial analysis in Figure 3, demonstrating that the performance gain comes from improved decoder robustness, not from a reduction in the quantization error itself. This confirms the authors' central hypothesis and provides a clear understanding of the method's mechanism of action.

### Weaknesses
1. The paper mentions selecting the loss weight λ_guide, but a sensitivity analysis showing how performance varies with this weight would enhance the experimental rigor and provide practical guidance for future work.

2. The method's effectiveness is demonstrated exclusively on single-codebook models. Its applicability and potential benefits in common multi-codebook architectures (e.g., RVQ) remain entirely unexplored, which significantly limits the proven scope and generalizability of the proposed technique.

3. The proposed "self-guidance" is essentially a form of self-distillation, where a network branch with privileged information (pre-quantized latents) acts as a teacher. The paper fails to acknowledge this strong connection to existing paradigms, thereby overstating its novelty and positioning the contribution more as a clever engineering refinement than a fundamental advance.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the audio tokenizer task, aiming to improve the quality of audio codecs. The authors propose an approach that aligns hidden embeddings to better reconstruct fine-grained audio information.

### Strengths
- The proposed method is well-motivated and appears conceptually sound, leading to consistent performance improvements across multiple downstream tasks.

- Extensive experiments on various downstream applications demonstrate the effectiveness of the proposed approach.

- The experiments on Hidden Feature Alignment MSE provide a clear comparison of different methods in terms of information reconstruction capability. This analysis is valuable and could serve as a useful reference for future work on improving reconstruction quality in audio tokenizers.

### Weaknesses
- Could the authors provide more comprehensive experimental results in Table 5? For example, results for XCodec2 with a 16,384-sized codebook and XCodec2+SG with a 65,536-sized codebook would offer a fuller view of the model’s behavior under different configurations.

- While the proposed method is effective, its novelty appears limited. It remains unclear why this approach can mitigate quantization artifacts. The paper mentions decoder robustness, but the conceptual difference between decoder robustness and quantization error is not clearly articulated. Providing a deeper explanation of this relationship would help readers better understand the core contribution.

### Questions
I would appreciate it if the authors could elaborate on the conceptual and practical differences between decoder robustness and quantization error. How does improving robustness directly contribute to reducing quantization artifacts? A more detailed discussion or visualization would strengthen the theoretical understanding of the proposed approach.

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
5

### Summary
This manuscript proposes a technique to enhance the reconstruction fidelity of discrete speech codecs. The core idea involves a modification to the training objective: during training, both the continuous encoder-output features and the subsequent discrete quantized representations are passed to the decoder. An auxiliary feature-mapping loss (MSE) is then applied within the decoder to minimize the discrepancy between the internal representations generated from these two distinct inputs. The authors report performance on several standard reconstruction metrics (e.g., STOI, MCD) and simple downstream TTS tasks (e.g., WER, SIM).

### Strengths
The underlying motivation for this approach is sound. In VQ-based architectures, ensuring that the decoder is robust to the information bottleneck of the quantizer—by training it to map both pre- and post-quantization features to a similar internal manifold—is a reasonable objective beyond simply optimizing the codebook itself.

### Weaknesses
1. The primary concern is the limited methodological contribution (only an easy trick). The proposed method amounts to a straightforward technical adjustment (i.e., an auxiliary loss) rather than a substantial new approach. The introductory discussion in Section 3, which focuses on the well-established concept of information loss in discrete quantization, offers little new insight and feels remedial. The method itself is not particularly insightful or heuristic.
2. The most critical weakness is the lack of compelling empirical evidence. The results presented in Table 1 indicate that the proposed method yields minimal to no improvement over the baseline across the majority of metrics (STOI, MCD, WER, SIM, UTMOS). The PESQ metric, which the authors specifically highlight, demonstrates only a marginal improvement (reportedly a 0.3 change), falling short of demonstrating meaningful practical utility.
3. Given that the method is presented as a general "trick" adaptable to the VQGAN framework, its evaluation is insufficiently narrow. To substantiate its effectiveness, the technique should have been validated across a much broader spectrum of modern audio codec models, not just the single architecture presented. Furthermore, to claim generalizability within the VQGAN context, validation on other relevant modalities (e.g., image, video) would be essential.

### Questions
This work may meet the threshold for ICASSP/Interspeech, but it currently lacks the requisite novelty and impact expected for ICLR.

### Soundness
2

### Presentation
4

### Contribution
1
