# DiceFormer: Spiking Audio Transformer with Density-Aware Dice Attention

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Spiking Neural Networks (SNNs) have garnered significant attention due to their potential for low energy consumption. However, their application in the audio domain remains relatively underexplored. This work aims to close this gap by designing spiking transformers suitable for audio processing applications. We introduce DiceFormer, a directly trained spiking transformer that incorporates two novel components: (i) Spike Dice Attention (SDA), a spike-based attention module that leverages the Dice similarity concept to produce density-aware attention scores, which improve the modeling of spike-based representations; and (ii) Spike Audio Dice Attention (SADA), Spike Audio Dice Attention (SADA), an SDA-based extension specifically designed to handle the frequency–temporal features inherent in complex audio spectrograms. Extensive experiments demonstrate that DiceFormer achieves superior performance over existing state-of-the-art (SOTA) SNNs on mainstream audio datasets. Notably, when trained from scratch, DiceFormer achieves an mAP of 0.161 on AudioSet (20K) with only 54.3M parameters, substantially outperforming prior models. It also establishes new SOTA results on ESC-50 and SCV2, highlighting the promise of SNNs in complex audio processing.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes DiceFormer, an SNN transformer for audio with two key modules: SDA (Spike Dice Attention), a density-aware, linear-time attention based on a Dice-style score, and SADA (Spike Audio Dice Attention), a frequency/temporal dual-branch variant for spectrograms. It reports strong from-scratch results on AudioSet-20K (mAP 0.161 with DiceFormer-10-L), ESC-50, and Speech Commands V2, and shows SDA can be dropped into prior SNN ViTs for gains.

### Strengths
1) The paper diagnoses spike-density bias in dot-product/Hadamard attention and introduces a Dice-based score to normalize by token activity, keeping O(ND) complexity. 

2) DiceFormer-10-L reaches 0.161 mAP on AudioSet-20K and shows favorable parameter/energy figures compared with SNN/ANN baselines; smaller variants are also strong.

3) Replacing attention in prior SNN ViTs with SDA improves their scores, suggesting the mechanism’s usefulness beyond this architecture.

### Weaknesses
1) The paper lists SDA and SADA as two main contributions, but SADA is essentially a dual-branch time–frequency decoupling—a strategy already explored by DTF-AT (AAAI’24)，yet the contribution currently reads as incremental in architecture (single- vs. dual-branch) rather than conceptually new. 
2) Prior SNN Transformers (e.g., Spikformer, Spike-driven Transformer, QKFormer) emphasize avoiding MACs in favor of AC-style primitives for spiking self-attention. In contrast, Eq. (4) introduces an input-dependent division (normalizing by the sum of activities), which is not offline-precomputable and is often **unfriendly to neuromorphic substrates** (division/reciprocal are costly, latency-prone, and may break integer/accumulate-only pipelines). So, my major concern is that such division-based operations fundamentally limit neuromorphic deployability.  The paper should quantify and justify this choice.
3) DiceFormer uses **PLIF** neurons (learnable τ). It’s not specified whether compared SNN baselines in the authors’ re-implementations also use PLIF (or just LIF). If DiceFormer benefits from PLIF while baselines use LIF, that could confound fairness. 
4) The paper standardizes “10 layers = 5 SADA + 5 SDA” across S/M/L but it does not examine alternative SADA:SDA **ratios** (eg., 4:6 or 6:4) nor different **orderings** (e.g., front-loaded SADA vs. interleaved vs. back-loaded). Such ablations are important to verify complementarities and to identify where each module provides the largest gains.

### Questions
Please refer to the four points included in **Weaknesses** for details.

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
The paper proposes DiceFormer, a spiking audio transformer, for the tasks of audio representation learning. 

It consists of two key components, namely, Spike Dice Attention (SDA) and Spike Audio Dice Attention (SADA).

SA replaces element-wise attention with a Dice similarity based score, which is density-aware.

SADA decouples frequency and time branches before fusing them. 

Experiments on AudioSet-20K, ESC-50, and Speech Commands V2 sow its competitive performance.

### Strengths
+ The review appreciates the introduction of SNN to the audio domain, which is much less studied than the vision domain.

+ SDA seems to maintain a linear-time complexity like prior spiking Hadamard attentions, which has a good scalability.

+ The proposed method shows a very competitive performance on AudioSet-20K, ESC-50, and SC-V2 datasets. 

+ Overall this paper is easy-to-follow.

### Weaknesses
- While the reviewer acknowledges its contribution to the audio domain, it should be pointed out that, the technique novelties of the proposed method is limited, and the theory insight is neither sufficient. Dice similarity is a common and mature metric, but this work is not theoretically justified.

- Besides, the effectiveness and rationale of the Dice similarity metric should be compared with other commonly-used distance metrics, varying from cosine, Jaccard to F1. 

- For the state-of-the-art comparison, some concerns also remain, in terms of the fairness. While Table 1 is really nice, whether the compared state-of-the-art methods, such as DTF-AT, are under the same encoder or spiking confirguation or use the pre-trained model, is not justified. If not the same, then the state-of-the-art performance is less pausible.

- Besides, The authors seem to miss the comparison with [1], which is also a highly-cited work:

[1] Chen, Ke, et al. "Hts-at: A hierarchical token-semantic audio transformer for sound classification and detection." ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022.

- The ablation studies in this paper are also limited. For example, multiple hyper-parameters in SDA are not tested.

- Besides, the decoupling method in SADA should be compared with some other simipler alternatives to show its superiority. 

- The writing and presentation of this paper needs to be significantly enhanced. For example, 'INITIALIZE' and 'PROJECTION CONV' are not professional and the readers can only make educated guess.

### Questions
Please refer to the weakness section and addresses them point-by-point.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a highly compelling contribution with outstanding strengths across originality, quality, clarity, and significance. Its originality is exceptional, as it identifies and addresses the previously overlooked problem of density bias in spike-based attention mechanisms—a novel and crucial problem formulation. The proposed solutions, Spike Dice Attention (SDA) and its audio-specific extension SADA, are creative and well-motivated, representing a new class of density-aware attention for SNNs. The research quality is superb, backed by both theoretical analysis of SDA's properties and comprehensive empirical validation on three major audio benchmarks. The paper demonstrates state-of-the-art performance for SNNs, significantly outperforming prior work, and uses thorough ablation studies and correlation analyses to rigorously support its claims. The work is presented with excellent clarity, starting from a clear motivation and logically building up to the proposed architecture and its evaluation. Finally, the significance is substantial; by successfully adapting SNN Transformers to the complex audio domain and establishing new SOTA benchmarks, this work not only bridges the performance gap with ANNs but also provides a foundational and highly efficient architecture for future research in neuromorphic audio processing.

### Strengths
This paper presents a highly compelling contribution with outstanding strengths across originality, quality, clarity, and significance. Its originality is exceptional, as it identifies and addresses the previously overlooked problem of density bias in spike-based attention mechanisms—a novel and crucial problem formulation. The proposed solutions, Spike Dice Attention (SDA) and its audio-specific extension SADA, are creative and well-motivated, representing a new class of density-aware attention for SNNs. The research quality is superb, backed by both theoretical analysis of SDA's properties and comprehensive empirical validation on three major audio benchmarks. The paper demonstrates state-of-the-art performance for SNNs, significantly outperforming prior work, and uses thorough ablation studies and correlation analyses to rigorously support its claims. The work is presented with excellent clarity, starting from a clear motivation and logically building up to the proposed architecture and its evaluation. Finally, the significance is substantial; by successfully adapting SNN Transformers to the complex audio domain and establishing new SOTA benchmarks, this work not only bridges the performance gap with ANNs but also provides a foundational and highly efficient architecture for future research in neuromorphic audio processing.

### Weaknesses
1. In your core formulation for Spike Dice Attention, Equation (4), both the numerator and the denominator consist of computed variables rather than raw spike events. This computational pattern does not adhere to the fundamental spike-driven paradigm inherent to Spiking Neural Networks (SNNs). Consequently, the mechanism as formulated is conceptually inconsistent with the principles of SNNs.

2. While the energy efficiency of the proposed Dice Attention mechanism is a notable advantage, its case as a viable alternative to established attention mechanisms in ANNs is not sufficiently compelling if this is its sole benefit. Superiority in energy consumption alone does not guarantee substitutability; further evidence demonstrating comparable or superior performance, scalability, or other functional advantages is necessary to justify its broader adoption.

3. Theoretically, the frequency and temporal domains are not equivalent. Therefore, the approach of simply splitting the input along the channel dimension into frequency and temporal branches and processing them with an identical attention mechanism is conceptually unsound.

### Questions
See the Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a directly trained spiking transformer (DiceFormer) with the proposed spike dice attention and spike audio dice attention. Spike dice attention is a spike-based attention module that leverages the dice similarity concept to produce density-aware attention scores, which improve the modeling of spike-based representations. Spike audio dice attention is an SDA-based extension specifically designed to handle the frequency–temporal features inherent in complex audio spectrograms. Extensive experiments demonstrate that DiceFormer achieves superior performance over existing state-of-the-art (SOTA) SNNs on mainstream audio datasets.

### Strengths
1.	The paper introduces a novel density-aware spiking attention mechanism (SDA) with clear motivation and theoretical grounding, effectively addressing the spike-density bias issue in existing spiking transformers.
2.	The model includes a well-designed adaptation for audio processing, explicitly decoupling temporal and frequency features in a biologically and computationally coherent manner.
3.	The paper provides comprehensive experimental results across multiple audio classification benchmarks, demonstrating consistent and strong performance compared to both SNN and ANN baselines.

### Weaknesses
1.	Although the proposed SDA is positioned as a general spiking attention mechanism, the experiments are limited to audio classification tasks. Evaluating SDA, SSA, and SDSA on vision benchmarks such as ImageNet would strengthen the claim of generality and could also serve as potential pretraining for audio classification.
2.	The Spike Audio Dice Attention (SADA) module shows limited novelty. The idea of decoupling time and frequency attention has already been explored extensively in prior works on audio transformers (e.g., Septr[1], DTF-AT[2]). The contribution here appears to be mainly an adaptation of that concept to a spiking setting rather than a fundamentally new architectural design.
3.	It is unclear whether the Dice score operation used in SDA is hardware-friendly for neuromorphic deployment. Since hardware efficiency is a key motivation for SNNs, an analysis or discussion of its implementability on neuromorphic hardware would be valuable.
4.	It is somewhat unexpected that DiceFormer surpasses pretrained ANN models such as AST on AudioSet, despite not using large-scale pretraining (e.g., ImageNet pretraining or SSL) and operating with only four timesteps. This raises concerns about training fairness and reproducibility. The authors are encouraged to clarify the experimental setup and indicate whether the code and pretrained checkpoints will be released to ensure transparency.
[1] Ristea, Nicolae-Catalin, Radu Tudor Ionescu, and Fahad Shahbaz Khan. "Septr: Separable transformer for audio spectrogram processing." arXiv preprint arXiv:2203.09581 (2022).
[2] Alex, T., Ahmed, S., Mustafa, A., Awais, M., & Jackson, P. J. (2024). DTF-AT: Decoupled Time-Frequency Audio Transformer for Event Classification. Proceedings of the AAAI Conference on Artificial Intelligence, 38(16), 17647-17655. https://doi.org/10.1609/aaai.v38i16.29716

### Questions
See in Weakness.

### Soundness
3

### Presentation
3

### Contribution
2
