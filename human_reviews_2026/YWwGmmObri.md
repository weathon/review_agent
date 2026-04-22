# Primus: Enforcing Attention Usage for 3D Medical Image Segmentation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Transformers have achieved remarkable success across multiple fields, yet their impact on 3D medical image segmentation remains limited with convolutional networks still dominating major benchmarks. In this work, a) we analyze current Transformer-based segmentation models and identify critical shortcomings, particularly their over-reliance on convolutional blocks. Further, we demonstrate that in some architectures, performance is unaffected by the absence of the Transformer, thereby demonstrating their limited effectiveness. To address these challenges, we move away from hybrid architectures and b) introduce Transformer-centric segmentation architectures, termed Primus and PrimusV2. Primus leverages high-resolution tokens, combined with advances in positional embeddings and block design, to maximally leverage its Transformer blocks, while PrimusV2 expands on this through an iterative patch embedding. Through these adaptations, Primus surpasses current Transformer-based methods and competes with a default nnU-Net while PrimusV2 exceeds it and is on par with the state-of-the-art ResEnc-L architecture across nine public datasets. In doing so, we introduce the first competitive Transformer-centric model, making Transformers state-of-the-art in 3D medical segmentation. Our code will be published.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors investigate the performance dependency of convolutional networks and Transformers in 3D segmentation tasks. They demonstrate that CNN–Transformer hybrid architectures rely heavily on the convolutional component for achieving strong segmentation performance, while the Transformer’s contribution remains minimal. To address this imbalance, the authors propose two variants of Primus that incorporate recent Transformer design advances—such as Rotary Positional Embeddings (RoPE) and EVA-style MLPs—to reduce reliance on CNNs and enhance the Transformer’s contribution in 3D segmentation.

### Strengths
- The paper presents a investigation comparing CNN and Transformer architectures for 3D segmentation.

- It provides many ablation studies on architectural design choices, offering valuable insights that can guide future research on Transformer-based 3D segmentation.

### Weaknesses
- Unclear performance gap explanation: the authors investigate the issue that CNN–Transformer hybrid architectures heavily rely on the CNN module for 3D segmentation. However, it remains unclear why CNNs consistently outperform Transformers in this domain, or how the proposed design choices specifically address this performance gap. A more detailed analysis or visualization explaining the inductive bias difference would strengthen the argument.

- Misleading problem statement: the paper argues that future 3D segmentation models should move toward pure Transformer architectures, citing advantages in self-supervised learning and multi-modal integration. However, this justification is not fully convincing. Hybrid CNN–Transformer architectures (e.g., KL-VAE, VQ-VAE, and their variants) have demonstrated strong self-supervised learning capabilities, and current research trends do not exclusively favor pure Transformers for such tasks. Moreover, hybrid models can effectively perform multimodal fusion through either Transformer or CNN blocks, provided feature dimensionalities are compatible. Thus, the necessity of abandoning CNN-based architectures remains questionable.

- High parameter count and computational cost: the proposed model family includes four sizes—Small, Base, Medium, and Large. While the Large variant achieves competitive results with nnUnet, it requires around 300M parameters and 560B FLOPs, significantly higher than other more lightweight architectures. Although the paper introduces smaller variants (Small and Base), their performance gains over prior work appear marginal, suggesting that improvements may primarily stem from scaling up model size rather than architectural innovation.

- Outdated baseline comparisons: the baseline comparisons in Table 6 seem outdated. Given the field's rapid pace, works from 2023-24 would also be acceptable. However, most of the comparison baselines are from 2021.

### Questions
- Pleas see weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents PRIMUS and PRIMUS-V2, two Transformer-centric architectures for 3D medical image segmentation. The authors argue that previous hybrid CNN–Transformer models rely heavily on convolutional components and fail to demonstrate the effectiveness of Transformer blocks. PRIMUS is designed to “enforce attention usage” through architectural changes such as high-resolution tokens, 3D rotary position embeddings, and lightweight decoders. Experiments across multiple datasets show that PRIMUS and PRIMUS-V2 achieve performance comparable to or slightly exceeding strong CNN baselines such as nnU-Net and ResEnc-L.

### Strengths
1. Comprehensive empirical analysis: The paper provides an extensive evaluation across nine public datasets and multiple baselines, demonstrating strong experimental rigor.

2. Clear architecture design and ablation studies: The paper systematically explores the effects of each architectural modification, such as patch size, position embedding, and tokenizer design.

3. Good reproducibility and transparency: The authors provide detailed training configurations, public datasets, and plan to release code integrated with nnU-Net, which increases reproducibility.

### Weaknesses
1. Questionable motivation: The motivation is somewhat inconsistent. In the introduction (lines 54–68), the authors acknowledge that CNNs currently outperform Transformers in 3D medical segmentation. This raises the question of why a Transformer-centric model is necessary. Furthermore, the stated advantages of Transformers (e.g., efficiency in self-supervised learning and multimodal integration) could arguably also be achieved with hybrid CNN–Transformer models, which are dismissed too quickly. The motivation therefore needs substantial clarification and rewriting.

2. Outdated baselines: Most of the compared methods are from 2024 or earlier. Considering that this paper targets ICLR 2026, comparisons to more recent 2025 Transformer or CNN-based segmentation models are needed to ensure fairness and relevance.

3. Limited performance gain: The proposed method achieves only marginal improvements (see Table 6), and sometimes even lags behind strong CNNs such as ResEnc-L. This weak advantage raises concerns about whether it is truly necessary to pursue a Transformer-centric architecture for 3D medical segmentation.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Primus and PrimusV2, two transformer‑centric architectures for 3D medical image segmentation, and a diagnostic study that quantifies how much current transformer models actually depend on attention. The paper introduces a UNet Index to measure non‑transformer capacity, shows that removing attention often barely hurts performance in popular hybrids, and then designs Primus with high‑resolution tokens, 3D Rotary Position Embeddings, Eva‑02‑style SwiGLU MLPs, LayerScale, and a lightweight decoder. PrimusV2 adds an iterative tokenizer to better capture small structures. Across nine datasets, PrimusV2 is competitive with a strong CNN (ResEnc‑L) and clearly outperforms prior Transformer baselines under a unified nnU‑Net training setup, with ablations and scaling experiments that support the design choices.

### Strengths
- The UNet Index plus the remove‑the‑Transformer ablation offer a clear, quantitative look at hybrid designs and motivate a truly attention‑centric approach; this analysis helps explain why many prior hybrids trail well‑tuned CNNs. 
- The architecture is carefully built from principled components for the 3D setting (8×8×8 tokens, 3D RoPE, SwiGLU, LayerScale), and the authors run systematic ablations that show steady gains and improved stability. 
- The evaluation spans nine datasets with consistent training in nnU‑Net, qualitative examples, and clear reporting that PrimusV2 matches or exceeds nnU‑Net on most sets and approaches ResEnc‑L on average.

### Weaknesses
The baseline comparison omits several recent and strong segmentation models, which makes the comparisons feel dated, in particular, there are no results against MedNeXt, UNETR++, SwinUNETR‑V2, or SegMamba on the same splits and preprocessing, so it is hard to judge progress relative to the current works.

### Questions
Could you clarify why comparisons to recent methods such as MedNeXt, UNETR++, SwinUNETR-V2, and SegMamba were not included, and can you compare with them under the same setting on the reported datasets?

### Soundness
2

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
3

### Summary
This manuscript addresses a critical gap in 3D medical image segmentation: despite the success of Transformers in other domains, state-of-the-art methods in this field remain dominated by convolutional neural networks, as existing Transformer-based models suffer from over-reliance on convolutional blocks and redundant Transformer layers. 
To solve this, the authors propose two Transformer-centric architectures: Primus and PrimusV2. Key innovations include 8×8×8 small-patch tokenization for Primus, iterative patch embedding via residual blocks for PrimusV2, 3D axial Rotary Position Embedding, and a lightweight convolutional decoder. Experiments on 9 public datasets show Primus competes with default nnU-Net, while PrimusV2 matches the SOTA CNN ResEnc-L, marking the first competitive Transformer-centric model for 3D medical segmentation.

### Strengths
1.	Domain-Specific Architectural Innovations: Innovations are tailored to 3D medical images’ unique challenges: (1) 8×8×8 patch tokenization addresses the local detail loss from UNETR’s 16×16×16 patches (critical for small lesions like SBM’s <0.05 cm³ metastases); (2) 3D RoPE resolves spatial awareness issues of absolute position embeddings (ablations show DSC improves from 77.13 to 87.84, Table 4); (3) PrimusV2’s iterative patch embedding solves Transformer’s difficulty in learning low-level features with small data—all supported by clear ablation results (Table 3, Table 12).
2.	Comprehensive, Reproducible Experiments: The work benchmarks against 6 SOTA baselines across 9 datasets, covering multiple modalities (CT, MRI) and tasks. It also provides implementation details and plans to release code, ensuring reproducibility.

### Weaknesses
1.	The manuscript analyzes 9 mainstream architectures and finds that their original studies did not verify performance changes after removing Transformer blocks. However, some of these models (e.g., Transfuse, TransBTS) did conduct ablations on attention windows, position embeddings, or Transformer layer counts. Could you clarify: (1) Whether you reviewed the ablation sections of these original studies to confirm they indeed omitted "Transformer block removal" experiments? (2) For models that did test partial Transformer component adjustments (e.g., reducing attention heads), why do you think such ablations fail to reflect the redundancy of Transformer blocks as a whole?
2.	Primus-L and PrimusV2-L have 325.5M and 326.1M parameters, respectively, far exceeding the default nnU-Net’s ~30M parameters. Even with a lightweight decoder, the 8× longer token sequence (from 8×8×8 patches) increases inference latency, which is critical for clinical deployment.
3.	The core of the residual blocks in PrimusV2 lies in convolutional operations, whose role is to pre-learn low-level features, which should originally be learned by the Transformer. The ablation experiment in Appendix C.2 of the manuscript shows that for PrimusV2-M using Minimal Residual Downsampling (MRD), after removing the Transformer blocks, the average Dice Similarity Coefficient still reaches 83.98. In contrast, for Primus-M using "convolution-free Default Projection (DP)", the average DSC drops to only 19.78 after Transformer removal. This indicates that the convolutional residual blocks in PrimusV2 can independently learn a large number of effective features and maintain relatively high performance even without the Transformer, which contradicts the core logic of "Transformer-centric" architecture that requires "the Transformer to dominate representation learning".

### Questions
1.	Please refer to Weakness 1.
2.	Regarding computational cost: Have you compared the inference latency and memory usage of Primus/PrimusV2 with baseline models on standard clinical hardware to evaluate deployment feasibility?
3.	Please refer to Weakness 3.

### Soundness
3

### Presentation
4

### Contribution
3
