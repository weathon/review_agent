# DPMFormer: Dual-Path Mamba-Transformer for Efficient Image Super‑Resolution

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
Vision Transformers have achieved outstanding performance in image super-resolution (SR), but existing lightweight models rely on window-based attention, limiting their ability to model global dependencies essential for high-quality reconstruction. To address these challenges, we present DPMFormer, a Dual-Path Mamba–Transformer architecture for lightweight image super-resolution.  Rather than a simple combination of a state-space model and a Transformer, the design couples two streams throughout the network. On the Transformer side, an Enhanced Transformer Layer (ETL) replaces self-attention with Spatial–Channel Correlation (SCC) and a Depthwise-SwiGLU Feed-Forward (DW-SwiFFN). On the Mamba side, Lightweight Bi-directional Mamba Layers (LBi-ML) implement single-pass bidirectionality via channel split and sequence reversal with additive cross coupling. The streams interact at two levels: within each block, a Cross-Attention Layer (CAL) performs fixed, non-overlapping cross fusion,
and across blocks, Inter-branch Exchange Bridges (IEB) use resolution-preserving 1 × 1 adapters around tokenization to align channel spaces in both directions. Besides, we employ RMSNorm to reduce normalization overhead and, under our setup, observe modest, configuration-dependent gains. Extensive experiments show that DPMFormer reduces FLOPs by 47.3G (21\%) and parameters by 21K under 2 × upsampling compared to HiT-SR, while almost achieving state-of-the-art performance across five benchmarks. Measured on an RTX 4090, our method reaches 668 ms latency, yielding 1.59 × and 2.33 × speedups over MambaIR and CATANet, respectively. The code will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes DPMFormer, a dual-branch architecture that integrates Transformer and Mamba modules for efficient image super-resolution. The Transformer branch employs a Spatial–Channel Correlation attention and a depthwise SwiGLU feed-forward (DW-SwiFFN) to enhance local feature modeling, while the Mamba branch introduces a Lightweight Bidirectional Mamba (LBi-Mamba) for linear-time global dependency capture. The two branches interact via Cross-Attention Layers (CAL) within blocks and Inter-branch Exchange Bridges (IEB) across stages. Experiments on five benchmark datasets show that DPMFormer achieves competitive PSNR/SSIM with fewer parameters and FLOPs than prior lightweight SR models, such as HiT-SR and MambaIR.

### Strengths
1.Proposes a clearly motivated dual-path fusion leveraging complementary strengths of Mamba and Transformer.

2.Strong quantitative performance–efficiency trade-off, reducing FLOPs by ~20 % vs HiT-SR with comparable PSNR.

3.Extensive ablations (DW-SwiFFN, RMSNorm, IEB variants) demonstrate careful engineering and reproducibility.

4.Reproducibility statement is complete and code release is promised.

### Weaknesses
1.Innovation marginal: The dual-branch idea has been explored in prior hybrid SR models; more theoretical or analytic justification of the coupling design would enhance novelty.

2.Limited qualitative diversity: Most visual comparisons are standard; additional challenging scenes or real-world degradations would strengthen claims.

3.Missing complexity analysis: An explicit breakdown of runtime cost per module (ETL vs LBi-Mamba vs IEB) would help understand where efficiency gains arise.

4.Minor clarity issues: Equations (2)–(6) lack dimensional definitions; figure readability (font size) could be improved.

### Questions
1.How sensitive is the performance to the choice of window sizes (r) in ETL and CAL?

2.Could the proposed LBi-Mamba be applied to other low-level tasks (e.g., denoising, deblurring)?

3.Is the training stable when coupling both branches with IEB — any gradient conflict observed?

4.How does DPMFormer scale to higher resolutions (e.g., 4 K images) given linear Mamba dynamics?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work proposes a combination backbone (DPMFormer) of Mamba-Transformer for efficient SR. Generally, DPMFormer consists of several structural adaptations, including the mamba block, attention block, and FFN, to improve the modeling capability of multiple-range correlations. Overall, it brings improvement to a certain extent on the ESR tasks.

### Strengths
- Some visual results are good, and the overall results advance existing models to a certain extent.  
- The method contains multiple refinements and conducts multiple ablation studies to validate them.
- The paper is easy to follow.

### Weaknesses
- The DPMFormer offers barely new insight for the SR task or efficient backbone design. The key motivation of the model is still based on a combination of validated designs, such as the mamba block and information cross module, which have been well explored. The backbone of DPMFormer is rather bloated and complex, and lacks sound theoretical analysis. 
- For an efficient task, inference performance should be evaluated in multiple dimensions, like memory, activations, and run time on more practical mobile devices.  The comparison in manuscripts supports DPMFormer being a lightweight SR model, but the complicated design suggests that it is far from efficient, especially compared with a convolutional-based model.
- The improvements over existing methods are limited, only 0.0 dB, and all experiments are conducted on synthesized data, hardly proving its effectiveness on real-world applications.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces DPMFormer, a dual-path Mamba–Transformer hybrid designed for all-in-one image restoration. It combines a Dual-Path Mamba Block (DPMB)—one branch using Mamba for long-range dependency modeling and the other using a lightweight Transformer for local feature aggregation. A Path Interaction Unit (PIU) fuses global and local cues, while a Degradation-Aware Guidance Module (DGM) provides task conditioning via learned degradation priors. Experiments on multiple degradation benchmarks (rain, haze, low-light, noise) show improvements over several transformer and Mamba-based baselines

### Strengths
1. This paper proposes a hybrid Mamba–Transformer architecture, integrating sequence modeling and spatial self-attention. The dual-path structure is intuitively appealing for balancing long-range reasoning and local fidelity. Degradation-aware conditioning provides some adaptivity for mixed degradations.
2. Experiments cover common restoration tasks and compare with both transformer (Restormer, Uformer) and Mamba-based baselines. 
3. Ablation studies isolate the effect of each module (Mamba path, Transformer path, DGM).
4. Extends Mamba-based modeling into restoration, which remains relatively new.

### Weaknesses
1. Combining Mamba and Transformer paths is a logical but incremental step; there is little theoretical or architectural innovation beyond simple concatenation and gating.
2. The PIU fusion resembles standard cross-attention or gating mechanisms used in hybrid CNN–Transformer or Swin–MLP models.
3. The paper lacks rigorous analysis on why or when the Mamba path improves over pure Transformer designs. No detailed exploration of information flow or path synergy (e.g., attention entropy, frequency response, or token dependency visualization).
4. Reported gains are modest (≈0.2–0.4 dB PSNR) and often within noise margins. On several datasets, DPMFormer lags behind recent AIR systems (PromptIR, UniRestorer) in unseen or composite degradations.
5. All experiments are conducted on synthetic benchmarks; no evaluation on real-world degradation datasets or perceptual metrics (LPIPS, NIQE). Efficiency and scalability (especially GPU memory and throughput vs. Restormer or VMamba) are not reported.

### Questions
1. Could the authors provide FLOPs and throughput comparisons with Restormer and VMamba to justify efficiency claims?
2. What are the qualitative differences between features extracted by the Mamba path and Transformer path? (e.g., visualization or layer attention maps)
3. How does DPMFormer perform on real-capture datasets such as LOL-V2, RainDS, or SOTS-real?
4. Have the authors compared their design to Swin-Mamba or other existing Mamba–Transformer hybrids?
5. Does the DGM generalize to unseen degradation mixtures, or is it trained with supervision on specific degradation types?
6. How sensitive is performance to the relative weighting or depth of the two paths? Could a single-path Mamba or Transformer with the same parameter budget achieve comparable results?
7. The PSNR gains are small—can the authors include statistical significance or variance over multiple runs?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes DPMFormer, a dual-path architecture combining window-based Transformers and Mamba blocks for lightweight image super-resolution (SR). The authors introduce cross-attention layers (CAL) and inter-branch exchange bridges (IEB) to fuse local and global features. Experimental results show that  the proposed method achieves competitive performance on standard benchmarks.

### Strengths
1. Ablation studies validate the contribution of individual modules.
2. DPMFormer shows competitive PSNR/SSIM on standard datasets.

### Weaknesses
1. This paper claims to address the limitations of window-based attention mechanisms in terms of global feature dependency, yet directly replaces window-based attention with DW-SwiFFN during model design. This manner is inconsistent with the original intent of enhancing its global modeling ability.
2. The author states in the abstract that global modeling is essential for high-quality reconstruction, yet provides no supporting references or experimental evidence.
3. The paper emphasizes efficient image SR, but DPMFormer’s inference latency (668ms) is higher than CATANet (516ms) and significantly slower than efficient CNNs.
4.  For the ×4 SR task, DPMFormer and CATANet exhibit comparable performance, but the former requires nearly double the number of parameters, undermining its complexity advantage.

### Questions
1. No analysis is provided on memory usage, MACs, or deployment feasibility on edge devices, which is critical for efficient SR.
2. The narrative flow of the manuscript requires reorganizing to strengthen its motivation, and the layout must be refined and optimized to enhance overall readability.

### Soundness
3

### Presentation
2

### Contribution
2
