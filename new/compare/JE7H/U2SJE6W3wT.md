---
job_id: 27abd439-8193-47fb-99c2-4c567e7ab238
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: U2SJE6W3wT.pdf
paper: Improved Adversarial Diffusion Compression for Real-World Video Super-Resolution
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a compressed diffusion-based model for real-world video super‑resolution, with architectural and training innovations in generative modeling, representation learning, and adversarial distillation, which fits squarely within ICLR’s core topics.

## Minimum Quality
Pass ✅.  
All major sections (Abstract, Introduction, Related Work, Method, Experiments, Conclusion) are present and reasonably detailed. The work is technically sound overall, with substantial experiments, nontrivial modeling choices, and clear positioning within diffusion-based Real‑VSR. No fatal methodological or theoretical flaws are evident that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
No signs of prompt injection, hidden text, or other manipulative content targeting automated reviewers are present in the paper.

---

# Expected Review Outcome:

## Summary

The paper proposes AdcVSR, a compressed one‑step diffusion‑GAN hybrid for real‑world video super‑resolution that distills a heavy 3D DiT‑based teacher (DOVE) into a compact student. The student uses a 2D Stable Diffusion UNet backbone (as in AdcSR) augmented with lightweight 1D temporal convolutions (“2D + 1D”), and is trained with a dual‑head, dual‑domain adversarial distillation scheme that disentangles spatial detail and temporal consistency objectives. Experiments on multiple synthetic and real‑world benchmarks show that AdcVSR achieves competitive perceptual quality and temporal consistency while reducing parameters by about 95% and speeding up inference by 8× compared to DOVE.

## Strengths

1. **Clear and useful problem setting: compressing heavy Real‑VSR diffusion models.**  
   The paper targets a very practical and relevant problem: current one‑step diffusion‑based Real‑VSR models (SeedVR2, DOVE, DLoRAL) are still prohibitively large and slow for many applications. Focusing on compressing a strong DiT teacher into a significantly smaller student is both well‑motivated and clearly articulated in Sec. 1 and Fig. 1.

2. **Simple but effective “2D + 1D” architecture validated empirically.**  
   The core architectural idea is to reuse an aggressively pruned 2D SD2.1 backbone (AdcSR) and inject 1D temporal convolutions after each UNet block to restore temporal awareness without expensive 3D attention (Sec. 3.2, Fig. 2a). The ablations in **Table 2** and **Table 5** are quite convincing:  
   - On UDM10, the “2D only” AdcSR variant has much worse DISTS (0.2418) and warping error \(E^*_{\text{warp}}=4.43\) than both the pruned 3D DiT and the proposed 2D+1D student, clearly showing that 2D alone cannot learn the temporal behavior of a 3D teacher.  
   - The 2D+1D design nearly matches the pruned 3D model’s DISTS (0.2112 vs 0.2098) but *improves* warping error (1.67 vs 2.53) with only 0.55B parameters vs 8.36B.  
   - **Figure 5** qualitatively supports this: the 2D student exhibits visible temporal flicker in high‑frequency regions of buildings, while the 2D+1D model restores smooth temporal profiles while keeping textures sharp.  
   This substantiates the claim that simple 1D temporal convolutions are sufficient to reclaim most of the temporal modeling needed for Real‑VSR in this setup.

3. **Carefully designed dual‑head, dual‑domain discriminator with meaningful supervision semantics.**  
   The adversarial distillation scheme is the most interesting conceptual contribution. Instead of a single discriminator signal, each discriminator (pixel and feature domain) has two heads that separately target “detail” and “consistency”, and these are trained with five types of curated data combinations and frame shufflings (Sec. 3.3, Fig. 2b).  
   - **Equations (2)–(3)** clearly integrate adversarial terms \(\text{Softplus}(-\mathcal{D}_{\text{pixel}}(\cdot))\) and \(\text{Softplus}(-\mathcal{D}_{\text{feature}}(\cdot))\) on top of L1 and DISTS / feature losses, so the GAN component is not a bolt‑on but properly incorporated into the objective.  
   - **Equations (4)–(5)** define a multi‑attribute label set \(y_d, y_c \in \{-1,0,1\}\) and the dataset \(\mathcal{S}\) that includes student outputs, real videos, shuffled videos, repeated images, and random image sequences. This gives a clear semantics: shuffled videos penalize inconsistency, repeated images enforce perfect consistency and fine detail, and random image sequences enforce detail realism but penalize inconsistency.  
   - The ablation in **Table 3** shows that the dual‑head, dual‑domain design improves both CLIPIQA and warping error compared to single‑head or single‑domain variants, indicating that the detail-consistency disentangling is not cosmetic.

4. **Strong efficiency vs. quality trade‑off, with thorough quantitative comparison.**  
   **Table 1** and **Figure 4** together provide a convincing picture that AdcVSR hits a very favorable sweet spot:  
   - It uses only 0.57B parameters and 0.55s latency for a 25×512×512 video, versus DOVE’s 10.55B and 4.42s, while retaining fairly close quality on most metrics.  
   - It attains the *lowest* warping error \(E^*_{\text{warp}}\) on both UDM10 (1.67) and VideoLQ (6.74), suggesting very strong temporal consistency compared to both Real‑VSR and Real‑ISR competitors.  
   - It is dramatically more efficient than multi‑step diffusion baselines like Upscale‑A‑Video, MGLD‑VSR, and STAR (order‑of‑magnitude speedups and large parameter reductions).  
   - **Figure 4** visualizes this trade‑off effectively: AdcVSR is at the top‑left of the curve (low warping error, low inference time) with one of the smallest bubbles, making the narrative about “compressing a 3D DiT teacher into a small yet competitive Real‑VSR” very concrete.

5. **Extensive experiments and non‑trivial ablations.**  
   The experimental section is fairly comprehensive:  
   - Multiple synthetic and real‑world datasets (UDM10, SPMCS, YouHQ40, RealVSR, MVSR4x, VideoLQ), with both fidelity (PSNR, SSIM) and perceptual/no‑reference metrics (LPIPS, DISTS, MANIQA, CLIPIQA, MUSIQ, DOVER) plus explicit temporal metrics \(E^*_{\text{warp}}\).  
   - Ablations on architecture (2D vs 3D vs 2D+1D in **Tables 2 & 5**, **Fig. 6**), discriminator variants (**Table 3, Table 6**), distillation setups with and without adversarial learning and with different teachers (**Table 4**), and curated data usage in \(\mathcal{S}\) for discriminator training (**Table 7**).  
   - **Figures 3 and 8** provide qualitative comparisons that clearly show the trade‑offs: Real‑ISR baselines (PiSA‑SR, AdcSR, HYPIR) are sharp but flickery and temporally unstable (as seen in temporal profiles), while AdcVSR produces comparably sharp details but cleaner and more stable temporal stripes.

6. **Good clarity and reasonable reproducibility.**  
   The method description, including equations, loss definitions, and training algorithm (Algorithm 1 in the appendix), is reasonably clear. Implementation details in Sec. 4.1 specify pruning ratios, kernel sizes, learning rates, loss weights, batch size, training schedule, and hardware. This is not enough to re‑implement blindly, but it is well above the bar of “vague description”.

## Weaknesses

1. **Limited conceptual novelty beyond combining known components, with relatively incremental technical depth.**  
   While the paper is practically compelling, most of the core ideas are recombinations or straightforward extensions of existing work:  
   - The 2D backbone and ADC framework are taken directly from AdcSR, with the main architectural change being the insertion of temporal 1D convs after each block. This is a very natural extension once one decides to adapt an image‑SR compressor to video. The paper does not explore more interesting design space (e.g., causal vs bidirectional 1D convs, shared vs stage‑specific parameters, adaptive temporal receptive fields).  
   - The adversarial distillation is conceptually similar to previous adversarial diffusion distillation works (ADD, ADD‑SR, etc.), with the main twist being two heads and some heuristic label design. While this is a useful engineering insight, it is not particularly deep theoretically, and the justification is largely empirical.  
   The paper does a good job arguing *why* these combinations matter, but from a representation learning / generative modeling perspective, the advance feels more like careful system engineering than a fundamentally new modeling principle.

2. **The discriminator formulation in Eq. (4) is mathematically odd and under‑explained, especially for unlabeled (0) samples.**  
   In **Equation (4)**, the discriminator loss is  
   \[
   \mathcal{L}_{\text{disc}}=\sum_{(\mathbf{s},y_d,y_c)\in\mathcal{S}}\left[\text{Softplus}(-y_d[\mathcal{D}(\mathbf{s})]_d)+\text{Softplus}(-y_c[\mathcal{D}(\mathbf{s})]_c)\right],
   \]  
   with labels \(y_d,y_c\in\{-1,0,1\}\). When a head is “unlabeled” (0), this reduces to \(\text{Softplus}(0)=\log 2\), a constant independent of discriminator parameters. So those terms contribute a constant offset and effectively drop out of the gradient.  
   - This is mathematically fine but means the 0 label is a *no‑op*. The text (Sec. 3.3, Page 6) states: “real video details are left unlabeled for the detail head” and describes 0 as an “unlabeled” label, but it never explicitly explains that such samples simply do not affect that head’s training at all. Given the somewhat complex construction of \(\mathcal{S}\) in Eq. (5), the reader is left to infer this behavior.  
   - More troubling, the notation \(\mathcal{D}_{\text{pixel}}(\mathbf{x})\) and \(\mathcal{D}_{\text{feature}}(\mathbf{f})\) is later treated as a 2‑dimensional vector (detail and consistency heads), but in **Equations (2)–(3)** the same \(\mathcal{D}_{\text{pixel}}\) and \(\mathcal{D}_{\text{feature}}\) appear without specifying which head is used in the generator loss. From context it seems both heads contribute, but the exact aggregation (sum, average, weighted) is not spelled out.  
   These issues do not appear fatal, but they make the theory presentation and the precise optimization objective needlessly opaque and should be clarified.

3. **Teacher‑student distillation objective is heavy on heuristics; missing more principled analysis of error vs adversarial terms.**  
   The generator objective in **Equations (1)–(3)** mixes L1 in pixel and feature space, DISTS, and two adversarial terms weighted by \(\lambda_{\text{pixel}}, \lambda_{\text{feature}}, \lambda_{\text{adv}}\). The paper asserts that pure regression is too strict for a small student and that adversarial learning “relaxes” the requirement. However:  
   - There is no analysis of how sensitive performance is to the choice of \(\lambda\)s, nor any explanation of why \(\lambda_{\text{pixel}}=0.1\) and \(\lambda_{\text{feature}}=1.0\) is a good trade‑off.  
   - It is not discussed whether the student is expected to match the *teacher’s* distribution or the *real data* distribution when teacher and real differ (and they do, as shown by ablations in **Table 7** using teacher videos as consistency reference).  
   - There is no ablation of “dual‑head discriminator but *no* pixel/feature L1 to teacher” to verify that the teacher’s supervision is actually necessary, versus purely real‑data GAN training with the same architecture.  
   This leaves some uncertainty about how much of the gains come from the architecture vs. from the specific mix of losses, and it weakens the claim that the approach is a principled “adversarial diffusion compression” as opposed to a tuned cocktail.

4. **Positioning vs. closely related Real‑VSR diffusion and compression work is incomplete.**  
   The paper discusses many Real‑VSR and one‑step diffusion works, but omits several highly relevant recent efforts on efficient or streaming diffusion‑based VSR and temporal consistency:  
   - FlashVSR (Zhuang et al., 2025) introduces a real‑time diffusion‑based streaming VSR framework with a strong focus on efficiency and scalability, which is very close in spirit to this paper’s goal of compressing diffusion for fast VSR. It should be explicitly compared and discussed in Sec. 2 and Sec. 4.2 in terms of latency, parameter count, and consistency trade‑offs.  
   - InstaVSR (Hu et al., 2026) targets efficient and temporally consistent diffusion‑based VSR with lightweight designs. This is directly relevant to both the 2D+1D architecture and the temporal consistency discussion and should be covered.  
   - An et al. (2025), “Spatial Degradation‑Aware and Temporal Consistent Diffusion Model for Compressed Video SR”, also focuses on explicit temporal consistency modeling within a diffusion framework. It is missing from the related work and should be cited and contrasted, especially in Sec. 3.1 where the “conflict between details and consistency” is discussed.  
   These omissions weaken the positioning and make it harder to gauge how competitive or conceptually different AdcVSR is relative to other recent efficient diffusion‑VSR designs.

5. **Evaluation scope is narrow on some important dimensions (long videos, higher resolutions, streaming, robustness).**  
   All experiments restrict to 25‑frame clips at 512×512 resolution, both for training and testing. This matches DOVE’s setup but raises questions:  
   - How does AdcVSR behave on longer sequences (e.g., 100–300 frames)? Are the 1D temporal convolutions applied in a sliding‑window fashion or over the full sequence, and does consistency degrade with increasing temporal distance?  
   - How does the model scale to higher resolutions (e.g., 1080p or 4K)? Given that a key selling point is efficiency, it would be useful to see runtime/quality scaling curves to verify that the compression benefits persist.  
   - The design does not appear causal; 1D convs operate over the full temporal dimension of the current clip. This makes AdcVSR unsuitable for truly streaming / online scenarios, which is an increasingly important use case for VSR. The paper does not discuss this limitation.  
   These are not fatal issues but they limit the practical takeaway for deployment scenarios beyond the specific 25×512×512 regime.

6. **Qualitative analyses highlight advantages but are somewhat anecdotal and lack user studies.**  
   **Figures 3 and 8** show compelling examples where AdcVSR yields sharper and more temporally stable results than others, especially Real‑ISR baselines. However:  
   - The paper repeatedly argues that no‑reference perceptual metrics (MANIQA, CLIPIQA, MUSIQ, DOVER) align “better with human perception” than PSNR, but does not provide any human study or preference test to back this for the specific model class and data considered.  
   - There is no systematic qualitative analysis of failure modes: where does AdcVSR lose to its teacher in detail or consistency, or when does it hallucinate obviously incorrect textures? The Limitations section briefly mentions this in generic terms (foliage, water, glass) but without concrete examples.  
   Given that the main selling point is perceptual quality and consistency, some more careful qualitative evaluation (even a small user study or a curated failure case gallery) would strengthen the case.

7. **Some training details and dataset mixing choices are unclear.**  
   While Sec. 4.1 provides many hyperparameters, there are still gaps that matter for reproducibility and understanding:  
   - The relative sampling of the five data types for discriminator training (student outputs, real videos, shuffled videos, repeated images, random image sequences) is not specified. Are they all equally likely, or is there a curriculum? This can have a strong effect on the balance between detail and consistency signals.  
   - It is not entirely clear whether OpenVid‑1M videos and LSDIR images share similar content/distribution with the Real‑VSR synthetic training pipeline. There is no discussion of possible domain mismatch, nor of whether using Real‑ESRGAN‑style degradation pipelines on real videos would further help robustness.  
   - The training time is reported as “about one day on 8×H20”, which is commendably efficient, but batch size and LR scheduling for the discriminators vs generator are only sketched. Precise settings (e.g., which optimizer betas, weight decay) would help.

8. **Compression claims would be stronger with more systematic comparisons to other compression strategies.**  
   The main competitor in the “compression” space is AdcSR and TinySR / PassionSR are only mentioned in related work (Sec. 2) as ISR methods. There is no attempt to adapt any of these ISR compression methods to videos (even in a naive per‑frame way plus a simple temporal smoothing), which would be a more direct competitor to AdcVSR’s proposed improved ADC. A small experiment comparing:  
   - “AdcSR + simple temporal smoothing / optical‑flow consistency regularization”  
   - “TinySR / PassionSR + such temporal fix”  
   would help clarify how much of the gain truly comes from the proposed dual‑head adversarial Real‑VSR‑specific training versus just adding temporal regularization on top of a compressed Real‑ISR model.

Overall, the paper is technically solid, empirically strong, and practically important, but its core contributions are more engineering‑oriented than conceptual, and some methodological choices (especially in the discriminator and loss design) would benefit from deeper explanation and analysis.

## Potentially Missing Related Work

1. **Zhuang et al., “FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution”, 2025.**  
   - Directly relevant as a diffusion-based framework focusing on real-time and streaming VSR, with similar efficiency and scalability goals.  
   - Should be cited and discussed in Sec. 2 (Real‑VSR / One-Step Diffusion) and compared in Sec. 4.2 at least qualitatively, including efficiency vs. quality trade‑offs and whether FlashVSR uses any temporal consistency mechanisms comparable to the proposed 2D+1D design.

2. **Hu et al., “InstaVSR: Taming Diffusion for Efficient and Temporally Consistent Video Super-Resolution”, 2026.**  
   - Proposes a lightweight diffusion architecture targeting both efficiency and temporal consistency, closely aligned with AdcVSR’s objectives.  
   - Should be added in the related work discussion on efficient one‑step diffusion Real‑VSR (Sec. 2) and mentioned when contrasting AdcVSR’s 2D+1D design and dual‑head distillation in Sec. 3.1–3.3.

3. **An et al., “Spatial Degradation-Aware and Temporal Consistent Diffusion Model for Compressed Video Super-Resolution”, 2025.**  
   - Focuses on temporally consistent diffusion for compressed video SR, which is another form of real‑world degradation. The paper’s claim that detail vs. consistency objectives are competing (Sec. 3.1) would be stronger if it connected with this work’s approach to handling temporal coherence in diffusion.  
   - Should be cited in Sec. 2 (Real‑VSR and One-Step Diffusion) and referenced in Sec. 3.1 when discussing prior work on the conflict between detail and temporal consistency.

4. **Zhang et al., “Q-Insight: Understanding Image Quality via Visual Reinforcement Learning”, 2025.**  
   - Proposes an IQA method based on visual RL, potentially complementary to the IQA metrics (MANIQA, CLIPIQA, MUSIQ) used in this paper.  
   - Could be mentioned in Sec. 4.1–4.2 when discussing evaluation metrics, as a possible alternative or future metric for quality assessment of SR outputs.

5. **Li et al., “Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment”, 2026.**  
   - Further develops RL-based IQA, which may provide more nuanced assessment of perceptual quality than current metrics.  
   - Similarly, could be discussed in the evaluation section as a direction for richer quality evaluation, especially for video.

(Other listed works like FakeShield are only tangentially related and need not be cited.)

## Questions

1. **Clarification on which discriminator heads are used in the generator loss (Eqs. (2)–(3)).**  
   In \(\mathcal{L}_{\text{pixel}}\) and \(\mathcal{L}_{\text{feature}}\), \(\mathcal{D}_{\text{pixel}}\) and \(\mathcal{D}_{\text{feature}}\) are multi‑head discriminators, but the equations treat their outputs as scalars. Do you use the *sum* of the detail and consistency heads, an average, or only one head for the generator adversarial term? Clarifying this is important to interpret the role of each head and how gradients are distributed.

2. **On the semantics and impact of the 0 labels in Eq. (4).**  
   Since \(\text{Softplus}(-0\cdot D)=\log 2\) is constant, unlabeled samples do not contribute gradients for that head. Could you confirm that this is intended, and comment on whether you tried weighted soft labels instead (e.g., 0.5) to weakly supervise detail or consistency for those data types? Some insight or an ablation here would strengthen the motivation for the current label scheme.

3. **Details of curated data sampling for discriminator training.**  
   How are the five data types (student outputs, real videos, shuffled videos, repeated images, random image sequences) mixed in each discriminator update? Are batch proportions fixed or adaptive? Did you observe training instabilities when changing this mixture, and if so, what schedule worked best?

4. **Behavior on long sequences and higher resolutions.**  
   Have you tested AdcVSR on sequences longer than 25 frames and on resolutions higher than 512×512 (e.g., 1080p, 4K)? If so, can you provide preliminary numbers or qualitative insights (even in the rebuttal) on whether warping error and perceptual quality degrade significantly? If not, can you comment on whether the 1D temporal convs are applied over entire sequences or in a sliding window and how you would expect the model to scale?

5. **Streaming / causal usage.**  
   Are the 1D temporal convolutions implemented causally (only past frames) or with symmetric kernels across time? If they are non‑causal, do you foresee challenges or modifications needed to deploy AdcVSR in real‑time streaming or low‑latency settings? A brief discussion of possible causal variants would be helpful.

6. **Sensitivity to loss weights and teacher choice.**  
   **Table 4** provides some evidence on teacher choice but not on the relative importance of pixel vs feature vs adversarial losses. Did you experiment with e.g. setting \(\lambda_{\text{pixel}}=1\) or reducing \(\lambda_{\text{feature}}\)? How robust is performance to these changes? In addition, could a stronger Real‑VSR teacher (if available) further improve performance, or do you observe diminishing returns due to the student capacity limit?

Addressing these questions and clarifications in the rebuttal would increase my confidence in the method and may positively influence my assessment.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The method is technically consistent and the experiments are extensive and reasonably well‑controlled. Some parts of the loss formulation (multi‑head GAN objective, unlabeled labels) are under‑explained but not obviously wrong, and the empirical validation is strong.

## Presentation Rating

3: good.  
The paper is generally well written, with clear figures (especially Figs. 1–4) and tables (1–4, 5–8). Some mathematical notation around the discriminators and generator adversarial losses could be clarified, and related work is missing a few closely related recent papers.

## Contribution Rating

3: good.  
The work makes a solid contribution by showing that a heavily compressed 2D+1D diffusion‑GAN hybrid can effectively inherit a 3D DiT teacher’s Real‑VSR capability with large efficiency gains. The core ideas are more engineering‑oriented than conceptually deep, but they are non‑trivial and the results are practically valuable for the community.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper presents a well‑executed and practically meaningful compression approach for diffusion‑based Real‑VSR, with strong empirical evidence and a thoughtful dual‑head adversarial scheme. While the conceptual novelty is moderate and some design choices are heuristic, the methodological soundness and experimental thoroughness justify a positive recommendation as a solid poster‑level contribution.

## Reviewer Confidence

4: confident.  
I am familiar with diffusion‑based SR and video restoration, and I carefully checked the loss formulations, discriminators, and key experimental comparisons. Some implementation details remain unspecified, but I am reasonably confident in my assessment of the paper’s contributions and limitations.