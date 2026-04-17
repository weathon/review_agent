---
job_id: bf62198a-7c99-4015-a25c-922c602a5192
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: pW6rFymZ8F.pdf
paper: EMBodiedMAE: A Unified 3D Multi-Modal Representation for Robot Manipulation
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a multi-modal masked autoencoder and large-scale 3D robot dataset for representation learning in manipulation, which fits directly within ICLR’s scope on representation learning, generative/self-supervised methods, and robotics.

## Minimum Quality
Pass ✅.  
The paper is in English and includes all major sections (Abstract, Introduction, Methodology, Experiments/Results, Related Work, Conclusion). The method is technically coherent, experiments are extensive with strong baselines, and there are no obvious fatal methodological or statistical flaws, although some aspects could be more rigorously analyzed.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, attempts to influence automated reviewers, or other manipulative content within the provided manuscript.

---

# Expected Review Outcome:

## Summary

The paper introduces **EmbodiedMAE**, a multi-modal masked autoencoder that jointly learns from RGB, depth, and point cloud inputs for robot manipulation. The model uses stochastic modality-wise masking with a shared ViT encoder and a cross-attention decoder, and is first trained at ViT-G scale on a new 3D robot dataset, **DROID-3D**, then distilled into smaller variants. Extensive experiments across 70 simulated tasks (LIBERO, MetaWorld) and 20 real-world tasks (SO100, xArm) show consistent gains over strong vision foundation model (VFM) baselines in both RGB-only and 3D (RGBD / point cloud) settings.

## Strengths

1. **Ambitious and comprehensive empirical evaluation.**  
   The paper evaluates EmbodiedMAE on:
   - 40 LIBERO tasks and 30 MetaWorld tasks, plus  
   - 10 SO100 and 10 xArm real-world tasks,  
   all with the same RDT-based policy backbone (Figure 5).  
   Learning curves in **Figure 6** on LIBERO show the Giant/ Large EmbodiedMAE variants dominating DINOv2, SPA, SigLIP, R3M, and VC-1 across all suites, both in final success rate and sample efficiency. **Table 1** further shows clear MetaWorld gains: e.g., in RGBD, EmbodiedMAE attains 76.2% average success versus 54.4% for DINOv2-RGBD, and in point cloud, 77.7% versus 65.8% for DP3. This breadth and consistency of results is a strong point.

2. **Well-motivated 3D data effort and clear qualitative evidence.**  
   Section 2.1 and **Figure 2** convincingly illustrate that depth in BridgeDataV2 and RH20T is noisy/low quality, and that AI-estimated depth (SPA’s CrocoV2-based pipeline) is temporally inconsistent on DROID. By contrast, their ZED-SDK processing yields temporally consistent and metrically calibrated depth. Constructing **DROID-3D** (76k trajectories, 350h) with synchronized RGB, depth, and point clouds on *the full DROID* is a substantial and practically valuable contribution for 3D embodied learning.

3. **Sensible and scalable multi-modal MAE design.**  
   The encoder uses modality-specific patchifiers (RGB/depth patches, DP3-style grouped point cloud tokens) feeding a shared ViT initialized from DINOv2. The masking strategy in Section 2.2 fixes the total number of unmasked tokens and allocates them across modalities using a symmetric Dirichlet over modality fractions \((\lambda_I,\lambda_D,\lambda_P)\). This is a clean extension of MultiMAE-style token budgeting to the embodied 3D setting and seems to work well in practice.

4. **Decoder and loss are conceptually sound and technically well grounded.**  
   Section 2.3 specifies a cross-attention decoder that explicitly fuses visible tokens from all modalities. The reconstruction loss in **Equation (1)** cleanly sums MSE terms for the masked RGB, depth, and point cloud targets, with \(g_I, g_D, g_P\) normalized in line with MAE best practices (He et al., 2022). This design connects well to prior MAE literature, while enabling cross-modal prediction.

5. **Careful distillation strategy and informative ablations.**  
   Section 2.4 describes a feature-alignment distillation loss (**Equation (2)**) matching teacher and student hidden states at bottom/middle/top layers using SmoothL1 with learnable linear projections. The combined objective in **Equation (3)** is simple and standard. **Table 4** provides ablations on masking ratio, alignment locations, and the weight \(\beta\), suggesting robustness to hyperparameters and highlighting the importance of top-layer alignment (performance drops from 92.4 to 74.4 when “Top” is removed).

6. **Evidence of genuine cross-modal understanding.**  
   **Figure 3** is a useful qualitative probe of the learned representation. Under extreme masking (columns 1–9), the model reconstructs missing modalities reasonably well from the remaining one; in the depth-to-RGB and RGB-to-depth translations (columns 10–11), the outputs preserve geometry but exhibit the expected color ambiguity. The re-coloring experiment (column 12), where editing one depth-to-RGB patch only affects the corresponding object while preserving others, suggests some emergent object-level segmentation / binding in the latent space.

7. **Real-world gains and realistic discussion of 3D modalities.**  
   **Figure 7**’s rollout sequences show failure modes of baselines (lost objects, collisions) vs EmbodiedMAE’s more accurate grasps and placements, concretely illustrating the “3D perception” claim. In **Figure 8**, EmbodiedMAE-RGBD and EmbodiedMAE-PC significantly outperform DINOv2-RGBD and DP3 on xArm, with large margins in Pick&Place and Pot tasks. The paper also candidly reports that point clouds are sensitive to sensor noise and often underperform depth-as-auxiliary, which is examined in-depth in Appendix B (Tables 9–10).

8. **Practicality and usability.**  
   Section 2.5 and **Figure 4** show a HF-style API, and **Table 13** provides detailed latency numbers across model sizes, modalities, sequence lengths, and fp32/bf16. The Small and Base models have \(\approx 16{-}18\) ms per forward pass even with 3 modalities, which is useful information for robotics deployment.

## Weaknesses

1. **Novelty and positioning relative to prior multi-modal / 3D MAE work are underdeveloped.**  
   Conceptually, EmbodiedMAE is quite close to MultiMAE (Bachmann et al., 2022), extended to RGB+depth+point cloud and applied to DROID-3D. The paper rightly cites MultiMAE but does not sufficiently analyze how its contributions differ structurally beyond (i) the specific modalities, (ii) use of DINOv2 initialization, and (iii) embodied data. Moreover, several directly relevant works are not cited (see “Potentially Missing Related Work”), including 3D-MVP (3D multi-view MAE pretraining for manipulation) and multi-view masked world models, which already explore multi-view/multi-modal masked reconstruction for robot control. Without a more explicit comparison and discussion, it is difficult to gauge how much of the gains are due to the new architecture vs. just higher-quality 3D data and scale.

2. **Limited ablations at the *pretraining* level.**  
   The only core architecture ablations shown are in distillation (Table 4). There is no study of:
   - The effect of the Dirichlet concentration parameter \(\alpha\) in the masking distribution (Section 2.2). Since this directly controls the range of single-modality vs. multi-modality prediction regimes, it may have a strong impact on learned fusion.
   - The importance of the cross-attention decoder vs. a simpler per-modality decoder or concatenation-only design.
   - The benefit of pretraining on all three modalities versus subsets (e.g., only RGB+depth, only RGB+PC, or purely RGB MAE on DROID-3D).  
   As a result, it is hard to attribute performance gains. For example, is the large improvement of EmbodiedMAE-RGB over DINOv2 on LIBERO in **Figure 6** due mostly to domain-specific pretraining on DROID-3D (even with RGB only), or to the multi-modal masked learning objective?

3. **Mathematical specification of encoder–decoder interactions is incomplete and somewhat inconsistent.**  
   - In Section 2.3 and **Equation (1)**, the notation mixes \(g(h_I,h)\), \(g(h_D,h)\), \(g(h_P,h)\) with \(g_I, g_D, g_P\) in the text; the formula explicitly uses \(g\) with subscripts in the underbraces but not in the argument list, which is confusing. It would be clearer to define \(g_I(h_I,h)\), \(g_D(h_D,h)\), \(g_P(h_P,h)\) formally and use those consistently in the loss:
     \[
     \mathcal L_{\mathrm{MAE}} = \mathbb E\left[ \| \hat I_2 - I_2\|_2^2 + \| \hat D_2 - D_2\|_2^2 + \| \hat P_2 - P_2\|_2^2 \right],
     \]
     where \(\hat I_2 = g_I(h_I,h)\) etc.  
   - The decoder description says queries are “visible tokens concatenated with [MASK] tokens” and keys/values are “all visible patches,” but the exact cross-attention structure is not formalized: for instance, does each modality’s [MASK] attend to visible tokens of *all* modalities only, or also to its own unmasked tokens? Are there modality-specific positional encodings in the decoder? Since cross-modal fusion is core to the claimed contribution, more precise notation (e.g., defining query sequence \(Q_m\), key/value sequences \(K,V\), and attention blocks explicitly) would be helpful.
   - For the point-cloud encoder in Section 2.2, the paper specifies FPS with \(N\) centers and KNN with \(K\) neighbors, but never provides the actual \(N,K\) values used in experiments, nor the resulting token count \(L=N\). Given the ViT complexity scales as \(O(L^2)\), this matters when comparing RGB-only vs. 3D variants.

4. **Characterization of DROID-3D depth/point quality is largely qualitative.**  
   Section 2.1 and **Figure 2** show qualitative comparisons (“Low-quality”, “Inconsistent”, “Consistent w/ High-Quality”), but there is no quantitative evaluation of depth accuracy or temporal consistency (e.g., reprojection errors, geometric consistency metrics, or statistics over stereo baselines). The text states “hardware-calibrated metric depth” and “temporal fusion to reduce noise”, but does not detail:
   - Calibration procedures or how potential miscalibration across robots/scenes is handled.
   - Whether any failure cases (e.g., reflective surfaces, low-texture areas) remain and how frequent they are.  
   Given the strong emphasis on data quality as a central motivation, at least some quantitative checks (even for a small subset) would strengthen the scientific claim that DROID-3D’s depth/PCs are “high-quality” in a way that explains downstream performance.

5. **Policy evaluation lacks uncertainty quantification and some fairness details.**  
   - LIBERO and MetaWorld experiments report average success rates (Table 1, Figure 6), but there are no standard deviations or confidence intervals across seeds or runs, despite relatively modest numbers of trials (50 per task per seed; 3 seeds). This makes it hard to judge statistical significance of margins like the equal 73.0% average for SPA and EmbodiedMAE-RGB on MetaWorld in Table 1.
   - Real-world tasks on SO100 and xArm use 10 trials per task. **Figure 8** shows substantial gaps (e.g., xArm Pick&Place ~94% vs ~73% for DP3), but again, no variances; additionally, it is not fully clear whether all methods see identical demonstrations and hyperparameters for each modality setting. A clearer description of how baselines are tuned and whether data budgets are identical for each representation would increase trust in the comparisons, particularly because some baselines (e.g. DP3, SPA) are not native to the RDT policy architecture.

6. **Missing or under-discussed related work in multi-view/multi-modal pretraining for manipulation.**  
   The Related Work section is heavy on generic VFMs and 3D perception but light on prior *robotic* masked pretraining that uses multiple views or modalities. For instance, 3D-MVP (3D Multiview Pretraining for Robotic Manipulation) and Multi-View Masked World Models both study using multi-view masked reconstruction to learn manipulation-relevant features. These are arguably more directly comparable than, say, point-cloud foundation models trained on indoor scenes, yet they are neither cited nor contrasted. This weakens the positioning by making EmbodiedMAE look more unique than it is. I discuss these works explicitly in the next section.

7. **Some experimental design choices need further clarification.**  
   - Section 3.1 states that a “scaled-down RDT” is used as the policy backbone across all baselines, which is good for fairness, but does not specify whether vision encoders are frozen or fine-tuned, nor whether different encoders receive different learning-rate multipliers or normalization. This matters since some VFMs (e.g., SigLIP, DINOv2) are very sensitive to fine-tuning regimes.  
   - The paper mentions that unlike prior work it does not filter out unsuccessful demonstrations in LIBERO (Appendix A.2), which changes the dataset distribution. It is unclear whether baselines’ originally reported numbers are still comparable under this more challenging setting, or if all numbers are retrained from scratch under the new protocol. The main text hints the latter, but this should be explicitly stated near Figures 6 and 8.

Overall, while none of these issues are fatal, several are important for properly attributing where the improvements come from and for situating the work relative to existing multi-modal pretraining methods.

## Potentially Missing Related Work

1. **Qian et al., “3D-MVP: 3D Multiview Pretraining for Robotic Manipulation”, 2024.**  
   - Directly relevant: uses multi-view masked autoencoding on 3D inputs (multiple RGB views, depth) specifically for robot manipulation, with pretraining and downstream policy evaluation.  
   - How/where to add: should be discussed in **Section 4 (Related Works)** as a closely related robotic pretraining method; also merits comparison in **Section 3.3** as a baseline or at least conceptual comparison (e.g., EmbodiedMAE vs multiview-only pretraining).

2. **Seo et al., “Multi-View Masked World Models for Visual Robotic Manipulation”, 2023.**  
   - Directly relevant: proposes a multi-view masked world model architecture for visuomotor control, again very similar in spirit to using masked prediction as a pretraining signal for manipulation.  
   - How/where to add: should be cited in the discussion of masked autoencoding for robot learning in **Section 4**, and contrasted with EmbodiedMAE’s multi-modal (RGB/depth/PC) setup and DROID-3D-scale training.

3. **Yang et al., “CMViM: Contrastive Masked Vim Autoencoder for 3D Multi-modal Representation Learning for AD Classification”, 2024.**  
   - Relevance: combines masked autoencoding with contrastive learning over 3D multi-modal medical data. While the application domain is different, the methodology for 3D multi-modal MAE is conceptually close.  
   - How/where to add: should be mentioned in **Section 4** as an example of multi-modal MAE beyond robotics, to better contextualize EmbodiedMAE in the broader multi-modal representation literature.

## Questions

1. **Pretraining ablations and role of multi-modality.**  
   Could the authors provide results for:
   - A purely RGB MAE trained on DROID-3D with the same ViT backbone and masking schedule,  
   - And a “single-modality” variant of EmbodiedMAE pretraining (e.g., only RGB+depth or only RGB+PC)?  
   This would clarify how much EmbodiedMAE’s gains are due to multi-modal masked learning versus simply having a large, domain-matched MAE pretraining on DROID-3D.

2. **Dirichlet masking hyperparameter.**  
   What value of \(\alpha\) is used for the Dirichlet in Section 2.2, and did you experiment with different \(\alpha\) values? If yes, could you share at least a small ablation (even on a subset of tasks) to illustrate whether concentrating on single-modality reconstruction versus balanced multi-modal reconstruction meaningfully affects downstream policy performance?

3. **Decoder specification and implementation details.**  
   Could you provide a more formal description of the cross-attention decoder, including:
   - How queries, keys, and values are built across modalities (e.g., a precise equation for the attention block, indicating whether queries from masked tokens of modality \(m\) attend to visible tokens of all modes or just a subset),  
   - Whether decoder positional and modality encodings are distinct from those in the encoder, and  
   - How the normalization of \(g_I, g_D, g_P\) referenced after **Equation (1)** is implemented (e.g., unit \(\ell_2\) per patch, or per-channel normalization)?  
   This would remove current ambiguities around the core fusion mechanism.

4. **Fine-tuning regime for VFMs in policy training.**  
   Are the visual encoders frozen during policy learning, partially fine-tuned, or fully fine-tuned? Are the settings identical across all methods (e.g., same LR, weight decay, and number of trainable layers)? Please clarify in Section 3.1 and, if they differ, justify why the protocol is still fair.

5. **Quantitative validation of DROID-3D depth quality.**  
   Have you computed any quantitative metrics comparing ZED-SDK depth against CrocoV2 or native depth (e.g., stereo reprojection error, depth variance in static scenes, or alignment with known geometry)? Even a limited analysis on a subset would underpin Figure 2’s qualitative claims and help others judge whether DROID-3D truly provides significantly higher-quality 3D supervision.

6. **Reproducibility of real-world setups.**  
   For SO100 and xArm experiments, can you detail:
   - Whether the scene layouts and object positions are randomized across trials in a controlled way,  
   - How failures are defined for tasks like “wipe” and “unfold,” and  
   - Whether the same demonstration data and policy hyperparameters are used for all encoders in each setting?  
   These details would help other groups reproduce the real-world results.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The paper uses existing robot datasets and standard simulation benchmarks, and does not raise evident ethics issues (no new data collection from humans, no sensitive content, no obviously risky applications).

## Soundness Rating

3: good.  
The methodological components (multi-modal MAE, Dirichlet masking, distillation) are standard and coherently combined; experiments are broad and generally well executed. Some aspects (quantitative validation of DROID-3D, pretraining ablations, precise decoder formulation) could be stronger, but they do not appear to undermine the main empirical claims.

## Presentation Rating

3: good.  
The paper is mostly clear, with useful figures (especially Figures 2, 3, 6, 7, 8) and detailed experimental appendices. However, some notational inconsistencies (Equation 1) and underspecified architectural details (decoder cross-attention, point-cloud tokenization hyperparameters) detract from full clarity.

## Contribution Rating

3: good.  
The work meaningfully advances 3D multi-modal pretraining for embodied control by (i) providing a well-constructed DROID-3D dataset, (ii) proposing a practical multi-modal MAE architecture with large-scale training and distillation, and (iii) demonstrating strong gains on a wide suite of manipulation tasks. Conceptual novelty is moderate, but the empirical and practical contributions are significant.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper delivers a solid multi-modal MAE for robotics, backed by extensive experiments and a valuable 3D dataset. While the core ideas build fairly directly on existing MAE work and the positioning vs. closely related multi-view/multi-modal pretraining papers is incomplete, the combination of thorough evaluation, practical design, and resource contribution makes this a worthwhile addition to the ICLR program.

## Reviewer Confidence

4: confident.  
I am familiar with MAE-style vision pretraining, 3D perception, and embodied control, and I carefully checked the core equations and experimental sections. Some implementation details are still unclear, so there is room for the authors’ rebuttal to adjust my view, but major misunderstandings on my side are unlikely.