=== CALIBRATION EXAMPLE 59 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately conveys the paper’s focus on an embodied 3D multi-modal masked autoencoder for robot manipulation.
- The abstract clearly states the problem, the dataset contribution (DROID-3D), the model family, and the claimed evaluation scope.
- However, the abstract makes very strong claims that are only partially substantiated in the paper as presented:
  - “consistently outperforms state-of-the-art vision foundation models” across “70 simulation tasks and 20 real-world robot manipulation tasks” is a broad statement, but the experimental section does not always make clear whether comparisons are controlled fairly across all modalities and baselines, especially for point cloud and RGBD settings.
  - “reliable unified 3D multi-modal VFM” is a large claim for a paper whose evidence is mostly downstream policy performance; reliability for broader embodied AI settings is not yet established.
- The abstract does a good job of signaling the contribution, but it somewhat overstates the generality of the conclusions relative to the evidence provided.

### Introduction & Motivation
- The motivation is strong and well aligned with ICLR interest: representation learning for embodied AI, domain-specific 3D perception, and scaling data/model design for robot manipulation.
- The paper clearly identifies two gaps: lack of suitable 3D pretraining data for tabletop manipulation, and lack of effective architectures for multi-modal 3D perception.
- The introduction states contributions clearly, especially DROID-3D and EmbodiedMAE.
- That said, the introduction somewhat blurs the line between “representation learning” and “policy improvement.” The paper is framed as a general 3D VFM contribution, but the empirical evidence is almost entirely downstream control performance. For an ICLR standard, the paper would benefit from a more precise statement of what is actually learned and what generalization claims can be supported.
- The claim that existing 3D VFMs “sometimes even underperform simple MLPs” is plausible and referenced, but the paper should more carefully delimit when and why this occurs, since it is used as a major justification for the new architecture.

### Method / Approach
- The overall method is plausible and interesting: multi-modal MAE over RGB, depth, and point cloud; stochastic masking; cross-modal decoding; and distillation from a Giant teacher.
- The modality-processing pipeline is reasonably described, but there are important clarity and reproducibility issues:
  - In Section 2.2, the masking notation is hard to follow, and the paper does not fully specify exactly how masking is applied across modalities in practice when some modalities are absent at downstream inference.
  - The point cloud tokenizer is described as FPS + KNN + DP3 encoder, but critical hyperparameters such as the number of groups \(N\), neighborhood size \(K\), and whether these differ across scales are not clearly stated in the main method.
  - The decoder description says it uses cross-attention and a modality-shared ViT decoder, but the exact attention pattern, tensor shapes, and how modality-specific and shared tokens are interleaved are underspecified for reproduction.
- There are also logical gaps:
  - The paper claims a “unified 3D multi-modal representation,” but the training objective is reconstruction plus feature alignment. It is not fully explained why this should induce a representation that transfers robustly to control, beyond empirical evidence.
  - The distillation setup is potentially unusual: the teacher is frozen and the student is trained with MAE reconstruction plus intermediate feature alignment, but the relation between the teacher’s masked input distribution and the student’s smaller model is not fully justified.
- One serious issue is the mathematical presentation of the loss equations in Section 2.3–2.4. The parser garbles them, but even ignoring formatting artifacts, the paper should ensure the final submission cleanly specifies the reconstruction targets, normalization, and exact form of the feature alignment loss.
- The choice to initialize the ViT from DINOv2 weights while removing the CLS token is plausible, but the paper does not explain whether this changes the downstream token aggregation or if any pooling is used. That matters for reproducibility.
- Failure modes are acknowledged only indirectly. For instance, point cloud quality is said to be sensitive to sensor noise later in the appendix, but the core method section does not discuss how the architecture handles noisy or incomplete depth/point clouds during pretraining.

### Experiments & Results
- The experiments are broad and, in principle, well matched to the claims: simulation tasks, real-world tasks, and ablations on masking, alignment, and policy backbone.
- The use of both LIBERO and MetaWorld is reasonable for evaluating manipulation representations, and the real-world SO100/xArm results strengthen the practical relevance.
- However, several experimental issues would matter for an ICLR reviewer:
  - Fairness/completeness of baselines is not fully convincing. The main comparisons center on DINOv2, SigLIP, R3M, VC-1, and SPA, but for 3D settings, the most relevant baselines are not always matched in a controlled way. In particular, the “naïve DINOv2-RGBD” baseline is useful, but the broader comparison to 3D-specific pretrained models is limited.
  - The paper repeatedly claims “SOTA” but does not always show whether the strongest baseline is tuned equally well for each modality and each benchmark.
  - There are no error bars or confidence intervals in the main results tables/figures, despite repeated evaluation over multiple tasks and random seeds. This is especially important for real-world robotics, where variance can be substantial.
  - The ablation table in Section 3.5 is informative but limited. The most consequential missing ablation is a direct comparison of pretraining modalities and data sources: for example, RGB-only pretraining vs RGBD vs RGBD+PC, and the effect of using DROID-3D versus a lower-quality depth source. The appendix touches on point clouds, but the central narrative would benefit from clearer controlled evidence.
  - The “RQ1” visual prediction analysis is interesting, but it is qualitative only. Given the paper’s strong claim about cross-modal fusion, a quantitative reconstruction or modality-translation metric would materially strengthen the argument.
- The data scale and evaluation scope are impressive, but the paper should be careful not to conflate downstream policy success with representation quality in a general sense. Since the policy backbone is fixed, results are informative, but they still reflect a particular policy-learning setup rather than representation usefulness in general.
- The real-world experiments are a meaningful strength. Still, the paper should better document:
  - how many seeds/runs were repeated,
  - whether policies were re-tuned per baseline,
  - and whether the same demonstration counts and training budgets were used for all representations.
- Some appendix results are promising but raise questions:
  - The “Enhanced Point Cloud Quality” results suggest the model benefits substantially from preprocessing. This implies that real-world performance may depend heavily on carefully engineered sensor cleanup, which complicates the claim that the model itself robustly handles 3D perception.
  - The scaling/data-size ablation in Table 11 shows only a modest degradation from 100% to 25% data, but the paper does not provide statistical significance, so the “minimal and non-significant” wording is not fully justified.

### Writing & Clarity
- The paper is generally understandable, and the high-level narrative is coherent.
- That said, there are several places where clarity affects comprehension of the contribution:
  - The method section would benefit from a more precise, end-to-end schematic of data flow across the three modalities during pretraining and distillation.
  - The experimental setup is spread across the main text and appendix in a way that makes it hard to track what exactly was held fixed across baselines.
  - The tables/figures in the extracted text are sometimes hard to parse, but beyond parser artifacts, the paper should ensure the actual PDF presents the benchmark tables clearly enough to support the claims.
- The strongest clarity issue is conceptual: the paper alternates between describing EmbodiedMAE as a representation model, a foundation model, and a policy-enabling module. Those are related, but the paper would be clearer if it explicitly separated representation learning claims from downstream control claims.
- The visual prediction discussion in Section 3.2 is intuitive, but some claims there are stronger than the evidence shown. For instance, the inference that the model has “implicitly learned object-level semantic segmentation” from recoloring behavior is suggestive but not conclusive.

### Limitations & Broader Impact
- The paper does acknowledge one key limitation: EmbodiedMAE does not natively support language instruction and is currently vision-only.
- That is an important limitation, but it is not the only one:
  - The approach depends heavily on high-quality depth extraction and careful preprocessing, especially for point clouds in real-world settings.
  - The gains appear most convincing in precise tabletop manipulation; the paper’s own framing suggests limited evidence for broader embodied settings such as cluttered or outdoor environments.
  - The model’s utility depends on a downstream policy backbone (RDT/ACT), so its practical value is mediated by the policy architecture rather than standing alone as a general-purpose VFM.
- On broader impact, the ethics statement is minimal but acceptable given the scope. However, the paper should more directly discuss deployment risks in robotics: failures in grasping, collisions, and sensor-induced brittleness are relevant safety concerns. The paper shows failure cases, but it does not discuss their implications.
- Since the dataset contains robot interaction trajectories, there is little immediate privacy concern, but the paper should still note that manipulating physical robots and real-world platforms introduces safety and reproducibility constraints.

### Overall Assessment
EmbodiedMAE is a substantial and timely paper with a genuinely interesting idea: learning a unified RGB-depth-point-cloud representation for manipulation, supported by a new large-scale processed dataset and evaluated across a wide range of simulation and real-world tasks. The strongest aspects are the scale of the data effort, the breadth of evaluations, and the consistent downstream gains, especially in 3D settings. That said, for ICLR’s standard, the paper is currently stronger as an applied systems/benchmark contribution than as a fully compelling representation-learning paper: the method is under-specified in key places, the experimental evidence lacks error bars and some crucial controlled ablations, and several claims about generality and “foundation model” status are somewhat broader than the evidence warrants. I think the contribution is promising and likely useful, but it needs clearer methodological specification and more rigorous evidence to fully meet the strongest ICLR acceptance bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes EmbodiedMAE, a unified masked autoencoder for robot manipulation that jointly learns from RGB, depth, and point cloud inputs. The authors also release DROID-3D, a processed version of DROID with depth and point clouds, and claim strong gains over prior vision foundation models on LIBERO, MetaWorld, and two real-world robot platforms.

### Strengths
1. **Addresses an important ICLR-relevant problem: representation learning for embodied robotics with 3D perception.**  
   The paper targets the gap between general-purpose vision foundation models and manipulation-specific needs, which is a meaningful and timely direction for ICLR, especially given the growing interest in multi-modal and embodied representation learning.

2. **The dataset contribution is potentially valuable.**  
   The paper introduces DROID-3D, comprising 76K trajectories and 350 hours of robot data with synchronized RGB, depth maps, and point clouds. If the preprocessing quality is as described, this could be a useful resource for the community, since many prior datasets have incomplete or noisy 3D information.

3. **The method has a coherent design that aligns with the problem setting.**  
   The model uses stochastic modality-wise masking and cross-modal decoding, which is a natural extension of MAE-style learning to embodied multi-modal inputs. The choice to share transformer components across modalities is also sensible for efficiency.

4. **The evaluation scope is broad.**  
   The paper reports results on 70 simulation tasks and 20 real-world tasks across two robot platforms, and includes ablations on masking, feature alignment, policy choice, and point cloud processing. This breadth is stronger than many robotics representation papers that evaluate narrowly.

5. **There is evidence of scaling and practical deployment consideration.**  
   The paper includes Small/Base/Large/Giant variants, distillation from a large teacher, latency measurements, and discussion of real-world inference constraints. These are important for robotics and make the work more application-oriented.

### Weaknesses
1. **The novelty is moderate rather than clearly breakthrough-level by ICLR standards.**  
   The method is largely a combination of established ideas: MAE-style masking, multi-modal fusion, transformer backbones, and distillation. The main technical novelty appears to be the specific modality-masked cross-modal reconstruction setup for robot data, but the paper does not convincingly establish a new learning principle beyond a well-engineered adaptation of existing methods.

2. **The empirical claims are very strong, but the evidence is not presented with enough methodological transparency to fully support them.**  
   The paper repeatedly states SOTA across many tasks, yet key experimental details needed to judge fairness are thin in the main text: how exact baselines were tuned, whether all methods use identical policy training budgets, how hyperparameters were selected, and whether the reported gains are statistically significant. For ICLR, large performance claims need especially careful controlled evidence.

3. **The comparison against baselines may be incomplete or uneven.**  
   Several baselines are diverse, but the paper compares against a mixture of vision-centric, embodied-specific, and 3D-aware methods that may differ substantially in pretraining data, modality support, and downstream compatibility. The paper does not clearly establish that each baseline is adapted optimally for every input modality, which makes some comparisons harder to interpret.

4. **Some architectural choices are under-justified.**  
   For example, the paper omits modality-type embeddings and relies on projection bias to encode modality identity; this is asserted rather than carefully validated. Similarly, the choice to fix total unmasked patches and sample modality allocations from a symmetric Dirichlet is reasonable, but the paper does not provide strong evidence that this is better than simpler alternatives.

5. **The point cloud story is somewhat mixed.**  
   The paper claims 3D inputs help substantially, but also reports that point cloud performance can underperform RGB in real-world settings due to sensor noise, requiring enhanced preprocessing to work well. This raises concerns about robustness and suggests the representation may be more dependent on careful input engineering than the headline claim implies.

6. **Reproducibility is only partially convincing.**  
   The paper gives many hyperparameters and dataset descriptions, but some critical details remain vague: exact preprocessing for DROID-3D, distillation loss implementation, baseline training schedules, and the procedure for collecting and filtering real-world demonstrations. The reproducibility statement is also generic and promises code release, but does not yet provide enough detail for exact replication.

7. **The writing and presentation are uneven in places.**  
   The main ideas are understandable, but the paper sometimes reads like a polished benchmark report rather than a tightly argued research contribution. Some claims are repeated in a promotional way, and the core technical differences from prior multimodal MAE approaches could be clearer.

### Novelty & Significance
**Novelty:** Moderate. The work is a solid systems-level and empirical extension of MAE-style multimodal pretraining to embodied 3D robotics, but it is not obviously a fundamentally new learning paradigm.  
**Significance:** Potentially high if the dataset and reported results hold up under closer scrutiny, because better 3D representations for manipulation are an important problem. For ICLR, the significance is strengthened by scale and breadth, but the bar for acceptance also requires clear technical insight, and that is less compelling than the empirical results.  
**Clarity:** Moderate. The high-level pipeline is clear, but several implementation and comparison details are underspecified.  
**Reproducibility:** Moderate. Many training parameters are provided, but the experimental protocol and preprocessing pipeline need more precision.  
**Overall ICLR fit:** Borderline positive. The paper is relevant and ambitious, but ICLR typically expects either a sharper algorithmic contribution or very solid, carefully controlled evidence of a broadly useful new method. This submission leans more toward the latter, but would benefit from stronger methodological rigor and clearer ablation evidence.

### Suggestions for Improvement
1. **Strengthen the technical novelty discussion.**  
   Explicitly distinguish EmbodiedMAE from existing multimodal MAE and 3D pretraining methods, and clarify what is genuinely new beyond combining known components.

2. **Add more controlled ablations.**  
   For example, ablate modality-type embeddings, Dirichlet masking vs. fixed masking, cross-attention decoder vs. simpler fusion, and distillation losses at each alignment point. This would better justify the design choices.

3. **Improve baseline fairness and protocol transparency.**  
   Report how each baseline was trained, tuned, and adapted to each modality. If some baselines cannot naturally use depth or point clouds, state the exact adaptation strategy and its limitations.

4. **Provide statistical confidence.**  
   Include standard deviations or confidence intervals over seeds/tasks, especially for real-world experiments where variance can be large. This would make the claimed improvements more credible.

5. **Clarify DROID-3D construction in detail.**  
   Describe preprocessing, failure cases, filtering criteria, and the exact use of ZED SDK more precisely. A small diagnostic study on depth quality vs. downstream performance would strengthen the dataset claim.

6. **Separate the contribution of the dataset from the model.**  
   It would help to show whether the gains come primarily from DROID-3D, the architecture, or the distillation scheme. A factorial study would make the paper’s main message much more convincing.

7. **Discuss robustness and failure modes more honestly.**  
   The paper already notes point cloud noise in real-world settings; expand this into a clearer limitation section with quantitative failure analysis and conditions under which the method degrades.

8. **Tighten the exposition around the core representation learning idea.**  
   A schematic or algorithm box that precisely defines masking, fusion, reconstruction, and distillation would improve readability and make the method easier to reproduce.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add ablations that isolate the source of gains: pretraining on DROID-3D vs. the same model trained on original DROID, vs. DROID-3D with only RGB/depth/point cloud subsets. Without this, the core claim that the 3D augmentation itself drives the improvement is not convincing.
2. Compare against stronger and more directly relevant embodied baselines, including recent VLA/robot foundation models and 3D-aware policy representations on the same policy backbone and training budget. ICLR reviewers will expect the claimed “SOTA” to be tested against the strongest available embodied and 3D perception baselines, not just a limited set of VFMs.
3. Add a fair data-scale and compute-matched baseline showing whether EmbodiedMAE beats simply training a larger or longer-pretrained DINOv2/MAE-style vision encoder on the same robot data. Otherwise it is unclear whether the gains come from the multi-modal architecture or just more pretraining signal and compute.
4. Add a pretraining-only evaluation on standard 3D representation benchmarks or held-out reconstruction/transfer tasks, not just downstream control. The paper claims a “unified 3D multi-modal representation,” but the evidence is almost entirely policy success rates.
5. Include a full sensitivity study for point cloud generation quality and sensor noise on the same tasks, especially in real-world settings. Since the method’s 3D claims depend heavily on ZED processing and point-cloud preprocessing, the reported gains may not generalize to typical robotics sensors.

### Deeper Analysis Needed (top 3-5 only)
1. Add an analysis of why RGBD beats point clouds in real-world tasks despite PC helping in simulation. This discrepancy directly affects the paper’s claim that the model robustly “promotes policy learning from 3D inputs.”
2. Quantify which downstream gains come from cross-modal fusion versus reconstruction regularization versus feature distillation. Without decomposing these terms, the mechanism behind the method’s success is not established.
3. Analyze performance as a function of task geometry, occlusion, and localization difficulty. The paper repeatedly claims improved spatial understanding, but does not show whether gains are concentrated on tasks that actually require 3D reasoning.
4. Report variance, confidence intervals, or statistical significance across seeds/trials. ICLR expects robust evidence; many of the claimed improvements are small enough that significance matters.
5. Provide a compute/data efficiency analysis that includes total pretraining cost, distillation cost, and downstream fine-tuning cost relative to baselines. The paper claims efficiency and scalability, but the current evidence does not separate efficiency from raw resource usage.

### Visualizations & Case Studies
1. Show failure cases where EmbodiedMAE still fails on real robot tasks, especially under clutter, reflective objects, occlusion, and camera miscalibration. This would reveal the true limits of the 3D representation rather than only curated success rollouts.
2. Visualize attention or token attribution across RGB, depth, and point cloud branches during successful and failed predictions. Without this, the claimed cross-modal fusion remains qualitative.
3. Add side-by-side rollout comparisons on the same task with RGB-only, RGBD, and PC inputs under identical conditions. This would make it clear whether 3D inputs genuinely improve localization or just occasionally help.
4. Show reconstruction maps or point-cloud completion examples on held-out scenes, especially for fine-grained contact regions. This would test whether the MAE objective actually learns manipulation-relevant 3D structure.
5. Include a qualitative comparison between DROID-3D depth/PC quality and the real-world sensor inputs used at test time. This would expose whether the method depends on a train-test sensor-quality match.

### Obvious Next Steps
1. Extend the method to language-conditioned VLA training on DROID-3D, since the paper itself admits the current model is vision-only. Without this, the contribution is less aligned with the dominant embodied AI direction at ICLR.
2. Release and benchmark DROID-3D as a standalone resource with standardized evaluation splits and reconstruction/representation benchmarks. The dataset claim is important enough that it needs to be validated independently of the policy results.
3. Test transfer to additional robot platforms and sensor setups beyond SO100 and xArm. The current evidence is too platform-specific to support a broad “unified 3D representation” claim.
4. Evaluate whether the learned representation improves downstream finetuning efficiency under very low-data regimes. That would directly test the foundation-model claim, not just peak success rates.

# Final Consolidated Review
## Summary
This paper proposes EmbodiedMAE, a multi-modal masked autoencoder for robot manipulation that jointly pretrains on RGB, depth, and point cloud inputs. It also introduces DROID-3D, a processed version of DROID with extracted depth maps and point clouds, and reports improved downstream performance on LIBERO, MetaWorld, and two real-world robot platforms.

## Strengths
- The paper tackles an important and timely embodied-representation problem: learning 3D-aware visual backbones for manipulation rather than relying on generic image-pretrained models.
- The dataset contribution is substantial in scale and likely useful: DROID-3D contains 76K trajectories / 350 hours, and the authors provide a concrete pipeline for producing temporally consistent depth and point clouds from DROID recordings.
- The overall method is coherent and well matched to the task: modality-wise stochastic masking, cross-modal reconstruction, and distillation from a larger teacher are sensible design choices for learning multi-modal manipulation representations.
- The evaluation is unusually broad for this area, spanning 70 simulation tasks and 20 real-world tasks on two robot platforms, with ablations on masking, distillation alignment, policy choice, and point-cloud processing.
- There is credible evidence of scaling and practical deployment awareness: the paper reports Small/Base/Large/Giant variants and includes latency measurements and real-world robot experiments rather than only simulation.

## Weaknesses
- The paper’s central claims are broader than the evidence actually establishes. Most results are downstream policy-success numbers under a single policy backbone, so the manuscript does not fully justify calling EmbodiedMAE a generally “reliable unified 3D multi-modal VFM” or a broadly validated representation beyond these manipulation settings.
- The experimental evidence is not sufficiently controlled to support the strength of the SOTA claims. The paper compares against several strong baselines, but it is still unclear whether all methods were tuned and adapted equivalently across RGB, RGBD, and point-cloud settings, and the main results lack error bars or confidence intervals.
- The method is under-specified in key places for a paper making a systems-level foundation-model claim. The masking scheme, point-cloud tokenization hyperparameters, decoder attention structure, and exact distillation implementation are not fully spelled out in a way that makes faithful reproduction straightforward.
- A major part of the story depends on careful sensor preprocessing, especially for point clouds. The appendix shows that enhanced point-cloud processing materially changes performance, which weakens the headline implication that the learned representation alone robustly solves 3D perception in real-world robotics.

## Nice-to-Haves
- A factorial ablation separating the contributions of DROID-3D, the multi-modal architecture, and the distillation scheme would make the main message much more convincing.
- A quantitative reconstruction or modality-translation evaluation would strengthen the qualitative cross-modal fusion claims in the RQ1 analysis.
- Reporting variance across seeds and trials would help, especially for real-world experiments where some gains may be modest and task variance can be high.

## Novel Insights
The most interesting aspect of the paper is not simply “adding depth” to a vision backbone, but the attempt to build a single masked-reconstruction objective that spans RGB, depth, and point clouds while keeping the representation useful for control. The appendices suggest an important but somewhat uncomfortable lesson: much of the success hinges on matching sensor quality and preprocessing to the pretraining regime, especially for point clouds, so the representation is not fully decoupled from the data pipeline. In that sense, the work is best read as a strong domain-specific embodied pretraining recipe rather than a broadly solved 3D foundation-model problem.

## Potentially Missed Related Work
- **DROID / DROID-3D-related embodied robot datasets** — directly relevant as the source dataset and closest data-lineage comparison.
- **SPA (3D spatial awareness for embodied representation)** — especially relevant because it is one of the closest embodied 3D baselines and is already discussed, but it may deserve a more explicit comparison of data scale and 3D preprocessing.
- **3D Diffusion Policy / DP3** — relevant as a strong 3D-aware manipulation baseline on point clouds, particularly for interpreting whether the representation or the policy backbone drives gains.
- **VGGT / VGGT-DP** — relevant as a stronger geometric-vision comparison for the point-cloud / spatial reasoning angle, especially since the appendix already references it.

## Suggestions
- Add a controlled study comparing: original DROID vs. DROID-3D; RGB-only vs. RGBD vs. RGBD+PC; and with/without distillation, all under the same downstream budget.
- Include standard deviations or confidence intervals across seeds and task groups, and clearly state how each baseline was tuned and adapted to each modality.
- Provide a concise algorithm box with exact masking, tokenization, decoder fusion, and distillation steps so the method can be reproduced without ambiguity.
- Expand the discussion of real-world failure modes, especially cases where point clouds degrade or where enhanced preprocessing is required for the reported gains.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0, 4.0]
Average score: 5.0
Binary outcome: Reject
