# OccDriver: Future Occupancy Guided Dual-branch Trajectory Planner in Autonomous Driving

- Decision: Accept (Poster)
- Scores: 8, 6, 2

## Abstract
Trajectory planning for autonomous driving is challenging due to agents' behavioral uncertainty and intricate multi-agent interaction modeling. Most existing studies generate trajectories without explicitly exploiting possible scene evolution, while world models predict consequences from ego behavior, enabling more informed planning decisions. Inspired by the world model, we propose OccDriver, a novel rasterized-to-vectorized dual-branch framework for trajectory planning. This pipeline performs a coarse-to-fine trajectory decoding process: The vectorized branch first generate multimodal coarse trajectories; Then the rasterized branch predicts future scene evolutions conditioned on each coarse trajectory via occupancy flow prediction; Lastly, the vectorized branch leverages intuitive future interaction evolution of each modality from the rasterized branch and produces refined trajectories. Several cross-modality (occupancy and trajectory) losses are further introduced to improve the consistency between trajectory and occupancy prediction. Additionally, we apply a contingency objective in both occupancy space, considering marginal and joint occupancy distributions in different planning scopes. Our model is assessed on the large-scale real-world nuPlan dataset and its associated planning benchmark. Experiments show that OccDriver achieves state-of-the-art in both Non-Reactive and Reactive closed-loop performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes OccDriver, a dual-branch transformer framework that integrates future occupancy prediction into trajectory planning for autonomous driving. The model consists of a vectorized branch, which generates coarse-to-fine multimodal trajectories, and rasterized branches, which predicts future occupancy and flow fields for both joint and marginal prediction. The rasterized branch acts as a world model, providing future scene evolution guidance to refine the ego trajectory. Several cross-modality consistency losses, including occupancy interference, trajectory-occupancy alignment, and trajectory-occupancy collision losses, are designed to couple the two branches effectively. The authors further introduce a contingency planning objective that considers both short-term marginal and long-term joint occupancy distributions to handle behavioral uncertainty. Evaluations on the nuPlan benchmark show that OccDriver achieves state-of-the-art performance on both Non-Reactive and Reactive closed-loop metrics, especially improving safety (collision, TTC) without sacrificing comfort or progress.

### Strengths
The proposed rasterized-to-vectorized pipeline elegantly combines probabilistic world-model reasoning with precise trajectory planning, bridging two dominant paradigms (rasterized and vectorized) in motion planning.

Incorporating marginal and joint future occupancy prediction as explicit supervisory signal and guidance mechanism through contingency planning, seems to be a quite novel idea

The experiments are thorough, covering both Val14 and Test14-Hard nuPlan benchmarks, with convincing performance gains in safety and closed-loop metrics. The ablation studies clearly demonstrate the contribution of each module.

### Weaknesses
The method adds nontrivial computation due to occupancy decoding and marginal distribution estimation. Although inference time is reported (≈23 ms), the training cost and scalability with agent count are not analyzed in depth.

The paper lacks a deeper analysis of why occupancy-based guidance ensures optimal or safer planning.

The fonts are small in current figures. Fig.2 a bit cluttered and difficult to follow. The implementation of contingency planning could also be summarized in a figure.

### Questions
1. Does marginal occupancy pruning significantly affect safety performance? 

2. How are the short-term (Ts) and long-term (Tf) horizons chosen for contingency, and how robust is performance to these hyperparameters?

3. How does OccDriver conceptually differ from recent world-model planners (e.g., DriveDreamer) that also predict scene evolution before planning? Could OccDriver be extended to reinforcement-based fine-tuning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents OccDriver, a rasterized-to-vectorized framework for motion planning in autonomous driving. Motivated by the limitations of trajectory-based or occupancy-based approaches, it integrates a vectorized trajectory decoder with an occupancy world model that predicts future scene evolution conditioned on coarse trajectories. Cross-modal losses—including occupancy interference, trajectory–occupancy alignment, and collision penalties—enable explicit safety-aware guidance. Additionally, a contingency strategy based on short-term marginal and long-term joint occupancies enhances robustness under uncertainty.Evaluated on the nuPlan benchmark, OccDriver achieves SOTA closed-loop driving scores and superior safety metrics, demonstrating its effectiveness in reliable and interpretable planning.

### Strengths
- The paper proposes an innovative Dual-branch Planning Framework with a coarse-to-fine decoding mechanism: the coarse trajectory decodergenerates a preliminary behavioral framework, while the fine trajectory decoder optimizes trajectory with future scene information. This hierarchical design balances planning efficiency and accuracy, addressing the computational cost-rationality trade-off in traditional single-stage methods.
- Marginal prediction (MP) and contingency planning (CP) modules are introduced for emergency risk: MP captures individual agents’ short-term marginal occupancy, and CP generates emergency trajectories. Ablation experiments (Table 3) confirm both modules improve safety metrics and driving scores, verifying the risk-aware design’s practical value.

### Weaknesses
- The paper demonstrates the effectiveness of the proposed Dual-branch Planning Framework by presenting the performance of the fine trajectory that incorporates future scene information and occupancy guidance. However, it lacks direct quantitative and qualitative comparative analysis between the coarse trajectory and the fine trajectory.6.2 The paper only dissolved M=1, 3, and 6, without verifying 
the monotonic performance growth of more M or identifying the optimal "peak" M. If there is a maximum peak M, there is still a lack of strict explanation for why a specific M (if optimal) performs the best.

### Questions
- Given that the fine trajectory decoding is conditioned on the latent variableQ_c(coarse trajectory feature), what is the necessity of explicitly decoding the coarse trajectoryY_c? Additionally, is there confusion regarding whether direct supervision is applied toY_c?
- Supplementing direct quantitative and qualitative comparisons between the coarse trajectory and the fine trajectory would more effectively demonstrate the validity of the "coarse-to-fine" framework proposed in this paper.
- The authors should conduct more comprehensive ablation studies on the number of modalities.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes OccDriver, a dual-branch rasterized-to-vectorized trajectory planner that integrates future occupancy predictions as guidance for trajectory generation in autonomous driving. The framework consists of a rasterized branch that predicts future scene evolution in occupancy space and a vectorized branch that plans ego trajectories conditioned on this predicted scene. The authors utilize several cross-branch loss functions and a contingency planning objective to enhance safety and robustness. Experiments on the nuPlan dataset demonstrate state-of-the-art closed-loop results compared with existing learning-based planners. While the paper is technically sound and well-engineered, its conceptual novelty is insufficient for a top-tier contribution (see Weaknesses).

### Strengths
1. The quantitative results on the nuPlan benchmark show competitive performance.
2. The ablation study is detailed and systematic.
3. The paper is readable, the methodology is well-documented, and architectural details are described thoroughly in the appendices.

### Weaknesses
1. My main concern is the novelty of this paper. The concept of using occupancy prediction to guide planning is not new. Even UniAD[1] in 2023 has already explored joint designs between occupancy forecasting and trajectory generation. It seems that the main difference here is to adopt a parallel dual-branch structure rather than a cascaded one, which feels like an architectural variation rather than a fundamentally new paradigm. The claimed “future occupancy guidance” and “dual-branch feature interaction” are incremental extensions of prior occupancy-assisted pipelines.
2. The related work section misses several closely aligned approaches (e.g., some end-to-end methods like World4Drive[2])
3. The evaluation relies on a single dataset, which limits claims about robustness and generalization. To substantiate the method’s effectiveness, it should include additional benchmarks such as NavSim[3].


[1] Hu, Yihan, et al. "Planning-oriented autonomous driving." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.
[2] Zheng, Yupeng, et al. "World4Drive: End-to-end autonomous driving via intention-aware physical latent world model." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.
[3] Dauner, Daniel, et al. "Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking." Advances in Neural Information Processing Systems 37 (2024): 28706-28719.

### Questions
Minor:
1. Discuss failure cases and qualitative scenarios where occupancy guidance meaningfully alters the decision.
2. No ablation isolating the dual-branch architecture itself (without losses) is clearly analyzed.

### Soundness
2

### Presentation
2

### Contribution
1
