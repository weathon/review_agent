# Learning Open-World Visual-Tactile Grasp Stability Prediction with Synthetic Data

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Grasp stability prediction with visual-tactile data is an important problem in robotics. Most prior work learns these predictors with limited real-world data. Moreover, their evaluation has also been restricted to a simple and unitary laboratory environment. Our work studies open-world visual-tactile grasp stability prediction, i.e. the predictor should zero-shot generalize to novel objects in novel environment. Towards this problem, we propose to learn with synthetic visual-tactile data, generated with FEM-based simulation and ray-tracing rendering. In our experiment, we show that our simulation pipeline has much higher physical fidelity, compared to the rigid-body simulation. Furthermore, the predictor trained on our synthetic dataset has higher accuracy on open-world grasp stability prediction tasks than models trained on real-world dataset or on synthetic dataset from rigid-body simulation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles open-world visual-tactile grasp stability prediction and proposes training entirely on synthetic data generated via FEM-based simulation (Taccel/IPC) plus ray-traced RGB and a custom tactile renderer. The pipeline produces ~30k visual-tactile pairs from 10k+ grasps across 453 objects, and the authors additionally collect a 333-grasp real-world “open-world” test set for zero-shot evaluation. A model trained on the synthetic set achieves 77.5% average accuracy on the real test set, outperforming training on an existing real dataset (9.2k samples) and on a rigid-body (Taxim/PyBullet) synthetic set. The paper also reports a smaller sim-to-real gap for FEM vs. rigid-body simulation on 5 grasp-annotated objects.

### Strengths
- Clear open-world setup with a purpose-built real evaluation set; synthetic data scale and diversity are well described (objects, scenes, lighting). 
- Technically sound switch to FEM-based contact modeling and an improved tactile renderer; qualitative and quantitative evidence suggests reduced sim-to-real gap. 
- Strong end-to-end gains: synthetic V+T training substantially beats real-only or rigid-body synthetic baselines; ablations show tactile is necessary and V+T is best.

### Weaknesses
- Real evaluation size is modest (266 test grasps) and human-actuated; potential bias across scenes/objects is under-analyzed (no per-scene/category/error breakdowns).
- Simulator fairness: in Sec. 5.1 you grid-search per-object friction $\mu_o$ and $F_\text{stop}$ to match outcomes, which can overfit the annotated set and handicap Taxim; cross-object fixed configs or cross-validation would be more convincing.
- Dataset preparation relies on LLM-estimated mass/friction and then clips/mass-maps to $[0.01,0.2]$ kg; this may distort contact dynamics. Reproducibility also hinges on releasing assets, renderer settings, and the 2500+ GPU-hour simulation outputs.

### Questions
- Tactile renderer: you intersect rays with the object (not the gel) and clip by gel thickness. Please provide the exact depth computation formula, LED positions/intensities, and any per-pixel normalization. How sensitive are results to these choices?
- Sec. 5.1 hyper-parameters: when tuning $\mu_o$ and $F_\text{stop}$ per object, which labels/frames were used, and how did you prevent test leakage? Can you report accuracy with (i) one global $(\mu_o,F_\text{stop})$ across all objects and (ii) parameters calibrated on a subset only?
- Simulation ablations: report performance sensitivity to $\Delta t=0.01$, $\kappa=3\!\times\!10^6$, $\hat d=5\!\times\!10^{-4}$, $\epsilon_v=10^{-3}$, $\epsilon_r=5\!\times\!10^{-5}$, and to the mass mapping/clipping. Also, will code, datasets (synthetic + real), and exact random seeds be released?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a simulation pipeline to generate synthetic dataset for training visual-tactile grasp stability predictors. The contributions are mainly based on a previous work Taccel for simulation. The authors claim that the trained grasp stability predictor on the synthetic dataset has higher prediction accuracy than baselines.

### Strengths
This paper collected a relatively large number of objects in the open-world dataset.

### Weaknesses
- This paper builds primarily on a previous work Taccel. The contributions appear to be mainly incremental and focused on engineering refinements, which somewhat limits the novelty and significance of the work.

- The paper would benefit from citing and discussing several relevant previous work, particularly to clarify how it advances beyond existing approaches. Please find more details below.

- The experimental results presented are somewhat limited in scope. Providing more comprehensive evaluations or additional analyses would make the contributions more convincing. Please find more details below.

- The authors mention grasping deformable objects as future work. However, using FEM-based simulation for rigid objects seems like mathematically and computationally overkill. It would be better to discuss more on this point.

### Questions
- The authors claim one of their contributions is to train visual-tactile models for grasp stability prediction but discussion and comparison of relevant work is missing, such as paper [1,2].

- In the related work section, several tactile sensor simulators are mentioned, but only Taxim is compared against the proposed pipeline in the experiments. It would be more convincing to compare with more simulators.

- The experiments focus exclusively on the GelSight Mini tactile sensor. The impact and generality of the work could be enhanced by considering other widely used tactile sensors, such as curved tactile sensor GelSight360. 

- The evaluation currently covers a limited number of scenarios. Including a broader range of test cases or environments would provide a more comprehensive assessment.

References

[1] Cui, Shaowei, et al. "Self-attention based visual-tactile fusion learning for predicting grasp outcomes." IEEE Robotics and Automation Letters 5.4 (2020): 5827-5834.

[2] Cui, Shaowei, et al. "Grasp state assessment of deformable objects using visual-tactile fusion perception." 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a custom tactile simulator and renderer for large-scale synthetic data generation to train a grasping stability predictor. The authors demonstrate that the predictor generalizes to real-world and novel objects.

### Strengths
* The extension to diverse-object grasping prediction is strong and well supported.
* The authors describe the experimental settings in detail, making the work easier to validate and replicate.
* The ablation study shown in Fig 6 demonstrates the effectiveness of vision+tactile sensing compared to baselines.

### Weaknesses
* The proposed simulated rendering is better than the baseline, but it still appears quite different from real tactile patterns. According to Figs. 2 and 4, the simulated rendering looks bulky compared with fine-grained real tactile signals. Additionally, the simulated contact patterns differ: in Fig. 4 (right column), the real tactile image shows one line of contact points, whereas the simulated image shows two rows. This raises concerns about the effectiveness of the simulated tactile rendering. If the selected results differ so much from real observations, how can we be confident the stability predictor will produce useful predictions?
* The authors attribute the failure of Taxim to convex decomposition, but this argument is not fully convincing. The fidelity of convex decomposition can be tuned; if slower simulation is acceptable, artifacts can, in principle, be mitigated.
* Some contributions are not well explained. The paper appears to emphasize two main contributions: (1) novel-object grasping prediction and (2) a custom tactile simulation renderer. However, the renderer is not sufficiently elaborated and reads as an engineering addition on top of Taccel. Moreover, its quality does not align well with real measurements, as noted above.

### Questions
* Fig. 2 seems to indicate that the simulated tactile image differs substantially from the ground-truth tactile image. Is this reading correct? Additionally, the contact positions appear offset.
* How does the framework close the sim-to-real gap given the pronounced differences in visuo-tactile renderings? Directly evaluating the predictor in the real world may be brittle under such a gap.
* A suggestion: include three bullet points summarizing the paper’s contributions at the end of the introduction. This would make the paper clearer and aligns with common practice in the robotics community.

### Soundness
3

### Presentation
3

### Contribution
3
