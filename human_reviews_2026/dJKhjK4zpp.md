# BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Diffusion-based planners have shown strong potential for autonomous driving by capturing multi-modal driving behaviors. A key challenge is how to effectively guide these models for safe and reactive planning in closed-loop settings, where the ego vehicle's actions influence future states. Recent work leverages typical expert driving behaviors (i.e., anchors) to guide diffusion planners but relies on a truncated diffusion schedule that introduces an asymmetry between the forward and denoising processes, diverging from the core principles of diffusion models. To address this, we introduce BridgeDrive, a novel anchor-guided diffusion bridge policy for closed-loop trajectory planning. Our approach formulates planning as a diffusion bridge that directly transforms coarse anchor trajectories into refined, context-aware plans, ensuring theoretical consistency between the forward and reverse processes. BridgeDrive is compatible with efficient ODE solvers, enabling real-time deployment. We achieve state-of-the-art performance on the Bench2Drive closed-loop evaluation benchmark, improving the success rate by 7.72% and 2.45% over prior arts with PDM-Lite and LEAD datasets, respectively. Project page: https://github.com/shuliu-ethz/BridgeDrive.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes BridgeDrive, a planning method based on diffusion bridges that addresses the theoretical inconsistency in prior anchor-based diffusion policies such as DiffusionDrive. By enforcing a symmetric and theoretically consistent forward–reverse process, BridgeDrive improves closed-loop performance. The paper further investigates temporal speed waypoints vs. geometric path waypoints representations and demonstrates that the geometric representation leads to less likely to violate route lane constrain. Experiments on Bench2Drive show clear improvements over prior state of the art.

### Strengths
1. By incorporating bridge diffusion, the paper builds an explicit and theoretically complete formulation that connects anchor information with trajectory generation in anchor-based diffusion policies. This directly addresses the theoretical shortcomings arising when truncated diffusion processes ignore boundary consistency and rely on heuristically truncating the diffusion process.
2. The comparison between temporal and geometric trajectory parameterizations provides valuable insight for practical control and shows the superiority of geometric spacing.
3. The method achieves notable improvements on Bench2Drive, demonstrating strong closed-loop performance benefits.
4. Real-time inference and robust multi-modal trajectory generation contribute to the practicality of the approach.

### Weaknesses
1. The training and evaluation are limited to Bench2Drive simulation. Validation on larger-scale real human driving datasets, e.g. NavSim,  would be crucial to support claims of real-world generalization.

2. The influence of scene context versus anchor selection on the final trajectory is insufficiently analyzed. More evidence is needed to show that the trajectory is shaped jointly by scene understanding and anchors, instead of the anchor having disproportionate dominance.

### Questions
1. Could you provide quantitative or qualitative results confirming that both scene features and anchors significantly contribute to the trajectory generation (for example, attribution studies or case analyses)?

2. How do the number, diversity, and classification accuracy of anchors impact planning robustness? Any sensitivity or failure-case analysis?

3. Are there any experiments on large-scale datasets such as nuPlan or NavSim to support broader applicability claims?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces BridgeDrive, which adapts the Denoising Diffusion Bridge Model to generate geometric path waypoints from an anchor distribution, achieving state-of-the-art performance on the widely used Bench2Drive benchmark.

### Strengths
1. The paper point out that the problem of DiffusionDrive is that it is not a theoretical diffusion process. This issue has misled the community. The authors identified this and provided a solution.
2. Using Bridge Diffusion to solve this makes sense.

### Weaknesses
1. Given the authors' claim that geometric waypoints outperform temporal waypoints, it is better to validate whether this indicates a bias in the Bench2Drive benchmark. Therefore, verification across more benchmarks is highly recommended.

2. The architecture proposed in Figure 2 lacks ablation studies on its structural design.

3. The use of anchors greatly simplifies the true trajectory distribution, raising the question of whether using diffusion is truly necessary. A compelling comparison with SOTA anchor-based methods is needed to justify the adoption of bridge diffusion.

### Questions
1. Why is the Think2Drive expert used as a baseline when diffusion-based methods appear to perform worse than other approaches? Using the same expert for all methods facilitates a fair comparison.

2. Is Bridge Diffusion more challenging to train than standard diffusion models, and does it still support both classifier-free guidance and classifier guidance?

### Soundness
2

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
5

### Summary
The authors present BridgeDrive, which proposes an anchored guided diffusion formulation for closed-loop trajectory planning. BridgeDrive demonstrates SOTA performance on the Bench2Drive benchmark and provide useful insights on parameterization of diffusion models. The only concern is whether the experiments setup is fair for DiffusionDrive( scoring-based v.s predicted anchors), and whether the performance will be highly correlated with how well the anchor prediction is.

### Strengths
- Strong empirical SOTA performance on the Bench2Drive benchmark
- Provide good insights on how to parameterize diffusion models output: geomery points verus waypoints

### Weaknesses
### 

- The main comparison of this work is against DiffusionDrive. The primary comparison is to DiffusionDrive, but the two methods differ significantly: DiffusionDrive scores across *all* anchors and selects the best, while BridgeDrive first classifies one anchor then denoises around it. Please add this additional results for the scoring-based formulation in Table 1 and Table 2
- Why do we care about whether the forward and reverse are symmetric, and how does that affect the performance?  Following the previous points, DiffusionDrive provides a good way for fast inference speed (2 denoising steps compared to 20 steps).
- Couldn’t open the supp videos

### Questions
- What if the classified anchors are wrong for BridgeDrive, and how does this affect the perfromance? Does the diffusion models assumes that it always starts from the right anchor?
- Can BridgeDrive, for example, hanlde multiple anchors and then select the best result based on scoring, since this is more practical and may potentailly improve its out-of-distribution capabilities.
- How does the performance degrade by decreasing the diffusion timesteps for Bridgedrive? And why is Diffusiondrive’s sampling speed not ~10x faster than BridgeDrive since diffusion steps are 10x smaller?
- Table 1 miss diffusiondriveo^geo

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
BridgeDrive introduces a diffusion bridge policy for closed-loop trajectory planning in autonomous driving. Unlike previous diffusion-based planners (e.g., DiffusionDrive) that are mainly tested in open-loop settings, BridgeDrive applies a symmetric diffusion bridge framework connecting expert priors and ground-truth trajectories. This design resolves the theoretical inconsistency of prior truncated diffusion models and enables more stable, consistent, and safety-oriented planning. Experiments on the Bench2Drive benchmark demonstrate state-of-the-art results, surpassing SimLingo by +1.8 in Driving Score and +5% in Success Rate, especially in challenging tasks like merging, overtaking, and traffic sign compliance.

### Strengths
1. Recent works utilize diffusion models to enhance autonomous driving planning tasks seems just experiment on open-loop benchmarks (DiffusionPlanner and DiffusionDrive), while this work extend their experiment to close-loop benchmark, which is more chllenging and I think this is a good contribution to the community of autonomous driving.
2. The model BridgeDrive gets SOTA results on close-loop benchmark Bench2Drive.
3. The authors honestly provide the "Limitations and future work" section.

### Weaknesses
Theoretical clarity – Some diffusion concepts (e.g., “truncated diffusion,” the exact bridge formulation) are insufficiently explained for general readers.

Fairness of comparison – DiffusionDrive was reimplemented for Bench2Drive, which might affect comparison reliability.

Comfort and smoothness – Prioritizes safety over comfort, leading to frequent braking behavior.

Dependency on LiDAR – Relies on LiDAR input, limiting adaptability to camera-only settings.

### Questions
1. Could the author provide detail explain to why "DiffusionDrive that trying to  leverage typical human expert driving behaviors introduces a theoretical inconsistency: its denoising process does not match the forward diffusion process that it is trained on, which diverges from the core principle of diffusion models and can lead to unpredictable behavior and compromised performance." in line 36-43.
2. About Fig1. Intuitively, the startpoint of the denoise process should be noisy, but the waypoints in the leftmost subfigure seem very smooth. So these waypoints are not noise sampled from Gaussian but clustered resulst from expert drivers' behavoir?
3. How do you get the expert prior? Using the data of human drivers' behaviour data on open-loop benchmarks, like nuScenes? Or the "robot expert" trained via reinforcement learning on Bench2Drive?
4. About section 3.2 "DiffusionDrive with Truncated Diffusion". In this section, the author mentioned the asymmetry between the forward "add noise process" and the inverse "denoise process" of DiffusionDrive, whose starting point of forward process is expert prior (anchor) while the endpoint of inverse process is ground truth trajectories. I understand this. But, what does the "truncated" mean? In your paper (line 129-131), it seems that you want to express: timestep t is a pivot, before t, we add some noise, and after t we start denoising, which is really confusing.
5. You method name is "BridgeDrive", where can show "Bridge"? In line 191, you mentioned that you will learn a bridge model $p_\theta(x_t | x_T, z)$, so, you actually bridge the startpoint (ground truth traj) and endpoint (expert prior) at the diffusion process? And the endpoint is not pure gaussion noise traj? And in inference, you directly denoise from expert priors? That sounds plausible.
6. Finally, I think most of the ideas in this paper are based on the weaknesses of DiffusionDrive, so the comparison with DiffusionDrive is important. The problem is, DiffusionDrive only provides its code on NAVSIM and nuScese instead of Bench2Drive. So, to compare with DiffusionDrive, the author can:
  1. Implement your idea based on DiffusionDrive on NAVSIM and nuScenes, then test on the same testset.
  2. Reimplement DiffusionDrive to Bench2Drive, and then test directly, as you have implemented your methods on Bench2Drive.
  The author chooses the second method and although they provide detailed implementation details of how to reimplement DiffusionDrive to Bench2Drive, the fact is reimplement a method from one benchmark to another is challengeing, we cannot guarantee that during reimplementation, there maybe some errors lead to performance drop. So, I recommend the author can adopt the first compare method, in that way, DiffusionDrive have offical code and you can easily implement your method on it and avoid cross benchmark implement. And the result will be more convincing.

### Soundness
3

### Presentation
3

### Contribution
3
