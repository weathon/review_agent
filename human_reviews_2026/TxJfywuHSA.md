# Fourier Features Let Agents Learn High Precision Policies with Imitation Learning

- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Various 3D modalities have been proposed for high-precision imitation learning tasks to compensate for the short-comings of RGB-only policies.
Modalities that explicitly represent positions in Cartesian space, such as most point cloud encoder architectures, have an inherent advantage over purely image-based ones, since they allow policies to reason about geometry.
Despite the effectiveness of such architectures, a number of hybrid 2D/3D architectures have been proposed in the literature, indicating that this performance can often be task-dependent.
We hypothesize that this discrepancy may be due to the spectral bias of neural networks towards learning low frequency functions, which especially affects architectures conditioned on slow-moving Cartesian features.
We thus propose to use a parametric projection to map point clouds from Cartesian space into high-dimensional Fourier space when using a point cloud encoder. 
We experimentally validate the use of these Fourier features on challenging manipulation tasks from the RoboCasa and ManiSkill3 benchmarks, and on a real robot setup.
Despite their simplicity, we find that Fourier features provide robust and significant benefits across diverse encoder architectures and tasks.
These results indicate that Fourier features let policies leverage geometric details more effectively than Cartesian features, showing their potential as a general-purpose tool for point cloud-based imitation learning.
The overview and demos are available on our [project page: https://fourier-il.github.io/fourier-il](https://fourier-il.github.io/fourier-il/).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies spectral bias in point-cloud–conditioned imitation learning (IL) policies and proposes a simple, architecture-agnostic fix: apply a NeRF-style Fourier feature mapping to Cartesian 3D inputs before point-cloud encoding. The authors instantiate this on two representative encoders—PointPatch and DP3—and evaluate on RoboCasa (16 high-precision kitchen tasks, 50 demos each) and ManiSkill3 (4 tabletop tasks, 500 demos each). They report consistent gains from the Fourier mapping: e.g., RoboCasa mean success improves from 21.0%→40.0% (PointPatch) and 19.1%→26.3% (DP3), with notable per-task jumps like CloseDrawer 33.3%→70.0%. On ManiSkill3, average success rises from 50.8%→57.5% (PointPatch) and 58.8%→64.2% (DP3). The approach is intentionally minimal (fixed log-spaced bands; variable-magnitude jitter augmentation) and claims broad applicability to 3D-based IL.

### Strengths
1.Clear, simple idea with broad compatibility. The paper targets a real pain point—networks’ low-pass bias on slowly varying XYZ—and plugs in a standard Fourier mapping that can sit in front of most point-cloud tokenizers, not just a bespoke architecture.

2.Solid experimental coverage. Two encoders (local patch tokens vs. global DP3 token) and two popular benchmarks (RoboCasa, ManiSkill3) under a multi-task IL setup; consistent benefits across most tasks and encoders, with visual qualitative evidence.

3.High-precision tasks emphasized. The study focuses on tasks where small geometric distinctions matter (insertions, buttons, levers), which is where spectral bias plausibly bites most; the per-task tables quantify where improvements are largest.

### Weaknesses
1.Limited novelty. The core technique (Fourier features / positional encodings) is well-established; the main contribution is a systematic application and study in point-cloud IL.

2.Real-robot validation absent. Claims emphasize high-precision manipulation, but results are purely in simulation (RoboCasa/ManiSkill3). The paper lacks real-world experiments. I would like to see solid real-world experiments to support your claim.

### Questions
1.Coordinate bounding & periodicity. You note the mapping is periodic and requires points to lie in [−λmax/2, λmax/2]. How are coordinates normalized/cropped in multi-view reconstruction, and what happens to points outside bounds during exploration or camera drift?

2.Sensitivity to frequency design. How sensitive are gains to L, λmin, λmax? Could learned Gaussian RFF or learned sinusoidal frequencies outperform fixed log-spaced bands here? Please include a small sweep or a learned-RFF variant. 

3.Why DP3 sometimes drops. On ManiSkill3 PullCube, DP3 + FF underperforms vanilla DP3 (91.7→80.0). What failure mode explains this, and can frequency ranges be task-adapted to mitigate regressions? 

4.How does the method handle depth noise, extrinsic/intrinsic miscalibration, or partial occlusion? I would like to see solid real-world experiments to support your claim.

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
This paper proposes encoding 3D point-cloud positions with Fourier features to help an imitation-learning policy focus on geometric details. The authors conduct experiments on RoboCasa and ManiSkill3 to demonstrate effectiveness. However, the paper still lacks real-world experiments and clear, significant contributions.

### Strengths
* This paper highlights a trick that many robotics papers overlook.
* The paper is well presented and easy to read.

### Weaknesses
* The novelty is limited. While applying Fourier features is a reasonable addition to the imitation-learning network, this is an incremental contribution, and its impact is difficult to validate without large-scale real-world experiments.
* No real-world experiments. The authors evaluate only on RoboCasa and ManiSkill3, which are known for simplified physics and susceptibility to overfitting. Without convincing real-world results, it is hard to accept this as a substantial contribution to the robotics community.
* The comparison in Fig. 5 is not fair. The proposed policy is compared qualitatively to a baseline, but the two scenarios differ, making it difficult to conclude that the proposed method attends better to details.
* The evaluation appears too noisy to support strong conclusions. For example, in RoboCasa (Fig. 4), PP+FF outperforms DP3+FF, while in ManiSkill3 the opposite holds. This suggests the current benchmarking is not informative enough to determine whether the trick consistently helps.

### Questions
* Why was EDM chosen as the action-conditioned diffusion framework instead of the more commonly used DDPM? Are there specific considerations driving this choice?
* According to iCT ([https://arxiv.org/abs/2310.14189](https://arxiv.org/abs/2310.14189)), Fourier features are sensitive to hyperparameters. How were the hyperparameters selected here? The current choices appear somewhat arbitrary.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates how Fourier features help mitigate spectral bias. The method is simple — augmenting point cloud encoders with Fourier feature embeddings to enable the network to capture high-frequency geometric cues. Experiments on several tasks from RoboCasa and ManiSkill3 demonstrate consistent improvements on both DP3 and PointPatch.

### Strengths
1. The idea is straightforward and reasonable at a high level. The experiments conducted in simulation convincingly demonstrate its effectiveness across two 3D baselines (PointPatch, DP3).

2. Implementation details (such as training details) are sufficient and shows good reproducibility.

### Weaknesses
1. Lack progressive ablation studies. The work employs VariableJitter augmentation to stabilize training with Fourier features. However, this setup lacks fair and step-by-step ablation studies. There is a critical hypothesis that VariableJitter itself may contribute to the observed improvements, or it fits Fourier features. To properly isolate the effects, the paper should include step-by-step comparisons across the following variants: baseline, baseline+aug, fourier, fourier + aug.

2. Lack direct evidence of mitigation spectral bias. The improvements should be supported by direct visualization or spectral analysis between with/wo fourier.

3. Absence of real-world experiments. Real world 3D point clouds contain more noise (e.g. point sparsity, unstable depth sensors, noise or occlusion artifacts), which could lead to unstable learning of policy or sensitivity to spurious geometry. Besides, simulation engines has coarse contact resolution compared with real-world ones. Experiments on real hardware (even under noisy or partially occluded conditions) should make the results much more convincing. 

4. Limited contribution scope. The second contribution appears to be a standard validation of first contribution rather than an independent contribution. 

5. Limited scope to 3D policy. The motivation regarding spectral bias should also apply to 2D inputs. The authors' justification about RGB is sensitive to viewpoint or lighting is not convincing, especially the experiments presented in the paper are not directly related to these factors. Demonstrating method effectiveness in 2D would broden the research scope.

### Questions
See the weakness.

### Soundness
2

### Presentation
1

### Contribution
2
