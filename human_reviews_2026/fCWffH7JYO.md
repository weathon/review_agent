# Analyzing the Role of Spinal Joint Dynamics in the Movement of a Sprawling Robot

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 2, 8

## Abstract
Sprawling locomotion in vertebrates, particularly salamanders, demonstrates how body undulation and spinal mobility enhance stability, maneuverability, and adaptability across complex terrains. While prior work has separately explored biologically inspired gait design or deep reinforcement learning (DRL), these approaches face inherent limitations: open-loop gait designs often lack adaptability to unforeseen terrain variations, whereas end-to-end DRL methods are data-hungry and prone to unstable behaviors when transferring from simulation to real robots. We propose a hybrid control framework that integrates Hildebrand’s biologically grounded gait design with DRL, enabling a salamander-inspired quadruped robot to exploit active spinal joints for robust crawling motion. Our evaluation across multiple robot configurations in target-directed navigation tasks reveals that this hybrid approach systematically improves robustness under environmental uncertainties such as surface irregularities. By bridging structured gait design with learning-based methodology, our work highlights the promise of interdisciplinary control strategies for developing efficient, resilient, and biologically informed spinal actuation in robotic systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates how active spinal joint dynamics influence the locomotion performance of sprawling quadruped robots inspired by salamanders. To this end, the authors design a salamander-like quadruped robot with four legs (each with two degrees of freedom) and an optional one-degree-of-freedom spinal joint for lateral bending. The robot’s legs are controlled using a biologically inspired Hildebrand gait, while the spinal joint is adaptively modulated by a deep reinforcement learning (DRL) policy to improve locomotion efficiency and robustness. The experiments compare four locomotion strategies—Hildebrand-only, RL-only, torque-limited RL (RL*), and the hybrid Hildebrand + RL approach—in both simulation (MuJoCo) and real-world hardware. The results demonstrate that the hybrid Hildebrand + RL framework achieves the best balance between stability and adaptability, outperforming both purely model-based and purely learning-based baselines in reaching the target efficiently.

### Strengths
- Biologically grounded motivation with strong interdisciplinary reasoning.
- The writing is clear and well-organized, making the technical content easy to follow for readers from both robotics and machine learning backgrounds.
- The experiments are comprehensive, demonstrating results in both simulation and real-world hardware, which strengthens the paper’s practical relevance and credibility.

### Weaknesses
Novelty: The paper’s use of a hybrid controller combining a biologically inspired gait (CPG/Hildebrand) with deep reinforcement learning is not clearly novel. For example, CPG‑RL: Learning Central Pattern Generators for Quadruped Locomotion (Bellegarda & Ijspeert, 2022) already integrates central pattern generators with RL for quadruped locomotion.The paper under review should better articulate how its contribution exceeds or differs from prior CPG + RL frameworks.

Results are not fully convincing: In the current era, complex quadruped systems can be trained in simulation and transferred to real hardware with relative success.Thus it is hard to be convinced that the relatively simpler robot system in this paper faces major sim-to-real difficulty, or that the added spinal joint truly drives the transfer gap.

Ambiguity in focus: The paper’s title and introduction suggest an analysis of spinal joint functionality (inspired by animal biology) via combining biomechanical insight and learning. Yet in the current version the emphasis appears to be mostly on “how using RL to control an extra DoF (spinal joint) affects policy training/performance.” This mismatch creates unclear expectations: is the goal biological insight or robotic control improvement? The clarity of the paper’s scope and claims would benefit from refining.

### Questions
N/A

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
4

### Summary
This paper studies locomotion control in a salamander-inspired quadruped robot with an articulated spine joint in addition to 8 leg joints (2 per leg).  Several methods are compared: A classical Hildebrand-style open-loop trajectory, a pure RL controller, and a hybrid that uses a Hildebrand trajectory for the legs but RL for the spine.  Experiments include both simulation and hardware execution, and both smooth and rough terrain in both cases.  The results suggest that the hybrid approach is particularly useful in the presence of rough terrain.

### Strengths
- To my knowledge, an articulated spine is an interesting and under-explored form factor for quadruped robots.
- Overall the paper is clearly written and easy to follow.
- The paper considers a fairly wide range of different experimental conditions, and averages performance over multiple runs to bolster reproducibility.

### Weaknesses
- I am not sure how relevant this paper is to ICLR, which is focused on deep learning architectures.  This paper is primarily focused on robotics, not deep learning.  It does use reinforcement learning, but does not define the network architecture of the agent and value function, nor does it analyze their learned representations.  The robot form factor and hybrid controller may be novel, but as far as I can tell, there is no novelty in terms of the deep learning architectures used (assuming they used standard MLPs common in PPO and SAC).
- The paper reports difficulties with sim-to-real transfer, but does not mention using any domain randomization during training, which is standard practice to improve robustness.  Also, if I understood correctly, torque limits are applied at inference time but not during training, which introduces an unnecessary distribution shift that might negatively impact performance.  This raises a concern that the results may not reflect best practices in RL and might be biased by sub-optimal implementation.
- The paper reports overall task performance using the actuated spine, but would benefit from a deeper analysis of the spine behavior, since that is the primary novelty of the work.  In particular, it would be interesting to plot the trajectory of the spine over time, quantify its amplitude and periodicity, etc.
- There are some minor presentation issues that could be improved:
    - The authors should use \citep for citations where the author name is not also a noun in the sentence, so that the parentheses enclose the author name
    - Lines 160-161: The full names of SAC and PPO should be introduced immediately before their first occurrences, with citations for each
    - For better image quality, Figure 4 should use a latex table and vector graphics image instead of two raster images
    - Line 217 is missing a space after a comma

### Questions
- The paper mentions that various hyper parameters were "tuned to optimize performance," but not how the tuning was done.  Was it manual? A grid search, and if so, over what ranges? Using a software library like optuna?  The answers should also be included in the main text.
- I am surprised the RL performance degraded so significantly when introducing only one additional action dimension for the spinal joint.  Can the authors provide any evidence-based insight into why this effect was observed?  Is it possible that with more tuning or a different architecture the pure RL would work better?

### Soundness
2

### Presentation
3

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
This paper investigates the functional role of active spinal joints in sprawled quadruped locomotion, inspired by salamanders, through a hybrid control framework that combines biologically grounded gait design (Hildebrand method) with deep reinforcement learning (DRL). The authors develop a 9-DoF salamander-like robot, modeled and 3D-printed based on amphibian biomechanics, and compare locomotion across several control strategies—pure Hildebrand gaits, unconstrained DRL, torque-limited DRL, and a hybrid Hildebrand+RL approach.

The study evaluates locomotion performance on flat and rough terrains within MuJoCo simulation and partially on the physical robot, focusing on stability, goal accuracy, and traversal speed. Results demonstrate that pure RL policies produce unrealistic “cheetah-like” behaviors unsuited to the physical platform, while the hybrid model achieves robust and efficient crawling with ~38% faster traversal speed in real-world trials.

### Strengths
* The salamander robot platform is novel

### Weaknesses
* Scope of contribution: The paper’s primary contribution lies more in the robotics and bio-inspired locomotion domain than in advancing the machine learning methodology itself. The deep reinforcement learning component relies largely on off-the-shelf algorithms (e.g., SAC) without introducing novel learning techniques or insights that would meaningfully impact the learning community.
* Similar salamander-inspired or sprawling robot designs have been explored in prior literature, and RL-based locomotion frameworks for such morphologies are already well established. To further strengthen the work, the authors could consider integrating GPU-accelerated simulation or parallel training pipelines to improve policy robustness and sample efficiency, following approaches such as [1].

[1] Qu, Tomson, et al. "Versatile Locomotion Skills for Hexapod Robots." 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024.

### Questions
See weakness

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work introduces a hybrid control framework that combines Hildebrand’s biologically inspired gait design with deep reinforcement learning to enable robust, salamander-like crawling in quadruped robots. Experiments across various robot configurations show that this approach enhances stability and adaptability under environmental uncertainties, demonstrating the benefits of integrating structured gait design with RL.

### Strengths
The paper proposes a novel hardware platform and demonstrates the benefits of its hybrid approach by showing positive transfer from sim-to-real while also outperforming classical techniques. The paper is well written and easy to follow. I also enjoyed the discussion on the limitations and future work.

### Weaknesses
I mainly see two weaknesses. First, while the paper is a good contribution to the robotics community, I am unsure about its fit to the ICLR venue. Overall, the learning methods applied here are not very novel. In my opinion, the main contribution of the paper lies on the hardware side. Second, I was wondering how the sim parameters were set; moreover, was a system ID of the sim parameters performed based on a small fraction of real data?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
