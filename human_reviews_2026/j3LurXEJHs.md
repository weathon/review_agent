# Interpretable Brain-Inspired Representations Improve RL Performance on Visual Navigation Tasks

- Decision: Reject
- Scores: 6, 4, 2, 8

## Abstract
Visual navigation requires a wide range of capabilities in an agent. A crucial one is the ability to determine the agent's own location and heading in an environment. However, existing navigation approaches either assume this information is given, or use methods that lack a suitable inductive bias and accumulate error over time. Inspired by neuroscience research, the method of slow feature analysis (SFA) overcomes these limitations and extracts agent location and heading from a visual data stream, but has not been combined with modern, deep reinforcement learning agents. In this paper, we compare SFA representations with those learned by convolutional neural networks in deep RL agents. We also demonstrate how using SFA representations can improve navigation performance. Lastly, we empirically and conceptually investigate the limitations of using SFA and discuss how they currently prevent it from being used more widely for visual navigation in RL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper investigates whether “interpretable” representations learned by hierarchical Slow Feature Analysis (hSFA) can improve the performance of reinforcement learning agents (PPO) in visual navigation. It compares three representations end to end: pretrained hSFA, representations learned during training by CNNs (NatureCNN and a CustomCNN mirroring hSFA structure), and a PCA baseline. The evaluation spans four MiniWorld environments and reports both interpretability of representations (reconstructing position and heading) and navigation efficiency (average episode length). The main finding is that hSFA outperforms CNNs and PCA on the StarMaze variants, with the first 32 components supporting linear reconstruction of heading, while limitations are discussed such as lack of gradients, coverage requirements, and sensitivity to symmetry.

### Strengths
* **Clear motivation and task framing**

  * The paper targets the gap in “self-localization and heading estimation” for visual navigation and surveys limitations of three families of approaches, highlighting hSFA’s inductive bias. This sets a precise theoretical starting point and comparison frame for later design choices.
  * It emphasizes error accumulation in odometry-like integration and uses this to motivate the stability of hSFA.

* **Strong evidence for representation interpretability**

  * Visualizations of position coding show components with distinct global and local spatial sensitivities.
  * Heading can be linearly reconstructed from the first 32 components, with a clear treatment of the circular nature of angles.

* **Insightful analysis of behaviors**

  * On StarMazeArm and StarMazeRandom, hSFA exhibits goal-directed behavior that first orients then proceeds directly, contrasted with CNN and PCA agents that “chase only when visible” or wander.
  * In WallGap, environmental symmetry leads to indistinguishable positions for hSFA and the paper offers a reasonable explanation.
  * In FourColoredRooms, hSFA underperforms CNNs, suggesting CNNs remain strong at extracting certain visual cues.
  * The discussion connects performance differences to the lack of explicit planning or mapping.

* **Openness and reproducibility**

  * Anonymized code links and an environment fork are provided. Training steps, policies, and the SB3 implementation are clearly described.

### Weaknesses
* **Baselines and task settings could be stronger**

  * No memory-based visual navigation baseline such as CNN with RNN or LSTM is included to test “integration over time.”
  * There is no perceptual baseline with explicit supervision for heading and position (for example, predicting φ and (x, y) then feeding PPO).
  * Only on-policy PPO is used. There are no off-policy agents or agents augmented with planning or mapping.

* **Limited systematic analysis of coverage and symmetry**

  * Pretraining requires about 80k random trajectories and stresses coverage of position–heading combinations, but there is no degradation curve for undercoverage.
  * The symmetry analysis in WallGap is mostly qualitative and visualization-based, without an information-theoretic lower bound or formal recoverability condition.
  * There is no sensitivity threshold study for “slightly breaking symmetry,” such as adding mild texture noise.
  * There is no report of how different hSFA dimensionalities k affect success rates or errors.

* **Environment-specific pretraining**

  * hSFA must be pretrained separately for each environment and cannot be directly transferred to a new environment. This limits applicability in multi-task or real-world settings.

* **Limited representational generality**

  * The learned dimensions encode position–heading combinations specific to a given environment rather than abstract spatial coordinates or goal-relative positions, which restricts cross-environment policy generalization.

### Questions
* **Baselines and task difficulty**

  * Would the authors add a CNN+LSTM or RNN baseline within the same PPO framework, swapping only the feature head and memory module, to test the benefit of temporal integration?
  * Would the authors include an explicitly supervised localization/heading head trained on rendered ground truth, to quantify the trade-off between unsupervised interpretability and supervised performance?
  * Could the authors design occlusion, noise, and lighting perturbation experiments and report robustness curves as performance versus perturbation strength?
  * Can the authors evaluate agents that include mapping or short-horizon planning modules, reusing the same representations, to test gains in more complex settings?

* **Coverage and symmetry analysis**

  * Can the authors report degradation under undercoverage by varying pretraining samples from 10k to 80k and measuring localization error and RL performance, and provide an estimate of minimal usable coverage?
  * Can the authors give an information-theoretic lower bound or formal unidentifiability condition for WallGap and then test the minimal symmetry breaking needed to restore identifiability?

* **Transfer and generalization**

  * Have the authors considered transferring hSFA representations to new environments via joint training, environment-conditioned modulation, or hybridization with more transferable backbones such as CNNs or transformers to improve cross-environment generalization?

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
3

### Summary
This article introduces the slow feature analysis (SFA) method to deep reinforcement learning (RL) researchers as a novel feature extraction technique. The authors argue that SFA possesses several advantages for visual navigation tasks, particularly its ability to identify the agent's location and heading. They conducted experiments in four MiniWorld environments and visualized the features extracted by their proposed hSFA method alongside baseline methods. Their visualizations demonstrate that different channels of hSFA are activated at distinct spatial locations, and the extracted features exhibit a notable correlation with the agent's heading. Such patterns are not clearly observed in CNN and PCA feature extractors. The authors also discuss the limitations of SFA to provide insights for future research.

### Strengths
The SFA method exhibits clear patterns related to location and heading of the agent. The visualization clearly support the main arguments. Overall, the work is easy to follow.

### Weaknesses
The empirical results appear somewhat inconclusive. Specifically, hSFA method outperforms NatureCNN in two environments but underperforms a standard CNN in the other two. Furthermore, the experiments are conducted in environments with relatively simple physics. It remains a question whether SFA alone can scale effectively to environments with more complex physical interactions.

### Questions
1. Beyond location and heading, is the features learned by hSFA capable of represent or distinguish specific objects (e.g., walls or obstacles) within the environment?
2. How would the agent's performance change if it were provided with ground-truth location and heading directly? A comparative analysis with such a baseline may yield valuable insights into the empirical results.
3. Given that hSFA and CNNs/PCAs extract complementary types of features, what would be the potential of a hybrid feature extraction architecture for improving navigation performance?

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
3

### Summary
The paper investigates whether hierarchical Slow Feature Analysis (hSFA) can yield interpretable spatial representations—explicit location and heading—that also help RL on visual navigation. The pipeline pretrains hSFA offline from egocentric videos collected with a random policy (target removed), then freezes the extractor in a PPO agent. On Miniworld tasks, hSFA features are linearly decodable for heading and exhibit topographic spatial selectivity; PPO+hSFA outperforms CNN/PCA on localization-centric mazes (StarMaze) but does not dominate on symmetry-heavy or cue-driven tasks (WallGap/FourColoredRooms). The paper discusses why SFA helps (stable/slow spatial codes) and where it struggles (symmetries, offline coverage, lack of end-to-end gradients), and suggests gradient-based SFA and hybrid SFA+memory/mapping as next steps.

### Strengths
* **Clear motivation.** Articulates why localization-relevant inductive bias (slowness) could aid navigation versus integration-based odometry or generic CNNs.
* **Interpretability evidence.** Spatial maps and linear probes show human-readable structure for heading/position.
* **Task relevance.** Benefits materialize where self-localization is the bottleneck (StarMaze).
* **Transparent negatives.** Honest about failure modes (symmetries/partial observability; cue-centric tasks).

### Weaknesses
* **Per-layout pretraining assumption.** The hSFA extractor is pretrained separately for each environment layout. This is not an accepted assumption in standard embodied navigation and severely limits external validity. At minimum, show (i) cross-layout generalization (train on some layouts, test on unseen) and (ii) cross-environment/domain transfer (apply a pretrained extractor to totally different new environments) without re-pretraining.
* **Missing strong baselines (pose-interpretable).** Since the interpretability claim targets pose (location & heading), the most relevant comparators are explicit localization pipelines:

  * **(V)SLAM/VO → policy:** Feed oracle (or realistic noisy) pose ((x,y,\theta)) from a classical VSLAM/VO (e.g., ORB-SLAM/ORB-SLAM3) to the policy as a yardstick for “interpretable pose.”
  * **Neural-SLAM / mapping-augmented agents:** Lightweight learned mapper + policy to test whether hSFA’s gains come from having any localization/mapping signal vs. the specific SFA bias.
    If pretraining is allowed, also compare to navigation foundation/backbone encoders (large pretrained embodied vision models).
* **Limited robustness/generalization.** No tests for texture/lighting/layout changes, camera/FoV shifts, or sensor noise.

### Questions
1. **Per-layout pretraining:** Can you pretrain once across many layouts and evaluate on held-out layouts with zero re-pretraining?
2. **Cross-environment transfer:** Pretrain an hSFA extractor in Environment A and apply it to a totally different Environment B (different topology/appearance) without re-pretraining. How does PPO+hSFA perform vs. CNN/PCA and mapping/memory baselines?
3. **(V)SLAM/VO baseline:** Add a baseline based on a standard VSLAM/VO into the policy, to provide a clear "explicit" pos yardstick for your interpretability claim.
4. **Neural-SLAM / mapping baseline:** Add a learned mapping agent (e.g., lightweight Neural-SLAM/active mapping) to test whether improvements arise from having localization/mapping at all vs. the SFA bias.
5. **Navigation foundation/backbones:** If pretraining is allowed, compare against pretrained embodied encoders (navigation foundation models) to contextualize SFA’s benefits.
6. **Robustness:** Report results under texture/lighting/layout perturbations, camera FoV/noise, and mild domain shifts.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Estimating the location and heading of an agent in an environment is an important question in Reinforcement Learning setups. In this manuscript, the authors argue that features learned using Slow Feature Analysis (SFA), a method from computational neuroscience, might provide a better alternative to tackle this problem. To demonstrate the feasibility of their approach, they first train hierarchical SFA networks, along with other control networks on randomly sampled 80,000 images from five different environments. They then train PPO agents including their feature extractors on various RL environments and demonstrate the benefits of their approach.

### Strengths
Overall, the manuscript was a delight to read--it is very well written and easy to read. The authors also provide the code to replicate their results, and discuss the limitations of their approach--a rare sight in AI these days. While the SFA idea itself might not be novel, its application to RL problems is interesting and provides significant insights to the field. I have a few suggestions which, I hope, help the authors improve the manuscript in weaknesess.

### Weaknesses
1 - The manuscript argues for the use of temporally smooth or "slow" features. Thus I believe an important comparison while arguing for its benefits would be to show the breakdown of the system when the features are not slow. I am unsure exactly how one would go about implementing a Fast Feature Analysis, but maybe by altering the estimation of the temporal derivatives? I imagine instead of doing $y(t+1) - y(t)$, the authors could try $y(t+n) - y(t)$ with an increasing $n$ (hopefully) breaking the performance?  

2 - I find the justification of 80,000 images a little unsatisfactory, especially given the fact that the visual cue of the target ends up encoded. Maybe a better alternative could be to just show the performance of the models (on y-axis) across different dataset sizes (on x-axis). A plot like this will also help elucidate the sensitivity of the different methods used (maybe PCA fares better than hSFA with just 40,000 images). 

3- Since performance often also depends on the parametric sizes, it would be good have a table listing those in the manuscript/Appendix.

4 - While not directly used for RL problems, I think there are also other efforts in AI that aim to implement similar constraints (in essence). The authors could have a look at and maybe better link their work to those of Contrastive Learning Through Time, or even models like V-JEPA, etc. I believe there are also efforts to link Object Permanence and Temporal smoothness in the literature. 

Minor : 
1 - There are a couple of typos in the article.

### Questions
I have written several suggestions in weakness section to hopefully improve the paper.
Overall, the manuscript offers a compelling approach to learn representations for RL problems and, with some revisions, could be well-positioned for publication.

### Soundness
3

### Presentation
3

### Contribution
3
