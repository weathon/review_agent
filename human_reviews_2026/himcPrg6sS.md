# Leveraging Physics-Based Models for Rapid Adaptation in Reinforcement Learning

- Decision: Reject
- Scores: 2, 4, 8, 2

## Abstract
A central challenge in reinforcement learning (RL) is achieving agents that generalize and adapt to new tasks and conditions. Many works address this via offline RL which is constrained by dataset coverage, or online RL which requires costly and potentially unsafe exploration. We propose a framework for rapid adaptation of RL agents by augmenting model-based RL with physics-informed data augmentation. Specifically, we use lightweight analytical models to generate stable, physics-grounded rollouts that complement real interaction data and allows the model-based RL agent to adapt in just a few trials. We validate our approach in autonomous racing, an extreme testbed with fast dynamics and strict safety constraints, using Assetto Corsa paired with lightweight vehicle models for data augmentation. Across diverse tracks and surfaces, our method achieves faster convergence, lower lap times, and fewer incidents than a set of strong baselines.
Although demonstrated in racing, our framework is domain-agnostic, offering a practical path to data-efficient control wherever simple models exist as priors.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a model-based framework for offline-to-online learning that leverages synthetic data that is generated from numerical solutions to dynamics ODEs (i.e. first-principles models).

The main contributions of the paper are extensive evaluation of the proposed method in the “Assetto Corsa” racing simulator. The authors also provide a faster extension of the simulator.

### Strengths
* The paper is easy to follow.
* The empirical results are convincing -- the method indeed works well in the Assetto Corsa Gym

### Weaknesses
Novelty: The method appears to have been proposed in several previous works. The primary contribution here lies in leveraging ODEs as a data augmentation technique, which differentiates this study from earlier approaches.

Specificity: While the empirical results are impressive within the race-driving context, the experiments are limited to this specific domain. Broader validation across more applications would strengthen the generality of the findings.

Assumptions: The approach assumes that the given ODEs accurately capture real-world dynamics. However, in practice, the sim-to-real gap remains a fundamental challenge when relying on first-principles models. Although the experiments provide evidence that this challenge can be mitigated, a more explicit discussion of this assumption and its implications would strengthen the paper.

A minor weakness is the repeated mention of safety concerns, which are indeed critical in race driving, without explicitly addressing this challenge.

In a broader sense, in many real-world scenarios, writing down the ODEs that govern the dynamics is too hard.
A related and potentially more impactful direction would be to augment data from physics engines that solve the underlying dynamics. A key research question then arises: how can one augment simulation-generated data intelligently, without introducing bias toward the simulated environment and thereby amplifying the sim-to-real gap?

### Questions
When you train SAC from scratch, do you augment it with the ODE data? What Update to Data ratio do you choose for it?

### Soundness
3

### Presentation
2

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
This paper presents PIA-MBRL (Physics-Informed Augmentation with Model-Based RL), a framework for rapidly adapting reinforcement learning agents to new tasks by integrating lightweight analytical models into a model-based RL (MBRL) pipeline. The key insight is to use stable ODE-based vehicle models to generate physics-grounded synthetic rollouts, which are merged with limited high-fidelity simulator data (Assetto Corsa Gym) to improve sample efficiency and generalization. The approach builds on TD-MPC2 and the offline-to-online adaptation paradigm of FOWM. Tested in diverse conditions within the autonomous car racing scenario, the method achieves faster convergence and improved task performance. 

Experiments in autonomous racing show that physics-informed augmentation enables agents to adapt to unseen tracks and surface conditions (e.g., dusty, low-grip) within a few episodes. The authors release over 1000 hours of racing data and ODE rollouts, along with an enhanced version of ACGym that supports dynamic model swapping and JAX-based real-time control.

### Strengths
- Justified design choice: Uses analytic ODE models as reliable, long-horizon data generators for MBRL. This idea is well-suited for the racing task.

- Engineering contribution: Extends ACGym with Linux support, modular dynamics swapping, and real-time TD-MPC2 in JAX.

- Extensive eval Thorough experiments across 15 tracks, 3 surface conditions, and multiple baselines (SAC, IQL, FOWM, TD-MPC2).

### Weaknesses
- Limited novelty in algorithmic form: Builds directly upon existing frameworks (TD-MPC2, FOWM), with the main innovation being the physics-based data source.
- Also, the methodology is only tested on a single task in a simulation environment. Based on the limited domain demonstrated, the paper's title should also incorporate the specific domain, e.g., add "...for autonomous racing"
- Another disadvantage of using a strong model prior assumption is the unmodeled effects. ODE models use fixed parameters and do not adapt to varying surface frictions or uncertainties. Despite being more efficient in terms of learning efficiency and even final performance, I do see that this approach has rather limited expressiveness when encountering more complex models.
- Generalization claims: While racing is a strong testbed, broader applicability (e.g., robotics or UAVs) remains unverified.

Apart from the methodology section,
- Sim2real verification: An alternative way to significantly improve the paper is to demonstrate that the approach can perform some sim2real verification to demonstrate the approach's effectiveness (not even necessarily for car racing at all, it can be even simpler tasks). This will strongly support the model learned, which can handle real-world scenarios.

### Questions
See the weaknesses above. I will be happy to see the paper further improved.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes an approach that leverages low-fidelity simulators to achieve more sample-efficient reinforcement learning on real systems. The algorithm first trains a policy offline using a dataset composed of both simulated and real-world data. This pretrained policy is then fine-tuned through online learning on the real system. The authors demonstrate that incorporating simulated data substantially improves key RL metrics such as safety and sample efficiency. They evaluate their method using the Assetto Corsa simulator as a proxy for the real system and employ an advanced bicycle model (including aerodynamics, drivetrain, and Pacejka tire dynamics) as the low-fidelity simulator. Finally, they conduct several ablation studies and experiments to validate the design choices and highlight the benefits of their approach.

### Strengths
The paper is clearly written and easy to follow. Each design decision is thoroughly ablated. The authors present a comprehensive set of experiments on the Assetto Corsa simulator. The simplicity of their method is another strong point.

### Weaknesses
It is not clear how similar the dynamics of the Assetto Corsa simulator are to those of their bicycle model.
In the result tables, the authors report only mean values; however, many of the numbers are quite close, and it would be important to include measures of variability (e.g., standard deviations or confidence intervals) to assess whether the reported improvements are statistically significant or fall within the range of uncertainty.
Several related works in the literature address similar problems and should be discussed in the related work section [1, 2, 3, 4].
Finally, evaluating the proposed method on a broader set of dynamical systems, as mentioned in the paper, would strengthen the empirical evidence for its general applicability.

[1] Rothfuss, Jonas, et al. "Bridging the sim-to-real gap with Bayesian inference." 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024.
[2] Hwangbo, Jemin, et al. "Learning agile and dynamic motor skills for legged robots." Science Robotics 4.26 (2019): eaau5872.
[3] Ha, Sehoon, and Katsu Yamane. "Reducing hardware experiments for model learning and policy optimization." 2015 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2015.
[4] Pastor, Peter, et al. "Learning task error models for manipulation." 2013 IEEE International Conference on Robotics and Automation. IEEE, 2013.

### Questions
Could you perform an ablation study on your simulator? It appears that you used a rather advanced bicycle model. Would a simpler model (e.g., one that does not model tire slippage or Pacejka tire dynamics) lead to similar learning performance?

In Figure 6, there seems to be a trade-off between the proportion of simulator data and real data used in the offline phase. Could you, for example, fix the amount of Assetto Corsa (AC) data and plot the performance as the amount of ODE data increases—that is, show how performance scales with a growing simulator dataset while keeping the real data constant?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes PIA-MBRL (Physics-Informed Augmentation for Model-Based Reinforcement Learning), a framework for efficient MBRL with physics informed models. The approach brings together many techniques from the MBRL literature. The paper demonstrates the approach on primarily in a high-fidelity driving simulator, and demonstrates that a simplified ODE model can be used for effective data augmentation in this setting.

### Strengths
- The paper evaluates methods across a wide range of different tracks
- The paper provides an open source simulation environment for future work

### Weaknesses
$\textbf{Clarity}$: After two close readings of the paper, I do not understand the details of the method or the logic behind the approach. The paper appears to be bringing together a number of different techniques, such as offline to online fine-tuning, online planning with a model (via TDMPC-2), Dyna-style data augmentation and physics informed models. The draft mentions that each of these techniques are used at a high level, but does not explicitly lay out exactly how these pieces fit together. 


$\textbf{Motivation}$: The method appears to be a “soup” of many existing techniques, and seems overly complex to implement. The paper does little to explain why all of these different techniques are needed together. There are no clear technical take aways from the paper that came through during my reading.

### Questions
- Can the authors clarify how each of the components of the framework fit together? 


- Why do we need all of these pieces together? 


- Why not use a simpler approach, and simply do MPC with a learned physics-based model like in numerous prior works?

### Soundness
1

### Presentation
1

### Contribution
1
