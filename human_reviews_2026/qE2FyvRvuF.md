# WMPO: World Model-based Policy Optimization for Vision-Language-Action Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Vision-Language-Action (VLA) models have shown strong potential for general-purpose robotic manipulation, but their reliance on expert demonstrations limits their ability to learn from failures and perform self-corrections. 
Reinforcement learning (RL) addresses these through self-improving interactions with the physical environment, but suffers from high sample complexity on real robots.
We introduce World-Model-based Policy Optimization (WMPO), a principled framework for on-policy VLA RL without interacting with the real environment.
In contrast to widely used latent world models, 
WMPO focuses on pixel-based predictions that align the "imagined" trajectories with the VLA features pretrained with web-scale images.
Crucially, WMPO enables the policy to perform on-policy GRPO that provides stronger performance than the often-used off-policy methods.
Extensive experiments in both simulation and real-robot settings demonstrate that WMPO (i) substantially improves sample efficiency, (ii) achieves stronger overall performance, (iii) exhibits emergent behaviors such as self-correction, and (iv) demonstrates robust generalization and lifelong learning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a system to train a VLA via RL on trajectories from a world model. A video diffusion model is transformed into a world model by additionally conditioning upon actions. For the reward model, a videoMAE model is finetuned on binary classification on successful/failed trajectories, and the model output probability is used as a reward. The used RL algorithm is GRPO, where the initial frame comes from a real dataset, and the trajectories in the cohort are different sampled videos starting from the same frame. For baseVLA, openvla-OFT is chosen. Across both sim and real, this method beats baselines, and the authors also show that the method improves generalization.

### Strengths
- The idea is simple and clearly explained
- The authors show solid improvement on sim and real

### Weaknesses
- only a single base-VLA is used (openvla-OFT)
- More real-world experiments would be good
- missing comparison to DreamGen
- the training of the world model lack details

### Questions
1. What is meant by 50/1000 steps?
2. Can you compare with PPO as well?
3. Can you try multiple different VLAs?
4. Can you do more real-world experiments?
5. Can you add more details regarding how the world model was trained? Hyperparameters, datasets and so on.

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
2

### Summary
The paper proposes World Model-based Policy Optimization (WMPO), a framework for improving Vision-Language-Action (VLA) models via reinforcement learning inside a learned pixel-level video world model. The core idea is to avoid costly real-world robot interactions by performing on-policy RL (specifically GRPO) entirely within an autoregressive video diffusion model pretrained on large-scale robotic data and fine-tuned on downstream policy rollouts. The framework includes: A pixel-space world model for visual fidelity; Policy Behavior Alignment to adapt the world model to policy-induced state/action distributions; A learned binary reward model for task success; Noisy-frame conditioning and frame-level action control to mitigate long-horizon prediction drift; Dynamic Sampling during GRPO to avoid vanishing gradients in sparse-reward settings. Experiments in MimicGen simulation and real-world tasks (e.g., “insert square into stick”) show consistent improvement over imitation learning baselines and offline RL methods such as DPO and limited online GRPO.

### Strengths
1. The paper addresses a critical bottleneck in VLA RL: sample inefficiency and brittleness of imitation learning. While prior works (e.g., RT-2, OpenVLA) have shown impressive generalization, they remain confined to IL and struggle to recover from failures. WMPO’s goal — learning to self-correct via on-policy RL in a world model — is both ambitious and well-justified.

2.  The authors’ fine-tuning of the world model on policy-generated trajectories is a principled way to close the distribution shift between expert demonstrations and actual policy rollouts.

3. The qualitative results showing self-correction are compelling. This is not merely improvement in success rate but evidence of learning novel recovery strategies absent in demonstrations.

### Weaknesses
1. Overclaiming of “On-Policy” Scalability Without Real Costs: While WMPO avoids real-world rollouts during optimization, it still requires 128–1280 real trajectories to fine-tune the world model and initialize policy behavior alignment. This is not zero-shot or low-data RL — it is offline world-model RL with modest real data.
Recent works like **IRASim** [1] and **World4RL** [2] also use diffusion world models but start from far fewer real trajectories (e.g., 50–100).
The paper does not compare directly to these methods, making it unclear whether the gains come from GRPO or from the specific world-model design.

2. Reward Model is a Black Box with No Robustness Checks. The reward model is trained on binary success labels from real trajectories. But: What if the world model generates semantically correct but visually shifted outcomes (e.g., object slightly displaced)?
There is no evaluation of reward hacking — e.g., does the policy learn to “fool” the reward model by generating plausible but incorrect motions?

3. Comparison to Offline RL is Weak. The DPO baseline is implemented naively: it uses trajectory-level preferences but does not leverage recent advances. Why not compare to IQL + diffusion policy? Also, DPO’s poor performance on background disruption (Tab 2) may reflect overfitting to visual cues, not the inherent weakness of offline RL.

4. The quality of experiments: One task, 30 trials for the real robot task, underpowered for statistical significance. Missing key ablations (e.g., world model w/o policy alignment, latent vs. pixel space, reward model threshold sensitivity).

[1] IRASim: A Fine-Grained World Model for Robot Manipulation 

[2] World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation

### Questions
Please address the aforementioned weakness as thoroughly as possible.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes WMPO (World-Model-based Policy Optimization), an on-policy RL framework for VLA models that replaces costly real-robot rollouts with trajectories “imagined” by a pixel-space video world model. Key ingredients are: (1) a diffusion video world model with noisy-frame conditioning for robustness and frame-level action control for precise action–frame alignment; (2) policy behavior alignment, i.e., fine-tuning the world model on the policy’s own rollouts to match failure as well as success states; (3) a lightweight clip-based reward model for outcome (0/1) labeling; and (4) on-policy GRPO training with dynamic sampling and no KL term (no reference model). The method aims to keep the visual state space aligned with pretrained VLA encoders by decoding back to pixels. Experiments on Mimicgen show consistent gains over GRPO and DPO baselines at two real-rollout budgets (e.g., mean SR 47.1% vs. 37.3% at P=128 and 57.6% vs. 42.4% at P=1280). WMPO also reports emergent self-correction, improved generalization under spatial/background/texture shifts, lifelong learning improvements via alternating policy/world-model updates, and a real-robot result on “Insert the square into the stick”

### Strengths
Clear, modular recipe: world-model rollouts + outcome classifier + GRPO, with practical choices (pixel-space decoding to match VLA features; noisy-frame conditioning; frame-level action injection). The write-up is concrete and reproducible. 


Consistent empirical gains over strong baselines across four Mimicgen tasks and two rollout budgets; improvements grow with budget (data-efficiency + scaling). 


Behavioral insights: convincing qualitative evidence of self-correction and reduced “getting stuck,” with trajectory-length analysis. 


Generalization: better robustness under spatial/background/texture disruptions than baselines.

### Weaknesses
Heavy compute / practicality: training uses 32× H100 for world-model/WMPO phases (plus 8× H100 for SFT). The paper would benefit from wall-clock, throughput, and ablations on smaller budgets/hardware. 

Model-world fidelity & safety: while qualitative results are strong, there’s limited quantitative assessment of rollout fidelity (e.g., per-step action-conditioned metrics), failure taxonomy, or safety constraints—especially since outcome-only rewards can reward shortcuts.

### Questions
1. please try to handle these weaknesses

2. Do you try to use some PeFT methods for low-resource environments? Maybe these discussions will help

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
WMPO proposes a novel policy finetuning method without using online interaction by using a world model on the pixel space, which allows robust finetuning of vision-language-action models on imagined trajectories. There are a few desirable properties of this method compared to traditional world modeling and finetuning objectives:
1. The world model can directly be trained in pixel space, which allows a pixel based policy to interface with the world model without extra decoders.
2. The method can be completely open sourced, and can be adapted into other distributions. 
3. This method allows on policy finetuning of VLAs using GRPO without any real-world demonstrations or preexisting datasets.
The authors then used GRPO to perform finetuning of a base OpenVLA-OFT policy, which shows desirable success rate, scalability, and robustness in RoboMimic. On the real world, WMPO also shows better performance compared to other finetuning methods such as direct GRPO and DPO.

### Strengths
1. The method section of the paper is concise and informative. 
2. I believe that there are sufficient ablations being done on the method, and I like that the authors have demonstrated good robustness of the method when dealing with OOD settings and desirable scalability.

### Weaknesses
1. I believe that the paper did not address how this method can be extended into a generalist setting. The scope of the environment is also rather limited, albeit there are adequate ablations being conducted.
2. In addition, the paper used OpenVLA-OFT as the base policy. This again limits how much promise the method can bring to generalist policies. If the authors can provide additional ablations without using OpenVLA-OFT, I believe this can strengthen the paper.
3. I believe that the paper did not address the question of how adaptable this method is when concerned with language-conditioned settings (even though OpenVLA is language conditioned, there are no mention of how to use language labels in this paper), furthering this discussion can also be beneficial to the rating of the paper.

### Questions
1. I might have missed this, but when you are increasing the rollout budget, do you use the larger set as well to finetune the world model?
2. One reason for not using real-world demonstrations at all is due to it being more expensive. Have you considered using a few real-world demonstrations as regularization in your method, and if so, how does it compare to only using imagined trajectories?
3. It seems that the authors chose to use on policy RL because of flaws with off policy RL [1] and propagating the correct value. Are there going to be any potential concerns when implementing such a method in the off-policy setting?

Minor remarks:
1. The main figure stated that BC from human demonstrations cannot achieve self-corrective behavior, but this is not correct when the training data contains corrective behavior. Similarly, you can do on-policy RL in real world if you can run it online.
2. Follow up on q3, it would be better to show differences in scalability if you have to use an off policy setting.

References:
[1] Park, S. et al., 2025. “Horizon Reduction Makes RL Scalable.” NeurIPS.

### Soundness
3

### Presentation
2

### Contribution
2
