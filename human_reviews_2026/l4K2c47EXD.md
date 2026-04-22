# Hierarchical Rectified Flow Matching with Mini-Batch Couplings

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Flow matching has emerged as a compelling generative modeling approach that is widely used across domains. During training, flow matching learns to model a velocity field. At inference, to generate samples, an ordinary differential equation (ODE) is numerically solved via forward integration of the modeled velocity field. To better capture the multi-modality that is inherent in typical velocity fields, hierarchical flow matching was recently introduced. It uses a hierarchy of ODEs that are numerically integrated when generating data. Each level of the hierarchy of ODEs captures the distribution of the next level, just like vanilla flow matching uses the velocity field to capture a multi-modal data distribution. While this hierarchy enables to model multi-modal distributions at any hierarchy level, the complexity of the modeled distributions remains identical across levels of the hierarchy. In this paper, we study how to gradually adjust the complexity of the distributions across different levels of the hierarchy via mini-batch couplings. We show the benefits of mini-batch couplings in hierarchical rectified flow matching via compelling results on synthetic and imaging data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Hierarchical Flow Matchining with mini-batch coupling, extending the existing HRF method. The coupling can be conducted in two levels: data distribution and velocity distribution, which is shown to gradually adjust the complexity of the distributions across
different levels of the hierarchy via mini-batch couplings.
Extensive numerical experiments are conducted to verify the efficiency.

### Strengths
The paper extends the exising HRF framework by introducing coupling, showing that the velocity distribution learnt from coupling data can recover the data distrbution as well. Through some 1D experiments, the authors show that coupling can reduce the multi-modality of velocity distribution, which is convincing. Results of large scale experiments also verify the effectiveness of proposed methods.

### Weaknesses
1. The proposed method is a naive combination of two existing methods: HRF and mini-batch coupling, and provides little new insights. It is a well known result that mini-batch coupling can help straighten the velocity and thus address multi-modality in some sense. The authors didn't provide further useful insights for why mini-batch coupling could benefit HRF other than some 1D numerical experiments. Can the authors develop some theories (even toy example is fine) to demonstrate why mini-batching coupling is good for HRF?

2. If I understand correctly, the velocity coupling requires a pretrained velocity model. This limits the practical application of the proposed method, which can only do distillation instead of training from scratch.

3. The authors only compared the proposed algorithm with FM, HRF and FM with coupling, claiming benefits in low NFE regime. The gains in high NFE regime seem quite marginal according to Figure 4(d). In terms of reducing NFE, the proposed method should also compare with other distillation/one-step FM algorithms such as [1,2]. 

[1] Geng, Zhengyang, et al. "Mean flows for one-step generative modeling." arXiv preprint arXiv:2505.13447 (2025).

[2] Frans, Kevin, et al. "One step diffusion via shortcut models." arXiv preprint arXiv:2410.12557 (2024).

### Questions
Please see weakness part.

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
This paper mainly focuses on gradually simplifying the complexity of the distributions across hierarchy in hierarchical flow matching using mini-batch optimal transport (OT) coupling. The paper empirically compares different schemes of coupling, including data coupling and joint data-velocity coupling. The paper experiments on synthetic and image datasets to show the effectiveness of the proposed method.

### Strengths
•	The paper is well-written and clearly-organized.

•	The empirical discovery that using mini-batch OT coupling in the current hierarchy level simplifies the distribution at the next level is interesting and could be considered in other scenarios of generative modeling like diffusion models.

### Weaknesses
•	The novelty of the paper is limited and incremental, the usage of minibatch OT coupling in flow matching is already proposed in [1]. The main point of the paper that using mini-batch OT inherently simplifies the velocity distribution is only intuitively explained without further theoretical mathematical evaluation.

•	The paper lacks further theoretical discussion and comparison between “data coupling” and “joint data and velocity coupling”.

•	The method is not simulation-free, which increases the training cost.

•	[1] Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, and Ricky TQ Chen. Multisample flow matching: Straightening flows with minibatch couplings. In Proc. ICML, 2023

### Questions
•	As shown in Figure 3(d), HRF2-D and HRF2-D&V performs nearly the same, with HRF2-D&V slightly better. In contrast, HRF2-D&V performs worse than HRF2-D when NFE is large on the real image datasets. How to choose between “data coupling” and “joint data and velocity coupling” given a certain task and a certain NFE? 

•	Could the authors provide a theoretical validation of “minibatch OT coupling inherently simplifies the velocity distribution on the next hierarchy” on a simple distribution like Gaussian distribution?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper extends the hierarchical rectified flow matching models by incorporating mini-batch optimal transport couplings between the source and target samples for the data and velocity. It is shown through theory and empirical evidence that the mini-batch coupling (for a large batch size) simplifies the distribution of velocity close to t=0 (closer to the source distribution) for rectified flow models. This observation motivated the authors to apply mini-batch OT in the two-level hierarchical rectified flow model (HRF2). The paper shows improvement in FID with a few NFEs for popular image benchmarks and synthetic datasets.

### Strengths
I like the presentation of the approach. The use of mini-batch OT in the two levels of hierarchical rectified flow is intuitive and well-motivated from experiments on bi-modal Gaussian data. The theoretical results are interesting and easy to follow. The experiments do show improvement in lower FID compared to baselines with low NFEs.

### Weaknesses
Major:

Novelty - The paper's main contribution is to use mini-batch OT for training the HRF2 model. The mini-batch OT has been previously used for CFM models (OT-CFM). I understand that this paper extends this to HRF models, but in my view, this is a very limited novelty.

Training cost - Flow matching training with the mini-batch OT is expensive. This method adds computing another OT map for the velocity distribution. In addition, HRF2 with velocity coupling requires a simulation from ODE flow for the velocity distribution. I would request the authors to add the training cost analysis of their method.

Other:

Suboptimal baselines - The OT-CFM results on CIFAR10 seem to have FID > 5 for NFE 100. However, Tong et al. (OT-CFM), Guo and Schwing (VRFM), and Samaddar et al. (Latent-CFM) all reported FID < 5 for NFE 100. This suggests that the baseline training or inference was suboptimal.

### Questions
1. In Fig. 1(a), after 100 steps of ODE steps, why does the velocity distribution not reach a univariate Gaussian (as shown in Fig. 2 in the original HRF paper, Zhang et al. 2025)?

2. Please enlarge the legends in the 2d sample plots, Fig. 3, top panel (two plots on the right).

3. Please report the overall performance of all the competing approaches at the final Euler step for the image benchmarks.

4. Can the author comment on the use of adaptive solvers instead of fixed-step Euler?

### Soundness
3

### Presentation
2

### Contribution
2
