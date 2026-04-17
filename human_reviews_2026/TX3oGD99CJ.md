# HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist Vision–Language–Action Policies

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
The development of foundation models for embodied intelligence critically depends on access to large-scale, high-quality robot demonstration data. Recent approaches have sought to address this challenge by training on large collections of heterogeneous robotic datasets. However, unlike vision or language data, robotic demonstrations exhibit substantial heterogeneity across embodiments and action spaces as well as other prominent variations such as senor configurations and action control frequencies. The lack of explicit designs for handling such heterogeneity causes existing methods to struggle with integrating diverse factors, thereby limiting their generalization and leading to degraded performance when transferred to new settings. In this paper, we present HiMoE-VLA, a novel vision–language–action (VLA) framework tailored to effectively handle diverse robotic data with heterogeneity. Specifically, we introduce a Hierarchical Mixture-of-Experts (HiMoE) architecture for the action module which adaptively handles multiple sources of heterogeneity across layers and gradually abstracts them into shared knowledge representations.
Through extensive experimentation with simulation benchmarks and real-world robotic platforms, HiMoE-VLA demonstrates a consistent performance boost over existing VLA baselines, achieving higher accuracy and robust generalization across diverse robots and action spaces.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
To tackle heterogenity in robot learning, this paper presents HiMoE-VLA. Which shares the pipeline with $\pi_0$ but changes the action expert to be a novel architecture named Hierarchical MoE. It contains Action-Space Regularization (AS-Reg) and Heterogeneity-Balancing Regularization (HB-Reg), where experts are dealing with action space heterogeneity and sharing common knowledge separately. Experimental results across 2 simulators and 2 real world robots (one is single arm and the other is dual arm) are sufficient.

### Strengths
- The author claims heterogeneity is crucial in robot learning and introduces the Hierarchical MoE method to tackle this.
- Experimental results across 2 simulators and 2 real world robots (one is single arm and the other is dual arm) are sufficient.
- All figures (except for figure 1) are clear and ablation studies are adequate.

### Weaknesses
- The performance is poor on CALVIN benchmark, the current SOTA method FLOWER (CoRL 2025) can reach 4,35 in D->D setting, while HiMoE-VLA is only 3.967.
- Figure 1 shows an overview of HiMoE-VLA. However, I cannot see any contributions in this figure, the overall pipeline is the same as $\pi_0$. And the proposed Hierarchical MoE is summarized with only a whole black block. This picture does not bring any useful information about Hierarchical MoE itself.



references:
- [1] FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies

### Questions
- The test setting is not straightforward. Since the author claims that HiMoE-VLA can tackle with different action space and robotic heterogeneity, then why not train a unified model with OXE, ALOHA and the testing benchmark CALVIN and LIBERO? In appendix C, the author says to fine-tune HiMoE-VLA "separately for each of the four task suite".  More direct experiments are needed to prove that HiMoE-VLA is good at handling heterogeneity.
- What about applying the proposed Hierarchical MoE to other domains? like images or languages. The proposed Hierarchical MoE looks general and can still make sense when action modality is not mentioned here.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a new VLA framework for handling heterogeneous robot data - including different action spaces as well as different embodiments and sensor configurations. The authors validate their method across a variety of simulated and real benchmarks and provide ablation studies to justify their design choices.

### Strengths
- The paper tackles an important problem of being able to learn robot policies from varies sources, including variations in embodiments and action spaces.
- The paper introduces two regularization components - one to handle variability in action spaces while the other handles knowledge sharing across embodiments.
- The authors conduct extensive experiments across both simulation and the real world to validate the usefulness of the proposed framework.
- The authors include detailed ablation studies to justify the design choices made in the proposed framework.

### Weaknesses
Including both weaknesses as well as questions tied to the weaknesses below.
- The authors mention being able to handle different sensor configurations (line 89). I am confused which of the experiments validate the variability in sensor configurations.
- It seems that the different action spaces are first projected into a unified vector representation, where each action space is consistently assigned to fixed positions within the vector. I am curious whether the network’s performance is sensitive to the dimensionality of each action space. In the current setup, the dimensions are relatively similar (7 for the single-arm setting and 14 for the bimanual case). However, if one were to train a single policy across embodiments with substantially higher-dimensional action spaces—such as 5-fingered hands or humanoids—would the same strategy still be effective? I would be interested to hear the authors’ perspective on this, acknowledging that it may be difficult to conduct such experiments within the rebuttal period.
- Missing citation in line 241.
- How many demonstrations are used for CALVIN and LIBERO? Also, does D->D on Calvin mean identical train and test settings? Further, in Tables 1 and 2, are all baselines finetuned on the same CALVIN and LIBERO datasets?
- The paper must discuss the limitations of the proposed method.

### Questions
It would be great if the authors could address questions in the weaknesses section.

### Soundness
3

### Presentation
3

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
The paper proposes a hierarchical mixture-of-experts (MoE) approach for a VLA framework aimed at robust transfer across diverse robot embodiments. To handle heterogeneous action spaces, two MoE modules—AC-MoE and HB-MoE—are introduced within a diffusion-based action module. For the vision–language component, a pretrained VLM is employed to extract intermediate key–value (KV) representations, which are fed into the action module. Experiments show that the proposed approach outperforms baselines on a range of manipulation tasks across four environments.

### Strengths
The work targets a generalist VLA policy applicable to multiple robot embodiments and environments, a direction with broad practical impact. The network design and the two specialized MoE modules are intuitive and well-motivated. Empirically, the method consistently outperforms strong baselines in two simulated and two real-world settings.

### Weaknesses
- The method appears sensitive to the choices of hyperparameters K and N (according to Table 6(c)).

- In the ablations, success rates are quite close. Considering the high variance over different runs, the incremental benefit of the proposed MoE modules seems to be marginal.

- Several details are unclear to me. Please refer to the questions.

### Questions
- HB-MoE appears very similar to DeepSeekMoE (Dai et al., 2024) in both architecture and training objective. What novel aspects distinguish HB-MoE from DeepSeekMoE?

- Is the specialization of MoE modules preferable to duplicating a single MoE module with the two regularization losses? Including this comparison in the ablation study would better support the hierarchical design choice.

- Some technical details are missing: How are heterogeneous actions encoded into a fixed-size vector? How are gating scores computed in the MoE modules (e.g., inputs, normalization, and temperature settings)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper targets a unified visuomotor-language policy that transfers across heterogeneous robot datasets with differing action spaces, embodiments, and sensors. It proposes HiMoE-VLA, a hierarchical MoE design with AS-MoE to handle discrepancies in action representations (e.g., joint vs. end-effector) and HB-MoE to balance broader cross-domain heterogeneity while retaining a shared transformer backbone; training combines flow-matching with specialization and balancing regularizers. Evaluations on CALVIN, LIBERO, and real-robot setups show consistent gains over prior VLAs.

### Strengths
- Tackles an important problem: transfer across diverse robot datasets and action spaces.
- Writing is generally clear; only minor phrasing/citation fixes needed.
- Strong results across sim and real benchmarks, outperforming existing VLAs.

### Weaknesses
- Core technical claims (e.g., non-transferability across action spaces) and architectural choice (MoE vs. simpler sharing/separate heads) aren’t fully justified (see questions).
- Limited insight into learned expert specialization/routing.
- Minor issues:
    - broken citation L249
    - CALVIN aggregation is sum and not avg as mentioned in the table headers.

### Questions
1. L60: Why are data from different action spaces “largely non-transferable”?
2. If the goal is to avoid cross-action sharing, why not use separate heads per action representation (or something like GR00T with an embedding indicating embodiment type as input)? If the goal is to share observation representations, why not fully share parameters (as in other VLAs)? Note this should be studied as an ablation of the method to avoid confounding factors from other sources like pre-training dataset, hyper-parameter tuning, etc.
3. What is the role of the shared expert in HB-MoE?
4. Can you provide expert routing analysis for AS-MoE and HB-MoE across datasets with different action spaces and observations?
5. What exactly is “MoE re-initialization during fine-tuning” (L240)?
6. In Table 6(b), are active parameters matched between “Full” and “w/o MoE,” or total parameters?
7. Will the codebase and dataset be open-sourced?
8. It is noteworthy that the ‘w/o MoE’ variant in Table 6(b) already outperforms all VLAs in Table 1. A breakdown of which elements of the training pipeline contributed most, perhaps in the appendix, would be valuable to the community.

### Soundness
2

### Presentation
3

### Contribution
3
