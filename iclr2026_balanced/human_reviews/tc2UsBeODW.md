## Human Reviewer 1

### Summary
This paper studies the role of VLMs as backbones for VLA models. The authors fine-tune 7 open VLMs on robot simulation data and evaluate them across three benchmarks (Calvin, SimplerEnv, Libero). Their results show that a simple architecture and training objective can achieve competitive performance against recent models with specific VLMs. The authors also show that performance on Calvin is correlated with the VLM’s VQA performance, and that fine-tuning on VQA data from robot tasks does not help improve performance. Finally, the authors show the importance of fine-tuning the vision encoder of the VLM.

### Strengths
1. A meta-analysis of the role of VLMs in VLA models, showing competitive performance with a simplified training framework.
2. A study of the relationship between robot task performance and generic VQA performance.
3. A study of the (lack of) usefulness of VQA data extracted from robot data.
4. An analysis of the importance of fine-tuning the vision encoder, likely due to the domain mismatch between pretraining and robot data.

### Weaknesses
1. While the experiments show that some VLMs can achieve competitive performance in the VLM4VLA framework, the paper lacks specific guidelines and insights into why specific models perform better. While the authors found VQA performance to be predictive of Calvin performance, this was not the case for the other benchmarks. This left me wondering how I would choose the next VLM to initialize my VLA from.
2. I am also concerned about the decision of using the same hyperparameters for all the models. While this drastically lowers the number of experiments, models of different sizes would likely require different hyperparameters to perform best. This can affect all the results and conclusions that the authors derive from their experiments.

### Questions
1. Which VQA benchmarks were used for the correlation analysis in Fig 3?
2. Have you tried correlating robotic task performance with other downstream tasks besides VQA?
3. What is the (lower-bound) performance of a VLM4VLA initialized from scratch?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper studies the impact of the underlying VLM on VLA policy performance through the VLM4VLA architecture. VLM4VLA trains a continuous regression head on top of the VLM for the robot action prediction task. The paper trains and evaluates VLAs based on several different VLM sizes across the CALVIN, SimplerEnv, and Libero simulated environments. The paper finds that the impact of base VLM capabilities varies greatly by environment. Furthermore, continuing to train the VLM on embodiment and 3D-specific tasks produces a slight degradation in VLA performance.

### Strengths
1. The paper addresses the important problem of the impact of VLMs on VLA performance, which has been relatively understudied in prior work.
1. The paper shows that general VLM capability does not necessarily correlate with VLA capability. This is an important finding since it contradicts the common intuition that a stronger VLM model is always better for VLAs. For example, recent VLA works use newer VLM bases, which this study shows is not necessarily a good decision.
1. The paper supports its claims with comprehensive empirical analysis across many different base VLMs and simulated environments.
1. The result showing that further training on VLM auxiliary tasks does not improve downstream VLA performance, as presented in Figure 4, is a novel insight. The paper supports this claim with comprehensive results that test a wide variety of auxiliary tasks. Like the VLM-to-VLA transfer result, this finding is also surprising and contrary to prior efforts that design auxiliary tasks to improve VLM embodied capabilities.
1. The results in Section 4.3 provide an important lesson by comprehensively demonstrating the importance of training the visual encoder, where freezing the visual encoder leads to a large performance drop.
1. The VLM4VLA architecture provides a consistent and simple way to adapt VLMs to VLAs. The paper also shows that this architecture produces results superior to prior works that use more complicated designs.
1. The paper includes sufficient reproduction details in the appendix.

### Weaknesses
1. The importance of the visual encoder can also be explained by several other factors beyond the need to finetune it. (1) The Qwen2.5-VL model is sensitive to image resolution, with higher resolutions using more visual tokens per image and typically producing better performance. It is possible that the visual encoder could be frozen if the image resolution were increased. (2) The VLMs are trained primarily on real images, while the selected benchmarks are not photorealistic and use only simple rendering (see the top of Figure 1). Thus, the visual encoder may need to be finetuned to overcome this domain gap, which would likely not be an issue with real robot data.
2. The linear correlation plot in Figure 3 is hard to interpret (see Question 2). The paper should report the strength of the correlation.
3. The paper appears to rely on VQA benchmarks as a measure of VLM capability in Figure 3. Are there more specific VLM capabilities, such as spatial understanding, that have a direct relationship with VLA performance?
4. The paper does not provide analysis of what drives VLA performance based on the base VLM. For example, why does Kosmos-2 achieve the highest success rate in Table 2? While showing that VLM capabilities do not necessarily correspond to VLA capabilities is a valuable contribution, providing initial evidence for why this occurs would strengthen the paper.
5. The paper does not experiment with larger VLM sizes. It is possible that larger VLMs learn more general features that improve VLA capabilities, which this paper does not rule out.

### Questions
1. Is it still important to train the visual encoder if the image resolution is increased, as discussed in Section 4.3 (see Weakness 1)?
2. What do the colors of the lines and shapes represent in Figure 3? Likewise, which points are used to fit the lines?
3. How does the paper compare the VLM performance of the models between “proprietary tasks” and “general-purpose VQA benchmarks” (L357–368)? Shouldn’t VLM general capability be assessed using the same benchmarks for all models?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
The authors propose VLM4VLA, a general framework for adapting arbitrary Vision-Language-Models (VLMs) into Vision-Language-Action (VLA) models, requiring only a minimal number of new parameters. The work also investigates how the choice of the base VLM affects the downstream VLA's performance.

### Strengths
1. The paper presents a framework for fairly comparing the performance of different VLMs on VLA tasks and provides an in-depth study into the reasons for performance discrepancies.
2. By using an MLP action head instead of a more complex diffusion-based one, the framework avoids introducing stochasticity. This ensures a "fair and reproducible" comparison across the different VLMs.
3. It systematically proposes three benchmarks for evaluating VLM capabilities: general capability, embodied-specific capability, and the vision encoder.
4. The experiments cover a wide range of models and test tasks, providing a strong empirical baseline for future work.

### Weaknesses
1. The study lacks real-robot experiments. The sim-to-real gap is a major concern in the VLA field, and this work doesn't clarify how different VLMs might affect the model's final generalization to real-world scenarios.
2. Diffusion action heads and MLP action heads may leverage VLM capabilities differently (e.g., many diffusion heads use the VLM's KV-cache for information interaction). The paper does not directly compare the impact of these two approaches on VLA performance.

### Questions
1. Could you provide a performance comparison and analysis for a model using both a diffusion action head and an MLP action head? Or, perhaps more directly, a comparison between using embeddings vs. using the KV-cache for information interaction?
2. Are there any additional experiments that could demonstrate the difference in generalization capabilities among the various VLMs when applied to VLA tasks?

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
This work conducts an empirical study (in simulation) to investigate the relationship between the capabilities of the VLM backbone and its downstream performance in a VLA. The authors propose VLM4VLA, an adaptation pipeline that adds a MLP head to the VLM backbone to ensure a fair comparison across models. Through experiments in simulation, the authors arrive at three findings: 1) a VLM's performance on general VQA benchmarks is often a poor predictor of performance on robotic manipulation tasks, 2) fine-tuning a VLM on auxiliary embodied VQA tasks doesn't reliably improve performance, and 3) fine-tuning the VLM's vision encoder is necessary.

### Strengths
Large-scale study: the primary strength of this work is the sheer number of models (7) and simulation environments (3) studied. This comprehensive evaluation provides a valuable perspective

VLM4VLA Framework: by using an MLP (non-stochastic), the authors maintain a simple adaptation module which enables isolating the VLM backbone as one of the only independent variables. Ensuring prompts are adjusted per model is further evidence of clean experimentation.

Findings: the findings that VQA performance is not predictive of manipulation performance, in simulation, is good to know for future work. However, I would like to add that I think the main reason researchers leverage a VLM backbone is to capitalize on the potential of generalization, not in-distribution robot manipulation performance (this must be taught via imitation learning).

### Weaknesses
Reliance on simulation: the paper's most significant weakness, which the authors acknowledge, is exclusive use of simulated environments. While this choice was made for reproducibility and scale, it limits whether or not the paper's main claims generalize to the real-world, which is ultimately what is needed in robotics. The sim-to-real gap is a well-known challenge in robotics and care must be taken to extrapolate findings from simulation to the real world. Moreover, Figure 3 supports this statement: the findings of VLA performance and VLM capability are not consistent across simulation environments. 

As another example of the potential pitfalls of simulation, as vision encoders are often pre-trained on real-world images, evaluating them purely in the simulated domain may add a confounding variable to the analysis, as most VLAs are only fine-tuned on real-world data. Therefore, I suggest that all main claims in the paper add the qualifier that they hold only in the simulation benchmarks tested. 

Novelty of vision encoder finding: while the result that fine-tuning the vision encoder is crucial, this observation is not entirely new. Other works such as OpenVLA [1] and Otter [2] have described similar findings. 

[1] Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., ... & Finn, C. (2024). Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246.

[2] Huang, H., Liu, F., Fu, L., Wu, T., Mukadam, M., Malik, J., ... & Abbeel, P. (2025). Otter: A vision-language-action model with text-aware visual feature extraction. arXiv preprint arXiv:2503.03734.

### Questions
Clarification of Pi0 baseline: could you elaborate a bit more on the specifics of why the official implementation of Pi0 was not used? In particular, were the official weights of the Physical Intelligence model used? 

Defining "VLM General Capability:" Figure 3 aggregates 'vlm capability' on the x-axis, but it is not precisely defined. What exactly does this mean?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4