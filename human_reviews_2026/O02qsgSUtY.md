# STEDiff: Revealing the Spatial and Temporal Redundancy of Backdoor Attacks in Text-to-Image Diffusion Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Recently, diffusion models have been recognized as state-of-the-art models for image generation due to their ability to produce high-quality images. However, recent studies have shown that diffusion models are susceptible to backdoor attacks, where an attacker can activate hidden biases using a specific trigger pattern, causing the model to generate a predefined target. Fortunately, executing backdoor attacks is still challenging, as they typically require substantial time and memory to perform parameter-based fine-tuning. In this paper, we are the first to reveal the **spatio-temporal redundancy** in backdoor attacks on diffusion models. **Regarding spatial redundancy**, we observed the *enrichment phenomenon*, which reflects the abnormal gradient accumulation induced by backdoor injection. **Regarding temporal redundancy**, we observed a marginal effect associated with specific time steps, indicating that only a limited subset of time steps plays a critical role in backdoor injection. Building on these findings, we present a novel framework, *STEDiff*, comprising two key components: *STEBA* and *STEDF*. *STEBA* is a spatio-temporally efficient accelerated attack strategy that achieves up to **15.07×** speedup in backdoor injection while reducing GPU memory usage by **82%**. *STEDF* is a detection framework leveraging spatio-temporal features, by modeling the enrichment phenomenon in weights and anisotropy across time steps, which achieves a backdoor detection rate of up to **99.8%**.  Our code is available at: [https://github.com/paoche11/STEDiff](https://github.com/paoche11/STEDiff).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents two key findings. First, it identifies an enrichment phenomenon where backdoor injection causes abnormal gradient accumulation in a few critical weight parameters. Second, it reveals that only a small subset of timesteps significantly influences backdoor injection.

Building on these insights, the authors propose two frameworks:
1. STEBA – a backdoor attack that targets optimization on key weights and crucial diffusion timesteps. It achieves a 15.07× speedup in injection and reduces *ideo memory usage by 82%.
2. STEDF – a detection framework that monitors spatio-temporal feature dynamics across timesteps to halt malicious generations mid-process, achieving up to 99.8% detection accuracy.

### Strengths
The paper is overall clear and well-organized. It provides insights into the mechanisms of backdoor injection in diffusion models, in the enrichment of gradients in key weight parameters and the temporal sparsity of critical timesteps. 

Their proposed methods, STEBA and STEDF, shows strong practical values, improving the efficiency of backdoor insertion and providing an effective mechanism for detection.

The experimental evaluation spans three widely used diffusion models (Stable Diffusion v1.5, v2.1-base, and Realistic Vision v4.0) and multiple trigger types, which reinforces the robustness and generality of the findings. Overall, the work offers meaningful contributions to understanding and mitigating backdoor vulnerabilities in diffusion models.

### Weaknesses
The study could be further strengthened by extending experiments to a broader range of diffusion model families to better assess generalizability. Also, it would be useful to evaluate STEDF under adaptive attacker scenarios to understand its resilience against under this adaptive threat model.

### Questions
For STEBA, 
- How does various parameters such as top-k and thresholds affects the attack effectiveness?  

For STEDF, 
- Which timestep found to be the most effective for detection? 
- What is the average compute savings in diffusion steps?

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
The paper observes redundancies in the backdoor attacks against diffusion models. The authors propose an attack method, STEBA, and a defense method, STEDF, based on the observations.

### Strengths
1. The observations seem intuitve and supported by experiments.
2. STEBA achieves high attack sucess rate with reduced computational cost in the paper's setting.

### Weaknesses
1. There lacks necessary understanding into the STEBA methods. It is unclear to me whether the observations about redundancy are confirmed by the optimizaiton results of STEBA. And whether the distribution of the most important parameters/time-steps follows certain rules.

2. The evaluation of STEDF is flawed. It is not mentioned what attack method was used, and what are the configurations of the attack and the baseline. It feels that the evaluation is weak since the baseline already achieves over 90% accuracy. Besides, it is unclear whether STEDF can transfer between different attack methods/datasets etc.

3. What does 'MSE' mean in Figure 4? Fonts in Figure 5(b) are too small to see.

### Questions
Please see weekness.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work reveals the novel phenomenon of text-to-image backdoor attack methods, denoted as spatial and temporal redundancy. They argue that the existing abnormal gradient accumulation brought by backdoor injection is regarded as spatial redundancy, and the subset of time steps that impact backdoor injection is considered as temporal redundancy. After recognizing two types of redundancy, this research proposes a novel framework for attack and detection in the field. The experimental results present their roadmap for discovering the phenomenon and the following framework, which provides a new perspective for this field.

### Strengths
1. The observations of the UNet, as well as the Transformer for the enrichment phenomenon (spatial redundancy), are good to demonstrate the existing defect of the VillanDiffusion.
2. The observations of the time steps correlation with ASR/FID are good to know that VillanDiffusion still has space for improvement.
3. For the defense, the authors provide more types of triggers to check the generalization (scope) of their proposed detection framework.

### Weaknesses
1. **Unclear base attack methods for analysis.** Based on Table 1, I conjecture that your analysis from Sec 2 to Sec 4 is based on the VillanDiffusion. Is that right?
2. **Comparisons with other attack methods.** However, several attack methods have been proposed today, such as BadT2I (as you mentioned in Sec 2.2), EvilEdit (1), and PaaS (2) mentioned in the survey paper (3). In my experience, EvilEdit and PaaS also rely on a few resources for consumption. Could you provide the comparisons with these methods? If not, please give convincing reasons.
3. Follow 2., as I know the attack behaviors in (1) and (2) are different from VillanDiffusion and RickRolling in their cross-attention maps, which makes me concerns about the generalization of your observation of the enrichment phenomenon. Could you provide more theoretical or empirical explanations about the enrichment phenomenon?
4. **Unclear about the analysis of the marginal effect in timesteps.**During the earlier timesteps, how do you obtain the images for calculating ASR and FID? ** Do you estimate the final image $x_0$?
5. **Unclear about the Trigger patterns.** Could you please provide the details of the trigger patterns in Tables 2 and 4? I might miss this part in the main article and in the Appendix.
6. The authors sometimes refer to $M_{be}$ as the benign model (Line 241) or baseline model (Line 247). **I suggest that the authors make the call consistent or clarify that it has the same meaning. **

- (1) Wang, H., Guo, S., He, J., Chen, K., Zhang, S., Zhang, T., & Xiang, T. (2024, October). Eviledit: Backdooring text-to-image diffusion models in one second. In Proceedings of the 32nd ACM International Conference on Multimedia (pp. 3657-3665).
- (2) Huang, Y., Juefei-Xu, F., Guo, Q., Zhang, J., Wu, Y., Hu, M., ... & Liu, Y. (2024, March). Personalization as a shortcut for few-shot backdoor attack against text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, No. 19, pp. 21169-21178).
- (3) Lin, W., Zhou, N., Wang, Y., Li, J., Xiong, H., & Liu, L. (2025). BackdoorDM: A Comprehensive Benchmark for Backdoor Learning in Diffusion Model. arXiv preprint arXiv:2502.11798.

### Questions
1. I wonder about the choice of the diffusion model 'Realistic Vision V4.0'. Is there any reason? What is the structure of this UNet or DiT model?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper "STEDiff" introduces a unified attack and defense framework that uncovers spatio-temporal redundancies in backdoor attacks on diffusion models . The authors identify two key phenomena: the enrichment effect (spatial redundancy in weight updates) and the marginal effect of timesteps (temporal redundancy in backdoor training). Building on these findings, they propose STEBA, an efficient attack method that reduces GPU memory and training time, and STEDF, a real-time defense mechanism that detects backdoors by monitoring behavior in diffusion dynamics. Their method significantly improves attack efficiency while maintaining high attack success rates. The study demonstrates that both attack and defense can be optimized by focusing on key weights and critical timesteps, reducing overhead while enhancing robustness.

### Strengths
1. Significant ASR Improvement with Lower Compute Cost: The Paper proposes a computationally-efficient backdoor attack on diffusion models. Demonstrate the backdoor attack on diffusion models can be achieved by controlling a few timesteps.
2. Novel Insight – Redundancy in Backdoor Training: This work is the first to pinpoint spatial and temporal redundancies in diffusion model backdoor attacks . Identifying that only a small fraction of model parameters and diffusion steps are truly responsible for the backdoor is a fresh and important insight.
3. Highly Effective Defense: The defense component, STEDF, demonstrates near-state-of-the-art detection performance. It can detect backdoor-compromised models with Backdoor Detection Rates ~98–100% across a wide range of trigger types, while maintaining very low false positive rates (often 0–2%)

### Weaknesses
1. Heuristic Methodology: The paper doesn't provide theoretical explaination for the method. Also lack of the analysis of various hyperparameters choosing.
2. Unclear Temporal Selection for STEBA: It's not surprise that diffusion models has reduntant steps because close timesteps have almost identical score or velocity field. The paper should include more detailed temporal selection algorithm and various amount of chosen timestep. It should also demostrate the results for such different settings. A better investigation should further cover the changing temporal dynamic across various amount of chosen timestep.
3. Unclaer Sampler Choice: It's trivial if only train on the timestep used by the specific sampler and achieve good FID and ASR on the identical sampler. For example, evaluate with 50 steps DDIM and backdooring on the these 50 step used by DDIM. The paper should cover a more comprehensive experiments to demostrate the generalization or failure on various samplers and sampling steps, including DDIM, DPM-Solver, PNDM, and UniPC while backdoored on fewer effective timestep.
4. Ignore the Usage of LoRA in VillanDiffusion: The paper doens't recognize the usage of LoRA in VillanDiffusion, which might be the root cause of enrichment effect.
5. Not Clarify the Contribution in Comparing to Previous HIdden-Activation-Based Backdoor Detection: Existing works have identify that activations in the hidden layers can pose strong signal for bnackdoor actication, like [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](https://ceur-ws.org/Vol-2301/paper_18.pdf). However, the paper doesn't emphasize the main contribuition and difference between this paper and prior works.

### Questions
1. How to choose the effective timestep for STEBA? Can you provide pseudo code and details? What if choosing different strategies and timesteps?
2. For each backdoored diffusion models, can ¥ou demostrate the utility and the ASR on various samplers and sampling stepsd? including DDIM with 100 steps, DPM-Solver with 20 steps, UniPC with 20 steps, and PNDM with 20 steps, which align with VillanDiffusion settings.
3. It looks like the experiment in section 9.3 and 9.5 don't recognize the usage of LoRA in VillanDiffusion. Can you conduct an experiment to demostrate if enrichment effect exists without LoRA? What's the consequence with and without LoRA?
4. Please survey the prior works on Backdoor Detection via Network Activation.

### Soundness
2

### Presentation
3

### Contribution
2
