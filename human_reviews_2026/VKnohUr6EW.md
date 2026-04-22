# SD3.5-Flash: Distribution-Guided Distillation of Generative Flows

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4

## Abstract
We present SD3.5-Flash, an efficient few-step distillation framework that brings high-quality image generation to accessible consumer devices. Our approach distills computationally prohibitive rectified flow models through a reformulated distribution matching objective tailored specifically for few-step generation. We introduce two key innovations: "timestep sharing" to reduce gradient noise and "split-timestep fine-tuning" to improve prompt alignment. Combined with comprehensive pipeline optimizations like text encoder restructuring and specialized quantization, our system enables both rapid generation and memory-efficient deployment across different hardware configurations. This democratizes access across the full spectrum of devices, from mobile phones to desktop computers. Through extensive evaluation including large-scale user studies, we demonstrate that SD3.5-Flash consistently outperforms existing few-step methods, making advanced generative AI truly accessible for practical deployment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper concentrates on two points:

1. a modification of the renoising part of the DMD algorithm
2. a comprehensive lightning pipeline including text encoder dropping, quantization, and diffusion step distillation

### Strengths
1. Good presentation
2. Diffusion acceleration is an important topic, and I appreciate the efforts in combining different acceleration methods.

### Weaknesses
### Experiments are inadequate to support the proposed methods:
  1. The ablative experiment is qualitative, but numerical results are necessary here to validate the strengths of the modifications
  2. Cross-base-model comparisons are not informative enough. Considering that the proposed method is evolved from DMD/DMD2, a fair comparison with these works (using the same base model and training pipeline) is necessary to substantiate the modifications as improvements.

### Timestep sharing, as the primary original methodological contribution, is not sufficiently explained and validated

If my understanding is correct, in traditional DMD, the noisy sample is obtained by first denoising (moving) the few-step input to the clean level (t=0) using the velocity predicted by the few-step generator, followed by renoising to the target timestep $t'$ with random noise. In the proposed method, the renoised sample with noise level $t'$ is obtained by directly moving the few-step input along the few-step predicted velocity by a certain interval. 

I have the following questions:

1. The sentence "Consequently, this forces us to share distribution matching timesteps with the student trajectory timesteps $t^s_i$, instead of random t in Eq. (3)" does not trivially hold to me. Why does this constraint exist? And what would happen if  $t'$ is randomly sampled without constraints?

2. The sentence "adding random noise for reaching timestep t can create noisy gradient updates." is not intuitive to me either. While score-based methods, diffusion models, and flow matching models have their respective backgrounds and mathematical deductions, in practice, they are doing almost the same thing: adding noise to clean data, and making the model predict a target quantity, such as noise, the original clean data, or velocity. Importantly, these quantities are all interchangeable. Therefore, why do flow matching models uniquely suffer from the unstable gradient problem? Do diffusion models suffer from similar issues? 

3. In-depth analysis is needed to explain what the unstable gradient problem is and how it negatively affects training. Quantitative experiments are also needed to support its superiority over the existing renoising method.

### Apart from the above limitations, the overall novelty of this work is limited. It is mainly a combination of well-established techniques.

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SD3.5-Flash, a distillation framework designed to efficiently convert a large, multi-step teacher model (SD3.5 Medium) into a high-quality, few-step (2-4 steps) student model suitable for deployment on consumer hardware. The core methodological contributions are two-fold:
Timestep Sharing: A simple trick that uses the student's intermediate trajectory points for gradient computation, avoiding the instability introduced by re-noising from trajectory endpoints.
Split-Timestep Fine-Tuning: A capacity-enhancement technique where the model is branched and fine-tuned on disjoint timestep ranges before being merged back, aiming to improve prompt alignment.
Through extensive qualitative, quantitative, and user-study-based evaluations, the authors demonstrate that SD3.5-Flash outperforms existing few-step distillation methods in image quality and prompt adherence while achieving significant speed-ups.

### Strengths
1. "Split-Timestep Fine-Tuning" is a creative approach to mitigating the capacity bottleneck in distilled models.
2. The work goes beyond a pure algorithmic contribution by integrating model distillation with critical deployment engineering. This holistic view from algorithm to practical deployment is highly valuable, which proves a potentially significant practical impact.
3. Large-scale human evaluation, broad quantitative metrics (FID, CLIPScore, Aesthetic Score, ImageReward), and latency and memory footprint analysis are valuable for other efforts.

### Weaknesses
1. The Timestep Sharing operation seems like a simple trick to integrate DMD and other denoising steps.
2. The efficacy and rationality of Split-Timestep Fine-Tuning should be highlighted and analyzed in detail.
3. Although the paper claims the efficiency of SD3.5-Flash on terminal devices like laptops or phones, the method does not design any optimization strategies to enhance this ability. To give convincing proofs, the author should analyze why the proposed model is superior after quantization compared to others.
4. Typos.
5. I suggest the authors release the code and finetuned models, which will be of high significance for the community.

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
SD3.5-Flash is an efficient few-step distillation framework designed to bring high-quality image generation from computationally intensive rectified flow models down to accessible consumer devices. Its foundation is a reformulated distribution matching objective tailored for the few-step paradigm, minimizing the gap between the original model and the distilled generator. The method introduces key innovations: "timestep sharing" to effectively reduce gradient noise and "split-timestep fine-tuning" to significantly improve prompt alignment. Coupled with comprehensive pipeline optimizations like text encoder restructuring, SD3.5-Flash enables both rapid, high-quality synthesis and highly memory-efficient deployment on resource-constrained hardware.

### Strengths
1. The paper exhibits outstanding clarity and excellent presentation; the framework is meticulously detailed, and the overall narrative ensures that the complex concepts are highly accessible to the reader.

2. The work is strongly motivated. Few-step distillation represents a crucial research area for rapid generative models, and the goal of enabling high-quality image synthesis on accessible consumer devices holds significant promise for the field's practical application.

### Weaknesses
1. Based on the results presented in Table 2, the proposed method appears to show only marginal, if any, advantage in terms of combined efficiency and capability when directly compared to the SANA-Sprint baseline, potentially weakening the claimed technical contribution of the work.

2. Although titled SD3.5-Flash, the core contribution seems presented as a general few-step distillation framework. The paper would be significantly strengthened by demonstrating its applicability and performance across other advanced flow-matching models, such as FLUX, to confirm the true generality of the approach.

### Questions
None

### Soundness
2

### Presentation
2

### Contribution
2
