# X-Distill: Cross-Architecture Vision Distillation Enables Data-Efficient Visuomotor Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Visuomotor policies often leverage large pre-trained Vision Transformers (ViTs) for their powerful generalization capabilities. However, their significant data requirements present a major challenge in the data-scarce context of most robotic learning settings, where compact CNNs with strong inductive biases can be more easily optimized. To address this trade-off, we introduce X-Distill, a simple yet highly effective method that synergizes the strengths of both architectures. Our approach involves an offline, cross-architecture knowledge distillation, transferring the rich visual representations of a large, frozen DINOv2 teacher to a compact ResNet-18 student on the general-purpose ImageNet dataset. This distilled encoder, now endowed with powerful visual priors, is then jointly finetuned with a diffusion policy head on the target manipulation tasks. Extensive experiments on 34 simulated benchmarks and 5 challenging real-world tasks demonstrate that our method consistently outperforms policies equipped with from-scratch ResNet or finetuned DINOv2 encoders. Notably, X-Distill also surpasses 3D encoders that utilize privileged point cloud observations or much larger Vision-Language Models. Our work highlights the efficacy of a simple, well-founded distillation strategy for achieving state-of-the-art performance in data-efficient robotic manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a distillation approach for learning robot manipulation policies in low data settings (i.e. having few demonstrations). The distillation approach uses Imagenet-1K as the dataset and DINO (VIT-L/14) as the teacher model. The readout class token is used to distill the pretrained features into a small ResNet-18 model. This Resnet model is then used as the vision backbone on top of a diffusion policy and is trained using imitation learning.

### Strengths
The idea of using distillation to train small vision networks for robot manipulation is interesting.

### Weaknesses
I have several major concerns with respect to the paper. I think this is a very preliminary submission and many related works, baselines are not implemented at all.

*Main objective*:  I am not at all sure why is using distillation a better objective than e.g. using parameter efficient finetuning (PEFT). Using PEFT such as adapters, lora etc. requires us to train only a few parameters while we get a lot of advantages in terms of useful priors, robustness etc.. Many prior works have explored such techniques in the context of robotics e.g. Sharma et al. Liu et al. These papers have shown advantages in low data regimes which is exactly what the proposed work is aiming for. However, unfortunately, none of these works are cited or being compared against. Hence, it is completely unclear if the distillation approach is even a good approach. In my opinion, the adapter based approach might be generally better since it requires even fewer parameters to adapt but this is something that the paper should really experiment. 

*Comparison to pre-trained model*: There is a huge discrepancy between small pretrained model (which is only at 75%). What happens when we compare these models on more demos? How many more demos do we need to bridge this gap? This ablation is not present in the paper.

Imagenet citation in line 144 is incorrect. The paper should cite the original dataset instead of AlexNet which actually solved the problem. This is minor but still important.

Sharma et al. Lossless adaptation of pretrained vision models for robotic manipulation

Liu et al. Tail: Task-specific adapters for imitation learning with large pretrained models

### Questions
see above

### Soundness
1

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
This paper proposes X-Distill, a cross-architecture knowledge distillation method that distills knowledge from a large ViT (DINOv2) into a compact CNN (ResNet-18) in an offline stage. This resulting encoder is then fine-tuned on an extremely small amount of robotic demonstration data (10 for simulation, 20-25 for real-world). The method demonstrates significantly better data efficiency and state-of-the-art performance compared to ViT-based baselines (e.g., DINOv2, $\pi_0$) and a from-scratch ResNet in this low-data regime.

### Strengths
- Impressive results, significantly outperforming SOTA ViT-based models like $\pi_0$ and DINOv2 with an extremely small number of demonstrations (10 for sim, 20-25 for real).
- Clearly written and easy to follow.
- Effective also in real-robot tasks.

### Weaknesses
- While X-Distill (CNN-based) clearly wins in extremely low-data regime, its performance might be surpassed by ViT-based models ($\pi_0$, Theia) if more data is provided (e.g., 50-100 sim demos, 50 real demos, which is (I think) realistically collectible amount).  The paper lacks an analysis of ***how low-data*** is required for X-Distill to be superior, making its application scope unclear. A scaling analysis with respect to data size is missing.

- How about  R3M [1]? Itis a highly relevant ResNet-based encoder pre-trained for robotics manipulation. Since R3M is also ResNet-based, it is crucial to understand if X-Distill outperforms it, and if so, why (e.g., DINOv2 distillation vs. Ego4D video pre-training).

- The ablation study in Table 2 shows that a larger CNN student (ConvNeXt, 89M) performs worse than the smaller ResNet-18 (11M), while the teacher model size (DINOv2-S vs L) has no significant impact. This is confusing, as it provides little guidance on what model size is optimal for a given data size. Furthermore, it is counter-intuitive that a larger, more powerful teacher model does not provide better representations to the student.


[1] Nair et al., R3M: A Universal Visual Representation for Robot Manipulation. CORL, 2022.

### Questions
- For the comparisons in Table 1, did all 2D vision baselines (X-Distill, Theia, DINOv2, ResNet-scratch, Depth-Anything) use the same Diffusion Policy head architecture?
- There seems to be an interesting discrepancy between Figure 4 (t-SNE) and Table 3 (Real-world performance). For the "Writing AGI" task, $\pi_0$ has a much better t-SNE clustering score (0.296) than ResNet-scratch (0.087), but its task performance is 0% while ResNet-scratch achieves 40%. Could you explain why $\pi_0$ fails at the task despite appearing to have better feature separability according to this metric?

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
4

### Summary
The manuscript proposes the use of CNN-based vision backbones for visuomotor policies. It advocates for this solution based on the inductive biases that make CNNs easier to train in low-data regimes. To generate descriptive features, the authors propose the use of model distillation, distilling a large, generalist DINOv2 VIT into a ResNET 18 backbone using the non-robotic images in the ImageNET dataset. 

Based on the proposed backbone architecture, a diffusion policy is implemented and evaluated in simulation and on real-world task. It is evaluated against different visual backbones, including transformers and large CNNs, and against pi0 as a generalist VLA policy. All models are trained on small demonstration datasets and results demonstrate the efficacy of the proposed approach. A further analysis of feature embeddings show the capability of the learned features to differentiate between different stages of robotic tasks.

### Strengths
- The approach shows strongly improved policy success rate with a simple and likely transferable approach.
- The resulting policy is evaluated in simulation and real-world experiments, and benchmarked against a range of visual backbones and policy types.
- The work approaches a robotic learning problem with only few demonstrations and no additional robotic data, which is a relevant practical scenario.

### Weaknesses
- The conceptual novelty of the work is limited, only applying a standard MSE distillation approach to DINO features.
- The evaluation of the work is limited to specialist visuomotor policies. Scaling of the approach to large data settings like VLAs or generalist diffusion policies like Octo [1] is not evaluated.
- The real-world evaluation settings only contain simple tabletop settings, more complex environments that test the generalization capability of the visual encoders with e.g. natural lighting or complex backgrounds are not contained in the test data.

[1] Octo Model Team, et al. "Octo: An open-source generalist robot policy." arXiv preprint arXiv:2405.12213 (2024).

### Questions
The work showcases an interesting observation in finding the right encoders for visuomotor policies. While the work is conceptually simple, it can provide an interesting stepping-stone toward this problem. However, to allow the reader to come to a valuable conclusion, the evaluation should demonstrate the limits of this approach and answer the following questions:
- How does the encoder scale with additional data? Does the encoder size correlate with the distillation data? Can a point be identified for VITs to be better?
- Can the approach profit from exposing the student model to large-scale robot data (OpenX Dataset)?
- Do the findings translate to generalist robot policies?

### Soundness
3

### Presentation
3

### Contribution
2
