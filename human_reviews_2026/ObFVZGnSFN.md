# DepthLM: Metric Depth from Vision Language Models

- Avg Score: 6.67
- Decision: Accept (Oral)
- Scores: 10, 4, 6

## Abstract
Vision language models (VLMs) can flexibly address various vision tasks through text interactions. Although successful in semantic understanding, state-of-the-art VLMs including GPT-5 still struggle in understanding 3D from 2D inputs. On the other hand, expert pure vision models achieve super-human accuracy in metric depth estimation, a key 3D understanding task. However, they require task-specific architectures and losses. Such difference motivates us to ask: Can VLMs reach expert-level accuracy without architecture or loss change? We take per-pixel metric depth estimation as the representative task and show that the answer is yes! Surprisingly, comprehensive analysis shows that text-based supervised-finetuning with sparse labels is sufficient for VLMs to unlock strong 3D understanding, no dense prediction head or complex regression/regularization loss is needed. The bottleneck lies in pixel reference and cross-dataset camera ambiguity, which we address through visual prompting and intrinsic-conditioned augmentation. With much smaller models, our method DepthLM surpasses the accuracy of most advanced VLMs by over 2x, making VLMs for the first time comparable with pure vision models. The simplicity of DepthLM also enables a single VLM to cover various 3D tasks beyond metric depth. Code and model are available at https://github.com/facebookresearch/DepthLM_Official.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper introduces DepthLM, which enables Vision-Language Model (VLM) to perform metric depth estimation with accuracy comparable to specialized vision models. The authors identify that VLM’s weakness in depth estimation is due to issues in pixel reference and camera ambiguity. To address this, DepthLM employs visual prompting for pixel localization and intrinsic-conditioned augmentation to unify focal lengths across datasets. Using only sparse text-based supervision (1 or 2-4 labeled pixels per image in large-scale training ), DepthLM outperforms large proprietary models like GPT-5 and Gemini-2.5-Pro and reaches parity with state-of-the-art pure vision models such as UniDepth.

### Strengths
1. DepthLM achieves expert-level metric depth estimation without changing the architecture or using complex losses in mainstream VLM. It’s very astonishing that VLM can beat expert depth estimation models in monocular depth estimation.
2. It’s also very astonishing that by querying a few pixels in each image, and tuning VLM to estimate depth for a few pixels of each image, VLM can beat expert depth estimation models. This finding is very interesting and meaningful.
3. The DepthLM framework extends beyond monocular depth estimation to other 3D reasoning tasks, like spatial VQA or robot manipulation, using one unified vision-language model. I think if general VLM can be tuned to be very good at depth estimation, it can also benefit downstream application that requires precise distance or spatial measurement, like robot navigation and manipulation.

### Weaknesses
**Major:**

1. The method requires significant computational resources. As stated in the paper, the training requires 23M-60M samples, with 2-4 pixels sampled per image, using 128 H100 GPU for 2-4 days. Nevertheless, I think if the training process can be integrated into the instruction-tuning stage of general VLM to improve depth perception ability, the computational overhead is acceptable.

**Minor:**

Since DepthLM explores leveraging VLM for depth estimation, the authors might consider discussing or citing a few related papers that also investigate VLM for depth estimation:
1. Zhang, R., Zeng, Z., Guo, Z., & Li, Y. (2022, October). Can language understand depth?. In Proceedings of the 30th ACM International Conference on Multimedia (pp. 6868-6874).
2. Hu, X., Zhang, C., Zhang, Y., Hai, B., Yu, K., & He, Z. (2024). Learning to adapt clip for few-shot monocular depth estimation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 5594-5603).
3. Chen, W., Shi, C., Ma, C., Li, W., & Dong, S. (2024). DepthBLIP-2: Leveraging Language to Guide BLIP-2 in Understanding Depth Information. In Proceedings of the Asian Conference on Computer Vision (pp. 2939-2953).
4. Zeng, Z., Wang, D., Yang, F., Park, H., Soatto, S., Lao, D., & Wong, A. (2024). Wordepth: Variational language prior for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9708-9719).
5. Cui, B., Huang, Y., Bai, L., & Ren, H. (2025). TR2M: Transferring Monocular Relative Depth to Metric Depth with Language Descriptions and Scale-Oriented Contrast. arXiv preprint arXiv:2506.13387.
6. Zeng, Z., Wu, Y., Park, H., Wang, D., Yang, F., Soatto, S., ... & Wong, A. (2024). Rsa: Resolving scale ambiguities in monocular depth estimators through language descriptions. Advances in neural information processing systems, 37, 112684-112705.
7. Zhang, W., Liu, H., Li, B., He, J., Qi, Z., Wang, Y., ... & Jin, X. (2025). Hybrid-grained Feature Aggregation with Coarse-to-fine Language Guidance for Self-supervised Monocular Depth Estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 6678-6692).
8. Zeng, Z., Ni, J., Wang, D., Rim, P., Chung, Y., Yang, F., ... & Wong, A. (2024). PriorDiffusion: Leverage Language Prior in Diffusion Models for Monocular Depth Estimation. arXiv preprint arXiv:2411.16750.

### Questions
1. In training, DepthLM conducts image resizing to unify focal length. Does inference also require unifying focal lengths, which requires camera intrinsics? If I understand correctly, unifying focal length during training is resizing all images to the same size, then by conducting random cropping, the model can generalize to different image sizes during inference. My question is, since the focal length is not unified during inference, could authors elaborate more about how the model can generalize to different image sizes with different focal lengths during inference?

2. This paper mentions that “Image diversity is more important than label density”. The experiment provides only an increase in the label density after the training samples (I assume it means the number of images) larger than 16M. I am quite curious about the contribution of label density and image diversity to the model’s performance, respectively. Could the authors provide the results that show the performance under the combination of different image diversity and label density? For example, draw several lines for Figure 4 (c), each line means sampling different numbers of pixels per training image. I know this experiment would require extensive computational resources, so it’s totally OK if only part of such an experiment could be shown.

3. How are the pixels sampled during evaluation? How many pixels are sampled per image, and how are they sampled? I didn’t find it in the paper.

4. How are labelled pixels sampled in each image? (How) Will the sampling strategy affect the performance in training?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The goal of this paper is to investigate whether it is possible to fine-tune a pre-trained VLM to achieve high accuracy in per-pixel metric depth estimation without modifications to the model architecture or training objective.

The authors fix the model architecture and conduct a set of experiments to investigate the best way to incorporate task-specific information for 3D depth estimation into the model's input prompt to achieve high-accuracy depth predictions. They fine-tune VLM to perform the 3D depth estimation task by generating a target distance in text modality.


The contributions of this work are as follows:
1) an approach for training VLMs to achieve high accuracy in per-pixel depth estimation
2) an accompanying benchmark dataset (mix of public datasets) adapted to the training and evaluation of VLMs on the 3D depth estimation task.

### Strengths
1) Approach works with two different VLM architectures.
2) It is the first VLM-fine-tuning-based approach for metric depth estimation that reaches specialised pure vision models.

### Weaknesses
1) It is not clear how the proposed pretraining unlocks 3D understanding. It looks like authors use the terms "3D understanding" and "per-pixel 3D metric depth estimation" interchangeably. 3D understanding is much broader than per-pixel 3D metric depth estimation. See questions.
2) High training cost. The model is trained on ""128 H100 GPUs for about 2-4 days"" compared to 6 days on 16 NVIDIA 4090 with half precision (best monocular metric depth estimation baseline UniDepthV2).

### Questions
1) Fine-tuned model does achieve the result comparable to specialized pure vision models, but it is fine-tuned just on the per-pixel depth estimation task. What is the difference vs expert pure vision approaches, except that the model is trained to output the text? Even in "Beyond metric depth" experiments, when the model is trained on five other tasks (which in fact are all depth estimation related), what original VLM properties, except executing these 5 tasks on which it was trained, are preserved?

2) Related to the above question: L37-39: Is it still a VLM if it is trained to execute one or five specific tasks? 

3) L117-118: how numerical 3D labels are chosen?

4) L324: "We tune the prompt for VLMs that are not trained directly on our task to maximize their performance". How was the prompt tuning done? Were all VLMs, selected as baselines, benchmarked using visual prompting?

5) L420: It -> it.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces DepthLM, a vision-language model designed for single-image depth estimation. The authors explore four key aspects: (1) how to prompt a VLM with a query point, (2) whether SFT or RL is more effective, (3) how to handle focal-length generalization, and (4) how much supervision per image is required. The method achieves promising results across experiments.

### Strengths
Clear and focused presentation. The paper is well-structured and easy to follow. Each experimental question is clearly motivated, investigated, and summarized. The writing is concise and avoids unnecessary technical ornamentation, which I find appealing.

Insightful experimental scope. The paper systematically studies several important dimensions of this task (prompting strategy, training scheme, focal length handling, and supervision density), offering valuable insights for the community.

Strong empirical performance. The model achieves competitive or superior results compared to prior methods, demonstrating practical effectiveness.

### Weaknesses
Overemphasis on visual prompting novelty. Prompting in the visual space instead of textual space is useful, but not highly innovative by itself. The paper seems to overstate the novelty of this contribution, dedicating perhaps too much discussion to it relative to its conceptual depth.

SFT vs. RL conclusion lacks depth. The conclusion that SFT outperforms RL feels premature. RL performance can vary significantly with algorithm choices (e.g., GRPO, PPO, DPO), reward design (e.g., continuous metric rewards vs. token-based or discrete depth-bin rewards), and training strategies (e.g., RL after SFT rather than combined in one stage). A more thorough investigation—or at least acknowledgment of these factors—would strengthen the claim.

Focal-length generalization clarity. The discussion on focal-length generalization is somewhat unclear. It appears the model is trained with unified focal-length settings and expected to generalize to unseen focal lengths during inference. Clarifying the exact assumption and mechanism would be helpful.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
