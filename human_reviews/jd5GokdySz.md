# Foundation Model-oriented Robustness: Robust Image Model Evaluation with Pretrained Models

- Decision: Accept (poster)
- Scores: 6, 8, 6

## Abstract
Machine learning has demonstrated remarkable performance over finite datasets, yet whether the scores over the fixed benchmarks can sufficiently indicate the model’s performance in the real world is still in discussion. In reality, an ideal robust model will probably behave similarly to the oracle (e.g., the human users), thus a good evaluation protocol is probably to evaluate the models’ behaviors in comparison to the oracle. In this paper, we introduce a new robustness measurement that directly measures the image classification model’s performance compared with a surrogate oracle (i.e., a zoo of foundation models). Besides, we design a simple method that can accomplish the evaluation beyond the scope of the benchmarks. Our method extends the image datasets with new samples that are sufficiently perturbed to be distinct from the ones in the original sets, but are still bounded within the same image-label structure the original test image represents, constrained by a zoo of foundation models pretrained with a large amount of samples. As a result, our new method will offer us a new way to evaluate the models’ robustness performance, free of limitations of fixed benchmarks or constrained perturbations, although scoped by the power of the oracle. In addition to the evaluation results, we also leverage our generated data to understand the behaviors of the model and our new evaluation strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed foundation model-oriented robustness, where foundation models are used as the oracle when testing vision models’ robustness. A pipeline is developed to maximize model’s misalignment while preserving semantics with an ensemble of foundation models, and a limiting computation budget. Lastly the authors presented a systematic study of Foundation Model-oriented Robustness (FMR) on ImageNet and ImageNet-C.

### Strengths
1. The proposed pipeline made use of existing foundation models as the oracle, addressing several issues of existing testing benchmarks, such as limited to specific curated perturbations or a distribution different from the training set.
2. The flexibility of the proposed method allows robustness evaluation on a wide range of image datasets and vision models.
3. The authors presented a thorough study of vision models (CNN or transformer based), and different augmentation methods.

### Weaknesses
1. The perturbations produced by the model seem limited to those barely visible to human beings. Is this desirable? For now all perturbations seem weak, they are small perturbations around the image, barely visible to humans. There are many strong generation models, such as edge-conditioned ControlNet, or inpating based diffusion models that can generate very diverse outputs in terms of colors, and styles. I think these strong perturbations would be very interesting and naturally suitable in this framework.
2. There’s limited understanding of what the perturbed images are. This is important to support the “diversity” argument in the introduction. How similar are the images to ImageNet-C perturbations?
3. It’s challenging to read the texts in Figure 1…

### Questions
1. As discussed in the introduction, two main limitations of existing evaluations are the limited type of perturbations, and testing data in a different distribution from training set. How well did this work improve these issues over previous evaluations? Are the perturbations more diverse? For the data distribution, there is also a gap between ImageNet training and foundation model training left unaddressed.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a novel approach to evaluate machine learning models' robustness. Rather than relying solely on fixed benchmarks, the authors propose a method that directly measures image classification models' performance in comparison to a surrogate oracle, a set of foundational models. They extend image datasets with perturbed samples that maintain the original image-label structure but are distinct from the original set. This approach allows for a more comprehensive evaluation of models' robustness, overcoming limitations associated with fixed benchmarks and constrained perturbations. The paper not only presents evaluation results but also provides insights into model behaviors and the new evaluation strategies employed.

### Strengths
The paper discusses the evaluation of model performance, which is an important topic.

### Weaknesses
1. The text in Image 1 is too small to discern clearly.
2. I am skeptical about the significance of the paper. I really don't understand why the performance needs to be compared with the foundation model. If there are already foundation model for the given task, is it really necessary to train a new model with similar architecture from scratch?
3. If users need to train different generative models for various downstream tasks, I think this evaluation approach is not user-friendly.
4. Is there a possibility that because the evaluation method proposed in this paper uses VQGAN, and DAT also utilizes VQGAN during training, DAT performs best under this specific evaluation scenario? If that's the case, the evaluation method proposed in this paper might not provide enough insight to the readers.

### Questions
See above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a method to generate perturbed images which have to take the "to-be" evaluated models into account and uses the prediction of the pre-trained large foundation models as "oracle/ground-truth/constraints". These perturbed images could serve as "dynamic" evaluation benchmarks for robustness evaluation compared to previously fixed benchmarks.

### Strengths
1) The paper is well-organized and the writing is clear and easy to follow;
2) The problem is clearly stated and the experiments have included enough details.
3) The authors have provided a nice discussion, which really benefits readers to better understand this proposed method.

### Weaknesses
1) \theta is an input for Algorithm-1, but do not see where it has been used in Algorithm-1.

2) It is not clear to me why this method has any advantages over other benchmarking datasets. The big difference is this proposed method will take the "to-be" evaluated model into account when generating the "adversarial examples". It looks like this proposed method could generate "personalized" "adversarial" examples for the "to-be" evaluated models. So each "to-be" evaluated model may be evaluated on totally different datasets. But in order to fairly compare across different "to-be" evaluated models, the authors normalize their performance by the pre-trained large foundation model? However, I do not see any benefits as to why this evaluation pipeline could better reflect real-world scenarios.

3) It is not clear what "SA" means. Is SA the clean test accuracy of the foundation model? Also, there is an abusive use of "SA" to present two different terms in this paper---- SA for "standard accuracy" and SA for "self-attention".

4) If the perturbed generated images are the images that cause the maximum classification loss of the "to-be" evaluated model, then it will definitely result in a very bad classification performance of the "to-be" evaluated model on the perturbed images. It is not clear to me how the authors do this kind of evaluation. From my current understanding, the generated images have already had the knowledge of the "to-be" evaluated model. Then I doubt how "PA/SA" could be a fair comparison for different "to-be" evaluated models.

5) The overall idea looks very similar to "adversarial training". The big difference is instead of using "norm-bouned" perturbation, here the authors use the prediction of large foundation models as a constraint to limit the perturbation to not go too far. So from this perspective, I only see limited novelty of this proposed method. And even, the perturbed images may be eventually biased towards the foundation models.

6) Could we also report the performance of the "to-be" evaluated models on some of fixed benchmarks as a robustness reference?

### Questions
see [weaknesses].

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
