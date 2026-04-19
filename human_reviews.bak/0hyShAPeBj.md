# IT$^3$: Idempotent Test-Time Training

- Decision: Reject
- Scores: 5, 5, 3, 3

## Abstract
This paper introduces Idempotent Test-Time Training (IT$^3$),
a novel approach to addressing the challenge of distribution shift.
While supervised-learning methods assume matching train and test distributions, this is rarely the case for machine learning systems deployed in the real world.
Test-Time Training (TTT) approaches address this by adapting models during inference, but they are limited by a domain specific auxiliary task. IT$^3$ is based on the universal property of idempotence. An idempotent operator is one that can be applied sequentially without changing the result beyond the initial application, namely $f(f(x))=f(x)$. 
An idempotent operator is one that can be applied sequentially without changing the result beyond the initial application, that is $f(f(X)=f(X)$.
At training, the model receives an input $X$ along with another signal that can either be the ground truth label $y$ or a neutral "don't know" signal $\mathbf{0}$. At test time, the additional signal can only be $\mathbf{0}$. When sequentially applying the model, first predicting $y_0 = f(X, \mathbf{0})$ and then $y_1 = f(X, y_0)$, the distance between $y_1$ and $y_2$ measures certainty and indicates out-of-distribution input $x$ if high.
 We use this distance, that can be expressed as $||f(X, f(X, \mathbf{0})) - f(x, \mathbf{0})||$ as our TTT loss during inference. By carefully optimizing this objective, we effectively train $f(X,\cdot)$ to be idempotent, projecting the internal representation of the input onto the training distribution.
We demonstrate the versatility of our approach across various tasks,
including corrupted image classification, aerodynamic predictions,
tabular data with missing information, and large-scale aerial photo segmentation. Moreover, these tasks span different architectures such as MLPs, CNNs, and GNNs.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper presents a new method for test-time training: at training time the model is trained to be Idempotent, and at test-time two copies of the models are leveraged (one frozen and one adapted) to encourage the model to be idempotent under distribution shifts. Experiments are carried out on different tasks to demonstrate how versatile the proposed method is.

### Strengths
The main strengths of this paper:

1- Simplicity of the approach: the method is easy to understand and to implement. 

2- The paper is generally well-written and easy to follow.

3- The breadth of the experiments: The authors are commended for the variety of different tasks that they tested their proposed methods on.

### Weaknesses
The main weaknesses of this paper are:

1- The method section, while simple, is a bit confusing. Why does the paper present two variants of IT$^3$? The experiments in section 4.6 and Table 1 clearly favors the online version of the method over the one with frozen predictor. This questions why not leveraging the online version in all the experiments? Is there an experimental setup where the offline one is much better than the online one? Why doesn't the paper present one method and treat the other as a special case/variant that is less powerful?

2- The main weakness of this paper the lack of baselines in the experimental section. In all of the presented results, a very suboptimal version of TTT is compared against in **one experiment** only. This really questions how strong is IT$^3$/IT$^3$-online is when compared against strong baselines. Here are some suggestions for necessary experiments:

2a. Since TTT/TTT++[A] are directly comparable with IT$^3$, then I suggest to *at least* have them in all of the experiments with one-to-one comparison in terms of batch size and model architecture. Also, consider adding the performance of IT$^3$ under batch size=1 in Figure 3 results.

2b. Another strong baseline that is suitable for both classification and regression tasks is ActMad [B]. A direct comparison against this baseline is also necessary is all the presented experiments.

2c. Another line of work that is directly comparable is Test-Time Adaptation (TTA). TTA works under a more conservative setup where no control over the training process is assumed. It is also important to compare against the current SOTA TTA method EATA [C] or more closely the dataset distillation method from [D] to further demonstrate the superiority if the proposed method.

2d. Since this is a 'test-time' method, a discussion on its computational requirements is necessary. How would the performance be when evaluated under computational constraints [E]?

2e. Benchmarks used in this work are somewhat small scale. Experiments on larger benchmarks such as ImageNet-C [F] and ImageNet-3DCC [G] in the classification setting are necessary. Similar arguments follow for regression tasks where for example one can follow the object detection experiments from ActMAD. 

3- In section 4.1, I am not sure about the Distribution Shift introduced in this experiment. For instance, the performance of non-optimized does not constantly degrade. Why zeroing out features is a good way of modeling distribution shift instead of adding random noise to the features? Can you please comment on this and provide justification for their choice of distribution shift.

4- Missing references: [B, C, D, E, F, G].

[A] TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?, NeurIPS 2021

[B] ActMAD: Activation Matching to Align Distributions for Test-Time-Training, CVPR 2023

[C] Efficient Test-Time Model Adaptation without Forgetting, ICML 2022

[D] Leveraging Proxy of Training Data for Test-Time Adaptation, ICML 2023

[E] Evaluation of Test-Time Adaptation Under Computational Time Constraints, ICML 2024

[F] Benchmarking Neural Network Robustness to Common Corruptions and Perturbations, ICLR 2019

[G] 3D Common Corruptions and Data Augmentation, CVPR 2022

### Questions
Please refer to the weaknesses section (especially point 2). I am happy to raise my score if my concerns are resolved.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work presents IT$^3$ a method for test time training that relies on the concept of idempotence. During training, the authors train the neural network to be able to predict the ground truth label by either conditioning on it (by concatenating it to the input) or by assuming a placeholder value for it (by concatenating a $\mathbf{0}$ value to the input). At test time, the authors then fine-tune the model on unlabelled data by matching the predictions of the model when using $\mathbf{0}$ as the label with that of the model when conditioning on its own output at the $\mathbf{0}$ label. This essentially leads to a soft idempotence constraint which, according to the authors, allows the model to move out-of-distribution data closer to in-distribution ones and thus improve performance when distribution shifts happen at test time. The authors then evaluate IT$^3$ on a variety of tasks and architectures.

### Strengths
* The paper is mostly well written; the ideas are explained clearly and reasonable intuitions are being given. Same goes for the experimental evaluation and discussion of the settings. 
* The idea is simple and I find it quite interesting. While the main architecture of the method relies heavily on the prior work Durasov et al., the application on the TTT setting is novel, since, as far as I know, idempotence hasn’t been explored in such a scenario. 
* The tasks considered in the experiments are quite diverse, which is nice to see. They range from simple tabular data, to standard image classification, to regression and dense prediction with various architectures. 
* The authors demonstrate improvements with IT$^3$ on all settings considered upon (admittedly very simple) baselines.

### Weaknesses
* The main weakness I see in this work is the (almost complete) lack of proper baselines. For example, on only the CIFAR task there is another TTT baseline but on all of the others the authors just compare against not doing anything. This make it hard to gauge significance of the method against other prior art.
* The work makes claims about how idempotence can be seen as a generalisation of a projection and that it allows to map the OOD data to the in-distribution. Thus, while the authors do spend some time to explain why would intuitively their method works, they do not have any ablation studies to verify that these exact intuitions hold in practice. 
* IT$^3$ as a method requires two sequential forward passes through the model, so in practice it can be slow and the authors do not discuss the efficiency of IT$^3$ relative to other works.

### Questions
As mentioned before, I find this work quite interesting but the lack of proper baselines push me towards a weak reject opinion. As for specific questions and ways that the authors can improve their work:
*  More baselines are needed on all tasks; even if the method does not translate exactly to the setup considered, the authors could perform minor adaptations so that it does. For example, why not consider additional self-supervised tasks as discussed at Sun et al. (2020)? In the case where simple rotation prediction might not apply, something simple like denoising an artificially corrupted image could still work as a self-supervised task. Another example would be on the online setup; there, methods from the TTA literature could be applied, such as the work of [1] which works even on the instance level and without batch normalisation.
* Apart from the CIFAR-C case, most other distribution shifts are generated by just partitioning the datasets according to some rule and then training on a subset while considering the other as OOD. This is a bit of a constrained setting and I would encourage the authors to consider more diverse shifts, as that would better highlight the usefulness of IT$^3$. For example, why not add noise to the road segmentation task, in the same way that it was done for CIFAR 10? This could be plausible real world setting where there is a fault in the sensor, thus the images come out corrupted. 
* How is the label encoded in the input in the various settings considered? This is an important information for reproducibility. Furthermore, is the loss at Eq. 2 used for all settings (even when it might not make much sense, such as classification)? 
* In the online setting, the authors consider a smooth transition between the distribution shifts, which might not be practically realistic. How does the method behave when the transition between distribution shifts is not smooth? 
* How many update steps on each datapoint do the authors do? Does the test time optimization reach a fully idempotent model and does “more idempotence” provide better performance? 


[1] Towards Stable Test-Time Adaptation in Dynamic Wild World, Niu et al., ICLR 2023

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces $IT^3$, a generic method for addressing distribution shift challenges. By enforcing idempotence, $IT^3$ sequentially adapts to potential distribution shifts during inference. The authors show the performance of $IT^3$ across diverse scenarios.

### Strengths
The authors extend the concept of TTT (Sun et al., 2020) by incorporating idempotence, offering a simple yet elegant solution. The approach is intuitive and Figure 1 is particularly helpful for quickly grasping the core idea, even for those who are not experts in distribution shift.

### Weaknesses
1. (cf Figure 1) During training, in addition to feeding the standard $(x, y)$ pair to train the model, the authors also input the $(x, 0)$ pair to ensure the model satisfies the property of idempotence, referring to the zero as a neutral “don't know” signal. While this approach may work in classification tasks where zero could represent a new “don’t know” class, in regression tasks, it is unclear how the model differentiates this zero from an actual label of 0.
2. (Line 211) Online $IT^3$ appears to rely significantly on the use of Exponential Moving Average (EMA). However, the authors did not provide a citation for this technique. 
3. In the first experiment (Section 4.1 on tabular data), the method of randomly substituting features with zeros in the test set may resemble a missing data scenario rather than a true distribution shift. In other experiments, the authors simulate distribution shift by splitting the training and testing data based on a specific covariate to create distinct distributions. It is unclear why the same method was not applied for the first experiment. If the authors prefer using the zeroing method, they should include figures or statistical tests to substantiate that the training and testing data are distributionally different, rather than relying solely on intuition.
4. There are unnecessary abbreviations throughout the manuscript. For instance, “such that” is shortened to “s.t.” in line 490. The proposed method, $IT^3$, is inconsistently referred to as ITTT in parts of the manuscript, such as in Figure 13.
5. Figures 14 to 16 are not mentioned or referred to in the main text. This omission is unusual and may confuse readers as to the purpose or relevance of these figures.
6. Figures 12 and 15 have the exact same title.

Some minor improvements and spelling corrections for clarity:
1. (Line 22) $x$ (not $\mathbf{x}$) is not mentioned before.
2. (Line 77) $y_2$ is not mentioned before.
3. (Line 85) "th input" should be corrected to "the input".
4. (Line 131) Missing right parenthesis: $f(f(z))$.
5. (Line 490) “s.t. that” should be replaced with “such that”.
6. (Line 495) Remove the extra colon (":").
7. In Table 1, the title references “qualitative” results, but the data presented are numerical and should be described as “quantitative” results.

### Questions
1. Do you use a special encoding or placeholder value instead of 0 for regression tasks?
2. If the purpose of the neural "don't know" zero is purely for contrast and its specific value is not important, will it be more computationally efficient to use a representative value such as the median of $y_i$ where $i \in \text{training set}$?
3. In EMA (Morales-Brotons et al., 2024), when $\alpha = 1$, the online $IT^3$ aligns with the offline version, and when $\alpha = 0$, it encounters the collapse issue described in Section 3.2. Could you provide guidance on selecting the value of $\alpha$ or share experimental results demonstrating performance across different $\alpha$ values?
4. The right panel of Figure 2 and Figure 16 both appear to represent the car data. Could there be a potential mistake or duplication here?

Morales-Brotons, D., Vogels, T., & Hendrikx, H. (2024). Exponential moving average of weights in deep learning: Dynamics and benefits. Transactions on Machine Learning Research.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
- This paper proposes a test-time-training based approach to address the distribution shift or OOD generalization problem by learning models that are idempotent.

- In particular, this paper proposes a method where models $f_{\theta}: \mathcal{X} \times \mathcal{Y} \to \mathcal{Y}$ are trained by minimizing both the difference between $f_{\theta}(x, y)$ and $y$ as well as the difference between $f_{\theta}(x, 0)$ and $y$, resulting in a model that is idempotent on the training set. Then at test time, models are optimized on the test data before running inference to make them idempotent on test inputs, by minimizing the difference between $f_{\theta}(x, 0)$ and $f_{\theta}(x, f_{\theta}(x, 0))$ for an OOD input $x$.

- The paper shows empirically for several different settings that idempotent test-time-training improves classification accuracy on out-of-distribution samples.

### Strengths
- Idempotent training is an interesting and novel approach for addressing the out-of-distribution generalization problem.

- The paper shows successful application of the considered approach on a large number of experimental settings, including in online-learning settings.

### Weaknesses
- Test-time training requires undesirable extra compute for test-time optimization of the whole model. How much additional compute is needed compared to running inference on the base model? How does this method scale with model size?

- The paper lacks analysis that explains how or why idempotent training is expected to improve out-of-distribution analysis. Further investigation and ablations that provide intuition for how this method works would be valuable.

- The proposed method has not been evaluated on standard real-world out-of-distribution generalization benchmarks such as DomainBed [Gulrajani and Lopez-Paz 2020] and WILDS [Koh et al 2020]. The presented experiments are on smaller models/datasets.

References:

Gulrajani, Ishaan, and David Lopez-Paz. "In search of lost domain generalization." arXiv preprint arXiv:2007.01434 (2020).

Koh, Pang Wei, et al. "Wilds: A benchmark of in-the-wild distribution shifts." International conference on machine learning. PMLR, 2021.

### Questions
- Does the test time objective result in networks that are idempotent on OOD samples? Presumably once the function drifts from the fixed anchor function that test time loss no longer reflects idempotence? It would be good to see measures of idempotence on training and test samples.

- How does the intuition about idempotence being a generalization of orthogonal projection hold up? The proposed method considers idempotence only in the y-variable, not the entire function.

### Soundness
2

### Presentation
4

### Contribution
2
