# ReTrace: Reinforcement Learning-Guided Reconstruction Attacks on Machine Unlearning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Machine unlearning has emerged as an inevitable AI mechanism to support GDPR requirements such as revoking user consent through the "right to be forgotten". 
However, existing approaches often leave residual traces that make them vulnerable to data reconstruction attacks. 
In this work, we propose ReTrace, the first reconstruction attack framework that uniquely formulates unlearned data recovery on large-scale deep architectures as a reinforcement learning (RL) problem. 
By treating residual unlearning traces as reward signals, ReTrace guides a generator to actively explore the input space and converge toward the forgotten data distribution. 
This RL-guided approach enables both instance-level recovery of individual samples and distribution-level reconstruction of unlearned classes. 
We provide a theoretical foundation showing that the RL objective converges to an exponential-tilted distribution that amplifies forgotten regions. 
Empirically, ReTrace achieves up to 73.1\% instance-level recovery and reduces FID and KL scores beyond two state-of-the-art baselines. 
Strikingly, on the challenging task of text unlearning, it improves BLEU scores by nearly 100\% over black-box baselines while preserving distributional fidelity, demonstrating that RL can recover even high-dimensional and structured modalities. Furthermore, ReTrace demonstrates effectiveness across both convolutional (ResNet) and transformer-based models, with Distil-BERT as the largest architecture attacked to date. These results show that current unlearning methods remain vulnerable, highlighting the need for robust and provably private mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ReTrace, which seeks to reconstruct unlcearned training samples assuming access to the model before and after unlearning. ReTrace uses a reinforcement learning to learn a sample that is likely to minimize the loss functions of the two models before and after unlearning.

### Strengths
Understanding and objectively measuring the effectiveness of unlearning is an important problem. The paper presents a novel and interesting use of RL in this space. Overall, I think as a measurement method, there are some benefits to ReTrace as the results seem to suggest that exact unlearning is more effective than approximate unlearning, which is a nice, though somewhat expected result.

### Weaknesses
The paper claims to evaluate both ResNet and DistilBERT, but the majorit of results are in the image domain. Average attack success rates is only around 50% for exact unlearning and 60% for approximate unlearning, and worse on the text tasks, so the attack is not that  effective. As an actual attack, ReTrace doesn't seem that realistic: as it assumes that an adversary can access both pre- and post-unlearning models while running a costly RL process. The paper would have been stronger if the authors could have further compared the relative strenghts of different unlearning proposals using ReTrace and used their method to explain some of the differences.

### Questions
I found some results confusing: some trace score patterns in Figure 2 and Figure 6 are confusing as the trace score doesn't necessarily correlate with being unldearned or not —for example, (0,0) has a very low trace score despite being unlearned, while other points like (1,0) and (1,1) show high scores even though they are not unlearned. The bottom-right case (4,4) is inconsistent across black-, gray-, and white-box settings -- an explanation would be nice.

### Soundness
3

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
This paper presents ReTrace, a new reconstruction attack that reinterprets the problem of recovering unlearned data through the lens of reinforcement learning (RL). Instead of relying on static optimization or inversion techniques, ReTrace uses discrepancies between the original and unlearned models as reward signals to guide a pretrained generator toward regions of the input space that likely correspond to forgotten data. The approach integrates multiple trace signals—changes in predictions, losses, and gradients—across different access levels (black-, grey-, and white-box) and performs reconstruction through RL-guided latent exploration followed by a candidate refinement step. Through both theoretical analysis and experiments on multiple datasets, the paper shows that ReTrace can recover semantically meaningful data at both the instance and distribution levels, highlighting residual vulnerabilities in existing unlearning methods.

### Strengths
- **Novel and creative formulation**. This paper presents a genuinely original framing of reconstruction attacks. It adapts the RL-GAN-Net idea from Sarmad et al. (2019)—originally proposed for conditional image generation—and re-purposes it for unlearning data recovery. Using reinforcement learning to guide a generator’s latent exploration based on unlearning traces is both conceptually interesting and technically innovative.

- **General and modular framework**. The approach unifies different attack settings (black-, grey-, and white-box) under a single reward formulation, where prediction, loss, and gradient discrepancies can be combined or omitted depending on model access. This modularity makes the framework broadly applicable and easy to adapt to new unlearning scenarios.

- **Empirical evidence of residual traces after unlearning**. The experiments clearly demonstrate that models subjected to unlearning still leak identifiable information, confirming the practical relevance of the attack.

- **Largest-scale model targeted to date**. The paper extends reconstruction attacks beyond CNNs to transformer architectures, reporting results on DistilBERT, which the authors state is the largest model attacked to date in the unlearning literature. This demonstrates the framework’s scalability and suggests that the proposed RL-based formulation might have the potential to generalize to high-dimensional transformer settings.

- **Timely and relevant contribution**. The work addresses the emerging area of machine-unlearning security—a topic of growing importance for model safety, privacy, and regulatory compliance—and provides a concrete framework for analyzing these vulnerabilities.

- **Theoretical grounding supporting intuition**. The paper includes a concise theoretical analysis that explains why the proposed RL formulation works, strengthening the intuition behind the method.

### Weaknesses
I like this paper — it’s a creative and well-motivated idea. That said, there are a few points I would like to raise and hopefully discuss with the authors.

- **On the ambiguity in the RL formulation**. Section 3.2 introduces the RL framing with definitions of state, action, and policy, but these descriptions are somewhat abstract and internally inconsistent when mapped to the actual image-generation setup. The text defines the action as “sampling or refining a candidate x from the generator,” implying that the policy acts in the data space, while simultaneously describing the policy πϕ as “outputting candidates from the latent space,” implying it acts over z. This leaves it unclear whether the RL agent’s action space is x or z. Appendix D.1 later reveals that πϕ is in fact a small two-layer MLP producing latent vectors z that are then passed through a pretrained DCGAN G to obtain x = G(z). This architectural detail is crucial for understanding the proposed RL loop but is only specified in the appendix under the Experimental Setup Section. I would strongly recommend that the authors move this clarification into the main text so readers can immediately understand what components are being optimized and how gradients flow.

- **On the mathematical clarity and internal consistency**. While the paper’s theoretical framing is interesting, I found the mathematical presentation scattered and internally inconsistent. Symbols are introduced but never used or formally defined. For example, T(x) is defined once (Eq. 5) and never referenced again. The trace score s(x), which seems to be the central quantity guiding the policy updates, is described conceptually but never expressed mathematically. In addition, a reward r(x) is defined (Eq. 6); if I understood correctly, it corresponds to −s(x), but this relationship is never made explicit. In Eq. (7), it is also unclear what pϕ denotes—whether it is simply the policy distribution πϕ or a learned variant of the prior p0. The term Dpub is briefly defined as “a publicly available dataset with a similar distribution,” but its operational role remains vague. Is Dpub the same dataset used to pretrain the DCGAN generator, or is it only used for the KL regularization term? Clarifying this would help connect the regularized RL objective to the actual implementation.

- **on distribution-level comparison with baselines**. While the paper reports FID and KL metrics for ReTrace across datasets and access levels, it does not provide corresponding values for baseline methods (e.g., UIA, HRec). Since FID and KL are the primary metrics used to evaluate distribution-level reconstruction quality, the lack of direct comparison makes it difficult to assess whether ReTrace actually improves over existing approaches in recovering the overall deleted-data distribution.

### Questions
I would appreciate it if you could also answer my questions: 

**Q1**. I might have missed it in the paper, but I don’t fully understand— as also mentioned in my weakness section—whether the optimization in Equation (7) aims to maximize the trace score 𝑠(𝑥) or the reward 𝑟(𝑥) and what the motivation is for defining both and how they are related.

**Q2**. My understanding is that, according to Equation (9), the instance-level reconstruction step selects a single top-scoring candidate via arg max 𝑠(𝑥) and refines that sample. If that is correct, could the authors clarify how the multiple instances shown in Figure 4(a) are reconstructed? A related question: did the refinement step lead to a significant improvement in reconstruction quality?

**Q3**. I noticed that DCGAN is used as the generative model throughout the experiments. Could the authors elaborate on the reasoning behind this choice? Given the many stronger generative models introduced in recent years (e.g., StyleGAN, diffusion models), wouldn’t using a more advanced generator potentially improve reconstruction quality?

**Q4**. In the appendix, it’s mentioned that the DCGAN produces 32 × 32 images. If I understood correctly, some of your evaluation datasets (e.g., Food-101) are higher-resolution. Could you clarify how this resolution mismatch is handled? In particular, how are comparisons with baselines such as UIA and HRec made—do these methods also operate at 32 × 32 resolution, or were their outputs downsampled to ensure fairness?

As I mentioned before, I like the core idea and find it creative and promising. However, I would need the authors to clarify the points raised in the weakness section—particularly by adding distribution-level comparisons against baselines—and address the questions above before I could confidently give this paper a clear accept.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents an attack on machine unlearning using RL-based data reconstruction. The work uses residual learnings as rewards and is comprehensively evaluated on several data samples using blackbox access on standard benchmark datasets. The findings indicate the feasibility of performing such attacks on large scale datasets.

### Strengths
- important problem and timely issue
- well-carried out attack framework without much prior assumptions
- good comparison on several benchmarks and good theoretical foundation

### Weaknesses
- comparison is limited to a few models and datasets and can easily be expanded broader
-  Convolution Transpose,instance-wise unlearning,  or Masked Small Gradients methods could have also been studied
- would have been great to mention computational costs and complexities

### Questions
The paper presents a good study and evaluation of hidden traces and connections leading to success in attacks against unlearning models. The work is well presented, though some of the popular models and approaches for unlearning have not been explored. I wonder if the comparison can benefit from the larger set of models and methods studied in "Deep Unlearn: Benchmarking Machine Unlearning for Image Classification" in EuroS&P'25. 

What defenses can be used to mitigate the attacks mentioned in the paper?

It would be great to discuss the complexities of the attack and if it will be realistic to carry out against large datasets and bigger models.

### Soundness
3

### Presentation
4

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
The paper proposes RETRACE, a reconstruction attack on machine unlearning that frames recovery of deleted data as a reinforcement-learning (RL) problem. The method extracts “unlearning traces” by contrasting a pre-unlearning model  with a post-unlearning model at prediction, loss, and gradient levels, and uses these as rewards to train a generator. The theory claims the RL objective converges to an exponential-tilted policy that emphasizes high-trace regions; experiments report strong instance-level recovery and improved distributional alignment (lower FID/KL) versus UIA and HRec, plus preliminary text results with DistilBERT.

### Strengths
1. The paper proposes a new solution based on RL to address a newly proposed threat model against machine unlearning. 
2. The paper considered both instance-level machine unlearning and class-unlearning setting in their approach.
3. Authors consider three levels of access to the original model and retrained model (black-box, grey-box, and whitebox), and perform most their evaluation on the three settings.

### Weaknesses
1. Some notations are used without proper definitions and clarifications. For example in line 141, $x_{\pi_{(j)}} is not defined.
2. Some of the details of the proposed approach are missing, which makes it confusing for the reader. What is the initial $z$ that you use in PPO? The details are missing while I think it might affect the outcome. I think the paper would benefit more from a pseudocode instead of figure 1 that does not provide any useful information.
3. From assumptions 1 and 2 of the paper, it seems authors assume access to the population data of the forget class, which is a more restrictive assumption of access to the population data of the original training set as in prior work [1].
4. In general, while this paper builds on an earlier paper for its setting [1], I don’t believe the setting of the attack specifically is specifically relevant to unlearning, but it is more in general about getting two models, one of which is trained on a subset of the training data used for the other, and try to reconstruct that data. So I think it is basically a privacy attack using model differentiation.  That is also why in [1], they only use the retrained model for their evaluations. I believe the assumption about gaining access to the original model in the setting of unlearning is not realistic.
5. Although the link for the code has been added and reviewers are referred to in the reproducibility statement, the repository was empty and only the score computations method (equation 5) was provided.
6. Your generative model relies on a DCGAN for generating the data that is missing. However, the pretrained DCGAN data is already trained on the unlearned classes and is capable of generating samples from the unlearned class. Therefore, it wouldn’t be surprising if DCGAN is capable of generating samples whose MSE from the original forget sample fall within the same range as the variance of the samples of that class. Basically, if I want to rephrase what you point out about the MSE value it would be sth like: you have a sample from the ‘Bed’ class that has been removed from the training set. Then you use your method to generate the image of a sample that can be considered a ‘Bed’, but not necessarily the same sample that was unlearned. Given that DCGAN is able to generate images of ‘Bed’ achieving that would not be very surprising. I think some of the confusion about this could be resolved once you respond to weakness 2, specially if based on assumption 1 of the paper, the adversary starts with a prior for the unlearned class.
7. I think the provided figures are not very informative as they are now. For example, looking at figure 8, I don’t see any significance on the values of the selected squares. Even in figure 2 the advantage of white-box over black-box is not clear from the figure. In white box all the values seem to be larger, not only the forgotten samples.
8. The results in the table are not accompanied by standard deviations. For example in table 1 the difference in CS score or MSE score for white-box vs grey-box is only at most 0.02, which might simply be smaller than the variance of the data.
9. Your theoretical results rely on the assumption that the expectation of the score assigned to the samples from the forgotten samples are strictly larger than this expectation for the retained data. However, the score that you defined in equations 2,3, and 4 do not seem to necessarily follow this assumption. For example, if you train a model on the retained data and train a model on the whole data, the loss of these models, should be more similar on the retrained data compared to the forget samples. I think this assumption should be at least accompanied by some empirical observations.

[1] Bertran, M., Tang, S., Kearns, M., Morgenstern, J. H., Roth, A., & Wu, S. Z. (2024). Reconstruction attacks on machine unlearning: Simple models are vulnerable. Advances in Neural Information Processing Systems, 37, 104995-105016.

### Questions
1. In the setting of the problem, it is assumed that the adversary has access to the auxiliary public dataset $D_{\mathrm{pub}}$. However, it is not clear from the paper what the initial $z$ in the experiment is? It has to be clarified what initial $z$ is chosen in the optimization (the initial state). Could the authors please elaborate on the details (with specific examples on the dataset and forgotten sample ideally)?
2. Have you tested your method on a class that DCGAN has not been trained on? What would happen in that case?
3. Could the others provide some plots on the number of iterations used in PPO and how the metrics they use (e.g., MSE) changes along this optimization? 
4. Could the authors provide normalized values in figure 2 to show-case the improvement due to more information in the white-box setting. I would suggest computing the average value for all the 25 images in the patch and then reporting the ratio of the values for the red squares over the computed average value for the corresponding patch.
5. To maximize equation 7, as mentioned in line 214, you need to maximize the reward given in equation 6. For that you need to minimize the differences (due to negative signs). But this would mean samples that the retrained model and original model would act similar on (which would be the retained data) and for example for the retrained data, we would expect equation 2,3, or 4 achieve the smallest values. So why the model should converge to the forget samples. Could the authors please address this confusion I had when reading the approach?
6. In line 836, you mention exact unlearning method is implemented by “fine-tuning the model on the remaining data for the same number of epochs as the original training”. Why not training the model from scratch instead of fine-tuning the model on the remaining data. In practice the exact unlearning models are derived by fine-tuning the model from scratch because the fine-tuned model still contains information about the forget data and is not equivalent to the retrained model. 
7. Currently the authors only evaluate the effectiveness of their method that rely on SGD update (either gradient descent on remaining samples or gradient ascent on the forget samples). It would be interesting to see the effectiveness of the attack on the two following settings that are shown to be more successful than GA:
    - Using sparsification methods that only perform SGD updates on a subset of the parameters [1,2].
    - Unlearning methods that do not rely on SGD on either of the remaining sets and forget sets [3,4].

[1] Jia, J., Liu, J., Ram, P., Yao, Y., Liu, G., Liu, Y., ... & Liu, S. (2023). Model sparsity can simplify machine unlearning. Advances in Neural Information Processing Systems, 36, 51584-51605.
[2] Fan, C., Liu, J., Zhang, Y., Wong, E., Wei, D., & Liu, S. (2023, October). SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation. In The Twelfth International Conference on Learning Representations.
[3] Chen, M., Gao, W., Liu, G., Peng, K., & Wang, C. (2023). Boundary unlearning: Rapid forgetting of deep networks via shifting the decision boundary. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7766-7775).
[4] Ebrahimpour-Boroojeny, A., Sundaram, H., & Chandrasekaran, V. Not All Wrong is Bad: Using Adversarial Examples for Unlearning. In Forty-second International Conference on Machine Learning.

### Soundness
2

### Presentation
2

### Contribution
2
