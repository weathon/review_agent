# Training-free Linear Image Inversion via Flows

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 8, 8, 8

## Abstract
Training-free linear inversion involves the use of a pretrained generative model and---through appropriate modifications to the generation process---solving inverse problems without any finetuning of the generative model. 
While recent prior methods have explored the use of diffusion models, they still require the manual tuning of many hyperparameters for different inverse problems. 
In this work, we propose a training-free method for image inversion using pretrained flow models, leveraging the simplicity and efficiency of Flow Matching models, using theoretically-justified weighting schemes and thereby significantly reducing the amount of manual tuning.
In particular, we draw inspiration from two main sources: adopting prior gradient correction methods to the flow regime, and a solver scheme based on conditional Optimal Transport paths.
As pretrained diffusion models are widely accessible, we also show how to practically adapt diffusion models for our method.
Empirically, our approach requires no problem-specific tuning across an extensive suite of noisy linear image inversion problems on high-dimensional datasets, ImageNet-64/128 and AFHQ-256, and we observe that our flow-based method for image inversion significantly improves upon closely-related diffusion-based linear inversion methods.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes a flow matching model (or equivalently, diffusion model-based) based inverse problem solver. Unlike previous diffusion model-based approaches which are based on stochastic samplers, the proposed method is based on ODE integration. $\Pi$GDM approximation is used to estimate the time-dependent log likelihood is used. The method is shown to outperform previous arts ($\Pi$GDM, RED-Diff) on some datasets: ImageNet-64/128, and AFHQ-256.

### Strengths
1. To the best of my knowledge, this work is the first to use flow matching models (although they can be thought of as equivalent to diffusion models) to solve linear inverse problems.

2. The method is easy to understand, given that it builds on the prior approximation of $\Pi$GDM.

### Weaknesses
1. (Limited contribution) As stated in S1, the proposed method boils down to $\Pi$GDM that uses ODE sampler, rather than the stochastic DDIM sampler ($\eta = 1.0$) that $\Pi$GDM uses. The step size derived from Lemma 2 is also another way of stating what was already shown from DDRM and $\Pi$GDM.

2. (Choice of test dataset) On most of the diffusion model-based inverse problem solvers, testing is performed on two canonical datasets: FFHQ 256$\times$256 and ImageNet 256$\times$256. These two datasets are typically more challenging than the datasets that are used in this paper. It is hard to convince the strength of the paper when there is no specific reason to deviate from this standard. The method would be more convincing if the same superiority can be seen in such largely used benchmarks.

3. (Exposition) In the preliminaries, it is unclear why one has to start from *conditional* diffusion models and *conditional* flow models, when at the end of the day, unconditional models will be used to try to sample from the posterior distribution $p(\mathbf{x}_1|\mathbf{y})$.

### Questions
1. The authors repeatedly use the term *image inversion* throughout the work. Instead, I would advise to use the term *inverse problems in imaging*, which is a standard term that has been used for decades.

2. Is Algorithm 1 different from $\Pi$GDM other than the fact that it is an ODE sampler? Specifically, assume that $\eta = 0.0$ from the $\Pi$GDM DDIM sampling. Are these two equivalent in this case, given that VP-ODE is used?

3. (Implementation details) It is unclear why the authors trained a class conditional model, where the standard is to use class unconditional models. The former is known to perform better than the latter.

4. Why does VP-ODE perform (marginally) better than OT-ODE? The theory of flow matching models would state otherwise.

5. (pg 9. last paragraph) Did you mean to cite DPS when you mention non-linear observations? PSLD would not be able to solve non-linear inverse problems due to the gluing term used in the paper.

### Soundness
3 good

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
They propose a training-free method for image inversion using pretrained flow models. Their approach leverages the simplicity and efficiency of Flow Matching models, significantly reducing the need for manual tuning. They draw inspiration from prior gradient correction methods and conditional Optimal Transport paths. Empirically, their flow-based method improves upon diffusion-based linear inversion methods across various noisy linear image inversion problems on high-dimensional datasets.

### Strengths
- Theoretical analysis is conducted based on linear corruption regarding conditional flow matching.
- Extensive experiments are performed to validate the proposed method from various perspectives.

### Weaknesses
- I wonder whether the linear inversion in Sec3 can handle a wide range of data corruption in the real world.

I believe this paper adequately addresses the problem and conducts various experiments. Overall, I think it is a good paper and I would like to accept it. Honest speaking, I am starting to use it as my new baseline for another submission.

### Questions
as above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a training-free method for linear image inversion using pretrained flow models. The proposed method has theoretically-justified weighting schemes and thus require less hyperparameter-tuning. The authors show effectiveness of the proposed method on common high-dimensional datasets and compare to prior diffusion-based linear inversion methods.

### Strengths
1. The paper is clear and organized well. The visualized results are impressive.
2. The proposed method is well-motivated and is justified theoretically. Requiring less hyperparameter tuning is an important practical advantage for inverse problems.
3. Experiments is generally thorough and solid. Results reported on ImageNet and AFHQ demonstrated the effectiveness of the proposed method comparing to prior works.

### Weaknesses
[Minor]
1. In the paper, it seems DPS is assumed to have worse performance than ΠGDM as "ΠGDM is improved upon DPS". Empirically, is it also observed that DPS has worse performance than ΠGDM and the proposed method?
2. In Figure 2, the x-axis seems to be not continuous, but adjacent points are connected and interpolated with lines, which might be not intuitive.

### Questions
In Figure 5, ΠGDM shows green-dot artifacts, is this common for ΠGDM or rare cases?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper explores the application of flow models to linear inverse problems. It specifically extends recently introduced techniques from diffusion models to flow models and offers a method for converting diffusion models into flow models. The paper also conducts a comprehensive set of experiments to validate these propositions.

### Strengths
While many papers have recently concentrated on approaches that utilize diffusion models for inverse problems, there appears to be a noticeable gap in the literature concerning the application of flow models in this context. This paper aims to bridge that gap. The extensive numerical results appear to suggest that their approach outperforms diffusion models, although it's worth noting that, at times, distinguishing differences or identifying major issues in the produced images can be challenging.

### Weaknesses
The paper's main weakness lies in its writing. It appears that Section 2, labeled 'Preliminaries,' and the rest of the paper are written with the assumption that readers are already well-versed in flow-based models trained with flow matching. This might create challenges for readers who are not familiar with this background. Considering the relative novelty of flow matching, particularly in comparison to diffusion methods, it would be beneficial for the authors to provide some background information before transitioning to the conditional case

### Questions
- Could the authors offer some insight into why having probability paths that are straighter than diffusion paths might be advantageous for reconstruction in inverse problems?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
