# Epistemic Generative Adversarial Networks

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Generative models, particularly Generative Adversarial Networks (GANs), often suffer from a lack of output diversity, frequently generating similar samples rather than a wide range of variations. This paper introduces a novel generalization of the GAN loss function based on Dempster-Shafer theory of evidence, applied to both the generator and discriminator. Additionally, we propose an architectural enhancement to the generator that enables it to predict a mass function for each image pixel. This modification allows the model to quantify uncertainty in its outputs and leverage this uncertainty to produce more diverse and representative generations. Experimental evidence shows that our approach not only improves generation variability but also provides a principled framework for modeling and interpreting uncertainty in generative processes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method for incorporating Dempster-Shafer Theory into the GAN framework to achieve uncertainty modeling. By adjusting the model architecture and setting appropriate loss functions, a novel EpistemicGAN is introduced, demonstrating superior performance compared to standard GANs. Experimental results support this conclusion.

### Strengths
1. The introduction of Dempster-Shafer Theory into GANs represents a novel perspective that may contribute to a deeper understanding of GANs.

2. The paper provides an exceptionally thorough background introduction, enabling even readers with no prior knowledge of Dempster-Shafer Theory to comprehend its content.

### Weaknesses
1. The paper only provides a brief introduction to the methods and experiments. This may be due to page limitations. The authors should consider moving some background material to the supplementary materials, accessible to readers unfamiliar with Dempster-Shafer Theory, in order to preserve more space for describing the methods and experiments.

2. Regarding the methodology section, I have some concerns about the soundness of the method. I will detail my specific concerns in the Questions section. If the authors can provide a reasonable explanation, this would not be a weakness.

3. The experimental section introduces too few baselines. I understand the authors' concern that SOTA models are complex and may obscure the gains brought by the proposed method. However, models utilizing different distances (e.g., Wasserstein GAN or those using hinge loss) should be included to demonstrate the method's adaptability. Additionally, large SOTA models should be considered for additional results to demonstrate that the proposed method does not compromise model soundnes (does not degrade performance) within large, complex frameworks. The current experiments only prove effectiveness on DCGAN, which is a highly limited finding.

### Questions
1. Regarding the discriminator, how can we ensure the model outputs the expected belief pairs? The belief values are output by the model, and only the loss function constrains the sum of the two values to be 1. I believe this is insufficient. If it merely outputs probability values for real and fake, this requirement could still be met. Furthermore, this approach is quite similar to EBGAN's methodology, differing only in theoretical interpretation.

2. In the generator section, does the enhancement in diversity stem from uncertainty modeling or the added generator loss term? Since the authors claim this approach primarily improves output diversity, would directly constraining the variance of the latent space distribution be feasible? What is the significance of introducing uncertainty modeling? There is nerther discussions nor ablation experiments.

3. What does Figure 5 show? In my view, there is no apparent difference between the left and right images. There is no analysis.

### Soundness
2

### Presentation
2

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
The authors propose Epistemic GAN, integrating Dempster–Shafer (DS) evidence theory into both the GAN discriminator (which predicts belief functions rather than scalar probabilities) and the generator (which is redesigned to output region-wise mass functions using a Dirichlet parameterization that is later decoded into the final image). The authors also derive evidential adversarial losses consistent with belief axioms and add regularizers intended to balance diversity vs. stochasticity. Experiments on CelebA, CIFAR-10, and Food-101 report improved FID and Vendi diversity scores over a standard DCGAN baseline.

### Strengths
1.	Converting the discriminator output and the intermediate generator representation into a belief/mass function may be an original take on uncertainty-aware GAN training. The DS tutorial is clear and self-contained.
2.	The two-stage generator with a Dirichlet proxy for interval hypotheses is well-motivated and concretely described, with helpful schematics.
3.	The paper is generally well-written and easy to follow.

### Weaknesses
1.	The paper compares mainly to a “standard” DCGAN (from 2015). It omits established diversity-oriented GAN variants and modern strong baselines (StyleGAN, or diffusion-based contenders using diversity metrics). Without these, it’s hard to isolate how much of the gain stems from the evidential machinery versus general architectural changes. 
2.	The paper lacks ablations for (i) evidential loss in the discriminator only, (ii) mass-predicting generator only, (iii) the Dirichlet interval mapping vs. other parameterizations, and (iv) regularizer weights. This is important to validate the necessity of each component. 
3.	There’s no report of training time, memory, convergence behavior, or sensitivity to Dirichlet concentration \alpha and interval sampling noise; evidential training can introduce optimization quirks.
4.	All experiments are on relatively modest resolution/complexity datasets; there’s no conditional generation, no higher-resolution benchmarks.
5.	Generator-1/Generator-2 selection and capacity parity are unclear. The paper introduces two generator components but does not specify how they are selected/combined during training and inference, nor whether their total parameter count/FLOPs are matched to DCGAN. Without capacity-matched comparisons, the observed gains might simply arise from increased model size or ensemble-like effects rather than the evidential design.

### Questions
1.	Please broaden more SOTA baselines; optionally, compare to a diffusion baseline using the same metrics.
2.	Please add Ablation studies for discriminator/generator evidential components and regularizers; sensitivity to hyper-parameter and interval sampling variance.
3.	Compute/stability report: wall-clock, GPU hours, parameter counts, and failure modes.
4.	Please report architecture details, parameter counts per module, FLOPs, and training-time costs, and clarify whether the two generators are sequential stages or parallel heads and how their outputs are weighted/used.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper seeks to address the problem of mode collapse in generative adversarial networks (GANs) where the generator's outputs mostly  ignore the noise input and generate a few only a few samples instead of covering the data distribution more completely.

It does so by utilizing the Dempster-Shafer theory of evidence, an alternative perspective on viewing beliefs and uncertainty. To utilize the theory, this paper proposes some novel architectural modifications and subsequent modifications to the GAN loss to incorporate a measure of uncertainty in the discriminator and the generator. It does so by modifying the output of the discriminator to predict its "belief" that the input is real or fake instead of a single output, and adding an intermediate layer to the generator that predicts an interval per "region" over which to sample values.

### Strengths
* The underlying theory behind the proposed approach is interesting and could be of some interest to the community.
* The explanation of the Dempster-Shafer theory is also fairly well done.
* The modifications to the architecture and the loss function seem to be well motivated based on the above theory and its explanation
* In experimental evaluation, using the same base architecture and compute constraints is laudable.
* The datasets used for evaluation seem fine, more details on it in next section
* The two metrics being used for evaluation are also accepted and seem like good measures for how well the overall system is performing.

### Weaknesses
* While the underlying motivation and theoretical underpinnings are appropriate and understandable, I am not convinced that it's practical instantiation in the modifications to the discriminator and generator has been well justified or evaluated in the paper.
* There are two distinct architectural changes proposed in the paper. I was expecting each of these changes to be evaluated separately, not only on the final metrics, but on samples and the loss themselves.
* Specifically for the discriminator: No experiments showing whether the sum of real and fake beliefs are lower than one for a proportion of inputs. Nor any experiments showcasing examples when the discriminator is uncertain. The paper should present experiments to highlight how this architectural modification is helping. It should also show a hyperparameter sweep to indicate how sensitive the training is to the hyperparameter $\lambda$
* For the generator, the experiments don't seem to evaluate or showcase this region wise uncertainty. It is hard to visualize the how uncertainty across regions can correlate given the description of the approach in the text. Similar to above, experiments should show how sensitive the training is to the hyperparameters $\beta$ and $\gamma$.
* The use of the $b_{real} +b_{fake} \leq 1$ constraint needs to be explained better in the text. This constraint seems to be the most important bit in differentiating the modified discriminator architecture from a single output discriminator.
* While comparing to a single underlying architecture is commendable, the paper should also compare to some of the other approaches proposed to prevent mode collapse if that is what this architecture is supposed to do.
* There is no statistical test to indicate whether the proposed technique is statistically improving upon the baseline. How many times was the experiment run, what was the variance, is the result significant, all of these questions should be part of the experimental evaluation.
* Perhaps it is the number of samples, but I cannot differentiate between the two techniques in Figure 5, and thus the qualitative test is a failure in my mind.
* The DCGAN baseline seems like a very old one to compare to. Perhaps compare to some more recent baseline that has solved a lot of the inefficiencies and problems of early GANs?

### Questions
* What does a belief sum less than 1 mean? Why is the subsequent two output architecture more expressive compared to a single discriminator value?
* The paper mentions that the modifications to the discriminator allow gradients to the generator to reflect "not only correctness but also confidence levels". Perhaps the answer to the previous question will clarify this one as well, but how do the modifications provide this signal?
* The experimental section does not specify how the DCGAN generator was modified for the experiments. Where was the dirichlet layer added in?
* While using the same base architecture for fairness is laudable, do the proposed modifications increase the number of parameters significantly? Please post the number of parameters in the baseline and the proposed method with DCGAN

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Epistemic GAN, where uncertainty is embedded in both the generator and the discriminator by replacing probability functions with belief functions. The discriminator, instead of a definite real/fake probability, gives both the confidence in the input being real and it being fake, with both summing to no larger than 1. The generator, instead of being entirely deterministic, is separated into two modules, where the first module produces a feature map where each element is a parameter vector of a Dirichlet distribution with 3 categories, from which an interval can be sampled, which is then passed to the second module to produce the output image. The proposed model is compared with a baseline GAN on three datasets using FID and Vendi score, and shows minor improvements.

### Strengths
This paper introduces the concept of belief into adversarial training, which is interesting and merits further investigation.

### Weaknesses
1. My main concern lies in the disconnect between the proposed theoretical formulation and its practical implementation. Although the paper introduces the model using the language of belief and interval theory, these concepts are not meaningfully reflected in the training process. In the discriminator, the outputs $b_{real}$ and $b_{fake}$ are described as belief measures for real and fake samples, yet aside from a penalty enforcing $b_{real}+b_{fake} <=1$, they behave identically to a standard GAN objective. In fact, by reparameterizing $b^*_{fake}=1-b_{fake}$, the loss function effectively reduces to the vanilla GAN loss, suggesting no substantive difference in optimization dynamics. Consequently, the discriminator is likely to converge to $b_{real}+b_{fake}=1$, rendering the proposed belief interpretation redundant. Similarly, in the generator, the notion of “intervals” is introduced but never operationalized, the intervals are simply passed as numerical pairs without any mechanism enforcing or exploiting their interval semantics. This makes the architecture theoretically appealing but practically equivalent to a conventional GAN.
2. The experimental validation is also weak. The authors rely solely on a comparison against the outdated DCGAN baseline, without testing on stronger or modern models such as StyleGAN or BigGAN. Only a single figure is provided, showing a small set of low-quality samples from one dataset, making it impossible to assess diversity or fidelity. Subjectively, the generated images appear worse than the DCGAN baseline. Additionally, the presentation quality is poor,  the figure panels are misaligned, borders inconsistent, and captions off-center. Overall, both the experimental design and results presentation need significant improvement to substantiate the claimed advantages.
3. No Ablation Study. The contribution of each component (belief loss, Dirichlet variance, interval width) is not separately quantified.

### Questions
1. The Dirichlet distribution needs clearer explanation. My understanding is that it has exactly 3 categories, making it supported on a triangle where each point represents an interval in $[0, 1]$. Is this correct? If so, the utility of the variance term in the generator's loss function is questionable. I don't see how a larger variance in this distribution of intervals translates to higher variation in generated samples. Moreover, regularizing the first module's output alone won't ensure generator diversity, since the second module can still collapse everything to a few modes. Additionally, the variance and precision terms appear to counteract each other.

2. How is the model trained? Since the generator samples from a Dirichlet distribution, it cannot be trained end-to-end without modification. I assume reparameterization is used, similar to VAEs.

### Soundness
2

### Presentation
1

### Contribution
2
