# Linear combinations of latents in generative models: subspaces and beyond

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 5, 6, 5

## Abstract
Sampling from generative models has become a crucial tool for applications like data synthesis and augmentation. Diffusion, Flow Matching and Continuous Normalising Flows have shown effectiveness across various modalities, and rely on latent variables for generation. For experimental design or creative applications that require more control over the generation process, it has become common to manipulate the latent variable directly. However, existing approaches for performing such manipulations (e.g. interpolation or forming low-dimensional representations) only work well in special cases or are network or data-modality specific. 
We propose Latent Optimal Linear combinations (LOL) as a general-purpose method to form linear combinations of latent variables that adhere to the assumptions of the generative model. As LOL is easy to implement and naturally addresses the broader task of forming any linear combinations, e.g. the construction of subspaces of the latent space, LOL dramatically simplifies the creation of expressive low-dimensional representations of high-dimensional objects.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work examines the topic of finding low dimensional subspaces and semantically meaningful interpolant paths between the latent vectors of generative models, such as diffusion and flow models. The authors identify and validate normality tests that confirm their hypothesis that poor visual quality is a product of latent vectors being out of distribution for a model. They propose a simple and easy-to-implement approach for producing interpolants that lead to high quality generation.

### Strengths
The paper is very well written. The problem statement, background, identification of the issue, and proposed solution flow well and are easy to follow.

The visual evidence supporting the author’s proposed tests for identifying “failing” latents and for validating their proposed COG approach are compelling. Overall, the experimental validation is thorough and well executed.

### Weaknesses
The list below roughly follows the order of appearance in the manuscript. In my opinion W1, W2, W4, and W5 are the most important among these.

### W1: The functional form / actual mechanics of the normality tests should be added as an appendix section
For readers less familiar with the actual tests employed, this would be very helpful. For example, I was only aware of the Kolmogorov-Smirnov test of univariate distributions and am not sure how the multivariate version is defined. I would definitely recommend adding these details as an appendix section to aid clarity of exposition.

### W2: Figure 3 is hard to interpret
I found this figure hard to digest. Specifically, are the images from columns 2 and 3 generated from the same latent vector but with different seeds? 

### W3: “Correlation” between acceptance of an inversion and reconstruction quality should be quantified.
In the second paragraph of Section 3.2, the authors state:
> Figure 4 demonstrate a strong positive correlation between acceptance of an inversion (green)
$\mathbf{x}^{(*)}$ and its reconstruction quality,

This claim should be quantified if possible.

### W4: Figure 5 is also difficult to interpret
I found the bottom of Figure 5 also difficult to interpret. The relationship between Q-align score and acceptance under the normality tests does not appear to be clear cut and the confidence intervals seem quite wide.

### W5: The evidence in Section 3.2 seems anecdotal
Specifically the claims made in the third paragraph of this section seem to only be supported by the few examples in the grid of Figure 5. If there were some way to better quantify / make the evidence for these claims more robust, that would make the claim about the efficacy of the normality tests much stronger in my opinion.

### W6: Minor formatting issues
- Line 153: The “Shoemake (1985)” citation should be `\citep`
- Line 158: The “Song (2020a;b)” citations should be `\citep`
- Line 245: The citation for Kolmogorov-Smirnov seems to be malformed: “An 1933” and should also be `\citep`

### W7: Formatting suggestions
List below is not weaknesses per se, just some ideas I had about helping to make some of the figures easier to digest. Feel free to implement or ignore these: 
- In Figure 2, and other similar ones (e.g. Figure 6), perhaps place a red / some-colored border around the image in the first column that serves as the “anchor”/origin image and place a similar box around the corresponding image in the grid. Consider labeling the directions in the grid as well.
- In the paragraph for section 3.1, at the end of the paragraph, I would recommend bolding the normality tests that the authors found to be most effective.
- In Figure 5, it might be useful to also add the Q-align score as colored text to each image

### Questions
### Q1: In Figure 4, why is inversion with a higher budget worse than 1 step?
Columns with budget 2-32 appear to be worse than a single step. Why is this the case?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents Combination of Gaussian variables. It presents the theory behind the method and multiple experiments showing the effectiveness of the proposed method.

### Strengths
The method is simple and easy to understand.

### Weaknesses
I think the examples in the paper are not convincing that the proposed method is much better than the baselines. Some numbers indicate this but there is no human evaluation in the results. I think that better examples would have made the paper stronger.

### Questions
- In Figure 5, it is hard to tell if an interpolation succeeded or failed. I can't really make any clear decision by looking at the images. It would maybe be better to show interpolation between things which are more intuitive such as human faces or animals.
- Would it be possible to have a larger study of the correlation between the p-value and the Q-align values. For example, one can sample many examples and invert them using different budgets, then interpolate between pairs and check visual quality vs p-value, 
- In addition, some human evaluation would be good instead of the Q-Align metric.
- I think that for table 1 it would be good to have a MOS evaluation of the different methods, FID is not a good enough metric.
- In Figure 15, all interpolation methods seem to perform equally well. This makes me think that the proposed method is incrementally better at most. This is also the case in table 3.
- What would happen if figure 6 for a different interpolation method? Would it fail?
- There is no limitation section in the paper.
- In figure 14 it seems the CoG generates an image which is remarkably close to image 2 rather than an interpolation of the two images. Could it be that the generated images are just one of the two options image1 and image2? It seems the interpolation is not really an interpolation at all.
- Generally, I think most examples in the paper are not convincing and that examples that demonstrate the method better should have been chosen. For example, an interpolation between a dog and a cat is similar posture, or between two faces, or the same face with a different facial expression.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Generative models like diffusion/flow models can map data to Gaussian latent spaces, allowing additional manipulations. However, typical interpolation methods, like linear or spherical interpolation, often fail because they do not fully respect Gaussian distributional properties, leading to unrealistic generations. Current alternatives are either computationally costly or specific to certain models. This paper proposes Combination of Gaussian variables (COG), a simple, generalizable method that maintains Gaussian characteristics during interpolation and linear transformations, enabling effective latent manipulation across models and data types.

### Strengths
1. The core argument of the paper, "the norm is not a sufficient statistic to look at when determining whether the latent is good for generations", is intuitive and sound. Especially Figure 3 presents a convincing argument.
2. They paper has many illustrations of the different interpolation methods results. In particular, the ones of low-dimensional subspaces are very nice.

### Weaknesses
1. The contribution of this paper is limited. Norm being not enough is known, and the proposed interpolation method, under the setting of diffusion/flow model, is just rescaling the samples norm: $\alpha=0$ and we rescale the sample by the expected norm gain $\sqrt{\beta}$.
2. The normality testing part feels out of place and under-explored. I assume it is there to justify that the norm is not enough, and we need broad characteristics. In that case, Figure 3 alone is enough. Then the paper never specifies what broad characteristics we should/could look at or optimize. Lastly but more importantly, the exploration done in Section 3 never went anywhere: as listed in point 1, the proposed method also does not look at the broad characteristics (essentially still just the norm since $y$ to $z$ is affine), and is not connected to the normality testing in any way.
3. The actual advantages of the method are questionable. For example, there are no clear differences shown in Figure 1, compared to the (standardized/mode norm) Euclidean results. Also, I am wondering why COG is faster than Euclidean methods in Table 1?
4. The writing of this paper can be improved. For example: (1) the writing on normality testing part is not clear. How exactly are you doing the normality test, as it requires a set of data points? (2) what setting is Table 1 for, SD or ImageNet Inversion, since there is only one FID per task?

### Questions
1. I am confused by Figures 4 and 5. Is the 1-step inversion image correct? I expect an image of worse quality than a 2-step inversion following the trend. On this note, could you explain how you are doing the inversion, like one-step inversion and multi-step generation? Essentially, I am curious as to why a 1-step inversion reconstruction is this good, and the quality dips first and then becomes better as you increase the budget.
2. The claim made in Section 3.2, "stress some latents are rejected that lead to good reconstructions...", needs to be backed up, in addition to the illustration in Figure 5. There should be more larger-scale quantitative evaluations. Even Figure 5 itself is not very convincing: why is the snake interpolation better than the clock one?
2. Some Figures are not very readable. For example, the colored numbers in Figure 5, and the green crosses and red dots on the bottom row of Figures 4 and 5.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposed a novel method of linearly combining Gaussian latents of pretrained generative models (in the paper Diffusion model and flow matching model) that are simple, and shown in to be effective in linear interpolation and centroid determination tasks. The methodology is based on observation that latent samples from priors of pretrained generative models can lack Gaussian characteristics, which is the main reason for failure in the image reconstruction task.

### Strengths
- Well-motivated problem, the proposed method is novel, reasonable, and most importantly, relatively simple, yet seems to performing quite well in (certain) empirical benchmarks.

### Weaknesses
- The writing of the paper can be improved. Some parts are ambiguously written. For example, the notion of latent space $x_T$ can be unified for both flow matching and diffusion at time $T$, instead of writing $x(0)$ and $x_T$ to avoid confusion. The important observation, starting at line 206: "Having a latent vector with a norm that is likely for a sample from..." is extremely unclear. In fact, the sentence does not make sense. 
- I have a big question mark about the usage of the univariate testing method in Section 3 (stated in 3.1 in between line 250-254). Why don't the authors use a multivariate hypothesis testing method that accounts for the correlation between features/pixels. This makes more sense statistically. 
- The authors mentioned that "In this work we will propose a simple technique that guarantees that interpolations follow the latent distribution given that the original (seed) latents do, and which lets us go beyond interpolation". However, I failed to find a Lemma/Theorem that stated such a theoretical guarantee of their method. 
- Empirical results seem questionable: the authors claimed to follow closely benchmark protocols from NAO paper of Samuel et al. (2023) (in fact the year of publication for this citation in the paper was stated wrongly -- please fix). More specifically, the results from Table 1 of the paper is very different from Table 1 in Samuel et al. (2023) -- the two tables are supposed to report results from the exact same experiment.

Dvir Samuel, Rami Ben-Ari, Nir Darshan, Haggai Maron, and Gal Chechik. Norm-guided latent space exploration for text-to-image generation. Advances in Neural Information Processing Systems, 37, 2023.

### Questions
- Can COG perform rare-concept generation experiment similar to Samuel et al. (2023), section 5.1? I find the current empirical benchmarks that make comparisons with existing baselines quite lackluster.

### Soundness
2

### Presentation
2

### Contribution
2
