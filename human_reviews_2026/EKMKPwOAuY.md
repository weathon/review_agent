# Atrous Learning for Diffusion Models

- Decision: Reject
- Scores: 2, 2, 4

## Abstract
Diffusion models have shown remarkable success across a wide range of generative tasks. However, they often suffer from spatially inconsistent generation, arguably due to the inherent locality of their denoising mechanisms. For example, a diffusion model trained on natural images might generate hands with six fingers. To mitigate this issue, we propose atrous learning for diffusion models, a simple yet effective masking strategy that can be implemented with only a few lines of code. Experiments show that it is surprisingly safe to mask up to 98\% of pixels for diffusion model training. Our method attains competitive FID scores across datasets and avoids training instability on small datasets. Moreover, the masking strategy reduces memorization and promotes the use of broader contextual information during generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose "atrous learning for diffusion models", where they introduce masking in the denoising objective in an effort to avoid "spatial inconsistencies". They claim that their approach mitigates undesirable effects of locality biases that are inherent in diffusion models. Experiments are conducted on synthetic data and standard image datasets, where improved performance is observed compared to baselines.

### Strengths
1. The paper is easy to read and the overall methodology is simple and clear.
2. Along with the experiments, the authors offer some theoretical discussion and attempt to connect their work with existing literature.
3. The authors have taken care to include details such as masking ratio ablations and performance as a function of the training epochs.
4. I found the angle explored in Section 5.4, regarding memorization, particularly interesting. I encourage the authors to explore this further.

### Weaknesses
1. While the methodology is clear to me, I am not convinced about the motivation of this work. The connection with prior work on Ω-locality is, in my opinion, loose. I include specific questions relating to this point in the Questions section below.

2. The demonstrations are not convincing. The experiment in Figure 3 is not particularly informative as I struggle to see differences in the gradient heatmaps. I fail to see any correlation between the mask ratio and the standard metrics reported in Tables 1, 2 and it is not clear whether the reported improvements are statistically significant. For the experiment in Figure 6, none of the the samples appear sufficiently close to the training data and the proposed method's effect is therefore unclear.

3. The paper advertises the proposed method as consistently outperforming baselines in the abstract. I feel that this framing is not appropriate as the reported improvements are questionable. Larger scale (e.g. ImageNet) and more thorough benchmarking with competitors would be required to properly validate such claims. The angle explored in Section 5.4, i.e., mitigating memorization, is, in my opinion, a better fit for this work (e.g., see [1] for similar analysis on language models).

### Questions
1. As I understand, the binary masks do not impose any locality constraints, i.e., they are iid. How is it possible then that SMD mitigates locality bias? Increased variance, as mentioned in Proposition 1, is not sufficient to force the networks to explore non-local structures. For example, one can also decrease the training batch size to achieve higher variance but there is no expectation that this mitigates locality.

2. I fail to see any evidence for the claims made in Section 6 (starting at line 471). For example, how might SMD promote "large-scale structure first and filling in finer detail later"?

3. Beyond the samples of Figure 2, could the authors provide more examples of locality bias and how this might be mitigated by SMD? At present, I am not convinced with the visualization on this toy dataset.

[Minor] Figure 1: Supervised Singals -> Supervised Signals

[1] Be like a Goldfish, Don't Memorize! Mitigating Memorization in Generative LLMs, NeurIPS 2024

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors point out that recent work on diffusion/flow models where it has been shown that for a given coordinate in the output space, information from a local neighborhood is mainly used to score or predict at the coordinate. They hypothesize that this *locality* of the diffusion process harms the model performance and that more global representations can be learned. To encourage such behaviour, they propose masking the training data such that the models are forced to consider more spread out information. Qualitative and quantitative analysis is done on outputs from models trained with the masked approach (SMD), suggesting that masking does in fact help.

### Strengths
The paper is well motivated, and the theory is mostly easy to follow. The idea that encouraging more global use of information is promissing (and you could maybe have referenced other work encouraging such behaviour, e.g. dino). The flow is good and the theoretical analysis of the SMD gradient is nice.

### Weaknesses
The main weakness of the paper is the empirical validation. The datasets used are small and have low resolution, qualitative analysis needs to be more thorough, and quantitative analysis is lacking. The FID table is alright (50k images is good!) but still only compares a single model per setting, right?

More importantly, the main hypothesis that global information is not being used without masking is also not tested well enough. One could imagine masking having a different role, such as being a regularizer.

Finally, there are no statistics or error bars, for this to be a convincing study you need to train multiple models. Pointwise evaluations of FID scores are not enough. This is also seen in figure 4 where the curves are not exactly smooth. Or in table 2 where it is not clear whether the fluctuations in the numbers are due to noise.

Because of this, I can't recommend acceptance (empirical evidence for the local/global claim and some statistics are needed, happy to reconsider if shown some)


More comments:

I found figure 1 slightly confusing, i recommend explaining the variables and labels in the caption (also a typo in "Singals"). It is not clear how the multiple masks for SMD are used (the PSPC dynamics seem somewhat illustrated with the colors, maybe something similar can be done).

In section 5.1 on spatial consistency, it would be nice to see some kind of quantitative evaluation rather than just 9 samples from each setup.

In section 5.2. on contextual representations, the single outputs are nice but again, some kind of quantitative result would be nice, and more examples.

Figure 3 needs much larger text, and the plots are barely different to the untrained eye, especially for the same timesteps, maybe just crop around the center?

Figre 4 and 5 are also way too small. and the qualitative results unclear.

### Questions
I am somwhat confused by the statement in section 5.2 that says that the same noise provided to different models leads to two different images. Why shouldn't it? (the gradient difference could be visualized/quantified better)

Can you think of a concrete test for whether a model has learned to use global information instead of local information to predict?

How exactly are the recall and precision in Table 2 defined/calculated?

Small nit, why do you have different masking ratios in 4a than in 4b and 4c?

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
This submission proposes a new diffusion model to mitigate spatial inconsistency caused by the models' locality.
The proposed method utilizes atrous (dilated) convolution strategy combined with Simplified Masked Diffusion (SMD) training.
Compared with the original diffusion models, SMD provided training gradients with higher variance without bias.
Experiments are conducted using some toy data and CIFAR10 32×32, CelebA-50K 64×64, and LSUN Bedroom 32×32. Improved FIDs and precision and recall of generated images are reported.

### Strengths
- Masking strategy is . Deeply investigating the effect of masking in convolutional diffusion models is a meaningful direction of research. 
- Mitigating locality dependency seems a promising direction, especially non-attentive convolutional diffusion models.

### Weaknesses
- The organization/technical clarity of the paper is not excellent. "Atrous learning" is featured in the title and Introduction, but the relationship between atrous learning and SMD is not clear. Details of the network architecture and other training parameters are missing.

- Omega locality is defined in Sec 4.2 using a certain amount of text but is not used in the following theoretical and empirical analyses, which causes a feeling of somewhat shallow discussion.

- Experiments are limited in the small-scale datasets, and the generality of the method for larger scales are unclear.

### Questions
- Which network architecture was used and how atrous convolution was incorporated? Or atrous convolution is not used but the masked training is termed "à trou" (holed)?

### Soundness
3

### Presentation
3

### Contribution
2
