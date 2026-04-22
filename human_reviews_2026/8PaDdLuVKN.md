# There and Back Again: On the relation between Noise and Image Inversions in Diffusion Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Diffusion Models achieve state-of-the-art performance in generating new samples but lack a low-dimensional latent space that encodes the data into editable features. Inversion-based methods address this by reversing the denoising trajectory, transferring images to their approximated starting noise. In this work, we thoroughly analyze this procedure and focus on the relation between the initial noise, the generated samples, and their corresponding latent encodings obtained through the DDIM inversion. First, we show that latents exhibit structural patterns in the form of less diverse noise predicted for smooth image areas (e.g., plain sky). Through a series of analyses, we trace this issue to the first inversion steps, which fail to provide accurate and diverse noise. Consequently, the DDIM inversion space is notably less manipulative than the original noise. We show that prior inversion methods do not fully resolve this issue, but our simple fix, where we replace the first DDIM Inversion steps with a forward diffusion process, successfully decorrelates latent encodings and enables higher quality editions and interpolations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper analyzes DDIM inversion, specifically the non-gaussian properties of the final latent $\mathbf{\hat{x}}_T$ which has been mentioned multiple times in previous literature but not explored throughoutly like the authors have done in this work. The authors show that the inverted latents quantitatively exhibit spatial correlations with the original image, especially in smooth (plain) iage regions, attributing it to the approximation errors in the early inversion steps. They propose a simple fix of replacing the first few steps with the original forward diffusion process. Their fix restores the gaussian properties of the final inverted latents, improving editability and text alignment when using DDIM inversion.

Overall, although this work has limited applicability (see weaknesses), this work is technically solid with extensive experiments that dive deep into the emprical findings that have been brushed off in previous works. Limited applicability do not outweigh its contributions in the form of empirical analyses.

### Strengths
- The main strenghts of this paper is in its arrangement and analysis of previous work. The authors go through great lengths in analyzing the claims of previous work that have mentioned this non-gaussian property of the final latent in DDIM inversion, and structures it well with their own quantitative experiments, corroborating past claims with evidence.
- Their quantitative experiments bring a clear insight that the early inversion steps were the root cause of the issue, emphasized more in what the authors call smooth (plain) image regions. Their proposed fix is simple and easy to implement, with convincing quantitative/qualitative improvements compared to the vanilla method of simply changing the target prompt starting from $\mathbf{\hat{x}}_T$.
- The experiments are wide breadth-wise, with multiple ablations that quantitatively show improvements.

### Weaknesses
Weaknesses:
- The main weakness of this work that I've found is the limited theoretical grounding of their explanation on why the early inversion steps causes correlation with the original image. Their explanation for why the early steps cause correlation is mostly empirical, and the connection they make to the ODE curvature could be formalized further. In order to prove why the proposed simple fix works should be first followed why the early inversion steps are theoretically the root cause. I do not see the simple, heuristic fix the authors propose as a weakness, just that the cause should be diven into deeper.
- The scope of their fix, especially in terms of image editing, is narrow as they demonstrate it on solely DDIM inversion. To the extent of my knowledge, most image editing tasks with text prompt inversion methods do not rely on plain DDIM inversion, but rather "image-to-noise inversion techniques" as described in section 2. Some of these inversion techniques might not be directly applicable to the proposed method, but when they are, I believe the proposed method should be demonstrated on them.

Comments:
- Though not strictly a weakness, It would be better if how the plain background or surfaces are mathematically calculated was included in the main text (even just a breif 2-3 liner) instead of fully allocating it to appendix I.

### Questions
- in appendix G, the authors mention: "Classifier-Free Guidance introduces additional errors to the DDIM inversion, so to focus solely on the inversion approximation error, we disable CFG by setting the guidance scale to w = 1. To the extent of my knowledge, this is the main reason multiple "image-to-noise inversion techniques" have been explored in literature. Although setting w > 1 does drastically alter the given image, I wonder if applying the proposed fix also provide a meaningful improvement in terms of fidelity or text alignment.

### Soundness
3

### Presentation
3

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
This paper sheds light on a known phenomena—the discrepancy between a real Gaussian noise and the inversion noise obtained through DDIM inversion—and provides a simple fix to mitigate it. When performing DDIM inversion on an image, the resulting noise is generally not Gaussian. The authors point out that this is particularly true in regions of the image that are flat (i.e. textureless). Through a set of experiments, they pinpoint the reason of this discrepancy to lie in the very first few steps of inversion. To solve this, the paper proposes to replace the first few steps of inversion by a forward noising step. This is generally done for only ~4% of the denoising path. 

Experiments ablate the threshold to which one should do forward noising, and validate that this fix indeed improves inversion quality for editing tasks.

### Strengths
The paper provides a structured analysis of the problem, which can be valuable to the community. While many of its conclusions are already well-established within the field (DDIM inversion is not a Gaussian noise etc.), the experiments presented in the methods section offer tangible evidence that reaffirms these claims. Furthermore, the paper proposes a simple solution that appears to effectively mitigate the identified issues. Overall, the presentation is clear and easy to follow.

### Weaknesses
My main concern with the paper lies in its contributions. The method is organized into three subsections, in which the authors address the following questions:

- Sec 3.1: Are there any difference between the original noise and the DDIM inversion noise? (Answer: Yes)
- Sec 3.2: How does it differ? (Answer: Loss of variance mostly in plain regions)
- Sec 3.3: Why? (Answer: Because it happens in the first few steps)

In my opinion, the first two sections primarily reiterate observations and facts that are already well known to the community, albeit supplemented with additional visualizations and plots that support these claims. The authors themselves mention prior work studying the same problem at the beginning of Sec 3.1. The third section has the potential to be more interesting; however, its conclusion appears to be based mainly on empirical observation, and the authors do not provide a convincing explanation for the observed phenomenon, which would have been the most compelling aspect of the analysis. Lastly, the proposed fix, while somewhat effective, appears ad hoc and lacks a principled justification.

### Questions
**DDIM Inversion Formula**

There appear to be different formulations of DDIM inversion in the literature. On one hand, Null-text inversion (https://arxiv.org/pdf/2211.09794, Sec 3.1) and Dhariwal & Nichol (https://arxiv.org/abs/2102.09672) both call the denoising network on $x_{t-1}$ with timestep $t-1$, i.e. $\epsilon^{(t)}\_{\theta}(x_{t}, c) \simeq \epsilon^{(t-1)}\_{\theta}(x_{t-1}, c)$. On the other hand, the authors in Eq. (3) seem to follow ReNoise (https://arxiv.org/pdf/2403.14602, Sec 3.1) and keep the same timestep but change $x_{t}$ to $x_{t-1}$, i.e. $\epsilon^{(t)}\_{\theta}(x_{t}, c) \simeq \epsilon^{(t)}\_{\theta}(x_{t-1}, c)$. In my understanding, the former formulation (calling the network on $x_{t-1}$ with its corresponding timestep $t-1$) represents the correct reverse ODE. This leads to the following questions:

- Which version exactly did the authors implement in the experiments?
- Have the authors noted any difference in the conclusions using the other version? 
- In particular, this choice changes the definition of the approximation error $\xi(t)$ in Eq. (4). How does that change the results and conclusions?

Note that in Appendix E, the authors use the latter formulation (non-matching timesteps) in the text, but the illustration in Fig. 8 seems to show the former formulation (matching timesteps). I believe this could be further clarified to improve the reproducibility of the experiments.

**Image Editing and Reconstruction**

Overall, I am not very convinced by the validation provided to measure the quality of image editing applications.

- In Fig. 7, the explanation suggests the method is superior to standard DDIM inversion because it performs similarly to original Gaussian noise in both image diversity and prompt alignment. However, this comparison may not fully validate the editing task. While using real Gaussian noise $x_T$ with a target prompt will expectedly yield a high-quality, prompt-aligned image, it offers no guarantee of fidelity to the source image. Successful editing requires preserving significant portions of the original image while making targeted modifications. Therefore, performance closer to real Gaussian noise does not, by itself, validate superior editing performance.
- A similar remark applies to Table 7 and Sec 4.3: CLIP text alignment alone is insufficient to assess image editing quality, as it does not measure the preservation of unedited regions. I would suggest considering metrics specifically designed for editing, such as **directional CLIP similarity** (StyleGAN-NADA, https://arxiv.org/abs/2108.00946) or **AugCLIP** (https://arxiv.org/abs/2410.11374), for a more comprehensive assessment.
- Image editing performance often differs between generated and real images. I may have missed it, but are there any editing results with real images? If not, adding a few such examples would significantly strengthen the assessment of the method's practical utility.

**Result with Large Latent Diffusion Models**

Large text-to-image diffusion models like Stable Diffusion are very common in the field, yet not fully assessed in this paper.

- Considering the widespread use of Stable Diffusion models (SD2.1, SDXL, SD3), I was wondering why the authors did not include analysis with these models in the first part of the paper (beyond the results with SDXL in Sec. 4.3)?
- Did the authors observe any correlation between the quality of standard DDIM inversion and the size of the model's training dataset? One might expect that models trained on larger datasets are less prone to inversion errors, and that the proposed fix would have less impact in these cases. I would be happy to get the authors' perspective on this.

**Other questions and notes**

- What model was used in Figure 1, Figure 6 and Figure 13?
- The CFG scale can significantly impact experiment outcomes. The authors mention in Appendix G that they use a CFG scale $w=1$ for conditional diffusion models. Does this apply only to the image editing part, or to all experiments in the first part of the paper as well?
- In Sec 4.3, I believe it could be more explicitly mentioned that the results in Table 7 are based on a comparison with an equal number of function evaluations (NFEs).
- Appendix H shows interesting differences in the shape of the most probable triangles between models. Do the authors have an explanation for what might be causing these vastly different triangle shapes?
- In Fig. 3, perhaps a different color scheme or the addition of level set lines could help better visualize the intended point. If I understand correctly, the bottom-left corner (distance between $x_T$ and $x_T$) should be 0. However, its color appears similar to the top-right corner, which would suggest that the image latent $x_0$ is almost the same as the inverted noise $\hat{x}_T$. This seems counter-intuitive, and a clarification would be helpful.
- Appendix N shows an interesting experiment using the $L_2$ distance. In general, it is understood that the noise is not necessarily close to the image in an $L_2$ sense, but rather that they are correlated. Out of simple curiosity, does the mapping become feasible if a Pearson correlation metric is used instead?
- Lastly, I noted a minor citation issue: the reference to Samuel et al. appears to be missing the year, showing up as "Samuel et al." in the text instead of "Samuel et al. 2025".

I thank the authors for their time and look forward to any clarifications they can provide.

### Soundness
2

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
4

### Summary
The authors examine DDIM inversion in diffusion models, a process of recovering an approximate noise map that reconstructs a user provided image. First, they study the statistical relation between pure normal gaussian noise, the corresponding sampled image, and the noise obtained via inversion of that same image. They show that the inverted noise maps differ from the standard gaussian distribution, having especially high neighboring pixel correlations in areas where the input image is smooth. Secondly, they claim that these statistical deviations lead to higher inversion error, and worse performance when used for of text-based editing and image interpolation. Finally, the authors claim that this error stems from low accuracy in the initial inversion timesteps and propose a simple fix of using random added noise instead, which improves the inverted noise statistics and its applicability to interpolation and editing.

### Strengths
1.	The authors provide a nice observation regarding the statistics of inverted noise latents compared to the native noise space, and the repercussions regarding the usage of these latents for image manipulation and editing.
2.	The authors provide a simple remedy to fix the inverted latent statistics, and as such their downstream editing potential, with minimal cost of reconstruction quality.
3.	The authors provide thorough quantitative evaluation of the tasks presented (image interpolation and text-based image editing).

### Weaknesses
1.	Adding random gaussian noise instead of inversion in the first steps. The authors claim that the last steps are not important for reconstruction quality, but the diffusion process acts as a coarse-to-fine spectral regressor throughout the whole process [1]. This means that the last steps should correspond to fine-grained details, such as textures. This can be observed, for example, in the results in figure 6 (top right) where the tower in the background is generated with small windows,  which did not exist in the original image.  An ablation on the effect of the added noise for different amounts of timesteps is recommended.

2.	Hard to qualitatively evaluate the results for inversion reconstruction and text-based image editing. Only a few results are present in the main paper, and the results in the supplementary section are highly compressed, while no further image files were provided in the zip. I would recommend adding more results to the paper, especially for the editing experiments, since current metrics cannot fully replace human assessment.

[1] Rissanen et al. 2023. Generative Modelling With Inverse Heat Dissipation

### Questions
Please see weakness section. Addressing the raised concerns could affect my final rating for the paper.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a systematic analysis of DDIM inversion in diffusion models, focusing on the discrepancy between the original Gaussian noise and the latent representation of DDIM inversion. The authors identify that inversion errors primarily accumulate in early steps and are particularly pronounced in plain image regions. Based on this observation, the paper proposes a simple method that replaces the first inversion steps with forward diffusion noise.

### Strengths
- The paper provides a thorough and insightful diagnosis of the underlying causes of latent distortions introduced during inversion.
- The experiments are conducted across a diverse set of diffusion models, supporting the generality of the findings.
- The proposed forward-diffusion replacement is conceptually simple, easy to integrate, and empirically improves both interpolation smoothness and the diversity of editing outcomes.

### Weaknesses
- The analysis focuses on diffusion-based models and does not provide evidence that similar problems occur in flow-matching models (e.g., FLUX, Stable Diffusion 3)
- The paper evaluates mainly interpolation and text-guided editing, but does not explore other inversion-driven applications (e.g., local editing, style transfer).

### Questions
- Have the authors tested whether similar inversion-induced correlation patterns appear in flow-matching models such as FLUX or Stable Diffusion 3?
- Can the proposed method enhance performance when combined with various editing engines such as MasaCtrl or Pix2Pix-Zero? Demonstrating improvements across multiple editing scenarios would further strengthen the paper.
- Tables 4 and 7 suggest that increasing the proportion of forward diffusion replacement decreases reconstruction fidelity, while improving editability. This appears to reflect an inherent trade-off between preserving the original image information and increasing the degree of manipulation. It would be valuable to analyze or formalize this trade-off explicitly.

### Soundness
4

### Presentation
4

### Contribution
3
