# Two-Stage Diffusion Models: Better Image Synthesis by Explicitly Modeling Semantics

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3

## Abstract
Recent progress with conditional image diffusion models has been stunning, and this holds true whether we are speaking about models conditioned on a text description, a scene layout, or a sketch. Unconditional image diffusion models are also improving but lag behind, as do diffusion models which are conditioned on lower-dimensional features like class labels. We advocate for a simple method that leverages this phenomenon for better unconditional generative modeling. In particular, we suggest a two-stage sampling procedure. In the first stage we sample an embedding describing the semantic content of the image. In the second stage we use a conditional image diffusion model to sample the image conditioned on this embedding, and then discard the embedding. The combined model can therefore leverage the power of conditional diffusion models on the unconditional generation task, achieving large improvements in unconditional image generation. The same method can be generalized to yield similar improvements for image generation conditioned on a low-dimensional signal like a class label.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper revisits the unCLIP paradigm proposed by Ramesh et al., 2022, which consists of two cascaded diffusion models, one trained on CLIP latent text embeddings and another one mapping from latent text embeddings to the image space. Unlike unCLIP, the paper proposes to train the latent diffusion models on CLIP image embeddings rather than text embeddings, which enables unconditional image generation. The proposed model is evaluated on different variants of AFHQ, FFHQ, and ImageNet, and is compared to EDM (Karras et al. 2022) among other baselines.

### Strengths
Improving unconditional image generation models is an active research area, and lags substantially behind class/text conditional generation. Making progress in this area is important. Further, techniques relying on multi-stage modeling/latent modeling like the proposed one have proven effective in making diffusion models more efficient.

### Weaknesses
I see two main weaknesses: 
1. The lack of novelty. The proposed method is very similar to unCLIP. 
2. The method is arguably more complicated than (Hu et al., 2022) which simply clusters image embeddings obtained by a self-supervised representation and uses the cluster indices as conditioning signal. While the paper compares to this approach on AFHQ/FFHQ, I’m not fully convinced that the proposed method is superior. I would expect a comparison on ImageNet to be convinced that the additional complexity of the proposed approach is justified, since Hu et al. get similar improvements.

Given these two points, I’m leaning towards rejecting this paper.


Minor points:
- Typo page 2 bottom “that the all images”
- I found the terms lightly/strongly conditional somewhat confusing. Maybe it would be simpler to just use class/text conditional?

### Questions
- Do the authors have any explanation why the 2SDM outperforms 2SDM with oracle in Figure 5 right?
- Did the authors consider any other image embeddings besides CLIP? For example DINO might be better aligned with ImageNet. Also it would be interesting to see how well the first diffusion model can learn the embedding, and how this affects the quality of the end-to-end model.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes an unconditional image generation pipeline which is split into two parts: first, a model generates a random condition (in this case a CLIP image embedding), then, a second model is conditioned on this condition to generate the actual image. Compared to baseline unconditional single-stage models, the new approach performs better while leading only to a small overhead in training and sampling cost.

### Strengths
The approach tackles unconditional image generation, and improvements in that area could potentially also translate to conditional generation pipelines.

The approach of splitting the unconditional generation into two parts seems novel and the results indicate that this does indeed lead to improvements, at least on the relatively small datasets and image resolutions that it was tested on.

### Weaknesses
While the approach seems to lead to improved performance it's not clear to me why this is the case and there is only very little analysis around this.
Is it that unconditional sampling of CLIP image embeddings is somehow important or easier than sampling an image directly? Or is it the two-stage pipeline itself that is the important part? Could the condition generation and subsequent image generation be done in a single pipeline with end-to-end training? What exactly is the interaction between the first and second stage models?

### Questions
How well do you think this would work for more complicated domains and datasets?
How do you think this approach could benefit/improve conditional generation pipelines such as text-to-image?
How well do you think this would work with more specific conditions in the first stage (e.g., depth maps, edge maps, etc)?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a two-stage approach, 2SDM, for sampling from diffusion models. The goal is to improve the performance of unconditional generation, which has a gap in performance compared to conditional generation. In the first stage, an auxiliary diffusion model is used to generate an embedding, which is subsequently used in the second stage by a conditional diffusion model to synthesize an image. The authors demonstrate that 2SDM yields better performance across almost all experiments in terms of quality and diversity with little to no increase in sampling speed.

### Strengths
- The proposed two-stage approach is straightforward, extending UnCLIP to the unconditional setting.
- The authors provide some context and insight into what it means for conditional generation to be better than unconditional generation, and why this may be the case.
- Experiments demonstrate superior performance over the baselines with negligible impact on the sampling speed.

### Weaknesses
While the method is straightforward, the paper is a bit difficult to understand overall. The finer details are unclear. For example:
- It is unclear what the authors mean when they mention "discarding" the conditional embedding y after sampling.
- The details about the auxiliary model in Section 4 are unclear. For example, what is a_\sigma? Maybe reiterating some of the variables in Equation 4 would be helpful, too.
- In the results overview of Section 5, the authors describe that Figure 4 (which seems to actually be referring to Figure 5) "compares against 'Class-cond', which is an ablation of 2SDM that applies to unconditional tasks". Given the label "Class-cond" it seems more intuitive that this would refer to the "lightly-conditional" task instead.

### Questions
- For explicit clarification, are the two models (auxiliary and conditional image) trained sequentially?
- The authors mention that they did not use classifier-free guidance in their results, which is common practice for diffusion sampling. It would be helpful to get some sense of how it affects the quality of the outputs.
- The experimental results are compelling and the method is straightforward, but the paper could greatly benefit from clearer communication of the proposed ideas and details.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
