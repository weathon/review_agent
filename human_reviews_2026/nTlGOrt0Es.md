# Equivariant Latent Alignment via Flow Matching under Group Symmetries

- Decision: Reject
- Scores: 8, 6, 6, 4

## Abstract
Geometry-aware generative models and novel view synthesis approaches have shown strong potential to improve visual fidelity and consistency. In parallel, equivariant representation learning has emerged as a powerful framework for constructing latent spaces where analytically known group transformations could act directly, capturing geometric structure in data and enhancing both interpretability and generalization.
However, we identify that existing approaches often suffer from \textit{latent misalignment}, a discrepancy between the intended group action and the actual required transformations in latent space, as the learned latents fail to consistently preserve the equivariant relations imposed by the underlying group symmetry. This misalignment degrades view synthesis quality and undermines the theoretical guarantees of equivariant representation learning. 
To address this issue, we introduce \textbf{Residual Latent Flow}, a flow-matching-based correction framework that corrects the misaligned latents, thereby improving compliance with the underlying equivariance relation.
We show experiments that flow-based correction significantly reduces latent misalignment and improves novel view synthesis quality, under orthogonal group $\mathrm{SO}(n)$, using synthetic image datasets with rotational freedom. Our method demonstrates the efficacy of combining flow-based correction with equivariant representation learning, resulting in a new powerful framework for learning a more consistent and accurate group symmetry-aware models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Residual Latent Flow to correct latent misalignment in equivariant representation learning. The flow-matching approach enhances latent alignment and novel view synthesis while preserving group structure across SO(2) and SO(3) symmetries.

### Strengths
- combines flow matching with equivariant representation learning to address a previously underexplored problem.
- well-structured, with clear motivations, derivations, and visualizations.
- offers a practical and theoretically grounded solution to a fundamental limitation in ERL, with potential impact on generative modeling and view synthesis.
- extensive experiments across multiple datasets, metrics, and model sizes.

### Weaknesses
- the method assumes access to a pre-trained equivariant encoder and decoder, which may limit its applicability if such models are not available or poorly trained.
- the flow model adds computational overhead, though the paper shows that even a smaller model brings improvements.
- while the method improves alignment, the final synthesis quality is still bounded by the decoder’s capacity.

### Questions
- how does the method scale to higher-dimensional or non-compact groups beyond SO(2) and SO(3)?
- could the flow correction be integrated into the encoder training loop, rather than applied post-hoc?
- the paper uses a linear interpolation path. have you experimented with stochastic paths, and if so, how did they affect performance?
- the decoder is fine-tuned on flow-corrected latents. did you observe any overfitting or degradation when fine-tuning was omitted?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies latent misalignment in equivariant representation learning(ERL). It means a persistent gap between the analytic group action expected in the latent space and the encoder output for the transformed input. The gap grows with larger rotations and harms geometric consistency and novel view synthesis quality. The authors propose Residual Latent Flow, a conditional flow matching correction that uses the same object and the same group element to transport an analytic latent to its paired target latent along a zero variance linear path while learning a velocity field that points from the source to the target. Integrating this field at inference maps the analytic latent to the correct target and keeps the intended group structure, which is implemented with an NFT based block diagonal parameterization of irreducible components. On synthetic benchmarks with planar and three dimensional rotations, the method improves image reconstruction metrics and reduces a latent angle error derived from the first order component, producing more faithful and symmetry aware novel view synthesis.

### Strengths
The paper clearly defines latent misalignment in equivariant representation learning and ties it to drops in geometric consistency and synthesis fidelity, then motivates a solution by identifying the conflict between the equivariance loss and the decoder loss. Built on Neural Fourier Transform, it uses a block diagonal, interpretable decomposition into irreducible components for fine grained diagnosis. The proposed Residual Latent Flow is a simple yet principled correction that conditions source and target on the same object and transform, learns a residual transport in latent space, and preserves group structure. Experiments on SO(2) and SO(3) show consistent reductions in misalignment and gains in reconstruction and novel view synthesis, indicating the method complements rather than replaces ERL.

### Weaknesses
From the method as written, I point out several potential weaknesses.
1. The reliance on a linear, zero-noise interpolation path may break under large rotations or strong lighting and occlusion shifts; without comparisons to curved or noise-injected paths, I remain unsure about error accumulation and robustness.
2. The method assumes paired supervision with the same object and the same transform, yet real data often has label noise or imperfect pairings; without mitigation strategies, I worry the learned correction becomes biased and brittle.
3. The evaluation centers on a degree-one block angle error, which can miss higher-order misalignment and bake in encoder bias; without multi-block or encoder-agnostic metrics, I doubt the coverage of the evidence.
4. As latent dimensionality and group complexity grow, compute, memory, and optimization stability may bottleneck; without scaling curves and throughput or latency reports, I remain uncertain about practical viability.

### Questions
1. How sensitive are results to using a linear, zero-noise path? Do small positive noise or non-linear paths materially change alignment and synthesis at larger rotations or under lighting/occlusion shifts?
2. How robust is the method when same-object, same-transform pairs are imperfect (label noise, calibration drift, identity mixups)? What failure modes appear, and are there mitigation strategies?
3. Does the evaluation capture misalignment in higher-order components, not just the degree-1 block? Can you report multi-block or block-weighted summaries and encoder-agnostic checks?
4. Is the method still practical in high-dimensional latents and more complex groups? Please analyze the practical feasibility of applying this method to more complex structures.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper addresses the problem of latent misalignment in equivariant representation learning for geometry-aware generative models and novel view synthesis (NVS).

The author address this problem by adding another flow matching process to correct the analytically rotated latents to their corresponding empirically encoded targets. Additionally, the author can finetune the decoder to take the flow-matching corrected latents to reconstruct target signal, which could further improve performance. 

The author evaluate the methods a several synthetic image benchmarks, and shows that the proposed methods improves both alignment accuracy and novel view synthesis quality compared with baseline equivariant encoders.

To confess, I didn't fully understand what's the relationship between equivelanet representation and task like novel view synthesis, and how the methods compares to popular methods like NeRF and 3D GS. So, I would say I have low confidence in my review of the paper.

### Strengths
1. The paper address a very critial problem in equivelanet representation learning: the mismatch between analytically rotated latents to their corresponding empirically encoded targets.  And the author address the problem with a method that makes a lot of sense, train a generative model (flow matching) to correct it in latent space. 
2. The results seems quite good, though its at pure syntheic dataset.

### Weaknesses
1. I find it slightly hard to understand what's the input, and how is latent analytically rotated, and what's the training loss, untill I read the supplementary.  Maybe moving some details to 4.1 or 3.2 from Appendix B.3 might make reader understand easier? 
2. The author should include some baseline results on NVS, rether than only comparing with yourself, and the vanilla baseline.  Also, the PSNR in Figure-4 seems comfusing, it will be better to show the absolute number.

### Questions
1. So, do I understand it correctly, by training on a dataset of tens of objects, each with multiview inputs. What you are trying to do is kind of letting the model maybe mermoize the objects multiview, and also get a latent space that enables you to traverse over the viewpoint sphere?  
2. If my understanding is kind of correct, the equivelanet representation should not be able to generate nove views of unseen objects during training.  Cause generate nove views of unseen objects during training is a probabilistic task, and seems the representation space and the decoder is determinstic model, which does not has the ability to sample novel views, especially unseen part of the object? 
3. I kind of don't understand why this equivelanet representation would help for novel view synthesis?  and how does it compare with NeRF and 3DGS?

### Soundness
3

### Presentation
3

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
- In latent-to-image equivariant generative models of images, it is expected that an analytical transformation of the latent will lead to the exact corresponding transformation of the generated image. E.g., rotating the latent should produce the corresponding rotation in the output image.
- However, this never occurs exactly in practice and the paper posits that this is because the latent must encode non-pose related information such as lighting.
- The paper proposes a framework that maps the latents to new latents that produce exactly the desired transformation when decoded. It does so by using a flow matching generative model.
- Experiments are presented against a baseline on three representative object-centered synthetic datasets with exact control over the appearance and pose of the objects. They show that the proposed latent realignment approach produces higher-quality object transformations to unobserved poses.

### Strengths
- The paper is quite well written and explained. It is also refreshingly does not read like it was over-edited by LLMs, making it much easier and more enjoyable to read.
- Extreme effort has been put into making the paper as accessible as possible to new readers and it is appreciated.
- The figures are very strong as they get the main thesis of the paper across well.
- Designing a posthoc correction algorithm for latents in an equivariant model to better respect equivariance constraints is an interesting idea in general.

### Weaknesses
### Methodology and framing:
The paper's framing is confusing. It (A) considers a simple autoencoder with an equivariance loss to be the end-all in equivariant representation learning and (B) assumes that the quality of representations should be evaluated solely by considering whether the decoded image was correctly transformed. While I'm sure that this approach existed, it is not cited anywhere in the main text (as far as I can see) and it also appears to be the sole baseline. 
- Regarding (A): There are many equivariant representation learners for image synthesis that do not take this form and it is unclear if the proposed methodology would extend to them. 
- Regarding (B): Network representations are commonly evaluated using a diverse set of downstream tasks and only image synthesis (on synthetic datasets) is presented here. For example, what about equivariant representations for other tasks that may benefit such as pose estimation?  Further, the focus on image synthesis raises non relevant but practically material concerns like the experimental trends very much depending on decoder capacity, latent size, the image loss used, etc.

Currently, the gains displayed in Table 1 are marginal relative to the effort required to train a secondary posthoc generative flow model to improve the original model. It is unclear if these small gains will transfer to more powerful representation learning or generative modeling strategies.

Lastly, as a more minor point, L053 claims that some portion of the empirical lack of equivariance is due to the data variability in non-closed-form ways (e.g. occlusion, lighting). While this is true to some extent, even with a perfect dataset, this empirical lack of equivariance will always persist with current architectures due to aliasing of intermediate representations (see [1](https://arxiv.org/abs/2106.12423) and [2](https://arxiv.org/abs/2210.02984)). This statement in the paper should be given more nuance.

### Experiments:
- All of the experiments in the paper are conducted on entirely synthetic programmatically generated images. While these are great for benchmarking as they give exact ground truth, IMO they are insufficient to demonstrate that a given algorithm is useful for real problems. This paper would need to design and execute experiments with a real dataset or two before it can be considered to be broadly useful.
- As far as I can tell, there are no baselines in the entire paper, just a single uncited method that just removes the flow matching from the proposed method (i.e. an ablation). This is insufficient and the paper needs significantly more experimental depth. There are many, many papers in this space about equivariant latents creating controllable outputs such as [1](https://proceedings.neurips.cc/paper_files/paper/2024/file/e63309e532688c722177f81e99f94f32-Paper-Conference.pdf), [2](https://proceedings.neurips.cc/paper_files/paper/2023/file/9bfc2c20fa2f56a18397eafe1be8a50a-Paper-Conference.pdf), [3](https://proceedings.mlr.press/v162/birrell22a.html), [4](https://arxiv.org/pdf/1909.11663), [5](https://arxiv.org/pdf/2202.07559), [6](https://arxiv.org/pdf/2008.11673), [7](https://arxiv.org/abs/2106.12423), and many others including all the references within them. At least two to three relevant baselines should be included.

### Presentation:
- From Section 3.2, it is clear *how* the flow matching is performed. However, unless I missed something, the paper never clarifies *why* it decided on a flow matching setup. One can imagine many different ways to achieve this latent alignment, and flow matching is not the first approach that would come to mind. Please elaborate on the rationale for this choice and its comparative advantage over other approaches for this problem.
- It is quite unclear from the paper whether the flow matching module is trained alongside the image representation learner end-to-end or if its trained post hoc. Please clarify.
- The paper takes too long to get to its contribution (on page 5) as it covers the reasonably well known fundamentals of equivariant learning in great detail. As a result, many key details that are much more specific to the paper, like Appendices B.2--B.4 are missing from the main text. The background of their chosen model based on this relatively new strategy called "Neural Fourier Transforms" should be expanded as well.

### Questions
It is possible that I misunderstood core contributions and I am very open to being corrected during the discussion period. As of now, the experiments come off as too preliminary for this venue and the methodological setup is confusing (see above for specifics) -- please focus on these aspects in the rebuttal.

### Soundness
3

### Presentation
4

### Contribution
2
