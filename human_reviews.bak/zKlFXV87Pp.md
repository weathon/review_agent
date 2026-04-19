# Distilled Decoding 1: One-step Sampling of Image Auto-regressive Models with Flow Matching

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Autoregressive (AR) models have recently achieved state-of-the-art performance in text and image generation. However, their primary limitation is slow generation speed due to the token-by-token process. We ask an ambitious question: can a pre-trained AR model be adapted to generate outputs in just one or two steps? If successful, this would significantly advance the development and deployment of AR models. We notice that existing works that attempt to speed up AR generation by generating multiple tokens at once fundamentally cannot capture the output distribution due to the conditional dependencies between tokens, limiting their effectiveness for few-step generation. To overcome this, we propose Distilled Decoding (DD), which leverages flow matching to create a deterministic mapping from Gaussian distribution to the output distribution of the pre-trained AR model. We then train a network to distill this mapping, enabling few-step generation. The entire training process of DD does not need the training data of the original AR model (as opposed to some other methods), thus making DD more practical. We evaluate DD on state-of-the-art image AR models and present promising results. For VAR, which requires 10-step generation (680 tokens), DD enables one-step generation (6.3$\times$ speed-up), with an acceptable increase in FID from 4.19 to 9.96. Similarly, for LlamaGen, DD reduces generation from 256 steps to 1, achieving an 217.8$\times$ speed-up with a comparable FID increase from 4.11 to 11.35. In both cases, baseline methods completely fail with FID scores $>$100. As the first work to demonstrate the possibility of one-step generation for image AR models, DD challenges the prevailing notion that AR models are inherently slow, and opens up new opportunities for efficient AR generation. The code and the pre-trained models will be released at https://github.com/imagination-research/distilled-decoding. The project website is at https://imagination-research.github.io/distilled-decoding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a method called Distilled Decoding (DD) that distills autoregressive models into fewer steps. It employs flow matching to establish a deterministic mapping from Gaussian noise to the output distribution of the pretrained autoregressive models. Extensive experiments demonstrate that DD outperforms the baselines, accelerating autoregressive models with minimal performance degradation.

### Strengths
1. The paper is well written and easy to follow.
2. The presented idea of compressing AR models with Flow Matching is intersting. Given that autoregressive models tend to be slow, accelerating them is a crucial challenge. The efforts presented in this paper could provide valuable insights for the community.

### Weaknesses
1. Lines 201-203 state that modeling the joint distribution of several tokens is impractical due to the vast possible space. However, it seems that the proposed method uses Flow Matching to learn the joint distribution of tokens (the full sequence in the one-step case). This is somewhat confusing—how does the proposed method tackle this issue?
2. The image quality shows significant degradation (as indicated by both FID scores and the figures), raising concerns about the practicality of the proposed method.
3. It would be beneficial to include experiments with text-conditioned LlamaGen to demonstrate that the method can be adapted to more complex scenarios.

### Questions
In table 1, why VAR-DD with one step achieves better performance than VAR-DD with two steps?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper is concisely written with a clear line of thought. The approach of constructing a continuous embedding space in AR and matching it to a Gaussian distribution is particularly interesting. This construction effectively combines the discrete cross-entropy model from AR-based methods with the probability distribution strategies that have proven successful in diffusion models, allowing the concept of consistency distillation from diffusion to be successfully applied within the AR field. The comparisons are comprehensive, and the experiments are well-executed.

### Strengths
The paper is concisely written with a clear line of thought. The approach of constructing a continuous embedding space in AR and matching it to a Gaussian distribution is particularly interesting. This construction effectively combines the discrete cross-entropy model from AR-based methods with the probability distribution strategies that have proven successful in diffusion models, allowing the concept of consistency distillation from diffusion to be successfully applied within the AR field. The comparisons are comprehensive, and the experiments are well-executed.

### Weaknesses
The paper successfully applies the consistency distillation (CD) technique from diffusion models to the AR field. However, the current results still fall short compared to the pre-trained model.

### Questions
My questions lean more toward potential future design possibilities, as the explanations provided in the paper are relatively clear.

Q1: The approach seems very similar to a variant of the Consistency Model (CM) applied to the AR domain, replacing the diffusion model in CM with an AR model. What are the advantages of this substitution? Line 319 mentions that, compared to diffusion flow matching, DD “has an easy way to jump back to any point in the trajectory.” This is noted as a characteristic, but what practical benefits does it provide? Additionally, the ODE sampling trajectory in CM is also theoretically fixed and reversible. If my understanding is incorrect, please correct me.

Q2: Is there a way for this method to be trained from scratch, similar to Consistency Training?

Q3: Currently, the method requires a dataset size of 1.2M. How significantly does the data volume impact performance? Related distillation methods, such as DMD[1], use relatively smaller datasets. Is there a possibility of a similar approach to DMD2[2] that could discard the need for noise-data pairs?

Q4: Q: In line 234, is $\pi$ the same as z used later on? There may be a misalignment in the notation here.

[1]. Yin T, Gharbi M, Zhang R, et al. One-step diffusion with distribution matching distillation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 6613-6623.
[2]. Yin T, Gharbi M, Park T, et al. Improved Distribution Matching Distillation for Fast Image Synthesis[J]. arXiv preprint arXiv:2405.14867, 2024.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Distilled Decoding (DD), a novel method to reduce the sampling steps of autoregressive image generation models. The approach reframes the traditional next-token or scale prediction process into a denoising procedure guided by flow matching. Starting from fully noisy tokens, DD employs an autoregressive process to map these tokens into image tokens in a flexible number of sampling steps. Experiments with the LlamaGen and VAR models demonstrate that DD significantly outperforms conventional few-step sampling baselines.

### Strengths
- The authors proposed a new framework to make few-step AR distillation possible. 
- The conversion from next image token prediction to next (set of) image token denoising is a very smart design. The method naturally combines the best of worlds in diffusion models / flow matching and autoregressive image modeling.
- The performance gain against baseline few-step samplers is huge.

### Weaknesses
- The authors propose a novel framework enabling few-step autoregressive (AR) distillation.

- Converting next-image-token prediction to next-image-token denoising is an ingenious design choice, seamlessly integrating the strengths of both diffusion models and autoregressive image modeling.

- The performance improvement over baseline few-step samplers is substantial.

### Questions
- The writing is somewhat unclear. It's not obvious whether $q_t$ represents tokens in the discrete codebooks or outcomes of flow matching. Algorithms 1-4 are much clearer than the main text, so refining the writing to enhance intuitiveness would benefit the readers.

- The performance gap between few-step AR samplers and the baseline remains quite large.

- Some baseline values, such as VAR/LlamaGen in Table 1, appear to be significantly lower than the original paper's reported results.

- It would be interesting to explore how DD performs in text-conditioned image generation tasks.

- The structure of DD closely resembles DMD in diffusion distillation, where a <noise, clean image> dataset is constructed; however, DMD isn't addressed in this paper. Adding a discussion on DMD would be beneficial. Additionally, transforming next-token prediction to AR-based token denoising at a high level seems conceptually similar to MAR (Li et al., 2024, cited). Including further discussion on MAR could also be helpful.

===

Update: I've read the author response and it has addressed my concerns, I will therefore keep my score and recommend acceptance.

### Soundness
3

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
The method maps autoregressive model output distributions to a Gaussian distribution through flow matching, creating a deterministic transformation. The goal is to then learn this mapping with a neural network. This enables parallel token generation while preserving the conditional dependencies between tokens in the AR model's output distribution.

### Strengths
The work presents a method that leverages deterministic flow matching to create training data (from an AR model) for a one-step image generation model. When trained on this data this model is a distilled version of the original AR model. The idea of using determinstic flow matching to create the data is novel and seems like a good and innovative candidate idea to achieve this. The paper evaluates the claims on class-to-image generation on ImageNet and compares to simple baselines, achieving acceptable FID scores, that are high but not too high.

### Weaknesses
A. It seems like the FID increases, although seemingly acceptable in numerical terms, give rise to blurry and artifact-ridden images, many of which don't even preserve the structure of the class they are trying to generate (monkeys without eyes etc.). Also, another thing that undermines my confidence is that no images are shown in the main paper and instead shown in the appendix. At least some examples are shown.

B. One big problem from (A) is that, since the paper's main premise is to distill an AR model into a one-shot model, good samples are required to demonstrate that this approach is viable. Given the current results we do not know if this approach can actually generate satisfactory images even when scaled or taken into another regime with larger data such as T2I.

I think these are the main issues for me to be convinced of this approach. My question to authors are:
- Do you expect the approach to become good enough to generate satisfactory images with different scale (data, model size)?
- If so, what leads you to believe this?

### Questions
1. Your method depends on DPM-Solver to generate the trajectories. Do you think this affects results in any way? What other dependencies does your method have that might affect results?
2. Do the hyperparameters for DPM-Solver affect the results drastically? Why did you pick the current hyperparams?
3. Do you think a larger amount of timesteps might be needed to get satisfactory results? Is this compatible with your method?
4. Skip baselines seem conceptually weak. The one-step baseline is guaranteed to fail given results on toy examples. Are there no better baselines - for example progressive distillation, that is used for diffusion models?

### Soundness
3

### Presentation
2

### Contribution
3
