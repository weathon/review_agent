# A Probabilistic Hard Concept Bottleneck for Steerable Generative Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
Concept Bottleneck Generative Models (CBGMs) incorporate a human-interpretable concept bottleneck layer, which makes them interpretable and steerable. However, designing such a layer for generative models poses the same challenges as for concept bottleneck models in a supervised context, if not greater ones.  Deterministic mappings from the model inner representations to soft concepts in existing CBGMs: (i) limit steerable generation to modifying concepts in existing inputs; and, more  importantly, (ii) are susceptible to *concept leakage*,  which hinders  their steerability. To address these limitations, we first introduce the Variational Hard Concept Bottleneck (VHCB) layer. The VHCB maps probabilistic estimates of binary latent variables to hard concepts, which have been shown to mitigate  leakage. Remarkably, its probabilistic  formulation enables direct generation from a specified set of concepts.  Second, we propose a systematic evaluation framework for assessing the steerability of CBGMs  across various  tasks (e.g., activating and deactivating concepts). Our framework which allows us to empirically demonstrate that the VHCB layer consistently improves steerability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on concept bottleneck generative models (CBGM) which are interpretable and steerable generative models. Existing CBGMs are deterministic models which limit steering to concepts predicted in existing inputs and may suffer from concept leakage. To overcome these limitations, this paper proposes a Variational Hard Concept Bottleneck (VHCB) layer which allows direct generation from sampled concepts and could reduce concept leakage. They evaluate the proposed method compared to the previous SOTA method and show improvements in steerability.

### Strengths
* The paper is fairly well-written and easy to follow with good illustrations.

* The paper focuses on the relatively under-explored topic of concept bottleneck generative models.

* The proposed VHCB layer seems to be novel and it makes intuitive sense to move from deterministic CB layers (like in CB-AE or CBGM) to probabilistic ones like VHCB, especially given that the generative task itself is not deterministic.

* I also appreciate the extensive experiments, especially the disentanglement experiments which were not reported in prior work.

### Weaknesses
* It is unclear to me how the error correcting codes help mitigate concept leakage. Rather it seems that increasing the dimensionality of the concepts (and the unsupervised embedding) would increase the chances of concept leakage. It would be good to elaborate on why (and which part of VHCB) actually helps to mitigate concept leakage.
    * Another related concern is that concept leakage might actually stem from the unsupervised embedding itself. It would be good to discuss this in the paper and comment on whether this can be removed from concept bottleneck generative models.

* Based on Table 10, it seems VHCB sometimes achieves better and sometimes worse FID scores than CB-AE, but there is no explanation/intuition offered for this.
    * Related to this, the FID scores are significantly higher when generating from random concepts which is attributed to some concept combinations potentially being out of distribution. I wonder if this can be improved by generating images from random concept samplings during training and applying a loss on the concept predictions (or an image generation loss to improve the quality of those images) similar to the intervention losses in CB-AE.
    * The higher FID scores when generating from random concepts also seems to be a major limitation because that was proposed as one of the main contributions of this work (which was not possible in the deterministic CBGM or CB-AE).

* Minor issues (typos):
    * L113: "Blottleneck"

### Questions
Please address my questions/comments from the weaknesses section.

### Soundness
2

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
4

### Summary
The paper introduces the Variational Hard Concept Bottleneck (VHCB) layer, a probabilistic extension of Concept Bottleneck Generative Models (CBGMs). Instead of previous approaches, it uses hard binary concept representations (binary latent variables with probabilistic smoothing) to reduce concept leakage and a probabilistic latent formulation to enable direct sampling from concept configurations. The approach is validated on StyleGAN2 and DDPM trained on CelebA-HQ and CUB-200-2011 datasets.

### Strengths
- Addresses an important and previously challenging problem in the concept bottleneck literature. 
- The side channel regularization is quite interesting. This 'independence regularization' term is quite different from those in the literature. It acts as a capacity control and seeks to prevent concept leakage into the unsupervised channel. 
- The evaluation, in the work, compared to the baselines, is quite thorough. 
- The paper also has a nice trick for reducing potentially incoherent steering on concepts. They restrict interventions to target one of the 100 most probable concept patterns observed in the dataset, selected via minimum Hamming distance. This ensures that interventions are more coherent. Of course, this is also limits the ability of the model to extrapolate.

### Weaknesses
- The probabilistic formulation is based on existing binary VAEs (from Martínez-García et al., 2025). This works applies it to CBGMs directly. 
- The paper introduces a binary side channel regularized toward a factorized prior via a KL penalty to mitigate concept incompleteness. This mechanism does not guarantee independence from the concept latents; it only limits capacity. The authors could strengthen their claims by including ablations on the KL weight, measuring mutual information between, and or visualizing variations induced by the unsupervised channel to confirm it captures residual, non-semantic factors.
- The current disentanglement evaluation, in the paper, is also not quite appropriate. It involves: sampling two independent latent pairs, swapping the unsupervised latents  to form mixed pairs, and regenerating images and comparing whether the concepts remain unchanged. This test examines whether observable concept labels change, but not whether these latents themselves are statistically independent. You could have apparent disentanglement (concepts unchanged) while they still encodes correlated concept information in latent space. If those classifiers are not sensitive to subtle variations, you can get artificially high “disentanglement” scores. I would recommend the authors consider other concrete measures of disentanglement. For example, the Glancenets paper that the authors cite has a measure, DCI, and toy experiments that they can use to stress test their approach.

### Questions
- Why a symmetric KL instead of standard cross-entropy or forward KL? 
- It is hard to disentangle whether leakage prevention comes from binary concepts or the KL regularization. Can the authors speak more to this? Can you train a simplified version of your model in a toy setting like in the work of GlancNets so that you can tease these issues apart? 
- I would suggest that the authors also look into measure of disentanglement or independence between latent factors to concretize their claims.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes the Variational Hard Concept Bottleneck (VHCB) layer that uses binary latent variables protected by error-correcting codes to map internal representations to hard human-interpretable concepts within pretrained generative models. Also, the authors introduce a systematic evaluation framework for assessing the steerability of CBGMs across various tasks, and demonstrate that VHCB consistently enhances steerability empirically. In general, the paper is clearly written, the metrics are sensible, and the authors also points out the impact of correlations and biases in training data, which is helpful for future work.

### Strengths
1. The authors introduce the limitations of existing approaches (i.e. relying on soft concept representations and susceptible to concept leakage), and the proposed VHCB layer is thus well motivated and is a clear conceptual advance.
2. The systematic evaluation framework for CBGMs proposed by this paper is thorough and comprehensive, which can be useful for many purposes.
3. The paper presents compelling empirical results that VHCB yields large gains in accuracy, and also transparently analyzes failures due to dataset correlations/bias.

### Weaknesses
1. The main comparison in this paper is VHCB vs. CB-AE. It would be helpful to compare the proposed method to several more baselines.
2. The authors use independent concept classifiers to obtain labels, which could introduce label bias that propagates into training and evaluation.

### Questions
1. Can you report a simple leakage diagnostic to empirically support the claim that hard concepts + compact s reduce leakage?
2. What would happen if you insert VHCB at multiple U-Net locations on DDPMs?

### Soundness
3

### Presentation
3

### Contribution
3
