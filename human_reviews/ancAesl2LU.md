# DomainStudio: Fine-Tuning Diffusion Models for Domain-Driven Image Generation using Limited Data

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 8, 3, 3

## Abstract
Denoising diffusion probabilistic models (DDPMs) have been proven capable of synthesizing high-quality images with remarkable diversity when trained on large amounts of data. Classic unconditional and modern large-scale conditional diffusion models are still vulnerable to overfitting when fine-tuned on extremely limited data. Existing works have explored subject-driven generation using a reference set containing a few images. However, few prior works explore DDPM-based domain-driven generation, which aims to learn the features of target domains while maintaining diversity. This paper proposes a novel DomainStudio approach to adapt DDPMs pre-trained on large-scale source datasets to target domains using limited data. It is designed to keep the diversity of subjects provided by source domains and get high-quality and diverse adapted samples in target domains. We propose to keep the relative distances between adapted samples to achieve considerable generation diversity. In addition, we further enhance the learning of high-frequency details for better generation quality. Our approach is compatible with both unconditional and text-to-image DDPMs. This work makes the first attempt to realize unconditional DDPM-based few-shot image generation, achieving better results than current state-of-the-art GAN-based approaches. It also significantly relieves overfitting for domain-driven text-to-image generation, expanding the applicable scenarios of modern large-scale text-to-image diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a new domain-driven generation approach called DomainStudio, which aims to learn the features of target domains while maintaining diversity using limited data. 
The key components of this approach are relative distance preservation and high-frequency detail enhancement. To achieve better diversity, the relative distance preservation is proposed, which uses pairwise similarity loss to keep the relative pairwise distances between generated samples similar to source samples. Additionally, high-frequency detail enhancement is proposed to address the lack of high-frequency details in generated results. This enhancement guides the adapted model to learn more high-frequency details from limited training data and source domains.
The paper investigates both unconditional image generation and text-to-image generation methods and conducts comprehensive experiments to demonstrate the effectiveness of DomainStudio. DomainStudio outperforms current state-of-the-art unconditional GAN-based approaches and conditional DDPM-based text-to-image approaches in terms of both generation quality and diversity.

### Strengths
1.	This paper is well-written, with a clear motivation and definition of domain-driven generation.
2.	Comprehensive experiments were conducted to evaluate the performance of both unconditional and conditional DDPMs fine-tuned on limited data, revealing a problem with existing state-of-the-art generative models. The experiments in Section 5 effectively demonstrate the effectiveness of the proposed method and each key component of the paper.
3.	The performance of this paper is significant, as it achieves better generation quality and diversity than current state-of-the-art unconditional GAN-based approaches and conditional DDPM-based text-to-image approaches.

### Weaknesses
The relative distance preservation is not the first time to be proposed. The cross-domain correspondence proposed in the paper “Few-shot image generation via cross-domain correspondence” is very similar to the one in this paper.

### Questions
1.	As mentioned earlier, relative distance preservation has been utilized in GAN-based methods. Could you please explain the differences between the relative distance preservation approach in this paper and the cross-domain correspondence approach in "Few-shot image generation via cross-domain correspondence"? 
2.	Assuming that StyleGAN2 models and DDPMs trained on large source datasets share similar generation quality and diversity, why does this paper outperform the CDC paper in terms of performance?
3.	In Section 4, what is the reason behind the statement that "Subjects in source and target domains do not need to be consistent with training samples"?
4.	Could you please explain why high-frequency enhancement not only improves image details but also enhances diversity, as shown in Tables 6 and 7? It appears that high-frequency enhancement and relative distance preservation have similar loss functions, resulting in similar effects.
5.	Are there any DDPM models for few-shot image generation other than subject-driven series methods? If so, what is their performance?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces DomainStudio, a strategy for fine-tuning pretrained diffusion models towards very small adaptation datasets (no more than 10 images), without losing the ability to generate diverse and highly detailed images. The proposed strategy can work on either unconditional image generation aimed at matching the style of the fine-tuning images, or text-conditioned image generation to generate images that match the style of the fine-tuning images while also matching specific structure or content described in the text prompt. 

The main idea of the method is the introduction of two regularization losses to supplement the standard diffusion training loss. One of these losses is designed to promote diversity in the predicted images, and the other is designed to promote high-frequency details. Combined, these regularizers produce noticeable improvement even compared to DreamBooth and other high-quality baselines, particularly for settings where there are very few images provided for fine-tuning.

### Strengths
Overall the paper is well-written and easy to follow, and the results are visually compelling.

### Weaknesses
Substantial suggestions for improvement
- Probably my most significant concern/suggestion is to include the quantitative evaluation results in the main paper, rather than putting them in the supplement. This is a core way that future papers will be able to build on your work and compare to it. I realize that in this field the metrics aren’t always super well-aligned with human perception of quality, so they may not be too meaningful, but in my opinion it would be better to include them and explain any caveats to the metrics rather than sweeping all of it under the rug (to the appendix).
- Likewise, I would strongly encourage the authors to include the ablation study results in the main paper. This is the only real justification for introducing two new losses; it merits inclusion in the main text.
- The introduction is long and the first half of it reads like a related work section. I would prefer if the authors more directly introduce the main idea of their contribution, and this would also save space that could be put towards the quantitative results and ablations. When citing related work, especially in the introduction, I would also encourage the authors to mention the name of the paper (e.g. DreamBooth) so that readers don’t need to flip back and forth to the references to understand what is being described.
- The related work section could also be improved and shortened. Currently it reads like a survey with each related paper described by a sentence, but not much higher-level analysis to help the reader group the related works into relevant categories and understand how these categories differ from each other and from the proposed method.

### Questions
Minor suggestions for improvement
- Figure 1 caption describes samples containing the “same subject as training samples”, but the examples shown (“a [V] house”) do not look like they are actually preserving the “identity” of a specific house from the training set.
- If you want a citation to not interrupt the flow of a sentence, you can use \citep rather than \citet. If you want to use the citation as a noun in a sentence, use \citet.
- The “Text-to-image Generation” paragraph describes DreamBooth as a “light-weight fine-tuning method”, but my impression was that DreamBooth is relatively expensive as a fine-tuning method because it edits the full weights.
- I don’t really buy the motivation for the high-frequency regularization—the sample images in Figure 2 and Figure 3 don’t strike me as particularly low-detail. Including the ablation in the main text would help motivate this loss, or perhaps there are other examples you can show to motivate why this loss is needed.
- It would be nice if the captions (or figures) for Figure 5 and Figure 6 more directly described the novelty of the contribution, and how the two figures fit together, since these are the main visual illustration of the method.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work proposes an approach for adapting the pre-trained denoising diffusion probabilistic models (DDPMs) to a target domain using limited data. It discussed three major contributions:
- analyzing the shortcomings of adapting DDPMs to target domains with limited data
- proposing a pairwise loss to keep the diversity of the image generation in the adapted generative model
- exploiting the frequency components for improving the generation quality
This work also claims to be the first one that achieves better generation performance for few-shot image generation (e.g., 10-shot) compared to SOTA works based on the GAN.

### Strengths
This work aims to address a crucial aspect of image generation with limited data in DDPMs. Even though this is studied in a length for GANs, there is a lack of efficient approaches for DDPMs under limited data or few-shot setups.

### Weaknesses
I have major concerns regarding the current version of this work:

I. All of the three contributions mentioned in the paper, have different types of flaws and problems:

a) Section 3 aims to study the adaptation of a pre-trained DDPM to a target domain using limited data for two different types of 
generations: unconditional generation and text-to-image generation. However, this part has nothing new, and lacks enough details:

- The major problem with this section is that this phenomenon is quite well-known in generative models, and simply running this for DDPMs is not adding any value in my opinion, and this can not be considered as a contribution!

- In Figure 2, the details of measuring the similarity are missing. For example, which features are used to compute the cosine similarity between two images?
In addition, this study is not enough. Using only two samples to discuss the diversity and the quality is not a technical way to discuss this and a more systematic approach for this is required (e.g., using quantitative measurements like FID for generated data).

- In Figure 3, the details are also missing. In addition, the approach used for the comparison (DreamBooth) is not related to the proposed method in terms of the task as this approach aims to implant a subject in a pre-trained model by linking it to a unique identifier.
The question is how Dreambooth is going to assign all these varied inputs into an identifier [v]? In this Figure, the Dreambooth is asked to learn the target domain distribution however it was originally designed to implant subjects in the same domain.

- Figure 3 gives a very bad impression that it is not clear to the authors which task they do have in mind, and they mix some unrelated concepts (generative tasks; subject diven generation vs domain adaptation; see [1]) together!

b) The pairwise loss proposed in sec. 4.1. is identical to the main idea proposed in the cross-domain correspondence (CDC) [CVPR'21] proposed for GANs, and this work fails to discuss the related work, borrowing the idea from CDC, and the possible new parts added for DDPMs.

c) Similarly, the third contribution, has been studied before in the literature in FreGAN [2] and MaskedGAN [3] and this work fails to discuss the related work and what is different in this paper.

II. My second concern is regarding the clarity of the tasks that this work aims to address. The unconditional generation task is clear from similar studies in the literature. But for the second task:
- the definition of the text-to-image generative domain adaptation is not clear to me, and there is no clear and technical definition in this paper
- This task seems to be **unnecessary**, as the pre-trained text-to-image model can handle this task and there is no need for adaptation. For the running example in most parts of the paper (e.g., the house painting by Van Gogh), the pre-trained text-to-image model (latent/stable diffusion) generates the required images using the proper prompt. For example, one can pass "The Yellow House in Vincent Van Gogh painting style" to stable diffusion / DALL.E and get much better results (I personally did) than what is shown in Figures 1 and 2.


III. The final concern is regarding the experimental results:

- For the first task (unconditional generation), the quantitative evaluation is shifted to the supp. which in my opinion is the most critical part that shows your approach can outperform state-of-the-art (as claimed in the abstract).
Checking the results in supp. (Tables 2 and 3 of supp), this paper only compared with old approached (DCL from CVPR'22) and excludes the recent works (AdAM from NeurIPS'22, and RICK from CVPR'23) that have much better performance than the proposed method. Including these two methods can clarify how efficient the proposed method is.

- The visual results provided in Figure 6 (for Raphael's painting) is not very appealing. The generator can not adapt to the target domain in this case and is almost similar to the source domain FFHQ.

- For the second task, as the pre-trained text-to-image model can already generate some images using a proper text prompt (as discussed before) experimental results should justify the necessity of tasks in terms of either failing the pre-trained model, or better performance using the proposed method.

I am willing to increase my scores if you can provide enough details regarding these weaknesses.

**References:**

- [1] A survey on generative modeling under limited data, few shots, and zero shot

- [2] FreGAN: Exploiting Frequency Components for Training GANs under Limited Data

- [3] Masked generative adversarial networks are data-efficient generation learners

### Questions
Please refer to weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a method for domain tuning without overfitting, using approximately 10 chapters of limited data. To achieve this, the paper employs relative distances preservation and high-frequency reconstruction loss.

### Strengths
The results seem good without overfitting while they tune the entire set of parameters instead of LoRA. Additionally, it makes sense to maintain the pairwise distance of previously generated images.

### Weaknesses
The paper focuses on comparisons with GAN-based approaches and lacks a comparison with LoRA, which is a more commonly used method. Also, there should be more ablation studies such as showing effects of each loss term.

### Questions
Is there an improvement in performance compared to the simpler LoRA method? It would be interesting to see the results for ablation studies such as with and without high-frequency reconstruction loss and pairwise similarity loss.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
