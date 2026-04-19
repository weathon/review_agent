# Unleash Data Generation for Efficient and Effective Data-free Knowledge Distillation

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 6, 3

## Abstract
Data-Free Knowledge Distillation (DFKD) has recently made remarkable advancements with its core principle of transferring knowledge from a teacher neural network to a student neural network without requiring access to the original data. Nonetheless, existing approaches encounter a significant challenge when attempting to generate samples from random noise inputs, which inherently lack meaningful information. Consequently, these models struggle to effectively map this noise to the ground-truth sample distribution, resulting in the production of low-quality data and imposing substantial time requirements for training the generator. In this paper, we propose a novel Noisy Layer Generation method (NAYER) which relocates the randomness source from the input to a noisy layer and utilizes the meaningful label-text embedding (LTE) as the input. The significance of LTE lies in its ability to contain substantial meaningful inter-class information, enabling the generation of high-quality samples with only a few training steps. Simultaneously, the noisy layer plays a key role in addressing the issue of diversity in sample generation by preventing the model from overemphasizing the constrained label information. By reinitializing the noisy layer in each iteration, we aim to facilitate the generation of diverse samples while still retaining the method's efficiency, thanks to the ease of learning provided by LTE. Experiments carried out on multiple datasets demonstrate that our NAYER not only outperforms the state-of-the-art methods but also achieves speeds 5 to 15 times faster than previous approaches. The code is available at \url{https://github.com/fw742211/nayer}.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this study, the authors address the challenge associated with generating pseudo samples from generators in the adversarial data-free knowledge distillation framework. They introduce the Noisy Layer Generation method (NAYER), an innovative approach that shifts the source of randomness from the input to a designated noisy layer. Instead of traditional inputs, the authors utilize label-text embedding (LTE), which encapsulates significant inter-class distinctions. This strategic incorporation of LTE enables learning high-quality samples faster. Simultaneously, the noisy layer augments sample diversity, ensuring that the model doesn't overly fixate on the label.

### Strengths
1. This research stands out as one of the first efforts in DFKD that harnesses a foundational model like CLIP.

2. While current state-of-the-art DFKD methods are often time-intensive and require prolonged training periods for knowledge transfer, the authors convincingly demonstrate in Table 2 that they achieve a marked acceleration. This efficiency is attributed to their use of latent text embeddings, which encode nuanced interclass relationships, thereby enhancing the generation process by exploiting these relationships.


3. Figure 3(d) highlights the generator's undue emphasis on labels within the Adversarial DFKD framework. This observation then drives the authors to inject randomness using noisy layers into the label-text embeddings sourced from CLIP, addressing the identified limitation.


4. Notably, the authors present comparisons and results on expansive datasets like ImageNet, seldom seen in similar works.


5. The 'Extended Results' section, located in the appendix, offers profound insights. It includes rigorous ablation studies, examining various language embeddings and their respective noise-embedding strategies.

### Weaknesses
Major Weaknesses:
1. Clarity on Noisy Layers: The proposed method heavily relies on the noisy layers. However, in its present form, the manuscript does not elucidate the mathematical intricacies of the noisy layer, denoted as $Z$. The authors seem to present this layer as an opaque entity. It remains unclear whether the noisy layer employed is analogous to the one proposed by Fortunato et al. [1]. Given the rarity of noisy layer implementations in literature, the authors should elucidate its underpinnings, possibly drawing comparisons to prior work. The primary novelty appears to stem from the introduction of this noisy layer, and without deeper insight into its operations, the paper seems to lack substantial technical contributions.


2. Ambiguities in Section 3.3: The closing remarks of Section 3.3, where the generation of MK synthetic images is broached, seem nebulous. A more comprehensive and systematic exposition of this segment would be beneficial. Furthermore, the authors' decision to employ BatchNorm with embeddings is not clearly justified. It would be insightful to understand the rationale and the potential implications of omitting this step.


3. Memory Overhead Concerns: While the study commendable reduces student training time, there's no discussion on the potential increase in memory overheads that might be attributed to the introduction of noisy layers.


Minor Weaknesses:
1. Citation Misrepresentation in Section 3.1: The initial sentences of the second paragraph of Section 3.1 mistakenly attribute an adversarial mechanism to Nayak et al. (2019). In reality, Nayak and his collaborators proposed a strategy for generating class impressions from pretrained teachers, leveraging these for knowledge distillation.

Reference:

[1] Fortunato, Meire, et al. "Noisy networks for exploration." arXiv preprint arXiv:1706.10295 (2017).

### Questions
See Weaknesses

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper explores a more effective data-free knowledge distillation. The authors believe that the previous DFKD-based generation generated samples from random noises, so no very effective information was extracted. Therefore, in the paper, the authors introduce a novel Noisy Layer Generation method (NAYER) that relocates the randomness source from the input to a noisy layer and utilizes the meaningful label-text embedding (LTE) as the input. Based on this method, this work It can achieve high efficiency while ensuring the diversity of data generation. Extensive experiments illustrate the effectiveness of the proposed method.

### Strengths
1. The authors focus on the relationship between the diversity and efficiency of sample generation, which is important for DFKD.

2. The code is released.

### Weaknesses
1. CLIP is introduced in this paper, and the training of CLIP requires a large amount of additional data. Other comparison methods do not seem to introduce additional data and only use teachers, which seems to be an unfair comparison. More explanation is needed here.

2. Lack of comparison with sampling-based methods [1][2]. More importantly, there is also design about noise in DFND[1]. Although not exactly the same, it should warrant comparison and discussion.

3. There is a lack of comparison with [3] in terms of generation efficiency. In addition, the distillation performance is not as good as [1][2].

[1] Learning Student Networks in the Wild, CVPR 2021

[2] Sampling to Distill: Knowledge Transfer from Open-World Data, arxiv 2023

[3] Up to 100$\times$ Faster Data-free Knowledge Distillation, AAAI 2022

### Questions
See Weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new method for data free KD method, Noisy Layer Generation (NAYER), which relocates the randomness source from the input to a noisy layer and utilizes the meaningful label-text embedding (LTE) as the input. LTE, generated by a pretrained text encoder, contains meaningful inter-class information, that enables the generation of high-quality samples with only a few training steps. LTE layer is initialized in each iteration for the diversity of generated images. Experiments suggest the proposed method outperforms other counterparts while being training faster too.

### Strengths
1. The idea of using pretrained text encoder to generate label-text embeddings as input for distillation is interesting and sounds novel to me.

2. To achieve diverse generated samples (which is a key problem in DFKD), they propose a noisy layer between the input and the generator. The noisy layer is initialized in each iteration, which effectively introduces more diversity for the synthetic samples.

3. The proposed method not only outperforms other DFKD approaches in terms of student performance but also is much faster in training.

### Weaknesses
1. Since this paper utilizes the pretrained text encoder in CLIP model, I think a similar idea for DFKD is to use pretrained text-to-image generation models, such as stable diffusion to generate pseudo data for distillation. It is advisable to add a set of comparison experiments to show the performance difference.

2. What is the effect of reinitializing the noisy layer in each iteration? What if it is not reinitialized? This key ablation study is missing now.

3. The presentation has some small issues to fix:

3.1 The text in Fig. 4 is too small, hard to make out.

3.2 Eq. (6) and (7) should have some punctuation. Make them in a sentence, not orphaned.

3.3 Missing period in the caption of Fig. 5.

### Questions
How many images are stored in the memory module M in each epoch? Does this affect the performance significantly (any ablation study about it)?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Data-Free Knowledge Distillation (DFKD) has made significant strides in recent years, with its core principle of transferring knowledge from a teacher neural network to a student neural network without requiring access to the original data. However, existing approaches face a major challenge when attempting to generate samples from random noise inputs, which lack meaningful information. As a result, these models struggle to effectively map this noise to the ground-truth sample distribution, leading to low-quality data and substantial time requirements for training the generator.

To address this issue, this paper proposes al Noisy Layer Generation method (NAYER) that relocates the randomness source from the input to a noisy layer and utilizes the meaningful label-text embedding (LTE) as the input. The significance of LTE lies in its ability to contain substantial meaningful inter-class information, enabling the generation of high-quality samples with only a few training steps. Simultaneously, the noisy layer plays a key role in addressing the issue of diversity in sample generation by preventing the model from overemphasizing the constrained label information. By reinitializing the noisy layer in each iteration, this work aims to facilitate the generation of diverse samples while still retaining the method’s efficiency.

### Strengths
- The paper introduces a DFKD method called Noisy LAYER Generation (NAYER) that relocates the source of randomness from the input to the noisy layer and utilizes label-text embedding (LTE) as the input. Using LTE as input allows for proficient generation of high-quality samples that closely mimic the distributions of their respective classes with only a few training steps. 
- Extensive evaluation is presented on standard benchmark datasets like CIFAR10, CIFAR100, TinyImageNet, and ImageNet. It achieves superior performance against the prior arts.

### Weaknesses
1) In the introduction section, paragraphs 4 and 5 seem disconnected from the rest of the introduction and disrupt the flow of reading. There are seveal references to different figures and experimental results that are presented in other pages and also in supplementary. This disturbs the flow of reading. The authors should rewrite these paragraphs in a way that simplifies the key ideas, potentially using examples to make them more accessible to a broader audience. It's important for the introduction to provide a smooth and coherent overview of the paper's content to engage a wider audience.

2) The paper proposes the use of label-text embedding (LTE), which may have a disadvantage compared to other methods that do not rely on such embedding. For example, in cases like classifying chemical compounds, where label-text embedding, like CLIP, may not be applicable. Other data-free knowledge distillation methods do not rely on such joint image-text embedding knowledge thus they world perform reasonably well for a wide variety of classification modalities (such as audio, chemical-compund, etc.). Highlighting the limitations of the proposed approach is essential for a balanced evaluation of its potential use cases.

3) The paper uses label-text embedding (LTE) obtained from a pre-trained model CLIP that is trained on image-text pairs from the internet. Drawing a comparison to CLIP, which also has knowledge of common objects and corresponding text, suggests that the proposed method is not entirely data-free. This raises doubts about whether it can be truly considered "data-free" distillation.

4) Limited novely: The proposed approach builds upon CLIP embedding for data-free knowledge distillation but offers limited innovation. Section 3.1 appears to reiterate the existing Data-free knowledge distillation framework without introducing fresh perspectives or insights beyond what has been discussed in prior works. Further, the use of label-text embedding followed by a randomly initialized layer makes the generator resemble a conditional GAN. There are various alternative ways to achieve the desired setup, such as: a) employing a conditional GAN where a noise vector-based embedding and an LTE-based embedding are concatenated to form the initial part of the generator network, b) using an equation like e + beta*(z~ N(0, I)) to model intra-class diversity and span the embedding space between classwise LTE embeddings, or c) exploring some form of linear combination of LTE embedding with specialized weight sampling. The inclusion of the proposed Noise Layer appears unnecessary and lacks clear justification.

### Questions
Please see the Weaknesses section.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
