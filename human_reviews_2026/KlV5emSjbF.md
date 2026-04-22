# Universal Multi-Domain Translation via Diffusion Routers

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Multi-domain translation (MDT) aims to learn translations between multiple domains, yet existing approaches either require fully aligned tuples or can only handle domain pairs seen in training, limiting their practicality and excluding many cross-domain mappings. We introduce universal MDT (UMDT), a generalization of MDT that seeks to translate between any pair of $K$ domains using only $K-1$ paired datasets with a central domain. To tackle this problem, we propose Diffusion Router (DR), a unified diffusion-based framework that models all central$\leftrightarrow$non-central translations with a single noise predictor conditioned on the source and target domain labels. DR enables indirect non-central translations by routing through the central domain. We further introduce a novel scalable learning strategy with a variational-bound objective and an efficient Tweedie refinement procedure to support direct non-central mappings. Through evaluation on three large-scale UMDT benchmarks, DR achieves state-of-the-art results for both indirect and direct translations, while lowering sampling cost and unlocking novel tasks such as sketch$\leftrightarrow$segmentation. These results establish DR as a scalable and versatile framework for universal translation across multiple domains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces universal Multi-Domain Translation, a problem setting for learning translation between any pair of K domains using only K-1 paired datasets. Specifically, this paper proposes a unified diffusion-based framework that avoids the need for training separate models for each domain pair. The framework is presented in two variants: iDR that routes translations through the central domain, and a dDR that distills iDR’s indirect path distributions into direct non-central mappings via a variational-bound objective and Tweedie refinement. To validate this approach, the authors construct three new image-based benchmark datasets and demonstrate that DR achieves state-of-the-art performance.

### Strengths
1. The problem formulation is well-motivated, and the background is clearly explained. 
2. Extensive experiments validate the effectiveness of the proposed method on the constructed star- and chain-structured image-domain conditional generation tasks.
3. The writing is generally clear.

### Weaknesses
1. The universal claim is not supported by intra-modality experiments. The authors motivate the problem with compelling cross-modal examples like image/text/audio translation. However, all experiments are within the image-to-image translation domain, and the current results only prove the method's capability as a multi-style image translation system.
2. The paper's contribution appears less like the introduction of a new paradigm and more like a new training setup for an existing problem. When viewed as a solution for multi-style image translation under a specific data-sparsity assumption, its conceptual novelty is reduced.  
3. iDR relies on multi‑step inference, and its outputs can accumulate approximation and routing errors. The proposed training of dDR distills from iDR’s generations, thereby inheriting these accumulated errors, which bounds dDR’s quality.  In addition, although DR is a single network whose parameter is roughly independent of the number of domains K, the number of non-central directed pairs to cover grows roughly as O(k^2). Together with Tweedie refinement for conditional sampling, this implies that total training time requirements scale up significantly with K.

### Questions
Minor clarifications for presentation. The X/Y format is not explicitly defined in tables. Please add a sentence to the table captions clarifying that the two numbers correspond to the forward and backward translation directions, respectively (e.g., "A->B / B->A").

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
This paper proposes a universal multi-domain translation model wherein a single model can be used to do pairwise translations across all the K considered domains with only requiring K-1 paired datasets.

### Strengths
- The proposed diffusion routers is novel
- The results are promising and outperforms baselines based on both qualitative and quantitative results
- They construct 3 different datasets and conduct extensive experiments on them to validate their approach

### Weaknesses
- The domains considered for all the datasets they constructed are quite limited. They only consider the image, segmentation, sketches, and depth, which are quite similar in structure only different appearance / details. 
- The baselines compared are also quite old (2018, 2022, 2023). Granted that there are relatively few works on multi-domain translation models, I think it would still be good to also compare with more recent single domain translation models see how good its performance is compared to a dedicated model. This would also show whether or not training on more domains could improve performance due to having more auxiliary information.
- For the domains involved, the recent text-to-image models can also already achieve this kind of translation. It would be good to also compare with them since in a way it is also able to perform multi-domain translation with a single model.
- The authors motivated a lot of their work with image-text-audio domains. However, none of the experiments show any translations between these domains.

### Questions
- How does the performance of the model change depending on what you consider as the central domain? Currently, the central domain is always fixed. 
- How well does the model perform on translating different modalities (audio-text-image) given that this is one of the motivating example in the introduction? 
- Can the model work if the different domains contain images of different classes? For example consider translating between sketch-image-grayscale. For domain pair is image to sketch you have shoes as examples and for the image-grayscale you have faces. The central domain image contains both face and shoes. This setup satisfies your K-1 paired examples with overlap in the central domain. How well can the model now translate between the other domains? 
- How about domains where structure is not preserved such as image to cartoon style? Current text-to-image models can generate a bunch of these paired examples, and different artist styles could also be considered as different domains.

### Soundness
4

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
3

### Summary
They propose a new approach for multi-domain translation using a diffusion model. Unlike existing settings, where we assume the presence of many fully aligned tuples or unlabeled paired domains, they assume access to one central domain that has the paired samples with other domains. Then, they propose an approach that can achieve domain translation between unpaired domains by using the central domain as the bridge between the unpaired ones. Specifically, they propose an architecture that can translate between different image domains and training objectives. They conduct experiments on image translation and show that their approach outperforms existing diffusion based approaches with a large margin.

### Strengths
1. Their idea of using the central domain as the bridge between two unpaired domains sounds reasonable and novel. Their derivation of the objectives also looks reasonable. 

2. The proposed architecture to achieve a single translation model, which specifies the origin and the target domain, also looks reasonable and might be novel. 

3. This paper is easy to follow. Presentation is overall clear.

### Weaknesses
1. They repeat the discussion on translating image, text, and audio. However, they lack experiments on such cases. I wonder why they do not include such results. Also, my concern is that in such a modality translation setting, the proposed framework might not work well. They propose to apply a single network to exchange the domains of inputs. Since their experiments are conducted only in the translation of images, the effectiveness of swapping modality is not clear. 

2. Lacking comparison to non-diffusion techniques in experiments. Readers should be curious about the performance difference between the proposed one and non-diffusion methods. 

3. The novelty in the proposed method is not very clear to me. I guess the field of domain translation using a diffusion model is a popular topic. Since I am not familiar with this field, I could not figure out how novel the proposed method is, compared to the existing ones. They need a more careful discussion in explaining their approach in the introduction and method sections. 

4. What if they have a designated model for one pair of domain translation and combine two models to bridge the domains? A single model still outperforms such an approach in terms of FID?

### Questions
Please answer the concerns described in the weaknesses.

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
3

### Summary
This paper introduces Universal Multi-Domain Translation (UMDT), a novel and practical problem setting for translating between any pair of K domains using only K-1 paired datasets structured around a central domain. To address this, the authors propose Diffusion Router (DR), a unified diffusion model that conditions a single noise predictor on both source and target domain labels to handle all translations involving the central domain. The authors create new large-scale benchmarks to evaluate their method, demonstrating that iDR outperforms existing baselines and that dDR can achieve direct translation, albeit with a trade-off in performance versus computational efficiency.

### Strengths
- The formalization of UMDT is a significant contribution. It addresses a critical limitation of existing multi-domain translation methods, which either require impractical fully-aligned data or are limited to a hub-and-spoke translation model. This problem setting is highly practical and forward-looking, with clear real-world applications.
- The core idea of the Diffusion Router is an elegant and parameter-efficient solution for the standard central-to-non-central translation task. The strong performance of iDR against other baselines validates this architectural choice and establishes it as a powerful model in its own right.
- The theoretical contribution for enabling dDR is clever. The formulation of a variational upper bound (Eq. 9) to align the direct and indirect translation paths is a non-trivial approach to learning from "pseudo-supervision" generated by the model's own indirect capabilities.

### Weaknesses
- The central weakness of this paper is that the main technical contribution, dDR, consistently performs worse than the simpler, two-step iDR baseline on the very tasks it was designed to improve. This is evident across all three benchmarks, where iDR's FID scores are superior to dDR's.The paper motivates dDR by claiming it overcomes the computational expense and potential quality degradation of the two-step iDR process. However, the results show that dDR introduces quality degradation. This undermines its primary justification and calls into question whether the proposed learning strategy is genuinely effective.
- While dDR is computationally cheaper at inference, the paper does not adequately quantify this benefit against the measured drop in quality. The faster inference speed is valuable, but if it comes at the cost of a significant increase in FID score, iDR remains the superior method in terms of output quality. The paper would be much stronger if it provided a clear analysis of this trade-off, perhaps identifying scenarios where the speed of dDR is worth the quality sacrifice.
- The authors acknowledge that their tractable objective relies on a single-sample Monte Carlo estimate inside a logarithm, which introduces bias. This technical compromise is a likely source of dDR's underperformance. The work would be more convincing if it included analysis isolating the effect of this bias.

### Questions
See Weakness

### Soundness
3

### Presentation
2

### Contribution
3
