# Multimodal Dataset Distillation Made Simple by Prototype-Guided Data Synthesis

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6, 6

## Abstract
Recent advances in multimodal learning have achieved remarkable success across diverse vision–language tasks. However, such progress heavily relies on large-scale image–text datasets, making training costly and inefficient. 
Prior efforts in dataset filtering and pruning attempt to mitigate this issue, but still require relatively large subsets to maintain performance and fail under very small subsets.
Dataset distillation offers a promising alternative, yet existing multimodal dataset distillation methods require full-dataset training and joint optimization of image pixels and text features, making them architecture-dependent and limiting cross-architecture generalization.
To overcome this, we propose a learning-free dataset distillation framework that eliminates the need for large-scale training and optimization while enhancing generalization across architectures. 
Our method uses CLIP to extract aligned image–text embeddings, obtains prototypes, and employs an unCLIP decoder to synthesize images, enabling efficient and scalable multimodal dataset distillation.
Extensive experiments demonstrate that our approach consistently outperforms optimization-based dataset distillation and subset selection methods, achieving state-of-the-art cross-architecture generalization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a prototype-guided data synthesis framework for multimodal dataset distillation that removes the need for large-scale training and optimization. It achieves efficient and scalable multimodal data generation, outperforming previous optimization-based methods while improving cross-architecture generalization.

### Strengths
1. The proposed method is learning-free, making it simple, efficient, and architecture-independent.
2. The framework works well across different architectures, showing cross-architecture generalization.
3. It synthesizes semantically aligned image–text pairs, improving multimodal learning performance.

### Weaknesses
1. Will averaging embeddings to construct prototypes cause degraded representations?

2. The method uses the unCLIP decoder to reconstruct images from CLIP embeddings/text. The capibility of the decoder could be essential. Do authors observe failure cases and how to process them? 

3. Besides Flickr30K and MS-COCO, have authors considered other data?

4. While the major classes are well represented in the proposed prototypes, have authors considered the performance on rare or long-tail classes?

### Questions
Please refer to 'Weaknesses'

1. Will averaging embeddings to construct prototypes cause degraded representations?

2. The method uses the unCLIP decoder to reconstruct images from CLIP embeddings/text. The capibility of the decoder could be essential. Do authors observe failure cases and how to process them? 

3. Besides Flickr30K and MS-COCO, have authors considered other data?

4. While the major classes are well represented in the proposed prototypes, have authors considered the performance on rare or long-tail classes?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a learning-free framework for multimodal dataset distillation, which avoids complex optimization by leveraging pre-trained models such as CLIP and unCLIP generation models, with new designs. The experiments validate the effectiveness.

### Strengths
1. The intuition is clear and novel. Instead of using costly optimization-based approaches for dataset distillation, the paper propose to reuse pre-trained models such as CLIP, unCLIP to derive the underlying multimodal embeddings, and use that for later dataset generation. This provides an inspiring direction  for multimodal dataset distillation.
2. The results show that PDS outperforms other baselines that are optimization-based, which shows good cross-architecture generalisation ability.

### Weaknesses
1. The "learning-free" claim needs qualification. While the distillation process itself avoids optimization, the framework is critically dependent on specific pre-trained models: CLIP and an unCLIP decoder. This reliance on a specialized generative model capable of conditioning on CLIP image embeddings may hinder the method's general usability.
2. The model may loss ability to use newer embeddings since it requires an generative model that can accept this embedding as as condition. This constraint may limit the model's performance.

### Questions
1. The results show a large effect of the distilled size ($M$) on performance (comparing $M=100$ and $M=300$). Can the authors provide a broader analysis of performance scaling as $M$ increases? More importantly, what practical guidance should be used to select a suitable $M$ for a new distillation problem?

2. The method relies heavily on pre-trained models (CLIP and unCLIP). It is unclear how much of the success is due to the PDS framework versus the prior knowledge embedded in these large foundation models. Could the authors comment on this dependency?

3. Could the proposed Learning-Free PDS approach be integrated with existing optimization-based methods? Combining PDS's efficient initialization with the precision of optimization could lead to a new state-of-the-art in distilled dataset quality.

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
4

### Summary
The paper proposes an approach to dataset generation in order to train multi-modal (text and image) models on a more compact amount of data, thus requiring less compute and energy. Different from existing distillation methods based on the optimization paths of the models, this paper looks at the latent space of an existing multimodal model CLIP and generates paired (image-text) samples directly from it after clustering. The experimental evaluation shows advancement over existing methods of dataset distillation.

### Strengths
The work proposes a strong case for need of more compact and easier produced datasets for multi-model training. The provided experiments are extensive and include possible ablation studies.

### Weaknesses
I am wondering if it is possible to prove in any way, that the learning result from such distilled\synthesized dataset is similar to the original learning result. The resulting performance is only a weak sign of similarity of the trained models.

Also, I find it contradictory, that the motivation for the proposed method is based on the need for large-scale dataset existing beforehand, while this method as well requires trained CLIP-model, which means that this large scale dataset was already used as well.

Finally, I find the argument about generating samples that are sufficiently different from the copyright prohibited images rather weak. First of all, it does not remove the copyright prohibited images from the original CLIP model, second there were no strict metrics provided proving that it is indeed the case that images would sufficiently differ compared to other methods (except for one Figure).

Minor:

- Referncing to tables and figures in Section3 is a bit cryptic - maybe more details on how exactly table proves the point would be helpful for the reader, who does not have to go to the tables beforehand then.

### Questions
1 - Why sampling inputs from the latent space clusters will lead to reduction in the data required for training a good model? Current empirical results show that the difference between performance on the full dataset and the synthesized one is significant. Is it even possible to catch up with the original performance in this way?

2 - How one can guarantee that generated samples indeed will be significantly different from the copyright data, if that one was used in training CLIP mode?

3 - How do you select amount of clusters for generating "prototypes"?

4 - Why not to perform clustering right away in the joint space of representations, but try to map clusters on each other later?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes the Prototype-Guided Data Synthesis (PDS) framework, a learning-free dataset distillation framework that eliminates large-scale training and optimization based techniques. The framework uses a pre-trained CLIP multimodal model to obtain image and text embeddings, the embeddings are pruned based on a similarity score and clustered to capture the dataset's semantic diversity through a Mini-Batch K-Means algorithm and matched using the Hungarian algorithm. An unCLIP decoder model is then used on the CLIP image embeddings to generate a synthetic image by retrieving the most similar caption most similar to the text prototype from the training set. The PDS framework outperforms multimodal dataset distillation baseline such as TESLA-VL and LoRS, also the framework outperforms dataset subset selection method such as Herding.

### Strengths
The paper proposes a novel dataset distillation method based on a pre-trained CLIP encoder and unCLIP decoder to extract image embeddings. These extract embeddings are then forwarded in an unCLIP decoder to generate a distilled dataset. The paper's methodological presentation and its contributions are well articulated in the text. Empirically, PDS achieves state-of-the-art performance compared with dataset subset selection and multimodal dataset distillation baselines, demonstrating the advantages of the proposed approach.

### Weaknesses
- The paper's presented  PDS framework is evaluated only with ViT-L/14 CLIP encoders. 
- The code to replicate results is not yet released (even anonymously).
- Which unCLIP decoder is used in the PDS framework? Currently, the authors provide a citation to Ho & Salimans (2022) and the guidance scale and sampling step hyperparameters, without specifying the model architecture used.

### Questions
- Can the PDS framework improvements over state-of-the-art methods be reproduced by using different CLIP variants? For example, clip-ViT-B-32 or clip-ViT-B-16? 
- Are the authors planning to release the code to reproduce the results reported in the paper? Please include scripts to reproduce all tables/figures. 
- Which unCLIP decoder is used? How sensitive are the results to the decoder's guidance scale and sampling step hyperparameters? 
- Could the sensitivity of PDS to the similarity-pruning threshold, the clustering seed, and the number of desired distilled samples (M) be assessed, with ablation studies on these factors?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The aim of this paper is to summarize big multimodal datasets in smaller sizes datasets of hundreds of pairs while losing as little performance as possible. This idea of dataset distillation is old, but has not been explored for multimodal datasets made of pairs. Indeed, these datasets are used as a bridge between two domains and the idea is to make this bridge very light and portable.

### Strengths
- The core concept is both timely and important. The paper correctly identifies that summarizing a link between domains is a different and more nuanced problem than summarizing a single domain. This is a valuable contribution to the field.
- The procedure described is logical and well-conceived. The idea of selecting only the overlapping pairs within matched clusters to form prototypes is a particularly clever mechanism for strengthening cross-modal alignment.

### Weaknesses
- A thing I found less convincing is what to do with these distilled, let’s say, 300 pairs. From what I understood it is not possible to use them to train a CLIP-like model from scratch (it would have been cool…).
- The application suggested—to use these pairs to link a vision space with a text space through a fine-tuned linear layer—is weaker than it seems. The two spaces are often already very aligned (see e.g., e.g., Huh et al., 2024, "The Platonic Representation Hypothesis"), so it's reasonable to expect they do not need much to be paired. Consequently, the authors should compare their method not only to other distillation techniques but also to methods specifically for multimodal latent space alignment, such as the relative representation alignment in ASIF (Norelli et al., 2023), that already uses far less pairs than usual multimodal datasets.

A cool application of the method I see is to make techniques like ASIF work with even less data, that is: hundreds of pairs. That would be very cool; the reduced dataset would also foster interpretability besides speed and privacy.

- The comparison with other data distillation techniques does not seem entirely fair, since this method saves a text embedding and not a text for each pair, which is significantly more information. In this sense, it would have been good to show performance also of the method selecting the closest caption to the embedding, especially since the authors already retrieve this exact caption to condition the unCLIP image generator.

### Questions
- Just a curiosity, have you tried to use unCLIP on the text embedding used in place of the image embedding? I would bet that performance is worse, but maybe not, I am not confident. Perhaps an average of the two is even better?
- One thing that surprised me is that you had clusters with no overlapping pairs at all. How many clusters? Also, I was expecting you to discard them since you cannot do the average, but instead you keep the centroid. Perhaps you want to motivate this choice more, since my first thought as a reader was that you would have discarded them.

---

Overall, I think the topic is important and the idea is interesting, but you could make it better shine! I will set my score at 6 with the hope of raising it.

### Soundness
3

### Presentation
2

### Contribution
3
