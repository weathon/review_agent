# SUBench: Benchmarking Spatial Understanding in Vision-Language Models

- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Recent advancements in vision-language models (VLMs) have shown remarkable success in general text-image retrieval. However, their ability to understand spatial relationships within images remains undertested. To mitigate this gap, in this paper, we introduce **SUBench**, a large-scale dataset of more than 50$k$ text-image pairs meticulously designed to evaluate a wide range of spatial relationships from real-world images. To curate the dataset, we designed an LLM-based framework to align human subjective descriptions and objective spatial relationships. In addition, unlike existing benchmarks, SUBench features a principled taxonomy of spatial concepts to ensure clarity and reduce ambiguity, alongside a scalable pipeline that systematically generates challenging hard negatives. Our experiments show that even state-of-the-art CLIP models struggle significantly on SUBench, revealing a critical blind spot in the spatial understanding capabilities of modern VLMs. Furthermore, we used the same approach to curate a set of training data and show that finetuning on this data not only improves performance significantly on SUBench but also enhances results on existing evaluation benchmarks. We will release the benchmark and believe SUBench will serve as a valuable resource to facilitate the development of more spatially-aware VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces SUBench, a benchmark to assess VLMs for image-text retrieval tasks that have a particular target spatial understanding. The authors also propose a new training method to mitigate the gaps in spatial understanding in image-text retrieval tasks.

### Strengths
* Formulation of the problem and the approach taken to mitigate it in a data-driven way
* Robust data curation procedure, which seems reasonably reproducible (especially the use of egocentric reference frames during data curation)
* New loss and the ablations presented to show their effectiveness

### Weaknesses
I think, in general, the paper could benefit quite a bit from important details that I feel should be included in the main text, or that are missing in the first place. I will include them in the next section.

### Questions
* Why not use a shape-optimized version of SigLIP [1, 2] or SigLIP-2 in Figure 1? Also, it seems like the comparison (Figure 1 and rest) is also not fair, as `TIPS-g/14-448 ` uses a resolution of 448x448 while the other models in the pool use 224x224.
* Could the authors provide a statistical analysis of the WebLI training dataset in terms of the spatial terms it includes in the captions? I think this will help with an even stronger motivation for the new dataset being introduced.
* As an extension of the previous question, could the authors try to re-caption the WebLI dataset (so that it includes spatial terms) and see if that alone could provide any gains? To ensure the model doesn't perform poorly on the non-spatial tasks, they could use the same contrastive loss on non-spatial original captions and their spatial-focused loss on the new synthetic captions.
* What is the extent to which a supervised fine-tuned model using the new loss works? Does it work beyond a certain number of spatial relationships and subjects in the captions? What's the performance ceiling there? These questions could help identify the limitations further.
*  Could the "Salience Principle" be experimentally validated? 
* In "Stage 3: Image Negation Pipeline (SUBench-T2I)", there is a use of an internal embedding model. Could the authors provide more details on that? The text reads as if it can be directly used to retrieve hard negative samples. If so, could the authors please clarify this more? 
* Consider including a statistical summary of the spatial-focus terms and relationships present in the new SUBench dataset.
* SA-1B doesn't have captions. Was it recaptioned? If so, the details seem to be missing.
* L337 -- how were those 600k training samples selected? Also, how were the 50k evaluation samples selected?
* It has been experimentally validated that the SigLIP pre-training strategy [3] is objectively better than CLIP. I wonder why the authors chose to use the CLIP pre-training strategy, despite that.
* What are the details of the spatial decoder used during the supervised fine-tuning process?
* Table 3 shows that just fine-tuning a pre-trained model isn't sufficient. In my opinion, that feels very restrictive. Could the authors run the following experiment? Train a LoRA [4] on a pre-trained model using the SUBench training dataset. Since users are free to choose the scale at which a trained LoRA should be applied, I feel controlling the LoRA scale could provide better trade-offs between the spatial and non-spatial performances than just naive fine-tuning.
* It's unclear how the negative pairs are used during training.
* When co-training, some details remain unclear. During co-training, both the WebLI and the SUBench training datasets are used. How is the spatial-focused loss applied to the WebLI samples?

**Misc**

* It would be helpful to the readers if a dataset scheme were included in the paper.
* Since the paper touches T2I datasets for spatial reasoning, the authors could consider citing SPRIGHT [5].

**References**

[1] Shape-optimized SigLIP: https://huggingface.co/google/siglip-so400m-patch14-384

[2] Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design; Alabdulmohsin et al.; 2023.

[3] Sigmoid Loss for Language Image Pre-Training; Zhai et al.; 2023.

[4] LoRA: Low-Rank Adaptation of Large Language Models; Hu et al.; 2021.

[5] Getting it Right: Improving Spatial Consistency in Text-to-Image Models; Chatterjee et al.; 2024.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces SUBench, a new large-scale dataset designed to evaluate understanding of spatial relationships in vision-language models. The dataset is curated via a LLM-based framework which involves creation of hard negative images and text descriptions, leading to the development of 50k parits. The evaluations show that state-of-the-art models like CLIP fall short on SUBench. Further, a spatial classification loss is introduced, which teaches the model to generate the right spatial "label" given an image-text pair. Results and ablations show that introducing this technique is effective and enables better performance over baseline fine-tuning techniques.

### Strengths
1. The paper is well written and easy to follow.
2. Adding a spatial classification loss is an interesting approach and seems to yield better performance than naive fine-tuning.
3. The scale of the benchmark is significantly larger than existing benchmarks such as VSR and OmniSpatial.

### Weaknesses
1. Questions around model generalization and results : The results in Table 3 show that the proposed model does well only on the subset presented in the paper (SU-Easy, SU-T2I) but other methods such as TIPS do better on well-accepted datasets such as COCO, DOCCI etc. This raises questions about the generalization of the method and if the model is overfit on the proposed dataset.

2. Regarding Section 3.1 --

i) Does mapping synonyms like “on”, “atop”, and “upon” to a single, unambiguous term not reduce the generalization of the benchmark and the trained model? 
ii) Performing "object grouping" further reduces the fine-grained nature of images -- while "a group of people" is logically correct -- spatial relationships also exist within that group, such as " a man with a blue hat and a woman with a blue shirt"
iii) Similarly, why filter out trivially true statements, especially for spatial relationships?  What are these trivially true statements that are removed? 

3. The internal embedding model for creation of SUBench-T2I will retrieve images which have the highest match - but these images can still be much farther than the original image - which will lead to data quality issues. Instead, why not use actual T2I models which now offer higher controllability during generation and has been used in prior work to create hard negatives (https://arxiv.org/abs/2411.02545) ? 

4. More fine-grained results would help paint a better picture of the shortcomings of existing models, such as which relationships do model fail on - horizontal, vertical, depth or adjacency etc.?

5.  As the authors acknowledged, the pipeline is completely LLM dependent -- having a user study that shows high alignment with LLM generated text would further strengthen claims.

### Questions
Please refer to weaknesses.

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
This paper introduces SUBench, a large-scale benchmark of 50k+ image-text pairs for evaluating VLMs’ spatial understanding. 

Current models excel generally but falter on fine-grained spatial relations. SUBench uses Gemini 2.5 Pro to align human-like descriptions with objective spatial relationships and generates hard negatives via a scalable, taxonomy-guided pipeline. 

The authors also propose a fine-tuning method adding a spatial decoder and classification loss to a standard VLM. Results show CLIP struggles on SUBench, while the proposed approach boosts performance and generalizes to existing benchmarks.

### Strengths
1. Identifies a Real Problem: The paper correctly highlights that current VLMs often possess superficial rather than genuine spatial understanding, which is a significant area for improvement in the field.

2. Introduction of Hard Negatives: The concept of explicitly generating both textual and visual hard negatives is a strong methodological contribution for creating a more challenging and diagnostic benchmark.

### Weaknesses
1. Data quality concerns: The LLM-based pipeline lacks quantitative validation—no human-vs-LLM agreement or human-annotated calibration subset is provided.

2. Limited methodological novelty: The approach—hard negatives and contrastive fine-tuning for spatial grounding—builds on prior work (e.g., SpatialVLM). The spatial decoder is a minor addition; the main contribution is the data pipeline, not architecture or learning objectives.

### Questions
1. How effective are LLMs as automated evaluators at filtering low-quality data? Can they quantitatively identify and correct subjective spatial biases (e.g., observer-centric descriptions)?

2. Does training with the spatial loss (L_spatial) hurt VLM performance on non-spatial tasks like attribute recognition or abstract semantic understanding?

### Soundness
3

### Presentation
3

### Contribution
2
