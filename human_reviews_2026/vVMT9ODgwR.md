# Oh-A-DINO: Understanding and Enhancing Attribute-Level Information in Self-Supervised Object-Centric Representations

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 6, 2

## Abstract
Object-centric understanding is fundamental to human vision and required for complex reasoning. Traditional methods define slot-based bottlenecks to learn object properties explicitly, while recent self-supervised vision models like DINO have shown emergent object understanding. We investigate the effectiveness of self-supervised representations from models such as CLIP, DINOv2 and DINOv3, as well as slot-based approaches, for multi-object instance retrieval, where specific objects must be faithfully identified in a scene. This scenario is increasingly relevant as pre-trained representations are deployed in downstream tasks, e.g., retrieval, manipulation, and goal-conditioned policies that demand fine-grained object understanding. Our findings reveal that self-supervised vision models and slot-based representations excel at identifying edge-derived geometry (shape, size) but fail to preserve non-geometric surface-level cues (colour, material, texture), which are critical for disambiguating objects when reasoning about or selecting them in such tasks. We show that learning an auxiliary latent space over segmented patches, where VAE regularisation enforces compact, disentangled object-centric representations, recovers these missing attributes. Augmenting the self-supervised methods with such latents improves retrieval across all attributes, suggesting a promising direction for making self-supervised representations more reliable in downstream tasks that require precise object-level reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper investigates how well self-supervised and object-centric visual representations preserve fine-grained object attributes necessary for distinguishing multiple objects in complex scenes. While large self-supervised models such as DINO, DINOv2, and CLIP exhibit emergent object understanding, the authors find that these representations mainly capture geometric properties (e.g., shape, size) but fail to retain surface-level cues like color, texture, and material.

To address this limitation, the paper proposes OH-A-DINO (Object-Aware DINO) that augments DINOv2 features with object latent vectors learned from segmented image patches. Experiments on CLEVR, CLEVRTex, and Stanford Cars show that OH-A-DINO improves multi-object instance retrieval, especially in color and material matching, indicating that object-centric latents is a promising direction for improving downstream tasks that require precise 
object-level understanding.

### Strengths
- **Novel yet Simple Method.** The most significant contribution of this paper lies in offering a novel perspective on integrating conventional self-supervised features with Object-Centric features. Previously, although Object-Centric models were favored for their characteristics, they were often criticized for their overly simplistic representation (using a few vectors to represent an image), which was considered insufficient for complex scenarios. The approach proposed in this paper, which involves using Object-Centric latent to enhance self-supervised features on the basis of self-supervised models, provides a solution that takes both aspects into account. The proposed OH-A-DINO introduces a clean, modular enhancement that does not require retraining the backbone. It elegantly combines global DINO features with locally learned VAE latents to recover missing attribute-level details.

- **Strong Empirical Results.** OH-A-DINO achieves large improvements in both single- and multi-attribute retrieval accuracy, particularly for color and material cues, and demonstrates consistent performance gains over all baselines.

### Weaknesses
- **Concerns about using PCA for segmenting.** According to my understanding, OH-A-DINO is divided into two functions: 1) extracting object masks, and 2) extracting object regions based on the masks, and using VAE to learn local features of each region to enhance global features. My concern lies in the former, that is, why does it use PCA plus threshold setting, a non-deep learning approach, to segment object patches instead of deep learning methods? For instance, since this paper is centered on Object-Centric, why not directly use Object-Centric methods to achieve segmentation? As far as I know, at least on CLEVR and ClevrTex, current Object-Centric methods have achieved nearly perfect segmentation results, which should be more reliable than PCA plus threshold and do not rely on manual parameter tuning. (Although this article mentions that OC models such as Slot Diffusion may lose some attribute information, it should not affect the application of OC models if they are only used to provide masks.)

- **Real world dataset choice.** Although the Stanford Car dataset was adopted as the real-world benchmark in the paper, a main concern is that the images in Stanford Car are all centered on a single vehicle as shown in Figure 5, which seems inconsistent with the "multi-object instance retrieval" task claimed in the paper. Using images with multiple objects, such as COCO which is commonly used in OCL, is obviously a better choice. Furthermore, similar to the previous weakness, in complex real-world scenarios like COCO, can the simple segmentation method of PCA effectively segment the approximate masks of objects? If not, would it be feasible to switch to the mainstream object-centric (OC) model for real-world scenarios, such as DINOSAUR + DINOv2? Even further, if we directly use Segment Anything to provide object masks, could this enhance DINOv2 in real-world scenarios to improve its retrieval capabilities?

### Questions
- Self-supervised models like DINO are typically trained based on the semantic consistency after geometric and color transformations of images, for instance, the features of an image after color transformation should be similar to those of the original image. Is this the reason why the models are insensitive to color, material, and texture? 

- What causes the Slot-based model to lose object attribute information? Logically speaking, the goal of SlotDiffusion (or other Slot-based model that reconstructs RGB pixels) is to generate the original image, so all the information in the image should be preserved in its slots, and information should not be lost.

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
3

### Summary
The paper investigates the attribute-level information encoded in object-centric representations from models such as CLIP, DINO, and SlotDiffusion. The authors find that the object embeddings produced by these models cannot be directly used for multi-object instance retrieval, a task that aims to retrieve objects sharing the same attributes as a given query object using cosine similarity. To address this limitation, the authors propose a two-step approach: first, they apply PCA to DINO features to segment objects in the scene; then, all DINO features of patches belonging to an object are fed into a VAE to learn disentangled features for each patch. The resulting feature is concatenated with the original DINO feature to obtain the final representation. Experimental results show that this enhanced feature improves multi-object instance retrieval performance on CLEVR and CLEVRTex, and captures color information more accurately than CLIP and DINO, as demonstrated on the Stanford Cars dataset.

### Strengths
The paper is well written and easy to read.

### Weaknesses
- My main concern about this paper is its unreasonable setting and metric: One major argument made in the paper is that, given a query object, cosine similarities on the **raw** representations produced by baseline models cannot retrieve objects with the same attributes such as color, material, or shape. The authors interpret this as evidence that the model fails to preserve these non-geometric, surface-level cues. This interpretation is not accurate: low cosine similarity does not imply that the information is missing from the representation—it may simply be encoded in a way that is not directly reflected in direct pairwise distances. For example, representations produced by slot-based models such as SlotDiffusion can reconstruct the original image with high fidelity, indicating that attribute-level information is well preserved. Moreover, numerous experiments (see SlotFormer) on VQA have demonstrated the effectiveness of these representations on downstream tasks where surface-level attributes are also relevant. Therefore, the idea of enforcing attribute-level similarity lacks motivation. On the other hand, I would expect representations produced by DINO and CLIP to lack certain information because they are not trained with a reconstruction objective.
- In addition to the lack of motivation for the proposed multi-object retrieval setting, the paper also offers limited technical contribution. The proposed method relies on simple heuristics of PCA to segment objects in the scene and then leverages a β-VAE to learn disentangled features for each patch. This is essentially a combination of well-known techniques.

### Questions
- What is the motivation for using cosine similarity in the multi-object retrieval setting? Why not just train a classifier to predict the attributes?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies fine-grained surface attributes to be essential for multi-object instance discrimination, which is lacked by SSL models like DINO. The authors propose Oh-A-DINO (Object-Aware-DINO), which augments DINOv2 representations with object-centric VAE latents trained on segmented image patches to improve this.
PCA-based segmentation extracts object regions from DINOv2 embeddings, followed by VAE training on patches to capture fine-grained attributes like color and material.
Empirically, Oh-A-DINO significantly improves previous methods across CLEVR, CLEVRTex, and Stanford Cars.

### Strengths
- The paper identified an interesting limitation in current SSL models on object-centric benchmarks that they struggle with surface-level attributes, making this a valuable research direction.
- The combination of global SSL features with local VAE latents provides a principled way to preserve both geometric and surface attribute information.
- The evaluation is thorough, covering both synthetic (CLEVR, CLEVRTex) and real-world (Stanford Cars) datasets. They effectively measure the model's ability to distinguish objects based on fine-grained attributes. The performance is strong compared to prior methods. Ablation studies properly isolate the contribution of different components.
- The delivery of the paper is clear, and I find it easy to follow.

### Weaknesses
- The evaluation focuses primarily on color, material, and basic geometric attributes. More complex attributes like texture patterns, semantic relationships, or fine-grained visual details remain unexplored. The generalizability to broader attribute types would be interesting.
- While CLEVR provides controlled evaluation, real-world evaluation is limited to Stanford Cars. More diverse real-world datasets spanning different domains would strengthen the claims.

### Questions
- Can the authors provide more failure case analysis or discussion of when the method might not work well? Understanding the boundaries and limitations would improve the contribution's practical value.
- How would SSL methods that employ reconstruction-based losses (eg, iBOT, SigLIP2, AM-RADIOv2.5) perform? How would recent advances in SSL (eg, Perception Encoder) and multimodal LLMs (eg, Qwen3-VL) perform on these benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper study how self-supervised models and slot-based methods understand objects in complex scenes. It find that models like CLIP and DINO good at shape and size, but not so good with color or texture. The authors propose a latent space with VAE regularization to fix this, which improve retrieval results. The idea is interesting and show potential in especially multi-object retrieval tasks.

### Strengths
- originality: 2/5,
- quality: 2/5,
- clarity: 4/5,
- significance: 2/5. Limited to simple image retrieval tasks.

### Weaknesses
W1
---
Table. 1.
The OCL baseline SlotDiffusion is relatively weak. In fact the object representation quality is highly affected by the object discovery (unsupervised object segmentation) accuracy. There are more advanced OCL methods, like SPOT, DIAS and SmoothSA, which should also be included as stronger OCL baselines.


W2
---
Line 201.
> collected from **a batch of t images** and apply PCA

This means an online induction based on multiple $t$ input image samples -- What if there is only one input available during inference?

This design could also be a bottleneck for real-world complex images like ones from COCO or ImageNet, where the borderline between foreground and background can be quite vague.



W3
---
Line 223,
> This yields a set of **object-level** latents

To put it in a rigid way, these are still **patch-level** latents with object/foreground mask augmentation, which is obtained in Section 3.2 (ii) "Refining object consistency" operation.



W4
---
Line 228,
> (CLS token in **DINOs** case)

"DINOs" should be "DINO's".



W5
---
Line 232,

> Retrieval is then performed by cosine similarity between v and v′ from query and candidate images
It is unclear your performance boost comes from the concat of global features (Figure 2, Line 167 and 168) or not. So for fair comparison, OCL representations should also be concatenated with the CLS token from DINO as the strong baseline.


W6
---
Line 234 or Appendix A:
> for each query patch vi we retrieve the patch with the highest cosine similarity

> $s_i^{max} = \max_j S_{ij}$

Intuitively, the max matching should be Hungarian matching. Otherwise, there might be multiple patches $i$ matched to the same $j$.


W7
---
Line 447:
> 6 STANFORD CARS: REAL-WORLD INSTANCE RETRIEVAL

The section label "6" should be "5.4", parallel to Section 5.2, as results on synthetic and real-world datasets respectively.


References
---
- SPOT: Self-Training with Patch-Order Permutation for Object-Centric Learning with Autoregressive Transformers
- DIAS: Slot Attention with Re-Initialization and Self-Distillation
- SmoothSA: Smoothing Slot Attention Iterations and Recurrences

### Questions
N/A.

### Soundness
2

### Presentation
3

### Contribution
2
