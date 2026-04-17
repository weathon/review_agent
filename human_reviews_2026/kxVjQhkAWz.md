# Dens3R: A Foundation Model for 3D Geometry Prediction

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Recent advances in dense 3D reconstruction have led to significant progress, yet achieving accurate unified geometric prediction remains a major challenge. Most existing methods are limited to predicting a single geometry quantity from input images. However, geometric quantities such as depth, surface normals, and point maps are inherently correlated, and estimating them in isolation often fails to ensure consistency, thereby limiting both accuracy and practical applicability. This motivates us to explore a unified framework that explicitly models the structural coupling among different geometric properties to enable joint regression. In this paper, we present Dens3R, a 3D foundation model designed for joint geometric dense prediction and adaptable to a wide range of downstream tasks. Dens3R adopts a two-stage training framework to progressively build a pointmap representation that is both generalizable and intrinsically invariant. Specifically, we design a lightweight shared encoder-decoder backbone and introduce position-interpolated rotary positional encoding to maintain expressive power while enhancing robustness to high-resolution inputs. By integrating image-pair matching features with intrinsic invariance modeling, Dens3R accurately regresses multiple geometric quantities such as surface normals and depth, achieving consistent geometry perception from single-view to multi-view inputs. Additionally, we propose a post-processing pipeline that supports geometrically consistent multi-view inference. Extensive experiments demonstrate the superior performance of Dens3R across various tasks and highlight its potential for broader applications.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Dens3R is a visual foundation model for dense 3D geometry prediction from unposed images. It jointly regresses consistent pointmaps, depth, and surface normals using a novel two-stage training framework. This approach builds an intrinsic-invariant pointmap by incorporating normals, ensuring high-quality, unified geometric perception across various tasks.

### Strengths
1. The model demonstrates SOTA performance across diverse benchmarks, including both indoor and outdoor scenes. Compelling visualizations showcase its superior accuracy and detail in depth and normal prediction, validating its effectiveness and robustness as a powerful geometry perception tool that consistently outperforms specialized methods.

2. A key strength is the well-motivated approach of jointly optimizing inherently correlated geometric quantities like depths and normals. Instead of predicting them in isolation, this unified framework explicitly models their structural coupling, ensuring geometric consistency.

3. The paper introduces an innovative two-stage training strategy. By leveraging surface normals—an intrinsic property—in the second stage, the model learns a representation robust to camera parameters and scale. This elegantly resolves monocular ambiguity and significantly boosts overall prediction accuracy and stability.

4. The paper is backed by comprehensive experiments, including thorough ablation studies that validate each key component. A particularly impressive highlight is its transferability to downstream tasks. The excellent performance on semantic segmentation (Fig. 8c) with a frozen backbone powerfully substantiates its claim as a versatile foundation model.

### Weaknesses
The network structure resembles DUST3R and may have sub-optimal performance when reconstructing long sequence inputs.

### Questions
See weaknesses.

### Soundness
4

### Presentation
4

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
This paper presents Dens3R, a regression-based 3D foundation model that unifies the prediction of multiple geometric quantities including depth, surface normals, and pointmaps from unposed image inputs. The method extends prior DUSt3R/MASt3R frameworks with a shared encoder-decoder backbone, a two-stage training strategy, and position-interpolated rotary positional encoding to improve robustness and multi-task consistency.

### Strengths
Well-engineered system that integrates multiple known effective components into a unified framework.

Demonstrates consistent performance improvements across several 3D geometry benchmarks (depth, normal, matching).

The paper is technically sound and clearly written, with solid experimental validation.

The incorporation of normal prediction and staged training improves empirical robustness and output consistency.

### Weaknesses
The core architectural and methodological ideas (multi-task learning, two-stage training, positional interpolation) are not novel and have been widely explored in prior work.

The backbone and representation design largely follow DUSt3R/MASt3R, with limited conceptual innovation.

The claim of being a “foundation model” is overstated, as the work focuses on supervised dense regression without demonstrating large-scale generalization or transfer capabilities.

### Questions
Can the authors clarify how much of the observed improvement comes specifically from the inclusion of surface normal supervision versus other training refinements?

How does the proposed “intrinsic-invariant pointmap” differ mathematically from the scale-invariant version used in Stage 1 — is it a new representation or mainly a training objective modification?

Since the positional interpolation for RoPE is borrowed from prior work, did the authors conduct ablations to quantify its contribution relative to baseline DUSt3R at higher resolutions?

### Soundness
3

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
The paper presents a geometric foundation model by designing a feed-forward framework based on the point map representation as in Dust3R, while also enabling a list of downstream tasks such as depth & normal prediction and semantic segmantation. A intrinsic invariant pointmap is proposed to enhance the geometric robustness during training by introducting a normal prediction head to recover the geometric details. The interpolatied RoPE is employed to handle multi-resolution visual input. Extensive qualitative and quantitative experimental results demonstrate the effectiveness of the paper.

### Strengths
(1) Overall, the paper is well presented and the qualitative images used for demonstrating the geometric details are impressive.

(2) Different from previous baseline methods such as Dust3R and Mast3R, the proposed framework is capable of predicting high-fidelity normal maps to recover the local geometric details, which is important for downstream computer graphics related applications.

### Weaknesses
(1) The motivation of designing a 'intrinsic invariant' pointmap is unclear to me. The pointmap is trained in a scale-invaraint manner by normalizing its geometric scale factors. Besides, the reason behind introducing the 'pointmap - normal' feature concatenatation can resolve the 'intrinsic-invaraint' ambiguity also remains unclear. Does it indicate that the normal could be further regularized by utilizing the information in the normalized pointmap?

(2) Although the positional-intropolated RoPE is technically reasonable, I think the novelty here is limited thus hard to be claimed as a technical innovation. By adapting different resolution images during training and inference, it is necessary to deal with the positional embedding in ViT with the flexible sequence length. So interpolating RoPE is a natural choice instead of a novelty. 

(3) One highlight of this method is the high-quality normal map prediction. However, the baseline methods used for comparison are trained on single view, leading to a unfair compairson. Since the paper follows the Dust3R's framework, a more practical baseline design is to compare the normal extracted from Dust3R's point map, which also uses a pointmap representation and trained on multiple views.

### Questions
On the normal prediction head, the authors mentioned to replace the 'one-to-many' mapping to 'one-to-one' mapping. Does this mean infer on the single image feature instead of using cross attention to aggregate the features from multiple views? 

Overall I think the paper proposed a technically feasible system, however some technical motivations and details look unclear to me. If the authors can address my concerns, I would consider to change my rating.

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
This paper proposes Dens3R, a model that predicts 3D geometry such as pointmaps, surface normals, depth, and image correspondences from unposed images. It uses a single transformer with shared weights and a two-stage training process: first learning scale-invariant pointmaps, then refining them into intrinsic-invariant ones using surface normal supervision. The model also adapts position-interpolated rotary positional encoding (RoPE) for high-resolution inputs and shows strong results in normal prediction and image matching.

### Strengths
* Addresses an important goal of predicting multiple 3D properties within one unified model.
* The two-stage training design is well-motivated and helps reduce monocular ambiguity.
* The position-interpolated RoPE is a simple and practical improvement for handling high-resolution data.
* Strong empirical results support the model’s effectiveness on normal estimation and matching tasks.

### Weaknesses
* Missing quantitative evaluation for depth prediction, which weakens the claim of a unified geometric model.
* Lacks ablation studies to verify the contribution of Stage 2, the normal loss, and RoPE interpolation.
* The description of the “Heads Training” process is unclear, especially regarding when and how the depth head is trained.
* Multi-view inference and computational cost are only briefly mentioned.

### Questions
1. Can the authors include standard depth metrics on datasets like NYUv2 or ScanNet?
2. Are the reported results from the unified model or after separate fine-tuning for each task?
3. How is the depth head trained, and is the matching loss still used in Stage 2?
4. Could the paper provide ablation results showing the effects of Stage 2, the normal loss, and RoPE interpolation?
5. What is the procedure and computational cost for multi-view inference?

### Soundness
3

### Presentation
3

### Contribution
3
