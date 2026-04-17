# Jodi: Unification of Visual Generation and Understanding via Joint Modeling

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Visual generation and understanding are two deeply interconnected aspects of human intelligence, yet they have been traditionally treated as separate tasks in machine learning. In this paper, we propose Jodi, a diffusion framework that unifies visual generation and understanding by jointly modeling the image domain and multiple label domains. Specifically, Jodi is built upon a linear diffusion transformer along with a Role-Switch mechanism, which enables it to perform three particular types of tasks: (1) joint generation, where the model simultaneously generates images and multiple labels; (2) controllable generation, where images are generated conditioned on any combination of labels; and (3) image perception, where multiple labels can be predicted at once from a given image. Furthermore, we present the Joint-1.6M dataset, which contains 200K high-quality images collected from public sources, automatic labels for 7 visual domains, and LLM-generated captions. Extensive experiments demonstrate that our Jodi excels in generation tasks and performs competitively in understanding tasks. Besides, Jodi exhibits strong extensibility to new visual domains. Codes, data, and model weights will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a joint modeling framework to train an all-in-one generative model that can do joint generation of images and caption, conditional image generation, and label prediction.

### Strengths
- I like the probabilistic perspective on joint modeling p(x, y) = p(x | y) p(y) = p(y | x) p(x). Although simple, it provides an intuitive motivation for why you would want to pursue such a method.
- Visual results clearly show that your method works.
- The paper is well-written and easy to follow.

### Weaknesses
see questions

### Questions
- Why would somebody want to use this instead of just chaining a few off-the-shelf models to do the target tasks? I get that there's some elegance to having it all in one model, but the pretrained models are already so powerful, that I'm skeptical that somebody would use this approach in practice. Furthermore, the results show that the specialist models often beat Jodi (although Jodi is better than the other joint models).
- What is the methodological novelty?
- Have you considered more modern fast attention mechanisms? [1] Or why not only train a few conditions at a time to get around the quadratic scaling?

[1] Guo, Han, et al. "Log-linear attention." arXiv preprint arXiv:2506.04761 (2025).

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
3

### Summary
The paper proposes Jodi, a unified diffusion model that can jointly generate images and their corresponding conditions (depth, mask, etc). To achieve this, Jodi considers a unified input format (role switching) and utilizes domain-invariant positional embeddings to optimize flow-based DiT model. Extensive experiments on various benchmarks demonstrate their methods efficacy in unified generation.

### Strengths
I like this paper due to the following strengths:

(1) Flexible Control: The framework naturally supports complex, multi-modal conditioning, offering unparalleled flexibility for creative applications.

(2) Enhanced Generalization: By forcing the model to simultaneously learn the generative process  and the analytical structure, JODI is likely to develop a richer, more robust latent representation of visual concepts.

### Weaknesses
(1) Lack of T2I evaluation. As demonstrated in Fig. 11, the generated image is the bridge between text and other condisions. Therefore, I would like to see some comparison with T2I methods on GenEval or other T2I benchmarks.

(2) The proposed method achieves condition to image by modeling the joint distribution of various conditions. However, some generation tasks (e.g., depth to image [a]) could be evaluated in a more precise manner. The authors should compare, or at least, discuss with such related works. 

(3) Minor: In Tab. 1-openpose, controlnet seems to achieve better result than yours in terms of LPIPS.


[a] 3dis: Depth-driven decoupled instance synthesis for text-to-image generation. In ICLR'25.
[b] 3DIS-FLUX: simple and efficient multi-instance generation with DiT rendering. In Arxiv.

### Questions
Please refer to "weaknesses"

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Jodi, a unified diffusion framework designed to jointly model the distribution of images ($x$) and multiple label domains ($y^n$), such as depth, normal, and segmentation. The goal is to unify visual generation and understanding, which are typically treated as separate tasks. The core technical contribution is a "Role-Switch" mechanism, where each domain is randomly assigned as a generation target [G], a condition [C], or ignored [X] during training. This principled approach, based on joint probability modeling, allows the single model to perform three distinct tasks: joint generation ($p(x, y, ...)$), controllable generation ($p(x|y, ...)$), and multi-label image perception ($p(y, ...|x)$). The framework is built on an efficient linear diffusion transformer to handle the computational load of many domains and is trained on a newly curated "Joint-1.6M" dataset.

### Strengths
1. Principled and Elegant Framework: The core idea of unifying $p(x|y)$ and $p(y|x)$ by modeling the joint distribution $p(x, y)$ is statistically elegant. The "Role-Switch" mechanism is a clever and direct implementation of this principle, forcing the model to learn a wide range of conditional and marginal distributions within one architecture.
2. The authors made smart architectural choices. The use of a linear diffusion transformer (Sana) correctly identifies and solves the $\mathcal{O}(M^2)$ computational bottleneck of multi-domain ($M$) attention. The "masked linear attention" and "domain-invariant positional embeddings" are solid supporting contributions that are well-motivated and essential for the framework to function.
3. The paper introduces the "Joint-1.6M" dataset (200K images + 7 predicted labels) and aggregates 90K images with ground-truth labels. This is a valuable contribution to the community that will enable future research in joint visual modeling.

### Weaknesses
The model's "unification" comes at a significant cost to specialist performance. Though perform better than omni-models on edge detection and normal estimation tasks, Jodi performs noticeably worse than SOTA specialist models on albedo estimation and depth estimation. For example, in depth estimation (Table 2), Jodi achieves 10.1 AbsRel on NYUv2, while the specialist Lotus-D achieves 5.1. In albedo estimation (Table 4), Jodi gets 15.5 PSNR, while the specialist RGB2X gets 20.6. The model excels at generality but does not "excel in... understanding tasks" as claimed.

### Questions
1. The perception results (especially Tables 2, 4) are lower than some SOTA unified models and specialist models significantly. Is this performance gap an unavoidable cost of the "jack of all trades" unification, or do the authors see a clear path for Jodi to actually surpass some specialist models in understanding tasks?
2. What was the sampling strategy for the [G], [C], and [X] roles? Were they sampled uniformly at random for all domains? Or was a curriculum used (e.g., more [C] roles early in training) to stabilize the learning of this complex joint distribution?

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
4

### Summary
This paper introduces Jodi, a diffusion-based model that tries to unify image generation and image understanding by jointly modeling images and multiple label modalities. With the proposed “Role-Switch” mechanism, the same model can generate images with labels, generate images conditioned on any combination of labels, or predict labels from an input image. It uses a linear diffusion transformer and domain-invariant positional embeddings to keep computation manageable and maintain consistency across domains. The authors also provide a new dataset covering 8 visual domains. Experiments show good performance on both generation and perception tasks and strong scalability. Overall, the idea is intuitive and nicely executed, and the results suggest that a unified approach like this is becoming useful.

### Strengths
* The role switch mechanism:  by randomly switching each domain between being generated, used as conditioning, or ignored, the model learns all the key distributions for both image generation and perception at once -- giving one network the flexibility to do many things.

* Paper is well motivated, and generally well written and easy to follow.

* The proposed method uses shared positional embeddings across domains (+ small role tags) so that the model knows which pixels align spatially, making it easier to keep different visual modalities consistent with each other.

* The proposed Joint generation could be very valuable in many downstream applications, such as artistic manipulation or asset usage in the traditional frameworks. 
  
* Experimental results validate the proposed method convincingly.

### Weaknesses
* “Extensibility to more domains” is claimed, but it’s unclear how much effort is needed for an entirely new modality.

* Baselines for multimodal generation/understanding might not share the same supervision or label availability. 
This needs to be clarified in the paper. 

* Randomly switching roles during training might make optimization harder or convergence slower, especially as domains increase.

* Even with linear attention, jointly modeling 8+ domains could still be memory-intensive and slow for large resolutions.

* The “multiple label domains” are pseudo labels (depth, normals, etc.), so errors/noise in these domains may propagate through the unified model.

### Questions
* How well does the method scale to more detailed label domains (e.g., full-class/instance segmentation or dense keypoints) without redesign?

* What are the failure cases on real or diverse datasets, especially with human subjects? Although the limitations are discussed, the failure cases are not provided (neither in the main paper nor in the supplementary).

* Can the authors provide per-domain trade-offs -- do any tasks degrade notably compared to specialist methods?
 The results currently reported appear not to be on fair ground comparisons.

### Soundness
4

### Presentation
3

### Contribution
4
