# ChainGeo: Enabling Effective Geometric Reasoning in Small VLMs through Interleaved Visual-Text Chains

- Avg Score: 3.00
- Decision: Reject
- Scores: 0, 6, 2, 4

## Abstract
Solving geometric problems requires linking visual perception with symbolic reasoning. However, small Vision-Language Models (VLMs) often fail to keep this connection. We introduce ChainGeo, a novel framework that enables small VLMs to perform complex geometric reasoning through interleaved visual-text chains. Our approach represents geometric elements as specialized tokens (e.g., [Point A], [Line AB]) that maintain explicit grounding in diagram regions, and act as bridges between visual features and symbolic reasoning. We further propose step-level consistency distillation to transfer complete reasoning processes from large teacher models, enforcing visual-textual coherence at each step. Experiments on GeoQA+ (72.1%), Geometry3K (64.7%), and We-Math (68.2%) show that our 2.7B model achieves performance comparable to GPT-4V while providing interpretable, grounded reasoning chains. In human evaluations, our model grounded visual references more accurately (75.3%) and reduced hallucinations by 36.6% compared with text-only baselines.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper propose to solve geometric problems through symbolic reasoning in smale VLM, thus build ChainGeo framework that enables complex geometric reasoning. The central idea is the specialized geometric tokens to represent symbolic elements, and consistency distillation from large VLM. Extensive experiments were conducted on three datasets with improved performance.

### Strengths
The technical contributions are very limited and the experimental settings are seriously flawed.

### Weaknesses
Presentation：

--The writing is poor and difficult to follow.

Method：

--Technical contribution is poor; 

--Limited novelty of representing geometric symbols as special tokens and distilling similarities.

Experiment: 

--The compared large VLMs (e.g. GPT 4V, Gemini1.5) are all outdated, making the results unreliable. 

--The compared specialized geometric solvers are all from before 2022.

--The authors only compare a limited number of early works, making it difficult to assess its contribution.

### Questions
--If the large VLMs possesses a strong geometric reasoning ability, what are the motivations and application scenarios for distilling down to a smaller model?

--Why not conduct experiments and comparisons based on methods from recent years?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ChainGeo, a framework designed to enhance complex geometric reasoning in small Vision-Language Models (1~3B parameters) by addressing the disconnect between visual perception and symbolic reasoning. It uses interleaved visual-text chains—specialized tokens grounded in diagram regions—to bridge visual features and logical reasoning, paired with step-level consistency distillation that transfers structured reasoning processes from large teacher models. ChainGeo enables small vision-language models to achieve geometric reasoning performance on par with large models such as GPT-4V, while being significantly more computationally efficient with inference speeds 15 times faster and producing reasoning chains that are more interpretable and verifiable. The approach is rigorously validated across multiple benchmarks, outperforming both general-purpose small VLMs and specialized geometry solvers.

### Strengths
1. The paper innovatively introduces interleaved visual-text reasoning chains into small-scale vision-language models (1~3B parameters), explicitly anchoring symbolic reasoning to visual elements throughout the inference process. Unlike traditional text-only chain-of-thought (CoT) approaches, this representation systematically addresses— for the first time—the problem of "visual amnesia" in geometric reasoning with small models.
2. The approach converts geometric elements (points, lines, angles, shapes) into explicit visual tokens and binds them to corresponding image regions, providing an interpretable interface for vision-symbol integration. This token-level grounding can be viewed as a novel multimodal intermediate representation, offering methodological innovation.
3. ChainGeo provides a systematic implementation pathway that enables vision-language models to explicitly align visual elements with symbolic reasoning steps, laying a foundation for future research in areas such as 3D geometry and physical scene understanding.

### Weaknesses
1. The distillation phase relies entirely on GPT-5 to generate reasoning chains, yet the paper does not disclose details about the teacher prompts, chain selection criteria, or quality control procedures. This component contributes significantly to the final results, but the lack of implementation details hinders reproducibility. It is encouraged to provide the teacher prompt templates and clear criteria for reasoning chain selection.
2. Although the authors emphasize that ChainGeo is a general-purpose framework, all experiments are conducted using a single architecture (Phi-2 + CLIP ViT-L/14). The absence of validation across different base models makes it difficult to assess the method’s transferability across varying language model scales or architectures. Reproducing and comparing results on other lightweight architectures—such as Qwen-VL-2B, LLaVA-1.5, or MiniGPT-4—would provide stronger evidence supporting ChainGeo’s claim as a broadly applicable training methodology.
3. The paper lacks sufficient training details: although it mentions losses such as "logical consistency loss" and "coherence loss," it does not provide explicit equations and hyperparameter choices in the appendix or methodology section to enable readers to verify the feasibility and reproducibility of the approach.

### Questions
1. Is the visual token generation process generalizable? The paper mentions that visual tokens are derived from a specially trained geometric element detector based on Faster R-CNN, but it is unclear whether this detector was tailored specifically to a particular dataset (e.g., GeoQA+). If applied to other types of geometric diagrams—such as hand-drawn textbook problems—would the detector maintain its performance?
2. For the right subfigure in Figure 2, it is recommended to include a more intuitive visualization that clearly illustrates the positive correlation between "attention retention" and "visual grounding accuracy," thereby strengthening the claim that ChainGeo effectively mitigates "visual amnesia" and ensures reliable visual grounding.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces ChainGeo, a framework designed to enable small Vision-Language Models (1–3B parameters) to perform geometric reasoning through interleaved visual–text chains. The core idea is to represent geometric primitives (e.g., points, lines) as specialized tokens that explicitly link symbolic reasoning steps with corresponding regions in the diagram.

### Strengths
The proposed token-level grounding approach offers a computationally efficient way to encode geometric primitives, suitable for smaller models.

They conducted experiments and evaluated their model on the public benchmarks, suggesting enhanced alignment between visual and symbolic reasoning.

### Weaknesses
The contribution and novelty of this paper are limited. The observation that MLLMs fail to perceive diagrams correctly has been extensively explored in prior works such as MathVerse, MathVista, Primitive Vision, and MAVIS, which already provide detailed analyses of perception versus reasoning errors. Similarly, the discussion of visual information loss and attention misalignment is not a new insight—earlier studies (e.g., Through the Magnifying Glass: Adaptive Perception Magnification for Hallucination-Free VLM Decoding) have investigated similar issues and proposed mitigation strategies. A more comprehensive literature review would help situate this work within the existing research landscape.

Regarding the proposed solution, the use of a detector-based mechanism to localize geometric elements is not novel and closely resembles approaches in Primitive Vision (Shan Zhang et. al., Primitive Vision: Improving Diagram Understanding in MLLMs, ICML 2025 ). The authors of Primitive Vision explicitly acknowledged the uncertainty of detector outputs and therefore adopted selective feature maps as soft proxies for geometric regions, reducing error propagation from imperfect detections. But in this work, authors do not clearly explain how they ensure the reliability of detection results, especially since training on synthetic datasets often limits generalization to real-world diagrams. The paper also lacks a detailed description of the synthetic data generation process and its potential biases.


The proposed interleaved visual-text reasoning framework also explored by previous works, such as MINT-CoT: Enabling Interleaved Visual Tokens in Mathematical Chain-of-Thought Reasoning, which addresses similar geometric reasoning challenges. Finally, the method appears restricted to planar geometric problems and does not generalize to other mathematical domains (e.g., graph-based or symbolic reasoning). The analysis of model responses remains shallow, with limited qualitative or interpretability discussion.

### Questions
The proposed method appears tailored to 2D geometric problems. How would ChainGeo extend to other domains—such as graph-based reasoning, algebraic problem solving, or multi-step spatial reasoning—where visual grounding is less explicitly defined?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ChainGeo, a framework that enables small Vision-Language Models (VLMs, 1–3B parameters) to perform complex geometric reasoning through interleaved visual-text chains. The core idea is to represent geometric elements (e.g., [Point A], [Line AB]) as specialized tokens that are explicitly grounded in regions of the input diagram, serving as bridges between visual perception and symbolic reasoning. The authors further introduce step-level consistency distillation, which transfers full reasoning processes from a large teacher model while enforcing visual-textual coherence at every step. Experiments on GeoQA+ (72.1%), Geometry3K (64.7%), and We-Math (68.2%) show that their 2.7B model achieves performance comparable to GPT-4V, while producing interpretable and grounded reasoning chains.

### Strengths
- Precise problem framing: Clearly identifies “visual amnesia” as the key bottleneck in small VLMs for geometry.
- Elegant mechanism: The interleaved visual-text chain with dynamic grounding tokens is simple yet highly effective.
- High efficiency: 2.7B model runs ~15× faster than GPT-4V while achieving >90% of its accuracy—ideal for educational deployment.

### Weaknesses
1. Limited generalization: The method depends on a pre-trained geometric detector, which may fail on hand-drawn sketches, 3D diagrams, or dynamic constructions (acknowledged in Appendix F).
2. The core components are not fundamentally novel:
   - Visual tokens with grounding appear in prior work (e.g., KOSMOS-2, DocLLM);
   - Step-wise distillation for reasoning is conceptually similar to Hsieh et al. (2023)’s “Distilling Step-by-Step”;
   - Geometric element detection builds on standard object detection frameworks.
Thus, the primary contribution lies in systematic engineering and domain-specific adaptation, rather than a breakthrough in representation learning or distillation theory.

### Questions
1. Detector generalization: How robust is the geometric element detector on real-world diagrams with sketchy or noisy styles? Was cross-dataset generalization evaluated (e.g., training on GeoQA+ and testing detection on We-Math)?
2. Token vocabulary extensibility: How does the system handle non-standard geometric elements (e.g., arcs, sectors, irregular polygons)? Could unsupervised token discovery (e.g., via clustering) reduce reliance on predefined categories?

### Soundness
3

### Presentation
2

### Contribution
2
