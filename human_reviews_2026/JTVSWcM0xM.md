# GeneVLM: Automated Parsing Executable Digital Gene from a Single Image

- Decision: Reject
- Scores: 6, 0, 6, 6

## Abstract
Building structured world representations for robotic agents that can generalize and interact with the physical world is a core challenge in AI. The recently proposed $\textit{Digital Gene}$ offers a promising direction by representing objects as explicit, programmatic blueprints, addressing the generalization and interpretability bottlenecks of end-to-end learning paradigms. However, the practical application of this technology is hindered by a critical bottleneck: creating these genes for real-world objects. Prior methods rely on 3D data, which is difficult to acquire at scale, while parsing directly from ubiquitous 2D images remains an unsolved challenge.

In this work, we introduce $\textbf{GeneVLM}$, a vision-language framework that addresses this bottleneck by automatically parsing executable Digital Gene from a single 2D image. 
First, we propose a specialized and scalable model designed for the image-to-gene parsing task. Second, to enable its training, we design an efficient and scalable procedural pipeline to synthesize a diverse, multi-million-pair dataset of images and their corresponding Digital Genes. Third, to facilitate rigorous evaluation, we establish and release the first comprehensive, multi-dimensional benchmark for this task. Our experiments show that GeneVLM successfully recovers complex object structures and exhibits consistent performance scaling with increased model size, validating the effectiveness of our integrated approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes GeneVLM, a vision-language framework that automatically parses Digital Genes (structured, executable blueprints of 3D objects) from a single 2D image. Building on the Digital Gene concept introduced by Sun & Lu (2025), the work addresses a key bottleneck in structured 3D understanding: generating interpretable and executable object representations from everyday images.
Experiments show that GeneVLM reconstructs coherent object structures and scales predictably with model size, outperforming the Qwen-32B baseline.

### Strengths
1. The “image-to-Digital Gene” task is both original and practically important for interpretable robotic reasoning and structured 3D understanding.
2. Results demonstrate clear performance scaling from 7B to 32B parameters and reasonable generalization to real images. 
3. The implementation details, evaluation metrics, and prompts are meticulously documented.

### Weaknesses
1. Despite the ColorImage-Gene dataset, the work relies heavily on synthetic data, and performance on complex real-world images remains limited.

2. The authors acknowledge that predicted parameters may lack the accuracy required for physical interaction, which limits immediate real-world applicability.

3. The newly proposed VLM-as-judge metric is interesting but somewhat subjective. Moreover, the geometric metrics could be expanded (e.g., IoU, part-level correspondence).

### Questions
1. How robust is GeneVLM to occlusions, cluttered backgrounds, and partial object visibility?
2. Could the constrained decoding FSM handle previously unseen templates or dynamic object categories?
3. Can the proposed method generalize to real-world data?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper introduces GeneVLM, a vision-language framework designed to parse a "Digital Gene" from a single 2D image. The "Digital Gene" is defined as an explicit, executable, and programmatic blueprint (represented as a JSON file) that encodes an object's hierarchical components and geometry.

To solve this, the paper devises three components:

1. A procedural data pipeline that programmatically synthesizes a large-scale (multi-million) dataset of image-gene pairs, starting from a "seed collection" of Digital Genes and applying augmentations.

2. The GeneVLM model, a VLM based on Qwen2.5-VL that incorporates a specialized float-to-token quantization scheme to handle numerical parameters .

3. A constrained decoding method that uses a Finite State Machine (FSM) to ensure the generated genes are syntactically valid and executable.

### Strengths
The paper is well-written, and the problem it aims to solve (2D image to a programmatic 3D representation) is clearly defined. The illustrations for the data pipeline (Figure 2) and model architecture (Figure 3) are clear and helpful for understanding the proposed system. The goal of inferring structured, interpretable, and executable 3D representations from single 2D images is highly significant for AI and robotics. The paper correctly identifies data acquisition as a major bottleneck for structured 3D models.

### Weaknesses
1. Vague and Unnecessary Terminology: The paper's core concept of a "Digital Gene" is a confusing neologism. The paper's own related work appendix (Appendix B) positions this concept against existing explicit representations like MJCF/URDF and programmatic CAD/DSLs . However, the paper's description of a "Digital Gene" as an "executable blueprint that encodes an object's hierarchical components, geometry, and functional attributes" sounds identical to these existing formalisms. The paper fails to substantially differentiate this concept, making the contribution feel like a re-branding of a known problem (image-to-procedural-model) with a strange and unnecessary term.

2. Fundamentally Handcrafted and Over-Engineered System: The work claims to present a learning framework, but the "learning" is almost entirely superseded by complex, handcrafted engineering. This system does not solve the "bottleneck" of gene creation; it just shifts it.

    i) Data Generation: The data pipeline is not based on real-world data. It starts from a "seed collection of Digital Genes" (which are presumably handcrafted) and then applies a complex "procedural pipeline" of "Rule-Based Augmentation" and "Stochastic Removal". This is an entirely synthetic, over-engineered process.

ii) Specialized Tokenization: The model requires a specialized, non-standard quantization scheme to convert floating-point numbers into single tokens. This is a significant engineering workaround that adds complexity and suggests the VLM architecture is poorly suited for regressing the continuous parameters that are essential to the "Digital Gene" format.

iii) FSM-Based Decoding: This is the most significant weakness. The paper admits that standard auto-regressive decoding "often fails to generate syntactically valid Digital Genes". The solution is to use a "constrained decoding method" that relies on a "Finite State Machine (FSM)" to "mask out all others [invalid tokens]" at every sampling step . This means the model is not learning the grammar or structure of the "Digital Gene"; it is being force-guided by a handcrafted parser. This undermines the entire premise of "parsing" and "generation."

3. Insufficient Baselines: The only baseline is a zero-shot Qwen-32B model, which is fed a complex prompt (Appendix A.4). This is a weak baseline for a highly specialized task. The paper does not compare its performance to any other established method for single-image 3D reconstruction or image-to-program synthesis. The custom "Digital Gene" format makes the benchmark insular and difficult to compare against.

### Questions
1. How does the "Digital Gene" representation fundamentally differ from existing programmatic representations like hierarchical part grammars or CAD/DSL formats, which are also explicit, hierarchical, and executable? The distinction seems minimal, making the central contribution feel like a re-branding of a known concept.

2. The reliance on an FSM for constrained decoding is a major weakness, as it implies the model never actually learns the syntax of a Digital Gene. Can the model produce syntactically valid genes without this handcrafted FSM? If not, how much of the "parsing" task is truly being learned versus being hard-coded into the inference logic?

3. The data pipeline (Fig. 2) requires a "seed collection of Digital Genes" and "Rule-Based Augmentation". How is this procedural synthesis approach more scalable or less of a "critical bottleneck" than the manual creation of 3D assets it claims to replace? It seems to have the exact same manual-effort bottleneck, just at the "seed" level.

4. The custom float tokenization scheme is a complex piece of engineering. What was the performance of a simpler, more standard approach, such as having the VLM output text-based floating-point numbers (e.g., "0.596") and letting the standard tokenizer handle them?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces the image-to-Gene task and proposes GeneVLM, a vision-language model designed to parse executable digital genes—programmatic blueprints representing object structures—from a single 2D image. To train the model, the authors construct a dataset by modifying existing shape Genes using multiple strategies, resulting in both no-texture images and augmented color images via existing diffusion models. Experiments demonstrate the effectiveness of this dataset for the novel task. Extensive evaluations and ablation studies further validate the design choices in tokenization and training strategy.

### Strengths
* The image-to-Gene task is clearly motivated, aiming to produce a language-understandable representation of shapes from a single image.


* The dataset generation strategy is reasonable, and experiments demonstrate the usefulness of the generated data.


* The tokenization scheme and two-stage training strategy are well-designed, with ablation studies showing the contribution of each module.
* The proposed benchmark effectively demonstrates the method’s performance and capabilities.

### Weaknesses
* It is unclear whether all object categories are included in the training data, and whether the model can handle more complex instances or unseen categories. Including failure cases would help illustrate the limitations of the current approach.
* When generating synthetic training data, it is not clearly explained how modifications to shape parameters ensure the resulting shapes remain reasonable, or how the model determines which parts of a shape can be altered or deleted without invalidating the object.

### Questions
* In the dataset generation process, what information in the original Genes is used to support the different augmentation strategies?
* After applying these modifications, how do you ensure that the resulting shapes and structures remain reasonable and valid?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies a very important task of building structured world representations for robotic agents. Learning from those structured representations, agents can obtain better generalization ability. To have such representations, this paper studies recent digital gene, which gives concepts and instances some descriptions in JSON format. However, existing works have some scalability due to the need for a point cloud. Thus, this paper first studies how to construct such a representation from a 2D image. The authors proposed a whole pipeline of collecting data, finetuning a VLM, and evaluation. This is a very solid paper.

### Strengths
1. The problem studied in this paper is very important.
2. The content of the study is very solid. The authors provide a whole pipeline for generating a structured representation from 2D images, including dataset collection, fine-tuning VLMs, and evaluation.
3. The training data collected also considered the simulation to realism gap.

### Weaknesses
1. There are some issues with citation format in the paper. The authors use \citet for all citations. However, \citep should be used. For example, at line 173, it is "Our model, which we term GeneVLM, is built upon the Qwen2.5-VL Team (2025a) architecture."

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3
