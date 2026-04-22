# Quartet of Diffusions: Structure-Aware Point Cloud Generation through Part and Symmetry Guidance

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
We introduce the *Quartet of Diffusions*, a structure-aware point cloud generation framework that explicitly models part composition and symmetry. Unlike prior methods that treat shape generation as a holistic process or only support part composition, our approach leverages four coordinated diffusion models to learn distributions of global shape latents, symmetries, semantic parts, and their spatial assembly. This structured pipeline ensures guaranteed symmetry, coherent part placement, and diverse, high-quality outputs. By disentangling the generative process into interpretable components, our method supports fine-grained control over shape attributes, enabling targeted manipulation of individual parts while preserving global consistency. A central global latent further reinforces structural coherence across assembled parts. Our experiments show that the Quartet achieves state-of-the-art performance. To our best knowledge, this is the first 3D point cloud generation framework that fully integrates and enforces both symmetry and part priors throughout the generative process.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a part-level point cloud generation framework called the "Quartet of Diffusions," which integrates part and symmetry awareness into the generative process. The method uses four diffusion models to generate structure-aware 3D shapes by modeling shape latents, symmetries, parts, and assemblers. The Quartet framework is shown to outperform state-of-the-art methods in terms of diversity, quality, and symmetry.

### Strengths
- Part-level 3D generation is a critical research topic recently.
- Using separate diffusion models to represent parts and transformations between them is novel and interesting, ensuring global shape symmetry.
- The disentangled representation of parts, symmetries, and global structure offers fine-grained control over the generation process, allowing for targeted manipulation of individual parts.
- The method pipeline is complex but clearly written.

### Weaknesses
- Currently, part generation methods based on implicit representations (e.g., PartCrafter, PartPacker, OmniPart) have achieved very impressive results, almost reaching production-ready levels. However, this paper still insists on using point clouds as the 3D representation, and the experiments are only validated on the ShapeNet dataset. As this approach can only generate a limited number of points and coarse geometric details, this somewhat diminishes the contribution and novelty of the paper. Therefore, I recommend that the authors discuss in detail why they chose this approach and clarify the advantages of using point clouds as a 3D representation for structured 3D generation.
- The Quartet framework involves four separate diffusion models, which could introduce significant computational overhead during both training and inference, making it potentially less practical for real-time applications.
- Integration of four diffusion models will lead to accumulated errors, which are not analyzed.

### Questions
See the weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes “Quartet of Diffusions,” a structure-aware point-cloud generative framework that factorizes generation into four diffusion models: a global shape latent, a symmetry group, distinct parts, and an assembly model. Each model is learned separately and connected via conditioning, which enforces part symmetry and yields realistic assemblies. The method is reported to be computationally efficient in training (despite four models) and outperforms prior work on standard metrics (e.g., Chamfer / 1-NNA) and on a newly introduced symmetry-awareness metric (SDI).

### Strengths
-	Clear conceptual decomposition: Factorizing generation into interpretable aspects (shape latent, symmetry, parts, assembly) is an elegant way to inject structure and inductive bias that matches human intuition about object design.
-	Solid empirical gains: The method shows strong improvements on classical generation metrics (1-NNA / Chamfer) and on the proposed symmetry metric (SDI), indicating both fidelity and structural plausibility.
-	Symmetry guarantees: Enforcing symmetry in the part generation produces more realistic, consistent results for symmetric objects.
-	Reasonable compute: The paper claims modest GPU/training cost despite four separate diffusion models, which makes the approach more practical than it may first appear.
-	Rigorous presentation: The paper is mathematically well-presented where needed.

### Weaknesses
-	While the authors introduced an evaluation metric for symmetry-awareness, the work does not explicitly evaluate the part-awareness such as in SeaLion (Zhu et al., 2025).
-	The method is only evaluated on three object classes of ShapeNet. 
-	Related work could be improved and be made more readable. The section seems only like an assembly of many citations without having a flow content-wise.
-	The figures can be improved as often the point clouds overlap with text boxes, making the figures less readable and enjoyable (Fig. 1, 2, 3)

### Questions
-	By explicitly defining diffusion models for the four aspects (shape latent, symmetry, parts, assembly), the authors induce a bias into the model. Is it possible to investigate the impact of each of the four aspects individually?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposes a 3D point cloud generative model that considers part generation with symmetry. The model uses a four-stage diffusion process: it first generates a global shape latent, then symmetry groups (represented by reflection planes), followed by part geometry, and finally an assembler that composes the shapes. Experimental results show the framework achieves better performance than various point cloud generation baselines while using a lower training budget. The ablation study validates the effectiveness of the proposed method.

### Strengths
- The explicit consideration of symmetry in the 3D point cloud generation process is a novel exploration, and the evaluation, especially the SDI metric in Tables 1 and 3, clearly demonstrates the effectiveness of the proposed framework.
- Additionally, the proposed framework achieves state-of-the-art performance compared to existing point cloud generation methods on standard metrics (1-NNA in Table 1), showcasing the quality and diversity of generated shapes.
- A thorough ablation study in Table 3 validates the necessity of each introduced module.

### Weaknesses
Despite the strong performance shown, I still have some concerns regarding the evaluation setting.

1. Data Split Concerns. In Table 1, the evaluation protocol follows PointFlow (Yang et al., 2019), which uses the full category dataset for train/test splitting. Since this work uses ShapeNetPart, a subset of that data, it is unclear whether the train/test split remains consistent. If not, the test set may have leaked into the training set. Clarification is needed to ensure fair evaluation.
2. Comparison Baselines. The proposed framework is compared against many point cloud generation methods. However, none of them (including SALAD and SPAGHETTI) use part annotations for training their generative models. The proposed comparisons do not seem comprehensive. Instead, I would suggest comparing with methods that use part labels when training generative models. Some examples are provided below:
- StructureNet: Hierarchical Graph Networks for 3D Shape Generation. SIGGRAPH Asia 2019.
- SDM-NET: Deep Generative Network for Structured Deformable Mesh. SIGGRAPH Asia 2019.
- PQ-NET: A Generative Part Seq2Seq Network for 3D Shapes. CVPR 2020.
- DSG-Net: Learning Disentangled Structure and Geometry for 3D Shape Generation. TOG 2021.
- DiffFacto: Controllable Part-Based 3D Point Cloud Generation with Cross Diffusion. ICCV 2023.
- PASTA: Controllable Part-Aware Shape Generation with Autoregressive Transformers. Arxiv 2024.

I suggest making minor adjustments to the framework's claims: 

- The paper claims the framework supports fine-grained manipulation, but provides no visual examples or quantitative evaluations. This claim should be softened to avoid confusion.
- Various existing works consider symmetry in 3D shape generation (particularly for feature extraction), especially StructureNet and related approaches. More discussion on this point is recommended.

### Questions
Overall, I am convinced by the novelty and strong performance of the proposed framework. However, I have concerns about the fairness of the evaluation protocol and the absence of important baselines. Therefore, I am currently leaning toward rejection but am open to raising my rating if the following questions are addressed:

1. Data Split Clarification: Could you please clarify the train/test split used in your experiments? Since you use ShapeNetPart (a subset of the full ShapeNet dataset) while the baseline evaluation protocol follows PointFlow (Yang et al., 2019) which uses the full category dataset, it is unclear whether your split maintains consistency with the original protocol. Specifically, could you confirm that there is no data leakage between your training and test sets, and provide details on how many samples were used for training vs. testing?
2. Comparison with Part-Aware Baselines: The current baselines (including SALAD and SPAGHETTI) do not use part annotations during training, which makes the comparison potentially unfair given that your method explicitly leverages part labels. Could you include comparisons with some part-aware generative methods such as StructureNet (SIGGRAPH Asia 2019), SDM-NET (SIGGRAPH Asia 2019), PQ-NET (CVPR 2020), DSG-Net (TOG 2021), DiffFacto (ICCV 2023), and PASTA (Arxiv 2024)? This would provide a more comprehensive evaluation of your method's advantages when part supervision is available.

### Soundness
2

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
The paper proposed Quartet of Diffusion, a part-level, symmetry conditioned point cloud diffusion model that is trained on Shapenet dataset with segmentation labels. After sampling a global shape latent code, the paper then samples both the part level code and a symmetry group that the shape obeys. Afterward, a set of scales, translations, and rotations are sampled that are used to assemble the shapes together. The method shows superior generation metrics compared to other point cloud generation methods. It also supports part level generation which keeping the rest of the parts fixed

### Strengths
1. The paper proposed a novel symmetry conditioned point-cloud generation method. The symmetry diffusion formulation proposed in the paper seems novel and could potentially be a contribution to the shape generation field 
2. The generation quality surpasses many strong baseline including LION and ShapeGF. This suggests the overall generation quality.

### Weaknesses
1. I found the application provided by the authors to be insufficient to justify the modeling of parts and symmetry. Part-level generation is not a novel task and could be done without modeling the symmetry (See DiffFacto). It would be great to see further application enabled by the structure-aware modeling. 
2. The training requires part-segmentation labels as well as closed vocabulary part classes. This limits the generalization ability of the network to larger dataset such as Objaverse. 
3. The generated outcome is a point-cloud, which to me is not clear what it would be useful for. Can subsequent surface reconstruction algorithm be used to recover a higher fidelity shape?

### Questions
1. Are the assembled shapes always valid? Are there intersections or detachment between parts? Some metrics that quantifies this would be helpful (See SeaLION or DiffFacto for example metrics)
2. Some shapes don’t have all the parts. How does the part diffusion model shapes with missing parts?

### Soundness
3

### Presentation
3

### Contribution
3
