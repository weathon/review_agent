# DenseMatcher: Learning 3D Semantic Correspondence for Category-Level Manipulation from a Single Demo

- Avg Score: 7.50
- Decision: Accept (Spotlight)
- Scores: 6, 6, 8, 10

## Abstract
Dense 3D correspondence can enhance robotic manipulation by enabling the generalization of spatial, functional, and dynamic information from one object to an unseen counterpart. Compared to shape correspondence, semantic correspondence is more effective in generalizing across different object categories. To this end, we present DenseMatcher, a method capable of computing 3D correspondences between in-the-wild objects that share similar structures. DenseMatcher first computes vertex features by projecting multiview 2D features onto meshes and refining them with a 3D network, and subsequently finds dense correspondences with the obtained features using functional map. In addition, we craft the first 3D matching dataset that contains colored object meshes across diverse categories. We demonstrate the downstream effectiveness of DenseMatcher in (i) robotic manipulation, where it achieves cross-instance and cross-category generalization on long-horizon complex manipulation tasks from observing only one demo; (ii) zero-shot color mapping between digital assets, where appearance can be transferred between different objects with relatable geometry. More details and demonstrations can be found at https://tea-lab.github.io/DenseMatcher/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Summary: This paper introduces DenseCorr3D, a 3D matching dataset featuring colored meshes and dense correspondence annotations. It addresses the limitations of existing datasets that predominantly emphasize geometry. The authors propose DenseMatcher, a model that integrates 2D foundation models with 3D networks to significantly enhance dense correspondence accuracy. The effectiveness of DenseMatcher is demonstrated through applications in robotic manipulation tasks and color transfer experiments.

### Strengths
Strengths:

The authors have developed a dataset that is a valuable resource for the research community.

Despite its straightforward pipeline and principles, the proposed DenseMatcher effectively extracts semantic maps that facilitate subsequent tasks.

The introduction of the function map is promising, and the correspondence video demo on the accompanying website is impressive.

### Weaknesses
Weaknesses:

The range of tasks and the diversity of object categories provided in the dataset are limited.

Line 853 mentions the total time expenditure without delving into specific details, such as the time required for rendering images, particularly the computation comsumption of the function map.

The paper lacks an ablation study for the DINO and SD components. Previous zero-shot methods shows that the features provided by SD VAE may not be optimal. An ablation analysis for the feature backbone should be included in the experimental tables.

There is no discussion on whether the model incorporates augmentations for the pose of the mesh. Research has shown that semantic features can easily overfit to spatial position-related scenarios. If the input mesh's position changes, the resulting semantic map may become inaccurate. Therefore, it would be beneficial to include experiments that apply random rotations to the mesh as input.

### Questions
Additionally, it would be constructive to present examples of failure cases to provide a more comprehensive evaluation.

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
This paper studies the problem of dense surface-point matching between objects, where similarity is understood as a user-defined semantic and matches can be between objects of the same, but also different category. The contributed method combines features that encode the visual appearance with features that encode local geometry. The method is evaluated and compared against baselines on a self-created dataset and real-world robotic imitation of human demonstrations.

### Strengths
- The authors claim (and I am not aware otherwise, but also not super familiar with this subfield) to contribute the first method for 3D dense correspondences that combined visual appearance and geometric information. This very intuitively makes sense and makes especially the contributed dataset something that can have profound impact on the research on 3D correspondences.
- The method has directly been evaluated in a real-world application of mimicing human demonstrations with a robotic manipulator.
- The paper is very well written (the best in my review batch) and easy to follow.

### Weaknesses
- The experimental evaluation is limited to a self-contributed dataset and very few qualitative runs on a robotic application (where it is unclear if the method difference is statistically significant).
- The method design contains a couple of non-straightforward design choices without justifications or experimental evidence to back up these choices [**update from discussion with authors: these points are mostly addressed now**]:
  - Using the XYZ coordinates of the mesh vertices makes the method sensible to random transformations on the input mesh. There is no experiment evaluating whether the model is able to learn invariance over such random coordinate system changes.
  - The choice of negative cosine similarity in $L_\textrm{semantic}$ is quite particular. The authors do not explain why they would choose this over e.g. L1 or L2 distances and also do not ablate this choice.
  - Similarly, for $L_\textrm{preservation}$, the choice of a single linear layer for reconstruction might hinder the encoder network to learn a more useful non-linear function. The more standard choice would probably be to mirror the encoder architecture like in an autoencoder, but this is neither discussed nor evaluated.
- The method requires supervised training with an expensive 3D annotation workflow.

### Questions
- Section 4.1: I am not super familiar with the prior work on 3D dense matching, but this optimization formulation seems computationally expensive and as Section 4.4 shows also unstable. Why are other assignment and matching methods not compared as beaseline or ablation? e.g. Hungarian matching or the double-softmax used in [1]?
- line 200: The requirement of textured 3D assets is very limiting. It seems to me the method could also work from an untextured geometry asset and posed images, or am I missing something?
- line 242: Since the negative cosine distance is such an odd choice I suspect the authors were inspired here by related work? In that case it would be important to attribute this here with a reference.
- line 252: "object type and material" is misleading. Neither one of the frozen backbones captures this information, both are self-supervised encoders of visual appearance that might correlate with this information in some cases.
- line 254: What norm is used in the equation for $\mid\mid \cdot \mid\mid$? Why is that one choosen?
- Table 1: Please explain better the different ablation variants. Is "w/o Diffusion Net" directly matching the concatenation of $f_\textrm{multiview}$ and the HKS features? Or is it also using the XYZ features and therefore failing because of coordinate system change? 
- Section 6.2.3: I dont't think the comparison to Robo-ABC is entirely fair. It would be good to show both variants, with the full affordance memory and with the reduced form that is currently presented. The proposed method is very expensive in terms of the 3D data it requires, so really it needs to show that this additional information can compete with methods that are only based on cheaper and more abundant image data.
- Section 6.2.4: How is success determined in the experiments? Given the low number of overall trials, what level of statistical significance does the experiment currently have?



[1] Lindenberger, P., Sarlin, P.-E., & Pollefeys, M. (2023). LightGlue: Local Feature Matching at Light Speed. Retrieved from https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces DenseMatcher, an innovative method for computing dense 3D correspondences between objects with similar structures, geared towards applications in robotic manipulation. They propose that *semantic correspondence*—which aligns semantically similar parts across objects—provides more powerful generalization capabilities across categories compared to *shape correspondence*, which mainly focuses on geometry.

To facilitate the training and evaluation, they created *DenseCorr3D*, a new dataset comprising 589 colored object meshes across 23 categories, with dense correspondences organized into semantic group. DenseMatcher utilizes pre-trained 2D foundation models to extract multiview features, which are further refined using DiffusionNet. The enhanced features are then used to establish dense correspondences through a functional map. 

They provide comprehensive experiment results to demonstrate DenseMatcher’s effectiveness in 3D dense matching, zero-shot robotic manipulation, and color transfer tasks. DenseMatcher outperformed baseline methods on the DenseCorr3D benchmark and achieved a 76.7% success rate in real-world robotic manipulation, showcasing its robust generalization capabilities.

### Strengths
- Integration of 2D and 3D: DenseMatcher effectively combines 2D foundation models, like SD-DINO, for multiview feature extraction with DiffusionNet to refine features with geometry. This fusion enhances semantic understanding and generalizability in 3D correspondence.

- New 3D matching dataset: The authors introduce DenseCorr3D, the first dataset with colored meshes and dense correspondences, featuring 589 textured meshes across 23 categories. It advances research by supporting methods that account for both appearance and geometry.

- Enhanced functional map for accuracy: A novel regularization scheme promotes sparsity in DenseMatcher’s functional map, achieving a 43.5% accuracy improvement over baselines.

- The paper is well-written and easy to understand. The experiment results are comprehensive and promising.

### Weaknesses
- Limited analysis on varying topologies: While they analyze that previous methods struggle with different topologies, they do not deeply explore DenseMatcher's robustness on diverse object structures.

- Limitation to severe occlusion: The paper does not address how DenseMatcher handles significant occlusion. Since it relies on multiview feature extraction and functional maps, both susceptible to occlusion, further analysis of this limitation would strengthen the evaluation.

### Questions
- Performance on Varying Topologies: How does DenseMatcher perform with objects of varying topologies? Are there specific object structures or topological variations where its performance significantly degrades?

- Handling Severe Occlusion: Is DenseMatcher able to be adapted or extended to handle severe occlusion more effectively? What potential modifications could mitigate its reliance on multiview feature extraction and functional maps in such cases?

- More Benchmark Validation: Are there any benchmarks or experiments that could further validate DenseMatcher’s robustness against topological diversity and occlusion? How might these additional evaluations impact its overall effectiveness and applicability in real-world scenarios?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper proposes a framework and dataset for category-level object 3D dense matching. The DenseMatcher utilizes a 2D foundation model with 3D network refinement to reach generalization and 3D understanding. The author conducts robotic manipulation and zero-shot color mapping to validate the findings.

### Strengths
1. This idea is novel and underexplored in relevant areas, especially in robotic manipulation learning. Instead of simply augmenting the data with numerous demos, this paper can address sample efficiency by embedding semantic information.

2. This paper's writing style is straightforward, and it is easy to catch the main topic.

3. Utilizing the existing 2D network (DINO in this paper) with 3D networks is a simple but promising approach.

4. Experiments can thoroughly reflect the model's ability. In robotic manipulation tasks, it covered pick-and-place, long-horizon, and dual arm.

### Weaknesses
1. The statements of regularization terms in the methodology part are unclear and may cause ambiguity.

2. Some experiment details, like the description for each task, can be placed in the appendix and give a more precise visualization. The images in the robotic manipulation task are too undersized.

### Questions
1. In Sec 4.1 Preliminary, Functional Map, please give a detailed justification about how to regularize the term C as isometric in your context. 

2. In the appendix, please provide a detailed explanation, with proofs, showing how previous constraint terms ensure that the output is minimized in the semantic distance function.

### Soundness
4

### Presentation
4

### Contribution
4
