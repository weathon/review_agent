# ArtVIP: Articulated Digital Assets of Visual Realism, Modular Interaction, and Physical Fidelity for Robot Learning

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 2, 8, 6, 4, 6

## Abstract
Robot learning increasingly relies on simulation to advance complex abilities such as dexterous manipulation and precise interaction, necessitating high-quality digital assets to bridge the sim-to-real gap. However, existing open-source articulated-object datasets for simulation are limited by insufficient visual realism and low physical fidelity, which hinders their utility for training models to master robotic tasks in the real world. To address these challenges, we introduce ArtVIP, a comprehensive open-source dataset comprising high-quality digital-twin articulated objects, accompanied by indoor-scene assets. Crafted by professional 3D modelers adhering to unified standards, ArtVIP ensures visual realism through precise geometric meshes and high-resolution textures, while physical fidelity is achieved via fine-tuned dynamic parameters. Meanwhile, the dataset pioneers embedded modular interaction behaviors within assets and pixel-level affordance annotations. Feature-map visualization and optical motion capture are employed to quantitatively demonstrate ArtVIP’s visual and physical fidelity, and its applicability is validated through imitation learning and reinforcement learning experiments. Provided in USD format with detailed production guidelines, ArtVIP is fully open-source, benefiting the research community and advancing robot learning research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ArtVIP, a novel open-source dataset of articulated digital assets designed for robot learning. It tackles the shortcomings of existing datasets by providing high visual realism, modular interaction capabilities, and physical fidelity, all within the USD format. The dataset includes 206 articulated objects across 26 categories, along with complementary indoor scene assets. The authors demonstrate its effectiveness through evaluations of visual and physical properties and validate its utility in imitation learning and reinforcement learning tasks, showing improved sim-to-real transfer and robust policy training.

### Strengths
1. This work offers a substantial collection of 206 articulated objects and scene assets, meticulously crafted by professional 3D modelers. The emphasis on visual realism (meshes, textures, PBR materials) and physical fidelity (collision, optimized joints) is a significant improvement over existing datasets.
2. The embedding of customizable, reusable behaviors directly within assets is a novel and highly beneficial feature. This greatly simplifies the development of interactive functionalities in simulations, reducing overhead and accelerating research.
3. The paper provides objective evaluation methods for both visual realism (triangular faces, reconstruction performance, feature distribution visualization with t-SNE and CLIP) and physical fidelity (optical motion capture comparing real-world vs. simulated joint movements under force). This rigorous evaluation adds significant credibility to the claims of high fidelity.
4. The extensive experiments in imitation learning (ACT, DP) and reinforcement learning (EAGLE framework) clearly demonstrate the practical utility of ArtVIP. The results showing zero-shot deployment capability and improved performance with mixed real-sim data are particularly compelling.

### Weaknesses
1.  While the paper mentions streamlined processes and scripting tools, the asset creation remains a manual process by professional 3D modelers. The paper acknowledges this as a limitation, stating, "scaling to even larger datasets remains a non-trivial challenge." This could be a significant barrier to expanding the dataset's diversity and size in the future without further automation.
2.  The enhancement of the joint drive equation, especially the functions of $q$ and $\dot{q}$, is crucial for physical fidelity. While the appendix provides the equations, a more detailed explanation of the derivation or the empirical process for determining parameters like $\mu_s$, $k_{high}$, $k_{low}$, $\alpha$, and $\lambda$ would strengthen this section.
3.  While the RL experiments validate the dataset, they focus on a single CloseTrashcan task. Expanding the range of RL tasks to demonstrate the utility of ArtVIP across more diverse manipulation challenges would further strengthen the claims.

Typo: Line 431: ACT 39% on PullDrawer

### Questions
1. Regarding the Modular Interaction feature: Can the authors provide more technical details or an example of how a behavior like $\textit{toggle door}$ is embedded directly into the assets in USD format without writing additional Python scripts? What is the underlying mechanism for this modularity and reusability?
2. The paper mentions the trade-off between the number of triangular faces and simulation frame rate (Appendix G). How did the authors balance these aspects, and what specific optimization techniques were employed beyond merging redundant vertices to ensure a consistent frame rate above 60 Hz across diverse scenes and objects?
3. In the imitation learning experiments (Tab. 1), the "Sim-Only" models consistently performed worse than "Real-Only" models, even with equal data quantity. While the paper attributes this to persistent challenges in bridging the sim-to-real gap, could the authors elaborate on specific aspects of the remaining gap (e.g., visual discrepancies, unmodeled physical phenomena, or limitations in the teleoperation data collection) that still contribute to this performance difference?
4. Given the acknowledged challenge of scaling asset creation, what concrete steps are planned for future work on generative approaches (e.g., specific generative models, data requirements, or methodologies) to automate asset synthesis and broaden diversity?

### Soundness
3

### Presentation
4

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
This paper presents ArtVIP, an open-source dataset of 992 articulated digital-twin objects with a focus on visual realism, modular interaction, and physical fidelity. All assets are professionally modeled under unified geometric and material standards and provided in USD format for compatibility with Isaac Sim and other simulators. The dataset includes detailed annotations, embedded interaction primitives, and scene assets for direct use in robot learning. The authors evaluate ArtVIP through reconstruction and feature-distribution analyses, optical motion capture of joint dynamics, and downstream imitation and reinforcement learning tasks. The work aims to bridge the sim-to-real gap by offering high-quality, physically consistent assets and comprehensive guidelines for digital-twin creation.

### Strengths
The paper presents ArtVIP, a high-quality open-source dataset of articulated digital objects designed to enhance simulation fidelity for robot learning. Its key strength lies in the professional-level asset quality, with rigorous modeling standards ensuring both visual realism and physical fidelity. The dataset includes modular, interactive behaviors and pixel-level affordance annotations, enabling more complex manipulation tasks in simulation. 

Comprehensive evaluations including feature-level realism, physical motion consistency, and sim-to-real experiments in both imitation and reinforcement learning, demonstrate its practical utility. ArtVIP significantly advances the state of datasets for embodied AI by addressing limitations in existing resources and supporting immediate deployment in high-fidelity simulators.

### Weaknesses
While ArtVIP presents a high-quality dataset with clear efforts toward visual and physical fidelity, the evaluation methodology lacks quantitative rigor and convincing evidence. For the Visual Realism evaluation, the qualitative comparison in Figure 5 does not convincingly demonstrate that OmniGibson is inferior in geometry or texture realism. The statement that “reconstructions from ArtVIP assets exhibit higher structural fidelity and finer detail preservation” is insufficiently supported: no quantitative reconstruction metrics (e.g., PSNR, SSIM, Chamfer distance) are reported, and the visual difference is subtle and subjective. Likewise, the CLIP-based t-SNE visualization provides only superficial evidence: the feature clusters for Sim-ArtVIP and real data in Figure 5 (right) still show limited overlap, making the claim of stronger alignment somewhat speculative. Moreover, the paper does not provide dataset statistics (e.g., per-category object count, distribution of joint types, or texture resolutions) in the main text, which makes it difficult to assess the dataset’s scale and diversity without consulting the appendix.

### Questions
For the Reinforcement Learning part, the interpretation of the Pearson correlation between simulation and real trials as evidence of “high physical fidelity and visual realism” is unconvincing. A high correlation merely indicates consistent performance ranking across environments, not necessarily that the simulated physics or visuals faithfully reproduce real-world behavior. The authors should clarify (1) how the correlation is computed: over what quantities and across how many runs; and (2) why training exclusively in simulation would not trivially yield a similar correlation pattern. Without such clarifications, the claimed sim-to-real validity remains ambiguous.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces ArtVIP, a high-quality open-source dataset of articulated digital-twin assets designed for robotic learning.
It highlights the visual realism, physical fidelity and the modular interaction. Quantitative experiments validate the assets through reconstruction fidelity, optical motion-capture comparisons, imitation-learning and reinforcement-learning benchmarks.

### Strengths
1. Encoding interactive semantics (like damping or cross-asset effects) directly in the USD assets is novel and practically useful.
2. Zero-shot sim-to-real transfer in imitation learning demonstrate the benefits of the proposed dataset for robotics learning.
3. The paper is very well written and the analysis is comprehensive.

### Weaknesses
1. While fidelity is high, 992 objects are modest. Integration with generative pipelines would strengthen long-term impact.
2. The dataset emphasizes three key aspects: visual realism, physical fidelity, and modular interaction, but the paper lacks quantitative analysis demonstrating the individual contribution of each component. Conducting ablation studies that selectively degrade or remove each aspect and measuring the resulting impact on imitation-learning and reinforcement-learning policy performance would help substantiate the significance of these design dimensions and strengthen the paper’s empirical validation.
4. The results can be enhanced by comparing IL and RL policies learned on baseline datasets (BEHAVIOR).

### Questions
1. The modeling and tuning time is very long. I wondering for the modeling time, how many time is spent on geometry, visual texture and collision modeling respectively.
2. For RL, is the RL trained purely in sim and zero-shot generalize to real?

### Soundness
3

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
ArtVIP presents a dataset of almost 1000 digital-twin articulated objects in USD format, where all assets are visually realistic and exhibit physical fidelity. The dataset is manually crafted from 3D modeling experts following a given assembly guidelines. One key aspect of the paper is the interactive functionality of the different assets, where 5 key behavior primitives are integrated. This makes it possible to customize objects and still having those interaction functionalities, without the need of rewriting code. The dataset is evaluated on triangle count, reconstruction performance, feature visualization and robot learning.

### Strengths
- The overall paper is well written and motivated. It is clear what the goal of the paper is and why other approaches/datasets do not fulfill the requirements described in ArtVIP
- The dataset is crafted by experts following a specific assembly guideline, which should ensure that the objects are of high quality
- The dataset includes almost 1000 different assets which are articulated, as well as specific scenes and pixel-level annotations
- The evaluation of the dataset is thorough and includes different directions to showcase the advantages over other similar datasets
- Additional Imitation and Reinforcement Learning robot experiments solidify the usage of the dataset in some simple task settings
    - Results show that the Sim2Real gap is still given even with a more realistic dataset, but that it gets smaller

### Weaknesses
- Line 187-189: Any explicit source or statistic that confirms this?
- Visualized Feature Distribution: It is not clear from figure 5 on the right that ArtVIP object embeddings are actually that much closer to real world object embeddings. It looks more like they are still apart and ArtVIP is more closley related to OmniGibson. I think some other form of measurement for the feature distribution would be necessary.
- Claim (1) in line 430: Can you provide experiments using simulated data from other sources and show that the performance on the real robot is then actually worse? Otherwise it could be that ACT itself has some advantage for Sim2Real gap
- A comparison of time and money investment would be interesting compared to other datasets and also datasets which use learnable models to produce articulated objects
- It is also unclear how the higher polygon count affects the rendering speed. Additional comparisons of rollouts or training or robot models on the different asset datasets would help to understand the time difference for rendering.
- The dataset provides pixel-level affordance annotations, but there are no experiments showing the advantages of such an inclusion. Maybe you can train a vision-network to predict such affordances or use them for the robot learning models to better asses if an object is graspable or not. Further experiments would be helpful to verify the need for such annotations in this dataset.
- Minor Weaknesses
    - Figure 1 description is too short and uses an acronym which is not introduced before. I would also suggest to not use such a “teaser” figure, but rather put it below the abstract or even on top of page 2
    - Chapter 3 is an empty section
    - Figure 3 is never mentioned in the text
    - In Table 1 and 2 highlight important results with bold text or something similar

### Questions
- Can you provide additional robot experiments on other simulated datasets to verify the better performance of Sim2Real?
- How much slower is your approach in terms of rendering scenes given the higher polygon count?
- Why are you not evaluating the pixel-level annotations?
- It would be helpful if you can provide how much time and money in total the dataset needed.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes to create high-quality articulated object dataset using human modellers with an emphasis on quality than scaling. Qualitative and quantitative results show the scenes exhibit high photorealistic visual realism. The paper adopts a unified modelling and assembly guidelines given to human modellers and utilizes a top-down hierarchical mechanical modelling approach. Experiments are diverse and showcase various levels of improvements from visual realism compared to other sim provided assets, better intractability and well as imitation learning and reinforcement learning application for sim2real transfer.

### Strengths
In my opinion, below are the main strengths of the paper:

1. A high-quality 3D articulated object datasets combined with scene-level information i.e. kitchen etc. which exhibit greater photorealism and physical intractability. 

2. Experiments are diverse and cover a breadth of tasks such as evaluating photorealism, interactability, reconstruction performance evaluation, feature distribution analysis as well as downstream application to imitation learning and RL.

3. The paper is nicely written, easy to follow and the figures/qualitative results nicely complement the text in the paper.

### Weaknesses
In my opinion, below are the main weakness in the paper:

1. While the qualitative results do show higher quality assets, it's unclear how well these fair when compared to other low-effort feed-forward approaches [1,2,3,4]. A comparison to these feed-forward baselines for the experiments outlines in the paper would justify the time spent in creating the higher quality assets where low-effort approaches sometime run at 1Hz for eight or less objects from a single RGB-D image [2]. 

2. While qualitative results are appreciated, it is not clear how well the method compare to existing democratized approaches [1,2,3,4] to articulated object generation interms of reconstruction performance i.e. with a chamfer distance metric to compare the geoemtric fidelity of the produced articulated assets. 

3. For the imitation learning and RL results, it is unclear if the higher quality assets from the proposed approach led to higher success rate since these are RGB policies and it looks like significant effort was done in recreating table, robot as well as making some parts of the background similar to achieve sim2real transfer. Again, what if we replace the author's proposed assets with meshes from some of the feed-forward approaches, would it result in similar succes rate?

[Minor]

1. How much tuning of the parameters was carried out for interaction evaluation experiments. Can the same tuning effort be done for existing neural-network based articulated object generation approaches [1,2,3,4] and would it result in similar results?

2. Annotation time was not presented in the paper. Is this also significantly higher compared to other approaches that rely on distilled features etc? [5, 6]



[1] Liu et al. PARIS: Part-level Reconstruction and Motion Analysis for Articulated Objects, ICCV 2023
[2] Heppert et al, CARTO: Category and Joint Agnostic Reconstruction of ARTiculated Objects, CVPR 2023
[3] Lin et al. SplArt: Articulation Estimation and Part-Level Reconstruction with 3D Gaussian Splatting, ICCV 2025
[4] Mandi et al. Real2Code: Reconstruct Articulated Objects via Code Generation, ICLR 2025
[5] Yu et al. POGS: Persistent Object Gaussian Splat for Tracking Human and Robot Manipulation of Irregularly Shaped Objects, ICRA 2025
[6] Kerr et al. Robot See Robot Do Imitating Articulated Object Manipulation with Monocular 4D Reconstruction, COR 2024

### Questions
Please see questions in the weakness section, i look forward to seeing the author's response in the rebuttal.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces ArtVIP, an open-source dataset of high-fidelity articulated digital-twin objects aimed at improving robot learning in simulation. Built by professional 3D modelers under unified standards, ArtVIP achieves visual realism through precise geometry and textures, and physical fidelity via tuned dynamic parameters. It further includes modular interaction behaviors and pixel-level affordance annotations. Quantitative evaluations and experiments in imitation and reinforcement learning confirm its strong sim-to-real transfer, making ArtVIP a valuable and reproducible resource for the robotics community.

### Strengths
1. The paper makes a valuable and practical contribution by releasing a high-quality articulated object dataset that combines visual realism, physical accuracy, and modular interactions. The modeling pipeline and embedded behaviors are clearly documented, ensuring long-term usefulness for the robotics community.

2. The dataset’s open-source release in USD format, along with conversion tools (URDF/MJCF) and comprehensive production guidelines, greatly improves accessibility, reproducibility, and integration into diverse simulation workflows.

3. The imitation learning and reinforcement learning experiments convincingly demonstrate strong sim-to-real generalization, underscoring the dataset’s effectiveness for both visual and physical sim-to-real transfer.

4. The figures are clear and motivating, effectively conveying the dataset’s realism and interactivity.

### Weaknesses
1. The primary contribution lies in dataset engineering rather than methodological novelty. While the dataset’s quality is commendable, its scalability is constrained by manual modeling and tuning, which may limit extensibility. With the rise of generative pipelines such as RoboTwin[1] and Genesis[2], ArtVIP’s labor-intensive approach appears less sustainable for expansion.

2. The claimed physical fidelity mainly covers joint parameters such as damping, friction, and magnetic closure, but overlooks more complex mechanical dependencies within articulated systems. Works like AdaManip [3] and DoorGym [4] emphasize such mechanisms. For instance, rotating a doorknob before unlatching or twisting a pressure-cooker lid before lifting are emphasized in these works. These multi-stage kinematic couplings represent realistic physical constraints that ArtVIP does not yet model.

3. Although the paper includes imitation learning and RL experiments, the evaluation scope remains narrow, focusing primarily on sim-to-real ratios within ArtVIP. Direct benchmarks against other datasets (e.g., RoboTwin, AdaManip) under identical task setups would more convincingly demonstrate ArtVIP’s advantages and generality.

[1] Chen, Tianxing, et al. "Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation." arXiv preprint arXiv:2506.18088 (2025).

[2] Zhou, Xian, et al. "Genesis: A Generative and Universal Physics Engine for Robotics and Beyond." arXiv preprint arXiv:2401.01454 (2024).

[3] Wang, Yuanfei, et al. "Adamanip: Adaptive articulated object manipulation environments and policy learning." arXiv preprint arXiv:2502.11124 (2025).

[4] Urakami, Yusuke, et al. "Doorgym: A scalable door opening environment and baseline agent." arXiv preprint arXiv:1908.01887 (2019).

### Questions
1. Could the authors elaborate on how ArtVIP might scale to larger datasets or environments in the future? Given the heavy reliance on manual modeling, would integrating generative pipelines be a feasible direction for expansion?

2. How does ArtVIP compare with prior works that model realistic physical mechanisms rather than focusing mainly on joint parameters?

3. Would it be possible to include cross-dataset or benchmark comparisons (e.g., with RoboTwin or AdaManip) under identical task settings to better contextualize ArtVIP’s contribution?

### Soundness
3

### Presentation
3

### Contribution
2
