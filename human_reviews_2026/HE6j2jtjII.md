# UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Urban embodied AI agents, ranging from delivery robots to quadrupeds, are increasingly populating our cities, navigating chaotic streets to provide last-mile connectivity. Training such agents requires diverse, high-fidelity urban environments to scale, yet existing human-crafted or procedurally generated simulation scenes either lack scalability or fail to capture real-world complexity. We introduce UrbanVerse, a data-driven real-to-sim system that converts crowd-sourced city-tour videos into physics-aware, interactive simulation scenes. UrbanVerse consists of: (i) UrbanVerse-100K, a repository of 100k+ annotated urban 3D assets with semantic and physical attributes, and (ii) UrbanVerse-Gen, an automatic pipeline that extracts scene layouts from video and instantiates metric-scale 3D simulations using retrieved assets. Running in IsaacSim, UrbanVerse offers 160 high-quality constructed scenes from 24 countries, along with a curated benchmark of 10 artist-designed test scenes. Experiments show that UrbanVerse scenes preserve real-world semantics and layouts, achieving human-evaluated realism comparable to manually crafted scenes. In urban navigation, policies trained in UrbanVerse exhibit scaling power laws and strong generalization, improving success by +6.3% in simulation and +30.1% in zero-shot sim-to-real transfer comparing to prior methods, accomplishing a 300 m real-world mission with only two interventions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper provides a new dataset of 100k assets for urban scene simulators, including physical properties, along with ~300 PBR materials for ground surfaces, and ~300 lighting skyboxes. The process for data filtering and annotation is documented as well.

The paper also proposes a real2sim process to convert ego-centric urban navigation videos into interactive simulators. This is done by processing videos to extract a scene graph, match the text and geometry of the scene objects to assets in the above database, and then place assets in corresponding locations in the scene. 

Experiments demonstrate the quality of this approach compared to a baseline model. Human preference scores for scenes (in diversity, coherence, and quality) favor the model over a baseline, nearing a ceiling set of scores from a set of human-crafted scenes. RL policy learning shows power law scaling both in the number of distinct scenes (from videos) and variations of scenes per video. Real-world evaluations add further rigor to the success of the sim2real transfer, including an extended navigation task with minimal human intervention for success.

### Strengths
# originality
- Combines a variety of techniques to produce high quality scenes. The novelty is not so much the problem or any individual component of the generation process, but the combination is a non-trivial addition given the strong results.

# quality
- Experiments show very strong sim2real results.
- Provides a massive and usable asset dataset that has been cleaned up and validated.

# clarity
- The data preparation process is thoroughly explained, as is the generation process. The plan to open-source will further facilitate reproduction, adoption, and extension.
- The narrative and steps were easy to follow.

# significance
- sim2real for urban scene navigation is directly valuable to the autonomous driving community. The assets will be of general value to downstream work on improved generation methods, with clear improvement over existing options and baselines (KITTI).

### Weaknesses
# originality
- No single part of the approach differs from established practices. This is fine given the power of the contribution.

# quality
- It would help to provide data on the scaling and costs of the different methods. For example: how many LLM/VLM calls are needed? how much wall clock time to process a scene? how do these vary in the size of the scene or length of the video?

# clarity
- The sim2real evaluation metrics would benefit from definitions (see below).

### Questions
# questions
- How does UrbanVerse scale?
	- How many VLM/LLM calls are needed?
	- How much wall time / compute?
	- How does UrbanVerse compare to UrbanSim in these dimensions?
- How does this scale with input video length or scene size (in meters, number of entities/assets, and so on)?
	- Is there any ablation data on training based on different video lengths or scene dimensions?
- lines 281-281: How sensitive are the outcomes to assigned parameters? How are they assigned?
- line 476: What were the two interventions needed?
- Table 2: Define the evaluation metrics. Some were not clear to me:
	- How does success rate differ to route completion? When can you complete the route without succeeding?
	- CT: Should the arrow be arrow down or up?
	- "Collision times" is ambiguous. It's not clear if this means "(mean) time to collision" vs "number of times there was a collision", for example.
	- The upward arrow suggests lower is better, implying the latter. But on that metric, PPO-UrbanVerse does quite poorly in Table 2 and for wheeled in Table 3. But well for quadruped? It's a bit confusing how to interpret this.


# suggestions
- Table 1: Separate "Ast." as a clearly marked final column as it is evaluated on different data (and thus not quite continuous with the other metrics).
- Figure 7: Would be nice to increase the scale to see where improvements asymptotically saturate. Given the layouts are limited to 32 environments it would suffice to increase the number of training cousins to show where success rate asymptotes (near 100%?).

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
5

### Summary
This paper presents UrbanVerse, a data-driven real-to-sim system that automatically converts city-tour videos into physics-aware, interactive simulation environments for training urban embodied AI agents. The system comprises two main components: (1) UrbanVerse-100K,  a curated repository of 102,530 high-quality metric-scale urban object assets, and (2) UrbanVerse-Gen, an automated pipeline that extracts scene layouts from real-world videos and instantiates simulation scenes by retrieving matched assets. The authors construct 160 training scenes from city-tour videos spanning 24 countries and demonstrate that policies trained in these environments follow scaling power laws and achieve strong sim-to-real transfer performance in mapless urban navigation tasks, including a 337-meter real-world deployment with only two human interventions.

### Strengths
1. UrbanVerse presents an effective framework that retrieves similar 3D assets through video reference and employs a mapping mechanism to construct street scenes that closely approximate real-world environments. This framework theoretically allow the system to recreate any scenario captured in city-tour videos.
2. The author establishes a comprehensive evaluation framework with six reasonable metrics (Cat, Ast, Cov, Lay, Loc, and Div) that assess different aspects of reconstruction fidelity. Combined with human evaluation, this hybrid evaluation thoroughly validates the effectiveness of the proposed framework. 
3. The navigation experiments prove that scene diversity significantly enhances the generalization capability of navigation agents, which highlights the urbanverse's potential capability.
4. The long-distance real-world deployment provides convincing evidence of UrbanVerse's excellent sim-to-real transfer capabilities.

### Weaknesses
1. UrbanVerse currently exhibits limited interactivity between agents and the environment. After reading the paper, I assume that the urbanverse only supports collision response now. The lack of dynamic environment interactions (pedestrians, moving cars, etc.) prevents UrbanVerse from serving as a general-purpose platform for diverse urban embodied AI research, limiting its utility to primarily perception-based navigation scenarios.

2. The paper does not convincingly articulate why researchers should adopt UrbanVerse over these established alternatives beyond improved visual realism. Without demonstrating substantial functional advantages—such as superior sim-to-real transfer, computational efficiency, or unique interaction capabilities—the contribution appears incremental. The claim that UrbanVerse addresses limitations of procedural generation is weakened by the fact that existing simulators already achieve effective sim-to-real transfer for navigation tasks.

3. Without clear documentation of how the community can leverage this work—whether through asset libraries, simulation APIs, or integration tools—the practical impact remains limited. The contribution of  this urban simulation is undermined by the absence of concrete plans for open-source release, developer documentation, or community engagement strategies.

### Questions
q1: Please supplement the functional innovations of UrbanVerse compared to other simulators (CARLA, MetaUrban, UrbanSim), including: supported embodied intelligence tasks, rendering efficiency comparison, supported agent types, multi-agent interaction, and environmental dynamic disturbances.
q2:Please describe the user-side pipeline of UrbanVerse. Based on the current content in the article, it is difficult to determine UrbanVerse's potential contributions to the community.

### Soundness
3

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
5

### Summary
The paper introduces UrbanVerse, a real-to-sim pipeline that converts crowd-sourced city-tour videos into physics-aware, interactive urban scenes in Isaac Sim. It builds UrbanVerse-100K (100k+ metric-scaled, attribute-annotated assets) and UrbanVerse-Gen to lift semantics, layout, ground, and sky from uncalibrated videos, producing 160 training scenes plus a 20-scene benchmark; KITTI-360 reconstruction metrics, human studies, and scaling experiments show improved scene realism and power-law gains in mapless navigation. Policies trained on these scenes outperform baselines in simulation and zero-shot real-world tests (up to ~90% success on a quadruped) and complete a 337 m route with two interventions.

### Strengths
1. The work shows extensive efforts for data curation to ensure a diverse and realistic dataset.
2. The paper shows human evaluation on scene quality. This is the most reliable comparison in the absense of a ground truth scene layout. The result shows scenes in UrbanVerse generally have better quality than UrbanSim.
3. Real world deployment experiments shows good sim-to-real performance of the RL-trained models on the UrbanVerse environment.

### Weaknesses
1. The work develops digital cousin from online videos, but does not provide valid reasons why previous 3DGS-based digital twin methods fail for the purpose. This motivation of the work is not fully consolidated.
2. The data curation pipeline utilized LLM to annotate the physical attributes of the objects. However, this is not verified by human and prone to error. The authors should show evidence that these annotations are correct and convincing.
3. The groud fitting step assumes a single ground plane. This fails to model uneven surfaces and curbs that are vital for the success of sidewalk navigation. These sides walk features can also be different for wheeled or legged robots.
4. The number of the scenes in the dataset is limited to 160. This is quite small considering the large number of city-touring videos available online. The limitation can be verified in Fig. 7 where the navigation performance grows linearly with the nubmer of digital cousins. It is unknown where is the margin for the scaling law on the number of environments.

### Questions
In Tab. 1, why VGGT has lower quality reconstruction than MASt3R? This seems contridictory to common knowledge that VGGT generally outperformes MASt3R in different scenes.

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
3

### Summary
The paper introduces UrbanVerse, a data-driven system that automatically converts real-world city-tour videos into interactive, physics-aware urban simulations. The framework comprises:

- UrbanVerse-100K, a large-scale annotated database of 100k+ metric-scale 3D urban assets with semantic and physical attributes.

- UrbanVerse-Gen, an automated pipeline that reconstructs scene layouts and lighting from uncalibrated videos and instantiates them into simulation-ready environments in Isaac Sim.

The authors demonstrate that policies trained on these automatically generated scenes follow scaling laws and transfer effectively to real-world navigation tasks—achieving up to +30% zero-shot sim-to-real improvement over prior baselines.

### Strengths
- Novel real-to-sim paradigm: The approach of transforming crowd-sourced city-tour videos into physically interactive simulations is innovative and highly scalable.

- Large, well-annotated asset database: The UrbanVerse-100K dataset provides a valuable and reusable resource with semantic and physical labels, addressing a major limitation of existing simulators.

- Strong quantitative and qualitative results: The system shows high reconstruction fidelity (93% semantic accuracy, 1.4 m localization error) and impressive sim-to-real performance across multiple robot platforms.

- Comprehensive experiments: The paper evaluates reconstruction quality, scaling effects (power-law validation), and real-world zero-shot transfer, providing a well-rounded empirical validation.

- Open-sourcing commitment: Promising for community impact, given the open release of code, assets, and simulation scenes.

### Weaknesses
- **Overly complex pipeline with high compounding error:**
    The overall data annotation and reconstruction pipeline involves numerous components—open-vocabulary detection, depth estimation, 3D lifting, GPT-4 labeling, geometry matching, appearance retrieval, and simulation instantiation. Each step introduces noise, and the compounded uncertainty across so many modules raises serious concerns about consistency and accuracy. While the qualitative examples look convincing, they may reflect cherry-picked successes rather than representative results at scale. 

- **Lack of failure case analysis:**
Following the previous, the paper only shows successful examples but omits clear failure cases or error breakdowns. Understanding when and why the system fails—e.g., mis-segmentation, inaccurate depth estimation, wrong asset retrieval—would be critical to assessing robustness and diagnosing limitations. Including a few representative failure visualizations would make the results more transparent and credible.
    
- **Questionable large-scale data quality:**
    The pipeline heavily relies on noisy, crowd-sourced internet videos, often with unstable camera motion, motion blur, or occlusions. Given such low-quality inputs and a long chain of automated processing, it is doubtful that most of the reconstructed scenes achieve the same level of fidelity as the few presented cases. The paper lacks quantitative or statistical analysis of **data noise, annotation error rates, or reconstruction failure modes**, which undermines confidence in the dataset’s overall reliability.
    
- **Limited scope of validation:**
    Despite the title’s emphasis on “urban simulation,” the evaluation focuses only on **navigation** tasks, without exploring manipulation, traffic reasoning, or social interaction scenarios. It remains unclear whether UrbanVerse scenes are truly general-purpose or mainly tuned for simple navigation.
    
- **System integration over algorithmic innovation:**
    The paper’s novelty lies mainly in combining existing large vision models (CLIP, SAM2, GPT-4.1, MASt3R, etc.) into a pipeline, rather than introducing new algorithms or learning frameworks. The contribution is thus primarily engineering-based.
    
- **Insufficient runtime and scalability analysis:**
    The pipeline’s cost—both in compute and manual verification—is not clearly reported. Given the dependence on heavy models and large-scale video inputs, the claimed scalability may be impractical without significant resources.

### Questions
- Can UrbanVerse handle dynamic agents (e.g., pedestrians, vehicles in motion), or is it restricted to static geometry?

- How does scene reconstruction cope with occlusions and low-light conditions in real videos?

- Would the system generalize to non-street urban contexts (e.g., parks, campuses, or indoor–outdoor transitions)?

### Soundness
3

### Presentation
3

### Contribution
3
