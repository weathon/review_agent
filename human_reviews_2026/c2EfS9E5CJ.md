# AdaViewPlanner: Adapting Video Diffusion Models for Viewpoint Planning in 4D Scenes

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Recent Text-to-Video (T2V) models have demonstrated powerful capability in visual simulation of real-world geometry and physical laws, indicating its potential as implicit world models. Inspired by this, we explore the feasibility of leveraging the video generation prior for viewpoint planning from given 4D scenes, since videos internally accompany dynamic scenes with natural viewpoints. To this end, we propose a two-stage paradigm to adapt pre-trained T2V models for viewpoint prediction, in a compatible manner. First, we inject the 4D scene representation into the pre-trained T2V model via an adaptive learning branch, where the 4D scene is viewpoint-agnostic and the conditional generated video embeds the viewpoints visually. Then, we formulate viewpoint extraction as a hybrid-condition guided camera extrinsic denoising process. Specifically, a camera extrinsic diffusion branch is further introduced onto the pre-trained T2V model, by taking the generated video and 4D scene as input. Experimental results show the superiority of our proposed method over existing competitors, and ablation studies validate the effectiveness of our key technical designs. To some extent, this work proves the potential of video generation models toward 4D interaction in real world.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes AdaViewPlanner, a two-stage framework that adapts a pre-trained text-to-video (T2V) diffusion model to plan camera trajectories (viewpoints) for a given 4D scene, with a focus on human motion. Stage I injects normalized SMPL-X motion into a frozen T2V backbone via a spatial motion attention branch to generate cinematic videos whose frames implicitly encode the planned viewpoints; Stage II then extracts absolute camera extrinsics with a camera-diffusion branch in an MMDiT multi-modal transformer, conditioned on the Stage-I video and the 4D motion, trained with a flow-matching objective.

### Strengths
1. Writing and presentation are clear, with well-motivated design choices and well-explained results.

2. The paper is technically solid and presents meaningful improvements both quantitatively and visually. 

3. The experiments are thorough, covering ablations, human evaluation, and clear baselines. 

4. The method is conceptually useful because it connects generative video modeling with geometric camera control.

### Weaknesses
1. The main limitation is the narrow evaluation scope: all experiments focus on human motion, leaving unclear how the approach generalizes to other 4D scenes or dynamic objects. 

2. I don't fully get the 4D scene concept in this paper. The generated scenes/humans are limited in view coverage.

3. The method depends heavily on accurate motion reconstruction via GVHMR, which may not be reliable in more complex scenes. 

4. Another issue is reproducibility—part of the evaluation relies on a proprietary model (Gemini 2.5 Pro), which makes results hard to verify independently. 

5. compute cost and inference efficiency are not reported, an important factor given the two-stage setup.

### Questions
1. How well does AdaViewPlanner perform on non-human or multi-object motion?

2. How sensitive is it to errors in motion reconstruction or camera intrinsics?

3. What is the computational overhead of the full pipeline, and could it be reduced through distillation or pruning?

4. Would a simpler cross-attention head perform comparably to the MMDiT design?

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
4

### Summary
AdaViewPlanner adapts pre-trained text-to-video diffusion models to the task of viewpoint planning in 4D scenes by a two-stage method: (1) an adaptive learning branch injects viewpoint-agnostic 4D scene representations into the pre-trained T2V model so the generated conditional video visually encodes candidate viewpoints, and (2) a camera-extrinsic diffusion branch performs hybrid-condition guided denoising to extract camera extrinsics from the generated video and the 4D scene. The paper reports that this approach outperforms existing competitors and includes ablations validating the main design choices

### Strengths
1. Clear novel idea - leverages strong priors in large pre-trained video diffusion models as implicit world models to support viewpoint planning, reframing viewpoint prediction as a video-conditioned denoising task.

2. Practical two-stage design - separating scene injection and extrinsic prediction makes the method compatible with fixed pre-trained T2V backbones, reducing the need for fully retraining large video generators.

3. Empirical support - experiments claim superiority over competitors and ablation studies that isolate the contributions of the adaptive branch and the extrinsic diffusion branch.

4. Broader implication - demonstrates a promising direction of reusing generative video priors for embodied perception and 4D interaction tasks beyond pure generation.

### Weaknesses
1. Dependence on pre-trained T2V quality - performance likely tied to how well the base video diffusion model captures geometry and viewpoint cues; limited discussion of failure modes when the generator hallucinate inconsistent geometry. Experimental sensitivity to the choice of pre-trained model appears underexplored.

2. Scalability and compute - adapting and running diffusion branches for viewpoint extraction may be computationally intensive for real-time or embedded planning; paper does not clearly quantify runtime or resource requirements for planning in practice.

3. Generalization to real-world 4D data - the paper summary does not specify the datasets used or robustness to real sensor noise, occlusions, and dynamic scene elements, leaving open questions about transfer from synthetic or curated benchmarks to real-world robotics settings.

### Questions
1. Which pre-trained T2V backbones were used, and how sensitive are results to that choice? Can you please provide results with open-source T2V models?

2. What datasets were used for training and evaluation, and how do results vary between synthetic and real-world 4D scenes? Can you please provide results with data that is widely available?

3. What are the runtime and memory requirements for viewpoint extraction per candidate or per scene, and can the method be optimized for real-time planning?

4. How does the method handle dynamic scene elements or moving objects in the 4D input, and does it distinguish viewpoint changes from object motion?

5. Are there common failure modes (e.g., hallucinated geometry, ambiguous viewpoints), and do authors have strategies to detect or mitigate them?

6. Can the approach be extended to plan multi-step camera trajectories (sequences of viewpoints) rather than single-viewpoint prediction?

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
4

### Summary
- This paper introduces AdaViewPlanner, the first method that adapts pre-trained Text-to-Video diffusion models for automatic camera viewpoint planning in 4D scenes.

- This paper uses a two-stage pipeline: (1) inject 4D scene into the T2V model to generate a video embedding implicit camera viewpoints, (2) extract camera poses via a dedicated diffusion branch conditioned on the video and scene.

- This paper provides outputs both coordinate-aligned camera trajectories and a video visualization, enabling prompt-controlled cinematography without requiring task-specific training.

### Strengths
- This paper leverages pretrained T2V priors, inheriting cinematic knowledge and strong generalization to diverse scenes—unlike previous specialized models requiring narrow datasets.

- This paper presents text-controllable viewpoint planning, enabling users to specify camera style and motion via natural language prompts.

- This paper ensures stable and effective design, with guided pose hints and a hybrid denoising branch that prevent training collapse and produce accurate, scene-aligned camera paths.

### Weaknesses
- Looking at 0001.mp4 (1 full result), there seems to be a tendency that the model does not fully reflect the motion, and I believe this should be mentioned in the limitations section. 

- Only Stage I and Stage II are presented as the ablation study, but you should also include a more detailed ablation study on newly introduced components, such as Spatial Motion Attention.

- The user study is very unclear. It states "Invite researchers," but it is not specified what kind of researchers were involved. It also does not explain what results were obtained from the survey, and no example of the questionnaire is provided. Furthermore, it is not stated whether approval from the Institutional Review Board (IRB) was obtained. While detailed information cannot be disclosed due to the double-blind policy, I believe that at least the basic principles should be followed.

- Would the authors be willing to add a limitations section? It is necessary to provide clear information about what the method can and cannot do.

### Questions
Mentioned in the weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes AdaViewPlanner, which leverages pre-trained Text-to-Video (T2V) models to automatically generate professional camera trajectories in 4D scenes. The core contribution is a two-stage pipeline: Stage I injects 4D human motion into a T2V model to generate cinematic videos with implicit camera movements using a guided learning scheme; Stage II explicitly extracts camera poses through a camera diffusion branch in an MMDiT framework. The method outperforms baselines on multiple metrics and demonstrates text-controllable, diverse camera trajectory generation.

### Strengths
1. Novel and well-motivated approach. First work to leverage pre-trained T2V models for automatic camera planning in 4D scenes, based on the insight that T2V models implicitly learn cinematographic knowledge. This approach effectively reuses foundation model priors for generalization and text-controllability.
2. Effective guided learning scheme. The curriculum learning strategy (providing ground-truth camera tokens with probability p) is critical for preventing training collapse. Ablations clearly show variants without this guidance or the video model fail to converge (Figure 7, Table 2).
3. Strong experimental results. Significant improvements over baselines across all metrics (>60% user preference, Table 1), with comprehensive evaluation addressing prior limitations. Thorough ablations validate each design choice.

### Weaknesses
1. Heavy reliance on synthetic data with limited real-world validation. Stage II training depends entirely on synthetic UE datasets (244k samples) and GVHMR reconstructions, with no evaluation on real captured videos. The sim-to-real transfer capability remains undemonstrated, and reconstruction errors from GVHMR directly propagate to training, potentially limiting real-world robustness.
2. Limited scope. Evaluation focuses solely on human-centric SMPL-X motion with no experiments on multi-agent scenes, non-human subjects, or general dynamic objects. (This can be future work)

### Questions
3D RoPE temporal sensitivity and motion speed dependency. Does the 3D RoPE encoding exhibit sensitivity issues across different motion speeds? Since actions vary from slow walking to fast dancing, have you analyzed whether the positional encoding properly captures these temporal dynamics? Table 2 shows modest improvement with 3D RoPE (122.13 vs 103.92 WA-MPJPE), but does performance degrade for extremely fast or slow motions? Have you examined whether motion-speed-adaptive encoding or temporal scaling strategies could improve results, particularly for rapid actions requiring finer temporal resolution versus slow motions needing different spatial emphasis?

### Soundness
4

### Presentation
4

### Contribution
4
