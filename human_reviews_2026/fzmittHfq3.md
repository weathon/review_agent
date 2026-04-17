# From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 8

## Abstract
Existing vision-language-action (VLA) models act in 3D real-world but are typically built on 2D encoders, leaving a spatial reasoning gap that limits generalization and adaptability. Recent 3D integration techniques for VLAs either require specialized sensors and transfer poorly across modalities, or inject weak cues that lack geometry and degrade vision-language alignment. In this work, we introduce **FALCON (From Spatial to Action)**, a novel paradigm that injects rich 3D spatial tokens into the action head. FALCON leverages spatial foundation models to deliver strong geometric priors from RGB alone, and includes an *Embodied Spatial Model* that can optionally fuse depth, or pose for higher fidelity when available, without retraining or architectural changes. To preserve language reasoning, spatial tokens are consumed by a *Spatial-Enhanced Action Head* rather than being concatenated into the vision-language backbone. These designs enable FALCON to address limitations in spatial representation, modality transferability, and alignment. In comprehensive evaluations across three simulation benchmarks and eleven real-world tasks, our proposed FALCON achieves state-of-the-art performance, consistently surpasses competitive baselines, and remains robust under clutter, spatial-prompt conditioning, and variations in object scale and height. Project page: https://falcon-vla.github.io/

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes FALCON, a Vision-Language-Action (VLA) architecture designed to enhance 3D spatial reasoning by injecting rich 3D spatial tokens—derived from a Spatial Foundation Model—into a dedicated "Spatial-Enhanced Action Head," rather than forcing them through the VLM (Vision Language Model) backbone. It introduces an "Embodied Spatial Model" (ESM) that (i) can operate RGB-only via VGGT-like spatial priors and (ii) optionally fuses depth and pose information without retraining or architectural changes, using a stochastic 3D-conditioning scheme to preserve robustness across modalities. Fusion strategies between the semantic action token and the spatial token were studied (cross-attention, FiLM-gated, element-wise addition), with element-wise addition selected for stability and generalization. Extensive experiments across CALVIN, SimplerEnv (WidowX/Google Robot), and 11 real-world tasks show that FALCON achieves SOTA or highly competitive results. Ablations demonstrate its modality transferability and support the claim that injecting tokens into the action head helps preserve VLM alignment.

### Strengths
1. Clear problem framing: Identifies the 2D–3D gap in VLA spatial reasoning and the modality transferability and alignment issues with prior explicit 3D pipelines and weak 3D cues.

2. Architectural novelty at the integration point: Instead of injecting spatial tokens into the VLM (which can disturb alignment), FALCON routes rich spatial tokens to a Spatial-Enhanced Action Head. This design decision is well-motivated and experimentally supported by a key ablation.

3. Practical modality transferability: RGB-only works via spatial foundation priors; optional depth/pose yields measurable gains, with a principled stochastic-conditioning scheme during training.

4. Extensive experiments: Strong results on CALVIN and SimplerEnv, clear real-world evaluation across 11 tasks, and targeted spatial capability tests (scale/height/abstract spatial prompts). Ablations on fusion strategies and token injection positions are informative.

5. Reproducibility-minded details: Training stages, losses, data splits, and hyperparameters are detailed; the ESM depth quality is quantitatively analyzed.

### Weaknesses
1. Positioning vs. contemporaneous 3D-enhanced VLAs: The related work section is broad, but could better differentiate FALCON from the latest VLA methods employing similar spatial decoupling strategies (e.g., avoiding direct modification of the VLM backbone). Clearer articulation of FALCON's specific innovations compared to potentially similar works and a systematic comparison of training costs (e.g., compute requirements) and data dependencies are recommended.

2. Theoretical grounding of fusion choice: The paper empirically finds "element-wise addition" to be the optimal fusion strategy but lacks theoretical support explaining this counter-intuitive result. Why does this non-parametric method outperform more complex parametric fusions (like Cross-Attention)? Further analysis, for instance, by quantifying the alignment drift (Δ) in the VLM's representation space under different fusion strategies, could provide deeper insight into why element-wise addition better preserves generalization.

3. Quantifying robustness: The proposed stochastic conditioning training strategy is an interesting design for enhancing modal robustness. However, the paper lacks a quantitative analysis of this robustness. It is recommended to add experiments that formally evaluate model performance under missing or noisy modal information (e.g., varying ratios r of valid depth pixels), potentially by plotting performance degradation curves.

4. Sensitivity to spatial prior source: FALCON's ESM module draws inspiration from VGGT. Given the existence of other spatial foundation models (like DUSt3R and MASt3R, cited in the paper), how sensitive is FALCON's performance to the specific choice of spatial prior model? Ablation studies, such as replacing the ESM backbone with DUSt3R or MASt3R, are suggested to test the generality of the FALCON architecture and assess potential domain adaptation issues.

5. Data efficiency substantiation: When evaluating few-shot generalization, the paper primarily reports final performance based on a fixed number (e.g., 20) of demonstrations. To provide a more comprehensive view of data efficiency, presenting learning curves—showing how success rate varies with the number of training demonstrations (e.g., 1, 5, 10, 20 demos)—is recommended to better illustrate the model's ability to learn from small samples.

### Questions
1. Could you quantify the representation drift in the VLM when spatial tokens are injected into the VLM vs. the action head (e.g., using CKA or feature-space distance to compare text-image alignment before/after injection)?

2. How sensitive are results to the Bernoulli probabilities p for depth/pose injection? Please provide a sweep and show their effect on RGB-only inference performance.

3. Can you provide success vs. number of real-world demonstrations curves for at least two tasks to substantiate few-shot efficiency?

4. Have you tried swapping VGGT with DUSt3R/MASt3R/CUT3R as the ESM backbone, holding everything else fixed? Was there any change in performance or modality transferability?

5. For long-horizon tasks (like CALVIN), what is the effect of the LSTM history length H and chunk size C on stability and success rate?

6. What is the runtime overhead (e.g., latency or FPS) of enabling the depth/pose inputs for the ESM compared to RGB-only inference on a single GPU?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
FALCON is a VLA that also accepts optional depth and camera pose inputs. These additional modalities can help with fine-grained manipulation tasks.  The approach will benefit from continued improvement in pointmap and camera pose estimation, and that technology is itself rapidly improving beginning with Dust3r). 

The proposed approach is to start with an off-the-shelf RGB + text VLA, Kosmos, and add a large downstream transformer that accepts the Kosmos original output plus optional additional depth and camera pose modalities. The majority of the paper discusses the design decisions around this second network, which the authors call the "Embodied Spatial Module" (ESM). Many of the design decisions build on VGGT (not worried about novelty; stating this to be clear). 

Results on CALVIN and SimpleBench allow comparisons to other VLAs, and results look strong. There are also some experiments on a real Franka.

Overall I feel the paper makes a useful contribution to the community. The empirical results are convincing but there are some small claims that do not feel fully substantiated and the paper could use some polish. With presentational changes I could be convinced to raise my score.

### Strengths
### Overall
Overall I feel the paper makes a useful empirical contribution -- especially if weights and code are released.

* Strong empirical results
* Straightforward method (this is a good thing!) with clear use case
* Clear description of the approach
* Good ablations of different design decisions
* Clear figures

Please don't take the length of this section an indicator. Mainly, these are the strengths that I'm seeing -- hopefully these are the same things the authors see too.

### Weaknesses
### Presentation
I found the paper a bit hard to read in its current form, and I feel the paper's impact may be limited by the writing and presentation.  Many parts of the paper could be tightened through removing unnecessary adjectives and reorganizing sections. 

**Organization:**
In the current version, even basic information about the training approach, including the losses, learning algorithm, and datasets are left entirely to the appendix. In contrast, the method section focuses almost exclusively on the architecture and has multiple paragraphs devoted to equations ablations the authors will run (e.g. Sec 2.4). 

In my opinion, this section could be reduced to equation 5 plus a little context, and the ablations could be moved to the experiments or appendix. This would make room for information about the training of FALCON.

**Wording**
Parts of the main paper are especially wordy and flowery.  Phrases like "Seamlessly integrating" are very common from LLMs but are hard to evaluate technically -- used three times in the paper. Wording like this could be cut to make space for some of the interesting ablations and details from the supplementary. I actually preferred the more direct writing in the supplementary!

---
### Experiments

**Cross-modality generalization**
There are lots of claims and experiments about cross-modality generalization, but I didn't find the experiments especially convincing and I don't feel these claims add much to the paper in their current form. Or maybe the authors could clarify the definition of "modality transferability" in the paper. 

I would like modality transferability experiments showing that a task could be learned (ABCD->D) entirely without depth, and then benefit from depth at test-time (and/or be learned with depth, and not suffer too much from losing depth at test-time) -- these would be more convincing than the current table 5.

However, I wanted to check my understanding about table 5, which the authors say L430-431
> "ESM achieves performance comparable to VGGT when using only RGB input. Moreover, its performance improves significantly when additional depth or camera pose information is accessible. This demonstrates the inherent strength of FALCON’s modality transferability"

VGGT is a pointmap estimation model (no actions), and based on the metrics, it looks like this is performance for depth estimation? So the ESM's depth estimation on CALVIN (which the model was trained on), is similar to VGGT (which VGGT wasn't trained on). And then when the GT depth is passed to the ESM, the depth estimation is close to perfect?

### Questions
* How come the authors add together imd + depth tokens (equation 4) and use this strategy, instead of concatenating and either dropping tokens or masking them (like in flow matching/diffusion)?
* When training and evaluating FALCON, do the authors use the depth and camera poses from the simulator/Realsense + IK? Or is VGGT used at test-time?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces FALCON, a spatial to action paradigm that injects richer and more representative 3D spatial tokens into VLAs through an improved injection scheme. The approach combines a 2D VLM backbone, an embodied spatial model for 3D encoding, and a spatial-enhanced action head for fusion. Experiments on CALVIN, SimplerEnv, and real-world tasks show better performance and improved modality transferability.

### Strengths
1. This paper focus on spatial reasoning in VLAs, which is a crucial research problem.
2. The experiments cover multiple benchmarks on both simulation and real-world. 
3. Clear qualitative visualizations are provided to support the contribution.
4. The pipeline is complete, which combines 2D VLM backbone, embodied spatial model for 3D encoding, and spatial-enhanced action head for fusion.

### Weaknesses
1. The core concept of this paper “3D spatial tokens” is ill-defined. This paper claims that 3D spatial tokens provide robust geometric priors. However, in line 132, the depth information is optional, and in line 224, the depth and/or pose is randomly injected. The resulting spatial tokens Tspl derives from DINO and an encoder with cross/self-attention. Thus, these spatial tokens are not truly 3D, they are 2D correlations.
2. In Eq. 4, they randomly inject depth and/or pose. The proposed stochastic conditioning breaks the well-defined mapping of Eq. 1. So, the expectations or joint distribution forms of F should be redefined.
3. The training pipeline requires further explanation. In appendix A.3, Stage 1 freezes all pre-trained components and optimize only the adapter parameters ΘD. The input for D is fixed (as ESM output is frozen), this training does not really learn cross-space alignment, it merely compresses the ESM's output. Stage 2 unfreezes VLM parameters ΘV and adapter parameters ΘD, while keeps ΘA and ΘG frozen. If my understanding is correct, here although VLM can learn new representations, the action control component is not adapted to new spatial features. So, the model cannot learn how spatial features influence actions, thus the Loss L cannot truly drive the entire system to learn the mapping from semantics and spatial information to action.

### Questions
1. As weakness 3, can you explain why freeze ΘA and ΘG in stage 2?
2. In Eq. 6, λ balances MSE and BCE losses, but its selection criteria are not described. Is λ chosen empirically, through grid search, or via adaptive re-weighting? How sensitive is the model’s convergence and performance to λ?
3. The ESM uses multi-task supervision, how each supervision type contributes to downstream VLA performance? What happens if one component is removed?
4. The pipeline adopts a two-stage post-training approach rather than joint training, how about joint end-to-end optimization? Why do two-stage?
5. Is the sampling probability p in Eq. 4 fixed over training?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a new VLA model, FALCON, which tries to incorporate the spatial 3D information into the VLA implicitly. The goal is to be able to utilize 3D information, if available, or otherwise be robust to different modality setups (full 3D, depth only, camera only, no 3D). The idea to achieve this is to utilize 3D feature information in models like VGGT, and additionally supply 3D information to the encoders of these models if they are available. Then, these spatial features are added to the action features before passing to the action head. This lets the action head utilize the 3D information for predicting actions. The paper conducts experiments on CALVIN, SIMPLER and real-world tasks. They show that the proposed method works better than prior 3D policies and 2D/3D VLAs, and the ablations show that the model works well in various modality setups and can leverage 3D information to improve performance, when available.

### Strengths
- The paper is very well-written and easy to follow
- The model design and ideas are clearly novel and overall make sense
- The problem setting is interesting and important — to utilize 3D information when available, but be robust in cases when it is not available. 
- The ablations are adequate and clearly show that the model is robust to various modality scenarios and can benefit form 3D data, when available.

### Weaknesses
My main concern is that the evaluation benchmarks chosen, especially the simulation ones are weak and do not necessarily benefit from 3D understanding and information. 

For instance, the paper makes this statement: “FALCON surpasses previous methods that rely on ground-truth point clouds (e.g., 3DDP (Ze et al., 2024) and 3D Diffuser Actor (Ke et al., 2024)), improving the Avg.
Len. by 4.13 and 1.05, respectively. This provides clear evidence of the effectiveness of our implicit
spatial information integration strategy” This statement appears wrong because the ablations in table-4 show that actually on Calvin, the performance is similar between RGB-only or RGB-D versions of their model, showing that additional 3D input doesn’t help much in this benchmark. And while it can be argued that FALCON’s implicit geometry is enough and that’s why explicit 3D doesn’t help, that is probably not true as the same ablations show that in real-world benchmarks performance increase from 60 to 80%. This makes me think that it is just that Calvin tasks don’t benefit much from 3D. Thus, it is unclear if the gains over prior methods especially the explicit 3D models like 3DDA is due to the implicit spatial information strategy or more parameters / better data. Concretely speaking, it is not convincing if implicit 3D, as proposed by this method, is enough or we need more richer 3D-aware models. 

It would be more convincing to show results on RL-Bench which contain several high precision tasks and where 3D policies (like 3DDA) have worked significantly better than 2D policies. Experiments there would help understand if the implicit 3D is strong enough or not.  


(Minor) 
Table-5 looks like it is testing monocular depth estimation and camera pose estimation — however that is not obvious from the description. It might help to explicitly say this

### Questions
As I describe in the weakness section, I suggest testing the proposed methods on richer settings like RL-Bench and compare with explicit 3D methods to eke out if the proposed implicit 3D representation is enough.

### Soundness
3

### Presentation
3

### Contribution
3
