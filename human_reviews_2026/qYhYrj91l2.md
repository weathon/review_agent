# VividCam: Learning Unconventional Camera Motions from Virtual Synthetic Videos

- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Although recent text-to-video generative models are getting more capable of following external camera controls, imposed by either text descriptions or camera trajectories, they still struggle to generalize to unconventional camera motions, which is crucial in creating truly original and artistic videos. The challenge lies in the difficulty of finding sufficient training videos with the intended uncommon camera motions. To address this challenge, we propose VividCam, a training paradigm that enables diffusion models to learn complex camera motions from synthetic videos, releasing the reliance on collecting realistic training videos. VividCam incorporates multiple disentanglement strategies that isolates camera motion learning from synthetic appearance artifacts, ensuring more robust motion representation and mitigating domain shift. We demonstrate that our design synthesizes a wide range of precisely controlled and complex camera motions using surprisingly simple synthetic data. Notably, this synthetic data often consists of basic geometries within a low-poly 3D scene and can be efficiently rendered by engines like Unity. Our video results can be found in https://anonymoususers196.github.io/VividCamDemo/ .

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes VividCam, a framework for training video diffusion models to generate unconventional camera motions using only synthetic training data rendered from simple low-poly 3D scenes in Unity. The method employs a dual-adaptation training scheme with appearance and camera LoRAs, optical flow loss, and style-aligned prompts to disentangle synthetic appearance from camera motion learning.

### Strengths
1. Good qualitative results showing diverse motions
2. Both text-based and trajectory-based control variants
3. Ablation studies on key components

### Weaknesses
1. Only tested on one base model (CogVideoX-5B).
2. No failure case analysis
3. Baselines (CameraCtrl, AC3D) weren't trained on the same synthetic data, making comparison unfair
4. Besides the camera control metrics (TransErr, RotErr), there only one appearance-related metric FVD, which is not enough, more metrics from different aspects are needed, for example the metric used to measure the dynamic degree.
5. The main technique innovations, LoRA tuning to avoid appearance leaky is widely used in many different image / video generation methods.
6. The dataset only focus on single domain, leading to low generalization capacities when used to generate other domain's videos.

### Questions
1. Since the CogVidX and VIVIDCAM-COG are text based camera control models, how to measure the transerr and roterr in Table 1?
2. Why only use 500 real and 500 synthetic videos for ablating the effectiveness of joint training?
3. What does the validation set consist of?
4. Why does Table 5 show mixed data performs worse than virtual-only data? This contradicts intuition and needs investigation.

### Soundness
2

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
The paper proposes a method to enhance existing video generation models to support unconventional camera controls. The proposed method first renders synthetic 3D scenes from unity with unconventional camera movements. The rendered scenes are low-poly for easy construction. Then, the paper discussed ways to disentangle the learning of camera movement from the learning of appearance. Specifically, the method first trains a LoRA to only learn the appearance. Then the method learns the camera movement. At inference, the appearance LoRA is dropped to enable unconventional camera control while generating realistic videos. The qualitatively results shows that the method can perform more arbitrary camera movements compared to prior methods such as CameraCtrl. The quantitative metrics also shows that the model has greater alignment with the camera condition.

### Strengths
1. The paper is clearly written, with good inspiration for the problem that it tries to tackle. The method description and the evaluation results are presented clearly.

2. The proposed finetuning method is sound and straightforward. The authors also considered the practical difficulty of data curation and decided to use low-poly and simple objects in the rendering pipeline.

3. Evaluation results clearly show the effectiveness of the method. The model can perform more unconventional camera movements compared to prior art.

4. The ablation studies show the effectiveness of the appearance LoRA, as removing the appearance LoRA causes degradation in appearance, justifying the method design.

### Weaknesses
1. The method requires the curation of a synthetic dataset. Alternatively, I do believe unconventional/artistic camera movements can be found in massively available gaming videos and movies, which can be extracted and used as conditions following general methods such as CameraCtrl2. It is a trade-off between scaling data and scaling manual work. This is a general weakness of methods requiring synthetic data, not a reflection on the novelty of the method proposed.

2. The authors should consider including related work [1], which also explores fine-tuning video generation models with a synthetic video dataset. Although the work doesn't directly focus on camera control, it has also demonstrated that synthetic videos can be used to support new camera motions.

3. The paper proposed "optical-flow-based" loss in equation 5. First, I think calling it optical-flow-based may be misleading. The loss is more like a temporal derivative consistency loss. But regardless of the naming, which is not a main concern, the paper does not seem to present an ablation study on this proposed loss term.

[1] Synthetic Video Enhances Physical Fidelity in Video Synthesis (ICCV2025)

### Questions
1. Regarding weakness 1, although the author has considered using low-poly to simplify the data curation process, the scene still requires textures (sky, ground), and objects, etc. What is the minimum extent that the scene can be? Can it be plain colors? Can it be wireframes?

2. Regarding weakness 3, it would be nice if the author could provide supporting justifications for the use of the loss term.

3. Are you planning on open-sourcing the curated data or synthetic scene generation code?

### Soundness
3

### Presentation
4

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
The paper proposes VIVIDCAM, which uses synthetic video with diverse camera motion to fine-tune video generation models to generate camera motion. With several disentanglement strategies, VIVIDCAM can learn robust motion representation from synthetic videos. Experiments show that the proposed method could generate real videos with unconventional motions.

### Strengths
1. The proposed method provides an efficient way to enable video generation models to generate diverse camera motions.
2. Extensive experiments are conducted to show the effectiveness of the proposed method.

### Weaknesses
1. There are four techniques for disentanglement, ie, dual-adaptation training, data with and without camera motion, optical-flow based loss, and special text prompt. However, the contribution of each one is not well present in the paper, although the ablation has explored two of them.
2. The training setting of the comparing methods and the proposed method seems to be different. Are the comparing method trained on the same data?
3. The training and testing combination of motion is unclear. What are the motion combinations from Table 1 in testing? Are these combinations seen in the training? Can the methods generalize to unseen combinations or even unseen motion (not defined in Table 1)?
4. The novelty of the proposed method is limited. It combines four techniques for disentanglement but doesn't provide new insights or methods for this task.

### Questions
1. What is the individual contribution of the techniques for disentanglement?
2. What is the training setting of the proposed method?
3. Waht are the motion combinations in the training and testing?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes VIVIDCAM, a method for training text-to-video diffusion models to follow specific camera motions, particularly "unconventional" ones for which real-world training data is scarce. The core idea is to use low-poly, synthetic videos (rendered in Unity) as the training data.

To prevent the synthetic, "low-poly" appearance from bleeding into the final output, the authors propose a dual-adaptation training strategy. First, an "appearance LoRA" is trained on static synthetic videos to learn and isolate the synthetic visual style. This training is guided by a special "virtual indicator" tag in the prompt. Second, a camera motion module (either a LoRA for text-based control or fine-tuning a trajectory encoder) is trained on synthetic videos that do contain camera motion.

At inference time, the appearance LoRA is discarded, and the model is expected to produce realistic videos that follow the camera motions learned from the synthetic data. The method is evaluated on text-based and trajectory-based camera control, using automated metrics (FVD, TransErr) and human studies.

### Strengths
The problem itself is significant. Enabling controllable, complex, and artistic camera motions is a key challenge for creative video generation, and the lack of diverse, well-labeled real-world data is a real bottleneck.

The idea of using synthetic data is a logical approach to solving this data scarcity problem.

### Weaknesses
The paper suffers from a significant lack of novelty, a mismatch between its tools and goals, and an unconvincing evaluation.

1. Critical Lack of Novelty: The central contribution, the "dual adaptation" or "dual LoRA" method for disentangling appearance from motion, is not new. This exact training pattern was established by AnimateDiff (Guo et al., 2023), which the authors cite as "inspiration." AnimateDiff's "Stage 1: Domain Adapter" is functionally identical to this paper's "Step 1: Appearance Adaptation." VIVIDCAM is a direct application of this existing 2-stage (adapter-then-motion) training strategy to a synthetic dataset. This is an incremental extension, not a novel framework, and represents a major overstatement of the paper's contribution.

2. Mismatch of Tool and Goal: The paper uses a powerful 3D rendering engine (Unity) capable of generating perfect, precise, 6-DOF camera trajectories (i.e., extrinsics). However, a large part of the paper focuses on using this data to train a model on vague, ambiguous text prompts like "pan left" or "push in." This is a baffling waste of the synthetic data's primary advantage. It's a step backward from existing works (including its own baselines like AC3D and CameraCtrl) that are focused on fine-grained, trajectory-based control.

3. Unconvincing Rationale: The paper motivates the work by the need for "unconventional" motions. However, the motions listed in Table 1 ("Simple" and "Composed") are entirely conventional (push, tilt, truck). These are readily available in existing real-world datasets. While the "Complex" motions (e.g., "seek object") are more interesting, training them from ambiguous text prompts is far less compelling than training a model to follow an explicit 3D "seeking" path, which Unity could have easily generated.

4. Insufficient Evaluation & Unconvincing Ablations: The primary evidence for the core disentanglement claim rests on the ablation in Figure 5. This figure not only fails to show temporal artifacts (by only showing static frames), but the visual difference between the full model and the "w/o Style-aligned Prompts" version is minimal. This does not convincingly demonstrate that the appearance LoRA and [VIRTUAL] tag are doing the heavy lifting the authors claim. Furthermore, the paper lacks direct side-by-side video comparisons to its baselines, making it impossible to truly judge the trade-off between motion accuracy and visual fidelity.

5. Poor Qualitative Fidelity: When compared to the baselines (e.g., vanilla CogVideoX or AC3D), the visual performance of the proposed method appears to be a significant regression. The generated videos, while perhaps following the motion, are less realistic and have lower fidelity than the state-of-the-art baselines they are built upon. The method seems to trade visual quality for motion control, which is a poor trade-off.

### Questions
1. The camera motion module (LoRA/encoder) is also trained with the <VIRTUAL> tag in the prompt. At inference, this tag is removed. How do you know the motion module isn't confused or degraded, as it was never trained on prompts without this tag? A key ablation is missing: what happens to motion accuracy if you keep the tag at inference (accepting the synthetic output)?

2. Given that AnimateDiff's "Domain Adapter" + "Motion Module" training pipeline is functionally identical to this paper's, could the authors please state clearly what the technical novelty of this paper is, beyond just applying this pattern to a new synthetic dataset?

3. Why was the decision made to degrade the perfect 3D trajectory data from Unity into coarse text prompts? Why not use the synthetic data for its primary strength and train a model that takes explicit 3D paths, target-of-interest coordinates, or specific orbit/dolly-zoom parameters as input?

4. Table 1 lists "Dolly zoom" as a capability. A true Dolly zoom has a specific mathematical definition (simultaneously changing focal length and camera distance to keep the subject size constant). Can the text-based model actually reproduce this effect from a simple prompt, or is it just generating a standard "zoom in"?

5. Could the authors provide a more in-depth analysis of why the appearance LoRA, trained only with a [VIRTUAL] tag, is sufficient to "absorb" all synthetic artifacts? The mechanism seems very similar to DreamBooth (specializing to a token) but is used for a different purpose (disentanglement), and this is not well-justified.

### Soundness
2

### Presentation
2

### Contribution
1
