# OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
General-purpose robots capable of performing diverse tasks require synergistic reasoning and acting capabilities.
However, recent dual-system approaches, which separate high-level reasoning from low-level acting, often suffer from challenges such as limited mutual understanding of capabilities between systems and latency issues. 
This paper introduces OneTwoVLA, a single unified vision-language-action model that can perform both acting (System One) and reasoning (System Two). Crucially, OneTwoVLA adaptively switches between two modes: explicitly reasoning at critical moments during task execution, and generating actions based on the most recent reasoning at other times.
To further unlock OneTwoVLA's reasoning and generalization capabilities, we design a scalable pipeline for synthesizing embodied reasoning-centric vision-language data, used for co-training with robot data. We validate OneTwoVLA's effectiveness through extensive experiments, highlighting its superior performance across four key capabilities: long-horizon task planning, error detection and recovery, natural human-robot interaction, and generalizable visual grounding, enabling the model to perform long-horizon, highly dexterous manipulation tasks such as making hotpot or mixing cocktails.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a single unified VLA model capable of both reasoning and acting, and adaptively switching between these two modes. It proposes a pipeline for synthesizing embodied reasoning-centric vision-language data to further enhance the model’s reasoning and generalization capabilities.

### Strengths
The paper is clearly written. The experimental design is appropriate and effectively demonstrates the proposed model’s capabilities. The model shows promising performance, particularly on long-horizon tasks, owing to the integration of reasoning and action within a unified framework.

### Weaknesses
Please refer to questions section.

### Questions
1. In Section 3.1, how is the adaptive inference capability learned? Is it acquired implicitly through training, or explicitly supervised with ground-truth signals? If it is the latter, how are such ground-truth labels defined and provided?

2. In Figure 2, the blue and green blocks are difficult to interpret. For instance, I understand the green part as representing System 1 and the blue as System 2. However, it is unclear why System 1 is further divided into instruction and reasoning inputs. How are these two types of inputs related to the two systems? The figure needs to be clarified.

3. The error detection capability is emphasized throughout the paper ( from the instruction to the conclusion) yet it is only qualitatively analyzed in Section 4.2, which is insufficient. If this ability is to be highlighted as a key contribution, it should be supported by quantitative evidence; otherwise, it would be better not to overemphasize it.

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
5

### Summary
OneTwoVLA introduces a VLA framework that allows a single model to adaptively decide when to reason and when to act, bridging high-level reasoning and low-level control. It employs special decision tokens ([BOR]/[BOA]) to autonomously switch between reasoning and acting modes. Experiments on long-horizon real-world manipulation tasks demonstrate that OneTwoVLA achieves superior success rates.

### Strengths
S1. It simultaneously equips the VLA model with reasoning capability and action prediction ability, which is important for robotics.

S2. The authors conducted a large number of experiments, including "out-of-lab" real-world tests. It is amazing.

S3. The writing is very thorough and careful, with a large appendix experiments and visualizations included.

### Weaknesses
W1. Lack of key ablations. The paper lacks experiments that use different reasoning components as conditions, e.g., using only “plan” or “Now I need to …”. Not every lab has the ability to collect all key-step annotations, and I also do not believe every reasoning element contributes positively to all tasks. For instance, in dynamic environments, is the “Historical Summary” really helpful? And in complex scenes, could detailed “scene descriptions” actually introduce unnecessary redundancy?

W2. Impact on generalization. Would using different subsets of reasoning outputs as conditions affect the model’s generalization ability?

W3. Table 14 only compares inference time across reasoning lengths. Did the authors investigate how different reasoning lengths affect manipulation accuracy? Also, what reasoning token length was used in the main experiments?

W4. What is the additional time cost introduced to the execution head by different reasoning token lengths?

W5. If the number of condition tokens increases, it may significantly slow down the DDIM output efficiency. In addition, could the authors provide a real-robot control frequency comparison experiment to support this?

W6. Since the authors already built an automated data labeling pipeline, could it be extended to modify pretraining datasets such as OXE? This could enable the model to develop reasoning-and-action collaboration abilities across broader domains, not just in the downstream tasks.

### Questions
Q1. In the out-of-lab experiments, under such complex backgrounds, especially when using a GoPro with a wide field of view, is it truly possible to achieve stable manipulation? Is this manipulation capability mainly derived from the pretrained knowledge of the base model?

Q2. The appendix layout could be improved, it contains excessive blank space and is overly long, making it difficult for reviewers to find key information efficiently.

Q3. It would be helpful if the authors could add a discussion comparing with Pi_0.5 and ThinkAct. Of course, a quantitative comparison is not strictly necessary.

Q4. This paper presents a very good idea, but it lacks detailed exploration and validation. I will adjust my score based on the authors’ response.

### Soundness
3

### Presentation
3

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
This paper presents OneTwoVLA, a unified vision-language-action model capable of both reasoning (System Two) and acting (System One) within a single framework. Unlike prior dual-system approaches that separate high-level reasoning from low-level control, OneTwoVLA adaptively switches between reasoning and action—engaging explicit language reasoning at critical moments while efficiently generating actions otherwise. The authors further propose a scalable pipeline for synthesizing embodied, reasoning-centric vision-language data to co-train with real robot data. Extensive experiments demonstrate that OneTwoVLA significantly outperforms existing methods across four key dimensions: long-horizon task planning, error detection and recovery, natural human-robot interaction, and generalizable visual grounding, highlighting the promise of unified models for embodied intelligence.

### Strengths
OneTwoVLA’s main strengths lie in its elegant unification of reasoning and action within a single adaptive model, which effectively eliminates the latency and coordination issues common in dual-system designs. It also demonstrates impressive generalization by leveraging a scalable pipeline that synthesizes embodied reasoning-centric vision-language data, significantly enhancing performance on unseen tasks. Moreover, the model enables natural and context-aware human-robot interaction, showing a level of interpretability and adaptability in current robotic systems.

One more thing:  The authors provide highly detailed supplementary content, including rich qualitative examples, visualizations, and additional experiments, which greatly improve the clarity and reproducibility of the work.

### Weaknesses
1. The reasoning-centric synthetic dataset appears essential to the model’s overall performance, but it remains unclear how much it directly contributes to robotic manipulation capabilities. An explicit ablation study comparing models trained with and without this reasoning data — or analyses showing how it benefits manipulation compared to standard vision-language data — would make the claim far more convincing.

2. The model’s generalization mainly lies in reasoning, while action-level generalization remains limited. Given that large language models already generalize well in reasoning for simple robotic tasks, it would be more meaningful to explore whether OneTwoVLA can generalize in low-level manipulation, such as grasping unseen objects.

### Questions
1. Is the sequential dual-phase design (reasoning first, then acting) fundamentally necessary? Have the authors considered a mechanism that allows simultaneous reasoning and acting, similar to how humans can reason while performing actions, rather than pausing execution each time reasoning is invoked? It would be valuable for the authors to discuss how such an asynchronous or parallel architecture might be designed and what challenges it would introduce.

2. Regarding the synthetic data filtering process, line 1536 mentions that incorrect images are excluded from evaluation. Could the authors clarify how this filtering was conducted — was it done manually, automatically, or through a hybrid approach? More detail on the evaluation pipeline and its reproducibility would improve transparency.

If the authors can address these concerns or provide more evidence supporting their design choices, I would consider raising my score.

### Soundness
2

### Presentation
4

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
- This paper introduces OneTwoVLA, a vision-language-action model that can perform both acting and reasoning.

- OneTwoVLA adaptively reasons at critical moments during execution (e.g., upon completing subtasks, detecting errors, or requiring human inputs).

- This paper also designs a pipeline for synthesizing embodied reasoning-centric vision-language data, used for co-training with
robot data.

### Strengths
- Improving the reasoning capabilities of current embodied systems is crucial, and enabling adaptive reasoning is particularly insightful.

- OneTwoVLA is highly practical, but it would be better if the authors could validate its generalization ability on public benchmarks.

### Weaknesses
- OneTwoVLA inherits the model architecture of pi0 (VLM for reasoning and flow-matching policy for acting). Why is it described as a single unified vision-language-action model?

- Regarding the issue raised by the authors that "System Two may produce intermediate contents that System One cannot execute," how does OneTwoVLA address this problem? Why does the standard two-system VLA suffer from limited mutual understanding, while OneTwoVLA does not?

- In my view, compared to previous works, OneTwoVLA does not introduce architectural innovations; essentially, it still relies on a VLM for reasoning and subtask generation, with an action expert responsible for execution. The main difference lies in the fact that the authors heuristically designed certain scenarios requiring re-planning and annotated corresponding data to fine-tune pi0.

- Currently, there are many works focused on embodied reasoning. Could the authors evaluate OneTwoVLA on some public benchmarks to demonstrate its multimodal reasoning and planning capabilities? For example, the following works:

[1] RoboBrain 2.0 Technical Report

[2] Embodied Arena: A Comprehensive, Unified, and Evolving Evaluation Platform for Embodied AI

- Could the authors report the reasoning time overhead introduced by adaptive reasoning? For the baseline, is it possible to improve performance by increasing the frequency of System 2 execution?

### Questions
Please see the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
