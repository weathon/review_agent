# Pixel Motion as Universal Representation for Robot Control

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
We present LangToMo, a vision-language-action framework structured as a dual-system architecture that uses pixel motion forecasts as intermediate representations. 
Our high-level $\textit{System 2}$, an image diffusion model, generates text-conditioned pixel motion sequences from a single frame and past motion to guide robot control.
Pixel motion—a universal, interpretable, and motion-centric representation—can be extracted from videos in a weakly-supervised manner, enabling diffusion model training on any video-caption data.
Treating the generated pixel motion as largely embodiment-agnostic $\textit{universal representations}$, our embodiment-aware $\textit{System 1}$ module translates these into robot actions via motion-to-action mapping functions, which can be either hand-crafted or learned with minimal supervision.
System 2 operates as a high-level policy applied at sparse temporal intervals, while System 1 acts as a low-level policy at dense temporal intervals.
This hierarchical decoupling enables flexible, scalable, and generalizable robot control under both unsupervised and supervised settings, bridging the gap between language, motion, and action.
Visualizations at https://anonymous.4open.science/w/LangToMo.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
LangToMo is a dual-system vision-language-action framework that uses text-conditioned pixel motion forecasts (via an image diffusion model) as universal, interpretable intermediate representations. A high-level System 2 generates pixel motion from a single frame and past motion, while a low-level, embodiment-aware System 1 maps this motion to robot actions through motion-to-action functions (hand-crafted or minimally supervised), enabling flexible, scalable, and generalizable control across unsupervised and supervised settings.

### Strengths
- This paper investigates a new hierarchical policy approach that uses pixel motion to interface between high- and low-level policies. It addresses an important area and offers insightful directions for future research.
- Writing: The paper is well written, the method is clearly presented, and the figures/tables are complete and easy to read.

### Weaknesses
1. **Motivation**: I am uncertain about the authors’ motivation for using pixel motion to construct a hierarchical policy. Prior dual system approaches (e.g., HiRT[1], LCB[2], OpenHelix[3]) leverage interactions between large and small models to increase control frequency, yet the paper does not explain how its hierarchical design relates to or improves upon this line of work. In addition, the authors should compare against these methods and clarify the advantages and drawbacks of using pixel motion versus other latent representations as the interface between System 1 and System 2.

[1] Zhang J, Guo Y, Chen X, et al. Hirt: Enhancing robotic control with hierarchical robot transformers[J]. arXiv preprint arXiv:2410.05273, 2024.

[2] Shentu Y, Wu P, Rajeswaran A, et al. From llms to actions: Latent codes as bridges in hierarchical robot control[C]//2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024: 8539-8546.

[3] Cui C, Ding P, Song W, et al. Openhelix: A short survey, empirical analysis, and open-source dual-system vla model for robotic manipulation[J]. arXiv preprint arXiv:2505.03912, 2025.

2. Compared to prior approaches that predict visual traces, the novelty of this work appears limited, and the authors have neither adequately justified nor empirically validated the advantages of pixel motion over visual trace/optical flow.

3. The experimental results are weak, especially on simulation benchmarks against advanced baselines. The appendix shows only moderate performance on CALVIN, and results on other environments (e.g., SimplerEnv) are missing.

### Questions
Refer to weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents LangToMo, a two-stage framework for predicting robot motion using pixel movement as an intermediate representation.
LangToMo consists of two systems:
(i) System 2 employs a diffusion-based model to generate pixel motion (PM), pretrained on the OpenX dataset and fine-tuned on downstream task demonstrations;
(ii) System 1 maps actions conditioned on the predicted pixel motion.
Experimental results demonstrate that LangToMo outperforms baseline methods on both Meta-World and real-world robotic manipulation tasks.

### Strengths
- The proposed two-stage framework preserves the original model’s capabilities while enabling the transformation from vision-language signals to action representations.
- Compared to related work, LangToMo employs a diffusion model to directly predict pixel motion instead of generating full video sequences.
- Surpass other baseline method in real world zero-shot tasks via large-scale pretraining.

### Weaknesses
- **Longer inference latency**
  
  Similar to UniPi, many steps of denoising are required when the diffusion model predicts pixels or PMs, which leads to long inference delays and false closed-loop control, which limits the model to static scenes.

- **Weak evaluation**  
  Choosing Metaworld benchmark in main text experiment for VLA models is less convincing. Metaworld tasks and scenarios are relatively simple, and accurate action prediction can be achieved using images alone without requiring text. The supplementary material shows that Calvin's experimental results are worse than those of VPP, which only performs pixel predictions. The reasoning seems insufficient ( VPP performs better even without using large amounts of data for cotraining). Therefore, additional ablation experiments are needed to clarify that the poorer performance is due to model size.
- **Real-world task problems**  
  The project link cannot be opened and the real-world video results cannot be seen. From the experimental content in the text, it seems that the scenes and tasks in real-world setting are relatively simple.

### Questions
1. Is the poor performance of Calvin due to the model size or the pipeline structure? You can add PM prediction channel to the SVD with post-training to make a fair compare with VPP in Calvin.
2. What is the failure case in the Calvin rollout? Is it due to semantic understanding causing PM prediction errors (wrong movement direction) or inaccurate action head mapping?
3. Please provide the frequency of your model deployment on real world.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes to use pixel motion as a control interface. System 1 translate language into pixel motions while system 2 translate the motions into robot actions. The authors show performance gain in MetaWorld experiments.

### Strengths
The writing is clear and easy to follow.

The method is able to leverage the unlabeled human data and enable scaled learning.

### Weaknesses
The novelty is limited. Many papers have explored the idea of extracting universal action representation from videos.

The performance is only evaluated on MetaWorld and the performance gain is marginal compared to ATM. More evaluations are needed.

See questions below.

### Questions
(1)	Another line of works that using latent actions to capture the pixel motions is missing, including works like LAPA, IGOR, Villa-X, UniVLA, etc. Comparison and discussion are needed.

(2)	As for the experiments, the author pretrain the models on the robot data (L361) and finetune the models on robot and human data. However, the pixel motions, the setting should be that we have many unlabeled video data (of human) and a small amount of labeled robot data.

(3)	In Table 5, why running two systems at the same frequency will lead to a performance drop?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes LangToMo, a dual-system vision-language-action (VLA) framework that uses pixel motion (optical flow) as a universal intermediate representation for robot control. System 2 is a language-conditioned diffusion model that predicts pixel motion from a single image and instruction and system 1 is a  mapping function that converts the generated pixel motion into robot actions.

### Strengths
1. The paper identifies a key bottleneck in robot learning from videos: the need for action supervision and embodiment-specific data. The idea of treating pixel motion as a universal, interpretable, and embodiment-agnostic abstraction is good.
2. Dual-system design also satisfied the real-time issue of robot policy

### Weaknesses
1. Although the idea of using pixel motion as action representation is nice, I feel the idea is widely studied in previous work. The author list the difference with previous works at Table 1. I feel the idea is a little bit incremental.
2. For the simulation experiments, the author only did experiments on 11 Metaworld benchmarks tasks, which is limited. Many previous works train language conditioned policy on the whole Metaworld benchmark. Also, Metaworld is not designed for language-conditioned tasks, maybe run methods on Calvin or Libera can better verify the effectiveness of the method.

### Questions
1. Could you include comparisons with more advanced vision-language-action (VLA) models? Since the proposed approach ultimately produces a language-conditioned policy, it should also be evaluated against general VLA policies, not only motion-based ones?

### Soundness
2

### Presentation
3

### Contribution
2
