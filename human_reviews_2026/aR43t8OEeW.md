# PhysLLM: Harnessing Large Language Models for Cross-Modal Remote Physiological Sensing

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Remote photoplethysmography (rPPG) enables non-contact physiological measurement but remains highly susceptible to illumination changes, motion artifacts, and limited temporal modeling. Large Language Models (LLMs) excel at capturing long-range dependencies, offering a potential solution but struggle with the continuous, noise-sensitive nature of rPPG signals due to their text-centric design. To bridge this gap, we introduce PhysLLM, a collaborative optimization framework that synergizes LLMs with domain-specific rPPG components. Specifically, the Text Prototype Guidance (TPG) strategy is proposed to establish cross-modal alignment by projecting hemodynamic features into LLM-interpretable semantic space, effectively bridging the representational gap between physiological signals and linguistic tokens. Besides, a novel Dual-Domain Stationary (DDS) Algorithm is proposed for resolving signal instability through adaptive time-frequency domain feature re-weighting. Finally, rPPG task-specific cues systematically inject physiological priors through physiological statistics, environmental contextual answering, and task description, leveraging cross-modal learning to integrate both visual and textual information, enabling dynamic adaptation to challenging scenarios like variable illumination and subject movements. Evaluation on four benchmark datasets, PhysLLM achieves state-of-the-art accuracy and robustness, demonstrating superior generalization across lighting variations and motion scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents PhysLLM, a framework that leverage large language models (LLMs) to improve remote photoplethysmography (rPPG) signal estimation. It involves a text prototype guidance strategy for promoting cross-modal alignment and a dual-domain stationary algorithm for solving signal instability.

### Strengths
- Proposes an attempt on robust LLM-based cross-modal framework for rPPG signal estimation.

### Weaknesses
- Unclear architecture and presentation. Some figures and subsections flow are difficult to follow. 

- Weak justifications on the framework motivations, especially why there is a need to use two LLMs just to output only the single and straightforward task/output (e.g., rPPG/HR)? Isn’t it too much more energy and latency for this task? 

- The reported performance improvements on MAE in many settings/datasets (including ablation studies) are just minor (<1 bpm) so it seems to be physiologically insignificant relative to computational and architectural complexity.  

- Lacks of diverse end-downstream evaluation, as only focus on simple HR task.  rPPG can be used for other tasks such as HRV, BP, stress detection, etc...

### Questions
- Please improve the presentation of the paper, such as: figure 2 (e.g., what is input of video encoder; unclear figure 2a), poor-quality figure 7. There is even “??” in line 190, and error vspace in 401. 

- Discussion on the contributions (and possible experiments on accuracy and efficiency) as comparing the work with Time-LLM [1], VL-phys [2], PhysDiff [3] both in intra and cross-dataset. 

- Please add more discussion/justifications on Lines 48-50, which is currently vague on why do we need text. For example, many general textual context are not related: how a general text like “The image features a young man with a beard, wearing a plaid shirt. He is looking at the camera” can help the LLM training to output rPPG/HR using Deepseek model? 

- Following that, discussion on to what extent is it necessary to include LLMs usage on the text processing here for the task instead of earlier language models.  

- Provide diverse end-downstream tasks. For example, extending the experiments more than just simple HR estimation would be a lot better (e.g., Question Answering or Reporting on more complex physiological parametters and health) at least can show more applications and more leverage the power of the given LLMs.

[1] Jin, Ming, et al. "Time-llm: Time series forecasting by reprogramming large language models." arXiv preprint arXiv:2310.01728 (2023). 

[2] Yue, Zijie, et al. "Bootstrapping Vision-Language Models for Frequency-Centric Self-Supervised Remote Physiological Measurement." International Journal of Computer Vision (2025): 1-22. 

[3] Qian, Wei, et al. "PhysDiff: Physiology-based Dynamicity Disentangled Diffusion Model for Remote Physiological Measurement." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 6. 2025.

### Soundness
2

### Presentation
2

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
PhysLLM introduces a novel framework for cross-modal remote physiological sensing by synergizing LLM with domain-specific rPPG components. The method tackles the challenge of extracting reliable physiological data from facial videos, which is often compromised by visual noise, lighting changes, and motion artifacts. Key contributions include a Text Prototype Guidance strategy for projecting physiological features into an LLM-friendly semantic space, and a Dual-Domain Stationary algorithm for stabilizing signals via adaptive time–frequency domain re-weighting. The results highlight the promise of combining LLM reasoning with specialized signal processing for resilient and interpretable health monitoring applications.

### Strengths
1. Novel cross-modal architecture: The paper introduces PhysLLM, a framework that integrates text, vision, and physiological signals with ideas like Text Prototype Guidance and adaptive cue prompting, representing a meaningful conceptual advance for rPPG.

2. Strong performance across datasets: It shows consistent improvements in both intra-dataset and cross-dataset generalization.

### Weaknesses
1. Lack of comparison on model size: Integrating LLMs significantly inflates parameter count and computational cost compared to prior rPPG methods. Since remote physiological sensing has strong real-time and mobile deployment requirements, storage and latency overhead are critical. The paper should provide model size, trainable parameter count, and inference efficiency comparisons with existing methods.

2. Effectiveness of semantic info: The LLM backbone is positioned as the central source of improved robustness, yet the paper does not isolate the benefit of semantic reasoning from general long-sequence modeling. It remains unclear whether the LLM performs meaningful language-grounded reasoning, or primarily acts as a temporal encoder with additional embedding capacity. A comparison replacing the LLM with similarly sized non-linguistic temporal architectures (e.g., ViT-based rPPG backbones, GPT-style time-series forecasters) would be important to confirm language contributes beyond generic model scale.

3. Statistical cue and prompt generation concerns: Cue captions depend on LLaVA-generated descriptions of a single frame, which may be noisy, biased by appearance (skin tone, beard), or misaligned with rPPG-relevant dynamics. The paper lacks ablations on prompt errors or failure cases (ex. when the cues are all incorrect and provide false information to the LLM).

### Questions
1. Section 4.3's title needs to be rearranged.

2. Why is PhysFormer++ excluded from several comparisons? Given its strong performance, including it in all datasets would strengthen claims of state-of-the-art accuracy.

3. How robust are cue captions in unconstrained environments? What happens if visual descriptions are inaccurate due to occlusion, masks, extreme lighting, or motion blur?

4. Since rPPG is typically real-time and mobile-oriented, can you provide inference efficiency metrics such as frames-per-second and VRAM usage?

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
The paper proposes PhysLLM, a framework that integrates large language models with rPPG pipelines. It introduces several interesting modules, such as Dual-Domain Stationary signal stabilization, Text Prototype Guidance for cross-modal alignment, and Physiological Cue-Aware Prompt Learning for leveraging task-specific priors. The work is technically ambitious and attempts to bridge physiological signal modeling with vision-language reasoning.

### Strengths
1. The idea of introducing LLMs into rPPG estimation is timely.
2. The framework is clearly described, and the modular design (DDS, TPG, APL) is logically motivated.
3. Writing quality and figures are generally good, helping the reader follow the methodology.

### Weaknesses
1. Limited benchmark coverage: the evaluation misses some widely recognized and more challenging datasets (e.g., V4V, VIPL-HR), which are important for validating cross-domain robustness in practical settings.
2. The choice of LLM backbone (DeepSeek-1.5B) and its integration details are only briefly discussed, it is unclear whether improvements mainly come from the LLM or other architectural refinements.
3. Some modules (e.g., APL and TPG) would benefit from clearer ablation or visualization to show how textual cues truly influence signal recovery.
4. The cross-dataset generalization results, while improved, are still marginal and may not fully demonstrate strong domain invariance.

### Questions
1. Could the authors clarify how much of the performance gain comes from the LLM integration itself, compared to the added DDS and TPG modules? An ablation or visualization isolating the LLM’s role would help.
2. The evaluation does not include large and diverse datasets such as VIPL-HR or V4V. Could the authors comment on how PhysLLM would generalize to those more complex scenarios?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes PhysLLM, a framework for remote photoplethysmography (rPPG) that couples a CNN-based rPPG backbone with an LLM through three key components: (i) a Dual-Domain Stationary (DDS) algorithm that stabilizes signals via complementary time- and frequency-domain smoothing with adaptive weighting, (ii) a Vision Aggregator (VA) that fuses multi-scale visual features using cross/self-attention, and (iii) Text Prototype Guidance (TPG) that aligns rPPG and visual features with a compact set of text prototypes to interface with an LLM.

### Strengths
1.	Strong empirical results. SOTA intra-dataset HR with gains on BUAA and MMPD. 
2.	Ablations and component evidence. Removing DDS/VA/TPG degrades metrics; including all yields the best performance, supporting the importance of each piece. 
3.	Task priors + LLM prompting. The cue design (task, visual, static/statistical) and adaptive fusion are thoughtful ways to inject physiological context into an LLM.

### Weaknesses
1.	Justification for LLM vs. sequence models. It remains unclear whether the LLM is essential beyond acting as a powerful sequence model; comparisons to strong non-LLM long-context baselines (e.g., state-of-the-art time-series Transformers without language pretraining) are missing. 
2.	Compute/latency & deployability. The paper does not quantify training/inference cost (LLM size, parameter-efficient tuning specifics, throughput on typical devices) or real-time feasibility, which is critical for rPPG applications.
3.	Data and evaluation breadth. While four benchmarks are covered, additional stress tests (e.g., extreme motion, low-light, occlusion regimes) and fairness (skin tone, age, gender) would strengthen claims of robustness and generalization.
4.	The comparison with the previous rPPG using LLM shown below is missing.

Yue, Z., Shi, M., Wang, H. et al. Bootstrapping Vision-Language Models for Frequency-Centric Self-Supervised Remote Physiological Measurement. IJCV.

### Questions
Please check the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3
