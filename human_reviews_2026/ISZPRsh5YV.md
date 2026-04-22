# MotionSight: Boosting Fine-Grained Motion Understanding in Multimodal LLMs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Despite advancements in Multimodal Large Language Models (MLLMs), their proficiency in fine-grained video motion understanding remains critically limited. They often lack inter-frame differencing and tend to average or ignore subtle visual cues. Furthermore, while visual prompting has shown potential in static images, its application to videos' temporal complexities, particularly for fine-grained motion understanding, remains largely unexplored. We investigate whether inherent capability can be unlocked to boost MLLMs' motion perception and enable distinct visual signatures tailored to decouple object and camera motion cues. In this study, we introduce $\mathtt{MotionSight}$, a novel zero-shot method pioneering object-centric visual spotlight and motion blur as visual prompts to effectively improve fine-grained motion understanding without training. To convert this into valuable data assets, we curated $\mathtt{MotionVid-QA}$, the first large-scale dataset for fine-grained video motion understanding, with hierarchical annotations including SFT and preference data, $\Theta{(40K)}$ video clips and $\Theta{(87K)}$ QAs. Experiments show $\mathtt{MotionSight}$ achieves state-of-the-art open-source performance and competitiveness with commercial models. Using $\mathtt{MotionVid-QA}$, we fine-tuned $\mathtt{MotionChat}$ on Qwen2.5VL-7B, which attains 48.3\% overall accuracy on FAVOR-Bench that is comparable to Qwen2.5VL-72B's 48.1\%. In summary, we present a novel zero-shot method and a large-scale, high-quality dataset specifically for fine-grained motion understanding. All the code and annotations will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper propose MotionSight, a novel visual prompting method for fine-grained video motion understanding by tailored decoupling object and camera motion. The experiments reveal the enhanced performance for fine-grained motion understanding without additional training data. The authors also collected and annotated the first large-scale fine-grained video motion understanding dataset MotionVid-QA.

### Strengths
1.Fine-grained video motion understanding is under explored and worth studying. The paper analyze this task from both method and data sides by designing tailored model and constructing a valuable dataset.

2.The experiment result is solid with high performance on several evaluation benchmarks for fine-grained video motion understanding.

3.Convincing visualization examples are provided to prove the effectiveness of the proposed method.

### Weaknesses
1.The overview of the framework and proposed framework may raise the concern about the processing cost as it contains many designed modules compared for fine-grained  video motion understanding task compared to single MLLM methods. I encourage the authors to introduce the training/inference cost and speed compared with the provided baseline methods in more detailed.

2.Although the performance of the model has improved to some extent after training with MotionVid-QA, the results in Table 4 have certain limitations. The author may compare the results of the model trained on datasets of similar size.

### Questions
1.See weakness.

2.Though the ablation demonstrates strong robustness to several hyperparameters, I wonder whether the proposed method is sensitive to the parameters of selected models of MLLM.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MotionSight, a zero-shot approach to enhance fine-grained motion understanding in multimodal large language models (MLLMs). The method decouples object and camera motion, introducing visual spotlight and motion blur as visual prompts to improve perception of action details and camera changes. Additionally, the authors construct the first large-scale open-source dataset, MotionVid-QA, containing 40K videos and 87K Q&A pairs for SFT and DPO training. Experiments show MotionSight outperforms existing open-source models on several benchmarks and rivals commercial models on certain metrics.

### Strengths
1. Clear problem definition and strong motivation: Addresses a notable shortcoming of MLLMs in fine-grained motion understanding.
2. Simple yet effective method: As a zero-shot approach, MotionSight improves model performance without training, offering broad applicability.
3. Notable dataset contribution: MotionVid-QA is the first large-scale open-source dataset focused on fine-grained motion understanding, with high-quality annotations and significant community value.

### Weaknesses
1. Conservative innovation: While extending visual prompting to video is useful, "spotlight" and "motion blur" are traditional enhancements lacking methodological breakthroughs. Although the motion decoupling strategy is reasonable, the process of "detecting first and then focusing" relies on existing detection/tracking models. If the detection fails or is missed, the effect will be greatly reduced.
2. Limited generalization analysis: Insufficient discussion of failure cases in complex scenes (e.g., multi-object interactions, low-light, non-rigid motions like smoke or water).
3. Dataset distribution and quality: MotionVid-QA is largely sourced from existing human-action datasets, potentially limiting diversity; non-human or synthetic motion videos are underrepresented. Human preference annotation introduces subjectivity; no inter-annotator agreement or statistical significance reported.
4. Incomplete efficiency analysis: Mentions ~75% latency increase but lacks detailed time/memory profiling across resolutions or frame counts.

### Questions
1. What is the failure-mode policy when Grounding DINO or SAM2 misses or hallucinates boxes? When a video contains both salient object motion and complex camera motion, why route to a single path instead of feeding both augmentations simultaneously?
2. What is the head-to-head win-rate between MotionSight captions and Tarsier2 captions in the preference round? Please report overall and per-motion-type ratios.

### Soundness
4

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
The paper introduces a novel zero-shot method called MotionSight to enhance fine-grained motion understanding in MLLMs without requiring additional training. It addresses the limitations of existing MLLMs in perceiving subtle inter-frame dynamics by decoupling object and camera motion using visual prompting techniques. Specifically, it applies a visual spotlight to highlight moving objects and introduces motion blur to emphasize camera movements. To support training and evaluation, the authors present MotionVid-QA, the first large-scale open-source dataset for fine-grained video motion understanding, containing ~40K video clips and ~87K question-answer pairs with hierarchical annotations. Extensive experiments show that MotionSight significantly improves performance on benchmarks like MotionBench and FAVOR-Bench, and when used to fine-tune MotionChat, it achieves competitive results with much larger models.

### Strengths
1. The paper introduces a novel zero-shot method, MotionSight, which is the first to apply visual prompting techniques specifically tailored for fine-grained video motion understanding. This includes the innovative use of a visual spotlight to highlight moving objects and motion blur to emphasize camera movements, both of which are unique adaptations from static image prompting.
2. The idea of decoupling object and camera motion is novel and addresses a significant gap in current MLLMs, which often struggle with subtle inter-frame dynamics.

### Weaknesses
1. More Experiments: Can you provide more results on video perception benches, such as mvbench, TOMATO bench, etc.
2. The accuracy of MotionSight is partially dependent on object detection methods. This means that the performance of the model can be significantly affected by the quality of the object detection algorithm used.

### Questions
1. Can the method improve performance on the spatial benchs?
2. Why choose to use the detection model instead of enhancing MLLM detection capability through RL?

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
The paper presents MotionSight, a zero-shot visual prompting framework that enhances MLLMs’ fine-grained motion understanding through two complementary visual cues, Visual Spotlight for object motion and Motion Blur for camera motion.
Building on this, the authors further construct MotionVid-QA, a large-scale video QA dataset automatically annotated by MotionSight and refined via multi-stage filtering and human preference alignment. Experiments demonstrate consistent zero-shot improvements on MotionBench and FAVOR-Bench, and show that fine-tuning on MotionVid-QA (MotionChat) can yield additional gains with competitive efficiency.

### Strengths
- The paper is well-motivated and clearly written, addressing an important gap in motion understanding for MLLMs.

- The proposed visual prompting (spotlight + motion blur) is intuitive and effective, yielding consistent zero-shot gains.

- The experiments are comprehensive, and the curated MotionVid-QA dataset provides a useful resource for future work.

### Weaknesses
- **Limited novelty of Visual Spotlight.**
   The proposed visual spotlight is conceptually similar to existing image-level attention prompting methods (e.g., *Attention Prompting on Image for Large Vision-Language Models*, ECCV 2024), which also use soft masks to highlight salient regions. A comparison with such methods in Table 6 would strengthen the claim of novelty.

- **Potential ambiguity in Motion Blur.**
   Although the temporal weighting distinguishes different frames, the resulting blur may introduce directional ambiguity. For instance, in Figure 4 (bottom right), both lower-left and upper-right motion directions could be plausible, suggesting that motion blur may not always provide a reliable cue.

- **Dataset quality and generalization concerns.**
   MotionVid-QA is generated using MLLM-based annotations (Qwen2.5-VL-7B), meaning its quality and difficulty are inherently tied to the model’s capacity. While filtering ensures correctness, it does not increase complexity, so the dataset might mainly contain simple QA pairs. As a result, MotionVid-QA may improve weaker models but offer limited benefit for stronger ones (e.g., InternVL3-78B). Evaluating SFT effects on multiple models would better validate its generality.

### Questions
Just curious whether the authors tried applying MotionSight on top of the fine-tuned MotionChat model to see if it yields additional improvements. This could establish a potential self-improving VLM pipeline.

---
Overall, I’m borderline on this submission (currently rating it 4, since no borderline is available, and because I still have some concerns and would like to see the authors’ reply and additional results). I’d be happy to update my rating if my questions are well answered.

### Soundness
3

### Presentation
3

### Contribution
2
