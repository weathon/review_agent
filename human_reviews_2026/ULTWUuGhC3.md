# Interleave-VLA: Enhancing Robot Manipulation with Image-Text Interleaved Instructions

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
The rise of foundation models paves the way for generalist robot policies in the physical world. Existing methods relying on text-only instructions often struggle to generalize to unseen scenarios. We argue that interleaved image-text inputs offer richer and less biased context and enable robots to better handle unseen tasks with more versatile human-robot interaction.
Building on this insight, we introduce Interleave-VLA, a robot learning paradigm extending interleaved image-text instructions from digital world to directly generating continuous action sequences in the physical world.
Interleave-VLA offers a natural, flexible, and model-agnostic paradigm that extends state-of-the-art vision-language-action (VLA) models with minimal modifications while achieving strong zero-shot generalization.
Interleave-VLA also includes an automatic pipeline that converts text instructions from Open X-Embodiment into interleaved image-text instructions, resulting in a large-scale real-world interleaved embodied dataset with 210k episodes.
Comprehensive evaluation in simulation and the real world shows that Interleave-VLA offers two major benefits: (1) improves out-of-domain generalization to unseen objects by 2× compared to text input baselines, (2) supports flexible task interfaces and diverse instructions in a zero-shot manner, such as hand-drawn sketches.
We attribute Interleave-VLA's strong zero-shot capability to the use of instruction images, which effectively mitigate hallucinations, and the inclusion of heterogeneous multimodal datasets, enriched with Internet-sourced images, offering potential for scalability. [Our project site](https://interleave-vla.github.io/Interleave-VLA-Anonymous/) has more information.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Interleave-VLA that predicts robotic actions based on multimodal (vision and language) instructions. To train Interleave-VLA, the authors propose a framework that automatically converts the text instructions in the existing large-scale robot learning dataset (Open X-Embodiment) to multimodal instructions. Experiments were conducted on real-world and simulated environments, and results demonstrate that learning from multimodal instructions significantly improves out-of-domain generalization.

### Strengths
- [S1] The paper explores a novel direction where VLA models utilize multimodal language instructions, and the proposed approach (Interleave-VLA) is simple and does not require any architectural modifications.  
- [S2] The experimental results seem interesting and clearly demonstrate the effectiveness of Interleave-VLA. 
- [S3] The paper is well-written and easy to understand. All figures and equations succinctly describe the details how Interleave-VLA works.

### Weaknesses
- [W1] When it comes to real-world scenarios, providing VLAs with multimodal instructions seems unnatural and cumbersome. Furthermore, additional complexity is required to convert textual instructions to multimodal instructions if users provide text (or verbal) instructions.
- [W2] The performance of the proposed system is inherently dependent on the generalization capabilities of off-the-shelf detectors, such as OWLv2 and Segment Anything.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces Interleave-VLA, a new paradigm for robotic manipulation that enhances existing Vision-Language-Action (VLA) models by enabling them to understand and act on **interleaved image-text instructions**. The authors argue that traditional text-only instructions are often ambiguous and limit generalization, especially for unseen objects. To address this, their paradigm allows for more flexible and less-biased inputs, such as "Put  on ".

To train this new paradigm, the authors developed a novel, automated pipeline to convert the large-scale Open X-Embodiment dataset into an "Open Interleaved X-Embodiment Dataset," comprising 210k real-world episodes. The paper's core contributions are:
1.  The Interleave-VLA paradigm itself, which is presented as a lightweight, model-agnostic adaptation for existing VLA models.
2.  The large-scale interleaved dataset and the automated pipeline to create it.
3.  A comprehensive set of experiments demonstrating that Interleave-VLA achieves 2x stronger out-of-domain (OOD) generalization in both simulation and the real world compared to text-only baselines.
4.  A novel analysis of "attentional hallucinations" in text-only models, which Interleave-VLA mitigates.
5.  Showcasing emergent, zero-shot capabilities, such as following instructions from hand-drawn sketches or web images.

### Strengths
This is a strong paper with several significant contributions.

* **Solves an Important and Intuitively Clear Problem:** The paper's premise is compelling and well-motivated. Using a visual example (a crop, a photo, or even a sketch) is an immediately intuitive and powerful solution. The paper effectively translates this intuition into a practical robot learning paradigm.
* **Comprehensive and Rigorous Experimentation:** The empirical validation is a key strength. The authors test their paradigm thoroughly across three distinct and challenging environments:
    1.  **Simulation (SimplerEnv):** They demonstrate a 2x performance gain on semantically out-of-domain tasks over strong baselines like the base $\pi_0$ model.
    2.  **Real-Robot (FANUC arm):** They replicate these findings in the real world, showing a 2-3x improvement in OOD success rates when using their pre-trained model. This real-world validation is critical and very well-executed.
    3.  **VIMA-Bench:** They show the generality of their *paradigm* by adapting it to a *different* VLA model (OpenVLA) and again showing superior performance, reinforcing their "model-agnostic" claim.
* **High-Value Dataset Contribution:** The creation and open-sourcing of the "Open Interleaved X-Embodiment Dataset" (210k episodes) is a substantial contribution to the community. The automated pipeline (Fig. 3) built from modern foundation models (Qwen, OWLv2, SAM 2) to generate this dataset is itself a clever and valuable piece of engineering, achieving high accuracy (95.6%).
* **Excellent Analysis and Insight:** The paper does not just present results; it provides a strong mechanistic explanation for *why* its approach works. The analysis of "attentional hallucination" (categorized as Attentional Bias, Diffused Attention, and Attention Leakage)  is insightful. Figure 5 provides a clear, qualitative "Aha!" moment for the reader, visually grounding the performance claims.
* **Impressive Emergent Capabilities:** Perhaps the most exciting result is the zero-shot generalization to *unseen instruction modalities*. The ability to handle hand-drawn sketches , internet images , and user crops —without any fine-tuning on such data—dramatically expands the practical utility and flexibility of the system. This demonstrates a deeper level of generalization than just handling unseen objects.
* **Clarity and Presentation:** The paper is exceptionally well-written and organized. The figures (especially 1, 3, and 5) are clear, informative, and effectively communicate the core concepts and results.

### Weaknesses
The paper is very strong, and the weaknesses are relatively minor or are areas for further strengthening.

* **Quantification of "Lightweight" Adaptation:** The adaptation is described as "minimal" , primarily adding special tokens (`<BOI>`, `<EOL>`) to the tokenizer for the $\pi_0$ model. While this simplicity is a strength, it's surprising that a model like Paligemma, which was *not* pre-trained on interleaved data, can learn this complex in-context visual grounding behavior so effectively from fine-tuning alone. The paper would be strengthened by a deeper analysis of this. Is the model truly "reading" the visual prompt, or is the prompt image just acting as a strong conditioning signal (e.g., a "blue" prior) that the model learns to associate with the text?
* **Computational Cost Not Quantified:** The authors correctly identify a key limitation: longer image token sequences increase computational demands. This is a very practical concern for real-time robotics. However, this cost is not quantified. The review would be more complete if it included a brief analysis of the trade-off: What is the actual increase in inference latency (e.g., ms per action) when using an interleaved prompt (e.g., text + 2 images) compared to the text-only baseline?
* **Analysis of Failure Cases:** The paper thoroughly documents the (many) successes of Interleave-VLA. However, a "Failure Modes" section is missing. When does Interleave-VLA fail? What if the user-provided sketch is highly ambiguous and matches *two* objects in the scene? What if the text ("red") and image (a blue object) are contradictory? Understanding the limitations of the proposed method is just as important as understanding its strengths.

### Questions
These questions are intended to clarify points that could strengthen my assessment.

1.  **Probing the Grounding Mechanism:** Regarding the adaptation of $\pi_0$, I am very impressed by the results given the base VLM was not pre-trained for interleaving. Have you run experiments to probe the nature of the learned grounding? For example, what happens in a contradictory scenario: if the text instruction is "pick up the **red** block" but the interleaved image prompt shows a **blue** block? Which instruction (text or image) does the model prioritize, and does this suggest a deeper "comprehension" or a learned attentional bias?
2.  **Quantifying Inference Overhead:** Could you please quantify the computational overhead mentioned in your limitations? Specifically, what is the approximate increase in the number of tokens processed and the corresponding impact on inference latency (e.g., actions per second) when using a typical interleaved instruction (text + 2 images) versus a text-only instruction?
3.  **Robustness of Sketch Generalization:** The zero-shot sketch generalization  is a fantastic result. How robust is this to the quality of the sketch? Could you provide a few examples of failure cases? I am curious about the limits of this capability, for instance, with sketches that are highly abstract, ambiguous, or could plausibly match multiple objects in the scene.
4.  **Impact of Dataset Pipeline Noise:** Your dataset generation pipeline achieves a high 95.6% accuracy. For the 4.4% of failures, what is the primary error mode (e.g., wrong bounding box, wrong object ID)? Have you observed any impact, even minor, from this label noise on the final trained policy's behavior?

### Soundness
4

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
This paper proposes Interleave-VLA, a model-agnostic robot learning paradigm that enables vision-language-action (VLA) models to comprehend interleaved image-text instructions and generate continuous physical actions, alongside an automated pipeline to construct a large-scale real-world interleaved dataset from Open X-Embodiment. Comprehensive evaluations in simulation and real robots show Interleave-VLA doubles out-of-domain generalization to unseen objects compared to text-only VLA baselines and achieves zero-shot performance on diverse inputs like hand-drawn sketches and web images.

### Strengths
Interleave-VLA effectively addresses the critical limitation of text-only VLA models—attentional hallucinations (bias, diffusion, leakage)—by leveraging in-context visual grounding from interleaved instructions, with minimal architectural modifications to existing VLA models (e.g., π₀, OpenVLA) ensuring broad adaptability.

The proposed Open Interleaved X-Embodiment Dataset fills a key gap in robotic multimodal data, integrating 11 diverse real-world datasets with standardized actions and high-quality interleaved annotations, enabling scalable training and cross-embodiment transfer.

### Weaknesses
1. Computational efficiency is compromised: interleaved inputs introduce longer image token sequences, increasing training and inference resource demands, which the paper only mentions briefly without exploring mitigation strategies.

2. To more comprehensively evaluate the effectiveness of the Interleave-VLA paradigm presented in this paper, it is suggested to introduce comparisons with additional recent VLA methods beyond the existing baselines to further validate its superiority in generalization and zero-shot capabilities.

3. It is recommended that several VLA works in 2025 be incorporated and discussed to contextualize the novelty of the proposed approach, and clarify its positioning against state-of-the-art advancements in architecture design and data scalability.

### Questions
See weakness

### Soundness
3

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
This paper introduces Interleave-VLA, a paradigm for robot manipulation that processes interleaved image-text instructions instead of relying on text-only instructions. The authors argue that text-only vla models struggle to generalize to unseen objects and scenarios, often suffering from attentional hallucinations where the model fails to correctly ground language to visual entities. To address this, they propose a model-agnostic framework that adapts existing vla models to handle mixed-media prompts. A core contribution is interleaved x-embodiment dataset of over 210k real-world trajectories, which was automatically generated by a pipeline that converts text instructions from the open x-embodiment dataset into an interleaved format. Through comprehensive evaluations in simulation and on real robots, Interleave-VLA is shown to offer two major benefits: 1) it improves ood generalization by over 2x compared to text-only baselines, and 2) it unlocks zero-shot capabilities, allowing the robot to follow instructions from novel visual modalities not seen during training, such as hand-drawn sketches, user-cropped images, and photos from the internet. The paper attributes this success to the explicit visual grounding provided by instruction images, which mitigates ambiguity and attentional failures.

### Strengths
1) Introduced a highly effective paradigm for improving robot policy generalization by leveraging interleaved image-text instructions, achieving >2x performance gain on OOD tasks.
2) A large-scale, real-world dataset (Open Interleaved X-Embodiment) and an automated pipeline for generating interleaved instructions.
3) The work is validated through comprehensive experiments in both realistic simulation and physical settings.

### Weaknesses
1) The paper claims to be the first framework to comprehend interleaved image-text instructions and directly generate actions on robots in the physical world. however, interleaved multimodal prompts for manipulation were already explored in simulation [1](as in related work section) and explicitly framed as interleaving text and visual tokens. 
2) There seems to be no quantitative grounding/hallucination analysis, despite rich literature and metrics for VLM hallucination like POPE etc.
3) FANUC evaluation averages over 12 trials per object and reports mean success/accuracy without uncertainty estimates; the paper does not report multi-seed variation for simulation either.
4) Interleaving increases sequence length and thus quadratic attention cost; practical deployment on real robots is sensitive to inference latency.
5) The authors attempt to isolate the source of performance gains by comparing Interleave-VLA (full) against a (partial) version trained on the same data but tested with text-only instructions. However, this analysis remains insufficient to attribute the gains specifically to the novel interleaving structure itself. The experiment conflates the effect of the input format (interleaving) with the effect of the input content (the addition of explicit visual goal images). The (partial) baseline lacks any visual goal information at test time, while the (full) version receives it so it is unclear how much of the significant performance jump is due to the proposed interleaving methodology versus the benefit of goal-image conditioning.

[1] VIMA: Robot Manipulation with Multimodal Prompts. ICML 2023

### Questions
Please see weakness section. My main concern is 1) and 5). How do you disentangle the gains from the interleaving structure versus the benefit of providing any visual goal conditioning?

### Soundness
3

### Presentation
3

### Contribution
2
