# Review

## Summary
This paper presents a Visuo-Tactile World Model (VT-WM) that integrates vision and tactile sensing to enhance robotic manipulation in contact-rich tasks. The model improves upon traditional vision-only world models by incorporating tactile data, which helps maintain object permanence and physical fidelity, even under occlusion or ambiguous contact states. Trained on various contact-rich tasks, VT-WM achieves better performance in object permanence and compliance with physical laws. In real-world experiments, it shows higher success rates in zero-shot planning for tasks like stacking, pushing, and wiping, demonstrating data efficiency and robustness, particularly in low-data scenarios.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. This paper addresses a significant gap by integrating tactile sensing with visual inputs in world models for robotic manipulation. This multimodal approach effectively enhances contact-rich task performance, overcoming common issues like object teleportation and unrealistic dynamics that can occur with vision-only models.

2. The paper is well-organized and clearly written, with thorough explanations of the model architecture, training process, and experimental setup. Figures and tables effectively illustrate the model’s performance and comparisons with vision-only baselines.

3. The proposed VT-WM model has broad implications for contact-rich robotic manipulation tasks, such as assembly, cleaning, and dexterous handling, where occlusions and complex interactions are frequent. By improving planning accuracy and physical fidelity, VT-WM could lead to more robust and reliable robotic systems in real-world applications.

## Weaknesses
1. While the paper demonstrates the effectiveness of VT-WM in contact-rich tasks, it lacks evaluation in more complex, multi-step tasks that require sustained contact and precise force control, such as in-hand manipulation or delicate placement tasks. These tasks often involve subtle contact states and dynamic force adjustments, which may challenge the model’s robustness.

2. The paper does not extensively discuss how VT-WM performs under different levels of occlusion or in environments with multiple similar objects. In real-world applications, partial occlusions or visual ambiguities are common, and the model’s ability to maintain object permanence and accurate contact perception in such cases could be crucial.

3. The experiments focus on a specific set of contact-rich tasks, but it remains unclear how well the model generalizes to new, unseen tasks or environments. Additional experiments across a wider variety of tasks would strengthen the claim of general applicability.

## Questions
1. How does the model handle situations with partial occlusion or multiple similar objects, where visual cues alone may be insufficient for contact-rich tasks? Would the tactile input alone be sufficient to disambiguate such states, or would additional sensors be necessary?

2. Could you provide more details on the computational cost and inference time required for VT-WM compared to vision-only models? Is VT-WM feasible for real-time control in dynamic environments?

3. Have you tested VT-WM in more complex, multi-step tasks (e.g., in-hand manipulation or dexterous assembly)? How does the model perform in such tasks compared to vision-only models?

4. How does VT-WM handle noisy or unreliable tactile data, which is common in real-world applications due to sensor degradation or wear?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4