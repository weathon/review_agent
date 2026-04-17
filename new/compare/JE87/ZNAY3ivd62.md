# Review

## Summary
The paper proposes GUI-Spotlight, a multimodal large language model (MLLM) designed to enhance graphical user-interface (GUI) systems by improving visual grounding—the ability to map textual references to specific on-screen elements. GUI-Spotlight uses a dynamic spotlight mechanism that iteratively narrows its focus with the help of multiple specialized tools (e.g., crop, extract, find color) to accurately target relevant regions on the screen. The model is trained in three stages: supervised fine-tuning (SFT) on multi-turn dialogues, reinforcement learning (RL) with a modified Group Sequence Policy Optimization (GSPO) algorithm, and further refinement with high-resolution samples. This approach enables GUI-Spotlight to outperform existing models on benchmarks like ScreenSpot-Pro and UI-Vision, achieving high accuracy with relatively small training data.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
- The GUI-Spotlight model introduces a novel spotlight mechanism that dynamically focuses on relevant screen regions, improving visual grounding accuracy.
- The model achieves high accuracy on benchmarks like ScreenSpot-Pro and UI-Vision with relatively small training data, demonstrating data efficiency.
- The use of multiple specialized tools (e.g., crop, extract, find color) allows for precise targeting and iterative refinement of search areas, enhancing the model's effectiveness in complex UI environments.
- The three-stage training process, combining SFT and RL with a modified GSPO algorithm, provides a robust framework for tool coordination and improves training stability.

## Weaknesses
- The paper does not include a comparison with SE-GUI-7B, a relevant baseline model.
- The model's iterative, multi-step reasoning approach may lead to slower inference times compared to single-step models.
- While GUI-Spotlight outperforms models like UGround-V1-7B and UI-TARS-7B, it still lags behind larger models such as UI-Venus-72B on some benchmarks, indicating a performance gap between 7B and 72B parameter models.
- The model's reliance on multiple tools and iterative processes may increase computational costs, potentially limiting its deployment in real-time or resource-constrained environments.

## Questions
- Could the authors provide a comparison with SE-GUI-7B to contextualize GUI-Spotlight's performance?
- How does GUI-Spotlight handle situations where the required tool is not in the predefined set?
- Could the authors elaborate on the model's inference time and its impact on real-world applicability?
- Are there any strategies to optimize the model for faster inference without compromising accuracy?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4