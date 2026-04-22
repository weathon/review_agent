# Understanding vs. Generation: Navigating Optimization Dilemma in Multimodal Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 8

## Abstract
Current research in multimodal models faces a key challenge where enhancing generative capabilities often comes at the expense of understanding, and vice versa. We analyzed this trade-off and identify the primary cause might be the potential conflict between generation and understanding, which creates a competitive dynamic within the model. To address this, we propose the Reason-Reflect-Refine (R3) framework. This innovative algorithm re-frames the single-step generation task into a multi-step process of "generate-understand-regenerate". By explicitly leveraging the model's understanding capability during generation, we successfully mitigate the optimization dilemma, achieved stronger generation results and improved understanding ability which are related to the generation process. This offers valuable insights for designing next-generation unified multimodal models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents R3, a Reason-Reflect-Refine framework for jointly improving unified multimodal model's understanding and generation ability. The proposed framework iteratively reasons on the generated image and refines it based on its own feedback, and further uses RL to finetune the model for better conducting R3.

### Strengths
- The proposed framework significantly improved the model's instruction following generation power as well as the understanding power.
- The use of RL to further improve the R3 power gives significant improvements in the designed scores.
- Such an approach may lead to more advanced multimodal reasoning paradigms.

### Weaknesses
- My major concern is that the effect of RL is unclear. Based on results in the paper, RL seems improved the average understanding and generation capability in one Reflection-Regenerate step. But it is unclear whether 1) RL improved the highest possible performance given an unlimited Reflection-Regenerate round budget, OR 2) RL reduced the number of Reflection-Regenerate rounds to have a converged performance.
- The proposed framework is only tested on one single model, BAGEL. It's necessary to test its generalizability over different MLLM backbones.

### Questions
See above.

### Soundness
3

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
3

### Summary
This paper discusses the long-standing trade-off between generation and understanding in multimodal large models. The authors argue that these two abilities compete for model capacity due to misaligned optimization objectives. To overcome this, they propose the Reason–Reflect–Refine (R3) framework, which reconceptualizes generation as a multi-step process: the model first reasons about the input prompt, then reflects on whether its generated output aligns with the intent, and finally refines the output accordingly. A Tree-based Reinforcement Learning strategy and stage-wise rewards enable stable optimization.

### Strengths
* This paper demonstrates the trade-off between understanding and generation, where fine-tuning for one capability degrades the other, and naive co-training yields negligible gains.
* The Tree-RL strategy and stage-wise reward formulation stabilize training.
* The results demonstrates consistent improvements across multiple benchmarks, including newly introduced VQA and ITA evaluations.

### Weaknesses
* The experiments does not seem sufficient. Many other unified understanding and generation models are missing from the comparison. Furthermore, the comparison against proprietary models should not be limited to just one; others, such as Gemini 2.5 Flash or Gemini 2.0 Flash, should also be considered.
* The proposed method performs worse than GPT-4o on both the GenEval++ and TIIF benchmarks, and this performance gap should be discussed and explained in more detail.
* The reward model design is more empirial and heuristic. The optimization may still bias toward the reward model rather than true bidirectional alignment with understanding and generation. More theoretical analysis is needed to explain the underlying optimization alignment mechanism.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the trade-off between generation and understanding capabilities in multimodal models by proposing the Reason-Reflect-Refine (R3) framework. The authors argue that this conflict arises from competing optimization objectives and propose decomposing generation into a multi-step "generate-understand-regenerate" process. It proposes R3 (Reason–Reflect–Refine): turn single-shot image generation into a multi-stage loop that (i) plans, (ii) self-evaluates alignment of the produced image to the prompt, and (iii) edits until a learned stop condition (“No further edit needed”). Training alternates policies for the Reason and Reflect-Refine stages with a tree-RL scheme (GRPO for text; FlowGRPO for diffusion) and stage-wise rewards from a VLM judge. Experiments (GenEval/GenEval++/TIIF) report sizable gains in instruction-following and smaller but non-trivial gains on bespoke understanding tests (VQA/ITA built over model-generated images and VLM judgments).

### Strengths
- Well-motivated problem: Recasting generation as generate→understand→regenerate is a well-motivated method aligning with the philosophy of multiple prior COT-related methods. The paper explains the loop clearly and gives a concrete stop rule. 

- Optimization design: The tree-RL split reduces variance compared to end-to-end trajectory optimization; stage-specific rewards and formatting checks are well-motivated.

- Empirical gains: On GenEval++ (GPT-4.1 judge), BAGEL+R3 improves overall score vs. BAGEL and edges Echo-4o; understanding proxies (ITA/VQA) also rise. The inference-time scaling plot shows most gains after the first reflect-refine, with saturation after ~4–5 turns—useful for deployment budgets.

### Weaknesses
- The whole framrwork idea does not look particularlu novel to me: Self-correction and iterative refinement for generation have been extensively studied.

- I am abit concerned about the practicability of this work. In practice, developers tend to develop a single-shot generation instead of allowing multiplt RR turns, which is expensive. 

- Evaluation validity: Generation quality relies on GPT-4.1 as the arbiter for GenEval++ and understanding relies on Gemini 2.5 Flash to create “ground truth” for VQA/ITA—both proprietary VLMs that may share alignment biases with the reward/eval models. This risks overfitting to judge preferences and limits claims about human-perceived quality. Human or crowd-sourced evaluations are highly recommended to validate the effect outside model-judge ecosystems.

- Because reflection quality and termination are rewarded via VLM scores and format checks, the system could learn to optimize the judge (e.g., produce edits that increase the VLM’s “alignment” without visibly improving images, or prematurely stop when the judge saturates). A test where the judge is swapped could be helpful.

- Generalizability: Authors themselves observe domain-specific improvements (training on “counting” helps counting much more than other attributes). That tempers the central claim that R3 “reconciles” the optimization dilemma broadly. Evidence beyond compositional instruction following (e.g., style/photorealism, long-prompt semantics) is limited.

### Questions
- Can you report human A/B on a held-out prompt set and the cost per image and effectiveness per RR turn? 

- What is the exact “immediate rollout” procedure (sampling temperature, selection by reward diversity, replay buffer size), and how sensitive are gains to it?

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
3

### Summary
This paper tackles the "optimization dilemma" in multimodal models, where improving generative capabilities often degrades understanding, and vice versa . The authors posit this conflict arises from competing training objectives for finite model capacity . They propose the Reason-Reflect-Refine (R3) framework, which recasts generation as an iterative "generate-understand-regenerate" process. The model first Reasons to create a plan and initial image , then iteratively Reflects on its output (using its understanding) and Refines the image until it aligns with the prompt. The framework is trained end-to-end with reinforcement learning, using a novel Tree-RL strategy for improved stability . Experiments show that R3 not only enhances generation quality but also improves the model's understanding capabilities (e.g., counting accuracy), effectively mitigating the conflict

### Strengths
Important Problem: The paper addresses the "generation vs. understanding" trade-off, a critical and widely recognized challenge in developing unified multimodal models.



Novel & Intuitive Framework: The R3 framework is a novel solution that embeds understanding as a functional sub-process within generation, creating a synergistic loop rather than a competitive one .


Strong Empirical Validation: The core claim is well-supported. Experiments show simultaneous improvements in generation (GenEval++) and dedicated understanding tasks (VQA/ITA), with clear gains in skills like counting .



Effective Training Strategy: The proposed Tree-RL strategy (Fig. 3) is a solid technical contribution, demonstrating superior stability and reward over standard full-trajectory RL (Fig. 4) .

### Weaknesses
1. The new VQA and ITA benchmarks rely on "ground truth" labels generated by Gemini 2.5 Flash. Thus, high scores may reflect better alignment with Gemini's judgment rather than an objective, absolute improvement in understanding.

2. I'm curious whether this method is effective for image editing tasks, especially when spatial understanding is required.
For example, suppose there's an image with four people, and my prompt is: "Move the third person behind the second person," instead of simply arranging all four people side by side.
Can the model correctly interpret and execute such spatially grounded instructions?

### Questions
Overall, I find this method to be simple, effective, and intuitive. I will take the opinions of the other reviewers into consideration as well.

### Soundness
3

### Presentation
3

### Contribution
3
