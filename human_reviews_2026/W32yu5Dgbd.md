# Adaptive Fast-and-Slow Visual Program Reasoning for Long-Form VideoQA

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2

## Abstract
Large language models (LLMs) have shown promise in generating program workflows for visual tasks. However, previous approaches often rely on closed-source models, lack systematic reasoning, and struggle with long-form video question answering (videoQA). To address these challenges, we introduce the FS-VisPR framework, an adaptive visual program reasoning approach that balances fast reasoning for simple queries with slow reasoning for difficult ones. First, we design efficient visual modules (e.g., key clip retrieval and subtitle retrieval) to support long-form video tasks. Then, we construct a diverse and high-quality fast-slow reasoning dataset with a strong LLM to align open-source language models' ability to generate visual program workflows as FS-LLM. Next, we design a fast-slow reasoning framework with FS-LLM: Simple queries are directly solved by VideoLLMs, while difficult ones invoke visual program reasoning, motivated by human-like reasoning processes. During this process, low-confidence fast-thinking answers will trigger a second-stage slow-reasoning process, and a fallback mechanism to fast reasoning is activated if the program execution fails. Moreover, we improve visual programs through parameter search during both training and inference. By adjusting the parameters of the visual modules within the program, multiple variants are generated: during training, programs that yield correct answers are selected, while during inference, the program with the highest confidence result is applied.  Experiments show that FS-VisPR improves both efficiency and reliability in visual program workflows. It achieves 50.4% accuracy on LVBench, surpassing GPT-4o, matching the performance of Qwen2.5VL-72B on VideoMME.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a neural-symbolic approach for language-based slow-fast reasoning. This paper designs several efficient visual modules, such as key clip retrieval and subtitle retrieval for long-form videos. This paper then constructs a diverse and high-quality fast-slow reasoning dataset. Experiments show that they can achieve stronger performance than GPT-4o on LVBench.

### Strengths
1. The experimental results look promising as it outperforms several competitive baselines such as QWEN and longvila.

2. We can observe that the proposed 8B model even outperforms the 72B model of Qwen, which is quite amazing.

### Weaknesses
1. The central claim of the evaluation results is that it outperforms open-source models. But in fact, this work isn't open-sourced. The attached supplementary code seems to be irrelevant to this submission and is on a toy dataset. The code is also specific in different situations. These facts raise overall concerns about the quality of this work.

2. From a methodology perspective, there are several things important to their contribution, but not clear:

(1) In Table 2, what are introduced by this work, and what are the existing methods? I think from VisualProg and follow-up works, these operators are mostly ready to use.

(2) How can the modules be connected if they have very diverse signatures? Is there any syntax checking algorithm?

(3) How can each module's effectiveness be justified? Note that Table 6 only ablates some parameters of some modules, while not all modules are justified. Also, the result difference of module ablations in Table 6 is quite small.

 (4) Is the parameter set in each module related to specific datasets? Those parameters may not be generalizable.

The main concern is that the claimed contribution #1 is weak and it's not novel, given so many related neural symbolic works.

(5) The training parameters are not described clearly. Ln 257 says "during training", it's not clear what it refers. Parameter search is not typically called a training procedure. This part needs to be clarified.

3. About the dataset construction. Is the dataset of 12K only enough to train the model? It's unclear whether the benefit comes from the training dataset or the model design itself. Also, it's not certain whether the data might contain samples close to the testing dataset.

### Questions
1. Why does InternVL have no results numbers listed? If so, why list it in Table 3?

2. Why not set the full column-width of Table 3?

3. Methodology questions: see above weakness section.

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
This paper proposes FS-VisPR a Fast-and-Slow Visual Program Reasoning for Long-Form VideoQA framework, the proposed method  consists of an adaptive visual program reasoning approach, that depending on the difficulty of the input question, the pipeline is able to adapt and choose "Fast Reasoning" for simple input queries, where a single VLM is used to give a direct answer, on the other hand, if the query is more complex, the pipeline uses "Slow Reasoning" by using visual program reasoning - a sequence of vision modules to gather evidence and answer the question -. To train the proposed model the authors construct a new dataset, that takes existing VQA datasets as a starting point and augment them with visual programs. FS-VisPR achieves better performances than Open-Source Models on LVBench, VideoMME and LongVideoBench and is competitive with Close-source models such as GPT-4o, Gemini-1.5-pro and Seed1.5VL-pro.

### Strengths
Paper Strengths:
1)  FS-VisPR a method that couples fast and slow reasoning for long VideoQA. The proposed approach demonstrates a practical recipe to tackle long-videoQA with test-time compute adaptively, that also provides interpretability from visual programs and confidence outputs.
2) A data construction pipeline that labels questions depending on their difficulty, and helps for aligning an openLLM to emit either a direct answer "fast reasoning" or start a visual program for "slow reasoning".
3) Broad benchmarking on LVBench, VideoMME, LongVideoBench with competitive or superior accuracy compared with open and close source models.
4) The paper is well written and easy to follow.

### Weaknesses
Paper Weaknesses:
1) The main comparison done in Table 3, is evaluating the proposed model FS-VisPR which uses fast and slow reasoning against single-pass videoLLMs for example against the backbone model Qwen2.5VL-7B, so improvements partly reflect extra test-time compute, not just a stronger backbone. Slow reasoning (which I would say is similar to chain-of-thought) is helping to get better performances, therefore, I think the comparison is not really fair. Have you tried to prompt the other models to perform chain of thought?
2) FS-VisPR uses FS-LLM (Qwen3-8B) to decide easy vs difficult, which is trained without any direct visual signal for a task that is related to video. This can create a suboptimal routing which can waste time. On the other hand,  FS-LLM does no see raw video while planning, from my understanding it writes programs from the question only, which I think can lead to noisy learning or suboptimal plans given by the LLM. Could this also mean that the LLM could be leveraging the well know problem in VQA datasets, which a model can reach the answer without seeing the visual context?.
3) I think is difficult to check the plans created by the model with the current Figures. The paper could benefit with some Figure showing this in a more visual way, for example if the model selected frames or OCR was used, it would be good to have some examples where you can see this visually , it could help to get a better understanding in my opinion.
4) While the paper motivates long videos and evaluates on long-form benchmarks, its analysis of length effects is limited to coarse Short/Medium/Long bins in a table. For example, in my opinion a curve of average accuracy vs video length can help to clarify the performance of the model and created a more focused analysis in long videoQA.
5) I think there is a typo on section 4.4 , maybe it should be Figure 5 and not Figure 8.

### Questions
Please refer to weaknesses for my questions and doubts. Overall, I think the paper brings a nice idea to use a fast-slow reasoning framework for long VideoQA, however I want to see the clarifications to my questions and concerns and sorry if I misunderstood something, I look forward to see the author's responses. Currently, I'm giving borderline reject, after rebuttal I will revise my decision.

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
This paper proposes a VideoQA method that adaptively assesses the difficulty of each queried question and adjusts the response strategy for more efficient response generation. The paper initially introduces Fast-Slow-Reasoning Data, which includes query difficulties with its visual program for answer generation, provided by GPT-4o. Using this data, a Multimodal LLM model (MLLM) is trained to determine whether a query is easy or difficult, and the visual program's parameters are enhanced through a simple parameter search using successful visual program examples from the collected data. Experimental results demonstrate competitive accuracy against proprietary closed-source MLLMs while improving calculation efficiency.

### Strengths
### S1. Auto-regressive multiple choice generation for confident calculation is interesting.

The proposed auto-regressive multiple choice generation for confidence calculation is a simple but effective idea. This potentially bridges recent MCQ-based benchmarks and open-ended question benchmarks, and also can be a general technique for explainable LLMs. I love to see this simple but effective trick.

### S2. Motivation is adequate.

The Adaptive VideoQA strategy is important to reduce MLLM's inference costs, making the motivation clear and reasonable.

### Weaknesses
### W1. A knowledge leakage from GPT-4o.

The method identifies the most confident visual program using examples from GPT-4o, intentionally leading to knowledge distillation.
Thus, comparing these results with other open-source MLLMs is unfair. Additionally, it might breach GPT-4o’s terms of use (see Ethics concerns for details).

### W2. Lack of sufficient significance.

Considering knowledge leakage, the work only shows significance if it surpasses GPT-4o in accuracy or efficiency, which is not quantitatively demonstrated. Comparing inference times with GPT-4o is difficult due to the lack of API support for internal execution timing.
The study should gather data from open-source MLLMs for comparative analysis.
Since current improvements rely on GPT-4o, the experiment does not adequately demonstrate the method’s impact.

### Questions
### Q1. 
Why is there a gap in accuracy between LVBench results in Table 3 (51.2) and Table 5 (47.1)? If the results in Table 5 do not use Qwen2.5VL-7/34B, please clarify the backbone model used in Table 5.

### Q2.
Given the current experimental setup, comparisons with open-source LLMs seem unfair. Please either avoid using GPT-4 data or provide a fair comparison using GPT-4 data.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper “Adaptive Fast-and-Slow Visual Program Reasoning for Long-Form VideoQA” proposes an adaptive reasoning framework for Video Question Answering (VideoQA), inspired by the dual-process theory of human cognition (fast vs. slow processing).

The model, called FS-VisPR, switches between Fast reasoning (direct VideoLLM answers) and Slow reasoning (program-based reasoning) based on the confidence of the model’s response, aiming to achieve both efficiency and interpretability in long-form video understanding.

### Strengths
1. Interesting conceptual motivation:

   The idea of transferring the System 1 / System 2 dual-process theory from human cognition into VideoQA reasoning is conceptually novel. The attempt to model “when to think and when to act intuitively” in an AI system is intellectually appealing.

2. Modular interpretability:

   The Slow reasoning mode executes explicit visual programs, providing a step-by-step, interpretable reasoning trace that can be visually inspected.

3. Evaluation on long-form VideoQA benchmarks:

   The experiments on datasets such as LongVideoBench and VideoMME highlight the method’s applicability to real-world long-duration video reasoning tasks.

### Weaknesses
1. Lack of clear quantitative advantage:

   Although the method is compared with large multimodal models (e.g., QwenVL-2.5, GPT-4V), the reported performance gains are modest (around +2–4 points) and without statistical validation.
   Compared to prior Visual Program methods (VisProg, ViperGPT, ProViQ, etc.), the paper does not make it clear which specific dimension—accuracy, efficiency, or interpretability—it actually outperforms.
   Despite claiming to improve “context understanding,” “efficiency,” and “transparency,” there are no explicit quantitative metrics for these aspects (e.g., runtime, computational cost, or program success rate).

2. Manual threshold tuning rather than adaptive control:

   The confidence threshold (θ) in Figure 4, which determines when to switch from Fast to Slow reasoning, was manually tuned through grid search rather than learned or automatically optimized (see Section 4.3 and Equation 2).
   Therefore, while the paper emphasizes “adaptive” reasoning, the adaptivity is rule-based and static, not learned or dynamically adjusted.

3. Dual-process analogy remains qualitative:

   Although the authors cite Evans (2003, 2008) and Evans & Stanovich (2013), the alignment with cognitive psychology remains purely qualitative.
   There is no quantitative comparison to human data such as reaction time distributions or confidence–accuracy correlations. The claim of psychological consistency is thus metaphorical rather than empirical.

### Questions
1. Incorporate an automatic or learned threshold optimization (e.g., Bayesian or reinforcement-based tuning) to justify the *adaptive* claim.
2. Introduce quantitative metrics for efficiency, transparency, and contextual understanding (e.g., inference time, module execution rate, interpretability score).
3. Provide a quantitative comparison with human data to substantiate the dual-process analogy.

### Soundness
3

### Presentation
2

### Contribution
1
