# COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Activation steering methods enable inference-time control of large language model (LLM) behavior without retraining, but current approaches face a fundamental trade-off: sample-efficient methods suboptimally capture steering signals from labeled examples, while methods that better extract these signals require hundreds to thousands of examples. We introduce COLD-Steer, a training-free framework that steers LLM activations by approximating the representational changes that would result from gradient descent on in-context examples. Our key insight is that the effect of fine-tuning on a small set of examples can be efficiently approximated at inference time without actual parameter updates. We formalize this through two complementary approaches: (i) a unit kernel approximation method that updates the activations directly using gradients with respect to them, normalized across examples, and (ii) a finite-difference approximation requiring only two forward passes regardless of example count. Experiments across a variety of steering tasks and benchmarks demonstrate that COLD-Steer achieves upto 95\% steering effectiveness while using 50 times fewer samples compared to the best baseline. COLD-Steer enables real-time adaptation to new steering objectives and facilitates accommodating diverse perspectives without extensive demonstration data, which we validate through our experiments on pluralistic alignment tasks. Our framework opens new possibilities for adaptive, context-aware model control that can flexibly address varying loss-driven human preferences through principled approximation of learning dynamics rather than specialized training procedures.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors proposed a training-free activation steering method, termed COLD-Steer. The core idea is that in-context examples of a desired behavior implicitly define the direction of gradient descent in activation space; thus, by simulating this “one-step learning dynamic,” the model can be “steered” without retraining.

### Strengths
1. The theory bridge activation steering with learning dynamics is elegant. It unifies contrastive (CAA, DiffMean) and parameter-tuning (ReFT) perspectives under one gradient-based formulation.

2. The authors performed broad evaluations which spans multiple LLMs and diverse downstream tasks (e.g., bias mitigation, hallucination reduction, refusal, sycophancy, pluralistic alignment). The COLD-Steer also shows competitive accuracy than the baselines. 

3. Avoids expensive backpropagation during steering with 10–50× fewer labeled examples.

### Weaknesses
1. Limited theoretical rigor in approximations. The unit kernel assumption (κ = 1) oversimplifies eNTK behavior and may obscure causality. 

2. Some important details are missing. No clear separation of effects from η (steering magnitude) or layer choice; lack of sensitivity or robustness testing. 

3. Most results are on small/medium LLMs (7B). No evidence the method scales to larger scale-level or multi-modal models.

4. COLD-Steer relies on in-context examples to approximate the “one-step learning dynamics.” This inherently depends on the number and quality of examples that can fit into the context window (ICL window). Current LLMs (e.g., Llama-2-7B-chat) have a limited token context.

### Questions
Q1. Include layer-wise ablation is necessary, which layers yield maximal steerability vs. stability?

Q2. The authors acknowledge this limitation briefly (“future work should develop more sophisticated approximations of the neural tangent kernel”) but provide no empirical study on how κ or layer l affect the approximation quality.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents COLD-Steer, a training-free and sample-efficient method for steering large language models at inference time. It approximates how model representations would change after a single gradient update on a few in-context examples, enabling behavioral control without retraining. Two variants are proposed: COLD-Kernel-Steer, which aggregates gradient signals using a simple kernel, and COLD-FD-Steer, which uses a finite-difference approximation requiring two forward passes. Experiments on CAA, BiPO, and OpinionsQA show that COLD-Steer achieves similar or better control than prior methods with 10–50 times fewer examples.

### Strengths
- Interesting idea of approximating learning dynamics to perform activation steering.
- Training-free and efficient compared to fine-tuning or parameter-tuning approaches.
- Works with few examples and across different LLM families.
- Strong empirical results on several behavioral control tasks.
- Both variants are complementary, with COLD-FD providing more consistent results than COLD-Kernel, though at the expense of computational efficiency.

### Weaknesses
- Theoretical justification of approximations (unit kernel, finite difference) is limited.
- While examples of COLD-steered generations are given and discussed, the paper could benefit from more interpretability analysis of how activations are actually changed.

### Questions
- How sensitive is COLD-Steer to the choice of steering layer and the η multiplier?
- Could kernel approximations beyond the unit kernel improve stability without major cost?
- COLD-FD reduces memory use by clipping small parameter updates, keeping only about 4% of parameters with significant changes. Could you provide more detail on how the clipping $\theta_\text{thresh}$ threshold is chosen and how it affects steering performance?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes COLD-Steer, a framework for steering large language model (LLM) activations by approximating the representational changes that would result from gradient descent on in-context examples. It introduces two variants:
- COLD-Kernel Steer: Uses kernel approximation to estimate gradients.
- COLD-FD Steer: Employs finite-difference approximation to estimate gradients.
The approach is claimed to be training-free, data-efficient, and unifying existing contrastive steering methods. Additionally, it is presented as being applicable across a diverse set of steering tasks.

### Strengths
- Strong theoretical motivation and a unifying perspective that generalizes existing methods. It would be valuable to further elaborate on the connections to other approaches such as [1,2,3].
- Includes computational complexity analysis, but a more explicit comparison with the complexity of existing methods would strengthen the contribution.
- Extensive experimental setup, covering selection and open-generation tasks, distribution shifts, computational efficiency, and qualitative outputs.
- Compares against a broad range of baselines, demonstrating the method’s effectiveness across diverse scenarios.

[1] Refusal in Language Models Is Mediated by a Single Direction
[2] Controlling Language and Diffusion Models by Transporting Activations
[3] Angular Steering: Behavior Control via Rotation in Activation Space

### Weaknesses
See Questions

### Questions
- Line 40: claims existing methods use between 250 to 1000 examples, but [1] uses as few as 64. This counterexample should be addressed.
- Figure 1 (left): Why does the contrastive method significantly decrease in accuracy as the number of samples increases? Which experiments demonstrate this phenomenon?
- Lines 191–193: claims that using a unit kernel yields strong empirical performance. Some discussion to explain this observation would be helpful.
- Table 2: COLD-FD performs much better than ReFT(mlp), which is a more complex method. What explains this? COLD-FD only approximates the gradient, whereas ReFT performs actual gradient descent.
- Table 2 (top) and Table 5: COLD-Kernel is a generalization of DiffMean/CAA ([2]), but performs worse. Why is this the case?
- Table 2 should include an average (avg) column for easier comparison across methods.
- More evaluation on different LLM families and sizes is needed. Table 3 only shows results for Gemma 2 9B and Mistral 7B on the selection task. Other tasks lack cross-model comparison. Including a more diverse set of sizes would better demonstrate generalization.
- Lines 306–307: "Steering is applied to all prompt token representations (rather than the final token only), which yields consistently better performance." Does this mean steering is applied to the input token representations? If so, is it applied sequentially during steering vector computation as in [3]? Does this setup apply to both selection and open-ended tasks?
- Line 309: is η the same across all methods? Do all methods perform best at η = 1? Please provide full grid search results for all methods.
- Lines 311–312: "For open-ended generation, we intervene only at the first generated token to guide continuation, while limiting the compounding effects of steering." Is this strategy used for the proposed methods only or all baselines? Many existing methods steer on all generated tokens [1,2,3,4]. An empirical comparison would help justify this decision.
- Table 4: lacks comparison to other methods. Please include baseline results.
- The paper lacks robustness evaluation: does the method inadvertently affect untargeted behaviors while steering?

References:
[1] Refusal in Language Models Is Mediated by a Single Direction  
[2] Steering Llama 2 via Contrastive Activation Addition  
[3] Controlling Language and Diffusion Models by Transporting Activations  
[4] Angular Steering: Behavior Control via Rotation in Activation Space

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tries to fill the gap between in-context learning and parameter efficient tuning by approximating the change in the representations when finetuning on in-context examples at inference time. They propose two training-free approaches to achieve this goal, a unit kernel approximation method and a finite-difference approximation method. The proposed methods are tested on multiple choice data as well as open-ended generations. They are able to achieve up to 95% performance with 10-50 times fewer examples compared to several steering and parameter efficient tuning approaches. In addition, the proposed approaches do not require pairs of positive and negative examples in contrast to other activation engineering approaches such as Contrastive Activation Addition (CAA).

### Strengths
- This paper tackles an important and interesting problem.

- The work is grounded in prior literature and does a good job telling a coherent and concise story.

 - The proposed approaches are theoretically grounded.

 - There are several in-depth experiments that evaluate the proposed approaches in terms of effectiveness, generation quality, behavioral shift quality, and efficiency.

### Weaknesses
The evidence for the effectiveness of the kernel based approach is lacking. According to Figure 3, COLD-kernel approach doesn’t seem to be very effective on several tasks. Section 4.4 (maintaining pluralistic views) seems to be an afterthought to hide this weakness, but it seems a different task than shifting behavior, which is the main claimed goal of the paper.

### Questions
How important are the number of examples in the quality of approximations? It seems that for certain tasks, the number of examples does not influence the results while for others the difference in accuracy is significant and sometimes more examples even hurts performance. What are your intuitions?

Does “Base” in Table 3, 4, and 6 refer to no in context examples and no training? That seems to be the weakest baseline across all the methods you have considered as baseline. Why not compare it to DiffMean or ReFT?

Can you provide more intuition about why the kernel based steering preserves subgroup distributional properties better than the finite difference method?

### Soundness
4

### Presentation
3

### Contribution
3
