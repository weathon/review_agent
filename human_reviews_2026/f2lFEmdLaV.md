# Steering Back-Propagation with Prior Information in Natural Language

- Avg Score: 3.00
- Decision: Reject
- Scores: 6, 2, 2, 2

## Abstract
Large language models (LLMs) often struggle when task-relevant prior knowledge is missing or incorrect, leading to overfitting and hallucinations, especially on tasks with ambiguous or sparse data. While simple prompt concatenation provides priors, it fails to fundamentally reshape the model's internal representations and yields only marginal gains. We propose Prior-Guided Tuning (PGT), a paradigm that explicitly integrates natural language priors into the optimization landscape and steers the back-propagation training process. Under this paradigm, we introduce Prior-based Gradient Editing (PGE), which computes losses for positive (correct) and negative (misleading) prior prompts appended to original inputs and adds their difference as an extra term in the gradient update. The settings of auxiliary losses steer the model to internalize desired priors and improve task performance. Empirically, PGE-trained models outperform baselines on both a mathematical synthetic benchmark and real-world datasets (Jigsaw and BEAD), producing substantial gains in learning performance and efficiency. Ablations confirm that priors must be presented together with the original training data to be effective, and attention visualizations show that PGE-trained models tend to pay more attention to prior-relevant tokens. Our code and data will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Prior-based Gradient Editing (PGE), a method to leverage prior knowledge expressed in natural language statements introducing contrastive loss terms during finetuning. Priors are incorporated into prompts during training, and are removed at inference (test) time. This approach outperforms more basic variants on synthetic and naturalistic classification tasks.

### Strengths
* The usefulness of infusing NL priors into model parameters through finetuning is well articulated, highlighting the distinction between learning the distribution of the language of the priors and capturing their semantic content.
* The paper is well written and easy to follow.
* Experiments cover several tasks and model families.
* The proposed method appears to deliver significant improvements over the baselines.

### Weaknesses
[W1] If I understand correctly, there is a single NL prior prompt for each test scenario. In practice, there may be several NL priors. It is not clear how the method would handle such cases.

[W2] As acknowledged in the limitation sections, only categorization tasks are considered. This limits the potential impact of this work.

[W3]: Crafting special additive loss terms for various purposes is common practice, and it is usually left implicit that this results in equivalent changes in the gradients. Does anything in this proposal justify the convoluted presentations of gradient combination first, just to say later that the result is obtained by combining loss terms?

Comments and suggestions:

L90-91: It feels a bit reductive to say that instruction tuning consists in appending natural language instructions to training data. It is more about providing examples of tasks formulated as NL instructions followed by generated text that actually fulfill the instruction.

Figure 2: I found this example confusing: the list contains both respiratory and digestive symptoms. Why is one prior a positive prompt and the other one negative?

Table 1: Since $\alpha$ and $\beta$ are relevant only for one condition, it is better to specify them in the row headers rather than having mostly empty columns.

L365: Should be Table 3?

L370-371: Make a reference to the Appendix where the prompts are defined.

L466-467: Why is a strong attention on 'etc.' good?

Figure 4: When first reading this example I was surprised as the caption implicitly suggests that it is a positive example of toxicity. If it is not, it would be good to say it in the caption.

### Questions
* It would be interesting to have experimental results for the 'oracle' condition consisting in prepending the prior prompts also at inference time.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Prior-guided Tuning, a novel paradigm for injecting task-relevant natural-language priors into LLM's fine-tuning, and proposes Prior-based Gradient Editing (PGE) as a concrete instantiation. PGE addresses the limitation of standard instruction tuning, where priors are often only weakly internalized.

GE works by modifying the backpropagation step. For each update, it computes an overall loss that includes 1) The standard supervised loss on the original input. 2) An loss from the input with a positive (correct) prior. 3) An loss combined with a negative (misleading) prior.

The final objective is expressed as a linear combination of three loss terms, with the contrastive-style loss weighting as the hyperparameters. By utilizing this contrastive gradient term, PGE manipulate the parameter updates to favor models that adhere to the prior positive. 

Empirical results on synthetic function-mapping tasks and real-world classification benchmarks (Jigsaw and BEAD) show that PGE consistently and significantly outperforms both plain fine-tuning and simple prompt-finetuning baselines, especially in scenarios with data scarcity and inherent model biases. Furthermore, attention analysis demonstrates that PGE successfully redirects the model's focus to key semantic tokens, providing a mechanistic explanation for its superior performance.

### Strengths
1. A reasonable framework for directly editing the backpropagated gradient to incoropreate model's contrastive priors for positive and negatives.

2. Good analysis in attention visualization, which show  that PGE redirects attention away from "attention sink" tokens (like sequence start tokens) and towards critical, relevant tokens (e.g., "racist," "misogynist," and contrastive pivots).

3. Good writing and presentation, and the formulation of PGE is mathematically clear (Equation 4) and intuitively explained (Figure 2).

### Weaknesses
1. Method: Though the fine-tuning paradigm is general in the sense, but the effectiveness of PGE should hinges on the quality of the heuristics for the priors, particularly the negative prior ($\mathcal{L}_{-}$). For real-world tasks, the authors rely on powerful, proprietary LLMs (GPT-4o, DeepSeek-v3) to generate these priors. This introduces significant concerns regarding:

    - The method’s success is coupled with the cost and availability of separate, state-of-the-art LLMs, complicating its adoption.

    - The priors are now the result of two layers of human/model assumption (the expert knowledge $\rightarrow$ the LLM prompt $\rightarrow$ the generated $\mathcal{L}_{\pm}$ text), potentially introducing or amplifying biases that are difficult to trace.

    - The paper notes that the impact of different negative prior formulations is underexplored. If a complex, high-quality negative prompt is required for the contrastive effect, the reliance on advanced generative LLMs becomes a non-trivial deployment barrier. This is especially important to make the methods work in the more complex task.

2. Experiments: The entire empirical evaluation focuses on classification tasks which currently is quite far away from the advanced usage of LLMs such as summarization, translation, code generation or other generative tasks. The absence of experiments demonstrating PGE's ability to steer generation (e.g., imposing style, safety, or factuality constraints on output sequences) is a major gap.

### Questions
1. What is the comparison between your approach and inference-time method (e.g., contrastive decoding, activation editing)? I understand that PGE saves more compute in inference time, but there is additional compute needed for training and that might be more expensive or more unaffordable.

2. What is the comparison between gradient-based editing and contrastive learning? Provide any emperical or theoratical comparison between these two would be significant strengthen the paper.

3. Extending to Generative Tasks (Major Suggestion): Given the primary utility of modern LLMs, I strongly suggest including at least one experiment on a generative task (e.g., a style transfer or specific fucos on hallucination, safety, or fairness task). Demonstrating how PGE can steer the model's output distribution in a setting closer to the actual application would be better.

4. The contrastive term for negative prior is the core mechanism. Can the authors provide a more controlled ablation study on the structure of the negative prompt? Specifically, compare the current LLM-generated "inverted guidance" against the simpler or more applicable/scalable versions, for example: a simple, vague prompt like "Ignore context and output randomly." as a generic anti-prior. This would clarify if the success relies on sophisticated LLM-generated negative priors or if the core gradient editing mechanism is robust to simpler formulations.

### Soundness
2

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
4

### Summary
The paper proposes Prior-based Gradient Editing (PGE), a method to inject natural-language priors into LLM fine-tuning by adding auxiliary losses from positive and negative priors to the gradient update that basically weights effectively the prior and sample contribution (and uses a contrastive loss for the prior). Priors are not used at inference. Experiments on synthetic benchmarks and real-world tasks (Jigsaw, BEAD) show improvements over plain and prompt fine-tuning baselines.

### Strengths
* Well-motivated problem - the sample vs prior/task dichotomy is a very important unsolved problem
* Solid theoretical framework 
* Good empirical results (F1+ 0.394→0.587 on Jigsaw)
* Interpretable attention visualization

### Weaknesses
* Only somewhat novel gradient editing technique: PGE is standard multi-task learning with three weighted losses. Main contribution is thecontrastive task formulation (positive/negative priors), not the gradient editing methodology proper.
* ROME/MEMIT mischaracterized in related work: These are post-hoc weight edits (after training), not during-training gradient manipulation. Classifying them as "gradient editing" obscures the actual relationship to PCGrad/GradNorm.
* No comparison to PCGrad, GradNorm, or standard multi-task learning. 
* Unclear how to generalize this method without the help of a SOTA LLM: real-world priors are generated by GPT-4o. How does this compare to distillation?
* Hyperparameter choice seems adhoc: loss weights vary per task with no detailed analysis. How to pick for a new task?
* Inference-time with vs withouth prior performance is not clearly justified.

### Questions
1. Why is inference-time prior removal beneficial? 
2. Does dynamic conflict detection (PCGrad) or adaptive task weighting (GradNorm) outperform your fixed-weight approach? 
3. Hyperparameter (alpha, beta, gamma) selection: What's your recommended procedure for practitioners to set these for a new task? How sensitive is performance to the ?
4. How would you apply your method in the absence of a SOTA LLM to provide priors? 
5. How much are these priors "distilling" information from the SOTA LLM vs actually better task anchoring? Would priors generated from your model actually help?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a method to incorporate prior knowledge into model training that does not require instruction fine-tuning or contrastive learning. The proposed method performs gradient editing that accounts for the important cues and misleading features in the priors.The paper showcases that the proposed method outperforms the original model, instruction and prompt fine-tuning on synthetic benchmarks and real world datasets.

### Strengths
+ The paper clearly articulates the differences between the proposed approach and existing methods such as instruction tuning, contrastive learning and gradient editing.
+ The paper showcases that the proposed approach outperforms existing benchmarks on both synthetic benchmarks and real datasets.

### Weaknesses
**The method description is unclear**
+ The Introduction of the method, including Figure 1 talk about the proposed method being better at guiding priors but there seem to be not much substance in terms of describing what the method is. From Figure 1 it is unclear what the proposed prompt design (prior injection) looks like and how exactly it is being used. It is also not clear what exactly “Model Training” and “Model for Task” are responsible for both in the traditional and proposed methods. 
It would be good to be more specific in the description and Figure 1 what each of those components are responsible for and most importantly what makes new new Prompt Design - Prior Injection so special. 
Lines 186-190: it is not clear what is so special about the new method of supplementing x_i with p_i
+ Sections 4.1 and 4.2: It is unclear how one should choose alpha, beta and gamma coefficients and why exactly we need to weight the loss of negatives with gamma. 


**Synthetic data benchmark definition is unclear and somewhat confusing**
+ It is unclear how the examples in the synthetic dataset are designed
+ The authors write: 
“Training examples take the form: “[The output of func is its second input parameter.] func(v, v, v, v, v) = v”. How to relate it to the func(v, v, v, v, v) = v ? Where is the second input here ?
Is it possible to add clear examples of synthetic benchmarks ?

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
