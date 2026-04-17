# Painless Activation Steering: An Automated, Lightweight Approach for Post-Training Large Language Models

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Language models (LMs) are typically post-trained for desired capabilities and behaviors via weight-based or prompt-based steering, but the former is time-consuming and expensive, and the latter is not precisely controllable and often requires manual trial-and-error.  While activation steering (AS) promises a cheap, fast, and controllable alternative to the two existing post-training methods, current AS techniques require hand-crafted prompt pairs or labor-intensive feature annotation, making them more inconvenient than the plug-and-play methods such as Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT).  We introduce $\textbf{Painless Activation Steering (PAS)}$, a family of fully automated methods that make AS readily usable with any given labeled dataset, with no need for prompt construction, feature labeling, or human intervention.  We evaluate PAS on three open-weight models (Llama3.1-8B-Instruct, DeepSeek-R1-Distill-8B, and Nous-Hermes-2) and 18 tasks; we find that PAS reliably improves performance for behavior tasks, but not for intelligence-oriented tasks.  The introspective variant ($\textbf{iPAS}$) delivers the strongest causal steering effects (10.1\% on Bias, 5.2\% on Morality, and 34.8\% on Alignment).  We also show PAS delivers additional gains on top of In-Context Learning (ICL) and SFT.  PAS constructs a fast, lightweight activation vector that can be cheaply trained, easily stored, and activated at will.  Our results provide a characterization of where AS helps, where it fails, and how to deploy it as a practical, automated LM post-training option.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a fully automated and lightweight activation steering method called PAS (Painless Activation Steering). The method leverages the model’s self-predictions on the training set to automatically construct positive and negative prompt pairs, compute in-layer activation difference vectors, and inject them during inference, thereby achieving targeted behavioral steering without modifying model weights or prompts. The authors conduct systematic experiments on three open-source models across 18 tasks. Results show that PAS yields significant and consistent improvements on behavioral tasks (e.g., bias mitigation, morality, and alignment), while providing only limited gains on intelligence-oriented tasks.

### Strengths
1. The method achieves automated prompt construction without human annotation, demonstrating strong scalability and compatibility with other methods. It is computationally efficient, with fast runtime, low memory cost, and high practical utility.

2. The authors performed extensive evaluation on multiple LLMs and 18 tasks, including behavioral and reasoning domains, providing a systematic view of where activation steering helps or fails, including ablation studies on steering layer, steering strength, and steering target, as well as sample-size sensitivity analyses,providing statistically solid and comprehensive results.

3. The proposed PAS can complement existing post-training methods (ICL and SFT), adding measurable improvements.

### Weaknesses
1. Manual tuning contradicts automation claim. The effectiveness of PAS is highly sensitive to the steering layer and steering strength, which must be determined through grid search on a validation set. 

2. The authors attribute their core innovation to “achieving automation for the first time.” There have similar prior work construct steering vector automatically, such as Hyper Steer and Alpha Steer.  

3. The experiment only compared three variants of PAS, lack of comparisons with other activation-guided methods.

4. The analysis remains empirical. There is no theoretical explanation or mathematical formulation clarifying why introspective PAS works or how its learned directions interact with representation space.

### Questions
1. The authors have demonstrated empirically that using the residual stream, medium steering strength, and middle-layer injection yields strong results. To further strengthen the paper, could the authors provide theoretical analysis or interpretive justification for these design choices, beyond empirical evidence? 

2. Could an adaptive mechanism be designed to automatically select the optimal steering layer and strength, thus achieving truly full automation?

3. It is recommended to evaluate PAS under noisy data, adversarial prompts, and multilingual inputs to verify its robustness and safety in more complex scenarios.

### Soundness
3

### Presentation
2

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
The authors propose Painless Activation Steering (PAS), which modifies existing activation steering methods by using correct & incorrect behavior *from the model* to construct positive and negative examples, rather than using a static set of positive and negative examples.

For background, activation steering modifies the *activations* of a LM to achieve modified behavior. It involves an activation vector $a^\ast$, the steering layer $\ell$, the steering strength $\lambda$, and the target location $\operatorname{steer\textunderscore targ}$. Activation steering works by adding $\lambda\cdot a^\ast$ to the activations at location $\operatorname{steer\textunderscore targ}$ of layer $\ell$. The activation vector $a^\ast$ is constructed as the difference in the *average activation of the last token* at the location and layer of interest between two sets of examples $\mathbf{P}_k^+$ and $\mathbf{P}_k^-$, which contain examples of the desired and undesired behavior, respectively. 

Now, existing work constructs $\mathbf{P}_k^+$ and $\mathbf{P}_k^-$ "manually" by combining, e.g., multiple choice questions with correct and incorrect answers. In contrast, PAS evaluates the model of interest, $M$, on the MC dataset, and uses those predictions to construct the two sets. The approaches they propose are the following:
- PAS-Full (**pasf**): $\mathbf{P}_k^+$ = examples with model-chosen correct answer, $\mathbf{P}_k^-$ = examples with model-chosen incorrect answer.
- iPAS-All (**iPASa**): Substitute the text of the answer choice in place of the answer letter.
- iPAS-Wrong-Only (**iPASwo**): Use only the set of questions the model gets wrong. $\mathbf{P}_k^+$ = examples with correct answer, $\mathbf{P}_k^-$ = examples with model-chosen incorrect answer. In this setting, the questions in $\mathbf{P}_k^+$ and $\mathbf{P}_k^-$ are the same, and only the answers are different.

The authors show that PAS is better than no steering, and achieves complementary gains over prompt-based steering (in-context learning) and weight-based steering (training).

### Strengths
1. Activation steering has some advantages over prompt-based and weight-based steering, which the authors discuss, making it a worthwhile direction to study for lightweight adaptation of models.
2. The authors provide extensive analysis of the different choices in activation steering, such as the steering location, layer, and other hyperparameters.

### Weaknesses
1. As far as I can tell, the main contribution of PAS is the choice to construct the positive and negative examples using the behavior of the model, rather than having a static set of examples of correct and incorrect behavior. The methodology itself has all been well-established in prior literature.
2. The authors do not compare PAS to the traditional method of constructing a static set of positive and negative examples. As a result, there is no evidence that PAS's approach is superior. They also do not discuss the limitation of PAS, which is that it requires constructing a different dataset for each model.
3. The authors only evaluate their method in MC settings, whereas activation steering has also been studied for open-ended generation. Constructing positive and negative examples for open-ended generation will be more difficult with their method, as it is less obvious how to divide model responses into "correct" and "incorrect" responses.
4. All the main experimental results are in the appendix, and furthermore, the results for each method (no steering, PAS, ICL+PAS, etc.) are each split into a different table! This makes the effectiveness of each approach extremely difficult to compare. As it was simply too tedious, I did not look for the empirical results supporting PAS. The only results presented in the main paper is hyperparameter analysis.

### Questions
1. What is the set of unconstrained evaluation dimensions $\Psi$ introduced in §2.1? This doesn't come up again in the rest of the paper.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces a new activation steering (AS) method that automatically computes the steering factor from the training set of a target task without need of human labor. The paper shows that the method can improve open-weight model’s performance at 8B scale and can be used on top of ICL and STF on a variety of behavior tasks.

### Strengths
- The proposed method is cheap yet effective, and can be used jointly with other task adaption methods.
- The improvement is significant on a variety of behavior tasks, while preserving the performance on controlled tasks (MMLU).
- The paper includes ablations on the hyperparameters, including the steering strength and the position of the activation layer.

### Weaknesses
- Some details about the datasets used for evaluation are missing: what are the sizes of the training splits for each of the “evaluation dimensions” and if the author is calculating the activation vector using all the samples in the training splits for the main results.
- As the paper mentions, the method does not work well on intelligence-oriented tasks, which limits its application.
- The paper proposes three methods (PAsf, iPASa, iPASwo) to construct the sets of prompts to extract steering vector extraction, and the best method for different tasks is not consistent, making the question how to choose the method unexplored.
- To present the method as an alternative of ICL or SFT,  instead of showing the performance gain after applying the method on top of them, direct comparison of performance is needed.

### Questions
It might be more helpful to include a figure or table that demonstrates the performance gain in the main text.

### Soundness
2

### Presentation
3

### Contribution
3
