# GPS: General Per–Sample Prompter

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
LLMs are sensitive to prompting, with task performance often hinging on subtle,
sometimes imperceptible variations in phrasing. As a result, crafting effective
prompts manually remains challenging and time-consuming. Recent automatic
prompting methods mitigate this difficulty but face three key limitations: (i) for
each new task, they require large datasets to train good prompts; (ii) they rely on
costly optimization loops that may take hours; (iii) they typically produce a single
task-level prompt that does not adapt to the individual input problem to be solved.
We propose GPS, the first general-purpose, per-sample prompting method. Without any task-specific tuning, GPS generates a tailored prompt for each unseen
input, improving performance across diverse tasks. The prompter is trained with
reinforcement learning on a suite of training tasks and includes a novel regularization for effectively adapting to per-sample prompting. Finally, we employ Minimum Bayes Risk decoding to stabilize inference.
Empirically, GPS demonstrates competitive performance: we attain second best
results among baselines on text simplification, third best results on summarization
and on-par results on classification, while not training on any of these tasks, in
contrast to the baselines. For in-domain prompting, we obtain sota on GSM8K.
Our work shows the potential of a novel and effective paradigm for automatic
prompting: generating adaptive, input-specific prompts without extensive optimization and without access to a task-specific training set. Code and data will be
released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
General-per-sample prompter is a trained model for generating task- and sample- specific prompts for novel samples and tasks. This improves on prior optimization approaches by dynamically generating sample-specific prompts and generalizing beyond in-distribution training tasks. The model is trained using RL with a novel regularizer intended to prevent reward hacking that might result from the prompter providing the answer directly to the LLM. Experimental results are presented showing the prompter's capability to generalize across tasks, and ablations are performed on the effects of varying regularization, decoding, and per-sample prompting methods.

### Strengths
GPS adapts to the task and generates a unique prompt for each sample.
GPS aims to generalize to unseen tasks, making it more versatile.
A novel regularization approach prevents reward hacking, which could generalize to other RL settings.
The prompter is decoupled from the evaluator, enabling a smaller model to boost the performance of a large model.

### Weaknesses
In general the experimental results show that the method is competitive but PRL seems to be a stronger method overall- the authors should discuss the relative advantages of their approach compared to PRL. Was PRL evaluated on unseen tasks?
Training tasks are largely reasoning tasks while test tasks are mainly classical NLP tasks, underscoring the need for ablations to understand whether GPS generalizes well to more complex tasks. The GSM8K results are encouraging but the absence of a holdout reasoning task is a liability.

### Questions
How did you choose the tasks you trained on? Did you perform any cross-validation by mixing the training tasks?

Explain the distinction from PRL and the benefits? Is PRL trained directly on the test tasks, explaining the superior performance? 

Is it necessary to calibrate r_{alignment} so that task rewards are balanced?

Have you performed any ablation over N in the decoding phase?  Compare with a baseline best-of-N prompting scheme?

It took me some time to figure out that GPS-J and GPS-SR-0.1 correspond to the two regularization methods- be sure to indicate this clearly in section 3.

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
The paper proposes GPS (General Per-Sample Prompter), a reinforcement-learning–trained prompt generator that, at inference, produces a unique prompt per input for an unseen task. Two regularization schemes (an LLM “Judge” that penalizes answer leakage and a sample-swap scheme) aim to prevent the generator from embedding solutions in the prompt. Inference uses Minimum Bayes Risk (MBR) selection over multiple candidate prompts/evaluator outputs. Trained on math/logic/programming tasks, GPS is evaluated out-of-domain on summarization (SAMSum), simplification (ASSET), and classification (SST-2/-5, MR, CR, AG News, TREC, SUBJ). Results show 2nd–3rd best performance on summarization/simplification and competitive classification averages, plus strong in-domain GSM8K, with ablations for regularization, MBR, per-sample prompting, backbone (Qwen vs Llama), and evaluator scale.

### Strengths
- The problem setup and the paper is well-motivated.
- Method simplicity and plausibility: two-LLM setup (generator/evaluator), GRPO optimization, and explicit reward decomposition (format/structure/accuracy). 
- Broad evaluation and ablations: out-of-domain generalization (trained on reasoning; tested on summarization/simplification/classification), backbone swap (Qwen→Llama), effect of regularization probability, per-sample vs no-per-sample, and MBR on/off.

### Weaknesses
- Effectiveness:
   - Underperforms PRL on most tasks. The central claim is that a task-agnostic, per-sample prompter should be at least competitive with task-specific prompt RL (PRL). Yet across the main tables (summarization, simplification, multi-task classification), GPS typically lags PRL. This weakens the paper’s core effectiveness claim.
   - Ask for a controlled, cost-normalized head-to-head. Please report a training budget comparison against PRL, including GPU hours, latency and token cost. 

- Novelty / Positioning

   - “First general-purpose per-sample prompter” is not well established. Instance-level prompt optimization is not new, and per-sample demonstration selection (retrieval-augmented ICL) is widely studied. The paper should more clearly delineate how GPS differs from (i) instance-dependent prompt tuning/selection and (ii) retrieval-based ICL that adapts the context per input.

   - Some other works generate per-instance keywords such as "https://arxiv.org/abs/2302.11520" is also considerred as per-sample PO.

- Missing Baselines:

  - Include some demonstration-selection baselines, since they are also per-sample strategies.

- Additional Evidence that would Strengthen the Paper:

   - What do optimized prompts look like?

   - Judge validation: How reliable is your Judge?

### Questions
See above.

### Soundness
3

### Presentation
2

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
This paper proposes a new framework called GPS, to customize the prompt for the inputs to the large language models (LLMs). The customization focuses on the instance-level and is done by a general prompt generator without finetuning on any data from the targeted downstream tasks. More specifically, a general prompt generator is trained via GRPO on the prompt generation task with the data from mathematical, logical, and programming tasks. Two regularization methods are developed to discourage the generator from inserting the answer to the input into the prompt template and hack the rewards. Experimental results on several NLP tasks show that the proposed method can achieve comparable performance with the existing prompt customization methods finetuned on the downstream tasks.

### Strengths
The strengths of the paper are listed as follows.
1. The paper is well motivated. The prompt design can affect the performance of the LLM and a general-purpose generator can reduce the human effort in prompt design significantly.
2. The two regularization methods are reasonable and reflect deep insights from the authors in the prompt customization task.
3. It is great to see that the authors provided a detailed ablation analysis in their methods, providing more insights into their method.

### Weaknesses
The weaknesses of the paper are listed as follows.
1. The experimental results are not convincing enough. I illustrated this point as follows:     
a. The evaluated downstream tasks are relatively naïve. More complicated tasks should be evaluated to show the effectiveness of GPS such as commonsense reasoning, information extraction and long-context understanding.     
b. The performance of in-domain performance in the math, logical reasoning and coding is also important to understand whether GPS is underfitting to the meta-tasks.       
c. It is not clear why Table 1 and Table 2 did not include the baseline named NI for the comparison. Besides, according to Table 3, the training-based GPS does not show a clear advantage over the NI, a baseline that designs the prompt by humans.    
2. The writing of the paper can be improved to make it easier to follow. For example, the contribution can be summarized into three key points, focusing on technical novelty, methodology, and evaluation. Secondly, since the training pipeline is based on PRL, the authors should highlight the difference between GPS and PRL in the related work section. Thirdly, in the method section, it would be better if the authors could formulate the prompt generation problem as an optimization problem to clearly define the objective, constraints, the learnable parameters, input and output to the problem.    
3. According to the proposed training pipeline, the general-purpose prompt generator is not model-agnostic and thus limits its impact. For example, the generated prompt may be overfit to the model family of the generator. So when the target LLM is not in the same family as the generator, the generated prompts may be suboptimal. The experiments do not provide enough evidence to address this concern since the authors did not compare with the existing baselines in their ablation study on cross-model performance.
4. Some important technical details are not clear. I raised the questions in the Questions section of the review.

### Questions
My questions are listed as follows.
1. What is the average prompt length GPS generated for each evaluated task? Will it generate a much longer prompt than the baselines and incur much higher latency for online inference?
2. What is the latency for a single sample inference with GPS? Will the introduction of an LLM-based prompt generator cause heavy overhead on the latency?
3. Could you provide more explanation about why picking math, logical reasoning and coding tasks as the meta-tasks to train the GPS? Why not the other tasks? Is the performance of GPS sensitive to the choice of meta-tasks and the relevant training dataset?
4. What are the differences between GPS and IPT or IDPG mentioned in the related work? Why not compare GPS with them in the experiments?
5. For equation (1), is there any scaling factor for each reward term?
6. In the proposed GRPO training, how to define a group when calculating the group average advantage?
7. In the proposed GRPO training, what is the decoding strategy of prompt generator and evaluator?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces GPS, a general-purpose, per-sample prompting framework. GPS generates unique prompts for each input without task-specific fine-tuning. It is trained via RLVR across diverse reasoning and programming tasks, with two regularization techniques (judge and sample regularization) to prevent label leakage. The model demonstrates strong out-of-domain generalization on summarization, simplification, and classification, and on GSM8K.

### Strengths
- The problem of per-sample prompting is quite important and well-motivated. 
- Method is evaluated across diverse domains—reasoning, summarization, simplification, and classification—with thorough ablations on regularization, decoding (MBR), and per-sample prompting.

### Weaknesses
- Given that the prompt generator model is RL-tuned. A fair comparison would be to compare to fully finetune the model on these tasks, with the same utility reward. Would that approach directly outperform the indirect optimization over the prompts? and what would the generalization performance be in that setting?
- I'd want to see a comparison with more recent prompt optimization approaches like in reflective prompt optimization GEPA.

### Questions
see above.

### Soundness
2

### Presentation
3

### Contribution
3
