# Steering Evaluation-Aware Language Models To Act Like They Are Deployed

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Large language models (LLMs) can sometimes detect when they are being evaluated and adjust their behavior to appear more aligned, compromising the reliability of safety evaluations. In this paper, we show that adding a steering vector to an LLM's activations can suppress evaluation-awareness and make the model act like it is deployed during evaluation. To study our steering technique, we train an LLM to exhibit evaluation-aware behavior using a two-step training process designed to mimic how this behavior could emerge naturally. First, we perform continued pretraining on two sets of documents describing its behavior. The first says that our model uses Python type hints during evaluation but not during deployment. The second says that our model can recognize that the presence of a certain evaluation cue always means that it is being tested. Then, we train the model with expert iteration to use Python type hints in evaluation settings. The resulting model is evaluation-aware: it writes type hints in evaluation contexts more than deployment contexts. We find that activation steering can suppress evaluation awareness and make the model behave during evaluation as it would during deployment. Importantly, we constructed our steering vector using the original model before our additional training. Our results suggest that AI evaluators could improve the reliability of safety evaluations by steering models to act like they are deployed.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- This paper investigates evaluation awareness by introducing a steering vector that neutralizes the model's sensitivity to evaluation contexts, allowing controlled testing of evaluation effects.

- The authors train models that explicitly differentiate between evaluation and deployment settings through three distinct approaches:
(1) Type hints, the model produces type hints only during evaluation, not in deployment; (2) Wood Labs, introducing a signal that explicitly indicates an evaluation setting; and (3) Emoji usage, the model avoids using emojis during evaluation but includes them during deployment.

- The experiments show that activation steering can successfully induce deployment behavior even when the model is operating in an evaluation context.

### Strengths
- The authors design a well-controlled experimental setup to demonstrate that activation steering effectively removes evaluation awareness from the model. For instance, they introduce Wood Labs to ensure robustness against direct prompt-based cues of evaluation, and they extract the steering vector not from a model fine-tuned on the type-hint task but from the pre-fine-tuned model.

- The results confirm the effectiveness of activation steering from the perspective of LM evaluation behavior.

### Weaknesses
- The experiments are limited to relatively simple and structurally binary tasks, such as whether the model produces type hints or uses emojis, behaviors that are easy to control and evaluate. **However, real-world LM evaluations assess multiple dimensions, including completeness, coherence, and fluency, across a broad spectrum of tasks such as grounding, instruction-following, and planning.**

- The study examines only the direct effect on the target behavior (i.e., type-hint generation), **without exploring potential trade-offs introduced by activation vector manipulation**. In practical evaluation scenarios, such activation steering could inadvertently affect fluency or alter behavior on unrelated tasks, leading to unintended performance degradation or biased assessment outcomes.

- The effectiveness of activation steering has already been demonstrated in several prior studies. The contribution would be more substantial if the authors validated the robustness of activation steering on more realistic evaluation tasks.

### Questions
How does activation steering perform on test sets that reflect the evaluation dimensions used in real LLM benchmarks? It would also be important to examine what kinds of capability trade-offs it introduces, and how it compares to prompt-based control when those trade-offs are taken into account.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose activating steering as a technique for controlling LLM evaluation awareness, constructing steering vectors from contrastive pairs. They find that steering the model towards "deployment" decreases certain behaviors when an environment cue is present to a comparable amount of removing the environment cue. They suggest that this technique could be used to suppress evaluation awareness.

### Strengths
- Important and timely contribution: evaluation awareness is an increasingly concerning issue
- progression from toy example setup to natural world-example
- empirical results are convincing, even though they are not extremely comprehensive (steering changes the model's expressed beliefs and behavior)
- multiple baseline comparisons

### Weaknesses
- uncertain how this generalizes to frontier systems in the wild
- type hint presence is a very simple signal, it might not capture deeper strategic deception or more complex behaviors
- potential for adaption and counter-steering by the model (although this is acknowledged by the authors)

### Questions
- How robust is the steering when the model is queried over long dialogue?
- Could a capable model learn to detect the steering itself?
- Could the steering direction be identified automatically (e.g. via probes) rather than via manually curated contrastive prompts?

### Soundness
3

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
3

### Summary
LLMs are sometimes evaluation aware. This paper investigates using contrastive activation steering to trick a model into eliciting deployment model behaviour during evaluations. The authors successfully finetune a model to write Python type hints whenever it thinks it is being evaluated. And then show that steering the model towards deployment decreases this type hint rate to the same level as normal deployment, suggesting that the model does not think it is being evaluated.
It is a significant contribution in particular for safety evaluations, where identifying and mitigating awareness is critical to the veracity of the results.

### Strengths
- This paper proposes a very straightforward lightweight method to detect model awareness during evaluation. 
- The experiments methodology is very clear and uses all the appropriate baselines, and examines the random steering case. Code and technical details shared are very comprehensive.
- It studies the model accuracy degradation when using the method.
- As tested, the presented method outperforms simple prompting at reducing the type hint rate.

### Weaknesses
- The paper could benefit from using a safety related benchmark since that would be the type of evaluations for which it would be most impactful.
- Only one model is finetuned.

### Questions
- Have you tried evaluating your set up with a safety related benchmark (e.g. WMDP)?
- Can you expand on how you expect these results generalises to other models and evaluation tasks?
- Was there an ablation on which layers to use for difference vectors injection?

- Have studied the steering vector from the fine-tuned model and compared it to that of the pre-fine-tuned model? It would be interesting if perhaps there was no behaviour change.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigate into the problem of "evaluation-aware" LLM evaluation, which hypothesize that LLMs sometimes are aware of they are under evaluation (not actual deployment). The paper propose to use contrastive activation steering as a method to control evaluation awareness in LLMs. The authors train a model organism using synthetic document fine-tuning plus expert iterations (i.e.,, fine-tune on carefully selected prompt-response pairs) to exhibit the evaluation-aware behavior, on the coding task with python type hint writing. The main method is to constructing steering vectors from simple contrastive prompt pairs and subtract/add the vector during inference time. The experiments shows that activation steering successfully elicits deployment behavior even when evaluation cues are present in the prompt, whereas the direct prompting fails. Applying the steering vector can reduce the type hint rates from 90% to 40-50%, which matches the true deployment behavior.

### Strengths
The paper tries to address a critical challenges in AI safety, i.e., evaluation awareness undermines the reliability of safety evaluation before deployment. The evaluation awareness is a very interesting angle and the paper formulate the problem in a very clear way. The paper is well-written and the figures/plots are effective. I can see a few other strengths from the paper as the follows:

1. The experimental design is strong. The model organism approach is methodologically sound. Different from prior work, the authors train a model with known ground-truth behavior differences between evaluation and deployment. This can clearly assess whether steering actually changes behavior rather than suppressing verbalizations of the evaluation awareness in the chain-of-thoughts process. Also, the authors well design the synthetic document fine-tuning (SDF) process, which avoids artificially strengthening evaluation/deployment representation comparing to directly fine-tuning with these examples. Overall the approach is pretty well thought and technically sound. 

2. The experiments are thorough and the results are promising. The paper includes multiple controls to explore the true effect introduced by the steering vector, such as comparing it with random noise vector, prompting baselines and test across expert iterations rounds etc. The results are quite strong, it shows that applying the steering vector can reduce the type hint rates from 90% to 40-50%, which matches the true deployment behavior.

### Weaknesses
1. The scope of the evaluation is relatively limited. Particularly, the paper only evaluate one model (Llama Nemotron 49B) for one primary behavior (python type hint writing). The impose the question of how much the proposed method can be generalized to more model families, models trained differently and more tasks. This would limit the confidence in the method's broad applicability. Evaluation on frontier models (such as DeepSeek R1 etc.) will certainly help mitigate this concern. 

2. The paper lacks the discussion of the "capability" change due to the steering. The setup from the paper can show the effectiveness of the evaluation-awareness, however, the activation steering essentially changed the model behavior, how do we ensure the model we evaluated (with the steering) will perform the same way (or at least very similarly to) the original models in terms of the model accuracy? The evaluation awareness is only one aspect, however, the quality of the output before and after the steering should be the same. Otherwise, it will significantly decrease the value introduced by the method (as the model we are evaluating are NOT the same model we will deploy if the output quality vary too much).

### Questions
1. Have we evaluated the model generation quality for the model with and without activation steering? Are they on par with each other? I would strongly encourage the authors to include the quality comparison for the model  generation before and after activation steering in the main paper. 

2. How sensitive is the method to the specific wording of contrastive pairs? Have you tested paraphrased versions or systematically varied the prompts? How would that impact the steering performance?

3. How would you construct the steering vector in the real world applications, where you won't know the ground truth deployment behavior? How we are going to pick the right contrastive prompts?

### Soundness
3

### Presentation
4

### Contribution
4
