# Implicit Chain of Thought Reasoning via Knowledge Distillation

- Decision: Reject
- Scores: 6, 3, 3

## Abstract
To augment language models with the ability to reason, researchers usually prompt or finetune them to produce chain of thought reasoning steps before producing the final answer. However, although people use natural language to reason effectively, it may be that LMs could reason more effectively with some intermediate computation that is not in natural language. In this work, we explore an alternative reasoning approach: instead of explicitly producing the chain of thought reasoning steps, we use the language model’s internal hidden states to perform implicit reasoning. The implicit reasoning steps are distilled from a teacher model trained on explicit chain-of-thought reasoning, and instead of doing reasoning “horizontally” by producing intermediate words one-by-one, we distill it such that the reasoning
happens “vertically” among the hidden states in different layers. We conduct experiments on a multi-digit multiplication task and a grade school math problem dataset and find that this approach is able to outperform baselines that directly produce the answer by a large margin.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a method to reason "vertically" across layers by using a chain-of-thought (CoT) model's internal hidden states.
The method relies on a teacher model that is originally trained to predict intermediate reasoning steps in natural language before outputing the final answer. The idea is to take the internal states of the teacher model across layers produced when generating the CoT steps. Then an emulator network is trained to predict a pre-defined sequence from the $L\times T$ matrix of states, with $L$ being the number of layers and $T$ being the number of tokens in the CoT steps. That (vertical) sequence of states is then used to predict the final answer.

Concretely, the method consists of three steps:
1. A network is first trained to predict the final answer to a reasoning question given the question and $L$ hidden states from the $L$-layered teacher model.
2. An emulator model is trained to predict the teacher’s vertical hidden states (when generating the CoT tokens) from the input (distillation step).
3. Finetuning the combine system that links the emulator with the network in step 1


The method is tested on 4 and 5 -digit multiplication and GSM8k, and compared to various GPT2 baselines, chat-GPT and GPT-4.
The method has a stronger performance than models trained to predict the answer directly, but a weaker performance than models with explicit CoT reasoning. The method is however faster than explicit CoT reasoning models.

### Strengths
The proposed method is original and interesting. It tries to address a challenging task for language models, ie: multi-step reasoning

The paper is well written and clear to understand.

The experimental section contains interesting ablation studies that shows the importance of various components. It is good to see the effect of selecting different hidden states from the teacher network, the importance of mixture on GSM8k, and the importance of optimizing both the emulator and student network weights after coupling the two.

### Weaknesses
1. Given that the method requires a teacher model that does explicit CoT reasoning to distill into the emulator, it should be evaluated against these models. Unfortunately, the proposed method is weaker than explicit CoT models, although faster.
Overall, this method trades interpretability and performance (explicit CoT) for speed, and it doesn't seem like a good trade-off given the extensive literature on making faster inference Transformers.

2. Another weakness of the proposed approach is that it requires significantly more training data. This is potentially due to the much different way of training the student network compared to the pre-trained model as mentioned by the authors. Could a non-pretrained model be better at this task?

3. Eventually, the literature review section is very light. GSM8k has been a popular benchmark for many reasoning methods. The paper could benefit from further discussion on related methods to do multi-step reasoning. In addition, fast inference transformers is also an active research domain. Some discussion about the field should be added.

Comment:

- In Section 3.1, the $L \times T$ matrix of the teacher model comes from a Transformer architecture, which (by default) has an attention matrix over **all** previous layer tokens.
So the assumption that “_Progressing diagonally, from z11 to zLL, we gradually add more intermediate tokens and layers_” is not always true. For this assumption to be true, you also need to assume that the attention matrix of the teacher transformer is autoregressive, ie: conditional from left-to-right. Such clarification should be added. It is only clear that this is the case by looking at the architecture chosen (GPT-2) which is indeed auto-regressive.

### Questions
With the current selection mechanism of hidden states described in Section 3.1, if the number of layers is greater than the number of tokens (L=4, T=3, delta=0.66), then it may be the case that $t_l$ doesn’t reach the last token in the sequence ($t_l$ = [1, 1, 2, 2] with L=4, T=3). How often is this happening? Could you find a better selection formula?

In Section 4.1, the author mentions that “_For training the teacher of implicit CoT, to minimize the gap between the number of transformer layers and the number of intermediate steps, we only keep the equations_”. Did you try to include the original explanation? What was the impact on performance?

Overall, this work proposes one way to combine a $T\times L$ weight matrix into a vector of L weights. It seems like the authors also tried “first column”, “top row”, and “bottom row”. Why not “last column”? That seems to be the one containing the most information.

Did you try to train a model from scratch? Since the task is very different from pre-training, maybe the same performance can be obtained with less data on a network trained from-scratch?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes implicit chain-of-thought reasoning, where a language model is trained to conduct CoT reasoning internally without consuming the context window. The proposed method first a student model to predict the output based on the selected hidden states from a teacher model that conducts CoT reasoning. Then they train an emulator to predict the hidden states to mimic the CoT reasoning process. Finally, both student and emulator models are coupled and trained from end to end so that during the inference time, the whole system can conduct implicit CoT reasoning.

### Strengths
1.	The proposed method distills the CoT reasoning capability from a large model to a small one. The small LM does not have to consume its context window to conduct CoT reasoning.
2.	The paper is well-organized and easy to follow.

### Weaknesses
1.	As indicated by the authors as well, such implicit CoT reasoning is not interpretable and it is hard to tell whether indeed the proposed system is conducting CoT reasoning or is simply learning some reasoning shortcuts.
2.	The proposed method may not generalize compositionally to questions requiring more reasoning steps or just out-of-distribution data. It forces the model to conduct CoT with limited computation.

### Questions
Besides maths problems, does implicit chain-of-thought also work on other types of reasoning tasks? If not, what are the barriers that implicit CoT faces?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new way to solve complex reasoning tasks (particularly, arithmetic reasoning tasks) without explicit chain-of-thought (CoT). to improve the efficiency of LLMs in reasoning. The authors propose a pipeline with 3 modules: a teacher model that encodes the chain-of-thoughts and provides representations for the CoTs; an emulator model that is trained to generate the encoding results from the teacher model during inference; and a student model that directly predicts the answer based on the encoding results from the teacher model/emulator. Small-scale empirical evaluations demonstrate the potential of the proposed method, which achieves similar performance with CoT prompting and higher efficiency.

### Strengths
The idea is straightforward and the motivation is clear.

### Weaknesses
1. **The writing is not clear and some paragraphs are not rigorous enough.** The proposed pipeline includes 3 different modules, and the explanations of how they work are hard to follow. In the `information Extraction` paragraph on page 5, the representation of the CoTs is extracted from the diagonal elements of the matrix $z$. However, matrix $z$ is often not a square matrix. In that case, what does it mean to extract the elements from $z_{11}$ to $z_{LL}$? Does it mean that the hidden states of tokens after the position $T$ are discarded when $T<L$? Also, how to process the case when $L<T$?

2. **Missing an important baseline.** The proposed method can be regarded as **compressing CoTs into a vector**. There could be multiple ways to compress CoTs into a single vector and then train another model to generate the compressed representations for CoTs during the inference process. Why it is the best choice to extract the intermediate states of the teacher model as the compression results? From my perspective, a more elegant way to compress CoTs should be to train an auto-encoder to map the original CoTs into a single vector. The auto-encoder ensures that the compressed vector contains all the necessary information to recover the CoTs and can be used to predict the answer directly. Specifically, denoting the teacher model as $f_{T}(\cdot; \theta_{T})$ with parameter $\theta_{T}$, the auto-encoder model as $g_{enc}(\cdot;\phi_{enc}) and g_{dec}(\cdot;\phi_{dec})$ with parameter $\phi$, the student model as $f_{s}(\cdot; \theta_{s})$. We extract the CoTs from the teacher model given a question $q$ as $c = f_{T}(q;\theta_{T})$ where $c = (c_{1}, \dots, c_{N})$ is the CoT. Then the auto-encoder is trained with the objective 

   $$\max _{\phi} P(\hat{c _ i}|z, \hat{c} _ {<i}), \text{where} z=g(c;\phi _ {enc}) \text{and} P(\hat{c _ i}|z, \hat{c} _ {<i}) = g _ {dec}(z, \hat{c} _ {<i};\phi _ {dnc})$$

   Then the student model is trained to maximize the generation probability of the target answer conditioned on the question and $z$. The proposed method is exactly a special case of the above framework, where the auto-encoder is replaced by the proposed information extraction method. The author should demonstrate why the proposed method outperforms the general framework above, or why we should choose the proposed design. Otherwise, the method is too intuitive.

3. **Insufficient empirical evaluation.** The authors only verify that implicit CoT can boost performance. However, it is still not comparable with standard CoTs. Also, the efficiency improvement is less significant (or necessary) considering the poor performance of implicit CoTs. Finally, I would like to see further discussions on why should we consider implicit CoTs besides the reason for efficiency, especially considering the significantly increased training cost and data collection cost.

### Questions
Please refer to the weakness above. Although I give a low rating to this paper, I would be delighted to increase my rating given the questions addressed.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
