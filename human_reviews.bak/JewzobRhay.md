# When Do Prompting and Prefix-Tuning Work? A Theory of Capabilities and Limitations

- Decision: Accept (poster)
- Scores: 8, 6, 6, 6

## Abstract
Context-based fine-tuning methods, including prompting, in-context learning, soft prompting (also known as prompt tuning), and prefix-tuning, have gained popularity due to their ability to often match the performance of full fine-tuning with a fraction of the parameters. Despite their empirical successes, there is little theoretical understanding of how these techniques influence the internal computation of the model and their expressiveness limitations. We show that despite the continuous embedding space being more expressive than the discrete token space, soft-prompting and prefix-tuning are potentially less expressive than full fine-tuning, even with the same number of learnable parameters. Concretely, context-based fine-tuning cannot change the relative attention pattern over the content and can only bias the outputs of an attention layer in a fixed direction. This suggests that while techniques like prompting, in-context learning, soft prompting, and prefix-tuning can effectively elicit skills present in the pretrained model, they may not be able to learn novel tasks that require new attention patterns.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents an analysis of how prompting, soft prompting, and pre-fix tuning work for transformer models and why these methods are not as capable as full-finetuning for transformer performance. The paper presents theorems on why soft prompting is able to elicit a wider range of behaviors than standard prompting, and prefix tuning than soft prompting. The paper then presents an explanation of why prefix-tuning cannot change the behavior of a transformer model as much as full fine-tuning. The paper then investigates why, if prefix-tuning is less powerful than fine-tuning, then why does it produce good results in practice? The paper posits that this result is due to prefix-tuning being very good for biasing a transformer toward performing pre-trained tasks and that the results in practice are a result of the model already understanding the task from pretraining.

### Strengths
The paper presents some very significant theoretical results and their implications for using transformer-based models. In particular, the idea that biasing the outputs through prefix (or even prompting or soft prompting) elicits pre-trained skills from the transformer model and that these pre-trained skills can be combined through this means, is important to really understanding both why techniques like prefix and prompting work, but also provide insight into the emergent behavior of models like LLMs. It also leads to the intriguing question of whether there is some basic set of tasks that something like an LLM needs to be pretrained on, in order to be able to practically accomplish just about any task in natural language. 

The paper is also thorough in its investigation of the phenomenon of prefix-tuning and prompting by including both mathematical arguments for the claims made as well as simplified examples with actual code.

### Weaknesses
The paper has some clarity and soundness issues, from my reading. For clarity, I having trouble interpreting the attention figures in Figures 1, 3, etc. to see the patterns that the authors are trying to call out. Perhaps the image captions could include some interpretation guidance for the readers (e.g, the figures are meant to be read left-to-right, where …). 

For soundness, there were a couple of areas, where I was not fully convinced of claims by the provided proofs. For Theorems 1 &2, I think can see why those are true, but having more of a sketch as to why they are true would improve both the clarity and firmly establish why soft prompting and prefix-tuning are more expressive in output generation than prompting. And in section 5.2, how does prefix-tuning change the attention of the next layer? Earlier on in the article, it is argued that changes to the attention layer have the form of $ W_{v} + \Delta W_{V}$ (i.e., equation 7), and yet the equations of section 5.2 do not have any alterations to $W_{V}$ or $H$. Rather it looks like the prefix-tuning changes the inputs to the next layer of attention rather than the attention block itself.

### Questions
In addition to the previously mentioned questions, in equation 7, what is the equivalence between the pre-trained ($t_i^{ft}$) and prefix-tuned ($t_i^{pt}$) model outputs, or is there one? While I generally by the argument that the change to $W_v$ means more changes to the outputs than adding the $W_V s_1$ term to the equation for the outputs, I am not convinced that that is always true or under what conditions it might achieve equivalence. Thus, to really cement the claim that pre-training holds more potential for outputs than prefix-tuning, it would be interesting to see where adding the bias term from prefix-tuning can be (and can’t be) equivalent to adding an update to $W_V$.

### Soundness
3 good

### Presentation
2 fair

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Context-based fine-tuning techniques like prompting, in-context learning, soft prompting, and prefix-tuning have gained popularity due to their ability to achieve good results with fewer parameters compared to full fine-tuning. However, we lack a theoretical understanding of how these methods affect the model's internal operations and their limitations. This paper reveals that while continuous embedding space is more flexible than discrete token space, soft prompting and prefix-tuning are less expressive than full fine-tuning. This means that techniques like prompting and in-context learning can leverage existing skills in a model but can't learn entirely new tasks that require different attention patterns. This understanding provides insights into the capabilities and limitations of these fine-tuning methods, helping researchers and practitioners make informed choices when applying them to various tasks.

### Strengths
S1. theoretical support to prompt learning is a pressing need and this paper provided a comprehensive view from theoretical analysis.

S2. I like their presentation, which is clear.

S3. their theoretical analysis is interesting.

### Weaknesses
Overall, this paper tried to solve a very interesting problem. I would be very happy to raise my score if the following concerns are addressed:


W1: prompting is not only used in linear data like text but also applied to non-linear data recently like graphs. It would be more solid to discuss them in the related work section (e.g. X Sun, et al. "All in One: Multi-task Prompting for Graph Neural Networks". KDD2023). It would be even better if the author could further confirm whether their theoretical analysis applies to the graph prompting area.

### Questions
see W1

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper discusses the roles and limitations of context-based fine-tuning approaches (e.g. prefix fine-tuning) from a theoretical perspective. By analyzing the effects of attention mechanisms and computation within the model, the authors illustrate that there are structural limitations of prefix fine-tuning, while prefix fine-tuning is expressive due to continuous space. These limitations preclude prefix fine-tuning from learning new attention patterns, which makes it less expressive than full fine-tuning. The paper then reveals that the success of the context-based fine-tuning approach depends on the eliciting of skills in the pre-trained model, rather than learning new skills.

### Strengths
- This paper studies a valuable problem, which focuses on the capabilities and limitations of context-based fine-tuning, making a valuable contribution to the research community. 
- The theoretical discussions effectively highlight the problems posed by context-based fine-tuning.

### Weaknesses
- It would be better to further test models larger than LLaMA-7B as models with more parameters may exhibit different properties.

- An additional ablation experiment may be required in subsection 'Prefix-tuning can combine knowledge from pretraining tasks to solve new tasks'. The authors utilize a 4-layer 4-head model to validate the point that prefix-tuning can learn a new task as long as the “skill” required to solve the new task is a combination of “skills” the pretrained model has seen. However, in a 4-layer 4-head model, a prefix-induced bias can have non-linear behavior when passed through non-linear MLPs and attention blocks as mentioned In Section 5.2. It would be better to further test a 1-layer 4-head model for further clarification.

- Minor grammar issues
There are also several minor grammar issues, just a few:
'However, generation generation is more interesting'
'This can be be clearly from the activations'

### Questions
-	Is the cross-layer effect enough to make the prefix fine-tuning learn new tasks when the transformer consists of more attention layers like many current language models?
-	Is there any potential improvements for tuning methods based on the analysis?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents an insightful theoretical analysis of the limitations of context-based fine-tuning methods like prompting and prefix-tuning. While these methods have been empirically successful, the paper argues they are structurally less expressive than full fine-tuning and cannot learn tasks requiring new attention patterns. The paper tests these theoretical claims with minimal transformers and discusses the practical implications of these limitations.

### Strengths
+ The paper provides a valuable theoretical perspective on the limitations of prompting and prefix-tuning. It contributes to the understanding of how these methods compare to full fine-tuning in terms of their ability to learn new attention patterns, which is a significant contribution to the field.

+ The authors support their theoretical framework with empirical evidence. The use of minimal transformer models for testing provides clear illustrations of the theoretical limitations in a controlled experimental setting.

+ The paper's findings have practical relevance for the design of fine-tuning strategies in NLP applications. It helps practitioners understand when to employ prompting and prefix-tuning and when to opt for more expressive fine-tuning methods.

### Weaknesses
It would be beneficial to validate the theoretical findings with a broader set of experiments, including a variety of tasks, models, and datasets to confirm the universality of the proposed limitations.

A comparison with other fine-tuning methods such as transfer learning or domain-adaptive pretraining could provide a more comprehensive view of where prefix-tuning stands in the spectrum of fine-tuning techniques.

The paper identifies important limitations but does not provide detailed potential solutions or alternative methods that could overcome these limitations. Expanding on this could make the paper more impactful.

### Questions
The theoretical framework presented is compelling, but how does it hold up against the more recent transformer models that might use different mechanisms or have additional layers/heads?

How can the results inform the development of more efficient training procedures that could circumvent the limitations of prefix-tuning?

Could the authors provide a more detailed discussion on the practical implications of these findings for various NLP applications?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
