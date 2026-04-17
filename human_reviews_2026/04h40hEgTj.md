# Decomposing Prediction Mechanisms for In-context Recall

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
We introduce a new family of toy problems to explore challenges with long context learning and associative recall in transformer models. Our setup involves interleaved segments of observations from randomly drawn linear deterministic dynamical systems. Each system is associated with a discrete symbolic label that must be learned in-context since these associations randomly shuffle between training instances.

Via out-of-distribution experiments we find that learned next-token prediction for this toy problem involves at least two separate mechanisms. One "label-based" mechanism uses the discrete symbolic labels to do the associative recall required to predict the start of a resumption of a previously seen system's observations. The second ``observation-based'' mechanism largely ignores the discrete symbolic labels and performs a prediction based on the state observations previously seen in context. These two mechanisms have different learning dynamics: the second mechanism develops much earlier than the first.

The behavior of our toy model suggested concrete experiments that we performed with OLMo training checkpoints on an ICL translation task. We see a similar phenomenon: the model learns to continue a translation task in-context earlier than it decisively learns to in-context identify the meaning of a symbolic label telling it to translate.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors aimed at creating a family of toy models for exploring the known challenge of long-context learning for LLM. The proposed toy model have different time series data interleaved with distinct labels. The authors found that LLM developed two distinct learning mechanisms in performing next token prediction on the toy model. The first mechanism focuses on identity regime change in the data, and the second one perform next token prediction based the data observed. The two mechanism also seem to follow different learning dynamic, and the second one developed earlier than the first.

### Strengths
1. The author aimed at a crucial problem in understanding LLM, namely the challenge of  challenge of long-context learning.
2. The designed toy model indeed is simple in structure on one hand, but capturing some nature of human languages on the other hand. 
2. Quite extensive numerical experiments are conducted.

### Weaknesses
1. The main message the author intended to convey is not very clearly presented. It appears to be the discovery of the capability of Transformers on developing distinct mechanisms for predicting different token positions in a single task via a study of a specially designed toy model. Although related statements in various places of the paper do not seem to always be precisely the same.  While the first two hypotheses are shown not to hold, the language in the description of the conjecture and its confirmation is very vague and puzzling.  Moreover, to confirm or deny such s strong conjecture, a much more thorough set of experiments needs to be designed and carried out, not simply an observation continuation and new initiation can not be distinguished. 

2. The connections between the observations, existence of distinct learning mechanism to the challenge of long-context learning are not explicitly stated.

### Questions
1. I feel that the critical question is that why and how the understanding of this toy model can help us understand the capability of long  long-context learning.of LLM. I don't see this is addressed in the paper. 
2. It appears that at any fixed time of learning, there will be much more time series data than discrete symbolic labels, is that the reason that the second mechanism develops much earlier than the first. 
2. How are the issues presented in Sec. 5(discussion) related to the problems and contributions presented in introduction? I failed to see the clear connections.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
ICL is a well studied phenomenon in the ML community. Various tasks, such as MQAR and regression, have been proposed to test the ICL capabilities of models in the past. The beauty of each is it both tests the model's ability to perform lookup operations (MQAR) and more complex operations only depending on the previous token (regression). This work combines these into a task using linear dynamical systems, where each system is marked in-context by a specific query label. Two observations are seen: the model uses the open-query label to perform the correct task, and the model uses past elements in the sequence to continue the task. These observations are validated by configuring the systems and states to align, allowing for a clear test of these observations in a controlled setting. Further investigating that these different mechanisms exist within these learned models, a mechanistic study is conducted separating out two circuits from within the model that have markedly distinct performance on the two different subtasks of recall and execution.

### Strengths
- This new tasks used to test ICL is appealing. It provides a nice link between standard ICL problems while keeping almost everything continuous, hence interpretable. The dynamics are quite intuitive yet retain significant depth to make an interesting analysis.
- The experiments investigate a variety of different interventions to test the hypotheses H1 and H2 and show, clearly, that models will learn to perform the correct task on the first token after the new-task identifier, and then relying on previous outputs to generate more tokens.
- The results regarding the disparity between 1-after and 2-after display a very interesting aspect of how these models learned to solve a task composed of a mixture of regression and associative recall
- The circuit analysis added depth into the difference between these two mechanisms as two truly separate aspects of the model.
- The writing is very clear, with claims and hypotheses which are most relevant to the reader highlighted in boxes, along with the distinct mechanisms all given separate colors for the reader to decern them.
- The toy model architectures are cleanly described in Appendix I.

### Weaknesses
- Much of the work (specifically the figures) is focused on the training dynamics of these model. While interesting, and should certainly be highlighted, the claims of the resulting model having these two distinct mechanisms to understand these linear-dynamics inputs typically are most important at the very end of training. This wasn't particularly central in the played results and rather had to be pulled out from the training dynamics
- The paper focuses on one specific task and found a property about ICL performance on this specific hybrid task. There was not any investigation into whether these same behaviors can be seen with other tasks (possibly other hybrid tasks), resulting in a possibly narrow applicability
- A few (and only a few) task design choices were not clearly described (see questions)

### Questions
- Is there any reason for selecting 5 dimensions? Did this coincide with the model able to learn it at the scales tested, where too much greater led to bad performance while any smaller made the task incredibly easy?
- Why train OLMo specifically and not some other language model like LLaMa? Was there any investigation into these tasks trained using different foundation models?

- Why specifically this setup? Did the close tokens help the model find the last token in the output of this current sequence? Why not always use the last token of any of the systems to be the input of the next one?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies mechanisms through which transformers can perform in-context prediction. 
In models trained on a novel synthetic task, the paper discovers two mechanisms ("label-based" and “observation-based”).
A further experiment on OLMo checkpoints provides further evidence from a translation task.

### Strengths
- The paper provides a new family of well-specified toy problems to study mechanisms used in Transformers for in-context recall

### Weaknesses
- The setup, as motivated in Section 1.1, appears quite specific. I was missing a motivation of why the setup is of broader relevance or interest, e.g. to language models, or the transformer architecture, etc. This is a concern especially as the paper mainly concerns empirical studies of toy models trained on a toy task.
- Interpretation of Section 3: Section 3.3, line 400: "0% edge overlap between the 1-after query and 2-after query circuits": As far as I understood the description in the section, the circuit finding strategy used here imposes no pressure towards overlapping circuits. Hence, it is conceivable that the reason for 0% edge overlap is just that the model has multiple redundant mechanisms for the two tasks, and the circuit finding algorithm happened to find different mechanisms when run on the two tasks. I'd appreciate if the authors can comment on this.
- Section 4: I didn't understand the task used here. On the one hand, the task is English-to-Spanish translation, on the other hand "we also change our analogous natural language setup to have in-context labels with no semantic meaning" What does this mean? How does this relate to the examples given in Appendix G?

### Questions
I'd appreciate if the authors can clarify if any of the weaknesses listed above may result from misunderstandings on my end. I'm happy to reevaluate my score on the basis of the response.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new methodology to study in-context behaviors in transformer models. They create a sequence which consists of segments of observations drawn from different distributions. Each segment begins with a special token, termed "symbolic punctuation label" (SPL), so model must choose between inferring the next observation based on the SPL or the observations in the context. They provide experimental evidence suggesting that the latter choice develops earlier in training than the first.

### Strengths
The paper proposes a new synthetic needle in a haystack task, which is interesting and novel. The experimental design of using 1) misdirected SPL and 2) synchronized observation are well-motivated. 
They also extend their analysis on OLMo checkpoints on translation tasks, which ground their findings with real-world evidence.

### Weaknesses
I find that many of the claims in the paper could be considerably strengthened and simplified. 
* I am not fully convinced that the label-based recall hypothesis (H1) is decisively ruled out. One could explain Figure 1b) simply from the fact that the model sees more observation tokens (the 1-after and 2-after query) than the open SPL token itself. It would be valuable to test whether increasing the representational weight or length of the SPL (e.g., by replacing each open-label with a multi-token sequence or embedding-enlarged symbol) causes the model to rely more on label information. 
* Figure 1a is not clear. the query and misdirection tokens (the parentheses) are identical. 
* The results on observation-based recall degrades with more systems (relative to the performance of 1-after query) is quite surprising and not well explained. Section 3 devotes much time to validating H1 and H2, but in my opinion does not spend enough discussion on how to explain the phenomena in Figure 3. For example, whether it reflects interference among observation traces, reduced signal-to-noise in embedding space, or limitations of attention span. A deeper discussion or ablation (e.g., varying the degree of interleaving or the correlation among systems) would strengthen the empirical interpretation.
* The paper concludes that “the model mostly leverages mechanistically different learned mechanisms for consecutive tokens,” but it remains unclear how many mechanisms exist in total (two, or more?). The pruning analysis isolates only two (corresponding to the 1-after and 2-after query positions), and while these show 0% edge overlap, this alone does not establish that the model’s behavior decomposes neatly into exactly two circuits. Clarifying the scope of this claim would strengthen the mechanistic argument of the paper.

### Questions
The paper would benefit from an ablation on the length of each interleaved segment. Since segment length determines how many observation tokens are available for inferring each system’s dynamics, varying it could reveal whether the distinction between label-based and observation-based recall arises from token exposure rather than a fundamentally different mechanism. For instance, longer continuous segments might strengthen observation-based continuation, while shorter segments could force greater reliance on symbolic labels.

### Soundness
3

### Presentation
2

### Contribution
2
