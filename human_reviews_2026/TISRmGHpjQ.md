# Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
Large language models demonstrate the intriguing ability to perform unseen tasks via in-context learning. However, it remains unclear what mechanisms inside the model drive such task-level generalization. In this work, we approach this question through the lens of off-by-one addition (i.e., 1+1=3, 2+2=5, 3+3=?), a two-step, counterfactual task with an unexpected +1 function as a second step. 
Leveraging circuit-style interpretability techniques such as path patching, we analyze the models' internal computations behind their performance and present three key findings. First, we identify a mechanism that explains the model's generalization from standard addition to off-by-one addition. It resembles the induction head mechanism described in prior work, yet operates at a higher level of abstraction; we therefore term it "function induction" in this work. Second, we show that the induction of the +1 function is governed by multiple attention heads in parallel, each of which emits a distinct piece of the +1 function. Finally, we find that this function induction mechanism is reused in a broader range of tasks, including synthetic tasks such as shifted multiple-choice QA and algorithmic tasks such as base-8 addition.
Overall, our findings offer deeper insights into how reusable and composable structures within language models enable task-level generalization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces an in-context learning task called "off-by-one additio", where examples like `2+2=5`, `3+3=7`, and `4+4=?` require the model to infer the hidden “+1” rule. Using mechanistic interpretability, the authors identify a three-part circuit of attention heads:

1. Group 3 – Previous-Token heads: detect local patterns by attending from each answer to the preceding “=”, capturing the off-by-one discrepancy.  
2. Group 2 – Function-Induction heads: attend from the test “=” to previous answers, encoding the learned “+1” transformation.  
3. Group 1 – Consolidation heads: attend to the current token and `<bos>`, integrating signals to produce the final output.

Ablation studies show that Group 3 → Group 2 → Group 1 forms a causal chain implementing the “+1” rule.  
The same mechanism generalizes to other tasks — off-by-*k* addition, shifted QA, Caesar ciphers, and base-8 arithmetic — revealing a reusable **function-induction circuit** within transformers.

### Strengths
1. The paper is clearly written and easy to follow, with a well-organized presentation of methods and results.  
2. The evidence and ablation studies are convincing, providing solid empirical support for the proposed mechanisms.  
3. The authors present an interesting and novel mechanistic explanation of in-context learning behavior in language models.

### Weaknesses
1. It is unclear how much of the observed flexibility of these attention heads originates from pretraining data. The paper assumes that the off-by-one addition task is novel, but it is plausible that similar unconventional arithmetic examples (e.g., math puzzles or riddles) appear in the pretraining corpus.  
2. This raises two key questions:  
   i. If the model were trained only on standard addition examples (e.g., `2+2=4`, `5+5=10`), would it still perform well on the off-by-one addition task?  (let's say we train them in an in context manner like the sequence of summations.)

   ii. If not, what aspect of the pretraining process enables these heads to adapt and represent such novel generalization?

3. These tasks appear quite simple; one would expect that even a relatively small transformer model, with significantly fewer layers, should be able to infer the in-context logic. With this in mind, I have the following questions:
i. If I replace the embeddings corresponding to these heads with zeros or random vectors, why can’t other heads—perhaps in later layers—compensate for that and still recover the in-context logic?
ii. What determines the specific role of each head? Is this role fixed during training?
iii. what happens if only use first few layers say 6 out of 24/36, can the model still recover the logic? Is the model uses other heads for doing so(if the answer is yes)?

Overall, I think the paper has a nice observation, but it is quite random to me that why certain heads are very important if we corrupt them other heads cannot compensate for that, and how much of all these in context learning capabilities is still linked to the pre-training phase.

### Questions
See Weaknesses

### Soundness
3

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
This paper investigates how large language models (LLMs) generalize to unseen tasks in-context by analyzing a counterfactual arithmetic task called off-by-one addition.
Using mechanistic interpretability techniques, particularly path patching, the authors identify a novel computational structure, the Function Induction (FI) mechanism, which generalizes the known “induction head” mechanism from token-level copying to function-level generalization.

### Strengths
- The off-by-one task is simple yet counterfactual, enabling precise mechanistic tracing. The experiments are carefully designed to isolate the function-induction process.

- The analysis spans six modern LLMs (Gemma-2, Llama-2/3, Mistral, Qwen-2.5, Phi-4), confirming the generality of findings.

- The introduction of Function Induction (FI) heads extends prior induction-head results to function-level abstraction, a conceptually and methodologically significant advance.

- The paper provides one of the most concrete demonstrations of compositional and reusable circuits in LLMs, aligning with recent interpretability goals.

### Weaknesses
- Ambiguous motivation:

In Line 41, the paper states that “our understanding is still limited, especially regarding more complex generalization scenarios involving unexpected elements or newly defined concepts in the task.” However, the actual experiments focus on a very simple and synthetic task (off-by-one addition). It is therefore difficult to claim that the study meaningfully addresses “more complex generalization scenarios.”

- Unclear mechanism of Group 1 (consolidation heads):

The discovery of the Function Induction (FI) heads is an excellent contribution, but the role of Group 1 heads, those that most directly influence model outputs, remains vague. These heads attend strongly to the first token (e.g., `<bos>`), yet the reason such attention is crucial is not sufficiently explained.

- Limited scope and real-world implications:

The paper’s experiments are all conducted on toy arithmetic or algorithmic tasks. While this design is justifiable for achieving clear mechanistic insights and is a well-accepted approach in the mechanistic interpretability community, the paper would be stronger if it discussed how the identified mechanisms might translate to more realistic or linguistically complex tasks.

Despite these limitations, I found this paper highly compelling overall. The experiments are exceptionally clear, the evidence is convincing across multiple models, and the identification of FI heads represents a meaningful step forward in mechanistic interpretability research.

### Questions
In Figure 2, Group 3 indeed behaves like a previous-token head, but it also shows strong attention to the very first token (e.g., `<bos>`). Why does it attend so strongly to the first token? What role does this attention play in the function induction process?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies in-context learning through the lens of a non-typical logic task: off-by-one addition. The idea is that inputs are two number sums (3 + 4) and outputs are the sum plus one (8). They use empirical mechanistic interpretability techniques to try to understand how the LLM solves the problem. In their results, they find a more abstract version of the induction head, they show multiple heads solve different parts of the task, and they show the same mechanism works with other, similar tasks beyond off-by-one addition. 

In more detail, the main results are as follows:

- Off-the-shelf LLMs solve the off-by-one addition task in context and perform better with more examples.
- Path patching can be used to identify which parameters are contributing to different parts of the task. Path patching works by taking activations from a base task (regular addition) and replacing them in a network prompted to do the contrast task (off-by-one addition). If, after the patch, the network solves the base task, this is evidence that the patched part of the weights were key to the contrast part of the task (the +1 operation, for example). In this paper, all patches are the weights for particular attention heads. They identify several heads (they call them function induction heads), for which indeed replacing weights of the contrast task with the base task leads the network to revert to base task behavior (100% accuracy on the base task, 0% accuracy on the contrast task).
- They explore several other tasks besides off-by-one addition, and show similar phenomena.

### Strengths
- The path patching results are quite compelling. It is very surprising to me that replacing the heads from the contrast task with those of the base task leads to any sensible behavior at all, much less reversion to correct base task performance.
- I appreciate that the authors presented several other tasks in section 5. The results are a bit more mixed on the other tasks, but are nonetheless consistent with the narrative presented in the paper.

### Weaknesses
- There was not enough description of the differences between the FI head and the FV head. All that is reported is that the heads appear at different layers. The paper should describe conceptual differences between the solution concepts. If there aren’t major conceptual differences, this is a concern. The fact that the heads appear at different layers could be explained by other factors, like different models.
- I would quibble with “Off-by-one addition is likely an unseen task to these language models and represents a novel challenge”. Model producers and benchmarks for in-context learning often include synthetic tasks that don’t correspond to typical learning tasks. I think it is likely safe to assume models are trained on this kind of thing as well.

### Questions
What is the FV head and how is it different than the FI head?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the mechanistic underpinnings of task-level generalization in LLMs through in-context learning. The authors use the novel, counterfactual task of off-by-one addition as a probe to understand how models adapt to new rules. This task requires a two-step process: performing standard addition and then applying an unexpected +1 function. Using circuit-style interpretability techniques, primarily path patching on the Gemma-2-9B model, the authors trace the internal computations responsible for this behavior. The authors find the identification of a circuit that generalizes the well-known induction head mechanism from the token level to the function level. Instead of just copying a token, this mechanism induces and applies an entire arithmetic function (+1) learned from the in-context examples. The circuit is composed of three distinct groups of attention heads: Previous Token heads, Function Induction heads, and Consolidation heads. Moreover, the authors show that this function induction mechanism is not specific to off-by-one addition. They provide evidence through head ablation experiments that the same core mechanism is reused by the model to solve a diverse range of other tasks requiring in-context rule adaptation, including off-by-k addition, shifted multiple-choice QA, Caesar ciphers, and base-8 addition. This suggests it is a general, flexible, and composable capability for task generalization.

### Strengths
[S1] This paper is well written and easy to follow even if without sufficient prior knowledge on this domain.

[S2] The experiment and analysis seem to be simple yet have effective coverage and depth.

[S3] The paper demonstrates analyses in the tasks where function-induction heads appear, suggesting that the function-induction heads in LLMs might be responsible for the capability to identify the off-by-one function and other new functions in in-context learning.

### Weaknesses
[W1] It is unclear how these results could generalize well and be universal across different models. Most analyses focus on the Gemma2-9B model, with function-induction heads only identified in other models but not sufficiently validated (we can see the results of Mistral/Llama in Appendix D.2) as done in Gemma2.  Moreover, usually the lines of mechanistic interpretability research (esp. induction heads) have started with simple toy models (e.g., a few layer attention networks) to clearly point out the target circuits in the model, and then expanded the analysis to LLMs. In contrast, this work just began with LLMs, where we could not exactly figure out the circuits. I think we might not be able to have higher confidence on the conclusion than the previous works.

[W2] It's not clear what aspect of function induction heads could be attributed as a generalization of induction heads. While induction heads could have a clear explanation about the role of key/value/query in the attention, the counterparts in function induction heads are unclear.

[W3] From the attention analysis in Figure 2 and Appendix D.2, I think it is quite challenging to distinguish among Group-1/2/3 heads at a glance. They all looked similar. Also, the analysis in the main paper is only about Group-2 heads (Figure 5/7). It may be necessary to clearly explain how we can classify those heads into Group-1/2/3 in the main text.

[W4] Limitation is not discussed in the main paper. Because the mechanistic interpretability may not perfectly explain how LLMs work in practice, the extensive discussion should be included.

[W5] While the results are interesting and validated to some extent, it is not really discussed how this finding could contribute to the progress of how LLMs are developed in practice. It is not clear what the takeaways from this analysis can be.

### Questions
Please also see **Weaknesses** section above.

[Q1] How should these findings suggest the way we evaluate, train, prompt, or otherwise use LLMs in practice? This paper may benefit more from an extensive discussion of how the function induction circuits on these tasks could be relevant for practical uses of LLMs in the real world.

[Q2] Could we expect/predict how these explanations transfer to larger models (more than 10B/100B/MoE, etc) well?

[Q3] Minegishi et al. (https://openreview.net/forum?id=LNMfzv8TNb) have studied the extension of induction heads similarly; the mechanism not only copying necessary tokens, but inferring the task from the context tokens and predicting the next-tokens corresponding to the query. Could you discuss the relevance and difference between them?

### Soundness
2

### Presentation
2

### Contribution
2
