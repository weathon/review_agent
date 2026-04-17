# Understanding Task Vectors in In-Context Learning: Emergence, Functionality, and Limitations

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Task vector is a compelling mechanism for accelerating inference in in-context learning (ICL) by distilling task-specific information into a single, reusable representation. Despite their empirical success, the underlying principles governing their emergence and functionality remain unclear. This work proposes the *Task Vectors as Representative Demonstrations* conjecture, positing that task vectors encode single in-context demonstrations distilled from the original ones. We provide both theoretical and empirical support for this conjecture. First, we show that task vectors naturally emerge in linear transformers trained on triplet-formatted prompts through loss landscape analysis. Next, we predict the failure of task vectors in representing high-rank mappings and confirm this on practical LLMs. Our findings are further validated through saliency analyses and parameter visualization, suggesting an enhancement of task vectors by injecting multiple ones into few-shot prompts. Together, our results advance the understanding of task vectors and shed light on the mechanisms underlying ICL in transformer-based models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a conjecture that task vectors (TVs) facilitate zero-shot inference by distilling from the original ICL demonstrations a single representative demonstration encoding. To support the conjecture, the paper uses linear-attention models to show that task vectors naturally emerge as a weighted summation of the in-context demonstrations. A key implication of this conjecture is that a single task vector, like a single demonstration, is limited to representing rank-one mappings. To test this limitation, the paper introduces a class of bijection tasks that require higher-rank mappings. Empirical results on practical LLMs validate that the standard TV method consistently fails on these tasks, while full ICL succeeds. Building on this finding, the authors propose an enhanced TaskV-M method that injects multiple task vectors, which successfully improves performance on these complex tasks.

### Strengths
- The organization of the paper is clear and easy to follow. Starting with a clear conjecture, the paper goes on to theoretically demonstrate the conjecture with a constructed simplified case.
- The use of a bijection task to identify a weakness of TV methods is sound and interesting.

### Weaknesses
- For table 1, while TV performs quite badly on bijection tasks, it seems that ICL is not much better. Given that ICL consistently outperforms TV on all tasks, what I can see most clearly is that the bijection task is much harder than other tasks, but not that "ICL preserves performance in many case".
- Some analysis seems to have been demonstrated by previous papers. For example, the findings of how information flows in ICL in Figure 3 (a,b) has been demonstrated in [1]. 
- I do not quite understand the implication of Figure 4c. It is difficult to see from Figure 4c that the performance of TV and 1-shot ICL are "parallel".



[1] Wang et al., Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning, EMNLP 2023.

### Questions
- Theoretically, can the TaskV-M method achieve the same performance as ICL?
- For the triplet tasks, how are the triplet demonstrations delimited? Why does the attention focus on the yi tokens instead of the delimiters?
- I might be a bit confused, but why is that, for the bijection task, in table 1, ICL outperforms TV, but in table 2, ICL underperforms TV?

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
4

### Summary
This paper looks into how task vectors emerge in ICL. The main idea is that a task vector, formed at a special prompt token, works like a distilled version of the whole demonstration set, basically acting as a single representative example. The authors support this through a theoretical analysis using linear-attention Transformers, showing that such task vectors can only express simple, rank-one mappings. They then back this up with experiments, including on large models like LLaMA-7B, which show that real models behave in similar ways. To deal with the limitation, they try injecting multiple task vectors into the prompt and find that this gives small but consistent gains. Overall, the paper offers a clear explanation for what task vectors are doing and what their limits are.

### Strengths
- The paper develops a clear and rigorous theoretical framework that explains the role of task vectors in few-shot learning. 
- A major strength lies in the bijection experiments, which directly validate the predicted limitations of single task vectors and demonstrate the practical implications of the theory. 
- The authors show that the same task vector behaviors emerge in large pretrained language models, reinforcing the relevance of the conjecture beyond simplified settings. 
- The proposed multi-vector strategy builds naturally on the analysis and provides an interpretable method to mitigate expressiveness constraints. 
- The clarity of writing, thoughtful experimental design, and theoretical depth make the paper both accessible and impactful within the ICL literature.

### Weaknesses
- The work builds on linear attention. I acknowledge that generalizing the proofs to standard Transformers is challenging, but the lack of a formal extension limits the scope.

- The multi vector strategy yields only moderate gains and likely needs further refinement for higher complexity tasks. Most of the tasks tested are short or simple, so it's still unclear how this approach would hold up on more complex reasoning problems.

- In Table 2 the task vector remains strong on bijection tasks, which softens the rank one limitation highlighted by Table 1. The paper should clearly explain why this happens and under which conditions the limitation manifests or is mitigated.

### Questions
- Please clarify why the task vector remains strong on bijection tasks in Table 2 despite the failure cases in Table 1. For example specify differences in prompt structure, presence of additional demonstrations, model nonlinearity, or training data effects that could compensate for rank limits. If this discrepancy is convincingly explained and bounded with clear conditions I am inclined to raise the overall score. 

- Could the multi vector approach be improved by learning to select or weight vectors adaptively rather than uniform insertion?

- Do you expect the task vector mechanism and its limits to extend to more compositional or long range reasoning tasks, and what evidence would most directly test this?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates task vectors in in-context learning (ICL), proposing that they act as distilled representative demonstrations. Through analysis of linear-attention models and saliency maps in large language models, the authors show that task vectors emerge from triplet-formatted prompts and encode task information via weighted combinations of examples. They identify a key limitation: task vectors are restricted to rank-one mappings, limiting performance on complex tasks. A multi-vector extension is proposed to address this, showing consistent improvements. The work provides useful insights into the structure and limitations of task vectors.

### Strengths
1. The paper presents a well-structured theoretical investigation into the emergence and functionality of task vectors in in-context learning (ICL), centered around the representative demonstration hypothesis and its implications under linear attention assumptions.

2. The qualitative analyses, including weight matrix visualizations and saliency maps, are clearly aligned with the theoretical framework and help to support the proposed mechanisms intuitively.

3. The paper evaluates its hypothesis across a diverse set of synthetic and linguistic tasks (e.g., linear regression, bijection), and the experimental outcomes reflect the predicted rank-one limitations, particularly in settings requiring high-rank mappings.

### Weaknesses
1. The theoretical analysis is limited to simplified linear attention models and does not incorporate components such as softmax attention, causal masking, or nonlinearity, which are central to modern transformer architectures.

2. While the appendix refers to prior approaches that leverage task or in-context vectors to improve performance, the empirical evaluation is limited to standard Few-shot ICL, TV, and the authors’ multi-vector variant. Including direct comparisons to these prior methods would help clarify the novelty and effectiveness of the proposed approach.

### Questions
1. The paper evaluates models such as GPT-J, Pythia, and Llama-2, but does not report results on more recent architectures like Qwen or the Llama-3 series. Do the rank-one limitations identified in this paper also persist in newer models such as Qwen or Llama-3 family?

2. The empirical evaluation focuses on synthetic and controlled linguistic tasks, leaving it unclear whether the observed rank limitations extend to more complex, real-world applications such as Question-answering or reasoning tasks. Have the authors explored such settings in their analysis?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper discussed three formats of in-context leanring: Single, Pairwise, and Triplet.
For Pairwise, the paper provides Theorem 1 to theoretically show that an L-layer linear transformer allocates one layer for embedding concatenation and utilizes the remaining L−1 layers to perform gradient descent.
For Triplet, the paper provides Theorem 2 to theoretically show that task vector naturally emerge from pretraining on triplet format.
These analyses are coupled with experiments in Sec 6.
The paper further show the task vector fails to predict on bijection tasks.
Sec 6 further shows multi-vector injection works better than vanilla

### Strengths
(1) The paper considers three ICL formats Single, Pairwise, and Triplet, making the view broad.

(2) The paper provides Theorems 1 and 2 for Pairwise and Triplet and gives clear explanations/intuitions.

(3) The paper identifies a task bijection that the one-shot approach TV cannot solve.

(4) The paper shows that multi-vector injection works better than the original task vector.

### Weaknesses
(1) Different from real-world LLM, the considered self-attention is without normalization, with position embedding as a one-hot vector.

(2) For Table 1, the conclusion is not well-supported. Will more advanced LLM lead to different conclusion?

### Questions
(1) I wonder whether L134, equation (2), should have "-" instead of "+" for loss.

(2) I did not understand L290-292. I believe the paper is discussing causal attention. Why is it said "as bi-directional attention allows"?

### Soundness
3

### Presentation
3

### Contribution
3
