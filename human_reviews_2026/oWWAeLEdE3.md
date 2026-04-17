# MoL: Adaptive Mixture-of-Length Reasoning for Efficient Question Answering with Context

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6, 2

## Abstract
We present Mixture-of-Length (MoL), an approach for Question Answering (QA) with context that aims to improve the balance between reasoning quality and response efficiency. Our method introduces a principled difficulty assessment based on information-theoretic principles and a dual-objective reward mechanism that adaptively modulates response length. In our experiments, MoL exhibits an emergent behavior termed "intelligent brevity": the model tends to produce shorter responses for simpler queries and longer ones for more complex inputs. This property is desirable for human-computer interaction and can reduce inference costs. A post-hoc analysis of internal activations suggests a correlation between this output adaptivity and the effective number of layers that contribute during inference. On multiple QA benchmarks, MoL demonstrates competitive accuracy while substantially reducing tokens compared to baselines, indicating that difficulty-aware length modulation is a promising direction for efficient QA with context.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a training framework called Mixture-of-Length (MoL). It produces short answers on easy questions and expand reasoning when the question is hard. The approach estimates task difficulty using average cross-document similarity and utilizing set-cover view of information redundancy across documents. They also train an LLM in GRPO-style with a dual reward that switches between a compression reward for correct answers and an extension reward for incorrect ones, with curriculum-style scheduling of the length coefficient. Experiments on HotpotQA, StrategyQA, and Loong datasets across several base LLMs show higher accuracy with fewer tokens.

### Strengths
- The problem tackled in the paper is important and well-formulated.
- The set-cover motivation and its practical proxy via average cross-document cosine similarity are clear and intuitive.
- Table 1 shows that accuracy increases with meaningful token reductions for several base LLMs, trained by the author's method.
- The paper is clear about limitations, ethics, and contains all needed information for reproducibility.

### Weaknesses
- Equating cross-document similarity with "easy" questions risks misclassifying questions that demand nontrivial reasoning over redundant sources. Note that authors treat a question as easy if its source documents look "similar", which they estimate by taking a few relevant sentences from each document and averaging how alike their embeddings are. This approach may work well on the evaluated datasets, but its generalizability to real-world scenarios remains uncertain. For example, imagine 10 near-duplicate news articles about a merger. A question like "By how many quarters did revenue grow between the two pre-merger reports?" requires number extraction, comparison, and arithmetic, despite the fact that the documents are very similar. In this case, high similarity means that their proxy call the question "easy", but the reasoning to answer the question correctly is non-trivial.
- Claims of "statistically significant" gains appear, but I did not find details on tests, confidence intervals, or seeds across runs.
- The paper compares "Tokens" across models and methods, but the tokenization standard (e.g., tokenizer family) and measurement point (prompt+answer? answer only?) are not specified; comparability is unclear when base models differ.

Minor issue:
- The text on some Figures (especially Fig. 5) is very small and hard to read.

### Questions
1. I would suggest a stress-test for your method to decouple document redundancy from reasoning depth, required to answer the question, using two new datasets:

- High-similarity + high-reasoning: near-duplicate docs but questions require multi-step computation/logic across them.
- Low-similarity + low-reasoning: stylistically diverse docs where each answer is a direct lookup.

If I understood the paper correctly, your proxy should call the first set "easy", and the second "hard", despite that the reality is the opposite. How will your method work in this situation?

2. At lines 209-212 you write: "The specific length thresholds are empirically set based on the response length distribution observed in the HotpotQA dataset". Can you please elaborate?

3. Why there is a "2" multiplier in the Formula 4?

4. The Figure 2 is hard to understand, since it doesn't really match with the text. What are these "compress" and "extend" stages, depicted on the Figure 2? In the text itself, I only see the description of the compression reward, not the distinct stage called like this. Can you please explain?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Mixture-of-Length (MoL), a reinforcement learning framework for question answering that adaptively controls response length based on question difficulty. Difficulty is estimated via an information-theoretic proxy: inter-document redundancy computed as the average cosine similarity of top-k question-relevant sentences (low similarity ⇒ harder).
MoL couples this with a dual reward scheme: correct answers receive a compression reward encouraging brevity, while incorrect ones get an extension reward promoting longer reasoning; a progressive schedule anneals the length coefficient over training.
Across three datasets, MoL improves accuracy while reducing output tokens. The approach generalizes to unseen QA datasets, and a post-hoc activation analysis suggests adaptive computation (fewer active layers on easy questions).

### Strengths
- The use of an information-theoretic proxy rooted in the set cover problem for assessing question difficulty is a solid theoretical motivation. 
- The dual-objective reward mechanism of MoL (compression and expansion) is well-formulated and empirically separated in ablations.
- The paper presents rigorous results across three QA datasets, as well as additional analyses and ablation studies that improve the reliability of central claims.
- MoL’s response-length adaptation is shown empirically with fewer tokens for easy and more for hard questions, outperforming strong baselines.

### Weaknesses
- Although Table 2 compares with passage-based similarity and HotpotQA labels, the difficulty prediction step relies on a Top-k sentence selection followed by embedding similarity. It is unclear how sensitive this is to k, embedding model choice, or sentence granularity. More systematically controlled ablation studies are needed.
- The paper shows strong overall gains, but lacks qualitative or quantitative analysis of scenarios where MoL fails — e.g., cases where over-compression leads to loss of crucial reasoning steps, especially in complex or long-context queries. Identifying such cases would strengthen the empirical claims and help users understand the trade-offs of the method.
- The analysis showing a correlation between shorter generated outputs and fewer activated layers is interesting, but the work does not demonstrate causal influence. It remains unclear whether the reduced computation is a deliberate model behavior or merely a post-hoc artifact of the compression objective.
- No human evaluation of response quality.

### Questions
1. On Difficulty Estimation: How sensitive is your difficulty metric to the choice of k (Top-k sentence filtering), the embedding model, and the document segmentation strategy? Did you experiment with alternative encoders or granularities? 
2. Error/Failure Cases: Can you provide examples where MoL over-compresses and fails to capture necessary reasoning steps (e.g. in complex multi-hop or long-context queries)? How often does this occur in practice, and are there mechanisms to detect or mitigate such failures?
3. Internal Activation Adaptation: Have you tried any controlled intervention experiments (e.g., freezing certain layers) to test whether the reduced activation in “easy” cases is causally induced by MoL, rather than being an emergent side effect of training?
4. Human Evaluation: Did you consider running a human evaluation on a subset of responses to measure perceived quality and readability? This would be especially valuable given the strong emphasis on token compression — are the shorter outputs still informative to end users?
5. Generalizability Across Tasks: Do you expect the MoL reward framework to extend effectively to other domains (e.g., dialogue reasoning, code generation, or fact verification)? What adaptations (if any) would be required?

### Soundness
3

### Presentation
3

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
This work proposes a reward function for reinforcement learning to improve the reasoning capability of large language models by adaptively controlling the length of generation based on the difficulty assessment. Basic idea comes from the prior studies in selecting the longer length of generation especially for complex reasoning tasks demanding longer inference, but shorter to avoid spurious reasoning. This work leverages the difficulty of a task measured by the similarity of retrieved sentences, i.e., the more difficult when almost no relevant sentences are extracted, and used it to adaptively choose the generation length heuristically. Experiments show that proposed approach achieves improvements when compared with related baselines.

### Strengths
- The proposed approach defines an interesting reward function to adaptively control the length of generation based on the difficulty of a task. The difficulty is measured by the similarity of retrieved sentences, such that the more similar implies the more easy to solve a task.
- Experiments are carried out systematically demonstrating gains on standard benchmarks when compared with other related methodologies. Further analyses show the generalization capacity over different domains and its relation of the difficulty assessment and accuracy.

### Weaknesses
- It is not clear whether the difficulty level could be attributed solely by the similarity of retrieved answer or not. There might be a case of a query which has a direct answer in a single sentence, and no similar sentences exist in the document pool. In this case, the task is treated as an extremely difficult instance leading to over estimates. I think it is better to show whether the difficulty assessment is correlated with the actual difficulty or not. It is possible to measure the accuracies in general by existing models or some evidence, e.g., the distribution of relevant passages in the document pool.
- The motivation of the heuristics to map the difficulty level to its target length is not clear. Probably it was judged empirically, but the more clear discussion will be necessary, e.g., the plot of the actual difficulty level of a task and its length for generation.
- Math notations are not consistent and several symbols are not defined. For example, $S_i$ is used as a subset of a document but $S_{ij}$ is defined as a scalar for similarity. $s$, Top-k and $Sim$ are not defined in Equation 2.

### Questions
See weaknesses.

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
The paper addresses the trade-off between reasoning quality and response efficiency in open-domain question answering with context. Traditional approaches often just compress all answers uniformly (risking under-explaining complex queries). This work proposes Mixture-of-Length (MoL), a framework to achieve ``intelligent brevity'': short answers for easy questions and longer reasoning for hard ones.

To achieve this property the authors proposed the following approaches. First, a question complexity via cross-document redundancy is estimated. Intuitively, if multiple context documents share overlapping information, the question is likely simple (high redundancy means low complexity), whereas if each document contributes unique facts (low redundancy), the question requires complex multi-hop reasoning. 

Second, a dual-objective reward mechanism (trained via reinforcement learning) adaptively modulates answer length. The model is rewarded for conciseness when correct and for elaborating more when the initial answer is incorrect. Concretely, MoL defines a compress reward for correct answers that encourages minimal sufficient explanation, and an extend reward for incorrect answers that encourages providing a longer, more detailed reasoning chain. The target answer length is dynamically set based on the estimated difficulty.

Empirical validation on three benchmarks - HotpotQA (multi-hop QA), StrategyQA (implicit reasoning QA), and Loong (long-document QA) – showing MoL achieves competitive or better accuracy with dramatically fewer tokens than baselines.

### Strengths
1) The proposed MoL framework intelligently balances brevity and thoroughness. Unlike prior methods, MoL's dual-objective approach allows it to produce concise answers for easy questions and expand reasoning for harder ones as needed.

2) By measuring cross-document similarity after extracting key sentences, the proposed difficulty metric can reliably distinguish simple retrieval questions (high redundancy across documents) from complex multi-hop ones (low redundancy).

3) MoL achieves significant token reduction without sacrificing accuracy, often even improving it. For example, with a Qwen-8B model on HotpotQA, MoL used ~50% fewer tokens than the original model yet boosted accuracy by +6.2 points.

4) Extensive experimental evaluation. The authors considered multiple base LLMs across three diverse QA benchmarks. They compare MoL with strong baselines and MoL outperforms them. The paper also includes an ablation study isolating the impact of each component. This comprehensive evaluation strengthens confidence in the results.

5) The paper is well-written and clearly explains the intuition behind MoL.

### Weaknesses
1) MoL approach may still struggle on cases where complexity isn't captured by document redundancy (for instance, a single-document question that requires complex logical reasoning or implicit knowledge). The method is currently tailored to multi-document scenarios, which might limit its direct applicability to single-document QA or other formats without further adaptation.

2) The framework relies on several hand-tuned thresholds and parameters that may need adjustment in new settings. It looks like prior choices of these hyper-parameters are somewhat arbitrary and task-specific. It’s not fully tested how sensitive the approach is to these values – a different domain might require re-calibration. The authors do include an ablation on some of these hyper parameters, but the complexity of tuning so many parameters could be a drawback.

3) The solution uses reinforcement learning approach GRPO which is known to be difficult to train and resource-intensive. The experiments used a high-end setup (64× A100 GPUs), indicating significant compute requirements to reproduce or fine-tune MoL. This raises concerns about accessibility and stability of the method – researchers or practitioners with less resource might find it hard to apply.

4) The evaluation focuses on F1 accuracy and token counts, which is appropriate, but it would be interesting to know if the conciseness achieved by MoL is indeed judged favorably by humans.

5) The paper reports a fascinating correlation between question difficulty, answer length, and the number of transformer layers effectively used. However, the authors acknowledge this analysis is post-hoc and correlational, not proving a causal link. This is not exactly a flaw in the method, but a limitation in our understanding of it. It opens questions: is MoL actually learning to skip computations for easier tasks, or is the layer activation difference a byproduct of generating fewer tokens?

### Questions
1) MoL's adaptive strategy relies on identifying when an initial answer is incorrect (to trigger extended reasoning). During training, this is determined by checking the answer's F1 against ground truth. How is this handled at inference time, when the ground truth isn't available?

2) The current difficulty assessment is tailored to multi-document QA, using cross-document redundancy as a proxy. How would MoL extend to scenarios in single-document QA or tasks like mathematical problem solving (where all information is in one passage but reasoning is needed)?

3) MoL introduces several hyperparameters (for reward balancing, difficulty thresholds, target lengths, etc.). Did the authors observe any sensitivity or need for tuning these on different datasets?

4) The paper finds a correlation between shorter answers and fewer activated layers. Can the authors shed more light on this adaptive computation aspect? For instance, did they analyse whether specific transformer layers are being skipped or pruned dynamically, or is it more that fewer tokens cause fewer layers to effectively contribute?

5) Given the reliance on RL and considerable compute, how feasible is it to apply MoL to much larger models or other settings?

6) Do the authors anticipate any challenges scaling to 65B or GPT-3-sized models with this approach? 

7) Are there any failure modes observed for MoL (cases where it chooses the wrong length adaptation)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Mixture-of-Length (MoL), a novel framework for question answering (QA) with context that aims to improve the balance between reasoning quality and computational efficiency. The authors identify a tension in current methods, which either compress responses uniformly (harming complex reasoning) or rely on simple heuristics for adaptive reasoning.

The authors demonstrate that this training leads to an "intelligent brevity" behavior, where the model naturally uses shorter responses for simple queries and longer ones for complex queries. Experiments on HotpotQA, StrategyQA, and Loong show that MoL-trained models achieve higher accuracy while simultaneously reducing the number of generated tokens compared to baselines . A post-hoc analysis also suggests this adaptive output length correlates with adaptive internal computation, where easy questions activate fewer transformer layers .

### Strengths
The paper tackles a significant and well-recognized problem: the trade-off between reasoning quality and efficiency (cost/latency) in LLMs. The goal of achieving "intelligent brevity" is highly desirable for real-world applications.

### Weaknesses
*The "Principled" Difficulty Metric Seems Oversold*

The paper's first contribution, the "principled difficulty assessment" is framed with a sophisticated theoretical grounding in information theory and the NP-hard Set Cover problem. However, the practical implementation is a simple heuristic: the average inter-document cosine similarity of k extracted sentences. This feels like a significant oversell: what is the **information theory** used? Under **which optimization** you derive the metric? What is the relationship between the NP-hard and the proposed difficulty calculation?


*The Difficulty Heuristic Is Potentially Flawed*

The core assumption that high similarity (redundancy) equals "easy" and low similarity equals "hard" is questionable and may not hold in many cases.

(Hard, High-Sim): A question might require synthesizing **subtle differences** between two **very similar documents**. The high similarity would incorrectly classify this as "easy".

(Easy, Low-Sim): A question might be a simple fact-retrieval task where the two facts just happen to reside in two completely dissimilar documents. The low similarity would incorrectly classify this as "hard".

*The Effectiveness Proof of Reward Mechanism*

The paper's core innovation is presented as the dual-reward mechanism (compress-if-correct, extend-if-wrong). However, this mechanism is entangled with a second mechanism: the dynamic L_target . The paper's claim of "emergent brevity" is muddled because it's impossible to know if the model is adapting because of **the dual reward** or simply because it's being **explicitly told to use a different target length**.

If the reward system can be truely useful to learn to reward or punish on the length, it is highly recommend to add a study to set the fix length for all difficulties, and check the performances.

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
2
