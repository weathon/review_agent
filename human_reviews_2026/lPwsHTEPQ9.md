# Let's Let's Let's Let's... Understand Looping in Reasoning Models

- Avg Score: 4.40
- Decision: Reject
- Scores: 6, 6, 4, 2, 4

## Abstract
Reasoning models (e.g., DeepSeek-R1) use extra inference-time compute to write long chains of thought and solve harder problems. Yet they often loop---repeating the same text---especially at low temperatures or with greedy decoding. We take a step toward understanding why. We evaluate several open reasoning models and see looping is common at low temperatures. Within a family, higher capacity models loop less and for distilled models, the student loops far more even when the teacher rarely does. This points to imperfect learning or errors in learning as a key cause. We then demonstrate two ways errors in learning can cause loops, using a simple graph-traversal setup. First, when the correct next action is hard to learn but an easy cyclic action is available, the model puts relatively more probability on the easy action and gets stuck. Second, errors across time steps in a chain of thought can be correlated, which drives repetition. Finally, we discuss potential avenues for reducing looping and implications beyond looping.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper contributes a systematic investigation of repetition phenomena in large language models (LLMs) and large reasoning models (LRMs): (1) It first contributes an empirical evaluation of how different factors of the model impact the likelihood of repetition; (2) It then introduces a theoretical analysis by studying the hardness of learning by traversing step graphs.

### Strengths
- The paper is well written.  I believe the related work section is especially well written. It first introduces the benefits of instruction-fine tuning. However, LRMs (which can also be seen as instruction-fine-tuned) are worse at controlling repetition. The writing has strongly motivated the paper. All figures are also clearly presented throughout the paper. 
- The paper has an extensive workload. It has evaluated a series of open-source models over a series of factors. These sound experiments will add to the valuable knowledge of this field. 
- I am not familiar with learning theory, but I believe it is creative to use the star graph traverse problem to study the hardness of learning. The authors also train transformer models from scratch to study the star graph.

### Weaknesses
The major weakness is that the paper lacks a formal linking between the star graph and the real-world reasoning problem (e.g. math reasoning) that the paper studied. 
- Can the authors formalize the real-world reasoning problems as the star graph?
- If not, can the authors show a proof sketch for how real-world reasoning problems can be *reduced* to the star graph (in the theoretical CS way)? 
- If neither is possible, can the authors further abstract the two problems and show their links? 

I would give 8 to the empirical aspect of this paper and only 5 to the theoretical part. So now I am rounding to 6 and would like to hear from the authors.

### Questions
The following points are NOT weaknesses. I only take this opportunity to have research discussions with the authors. 
 
Q1: Are there any other methods studying the repetition problem, which corroborates your findings?  How about other RL methods (e.g. MTCS-style reasoning) which might mitigate repetition? Aside from that, have you read the work from Ryan Cotterell's group that sees LMs as a Turing machine / finite state machine (if my memory is correct)? Because the paper studies state transition, I believe some discussions on the theoretical CS would add more value.

Q2: What is the difference between repetition in standard text generation, and reasoning generation? What characteristics of reasoning make the repetition more (or fewer)? 

Q3: It is not compulsory, but why didn't the authors evaluate close-source models?

### Soundness
3

### Presentation
4

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
This paper investigates the phenomenon of looping behavior in reasoning LLMs, where models repeatedly generate similar or identical text segments during inference. The authors analyze several open-source reasoning models and observe that looping occurs more frequently at lower decoding temperatures, in smaller-capacity models, and in distilled student models compared to their teachers. The authors attribute these behaviors to two underlying mechanisms: Hardness of Learning, where models overfit easy but unproductive reasoning paths when true progress steps are difficult to learn, and Temporally Correlated Errors, where small prediction biases compound across time, causing self-reinforcing repetition under deterministic decoding. These two mechanisms are demonstrated through a simplified graph-traversal setup(star graph).

### Strengths
- **Novel and Relevant Problem**: The paper tackles looping, a highly observed failure mode in reasoning LLMs,  and tries to identify and understand this issue.
- **Empirical Breadth Across Open Models**: The authors study looping across a variety of open-source LLM families, sizes, and training paradigms.
- **Star-Graph Demonstration as an Effective Illustration**: While the star-graph environment itself is not novel, the paper uses it effectively as a minimal and controlled demonstration tool to isolate the proposed mechanisms, which are themselves meaningful contributions.
- **Rich Analyses and Interpretive Insights**: The paper offers a wide range of analyses and observations that go beyond a single factor such as temperature. It provides a more holistic understanding of looping behavior across different conditions
- **Clarity**: The paper is clear. The writing, structure, and figures make the argument easy to follow.

### Weaknesses
- **Limited Dataset Diversity**: The experiments use the AIME math problems (~30 items). While each problem is complex and multi-step, broader reasoning domains (e.g., logic, science, code) would confirm its generality. The paper can benefit from including additional datasets to demonstrate that the observed looping mechanisms generalize beyond a single domain.
- **Looping detection**: Even though the n-gram repetition for k times is practical, it does detect semantic loops(the model repeats the same idea using different tokens).
- **Limited Generalization**: While the star-graph setup effectively isolates looping mechanisms, it remains simplistic and may not fully capture the richness of reasoning trajectories in real-world LLM tasks.

### Questions
- How sensitive are your looping measurements to the specific definition of “loop” (e.g., n-gram size, k repetition threshold)? Could small changes in these detection parameters significantly alter reported looping frequencies, especially for longer CoTs?
- Given that the Phi model deviates from the general looping–temperature trends, do you expect similar family-specific differences to arise in other reasoning models trained under distinct architectures or data regimes?
- Do you think your findings would extend to MoE reasoning models?

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
4

### Summary
This paper investigates the phenomenon of looping, repetitive token generation often observed in reasoning models such as DeepSeek-R1 or Phi-4 Reasoning, particularly under greedy decoding and low temperatures. The authors conduct a large-scale empirical study showing that smaller and distilled reasoning models loop more frequently than their larger counterparts, and that temperature generally mitigates looping. To explain this, the paper proposes a graph-based formalization of reasoning as a random walk over nodes, demonstrating two mechanisms through which errors in learning can lead to loops: (1) hardness of learning, where a hard-to-learn “progress action” has its probability diffused across many alternatives while an easy cyclic action dominates; and (2) temporal correlation of errors, where small correlated deviations across time steps cause repetitive selections. The authors further discuss implications for model training, temperature tuning, and reasoning stability.

### Strengths
1. **Timely and valuable question**: Understanding why reasoning models loop is a meaningful and underexplored problem. The looping issue directly relates to the stationarity of model decoding dynamics and the stability of chain-of-thought reasoning.

2. **Interesting research direction**: Formalizing decoding as a graph traversal is conceptually insightful. It provides a potential framework to reason about “deadlock” loops in probabilistic generation processes.

3. **Comprehensive empirical evaluation**: The analysis across model families (Qwen, Phi-4, OpenThinker, LLaMA) and the controlled graph experiments are carefully conducted and informative.

4. **Clear empirical trend**: The paper convincingly shows that looping decreases with temperature and model capacity, reinforcing the empirical validity of the observation.

### Weaknesses
1. **Gap between graph formalization and language model decoding.**
The connection between the proposed graph reasoning setup and actual language model decoding remains vague. While the analogy to random walks is appealing, the paper does not rigorously justify how node transitions correspond to token-level or reasoning-step-level dynamics in real models. The mapping between “actions” in the graph and “reasoning paths” in language models needs clearer formal grounding. Incorporating perspectives such as finite-state automata (FSA) or Markov chain abstractions could make this connection more rigorous.

2. **Unclear or overly intuitive definitions.**
The key definitions (e.g., “hard actions”, “cyclic actions”, “progress-making actions”) are intuitive but lack formal precision. The notion of “reasoning” itself is used informally, and the theoretical treatment risks being too heuristic for a work positioned as a formal analysis. More careful mathematical framing and clearer assumptions would strengthen the paper.

3. **Proof of Proposition 1 is unclear.**
The derivation in Appendix C appears inconsistent. For example, in the cross-entropy expression, the second term should arguably be $-(1-p)\log{1-q_0}$ rather than $-(1-p)\frac{1}{n}\sum_{i=1}^{n} \log{q_i}$. If the authors intend to split the hard-action probability among n indistinguishable actions, the derivation should clearly justify each transformation step. Revisiting this proof and illustrating the simplification process would improve transparency.

4. **Conceptual overlap with prior works.**
The two looping mechanisms, learning hardness and correlated errors, partly restate well-known properties of autoregressive models (e.g., local maxima trapping, exposure bias). The novelty is thus limited unless the graph formalization offers a more rigorous bridge to these effects.

5. **Writing clarity.**
While the paper is well-organized, some sections (especially section 3 and 4) are dense with narrative explanations but light on formal statements. More concise mathematical exposition and illustrative examples would enhance readability.

### Questions
Could the authors formalize how the random-walk state transitions relate to token-level probability distributions in LLM decoding?

Would viewing the process as a finite-state or Markov model help in proving convergence/stationarity properties more rigorously?

Could the authors validate their theoretical claims with diagnostic traces from real LLM decoding (e.g., token probability trajectories during loops)?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates why looping behavior appears in chain-of-thought (CoT) reasoning models. The authors find that looping primarily arises hard and biased predictions.

### Strengths
Extensive experiments are done across both large-scale reasoning LLMs and small models trained from scratch.

### Weaknesses
The findings are not particularly interesting. They confirm intuitive explanations (e.g., low temperature amplifies biases, worse models tends to loop more) rather than offering new theoretical insight or actionable mitigation strategies.

### Questions
What decoding method is used in the experiments? It seems to be greedy decoding, but temperature should not affect greedy decoding, since the argmax operation is invariant to temperature scaling.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates why reasoning language models (LLMs) exhibit persistent "looping", repetitive output generation, especially at low sampling temperature or under greedy decoding. Through empirical evaluations on a large suite of open-source reasoning models on math benchmarks (AIME problems), the authors document that looping is severe in smaller or distilled models, lessens with model scale, and is largely absent in their teacher or instruction-tuned counterparts. A key finding is that temperature reduces looping but may introduce its own accuracy trade-offs. The authors then introduce controlled graph-reasoning experiments to demonstrate two underlying error mechanisms: (1) "hardness of learning," where progress actions are hard to distinguish, resulting in probability being diffused across alternatives, and (2) "temporally correlated estimation errors," which become amplified into loops under certain sampling regimes. The implications for future model training and inference practices are discussed.

### Strengths
1. The paper contains a thorough and systematic empirical study of looping behavior in a wide array of modern reasoning-focused LLMs, including multiple sizes, instruction/distilled variants, and reinforcement learning (RL)-tuned models, using a challenging math dataset. 
2. Proposition 1 formalizes the hardness of learning scenario with a precise and correct description; the proof in Appendix C is sound and maps cleanly to LLM training as described in Section 3. The connection between indistinguishability, softmax allocation, and looping is well justified.
3. The graph-reasoning tasks in Section 3.1 and Section 4 demonstrate the identified error mechanisms in highly controlled settings, making the analysis and implications compelling. Figure 4 and Figure 5 (as well as their supporting metrics) structurally isolate how action ambiguity or error correlation leads to looping and are valuable for the transparency and reproducibility of the arguments.

### Weaknesses
1. The core observation, looping/repetition at low temperatures, has been well documented for LLMs since Holtzman et al. (2020) and subsequent works, though the controlled graph experiments and direct focus on "reasoning" models add new clarity. The main conceptual contributions lie more in consolidating, empiricizing, and framing recent issues than in presenting an entirely new mechanism or solution.
2. Though the focus on AIME questions is appropriate, the generalizability of findings to domains beyond math reasoning is left largely unexplored. All main looping/accuracy/temperature analyses (Figures 1–2) are on this single dataset; it would be valuable to know whether looping dynamics differ in language, code, or multimodal settings (see e.g., how behavioral features in FlowVQA/other logic-heavy tasks compare).
3. Contemporary approaches such as contrastive decoding, unlikelihood training, entropy penalization, or dynamic decoding strategies are mentioned but not directly compared as baselines in the experiments (Section 2 and Appendix B). Readers are left to infer whether these approaches can alleviate looping on reasoning tasks beyond what temperature sampling achieves.
4. The bulk of mitigation discussion focuses on adjusting sampling temperature, but could more structured mechanisms, e.g., entropy regularization, adaptive n-gram avoidance, or neural decoding algorithms, also be relevant? The absence of empirical ablation or theoretical remarks on these means the prescription feels somewhat narrow.

Holtzman, Ari, et al. "The curious case of neural text degeneration." arXiv preprint arXiv:1904.09751 (2019).

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2
