# Representation-Based Exploration for Language Models:  From Test-Time to Post-Training

- Decision: Accept (Poster)
- Scores: 8, 2, 8, 4

## Abstract
Reinforcement learning (RL) promises to expand the capabilities of language models, but it is unclear if current RL techniques promote the discovery of novel behaviors, or simply sharpen those already present in the base model. In this paper, we investigate the value of deliberate exploration---explicitly incentivizing the model to discover novel and diverse behaviors---and aim to understand how the knowledge in pre-trained models can guide this search. Our main finding is that exploration with a simple, principled, 
representation-based bonus derived from the pre-trained language model's hidden states significantly improves diversity and pass@k rates---both for post-training, and in a novel inference-time scaling setting we introduce. (1) For inference-time, exploration with representation-based diversity improves efficiency, consistently improving pass@k rates across a variety of models and reasoning tasks. For example, for Qwen-2.5-14b-Instruct we obtain over 50\% improvement in verifier efficiency on almost all considered tasks. (2) For post-training, we show that integrating this exploration strategy into an RL pipeline improves reasoning performance over that of the initial model and over standard RL post-training. For example, on AIME 2024, our post-trained Qwen-2.5-7b-Instruct's pass@80 matches the pass@256 of GRPO on the same model, demonstrating a 3x improvement in test-time sample efficiency. Overall, our findings suggest that deliberate exploration---with the right notion of diversity---is a practical path toward discovery of new behaviors beyond sharpening.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on the problem of exploration in the context of LLM reasoning. Existing RL methods are known to “sharpen” the distribution of responses v/s learning novel strategies for reasoning. To overcome this limitation, the authors propose RepExp—a simple yet principled exploration technique that leverages the internal representations of LLMs. Inspired by the literature of exploration in deep RL, RepExp encourages diversity in reasoning by providing an elliptical exploration bonus based on the model’s internal representations. The authors demonstrate the effectiveness of RepExp in two distinct settings: (1) a novel inference-time selection task, and (2) RL-based post-training. Across multiple datasets and problem types, RepExp consistently outperforms existing methods.

### Strengths
1. The paper is clearly written and highly accessible. Moreover, the exploration problem in the context of LLM reasoning is both timely and highly relevant, addressing a critical challenge within LLM reasoning.

2. The paper’s introduction of the inference-time selection task—serving both as an evaluation tool for exploration and as a standalone research problem—is an interesting and valuable addition to the study.

3. The proposed method RepExp is well motivated.

4. The results presented in this paper are both strong and compelling, and are effectively communicated through the well-structured Research Findings (RFs).

### Weaknesses
1. Section 2.1: (Line 143) ““Maximally diverse and high-probability of containing a positive response”: The inference-time selection problem is defined as selecting generations based on both diversity and quality. However, from my understanding, RepExp appears to select only for diversity. Could this lead to cases where the LLM produces hallucinated or incorrect reasoning that is nonetheless selected by RepExp due to its diversity?

2. In the RL post-training setting, the covariance matrix is computed using the hidden reprs of all the responses first, and then used to calculate div(x,y). From the motivation of elliptical bonuses (Lines 201-204), div(h,h<i), is the prediction error for a new “h” which is not in the training set. In this setting, isn’t “h” already in the training set? Is this approach then still principled? (prediction error is bounded by div(h|h<i))

3. “this method is history aware” – this line is a bit unclear, as it could be interpreted as div(x,y) is also conditioned on past-optimisation timesteps (which I assume is not the case). I think “history” can be a bit confusing for referring to responses which are often generated in parallel (which are therefore assigned the same generation timestep). 

4. How do other exploration-based methods, such as unlikeness or entropy, perform on the inference-time selection tasks? As the authors note, this setting provides a nice way to isolate and assess the impact of the exploration bonus.

5. (Line 403) “not optimized … hence does not yet give a wall-clock time improvement“ – doesn’t this method have a complexity of k (forward passes) * k generations * T? How can this be improved to be faster than naive autoregressive generation?

### Questions
1. Have you considered persisting the covariance matrix across training generations? — Can the LLM then explore novel strategies that are applicable across multiple problems?

2. Does RepExp lead to “novel” reasoning patterns? Moreover, is there a way to quantitatively assess the diversity of the generated responses—for instance, analogous to how state coverage is used in reinforcement learning?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
•	Principled and simple approach: The use of elliptical bonuses derived from model representations is conceptually clean and grounded in existing exploration theory, avoiding the complexity of auxiliary networks.
•	Broad empirical coverage: The paper evaluates across multiple tasks, datasets, and model families, providing a comprehensive empirical picture.
•	Clear dataset justification: The chosen benchmarks are well-motivated and represent a diverse set of reasoning and coding tasks.

### Strengths
•	Principled and simple approach: The use of elliptical bonuses derived from model representations is conceptually clean and grounded in existing exploration theory, avoiding the complexity of auxiliary networks.
•	Broad empirical coverage: The paper evaluates across multiple tasks, datasets, and model families, providing a comprehensive empirical picture.
•	Clear dataset justification: The chosen benchmarks are well-motivated and represent a diverse set of reasoning and coding tasks.

### Weaknesses
•	Overstated claims relative to results: The introduction makes strong assertions about “moving beyond sharpening” and “substantial improvements,” but empirical gains are modest and often inconsistent—particularly in RL post-training, where RepExp sometimes performs on par or worse than baselines (e.g., Figure 6).
•	Limited discussion of negative results: The method degrades performance for weaker models, yet this is not adequately analyzed or contextualized.
•	Unclear motivation and structure: The introduction is hard to follow due to numerous forward references and misplaced motivation (appearing in Section 2). The “Contributions” subsection mixes results with claims, making it difficult to distinguish novelty from outcomes.
•	Lack of practical relevance: The benefits at inference-time (reducing verifier calls) are meaningful only when verifier cost dominates, but the paper fails to demonstrate this in a realistic application scenario.
•	Inconclusive post-training value: For large k, RepExp sometimes underperforms Unlikeliness or GRPO; thus, it is unclear when and why this exploration actually helps.
•	Insufficient alignment between claims and framing: The paper promises exploration as a path toward discovering new capabilities, but the experiments focus narrowly on pass@k efficiency, not on qualitatively novel behaviors.

### Questions
1.	Can the authors clarify why weaker models degrade under representation-based exploration, and whether this correlates with representational quality or another factor?
2.	In the RL setting, what is the computational overhead of computing bonuses compared to standard GRPO training?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies representation based exploration for language models. An timely and highly important problem. Discovery of new and diverse behaviours would be a critical component for future revolutions in many AI4 science applications for example. 

The paper is exploring this question using many established based models and a decent selection of downstream tasks including AIME.

### Strengths
The paper is really nicely written and structured. There was a dedicated effort to make it easy to understand and follow the structure. 

The models studied include Llama, Mistral, Qwen and Phi and the tasks are representative.

### Weaknesses
Given the importance of the question in many domains I would have loved to see a similar study for protein language models and or how far the gap is in that area. That probably will however merit its own paper and is out of scope. 

Right now it states that "In addition, we have uploaded a zip file with the complete, anonymized
code for all our experiments and plots." Will the code, the data and the experimental setup be made publicly available after acceptance?

### Questions
See the question for code and data in weaknesses?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates representation-based exploration as a means to improve reasoning and diversity in language model behavior, both at inference time and during reinforcement learning post-training. The authors propose a simple yet principled elliptic bonus derived from a model’s hidden states to encourage exploration. Across several benchmarks (MATH, GSM8K, MBPP+, Game of 24, AIME), the approach yields significant improvements in verifier efficiency and maintains performance or improves at large Pass@k. The effect holds for larger models and, specifically, harder tasks.

### Strengths
- Adapting elliptical bonuses to language-model representations is well motivated, interesting, conceptually elegant, and grounded in prior exploration theory.
- Comprehensive empirical evaluation across multiple model families, scales, and sampling strategies.
- Significant improvements in sample efficiency.

### Weaknesses
- The proposed exploration strategy significantly degrades performance for smaller language models (e.g., Qwen-0.5B, Mistral-7B).
- The paper’s main narrative emphasizes discovering novel behaviors beyond sharpening, yet the results primarily reflect improvements in verifier sample efficiency. For large k, performance remains comparable to the base model, suggesting that the method broadens coverage rather than uncovering qualitatively new capabilities. The work would benefit from being reframed explicitly as a study in verifier-efficiency optimization rather than behavioral discovery.
- In the post-training experiments, the method consistently underperforms GRPO and Unlikeliness for small values of k.
- All benchmarks employed (GSM8K, MATH, MBPP+, AIME, Game of 24) feature cheap, automatic verifiers where verification cost is negligible relative to model inference. The paper would benefit from experiments in domains where verification is more expensive

### Questions
- Why did you choose Qwen-2.5-7b-Instruct for the post-training experiments?

### Soundness
3

### Presentation
3

### Contribution
3
