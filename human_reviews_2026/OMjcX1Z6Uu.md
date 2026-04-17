# Markovian Transformers for Informative Language Modeling

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Chain-of-Thought (CoT) reasoning often fails to faithfully reflect a language model's underlying decision process. We address this by introducing a *Markovian* language model framework with an autoencoder-style *reasoning bottleneck*: all information flowing from question to answer must pass through a bounded-length CoT, creating a bandwidth bottleneck analogous to the latent layer of an autoencoder. In practice, the KL penalty toward the pretrained distribution and the inductive biases of gradient descent discourage steganographic encoding, so the model learns to express its reasoning in natural-language steps from which the answer can be derived. We train this system with a GRPO-style policy gradient algorithm using parallel sampling, a frozen baseline CoT$'$, within-batch standardized advantages, and actor-reward (chain-rule) gradients. On QA tasks, Markovian training recovers most of the gains of a Non-Markovian GRPO variant while forcing the model to answer from the CoT alone (e.g., GSM8K: 19.6\% $\to$ 57.1\%; ARC-Challenge: 36.1\% $\to$ 79.9\%; on average within $\approx$3--4 pp of a Non-Markovian variant). Perturbation analyses across types and severities show that Markovian models incur systematically larger log-probability drops under CoT corruption than matched Non-Markovian baselines, indicating stronger causal reliance on the CoT. Cross-model evaluation confirms that learned CoTs generalize across architectures, suggesting they encode transferable reasoning steps rather than model-specific artifacts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Markovian Language Model framework. By training the system with GRPO-style policy gradient algorithm, the model yields large gains on QA tasks.

### Strengths
The proposed markovian transformers is interesting and I recognize its novelty. The paper did abundant experiments on various datasets, which makes the paper sound.

### Weaknesses
W1: Though you mentioned what is the baseline in your manuscript, but I didn't understand what your baseline is quite well. Also, I want to look at the comparison between yours method/architecture between the model/training method we are now using today. For example, I didn't see the comparison to LLM post-trained on CoT by teacher-forcing and RL methods. It is hard for me to judge its significance.

### Questions
Q1: In figure 1, while $o_1$ stands for question and $s$ stands for CoT, it is hard for me to understand what is the $o_i$ on the right. How do you get the $o_i$? If it is sampled according to $s_i$, then which part corresponds to the question?

Q2: In figure 2, can you explain why it actually performs worse on MMLU?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tries to improve the problem of chain of thought unfaithfulness, specifically when chain of thought isn't actually necessary for the model to reach its conclusions. The authors propose a Markovian framework wherein the model is blocked from attending to the original question during answer generation This forces all task-relevant information to be funneled through the chain of thought. To do this training, they use a GRPO-style algorithm and introduce further improvements like parallel sampling. After training, the models perform better on math reasoning benchmarks. The authors also use perturbation analyses on a Wikipedia text completion task to show that the chains of thought produced by their methods are more causally dependent than those produced by standard approaches.

### Strengths
- The problem (chain-of-thought unfaithfulness) is of broad interest to the field
- The method design seems novel and creative. It also seemed tricky to implement and the authors developed training strategies to make it work.

### Weaknesses
- Although the authors state that the approach improves performance on benchmarks, I don't think they compared to a baseline-- eg, what if you just did the training but didn't have the attention to the original question blocked?
- It would also be nice to have some qualitative discussions of the chains of thought that result from this training procedure, whether through example transcripts or through some grading of readability.

### Questions
- I'm not sure how to interpret figure 3. I'm assuming the x-axis ('sample') is number of tokens sample. Are there some examples of what these chain of thought transcripts look like? Are there parts of the transcript where the llama model has stated the answer?
- I also would like more clarification to interpret the perturbation analyses. Is this analysis testing how well the model adheres to the exact continuation of the wikipedia text? If so, why not do the same perturbation analyses on the previous QA tasks?

### Soundness
2

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
The paper introduces a Markovian Language Model (MLM) framework that enforces structural constraints on Chain-of-Thought (CoT) reasoning. The key insight is treating CoT as a "reasoning autoencoder" where the model must compress essential information into interpretable text that serves as a bottleneck between question and answer. By preventing the model from accessing the original question during answer generation (only seeing the CoT), the framework forces causal reliance on the CoT.

### Strengths
It is a nice and elegant idea to introduce causal reliance by construction. The distinction between "faithfulness" and "informativeness" is pragmatic and operationalizable. The reinforcement learning formulation seems sound and there is good improvement after training. There is interesting cross-model generalization where learned CoTs in one architecture transfer to another architectures

### Weaknesses
The paper oscillates between two different stories: compression (Wikipedia experiment) and sufficiency (for QA experiment).

The experiments are a bit superficial, there is only one LLaMA model and no baselines such as SFT and GRPO (yet the appendix F anyway shows some Wikipedia results for other models, then why not report results on the QA dataset as well?). 
It would be necessary to compare to other post-training baselines, especially on these datasets where any type of reinforcement learning. 
Furthermore, It is not clear what is the training data for the experiments in section 5.1, and there is no notion of uncertainty that is being reported (like std or confidence intervals). Appendix D.1 further discuss alternative reinforcement learning formulation, but it is not clear how and why the one in the main paper was chosen.

### Questions
How does the approach compare to baselines finetuning methods on QA dataset?
What was the training data in QA experiments?

### Soundness
3

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
The paper proposes a Markovian Language Model (MLM) framework that enforces a causal dependency between a model’s Chain-of-Thought (CoT) and its final prediction. The key idea is a “reasoning autoencoder” architecture that introduces a text-based bottleneck: the model must first generate a CoT, and only that CoT (not the original question) is used to produce the answer. The model is trained using a GRPO-style policy gradient algorithm, where the reward depends on how informative the generated CoT is for answer prediction. Empirical results show large improvements on reasoning benchmarks and higher perturbation sensitivity to CoT edits, suggesting the CoTs have become ''load-bearing.''

### Strengths
- The idea of enforcing a Markovian structure to make CoTs causally essential is novel, conceptually elegant, and well-motivated.

- The introduction of informativeness as a learning objective is interesting and moves beyond traditional notions of faithfulness or interpretability.

- The formalisation of the Markovian LM and integration of actor–reward gradients (where the reward depends on the same model parameters) are technically sound and well-presented.

- Empirical results show strong and consistent improvements on reasoning benchmarks.

- The perturbation sensitivity and cross-model transfer analyses go beyond accuracy metrics, probing whether the model is actually using CoTs.

### Weaknesses
- It is unclear which components of the method (Markovian bottleneck, actor–reward coupling, within-batch normalisation, or reward design) are responsible for the observed gains. The paper should include controlled ablations to isolate these effects.

- The informativeness criterion works well for deductive or mathematical reasoning tasks where the CoT captures logical steps. However, for non-deductive or knowledge-grounded tasks (e.g., MMLU, factual QA), informativeness alone may be insufficient.  Additionally, the authors claim that forcing the model to predict the answer only from CoT improves informativeness. Still, there is no baseline from which the model predicts (Question + CoT) under the same RL training. This comparison is critical to verify that the performance gains truly stem from the Markovian constraint.

- There is no comparison against recent process-supervised or fine-tuned CoT methods such as STaR. This makes it difficult to judge whether the proposed Markovian constraint provides a real advantage over established reasoning-enhancement techniques.

### Questions
Missing citations: 

1. Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning EMNLP Paul et.al. 2024
2. Truthful or Fabricated? Using Causal Attribution to Mitigate Reward Hacking in Explanations ICML Ferreira et. al. 2025

### Soundness
3

### Presentation
3

### Contribution
2
