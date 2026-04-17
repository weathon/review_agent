# Markovian Generation Chains in Large Language Models

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Large language models (LLMs) have already been widely used in our lives, so what happens when people repeatedly process text using these models? In this paper, we investigate the Markovian generation chain in LLMs: a fixed prompt is combined with the most recent output to produce the next output, and this procedure is repeated over multiple iterations. In our simulated iterative generation tasks (e.g., rephrasing and translation), the model's outputs may either converge to a set of similar results or continue to produce distinct outputs for a finite number of steps. While the outcome depends on the model, its configuration, and the input text, it is completely unlike the model collapse observed when models are iteratively trained on generated data. This process can be modeled and analyzed using a Markov chain, and it can be mapped to some real-world scenarios. Our study involves not only various LLMs but also Google Translate as a reference. At the sentence level, LLMs have the potential to increase the text diversity, for example, when the original text shows limited variation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper investigates repeated inference on textual data using LLMs, where the output of one turn is fed back as input for the next inference step. This is tested empirically via a rephrasing and translation task, and theoretically investigated via a proposed markov chain framework. 

The authors find that the iterative process behaves very differently depending on the decoding strategy, and conclude that this iterative inference process is unlike model collapse due to iterative training on repeated generated LLM data. Indeed, the paper finds that, contrary to some concerns, LLMs can potentially increase text diversity at the sentence level through this iterative sentence processing.

### Strengths
The paper is easy to follow and results/math is presented in a pedagogical way.
Interesting dichotomy between iterative inference with LLMs, and the more common modal collapse where models are recursively trained on LLM generated data.

### Weaknesses
**Concern 1:**

My main concern is regarding the significance of the contribution. 
Especially in relation to the scope and complexity of the experiments (See concern 2).

The results and findings from the experiments seem very intuitive, and almost self-evident. For example, since greedy decoding is deterministic and the number of possibilities relevant outputs are finite, we are bound to end up in a loop? Similarly, the more randomness/temperature the more different outputs one would expect.

This also goes for the discovered relationship between text length and the number of unique phrasings/translations. This also risks being so intuitive, that it might come across as non-interesting.

I will note that something does not need to be surprising to be informative. It could therefore be that these findings are crucial to the research community although not to me personally (hence my low confidence score in the review), but I was not convinced by the motivation in the paper. (See concern 3)


**Concern 2:**

The scope and complexity of the experiments seem (to me) to be too limited, to be of much interest. The paper focuses on single sentences in the current day where LLMs are capable of vast context sizes and generating long stories, programming projects, etc… So whilst it might be that recursive inference on LLM generated output leads to something interesting, I think readers would be more convinced if the scope of the experiments was increased.

I would suggest strengthening the appeal of the paper by perhaps setting up chains of agentic LLMs that pass information from one point to another. One could track how that initial information transforms, as more and more LLMs interpret/rephrase the information as they propagate it forward. Or something similar, where the scope is not only iterative transformations to individual sentences.

**Concern 3:**

The current motivation for why this task is important could use some strengthening. Some statements come across a bit hand-wavey, rather than feeling properly empirically/theoretically grounded. For example:

Line 23: “Researchers need to think more about where this chain will lead.” 
This sentence by itself is no motivation for why researchers should think about where this chain leads.

Line 109: “... which can be easily mapped to real-world applications such as translation…”
How exactly is this easily mapped to real-world scenarios? What scenario do you have in mind where LLM agents will, on the sentence level, transform/translate text during inference?

Line 113: “... which simulates the real-world phenomenon of text being processed multiple times by LLMs”
Similar to line 109 (and concern 2) I fear the reader might not trivially be able to see how your experiments map to real-world scenarios.

My suggestion would be to incorporate some of the final paragraphs in discussion and conclusion (Line 482) to an earlier part of the paper.

### Questions
Do you imagine it possible to expand your study/frameork from single-sentence transformations, to more complex and larger units of data?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the Markovian generation chain phenomenon in large language models (LLMs), where a model’s output is recursively fed back as the next input together with a fixed prompt. Through iterative rephrasing and translation tasks, the authors observe that outputs may either converge to a limited set of sentences (under greedy decoding) or remain diverse for multiple steps (under sampling). They formalize this process as a finite-state Markov chain and provide a theoretical analysis using entropy and KL-divergence properties to explain convergence patterns.

The paper claims to offer a new perspective for understanding repeated inference of LLMs and potential implications for text diversity and societal effects.

### Strengths
1. Clear presentation and solid structure — the manuscript is well written and easy to follow.

2. Systematic empirical setup — experiments cover multiple datasets (BookSum, ScriptBase, News2024) and several open models (Llama-3.1-8B, Mistral-7B, Qwen-2.5-7B, GPT-4o-mini).

3. Sound use of Markov-chain formalism — modeling iterative generation as a Markov process is mathematically reasonable and consistent with existing literature (e.g., Zekri et al., 2024).

### Weaknesses
1. Lack of novelty.
The main idea—treating LLM iterative inference as a Markov process—has already appeared in prior studies such as Zekri et al. (2024) “Large Language Models as Markov Chains” and other works. The theoretical framing and entropy analysis largely restate standard Markov-chain properties without introducing new modeling insights or learning mechanisms.

2. Conclusions are largely descriptive/common-sense.
The observed divergence under sampling and convergence under greedy decoding are expected outcomes of temperature-based sampling. The paper’s central message—that iterative generation can yield stable or diverse outputs—is already well known and does not advance understanding beyond descriptive confirmation.

3. No validation on downstream tasks.
The study remains entirely at the sentence-level simulation stage. There is no attempt to connect the theoretical model to real-world LLM performance (e.g., degradation of reasoning, factuality, or coherence under iterative prompting). Thus, practical relevance is limited.

4. Missing model-based improvement.
The paper does not propose enough modification or algorithm that leverages the Markov-chain view to improve existing models or inference processes. Without an actionable outcome, the contribution is also limited.

### Questions
1. Could your framework predict or mitigate real LLM degradation phenomena (e.g., semantic drift, factuality loss) in iterative settings?

2. Are there quantitative validations on downstream benchmarks (translation quality, summarization consistency) demonstrating that the Markov formulation offers predictive or corrective value?

3. Can you provide any model modification or inference adjustment informed by your theoretical analysis?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper investigates what happens during iterative inference when a LLM's output is repeatedly fed back as its next input along with a fixed prompt. The authors term this process a "Markovian generation chain". They draw a clear distinction between this concept and "model collapse," which is a phenomenon of iterative training.

Through simulations on rephrasing and translation tasks, the study finds two main behaviors: (1) Greedy Decoding quickly converges to short, repetitive loops of sentences. (2) Sampling-based Decoding can generate a large number of distinct outputs, potentially increasing sentence-level diversity rather than reducing it.

The authors propose a theoretical framework based on Markov chains, where sentences are discrete states. The LLM's operation acts as the transition matrix. In this model, the loops seen in greedy decoding are communicating classes, while sampling-based decoding explores this state space stochastically, eventually converging to a stationary distribution.

### Strengths
1. **Novel and Relevant Problem:** The paper addresses a highly practical and under-studied question. While model collapse (iterative training) is well-researched, this focus on iterative inference chains mimics real-world scenarios where users repeatedly edit, translate, or rephrase content using LLMs.

2. **Strong Empirical Evidence:** The simulations are thorough, covering multiple models, different domains, and the two distinct tasks. The inclusion of Google Translate is also a particularly effective comparison.

### Weaknesses
1. **"Diversity" vs. "Factual Drift"**: The paper primarily measures diversity as the "number of unique rephrasings". However, the provided example in Table 2 (Appendix) shows the Llama-3.1 model's output not just rephrasing "We begin with a prologue" but progressively adding new, unprompted information. The paper notes this as "information in the sentences having changed", but doesn't critically analyze this as a potential failure of semantic preservation (which the prompt explicitly requested ). It's unclear how much of the measured "diversity" is faithful paraphrasing versus hallucination and semantic drift.

2. **Applicability to real-world scenerios:** The simulations make sense, but it is hard to map this to any real world use case where a user will not repeatedly ask the same prompt. A genuine human-in-the-loop process  would involve user edits or, more likely, changing prompts at each step. This would create a non-stationary process, which the paper's static Markov model does not account for.

### Questions
1. **Explaining Model-Specific Behavior:** You show models behave differently in Figure 2, where Llama and Mistral are far more "creative" than GPT-4o-mini and Qwen. Qwen even appears to enter a loop under sampling. Why do you believe this is? Is it a result of different sampling parameters, pre-training data, or perhaps the fine-tuning process (e.g., some models being RL'ed into a lower-entropy, more stable state?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Given a passage, the model is prompted to produce a semantically equivalent paraphrase. The output is then recursively re-input for a total of 50 iterations. The paper analyzes what happens in this sequence, such as how many unique paraphrases are generated and how diverse the generated paraphrases are. While I appreciate the authors’ efforts at studying this phenomenon, I fail to understand why this process is significant at all and worth studying. The question of figuring out what happens when models are iteratively trained on its generated data recursively, which is studied in existing work and the authors point it out too, is clearly an important question because people use LLMs to generate text which is then put on the internet and becomes the training data for next iteration of frontier LLMs. But for the case of paraphrasing (inference) which is studied in this work, I don’t see why it is important and I don’t see a clear augment in the paper for why it is important either. I was also unable to see the precise contribution of the work from its introduction or abstract, and was unsure about what to take away from statements like “Researchers need to think more about where this chain will lead.”, “We also seek to explore the wider and deeper impact of LLMs on language and society.”

### Strengths
The paper carries out extensive analysis of how the paraphrases (when generated repeatedly by prompting the model on previous outputs) compare with each other, such as number of unique paraphrases and their textual similarity measured via automated metrics like BLEU and ROUGE, while using multiple LLMs. However the purpose of this analysis remains unclear to me.

### Weaknesses
As mentioned in the summary section

### Questions
NA

### Soundness
1

### Presentation
2

### Contribution
1
