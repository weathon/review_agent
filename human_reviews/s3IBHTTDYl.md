# Language Models Need Inductive Biases to Count Inductively

- Decision: Accept (Poster)
- Scores: 8, 5, 6, 8

## Abstract
Counting constitutes a core skill underlying a wide range of tasks, such as formal language recognition, multi-hop reasoning and simulating algorithms. Generaliz- ing counting inductively is central to task success on out-of-distribution (OOD) instances where testing inputs are longer than those seen in training. While there is a large body of literature reporting poor length generalization in language models, few papers have tried to distill the “reasoning” failure to the simplest case of count- ing failure. We aim to provide a broader picture on whether various language model architectures can a) learn to count, and b) generalize counting inductively. This work provides extensive empirical results on architectures ranging from RNNs, Transformers, State-Space Models and RWKV. We present carefully-designed task formats, auxiliary tasks and positional embeddings to avoid limitations in general- ization with OOD-position and OOD-vocabulary. We find that while traditional RNNs trivially achieve inductive counting, Transformers have to rely on positional embeddings (PEs) to count OOD. Further analyses on interpreting the learned solution reveal that different PEs encode different inductive biases that facilitate counting in different task formats. As counting is the basis for many arguments concerning the expressivity of Transformers, our finding calls for the community to reexamine the application scope of primitive functions defined in formal charac- terizations. Finally, modern RNNs also largely underperform traditional RNNs in generalizing counting inductively, hinting at the tradeoff modern RNNs struggle to balance between parallelized training and maintaining their recurrent nature.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper questions the common belief that counting is a basic, easy-to-learn skill of transformer-based language models. It provides a detailed study of what it means for a language model to learn to count and the conditions that affect its ability to do so. The authors set up different counting tasks of varying levels of difficulty to test when and how transformers can learn to count. They compare different types of transformer models (mainly those with different positional encodings) and include comparisons to classic and modern RNNs. Results show that counting isn’t a natural skill for transformer and modern RNN models, especially when they need to generalize to counts they haven’t seen in training—a key requirement for inductive counting.

### Strengths
The paper is well-written, organized, and easy to follow. The authors back up their claims with strong evidence and well-designed experiments, and they clearly explain the purpose and outcomes of each experiment. In particular, I like that:
- The paper questions a basic assumption and shows that counting isn’t necessarily something language models can be expected to master but instead something they have to learn under specific conditions. I find this to be an important and illuminating insight.
- By testing various transformers with different positional encodings and comparing them to RNNs, the paper provides a broad picture of how different architectures handle counting.
- The finding that counting isn’t a built-in skill could shape future work in language model design and applications, as it highlights the need to consider how well a model can generalize basic counting tasks.

### Weaknesses
I couldn't find any major weaknesses in the paper. The one small issue I ran into was with the term “number word,” which wasn’t immediately clear. Adding an example early on—like explaining that you use decimal numbers as “number words”—could help readers follow this part more easily.

### Questions
I don’t have any further questions. The paper does a thorough job of explaining its goals and findings.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors study the effect that the choice of position encoding in transformers has on their length generalization for counting. Specifically, they train transformers (<= 4 layers) on various counting problems, and evaluate their OOD performance. The counting problems require transformers to output a token representing the number of occurrences of certain input tokens. The variants of counting include shifted start, where the first input token is an offset to the counts transformers should output, modular arithmetic, and selective, where multiple classes of tokens are present in the input, and the counts of each class are to be tracked independently. Overall they find differences between the five position embeddings studied (NoPE, Sine, APE, ROPE and SPE), and conduct deeper analyses in the appendices to explain some of the more surprising findings.

### Strengths
- Comprehensive empirical evaluation of the 5 position embeddings.
- Analysis is reasonably thorough. e.g. I like the finding that RoPE fails to do modular counting, but not if there is a BOS token.

### Weaknesses
Counting setting far removed from practical language models.


Issues with overclaiming in the interpretation and discussion of results. Specifically:
- "poor results for 1L and 2L models suggest that counting in Transformers may require a non-trivial computation budget". I do not think the results are strong enough to support this claim. Firstly, 4 layers are still only a small fraction of most practical language models (e.g. even llama 8B has 32 layers), so "non-trivial" is some what of a stretch. Secondly, this studies the "inductive bias", i.e. the outcome of the particular training dynamics on this toy problem; it's entirely possible that some other synthetic set up, or even pretrained language models, can be more efficient. Claims like this is better substantiated with interpretability on pretrained language models, or an expressivity argument
- "Our results motivate the integration of multiple PE schemas to take advantage of orthongonal strengths." At no point do you study the setting of integrating several PEs. In fact, your results seem to suggest the opposite: that sometimes having PE with more degrees of freedom can do worse than having fewer (e.g. how NoPE can do better at some tasks)

### Questions
- Do you think that expressing numbers in bases (in the sense of base 10) is a source of inductive bias that can help length generalization? Would your results differ significantly? 
- Can you draw high level lessons from this? How would the findings here help improve the design of position embeddings?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work emphasizes the limitations in language models for counting tasks. The authors examine various architectures, including RNNs, Transformers, and modern state-space models with a suite of experiments. They find that while traditional RNNs can easily count inductively, Transformers struggle without additional design considerations, and modern state-space models face degraded performance.

### Strengths
1. This work focuses on a specific problem, the counting task, for the language models. The authors conduct many experiments to investigate the ability to count systematically. 
2. This paper not only focuses on the standard transformer architectures but also investigates many popular modern architectures.

### Weaknesses
1. **Lack of Insights:** Although this work conducts many experiments to support their findings, it offers limited insights into the reasons behind the poor performance of Transformers and modern RNNs. I encourage the authors to provide more intuition or explanations for the observed empirical phenomena.

2. **Lack of Generality:** This paper focuses on the counting task, which I acknowledge is an important task. However, it remains unclear how performance on this task influences real-world applications. The conclusions are specific to counting tasks and may not generalize well to real-world scenarios.

3. **Lack of Novelty:** While the paper addresses an important aspect of language model performance, it does not introduce significantly new concepts or methodologies. Similar studies[1] have already explored the limitations of Transformers in various tasks. This work just conducts some experiments in different architectures and does not offer sufficient contributions.

[1] When Can Transformers Count to n?

### Questions
How can the conclusions of this paper generalize to real-world tasks, such as math, code, and so on?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the ability of language models to learn counting inductively. Specifically, whether they can generalize counting beyond their training data. The experiments train Transformers with various positional embedding strategies (APE, SinePE, RoPE, SPE, NoPE). The core finding is that Transformers require specific positional embedding configurations to count effectively, while traditional RNNs handle counting tasks more naturally. The experiments reveal that different positional embedding approaches have distinct strengths: RoPE excels at unbounded counting with shifted starts, SinePE and APE perform well on both modular and selective counting, while NoPE and SPE are effective only for selective counting. These findings demonstrate that different architectural choices and embedding strategies significantly impact a model's ability to learn counting inductively, with implications for how counting capabilities should be implemented in language models.

Generally, this paper was very well written and easy to read. I would say that the biggest weakness is in the introduction and framing of the paper, where the formatting is unclear and the strong experimental results don't come through. If you can improve the first couple pages, this should be an excellent paper.

### Strengths
The generalization splits and counting tasks seem reasonable to me, clearly thought out well. The different performance across all of the kinds of positional encodings is very interesting, and I love the contrast between modular counting and typical unbounded state counting.

It's really interesting to focus on the positional embeddings as the design choice that might build in an inductive bias towards being able to count. It seems like core cognitive skills are implicitly built into different architectures, and perhaps papers like this one will lead to something exciting and new!

### Weaknesses
The abstract is very wordy and spends too much time explaining why counting is important. I would shorten the first half of the abstract to two or three sentences.

The quote at the beginning of the introduction, I’m not sure who the quote is being attributed to.

I am surprised you didn’t cite the bootstrapping counting paper by Steve Piantadosi, Josh Tenebaum, and Noah Goodman. This seems like an important citation, as this paper makes good on Carey’s earlier ideas in a computational framework.

The formatting for the inductive counting principle makes it unclear where this principle is coming from. Is it from an earlier work, or is it just a general idea that you are making precise here? The reference to ordered lists of words makes the definition seem a bit idiosyncratic, but maybe I’m wrong about this!

### Questions
I would really love to see you take this analysis into the interpretability setting. With controlled datasets like this, tools like activation patching a distributed alignment search could be used to localize counting states within network representations, and then you could get a mechanistic analysis of how this ability develops!

### Soundness
4

### Presentation
4

### Contribution
4
