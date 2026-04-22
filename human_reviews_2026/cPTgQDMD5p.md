# THE END OF MANUAL DECODING: TOWARDS TRULY END-TO-END LANGUAGE MODELS

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
The "end-to-end" label for LLMs is a misnomer. In practice, they depend on a non-differentiable decoding process that requires laborious, hand-tuning of hyperparameters like temperature and top-p. This paper introduces AutoDeco, a novel architecture that enables truly "end-to-end'' generation by learning to control its own decoding strategy. We augment the standard transformer with lightweight heads that, at each step, dynamically predict context-specific temperature and top-p values alongside the next-token logits. This approach transforms decoding into a parametric, token-level process, allowing the model to self-regulate its sampling strategy within a single forward pass.

Through extensive experiments on eight benchmarks, we demonstrate that AutoDeco not only significantly outperforms common decoding strategies but also achieves performance comparable to an oracle-tuned baseline derived from "hacking the test set"—a practical upper bound for any static method. Besides, we demonstrate an emergent capability for instruction-based decoding control: the model learns to interpret natural language commands (e.g., ''generate with low randomness'') and adjusts its predicted temperature and top-p on a token-by-token basis, which may open a new paradigm for steerable and interactive LLM decoding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces AutoDeco, a framework that aims to make language model decoding fully end-to-end by enabling models to predict their own decoding hyperparameters (temperature and top-p) dynamically at each generation step. The authors augment standard transformers with lightweight MLP heads that output token-specific decoding parameters, trained using pseudo-labels derived from optimization on ground-truth tokens. The approach claims to remove the need for manual or oracle-style hyperparameter tuning while maintaining negligible computational overhead.

### Strengths
1. Novel framing of decoding as a learnable, differentiable process. Treating temperature and top-p as learnable functions of context rather than fixed hyperparameters is an appealing conceptual shift that aligns with recent trends toward self-regulating and adaptive LLM inference.
2. Simple yet effective architecture. The addition of two MLP heads is computationally cheap and can be easily integrated into existing transformer architectures, which improves the practicality and reproducibility of the approach.
3. The observation that the model can interpret natural-language modifiers like “low randomness” to adjust temperature/top-p values is intriguing and opens a new line of research in interpretable controllability.

### Weaknesses
1. My concern is that it is not clear to me how they obtain labels for training prediction of temperature and top_p value per token. It is unclear how the argmax over continuous T > 0 is solved, what constraints or search ranges are used, and how noise in logits affects these derived labels. It is better to provide a simple example to explain this process. 
2. The experiments only compare against Greedy and Default Sampling (and an oracle). Missing are modern decoding methods such as Contrastive Search [1], Contrastive Decoding [2].

[1] https://arxiv.org/pdf/2210.14140
[2] https://arxiv.org/abs/2309.09117

### Questions
Please see the weaknesses

### Soundness
3

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
The paper introduces a method to learn the sampling hyperparameters (temp, top-p) end to end via finetuning, avoiding manual task-specific tuning. They also identify and give ways to solve challenges that arise when trying to do this naively. In the end, they find they can basically match task-specific manual tuning (the oracle upper bound).

### Strengths
- The idea is interesting and to my knowledge novel. 
- The problem space is richer than meets the eye (eg. how do you get training data for supervised training is surprisingly nontrivial). 
- The results are fairly convincing (Fig2). 
- They show this doesn't hurt performance (Fig3).

### Weaknesses
- In practice, nobody does pure autoregression in real world LLM usage at production-scale. Everyone uses speculative decoding of some sort, and it's not clear to me whether this sampling scheme permits that or breaks it. I would want to see an explanation or formal argument/construction for how speculative decoding would work when the target model has an AutoDeco head to be convinced this would not break speculation. Because if it does, then in practice it will never be used, which would be a major limitation. I think it can be made to work, though, it just needs maybe some explanation of how it would all come together. 
- The idea and science is good, but writing is mediocre in my opinion. The contributions section repeats the second contribution twice, omitting I imagine the third contribution (presumably the "emergent instruction-following"), the plots are not consistently formatted well (axes in Fig4 impossible to read), the name "AutoDeco" is neither catchy nor informative in my opinion, etc. 
- The "emergent ability" is not as surprising as the authors seem to think it is. You're training on a base model that conditions on language by construction, so I would expect this behavior as the language conditioning affects the model representations fed into the AutoDeco head. Maybe de-emphasize this, or figure out an alternative framing? But this is minor.

### Questions
See weaknesses.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work addresses an important practical issue in deploying large language models (LLMs): the need to manually tune decoding hyperparameters, which breaks the “end-to-end” workflow. To mitigate this, the authors propose AutoDeco, a method that augments standard transformer architectures with lightweight prediction heads that output context-dependent temperature and top-p values alongside next-token logits during decoding. Experiments across multiple benchmarks demonstrate that the approach can yield improved generation quality and adaptability.

### Strengths
1. The problem is timely and relevant. Removing manual tuning of decoding hyperparameters can significantly improve the practicality and usability of LLM-based systems.

2. The pseudo-label generation strategy for supervision is clever and helps circumvent the lack of direct ground-truth hyperparameter labels.

### Weaknesses
1. A key concern is that the model is trained to further increase the likelihood of the reference text. Since both pre-training and downstream fine-tuning typically already optimize for the likelihood of the ground-truth sequence, this additional adjustment may risk overfitting or reduce robustness in more open-ended generation settings.

2. While dynamic prediction of decoding hyperparameters is appealing, different applications may require different behavior. For example, customer support systems typically favor stability and consistency, whereas brainstorming or creative writing tasks may benefit from higher variability. It is unclear whether a single learned mechanism can generalize effectively across such diverse usage scenarios without explicit control, and it may limit adaptability when user preferences change.

### Questions
1. Can this framework be extended to other decoding parameters, such as top-k or repetition penalty?

2. In practical deployments, how should developers express or configure user-level preferences (e.g., consistency vs. creativity)? More discussion or empirical analysis would clarify how the method adapts to varying application requirements, especially in relation to the concern raised above.

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
2

### Summary
The paper introduces AutoDeco, a novel architecture designed to make large language models (LLMs) truly end-to-end by eliminating the need for manual decoding hyperparameter tuning (e.g., temperature, top-p). Traditional decoding strategies rely on static, hand-tuned parameters that must be manually adjusted for different tasks or even different parts of a generation. AutoDeco addresses this by augmenting the transformer with lightweight “decoding heads” that dynamically predict temperature and top-p values at every generation step.

### Strengths
Novelty and Conceptual Contribution: The paper identifies and addresses a fundamental yet overlooked bottleneck in LLM deployment, the manual, non-differentiable decoding process.

AutoDeco reframes decoding as a learnable and parametric component within the model itself, offering a principled step toward fully end-to-end generation.

### Weaknesses
1. While the emergent instruction-following behavior is a highlight, the explanation for why this arises is mostly empirical. A deeper analysis (e.g., probing whether linguistic cues correlate with latent space adjustments) would strengthen this claim.


2. Most benchmarks are reasoning or QA-oriented. It would be valuable to test AutoDeco on creative writing, dialogue, or long-form generation, where decoding choices play a larger role. Human evaluation or qualitative examples of improved text quality would strengthen the practical impact.

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2
