# What's the plan? Metrics for implicit planning in LLMs and their application to rhyme generation and question answering

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 8, 4

## Abstract
Prior work suggests that language models, while trained on next token prediction, show implicit planning behavior: they may select the next token in preparation to a predicted future token, such as a likely rhyming word, as supported by a prior qualitative study of Claude 3.5 Haiku using a cross-layer transcoder. 
We propose much simpler techniques for assessing implicit planning in language models. With case studies on rhyme poetry generation and question answering, we demonstrate that our methodology easily scales to many models.
Across models, we find that the generated rhyme (e.g. "-ight") or answer to a question ("whale")
can be manipulated by steering at the end of the preceding line with a vector, affecting the generation of intermediate tokens leading up to the rhyme or answer word.
We show that implicit planning is a universal mechanism, present in smaller models than previously thought, starting from 1B parameters. 
Our methodology offers a widely applicable direct way to study implicit planning abilities of LLMs.
More broadly, understanding planning abilities of language models can inform decisions in AI safety and control.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates whether LLMs exhibit implicit planning, meaning that they internally represent and use information about future outputs during text generation. Using rhyme generation as a controlled testbed, the authors introduce formal definitions and quantitative metrics of forward planning (encoding future goals before generation) and backward planning (conditioning ongoing generation on future goals). By manipulating hidden activations with mean activation steering, they show that altering representations at specific positions (e.g., line endings or newline tokens) systematically changes future rhyme families without modifying the input text, demonstrating controllable planing vectors. Experiments across various language models show that implicit planning behaviors are consistently observable and become more pronounced in larger and instruction-tuned models. The study further provides evidence that such planning mechanisms are structured and interpretable within model internals.

### Strengths
The authors' proposed steering vector based on the average activation difference appears to be directly effective, reflecting to some extent the underlying correlation between model activations and rhyme families.

### Weaknesses
1. The paper confines all of its experiments to the rhyme generation task, a setting that *has already been widely examined* in prior research. This narrow focus limits the generalizability of the findings and does not provide sufficient evidence to confirm the existence of an implicit planning mechanism.
2. The distinction between forward planning (the direct influence on the final output) and backward planning (indirect influence through the evolution of token probabilities) appears to describe two stages of the same representational process rather than two independent behaviors. The *lack of discussion on the connection* between these two aspects weakens the conceptual clarity of the work.
3. The proposed method and evaluation metrics are *relatively primitive and overly direct*. Modifying activation values essentially alters the contextual representation itself, making it difficult to isolate the planning effect. The experiments would benefit from deeper analyses, including an examination of cross-layer activation propagation and the relationship between attention distributions and rhyme outcomes.
4. The overall organization of the paper is confusing, and the figures are roughly presented and difficult to interpret. These presentation issues significantly reduce the clarity and accessibility of the work.

[1] Jack Lindsey, etc,. On the biology of a large language model. Transformer Circuits Thread, 2025.
[2] He, Jingkai, et al. History rhymes: Accelerating llm reinforcement learning with rhymerl. arXiv preprint arXiv:2508.18588, 2025.
[3] Jobanputra, Vedant Dhaval, et al. LLM-aided Evolutionary Algorithms for Haiku Generation. Proceedings of the Genetic and Evolutionary Computation Conference Companion. 2025.

### Questions
What is the relationship between the forward planning and backward planning proposed in this paper?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper investigates the concept of "implicit planning" in LLMs through the specific case study of rhyming poetry generation. The authors propose a set of simple, scalable metrics to quantify both forward planning (preparing for a future rhyme) and backward planning (adjusting intermediate tokens to lead to that rhyme). Using mean activation difference steering, they demonstrate the ability to manipulate the "planned" rhyme family across a diverse set of 23 open-weight models. The results show that planning capabilities for rhyming are present even in 1B parameter models and generally improve with model scale and instruction tuning. The paper concludes with a brief mechanistic analysis of the "rhyming circuit" in Gemma2 9B and with identifying specific attention heads involved in the process.

### Strengths
The paper's core strength is its departure from computationally intensive methods like "cross-layer transcoders" (I was not aware of these until this paper). The use of mean activation steering and straightforward metrics makes this work easily replicable and applicable across a vast array of models, as demonstrated by the authors' evaluation of 23 different models. There's a rich amount of steering literature to base this all on. 

The study isn't limited to a single large model but instead provides a panoramic view of how rhyming and planning abilities evolve across different model families (Gemma, Llama, Qwen) and sizes (1B to 32B). This comparison between and discussion around base and instruction-tuned models is particularly insightful.

The authors don't rely on a single metric. The combination of generation-based metrics (Fraction of Correct Rhyme), probability-based metrics (KL divergence, Top-1 difference), and regeneration metrics provides a multi-faceted and convincing case for both forward and backward planning - and "planning" at the log probs level might be controversial 

The analysis in Section 5.2, which pinpoints specific attention heads responsible for propagating the planning information, adds a layer of mechanistic credibility to the quantitative results and aligns with findings in related work.

### Weaknesses
The most significant weakness is the complete failure to discuss the role of tokenization. The model's ability to "rhyme" is profoundly influenced by its subword vocabulary!!!! For rhyme families like "-ight", words like "light", "night", and "right" likely share a common final token, making the task a simple exercise in token-level sequence completion rather than true phonological planning! The paper's entire analysis is conducted without ever acknowledging or controlling for this massive confounding variable. This oversight calls into question the interpretation of "planning" for certain rhyme families. I'm being very generous with my high scores in spite of this. 

The literature review is troublingly incomplete, which seems to have led to the first weakness. The paper misses several key pieces of prior work. Most notably, Roush et al. (2022) in "Most Language Models Can be Poets Too" dedicated significant discussion to the very problem of tokenization and vocabulary size impacting poetic and rhyming ability. It is a foundational paper for this specific topic and its omission is a serious oversight. Additionally, other works on steering model behavior, like SMC Steering by Lew et al, could have been mentioned to provide a broader context for the intervention techniques used.

Related to the tokenization issue, the paper notes that larger models are better but doesn't explicitly discuss why. A key hypothesis, supported by Roush et al., is that larger models tend to have larger vocabularies (i.e vocab size in subseqeunt versions of GPT has grown well beyond its original 50K tokens), meaning more words (and rhyming suffixes) are represented by single tokens, simplifying the rhyming task. This connection is a critical piece of the puzzle that the authors fail to mention!

I give a generous score given that I believe this paper in its current form has a glaring lack of discussion around Tokenization, however, I'm not convinced that this lack of discussion would invalidate the results and I think adding such a discussion is basically trivial, so I give a high score now - but if the authors do not commit to sufficiently cover tokenization in some way in the final manuscript I would be liable to reduce my score to weak accept.

### Questions
Could the authors comment on the role of tokenization in their findings? For rhyme families where the rhyming component is a single, common token (e.g., "-ight"), is the model truly engaging in "planning" in the same way it would for families where words are tokenized differently (e.g., words ending in "-ake" like "make", "take", "shake")? Have the authors analyzed if the performance of their metrics correlates with the token-level similarity of words within a rhyme family?

The circuit analysis points to specific heads in Gemma2 9B. Given the universal presence of the planning phenomenon across models, do the authors hypothesize that these are "rhyming heads" that are functionally homologous across different model architectures, or is it more likely that each model implements this capability with different, idiosyncratic circuits?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the implicit planning capability in large language models (LLMs), using rhyming poetry generation as the object of study. The authors propose a set of quantitative, scalable metrics to evaluate various aspects of implicit planning, including successful forward planning and successful backward planning.

The paper introduces the "mean activation difference steering" (MADS) method, which intervenes in the model's hidden states during generation to change the generated rhyme family. The authors created a dataset of rhyming couplets covering 10 rhyme families and used it to conduct a comprehensive evaluation of 23 open-source models (Gemma 2, Gemma 3, Qwen 3, Llama 3.1/3.2) across different series and scales (1B to 32B parameters).

### Strengths
- Clear Motivation: The paper is sharply focused on studying the implicit planning capability of LLMs, providing clear research intent.

- Simple and Effective Method: The paper proposes the mean activation difference steering method, which is simple and effective, serving as a replacement for more complex and computationally expensive techniques used in previous research.

- Extensive Experimentation: The authors validate their method across a wide range of popular model families and various model scales, leading to comprehensive experimental results.

### Weaknesses
- Writing Needs Improvement:

    - The paper's description of the activation manipulation position is somewhat confusing. Lines 157–159 mention that the manipulation target is the position of the newline character itself, while the immediately following formula on line 169 describes manipulation at the position preceding the newline character. This is confusing for the reader. Although the text immediately mentions that they experimented with both cases, I recommend the authors optimize the logical flow and presentation of this section.

    - Regarding the formula on line 169, the paper writes $P_{\text{RF}_2}^{\text{Train}} - P_{\text{RF}_1}^{\text{Test}}$. However, if I understand the proposed method correctly, the latter term should also be from the training set, meaning it should be $P_{\text{RF}_2}^{\text{Train}} - P_{\text{RF}_1}^{\text{Train}}$.

    - The figures in the paper are difficult to read due to small font sizes. It is recommended to adjust the typography in the visualizations.

- Limited Generalizability: The paper's study of LLM planning behavior is concentrated on rhyming poetry generation, and the generated content length is relatively short. Only a very brief discussion of Q&A tasks is included in the appendix. This limits the generalizability of the paper’s proposed metrics.

### Questions
- The proposed method appears similar to Classifier-Free Guidance (CFG) commonly used in diffusion models, but the paper does not mention this connection. Could the authors provide some discussion and analysis regarding the similarities or differences?

- How many training samples are required to obtain a relatively stable steering vector? The paper uses 85 samples. What would the effect of the steering vector be with significantly fewer samples, for example, less than 10?

### Soundness
3

### Presentation
2

### Contribution
3
