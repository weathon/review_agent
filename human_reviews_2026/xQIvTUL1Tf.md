# Efficient Reasoning with Hidden Thinking

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Chain-of-Thought (CoT) reasoning has become a powerful framework for improving complex problem-solving capabilities in Multimodal Large Language Models (MLLMs).
However, the verbose nature of textual reasoning introduces significant inefficiencies.
In this work, we propose **Heima** (as hidden llama), an effective CoT compression framework that condenses lengthy CoTs into a small set of abstract thinking tokens, preserving essential reasoning while removing redundancy.
We then conduct a theoretical analysis from an information-theoretic perspective, quantifying the information gap induced by compression, showing that reasoning capability is preserved when non-trivial mutual information is retained.
To further explore and quantify this information gap, we design the adaptive interpreter that maps thinking tokens back to variable-length textual sequences, thereby reconstructing the reasoning process.
Experiments across diverse reasoning benchmarks demonstrate that Heima improves reasoning efficiency, while maintaining or even achieving better zero-shot accuracy.
Moreover, the interpreter reconstructs coherent reasoning progresses from compressed thinking tokens, revealing that the information gap is minimal and validating the effectiveness of the proposed framework.
This work paves the way for scalable latent reasoning models and advances our understanding of efficient reasoning processes in large models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper considers multi-modal reasoning efficiency issues where input as multimodal (i.e., one image and one question), and textual reasoning with the final answers. Heima is proposed to condense each textual CoT steps into one specific token in the vocab via progressive distillation, and the paper additionally provide information-theoretic perspective analysis and tetxual CoT reconstruction via additional trained interpreter. The later part mainly is used to confirm the effectiveness of the proposed method, without other gains for the paper. The experimenal results validate the effective of Heima over two simple and weak baselines.

### Strengths
1. The targeted problem is interesting and valuable, but the aimed scope is limited.
2. The experiemntal results shows limited effectiveness.

### Weaknesses
1. the proposed method only consider one case that input is multi-modal, and the left part all based on textual reasoning and will not be affected by the input side. from this perspective, the contribution as first reasoning acceleration framework for MLLMs is weaken.
2. the progressive distillation is widely used in latent space reasoning, e.g., cocount.
3. despite the paper provide  information-theoretic perspective analysis, it seems no valueable insights, i.e., upper bound and lower bound, just extreme ideal situations.
4. also, the effectiveness can not be confirmed by the performance of interpreter, since there is no baseline or human evaluation. for example, although the reconsturcted sumamry by the interpreter is 4.1 out of 5 score (2.5/5 for caption), why 4 score is good enough? there is still a significant gap, weaken the statement of both  information-theoretic perspective analysis and value of the interpreter part.
5. other vicuna family model also is finetuned from llama family, it is more convincing to consider other family like qwen.
6. efficiency issue, since the method requires massive/multiple supervised fine-tuning in the progressive distillation stage.
7. baselines are too weak, whether or not the gain is worthy considering the cost to tune the model

### Questions
1. how many cot thinking tokens in total? what about the effect?

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
This paper addresses the inefficiency of CoT reasoning in MLLMs, which generates lengthy textual tokens and incurs high computational costs. To tackle the problem, this paper proposes Heima, a framework that compresses verbose CoTs into compact thinking tokens in latent space through progressive distillation. In addition, an interpreter is designed to reconstruct reasoning texts from thinking tokens for validation. Experiments show that Heima maintains or even surpasses the accuracy of the original CoT model while using only about 6% of the tokens, demonstrating efficient and effective latent-space reasoning.

### Strengths
1. The motivation of this work is interesting, as compressing Chain-of-Thought (CoT) to accelerate reasoning can effectively reduce computational costs, particularly in resource-intensive MLLM scenarios.
2. The proposed method is straightforward and effective, with comprehensive experiments validating its efficacy.
3. The information-theoretic analysis provides a solid theoretical foundation for the effectiveness of CoT compression, which is highly appreciated.

### Weaknesses
1. Although the idea of compressing CoT in MLLMs is promising, similar approaches have been extensively explored in the context of LLMs, suggesting that the methodological novelty may be somewhat limited.
2. Given that CoT reasoning demonstrates significant performance gains on more challenging tasks, the experimental evaluation should be extended to include more complex reasoning benchmarks, such as MathVision or OlympiadBench.
3. Beyond the LLaVA-Next and Llama3.2-11B-Vision models, it remains unclear whether the method generalizes effectively to other architectures, such as the Qwen-VL series. Theoretically, this should be a universally applicable approach.

### Questions
1. Can this method be effectively applied to long-chain CoT scenarios, such as those in QvQ or Virgo? If so, would the compression ratio be higher compared to standard CoT, and would there be a significant degradation in performance?
2. Additional experimental results should be provided to further substantiate the claims and explore the method's boundaries.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Heima, a framework for compressing Chain-of-Thought (CoT). Heima compresses the content of a CoT into an implicit Tinking Token, significantly reducing the number of tokens, and designs an interpreter to restore the CoT Token to text. The paper trains LlaVA-CoT and compares it with the original model on multiple datasets, demonstrating that Heima achieves excellent results in saving tokens.

### Strengths
1. This paper proposes a framework for compressing CoT, shrinking CoT into a single token, significantly reducing the length of the model's response while maintaining its quality.

2. This paper uses information-theoretic analysis and an interpreter to verify the effectiveness and rationality of the compression. The interpreter successfully reconstructs the reasoning process, confirming the minimization of the information gap.

### Weaknesses
1. The experimental setup in this paper primarily focuses on short-step reasoning. For longer, more complex reasoning tasks, such as intricate mathematical proofs or logical deductions, compressing the reasoning content into a single token raises questions about its impact on key information retention and model capability, requiring further experimental analysis. Adding longer-step reasoning might improve the paper's generalization ability.

2. In the experiments of Section 4, the paper only compares Heima with the pedestal model. Perhaps adding comparisons with other methods to improve reasoning efficiency would better highlight Heima's high efficiency.

3. In CoT task reasoning, the model may need to dynamically adjust subsequent reasoning paths based on intermediate results, such as in Rethinking. Heima compresses reasoning information into a single token. Will this affect the aforementioned dynamic adjustment, and will it limit the model's exploratory reasoning capabilities?

### Questions
Please see the  weaknesses.

### Soundness
3

### Presentation
3

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
The paper proposes **Heima**, a latent CoT–compression framework for MLLMs that replaces verbose textual chains of thought with a small number of **special “thinking tokens”**, learned via **progressive distillation**. It also introduces **LLM interpreters** that map the hidden states of these tokens back to text to estimate the “information gap,” alongside a brief information-theoretic argument (via DPI) that compressed tokens should retain task-relevant info.

### Strengths
- Experiments on MMStar, MMBench, MMVet, MathVista, AI2D and HallusionBench claim comparable accuracy to CoT baselines while generating far fewer tokens.
- **Interpreter idea for auditing latent reasoning**: training text-only LLMs to reconstruct reasoning from hidden states is a useful diagnostic to probe whether compressed tokens preserve semantics.

- **Token-efficiency results**: On several benchmarks, the method reports sizable **token reductions** (e.g., to ~6–10% of CoT tokens in places) with modest accuracy loss and sometimes gains relative to LLaVA-CoT.

### Weaknesses
- **Insufficient novelty and unclear justification.** The method is highly similar to *Coconut* (latent reasoning through continuous hidden states) and other latent CoT approaches such as *Cocomix*. The paper’s claim of being the first to extend such latent reasoning to MLLMs feels incremental.
- **Lack of rationale for multimodal adaptation.** The authors do not explain why a Coconut-style framework should work particularly well in the multimodal setting. Simply applying a latent CoT designed for text-only reasoning to MLLMs requires more theoretical or empirical justification.
- **Incomplete efficiency analysis.** The paper emphasizes token reduction but lacks concrete wall-clock latency, memory, or FLOP statistics under matched decoding setups.

### Questions
- Apart from the application domain, what are the most substantial differences between Heima and Coconut? Could other latent CoT approaches (e.g., Cocomix, CODI) also be applied in MLLMs?
- Coconut’s results were not particularly strong in pure NLP reasoning tasks. Why does a similar staged-distillation framework yield competitive results here?
- Why did the authors choose the Coconut paradigm over other latent reasoning methods? What multimodal properties make it more suitable?
- How does Heima perform when handling inputs with heavier visual content, such as multi-image or video scenarios?
- The experiments are limited to a narrow set of base models. How would **Heima** perform on more recent and widely adopted architectures such as **Qwen2.5-VL** or **InternVL 3**, which have stronger multimodal grounding?

### Soundness
3

### Presentation
3

### Contribution
2
