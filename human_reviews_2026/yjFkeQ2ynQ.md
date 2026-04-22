# All Code, No Thought: Language Models Struggle to Reason in Ciphered Language

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Detecting harmful AI actions is important as AI agents gain adoption. Chain-of-thought (CoT) monitoring is one method widely used to detect adversarial attacks and AI misalignment. However, attackers and misaligned models might evade CoT monitoring through *ciphered reasoning*: reasoning hidden in encrypted, translated, or compressed text. To assess this risk, we test whether models can perform ciphered reasoning. For each of 28 different ciphers, we fine-tune and prompt up to 10 models to reason in that cipher. We measure model accuracy on math problems as a proxy for reasoning ability. Across the models we test, we find an asymmetry: model accuracy can drop significantly when reasoning in ciphered text, even though models demonstrate comprehension of ciphered text by being able to translate it accurately to English. Even frontier models struggle with lesser-known ciphers, although they can reason accurately in well-known ciphers like rot13. We show that ciphered reasoning capability correlates with cipher prevalence in pretraining data. We also identify scaling laws showing that ciphered reasoning capability improves slowly with additional fine-tuning data. Our work suggests that evading CoT monitoring using ciphered reasoning may be an ineffective tactic for current models and offers guidance on constraining the development of this capability in future frontier models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates whether large language models (LLMs) can perform Chain-of-Thought (CoT) reasoning when the intermediate steps are written in ciphered, translated, or encoded text—a capability termed "ciphered reasoning." This capability is studied as a potential vector for evading AI oversight monitors. Through extensive experiments involving fine-tuning and few-shot prompting across 10 models and 28 different ciphers, the authors identify a notable asymmetry: models maintain high fluency/translation accuracy in ciphered text but suffer a significant drop in mathematical reasoning performance when forced to reason within that ciphered format. The paper correlates this performance drop with the cipher's prevalence in pretraining data and characterizes data/parameter scaling laws, concluding that ciphered reasoning is currently an ineffective evasion tactic for most models.

### Strengths
1.  **Systematic and Comprehensive Evaluation:** The study involves an impressive scale of experimentation, testing 10 models against 28 distinct ciphers using rigorous fine-tuning and prompting methods. This systematic approach provides the first large-scale empirical baseline for ciphered reasoning capability in modern LLMs.
2.  **Identification of a Critical Asymmetry:** The core finding—that models can fluently translate or comprehend ciphered text but struggle significantly when executing complex reasoning in that same format—is highly valuable. This asymmetry reveals a fundamental disconnect between the model's linguistic generation capacity and its internal computational mechanism, offering a clear target for future security research.
3.  **Insightful Correlation and Scaling Laws:** The analysis correlating cipher reasoning accuracy with pretraining data prevalence provides a plausible mechanism for the observed behavior. Furthermore, the identified scaling laws offer crucial guidance, suggesting that adversarial fine-tuning for this capability is prohibitively expensive for current frontier models.

### Weaknesses
1.  **Lack of Mechanistic Explanation:** The paper clearly states the existence of the asymmetry but stops short of investigating *why* it occurs. It fails to explore the underlying computational or architectural reasons for the collapse of reasoning ability when the surface tokens are altered. This leaves a significant open question regarding the core mechanism driving the reported phenomenon.
2.  **Limited Constructive Exploitation:** Although the paper identifies ciphered reasoning capability as correlating with pretraining data, it primarily utilizes this point for filtering guidance rather than designing a *new* method. For example, it does not propose techniques to intentionally break the correlation for defensive purposes, nor does it design ciphers that optimally exploit the weakness in models, minimizing the ratio of reasoning loss to fluency retention.
3.  **Reliance on Proxy for "Reasoning":** The study heavily relies on mathematical problem-solving accuracy as a proxy for general "reasoning capability." While math is a strong proxy for CoT-as-computation, the conclusions might not fully generalize to non-computational, symbolic, or logical reasoning tasks where the model might behave differently under ciphered conditions.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors study capabilities for encoded reasoning in many frontier models, spanning OpenAI, Anthropic, and open-weight models. They study many ciphers, ranging from very simple (adding a space between characters) to advanced ones (like base64). They elicit capabilities via both few-shot prompting and supervised finetuning. They find that, while models can often decode cipher text, doing ciphered reasoning results in substantial drops in accuracy.

### Strengths
Originality: Good. While prior work has found that steganography in frontier LLMs appears to be difficult, prior evidence was ad-hoc, and the paper improves substantially on prior works like https://arxiv.org/abs/2507.02737 

Quality: Excellent. Authors do extensive experiments to validate their claims and provide insight through follow up analysis. 

Clarity: Good. Paper is well written and easy to understand. 

Significance: Good. Beyond the object level finding, this work establishes a good playbook for evaluating frontier LLMs on their ciphered reasoning capabilities which could be easily incorporated into a standard evaluation procedure.

### Weaknesses
The authors only study instruction-tuned models. For the purpose of 'measuring whether current frontier models can do ciphered reasoning' I think this is fine. However, this is not a very general claim. Also, for the broader scientific purpose of 'determining the relative difficulty of ciphered reasoning vs plaintext reasoning' I think this is an unfair comparison, as mid-training datasets often contain examples of reasoning in plaintext. E.g. there is evidence that Qwen models in particular have been trained extensively on such datasets. https://x.com/stellalisy/status/1927392717593526780?s=46. I would appreciate some experiments on more diverse model families / base models. 

The authors mainly consider 'artificial' / pre-existing ciphers; however it is possible that language models can develop their own ciphers via semantic drift, as observed in https://www.antischeming.ai/snippets. This remains an important area for future work.

### Questions
Have you tried finetuning via RL? I expect that you would get better ciphered reasoning this way than training purely on SFT examples.

### Soundness
4

### Presentation
3

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
The paper systematically investigates large language models (LLMs) and their ability to reason in ciphered text, introducing the concept of “ciphered reasoning.” The authors evaluate 10 models across 28 different cipher schemes using both fine-tuning and prompting. They find that models can comprehend and accurately translate ciphered text but show a significant drop in reasoning performance, which correlates strongly with the prevalence of each cipher type in the pretraining corpus.

### Strengths
S1: The problem definition is novel — this is the first systematic and quantitative study of ciphered chain-of-thought (CoT) reasoning, providing valuable insights for AI safety and monitoring, especially regarding potential CoT-monitoring evasion risks.

S2: The experiments are comprehensive, covering diverse tasks, model types, cipher variations, and multi-scale settings (fine-tuning and few-shot prompting).

### Weaknesses
W1: The method itself lacks novelty; it combines existing techniques (fine-tuning, MATH benchmark, BLEU evaluation, cipher generation) with limited theoretical depth.

W2: The evaluation relies solely on MATH 500 as a proxy for reasoning, making the task scope too narrow.

W3: The definition and evaluation criteria for “gaming” behavior are not clearly or rigorously specified.

### Questions
The paper reports that billions of tokens are required to approach plain-text reasoning accuracy. Could the authors clarify how sensitive this result is to model initialization and learning rate scheduling? Also, is there evidence that smaller, more targeted datasets could close this gap faster?

### Soundness
2

### Presentation
3

### Contribution
3
