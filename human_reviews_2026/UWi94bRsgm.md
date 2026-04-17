# Hey, That's My Model! Introducing Chain & Hash, An LLM Fingerprinting Technique

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Growing concerns over the theft and misuse of Large Language Models (LLMs) underscore the need for effective fingerprinting to link a model to its original version and detect misuse. We define five essential properties for a successful fingerprint: Transparency, Efficiency, Persistence, Robustness, and Unforgeability. We present a novel fingerprinting framework that provides verifiable proof of ownership while preserving fingerprint integrity. Our approach makes two main contributions. First, a "chain and hash" technique that cryptographically binds fingerprint prompts to their responses, preventing collisions and enabling irrefutable ownership claims. Second, we address a realistic threat model in which instruction-tuned models' output distribution can be significantly altered through meta-prompts. By incorporating random padding and varied meta-prompt configurations during training, our method maintains robustness even under significant output style changes. Experiments show that our framework securely proves ownership, resists both benign transformations (e.g., fine-tuning) and adversarial fingerprint removal, and extends to fingerprinting LoRA adapters. We release our code at: https://github.com/microsoft/Chain-Hash.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper tackles the problem of model ownership attribution through fingerprinting. Fingerprinting embeds certain canary query-response pairs into an LLM which are only known to the model owner. The paper proposes a cryptographically secure method of generating responses and fine-tunes the LLM with this set. The fine-tuning also incorporates data augmentation with meta-prompts for enhanced robustness. The paper tests if such fingerprinting leads to drop in model performance, and whether these are persistent despite fine-tuning and system prompts.

### Strengths
1. I like the evaluation of robustness to prompt changes, which is a direction often overlooked in fingerprinting research.
2. The threat model of false claims of ownership is novel and seems realistic.

### Weaknesses
## Evaluations
I believe that the evals in the paper are incomplete

E.1. All the evals reported in the paper are on classification tasks, and there are no generative evals. Training models on incoherent text could lead to model generations being incoherent, which should be tested through evals like IFEval or GSM8k. 

E.2. Baseline comparisons are completely absent. There is a claim that Xu et al does not produce harmless fingerprints, but this is not substantiated fully (line 99). Similarly, it is unclear if Xu et al can also be augmented with meta-prompts to lead to better results in Sec 5.7. Other fingerprinting methods (such as Wu et al, Nasery et al) are not considered at all.

E.3. I am also unsure as to why fingerprinting leads to such a large gain in performance for Llama-3.1-8B on TruthfulQA in Table 1.

## Security 
Further, some of the security arguments in the paper seem under-specified or flawed both conceptually and empirically

S.1. The paper does not explore collusion resistance properly, with sec 5.5 noting that "Even if c models collude, each will retain at least one unique chain segment, making complete removal computationally infeasible". However, the paper does not detail how to overlap the chains, or how long these overlaps should be. I would like to point the authors to prior work (Nasery et al) which looks at this in more detail. Nasery et al show that one needs on the order of 1000 fingerprints for a proper collusion resistant scheme (even under 3-way collusion), which might degrade utility under chain-and-hash fingerprints.

S.2. I do not follow the unforgeability arguments of the chain-and-hash scheme completely. Let's say I do not use the chain to set responses, but use a simple random number generator. Why does this not work for unforgeability? The adversary would still need to guess out of a large set of responses, right? A formal cryptographic argument here would make the paper stronger.

S.3. The false positive analysis in lines 284-289 seems misleading to me. Why should $p_{adv}$ be $10^{-3}$, especially because the response words are fairly common? There needs to be an empirical justification for this.

S.4. Several adversarial capabilities are not fully utilized - e.g. Output Manipulation means that an adversary can paraphrase output queries to evade detection leading to much stronger attacks. 

## Persistence 
Finally, the persistence results are slightly confusing

P.1. I do not fully understand the required number of trials metric. Is the model prompted with **all** fingerprints for n trials and verification is said to be true if atleast 2 of these are answered correctly? If so where is the randomness coming from?

P.2. Further, the persistence after fine-tuning on a dataset like Alpaca is not the most convincing metric,  because there could be a big overlap between the fingerprint responses and the responses in the fine-tuning dataset (Alpaca etc). Two ways for better experiments are to either use different SFT data, or use different fingerprint responses. 

P.3. It also looks like the persistence is pretty low for instruct tuned models, requiring over 25 queries without any prompting. I wonder why that is the case, since it also undermines the claim "While heavy fine-tuning reduces the fingerprint’s effectiveness, it persists in most cases." (line 74)

References

Nasery, Anshul, et al. "Scalable fingerprinting of large language models." NeurIPS 2025.

Wu, Jiaxuan, et al. "Imf: Implicit fingerprint for large language models." arXiv preprint arXiv:2503.21805 (2025).

### Questions
I would like the authors to respond to the weaknesses above.

Apart from that

1. How much does random padding help?
2. What kind of arbitration mechanism do the authors envision for model attribution? For example, how would someone lay a claim to a model behind an API, who would arbitrate such claims and how would fingerprinting help here, both for proving model ownership and fighting false claims of ownership?

### Soundness
1

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
This paper addresses the critical issue of intellectual property protection for LLMs. It defines five essential properties (Transparency, Efficiency, Persistence, Robustness, Unforgeability) for effective LLM fingerprinting, then proposes the "Chain & Hash" framework to meet these properties. 

The core contributions include two aspects: first, a cryptographic "chain and hash" technique that uses secure hash functions to deterministically bind fingerprint prompts to predefined responses, enabling irrefutable ownership claims and preventing collisions; second, a training strategy integrating random padding and GPT-4-generated diverse meta-prompt configurations to enhance robustness against output style changes from meta-prompts or fine-tuning.  

Experiments on four models show the framework maintains >95% fingerprint strength while preserving utility on benchmarks like HellaSwag and MMLU. It resists benign fine-tuning and adversarial attacks (e.g., INT8 quantization, meta-prompt interference) and supports black-box verification and LoRA adapter fingerprinting.

### Strengths
1. Proposes the "Chain & Hash" cryptographic technique, which uses SHA-256 to bind fingerprint prompts to 256 predefined responses.

2. Supports black-box verification that only requires API access aligning with real-world scenarios.

3. Extends IP protection to LoRA adapters by embedding fingerprints directly into these parameter-efficient fine-tuning modules.

### Weaknesses
1. The benchmarks used in the paper to evaluate model utility were proposed between 2019 and 2022, and the paper fails to verify the framework’s performance on new benchmarks released in the past two years.

2. The paper relies on GPT-4 to generate diverse meta-prompts for enhancing fingerprint persistence. However, key implementation details  are not mentioned in either the main text or the appendix .

3. The paper only compares its method with the black-box technique proposed by Xu et al. (2024), resulting in a limited comparison with other approaches .

4. The paper lacks systematic testing on how the number of meta-prompts affects the fingerprint strength of natural questions.

### Questions
1. Could Chain & Hash be extended to collaborative or federated ownership verification scenarios?

2. Please quantify the computational overhead of fingerprint embedding compared with standard fine-tuning.

3. The 14% utility drop observed in Llama-3-8B-Instruct (LoRA) is notably larger than that in other models. Could the authors clarify the underlying cause of this gap? Is it related to LoRA hyperparameters (e.g., rank r=4/8/16, learning rate 1e-4–5e-4), model-specific architectural factors, or the interaction between LoRA adaptation and the hash-binding mechanism? A parameter sensitivity or ablation study would help determine whether this degradation is inherent or tunable.

4. Can you provide quantitative curves showing how fingerprint strength varies with fine-tuning intensity (e.g., epochs or sample size), and analyze which parameter updates (e.g., attention layers) most affect the hash-bound question–response mapping?

5. Please clarify the necessity of the “gray-box” assumption in evaluating fine-tuned Llama-3-8B models. Would pure black-box verification fail entirely without prompt adjustment, and how does this align with the intended verification protocol?Could combining Chain & Hash with differential privacy or encryption strengthen verification security?

6. Could integrating Chain & Hash with differential privacy or encryption improve verification security?

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
4

### Summary
The paper introduces Chain & Hash, a scheme to design and insert fingerprints into models satisfying five properties: transparency (does not harm the utility), efficiency (does not require too much inference), persistence/robustness (fingerprint keeps working even if the model is prompted, fine-tuned, or quantized), and unforgeability (an attacker cannot provide models or responses that give false positives).

### Strengths
1. The framework introduced to guide the design of fingerprints is very thorough and carefully crafted. All of the listed objectives are important for a practical and useful fingerprinting scheme.
2. A key contribution is the usage of cryptographic tools (hashing) to ensure unforgeability: a computationally bounded attacker cannot claim ownership of a model if they cannot influence its responses (e.g. by having injected them). 
3. The fingerprint insertion is also carefully designed to satisfy the fingerprinting objectives.
4. Experimental evaluation is very thorough, carefully testing each property.

### Weaknesses
1. One weakness is that, in order to prove ownership of a model, the owner must reveal the matching chain. Once the chain has been revealed, it can no longer be relied on in the future, since the questions and answers are known; anyone trying to avoid being fingerprinted can easily evade the fingerprint once it is known. This necessitates multiple chains, however the impact of multiple chains on transparency is not explored and multiple chains are only discussed briefly in the collusion section.

### Questions
1. The analysis of the false-positive rate is based on the assumption that $p_{\mathrm{adv}}$, the probability that a non-fingerprinted model would respond with the expected answer is 1e-3. However, this constant isn't justified. Since the answers are chosen randomly from a list of 256 options, the probability can likely be bounded by 1/256 (of course in practice, it will likely be much lower, but it's important to control the false-positive rate carefully when the number of fingerprints is low).
2. Related to 1, there is no analysis of how many chains are required to detect colluding parties. Is the number of chains required exponential in the number of models?

### Soundness
3

### Presentation
4

### Contribution
3
