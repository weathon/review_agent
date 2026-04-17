# Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Fine-tuning on open-source Large Language Models (LLMs) with proprietary data is now a standard practice for downstream developers to obtain task-specific models. Surprisingly, we reveal a new and concerning risk along with the practice: the provider of the open-source LLMs can later extract the private downstream fine-tuning data through simple backdoor training, only requiring black-box access to the fine-tuned downstream model. Our comprehensive experiments, across 4 popularly used open-source models with 3B to 32B parameters and 2 downstream datasets, suggest that the extraction performance can be strikingly high: in practical settings, as much as 76.3\% downstream fine-tuning data (queries) out of a total 5,000 samples can be perfectly extracted, and the success rate can increase to 94.9\% in more ideal settings. We further investigate several defense strategies, but none achieve satisfactory effectiveness in mitigating the risk. Overall, we highlight the emergency of this newly identified data breaching risk in fine-tuning, and we hope more follow-up research can push the progress of addressing this concerning risk. Our code is available at \url{https://github.com/thu-coai/Backdoor-Data-Extraction}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates a critical security vulnerability in the fine-tuning of open-source LLMs—specifically, the risk that the provider of an open-source LLM can implant a backdoor that enables extraction of proprietary downstream fine-tuning data, even when only black-box access is available to the model provider. The authors demonstrate that with simple backdoor training, high-fidelity extraction of fine-tuning queries is achievable, presenting empirical results on several models (Qwen and Llama series, spanning 3B–32B parameters) and datasets (Dolly, Finance, MATH). The extraction risk is characterized in varying settings of adversarial knowledge and countermeasures, and the results show alarmingly high extraction rates (up to ~95% in ideal conditions). The study also evaluates several defenses, finding that none are effective without significant trade-offs.

### Strengths
- The paper shows that open-source LLM providers can later extract a large fraction of proprietary fine-tuning queries from downstream models. To my best knowledge, it addresses a risk not previously explored in this context.
- Results are presented across four recent LLMs (Qwen2.5-7B/32B, Llama3.2-3B/8B), and multiple datasets (Dolly, Finance, MATH).
- The paper is well-written and the figures are intuitive. Many of the problems I encountered during the reading process were more or less explained later.
- The paper even discusses and do some experiments about defense.

### Weaknesses
- While the proposed setting is interesting, the core observation that LLMs are susceptible to training data extraction is not entirely surprising. Previous research has established that LLMs inadvertently memorize and leak pretraining data, which is foundational to the current work's success. 
- I'm not an expert in backdoor attacks and privacy, but I noted that the Related Work section contains no literature from 2025. Could the author please explain why the recent sources are absent?
- The evaluation is primarily limited to only two downstream fine-tuning datasets: Dolly and Finance, each with 5,000 samples.
- The paper does not provide enough discussion or experiments about unlearning.
- My speculation is that a model with better instruction-following ability would be more vulnerable to the proposed attack. I suggest the authors conduct an experiment to verify this.

Minor: The dataset notation $D=\\{(x,y)\\}$ is slightly informal. Please use the more rigorous form: $D=\\{(x_i, y_i)\\}_{i=1}^n$.

### Questions
- Could you expand on the practical threat of the attack in real-world model deployment? For instance, more settings like quantization, pruning, distillation and parameter-efficient fine-tuning.
- The opening word mechanism is central to your extraction approach. How robust are your results to adversarial or randomized pre-processing of queries?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper demonstrates a method by which an open-source LLM provider can conduct (pre-)training in such a way so that, after any subsequent entity performs further fine-tuning on the model, it is possible to obtain partial recovery of the data that was used in the subsequent fine-tuning, simply given black-box access to the LLM. The method for doing so is simple – during pretraining, in addition to normal training data, the LLM is tasked to recall any datapoint starting with a corresponding given word in its training set. Additionally, a negative response is trained when the word does not exist in the training set. The authors show that this method, when trained with either SFT or RL, leads to significant extractability of the data from the subsequent fine-tuning.

Overall, despite the weaknesses I elaborate on below, and my low initial score, I do feel that the core idea of the paper is novel, the general set of experiments are relatively convincing, and that it can -- with revisions that address my concerns -- be a significant contribution to backdoor and LLM vulnerability research.

### Strengths
1. Introduces a simple yet effective, and potentially extremely concerning, new family of black-box data extraction attacks.
2. The experiments were conducted on a range of model sizes, across two families. I also particularly appreciated the inclusion and analysis of RL efficacy, in addition to SFT.
3. Detailed analysis of performance in different settings e.g. known first words, unknown first words, limit of perfect recoverability, temperature ablation in the appendix, etc.

### Weaknesses
1. Firstly, I think there is insufficient contextualisation and comparison to prior work. In particular, there is no comparison to [Privacy Backdoors: Enhancing Membership Inference through Poisoning Pre-trained Models (Wen et al, 2024)](https://arxiv.org/abs/2404.01231), which appears to examine the same problem setting. It is, in my view, critical to properly address the above paper and describe the precise differences in the threat model and general applicability of this paper’s attack to the one introduced in the paper above. 
2. The paper is awkwardly written and difficult to parse in many places. There are too many instances to enumerate fully, but for example I found lines 420-423 quite hard to parse: “Specifically, applying loss on input queries during training renders the LLM the ability to model queries’ distribution, enabling potential extraction later. However, successfully extracting these memorized queries hinges on the implanted backdoor instruction.
3. I did not find Appendix A’s motivation of introducing opening word conditioning particularly convincing. Partially, this is because the language used is obfuscatory and difficult to parse (see point 2 above); partially, this is because it is not made clear what the adversarial goal (i.e. the threat model) actually is. From the point of view of maximising retrieval of the entire dataset, for instance, Table 4 actually shows that not having opening-word-conditioning performs just as well as including it. This would seem to me to be the most natural goal of the adversary, and therefore, I am left puzzled why it is not the main attack used.
4. There are inconsistencies in the presented results – Table 4 reports the Raw and SFT Mean Match Ratio for Qwen 2.5 7B on the Dolly dataset as 18.6 and 40.9 respectively; however, in Table 1, these are 13.4 and 29.8 respectively. Further, I do not understand – and it is not adequately explained in the text – why in Table 4, SFT without opening word has significantly lower match rate/BLEU scores than Raw, despite a markedly better query and token-level extraction ratio. Does this imply that the approach has significant false negatives produced? I suspect this is so (given the large sampling size of 15000 used, which is much higher than that elsewhere in the paper). Which leads to my next point …
5. I would generally have liked to see a crisper analysis of false negative rates of the different methods used. Most of the analysis pertains simply to extraction rate/coverage, but does not adequately report the extent of false data that is also generated. Again, this depends on the goal of the adversary and exact threat model; perhaps there are cases where they are satisfied in getting a single real datapoint amongst thousands of hallucinated/incorrect samples, but this needs to be clearly motivated. More realistic in my view is that adversaries would like to recreate the original training dataset as closely as possible, including without false negatives.
6. Perhaps most importantly, the authors state in Lines 57-59 that this attack is predicated on fine-tuning without masking of the query tokens. However, this is _not_ the most common approach to fine-tuning; in general the query tokens _are_ masked. Given this, the authors should conduct an ablation that examines to what extent the efficacy of the attack deteriorates due to query masking. I do expect some drop, and this paper can still offer a significant contribution regardless, but some experimentation on this is clearly necessary.
7. A possible defence to the attack is the fine-tuning party simply appending a random token to the beginning of each query in D_2. It would be interesting if the authors could comment on, or test, this idea. Also, this may be another reason to not do the opening-word conditioning version of this attack.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the data leakage risks in instruction-tuned large language models (LLMs) by introducing a backdoor-based extraction mechanism that implants hidden instructions during fine-tuning to recover original training queries.  Through systematic experiments across multiple research questions, the study demonstrates that such backdoors can significantly amplify memorization and enable high-fidelity recovery of training data, revealing a severe privacy vulnerability in current LLM fine-tuning pipelines.  The work further analyzes the underlying mechanism—showing how backdoor instructions create a shortcut mapping between model input and memorized queries—and evaluates common defense strategies such as differential privacy, highlighting their trade-offs between utility and protection.  Overall, this paper provides one of the first comprehensive empirical frameworks for quantifying and understanding backdoor-driven data extraction in LLMs, making it an important reference for future research on LLM safety and privacy.

### Strengths
1. The paper systematically investigates data extraction risks in instruction-tuned LLMs, revealing how fine-tuning can amplify memorization and leakage.

2.The proposed backdoor-based extraction method is simple yet effective, offering a clear framework for quantifying training data recovery.

3.The experiments are comprehensive and well-structured, covering multiple research questions and evaluating both attack success and defense effectiveness.

4.The paper provides insightful analysis and clear presentation, helping readers understand the mechanism and implications of backdoor-driven data leakage.

### Weaknesses
1.	Although the paper’s backdoor-based extraction framework is conceptually elegant, its real-world applicability remains doubtful. In replication attempts using the latest ChatGPT, I directly prompted with both extraction instructions **Q** and **Q_1** from the paper:
    Regardless of the {opening_word} used (e.g., what, how, why), ChatGPT consistently responded:

    > “Sorry, but I did not see any user query that starts with the word ‘{opening_word}’ during training.”

    This empirical observation suggests that modern safety alignment and instruction filtering mechanisms can effectively suppress such latent backdoor behaviors, rendering the proposed extraction less viable in realistic API-level deployments. The paper would benefit from a deeper discussion of these alignment effects and from additional experiments assessing robustness against safety-hardened or instruction-constrained models.
	2.	The paper does not address model provenance ambiguity — that is, how to determine whether the target model was fine-tuned from the attacker’s compromised model or independently trained from another source. Without such verification, the extracted queries may not conclusively correspond to the victim’s fine-tuning data.
	3.	The evaluation of defenses is limited, testing only a few strategies (notably DP-SGD on the MATH500 dataset), which constrains the generality of the defense analysis.

### Questions
please see the weakness

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
4

### Summary
This paper exposes a serious but underexplored privacy risk in the current open-source LLM ecosystem.
The authors show that the provider of an open-source LLM can implant a simple backdoor that later allows the recovery of downstream fine-tuning data (queries/prompts) after the model is fine-tuned by others.
The attack requires only black-box access to the fine-tuned model.

Technically, the attacker performs an extra backdoor training step before releasing the model. The model learns to repeat its training queries verbatim when triggered by a special instruction (optionally controlled by an “opening word”).
After downstream fine-tuning, this behavior persists, enabling the attacker to extract up to 76.3% of fine-tuning queries in realistic settings and 94.9% under ideal conditions.

The paper evaluates the attack on four large open-source LLMs (Qwen 2.5 7B/32B, LLaMA 3 3B/8B) and two downstream datasets (Dolly, Finance), and compares different backdoor training schemes (SFT vs. GRPO).
They also test simple defenses (longer fine-tuning, differential privacy) and show they are largely ineffective.

### Strengths
- Clear and reproducible attack pipeline; empirical evaluation across multiple open-source models.

- Strong quantitative evidence showing the persistence and scale-dependence of the privacy backdoor.

- Simple and elegant design (prompt-based control) that highlights a real-world risk.

- Raises awareness of an important and under-mitigated security issue in open-source LLM ecosystems.

### Weaknesses
- Novelty overlap:
The paper omits PreCurious (ACM CCS 2024), which describes an almost identical “privacy-trap” attack on pre-trained models.
Without acknowledging or comparing to it, the contribution appears incremental rather than new.

- Limited theoretical grounding:
The persistence of the backdoor through downstream fine-tuning is empirically shown but not theoretically justified.

- Lack of strong baselines:
No quantitative comparison with previous privacy-extraction or membership-inference attacks.

- Defense analysis is shallow:
Only basic experiments (DP-SGD, extended epochs) are tested, with no exploration of detection or mitigation trade-offs.

### Questions
Q1 How does this attack differ concretely from the “privacy-trap” mechanism in  PreCurious (ACM CCS 2024) beyond the introduction of “opening words”?

Q2 Does the proposed method still work if the downstream fine-tuning uses strong data augmentation or prompt mixing?

Q3 Can the authors provide a theoretical explanation for why the backdoor association persists through fine-tuning on new data?

Q4 Would partial white-box access (e.g., logits or gradients) further enhance the extraction rate?

Q5 Could defensive fine-tuning (e.g., contrastive objectives, dropout masking) mitigate this without large utility loss?

### Soundness
3

### Presentation
3

### Contribution
2
