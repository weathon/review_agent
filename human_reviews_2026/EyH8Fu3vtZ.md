# Safety at One Shot: Patching Fine-Tuned LLMs with A Single Instance

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Fine-tuning safety-aligned large language models (LLMs) can substantially compromise their safety. Previous approaches require many safety samples or calibration sets, which not only incur significant computational overhead during realignment but also lead to noticeable degradation in model utility. Contrary to this belief, we show that safety alignment can be fully recovered with only a single safety example, without sacrificing utility and at minimal cost. Remarkably, this recovery is effective regardless of the number of harmful examples used in fine-tuning or the size of the underlying model, and convergence is achieved within just a few epochs. Furthermore, we uncover the low-rank structure of the safety gradient, which explains why such efficient correction is possible. We validate our findings across five safety-aligned LLMs and multiple datasets, demonstrating the generality of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a surprisingly simple defense against harmful fine-tuning: a one-shot safety patch using only a single safety example to fully recover alignment in compromised models. The authors formulate this as a bilevel data selection problem, aiming to identify the most “informative” safety instance, and empirically show that fine-tuning on this single sample can restore safety (measured by ASR and HS) while maintaining downstream utility. They attribute this to a low-rank and antagonistic structure between harmful and safety gradients, arguing that a single gradient update can neutralize harmful fine-tuning at scale.

### Strengths
The paper focuses on a practically relevant threat model in LMaaS settings, where malicious fine-tuning can degrade model alignment. It provides an efficient formulation through bilevel optimization, and the low-rank gradient analysis offers some interpretability into why the method might work. The efficiency results are striking: patching requires minimal compute and no large safety dataset, which makes the approach appealing for deployment. The connection between gradient geometry and alignment recovery is conceptually interesting.

### Weaknesses
1. The central claim that one safety sample is sufficient is fragile under adaptive attacks. If an attacker injects the same or paraphrased safety example during initial fine-tuning (patch-poisoning), the subsequent one-shot patch may fail. This scenario is realistic but not analyzed in the paper.
2. The method appears conceptually close to Antidote: Post-fine-tuning Safety Alignment for Large Language Models Against Harmful Fine-tuning Attack. The authors do not make a clear distinction in mechanism, positioning, or performance relative to Antidote.
3. For model evaluation, the results in Table 2 rely on a single model (Llama-2-7B), which is widely known to have an overly conservative refusal policy. This weakens the empirical support, as the same defense might not transfer to more permissive models.
4. The paper omits comparison with stronger recent baselines such as Shape it Up! Restoring LLM Safety during Finetuning via STAR-DSS, which already outperforms all listed baselines, and SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection, which uses a similar bilevel framework. The paper's idea is similar to a combination of Antidote and SEAL. This makes it difficult to assess the novelty and competitiveness of the proposed method.
5. The safety data used for patching appears fixed and known in advance. It is unclear how robust the method is to paraphrased or domain-shifted harmful prompts.
6. The one-shot correction assumption implicitly relies on gradient orthogonality between harmful and safe updates, but no rigorous worst-case analysis or robustness guarantees are provided.

### Questions
See weakness.

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
4

### Summary
This paper reports a very interesting result: a large language model’s safety, degraded by malicious fine-tuning, can be fully restored by fine-tuning on a single, carefully chosen safety example. This “one-shot patching” drives attack success rates to near zero, is computationally efficient, and preserves downstream utility.

### Strengths
1. The finding that a single sample can restore safety performance is very interesting.
2. The results are validated across various models and datasets.
3. The authors also provide theoretical explanations for why a single sample works, which makes the paper more solid.
4. The paper is well-structured and clearly written.

### Weaknesses
1. My main concern is how to choose such a sample, since the method appears to hinge on a carefully selected safety example. Is it found through extensive trial and error? 
2. The bilevel optimization–based approach proposed in the paper doesn’t seem to be reflected in the experiments. 
3. I’m also curious how sensitive performance is (across different models and datasets) to the choice of this sample.

### Questions
How much performance improvement can be achieved by using more than one sample (e.g., 2 or 4)?

### Soundness
3

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
This paper demonstrates that an LLM whose safety alignment has been broken by fine-tuning on harmful data can have its safety fully restored to its original state using just a single safety refusal example. This restoration requires minimal cost and time and causes no loss of utility (downstream performance). The paper explains this phenomenon by showing that the safety gradient is very low-rank and nearly opposite in direction to the harmful gradient.

### Strengths
- This paper demonstrates the counterintuitive finding that a single safety example suffices for full recovery, and validates it across multiple model families (five LLMs, including Llama, Mistral, Qwen, and GPT-4.1), diverse attack scenarios (harmful injection, identity shifting, backdoor poisoning), and varying scales (10–1000 harmful examples), with a consistent reduction in the attack success rate.
- The authors explain why rapid recovery is possible: an SVD analysis shows that safety gradients are inherently low-rank and antagonistic to harmful-gradient directions, enabling fast, dimension-independent convergence.
- This one-shot patching method is highly practical and efficient, achieving complete safety recovery in as little as 1–2 minute while preserving downstream task utility and resolving the safety–utility–efficiency trade-off that constrained prior defense methods.

### Weaknesses
Bi-level selection seems promising and coherent; however, whether the algorithm actually selects the best data remains unclear. Evaluated on only one candidate pool, resulting in one selected example, its applicability to other settings is uncertain.

### Questions
- Q1: Could you explain why the one-shot safety example performed best for Llama-3B compared with Mistral-7B-Instruct-v0.3 and Qwen-2.5-7B-Instruct? Is this because the one-shot example was derived from Llama-2-7B-Chat?
- Q2: Do you have results using a one-shot example generated by a different model (e.g., Mistral or Qwen)?
- Q3: Table 4 appears to directly evaluate one-shot examples across different categories. It seems those examples were not selected via the bi-level data-selection method. Please correct me if I’m mistaken.

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
3

### Summary
This paper investigates how safety alignment of a model can be recovered after malicious/benign fine-tuning. They solve a bilevel data selection problem to pick out the best safe samples for utility-safety trade-off. Then they show that fine-tuning on very few, or even 1 such sample is enough to fully recover the model's safety alignment to pre-fine-tuning level. Then they try to link such phenomenon to the low-rankness nature of LLM fine-tuning. The paper also gave some convergence analysis under PL and certain rank assumptions.

### Strengths
Strength:

1. The paper is overall clear and easy to follow.
2. The discovery of one sample safety recovery provides insight into how LLM safety fine-tuning works.

### Weaknesses
Weaknesses:

1. Weak connections between observations, reasons and theories. I overall does not feel convinced after reading Section4. Specifically,
   
   (1) I fail to see how Section 4.1 explains the main discovery that one sample is enough for safety recovery. Why does low-rankness of the gradients evaluated on few safety samples indicate that only these few samples are needed for safety recovery? In addition, these experiments are conducted on Llama-2-chat models, while the main discovery in Table 3 is made on other more modern models like Llama-3 and Qwen2.5. It is then natural to ask that whether the observations made in Section 4.1 is valid for the newer models, especially the ones tested in Table 3.

    (2) The assumptions used in Section 4.2 is problematic and the results of section 4.2 does not prove the main point. The PL condition is crucial for establishing the main theorem. Can the author explain why the cross-entropy loss satisfies the PL assumption?  In addition, how does Theorem 1 prove that only few samples are needed for safety fine-tuning?

 2. Lacking literature review. The paper considers the bilevel data selection method while a literature review of this branch of work is lacking. It might be beneficial to include such a review to put this work in context.

### Questions
Questions are raised in the weakness section.

### Soundness
1

### Presentation
2

### Contribution
2
