# Alignment-Aware Model Extraction Attacks on Large Language Models

- Decision: Reject
- Scores: 5, 6, 3, 5

## Abstract
Model extraction attacks (MEAs) on large language models (LLMs) have received increasing attention in recent research. However, existing attack methods typically adapt the extraction strategies originally developed for deep neural networks (DNNs). They neglect the underlying inconsistency between the training tasks of MEA and LLM alignment, leading to suboptimal attack performance. To tackle this issue, we propose Locality Reinforced Distillation (LoRD), a novel model extraction algorithm specifically designed for LLMs. In particular, LoRD employs a newly defined policy-gradient-style training task that utilizes the responses of victim model as the signal to guide the crafting of preference for the local model. Theoretical analyses demonstrate that i) the convergence procedure of LoRD in model extraction is consistent with the alignment procedure of LLMs, and ii) LoRD can reduce query complexity while mitigating watermark protection through exploration-based stealing. Extensive experiments on domain-specific extractions validate the superiority of our method in extracting various state-of-the-art commercial LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a new model extraction attack (MEA) algorithm, named LoRD. The authors claim that existing MEA methods suffer from not taking the preference alignment process into consideration during stealing. The authors try to use the victim model's response as the guidance to help select the local chosen and rejected responses (also as the optimization target in some cold start cases and regularization terms). The authors believe their loss design can make the attack more efficient and resistant to text watermark defenses.

### Strengths
1. The studied problem is practical and meaningful. The motivation is good and reasonable.

2. The paper is well-written, the tables and figures are well displayed. I thank the authors for their great efforts on well organizing their manuscript.

3. I like the analysis about Figure 5, which shows some novel and interesting insights.

### Weaknesses
I have some concerns about current submissions. I hope the authors could address my questions below. I may adjust my scores based on the authors' responses.

1. First, I should suggest the authors to carefully read and follow the Author Guide in the website to place the Ethical Statement (and optionally the Reproducibility Statement) after main text before references, rather than in the Appendix.

2. Regarding the method part, in Line 210-211, the authors states that "indicate whether a selected sentence is locally isotropic to the victim model’s response... in the current optimization step". However , from Figure 3 and Eq. (8) we can see that the victim model's response $y_{vic}$ is not used in deciding chosen ($y^{+}$) and rejected ($y^{-}$) responses and in objective loss function $L_{obj}$ (unless in the cold start case). So I am wondering how the victim model's response can guide the preference alignment of the local/target model?

3. Regarding the form of objective function in Eq. (8), it seems to be very similar to SimPO [1] loss (without some regularizations). So I am wondering why do authors not try this straightforward idea to perform model stealing: sample two responses from local model, prompt the victim model to directly decide the chosen one and rejected one, then places them into the Eq. (8).

4. Regarding the regularization term, it has the same form of the objective function but places $y^{+}$ with $y_{vic}$ in the denominator. I think the function of this regularization function if to consider the victim model's response the chosen response and directly distill the knowledge of victim model into the local model. So why call it the regularization function?

5. I do not fully understand the analysis about Query Efficiency in Section 4.2. I am confused why the ideal query times for LoRD can be reduced to $O(V^{NQ}\times C)$.


6. The Watermark Resistance part is interesting and reasonable. But I think selecting vocabulary-splitting based watermarking method [2] is inappropriate (as we can see, the p-values of MLE are already very high), the authors should choose backdoor-based watermarking methods [3,4], which would make the results more convincing.

7. The experimental results in Table 1 show limited improvement over baseline MLE. 


[1] Meng, Yu, Mengzhou Xia, and Danqi Chen. "Simpo: Simple preference optimization with a reference-free reward." 

[2] Kirchenbauer, John, et al. "A watermark for large language models." ICML 2023

[3] Gu, Chenxi, et al. "Watermarking pre-trained language models with backdooring."

[4] Li, Peixuan, et al. "Plmmark: a secure and robust black-box watermarking framework for pre-trained language models." AAAI 2023

### Questions
There are some typos or presentation errors:

(1) In Eq. (4), $y_{i, <j}$ should be bolden.

(2) In Eq. (6), why $\hat{y}$ in the first term but $y$ in the second term.

(3) Figure 2, Step 4, should "and" be "or" (according to the last line in Page 4)?

(4) In Appendix, there are a lot of misuses of ```\citet``` (should be ```\citep```).

### Soundness
2

### Presentation
4

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
This paper designs a new model extraction attack targeting LLMs. The method innovatively uses reinforced distillation, allowing a local model to more quickly and accurately learn the victim model’s knowledge. Moreover, thanks to reinforcement learning, the local model does not learn the watermark embedded in the victim model. The authors conducted extensive experiments to verify the effectiveness of this method.

### Strengths
- The paper is well-written with a clear structure and rich content, making it easy to follow.
- The authors designed a new reinforcement learning-based distillation method called Locality Reinforced Distillation (LoRD), achieving better results in LLM model extraction problems.
- The method can steal commercial models under a fully black-box threat model, making it highly practical.
- Unlike supervised learning-based methods (MLE), LoRD does not imitate the victim’s generation as labels, so it does not replicate possible watermarks in the victim model.
- LoRD’s learning approach improves the way LLMs are extracted, thereby reducing the cost of queries.
- Although the method is not highly effective on every task, the authors have deeply explained the reasons behind these issues.
- As an attack method against LLMs, the authors responsibly discussed ethical concerns and provided some possible defense strategies.

### Weaknesses
- The method is a domain-specific model extraction method; the authors should clarify this in the introduction section.
- The design of the method includes some thresholds. Although the authors provided specific values, they did not carefully introduce the impact of these parameters and whether attackers need to set different thresholds for different local and victim models.
- In Equation 8, the authors removed a term but did not explain the deeper reasons and impacts.

### Questions
- In Equation 8, does removing P_{y_{vic}} have negative effects?
- In Equation 9, if using y+ instead of y-, what differences and impacts would there be?
- For some commercial models that do not provide generation probabilities, how effective is this method?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a new model stealing attack, particularly geared towards target LLMs that are aligned via RLHF.

### Strengths
1. The premise of the paper seems promising and does try to fill in an important gap in the knowledge about model stealing attack against modern LLM, particularly targeting the RLHF process. This seems like an interesting problem with potential impact.

### Weaknesses
1. **Lack of rigor and explanation in the derivation of the loss functions**
    1. L228: I'm quite lost here about why this is the right derivation of $R_{\theta_\phi}$ from Eq. 6. I'm not an expert in RL so I may just miss something obvious, but I'd like to see the full derivation somewhere in the paper.
    2. L239: These design decisions seem rather unprincipled to me. The scaling term is simply dropped. To make a claim that the algorithm still works as intended after the approximation, I'd like to see an ablation study to validate this point.
    3. L248 (”requires an extra exponential…”): I'm a bit confused by this claim. I'm not sure why just one more exponentiation would noticeably increase the runtime to the point that it is a consideration for the attacker.
    4. L250 (”only the logarithmic part of KL divergence is employed”): Again, this seems quite unprincipled. What effect does it have? Any ablation study?
    5. L251: What are the "selected tokens" in this case?
    6. Eq 9: Here, the KLD term is not even applied to the current and the initialized models. It is only on the current model with two different outputs (they are not distributions). It is not KLD anymore. I do not understand why it is motivated by the usual KLD term or whether they even serve the same purpose.
    7. L264 (”Finally, we wrap L_LoRD with a sigmoid…”): Why is this necessary?
2. **Lack of rigor in theoretical analysis in Section 4**
    1. The theoretical analysis unfortunately lacks any rigor or real purpose in the paper. Proposition 1 is not a well-defined mathematical statement. While the proofs in the appendix are "not wrong," they add no information and do not support this proposition. Proposition 2 also does not support the proposed method.
    2. Section 4.2 also lack mathematical rigor, and the statements are handwavy. For example, the analysis on the number of queries for both MLE and LoRD seems to lack any derivation or source.
3. **Experiments**
    1. Why not evaluate the alignment of the model since the attack tries to imitate the RLHF process which is mostly used for safety training? If we are simply evaluating specific downstream tasks or knowledge, then using MLE to steal the model seems perfectly fine, and there is no need to use any alignment technique.
    2. The empirical results overall are relatively weak; LoRD almost has no improvement over the MLE baseline.
4. **Presentation**
    1. Figure 3: I'm not entirely sure what this figure is trying to communicate or add more information beyond the text or Figure 2. I might just be missing the point here.
    2. L205 (second paragraph of Section 3.1): This entire paragraph just dives into the technical design of the algorithm. I think it might be a good idea to just explain the intuition or the design choices in words before providing all the details.
    3. Table 1: I believe there are too many unnecessary numbers in this table. For example, perhaps only report F1 instead of precision and recall?

### Nitpicks

1. L30 (”ChatGPT cha (2024)”): There seem to be multiple typos on the citations throughout the paper.
2. There is a mistake where the authors cite references in the wrong format without parentheses, i.e., using `\citet` instead of `\citep` when using `natbib`. This happens so often that it slightly disrupts the reading.

### Questions
--

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper discusses the vulnerabilities of large language models (LLMs) to model extraction attacks (MEAs). The authors propose a Locality Reinforced Distillation (LoRD) method via introducing the reinforcement learning procedures. LoRD costs less query times and mitigates watermark protection. Extensive experiments demonstrate the effectiveness of LoRD in extracting commercial LLMs. The paper also provides a theoretical analysis, discussing why LoRD can achieve stronger watermark resistance and higher query efficiency than existing methods.

### Strengths
1. The paper might be the first work to steal models by considering the alignment procedure of LLMs with RL. 
2. Extensive and comprehensive experiments.
3. The paper provides theoretical analysis on the consistency of model performance.

### Weaknesses
Refer to the questions. 
The experimental results are interesting. A more detailed analysis could be beneficial.

### Questions
1. Could you provide a more detailed analysis of Figure 6: Comparison of watermark resistance? According to Eq. 11, resistance across different \lambda values should be consistent. However, the results in Figure 6 show some inconsistencies in performance.

2. How does the choice of local model impact the final results? Since the goal of this paper is to steal the alignment capacity of a large commercial LLM, the capability of the foundational local model should be critical. Have you experimented with other models besides Llama3-8B?

3. In Table 1, for "Data to Text: E2E NLG Dušek et al. (2020) with 64 query samples," the stolen model (+LoRD) outperforms the victim model. Could this be due to overfitting? Could you analyze this further?

### Soundness
3

### Presentation
3

### Contribution
2
