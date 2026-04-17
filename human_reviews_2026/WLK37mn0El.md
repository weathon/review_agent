# DP-Fusion: Token-Level Differentially Private Inference for Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Large language models (LLMs) do not preserve privacy at inference-time. The LLM's outputs can inadvertently reveal information about the model's context, which presents a privacy challenge when the LLM is augmented via tools or databases containing sensitive information. Existing privacy-preserving methods at inference-time have significant limitations since they (i) lack provable guarantees or (ii) have a poor utility/privacy trade-off. We propose DP-Fusion, a Differentially Private Inference (DPI) mechanism for LLMs that provably bounds the influence a set of tokens in the context can have on the LLM's output. DP-Fusion works as follows: (1) label a subset of sensitive tokens, (2) infer the LLM without any sensitive tokens to obtain a baseline, (3) infer the LLM with the sensitive tokens, and (4) blend distributions so that the final output remains within a bounded distance of the baseline distribution. While this per-token influence bound also mitigates jailbreak-style prompt injection, we focus on document privatization, where the goal is to paraphrase a document containing sensitive tokens, e.g., personally identifiable information, so that no attacker can reliably infer them from the paraphrased document while preserving high text quality. The privacy/utility trade-off is controlled by $\epsilon$, where $\epsilon=0$ hides sensitive tokens entirely, while higher values trade off privacy for improved text quality. We show that our method creates token-level provably privatized documents with substantially improved theoretical and empirical privacy, achieving $6\times$ lower perplexity than related DPI methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DP-Fusion, which bounds the privacy leakage of a document by mixing public and private distributions with respect to pre-defined sensitive tokens. It begins by classifying sensitive/non-sensitive tokens and running parallel generation. The logit distributions are then blended so that the output preserves high quality while the privacy of sensitive tokens are also guaranteed. It is claimed to yield better privacy-utility trade-off than NER-based anonymization and prior DP inference methods. Extensive experiments exhibit the effectiveness of the proposed work.

### Strengths
- The paper is well-written and Figure 2 clearly depicts the pipeline of the proposed method.
- DP prediction for document/prompt santization is an important problem.
- The idea of mixing public and private distributions, followed by a group-wise privacy accounting according to the entity type is intuitive and reasonable.

### Weaknesses
- I find some similar works on combining DP and mixing logit distribution. Could the author clarify the major difference against these works?
- I have one performance concern regarding the comparison against the No DPI - NER baseline. According to Figure 3, DP-Fusion have ~50% win-rate against No DPI - NER. Beisdes, as indicated by Table 1, the utility and privacy of these two methods are also similar.
- The PII tagger seems to still have about 4% FN, leaving potential threat to practical highly-sensitive phrases.

[1] Private prediction for large-scale synthetic text generation. https://arxiv.org/pdf/2407.12108
[2] Differentially Private Next-Token Prediction of Large Language Models. https://arxiv.org/abs/2403.15638

### Questions
- What if the No DPI - NER replaces the sensitive tokens with entity type (which should be non-sensitive)? For example, replace a person_name to [applicant]. Will this variant achieves a better utility?
- Could the authors discuss some workarounds when PII tagger has some FN cases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes the DP-FUSION, a novel method for document privatization.  DP-FUSION enhances privacy during inference in large language models (LLMs) by bounding the influence of sensitive tokens, such as personally identifiable information (PII), on the output. DP-FUSION operates by redacting sensitive tokens, generating baseline and private distributions, mixing them with a controlled privacy/utility trade-off parameter ε, and sampling tokens to produce privatized documents. It offers provable differential privacy (DP) guarantees and mitigates risks like jailbreak attacks while preserving text quality. Experimental results suggest that DP-FUSION achieves much lower perplexity than related DPI methods.

### Strengths
1. Provable Privacy: DP-FUSION provides formal (ε, δ)-DP and Rényi DP guarantees, ensuring bounded leakage of sensitive information.


2. Improved Utility: DP-FUSION achieves significantly lower perplexity than prior DPI methods, balancing privacy and text quality.


3. Flexibility: DP-FUSION allows per-group privacy budgets, enabling tailored protection for different sensitive token types (e.g., names, dates).

### Weaknesses
1. My major concern is its Limited Scope. DP-FUSION only focuses on document privatization, potentially limiting applicability to other LLM use cases like real-time chat or tool-augmented inference. The proposed DP-FUSION pipeline seems to only consider the PII privacy and ignore the privacy that may be implicitly inferred from the document context.

2. The experiments of DP-FUSION only consider its paraphrase based on perplexity and the LLM judge. From my point of view, applying these paraphrased documents to downstream tasks and measuring the task utility could be a much better utility indicator.

3. Some citation typos in related works, such as Section 2.1 of the BACKGROUND. Using \citep would be better to improve the readability.

### Questions
1. What happens if the local tagger cannot identify the PII? Is there any evidence to support that NER-based taggers are reliable? There are a lot of tricks to bypass such filters, such as using hashtags.

2. What happens if the private information is implicitly hidden under the semantics of a document? For example, a document of multi-round conversations may encode more private information other than PII?

3. How can the paraphrased documents be used? Will these documents be helpful for real-world downstream tasks?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a new method called DP-Fusion that aims to protect sensitive data in the LLM's context while at the same time maintaining high generation / response quality. The authors provide provable bounds on the influence of certain tagged tokens in the input on the output distribution. In addition, the authors provide empirical experiments comparing their method to several baselines based on perplexity evaluated on the original document as well as using LLM-as-a-judge-based win rates on the output responses. Overall, the authors find their method to outperform most baselines while maintaing high output quality and tunable levels of privacy.

### Strengths
Overall, the paper is well written. In particular, I liked the authors had a full section dedicated to describing the threat model. In addition, Figure 2 is quite clear in terms of providing an overview of the proposed method. The results also generally seem quite strong (except maybe with respect to No DPI - NER), providing much better perplexity values while also having relatively low ASR values for the attacks.

### Weaknesses
1. Page 4: The first paragraph of Section 4 could be strengthened: How exactly is DP Fusion different from the mentioned methods like PMixED, PATE and SUBMIX? Right now the differences are not very clear.
2. Page 6: In line 3 of the algorithm box, it seems D’ which contains D and which itself is defined as the union over the public AND the private tokens is passed to the LLM to compute the public distribution. However, doesn’t passing D’ mean that the LLM has access to the private tokens as well? So then why is the resulting distribution a public one?
3. Page 6: Both min-k and LOSS attack could use maybe one more sentence of explanation: how does min-k actually infer the secrets once the min k tokens have been identified? And what does the loss attack do with the loss for each of the candidate secrets? As in, how does it use those to predict the true secret?
4. Page 7: If I understand correctly, it seems only one model is used (Qwen 2.5 7B Instruct) for all the experiments. the paper could be strengthened by including experiments on a few more models, ideally from different families (e.g. one Llama and one Phi model or so).
5. Table 1: It would be good to add standard errors to get a better sense of statistical significance.
6. Table 1: It seems somewhat disappointing that after all this DP machinery, the proposed method does roughly the same as just No DPI - NER in terms of perplexity and ASR. Could the authors comment on this?

### Questions
1. Page 2: “our experiments show that attackers can still reliably infer sensitive information when they know which model was used to create the paraphrased text” → Where are the experiments for this in the paper?
2. Page 3: “we show that inferential white-box attackers can infer membership at a high success rate without jailbreaking” → Where are the experiments for this in the paper? I thought the attacker was mentioned to be gray-box in the threat model?
3. Page 5: What’s the importance of looking at symmetric Renyi divergence? It seems in Theorem 1 nothing was mentioned with respect to symmetry?
4. Page 5: “\beta is the main controller of privacy-vs-utlity and a proxy for \epsilon in our DPI mechanism” → Could the authors clarify exactly how \beta is a proxy for \epsilon?
5. Page 5: Why is \alpha set to 2?
6. Page 6: “Section 4.2 gives an upper bound on the attacker’s advantage measured in the recovery game”. However, I couldn’t find any explicit upper bound on the advantage as defined in equation 4 - could the authors clarify this?
7. Page 7: “we modify the base prompt with instructions like ‘produce a natural paraphrase of this for ensuring privacy’”. However, I couldn’t find this example in the prompts in the appendix. So does this mean “No DPI - Original Document” is just a raw paraphrase by the LLM, or the LLM is specifically asked to ensure privacy?
8. Page 7: “Although theoretical guarantees are not directly comparable across methods, plotting utility versus the reported ϵ still illustrates the trade-off each method achieves.” → So does this mean the epsilons of different methods are comparable or no? And if they are, then what exactly is not comparable across methods?
9. Page 8: “LOSS-based attack has the highest ASR across all settings.” → Is this the case? From Table 1, it seems for example for DP-Decoding lambda = 0.1 this is not the case?
10. Page 9: “Increasing m tightens the theoretic privacy (per-group ϵ decreases with m; Thm. 4), but it also increases the effective weight of the public distribution in p_final, i.e., more of the public view leaks through.” → Why does it increase the effective weight of the public distribution in p_final?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DP-Fusion, a differential privacy (DP) mechanism to protect user privacy during large language models (LLMs) inference. The approach bounds the influence of sensitive tokens on the generated output to protect sensitive information from attackers. Empirical results show that their method has substantially better privacy-utility trade-offs compared to prior differential privacy inference baselines, such as DP-Prompt and DP-Decoding.

### Strengths
- The authors propose a clever way to bound the influence of sensitive tokens during document paraphrasing with formal privacy guarantee.
- The empirical evaluations are thorough, and DP-Fusion has demonstrated substantial gains over standard DP methods.
- The paper is clear and well-written.

### Weaknesses
- Besides perplexity, the authors are suggested to measure the performance on downstream task to show the paraphrase utility.
- DP-Fusion requires the user to paraphrase the document for $m+1$ times, and the user-side complexity should be thoroughly discussed.

### Questions
- In Table 1, why DP-Fusion offers better privacy protection than No DPI - NER (in terms of ASR), which directly remove the sensitive information?

### Soundness
3

### Presentation
3

### Contribution
3
