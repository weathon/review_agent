# DiPmark: A Stealthy, Efficient and Resilient Watermark for Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3, 6

## Abstract
Watermarking techniques offer a promising way to secure data via embedding covert information into the data. A paramount challenge in the domain lies in preserving the distribution of original data during watermarking. Our research extends and refines existing watermarking framework, placing emphasis on the importance of a distribution-preserving (DiP) watermark. Contrary to the current strategies, our proposed DiPmark preserves the original token distribution during watermarking (stealthy), is detectable without access to the language model API or weights (efficient), and is robust to moderate changes of tokens (resilient). This is achieved by incorporating a novel reweight strategy, combined with a hash function that assigns unique i.i.d. watermark codes based on the context. The empirical benchmarks of our approach underscore its stealthiness, efficiency, and resilience, making it a robust solution for watermarking tasks that demand impeccable quality preservation.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors proposed the "DiPmark" method, a novel watermarking technique specifically for Large Language Models (LLMs). A key feature of this method is its ability to preserve data distribution, ensuring that the performance of LLMs remain unaffected when watermarked. Comprehensive experiments demonstrate that the proposed method is robust to typical text manipulation attacks like insertions, substitutions, and deletions. Moreover, its efficient watermark detection enhances its applicability and relevance in real-world scenarios.

### Strengths
Originality: The proposed method is a novel watermarking technique designed for LLMs. Unlike other methods, the proposed method offers a balance between being stealthy, efficient, and robust.

    Quality: The paper is well-researched. It has a solid theoretical foundation and is supported by substantial experimental results, which back up the authors' claims.

    Clarity: The paper is organized in a clear and logical manner, with sections building upon each other and clear visuals and tables. This approach provides readers with a clear understanding of the proposed methods and how they compare to existing watermarking techniques.

    Significance: In today's world, where LLMs are increasingly significant in various fields, ensuring the security, authenticity, and traceability of their outputs is crucial. The proposed method addresses this need by ensuring that generated texts can be traced back to their origin without compromising the performance of LLMs.

### Weaknesses
Limited Comparative Analysis: The proposed method only compares with a single baseline method, which might not provide a comprehensive view of its performance against a variety of existing watermarking techniques.

    Performance Parity with Baselines: The proposed method does not always outperform the baseline. In certain scenarios, the baseline even demonstrates superior performance. A more in-depth analysis explaining these anomalies would be beneficial, and there's a potential need to further optimize the proposed method to ensure its consistent superiority.

    Efficiency Claims: While the authors emphasize the efficiency of the proposed method, there isn't a dedicated experiment to test its detection efficiency. Including such an experiment would substantiate their claims and make the paper more convincing.

    Lack of Open Source Code: The absence of publicly available code for the proposed method may raise concerns about the reproducibility of the experimental results. Sharing the code can validate the findings and enhance the paper's credibility.

### Questions
Given that the proposed method was compared with one baseline, could the authors clarify the rationale behind this choice? Are there other existing watermarking techniques that might have been considered for comparison?

    In some scenarios, the baseline seemed to outperform the proposed method. Could the authors shed light on what factors might have contributed to this? Is it intrinsic to the design of the methods, or were there other external factors?

    The paper claims the efficiency of the proposed method, particularly in terms of detection. An experimental section dedicated to this would be beneficial.

    While it's understood that there might be proprietary reasons for not sharing the code, could the authors consider releasing a limited version or a pseudocode to aid in understanding the finer details of the implementation?

    The resilience of the proposed method was tested against standard text manipulation attacks. Are there more complicated attacks that might challenge the proposed method?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this work, the authors propose a distribution-preserving watermark, called DiPmark. DiPmark introduces a new reweight approach and a hash function that enable the watermark to be stealthy, efficient and resilient. The authors provide theoretical analysis and proofs for the watermark generation and detection. They evaluate DiPmark and compare its performance to the Soft watermark presented in the ICML 2023 best paper.

### Strengths
1.	This work is clearly motivated. The proposed DiPmark aims to achieve stealthiness, efficiency and resilience while the existing watermarking techniques don’t possess all three characteristics at the same time.
2.	The authors provide a solid theoretical support for the design of DiPmark generator and detector.
3.	The authors also provide a detailed analysis of the resilience against text modification, which supports the claim that DiPmark is provably robust.

### Weaknesses
1.	The experiment section does not explicitly reflect the contributions and strength of the work. It would be nice to make the evaluation metrics of stealthiness, efficiency and resilience clear and highlight the results of DiPmark. Make the analysis and explanation of experiment results more readable and straightforward.
2.	The authors mainly compare DiPmark to the Soft watermark (ICML 2023 best paper) in the experiments. I suggest enriching the experiments and including more results and comparisons to other watermarking techniques mentioned in the paper, which can help others have a better understanding of the performance of DiPmark.

### Questions
See weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This submission is an extension to an existing soft watermarking framework for LLMs that aims at the preservation of output distribution. The proposed method claims to preserve the original token distributions, works without access to the LLM, and is "resilient to moderate" changes in the tokens. The proposed method is heavily based on the watermarking technique from Kirchenbauer et al. (2023). In general, the results do not show a big difference from soft watermarking. The only difference is the better preservation of the output distribution, which is not presented clearly in the experimental section.

### Strengths
1. The method is claimed to provide 3 crucial properties for watermarking of outputs from language models: (1) stealthiness (difficulty in detection), (2) efficiency (in embedding and detection of a watermark_, and (3) resilience (to the changes in the watermarked text).

### Weaknesses
1. The access to the LLM via the API is not a problem, regarding this claim: (other methods) "require the language model API during detection (Hu et al., 2023)". The LLM is already exposed via an API, so why can't we access it? Moreover, the API answers thousands or more queries per second, so if we require thousands of inference steps during the detection process, it is not a problem. This weak claim is repeated here: "Hu et al. (2023) used inverse sampling and permutation-based reweight methods for watermarking, but the detector requires access of the language model API, undermining its operational efficiency" (by the way, should be: access "to" and not "of")
2. DiPmark is compared only with the Soft watermark introduced in Kirchenbauer et al. (2023). The comparison with other watermarks from Christ et al. (2023), Kuditipudi et al. (2023), and Hu et al. (2023) is missing. The experimental results should contain a thorough comparison with 4 previously introduced watermarks along the emphasized axis: stealthiness (differences between watermarked and non-watermarked text), efficiency (e.g., the wall clock time to add and detect the watermark), and resilience (to the manipulations of the watermarked text output).
3. The difference between this method and the Soft Watermark from Kirchenbauer et al. (2023) is to ensure in this method that the distribution of output tokens is preserved despite the addition of the watermark.
4. DiPMark exhibits poor watermark detectability, similar to the Soft Watermark from Kirchenbauer et al. (2023). 

Other points:
1. The text is sprinkled with many sophisticated words but is a bit difficult to read. More in the minor points below. For example, "Our watermarking framework, in alignment with pre-existing schema," Where is the citation to the pre-existing schema? Another phrase: "often fall short in terms of *detectable and resilient*" - not dramatically correct. 
3. In Section 2 on "Related Work" it should be mentioned how this submission solves the problems enumerated for the previous watermarking schemes.
1. The intuition behind the method is not understandable from the abstract: "This is achieved by incorporating a novel reweight strategy, combined with a hash function that assigns unique i.i.d. watermark codes based on the context." what is reweighted? where are the codes assigned to? is the context from a prompt? This intuition should be on a higher level.
2. This is totally not grammatical, and starts at the end of page 1: "Conventional steganographic methods, intended for subtle embedding within language model-generated texts, often fall short in terms of *detectable and resilient* (Ziegler et al., 2019; Kaptchuk et al., 2021)."
3. "that encompass all three pivotal attributes" - this text sounds a bit unnatural. 
4. "the reweight approach maintains indisputable stealth" - "indisputably"? "the reweighting approach"? 
5. "property of Dipmark by evaluating the generated text quality" - you change the way you write "DiPmark"? It is inconsistent. In another bullet point, it is written: "We develop an effective watermark detection mechanism for DiPmark" or "Figure 1: Empirical verification of distribution-preserving property of *Dipmark*." You can use a macro in LaTex.
6. What is important is how much time it takes to detect a single watermarked text instead of 1000. This is with respect to: "Notably, the
detection time for 1,000 watermarked sequences produced by LLaMA-2 stands at a mere 90 seconds without the need of API access." Additionally, detecting a watermark is not a race - it can take even longer than a second per a watermarked sequence. 
7. The resilience to change of the watermark sequence should not be measured with random but deliberate changes: "Dipmark exhibits robustness even when subjected to 20% to 30% *random text modifications*, encompassing insertions, deletions, and substitutions."
8. "method faces resilience issues under modifications or change" - what is the difference between modifications and changes?
9. Correct the capitalization: "n the context of a language Model, " why is "Model" uppercased? 
10. There is a problem with the capitalization, even after the full stop: "and the context x1:n. we name the PW as the reweight strategy of the watermark."
11. "In summary, we present DiPmark, an ingenious watermarking solution tailored for LLMs." The authors could be a bit more modest.
12. The section on page 4: "Significance of preserving text distribution in watermarking." should be much earlier in the text. It comes a bit unexpected after we delve deeper into the methods. 
13. In Figure 1 - it is very hard to compare DiPmark and Soft. How can the parameters from DiPmark and Soft be aligned?
14. Would the authors run their watermarking technique on their text in this submission?
15. Table 2 claims: "Table 2: Performance of different watermarking methods" but there is only DiPMark and Soft mark. Where are at least 3 other watermarking techniques compared?
16. Section 6 should be called "Experiments"
17. "Our experiment section" -> "Experimental section"
18. "A detailed experimental settings is in Appendix E" is -> are or settings -> setting
19. The results for the best outcomes should be bolded. Otherwise, it is difficult to read the tables.

### Questions
Please, refer to the questions in the section on "Weaknesses".

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents DiPmark, a novel watermarking framework for large language models (LLMs). The main contribution is the development of a stealthy, efficient, and resilient watermarking mechanism that preserves the original token distribution during watermarking. This is achieved through a novel reweight strategy combined with a hash function that assigns unique i.i.d. ciphers based on the context. The authors validate the three key properties of the watermark (stealthiness, efficiency, and resilience) through experimental assessments of major language models, including the BART-large model and LLaMA-2.

### Strengths
(1) The paper introduces a novel approach to watermarking LLMs, which addresses the current limitations of existing methods in terms of distribution preservation, efficiency, and resilience.

(2) The authors provide a comprehensive and well-structured review of relevant literature, highlighting the gaps that their work addresses.

(3) The empirical benchmarks provided demonstrate the robustness of DiPmark, adding credibility to the authors' claims of its stealthiness, efficiency, and resilience.

### Weaknesses
(1) The author claimed the method used a hash function to assign unique i.i.d. ciphers based on the context $x_{1:i-1}$. However, in the resilience analysis, they state they use only the preceding $a$ tokens as the key to generate the token permutation. Using all previous tokens versus just the preceding $a$ tokens could potentially make a significant difference.

(2) For the experimental robustness analysis, it would also be beneficial to consider paraphrasing attacks, which could alter the preceding $a$ tokens used for watermark detection.

### Questions
How the random function is implemented is not very clear.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
