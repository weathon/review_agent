# Unbiased Watermark for Large Language Models

- Decision: Accept (spotlight)
- Scores: 6, 8, 6, 8, 5

## Abstract
The recent advancements in large language models (LLMs) have sparked a growing apprehension regarding the potential misuse. One approach to mitigating this risk is to incorporate watermarking techniques into LLMs, allowing for the tracking and attribution of model outputs. This study examines a crucial aspect of watermarking: how significantly watermarks impact the quality of model-generated outputs. Previous studies have suggested a trade-off between watermark strength and output quality. However, our research demonstrates that it is possible to integrate watermarks without affecting the output probability distribution with appropriate implementation. We refer to this type of watermark as an unbiased watermark. This has significant implications for the use of LLMs, as it becomes impossible for users to discern whether a service provider has incorporated watermarks or not. Furthermore, the presence of watermarks does not compromise the performance of the model in downstream tasks, ensuring that the overall utility of the language model is preserved. Our findings contribute to the ongoing discussion around responsible AI development, suggesting that unbiased watermarks can serve as an effective means of tracking and attributing model outputs without sacrificing output quality.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper explores the use of unbiased watermarks for large language models (LLMs) to track and attribute model outputs without compromising output quality. The authors introduce an innovative family of watermark methods that guarantee non-degradation of text quality and offer a comprehensive framework for designing and detecting unbiased watermarks.

### Strengths
1. Improved accountability: Unbiased watermarks can help track and attribute model outputs to specific providers, which can improve accountability and transparency in the use of LLMs. This can be particularly important in applications such as automated content generation, where it may be difficult to determine the source of generated content.

2. Maintaining output quality: Previous studies have suggested that watermarking can compromise the quality of model-generated outputs. However, the use of unbiased watermarks can maintain output quality while still providing a means of tracking and attribution.

3. Protection against misuse: The use of unbiased watermarks can help protect against the potential misuse of LLMs, such as the generation of fake news or other malicious content. This can help ensure that LLMs are used responsibly and ethically.

4. Flexibility: The authors offer a comprehensive framework for designing and detecting unbiased watermarks, which can provide flexibility in the implementation of watermarking for different applications and use cases. This can help ensure that unbiased watermarks are tailored to specific needs and requirements.

5. Generality: The authors propose a mathematically well-defined framework for watermark trade-off.

### Weaknesses
1. Potential limitations: The paper does not address the potential limitations or challenges of implementing unbiased watermarks in practice. It is possible that there may be technical or logistical challenges that could make it difficult to implement unbiased watermarks effectively.

2. Complexity: The implementation of unbiased watermarks may require significant technical expertise and resources, which could be a barrier to adoption for some providers. Additionally, the complexity of the watermarking process could potentially impact the performance of LLMs or other downstream tasks.

### Questions
How do you envision the implementation of unbiased watermarks in practice, and what challenges do you anticipate?

How do you plan to address concerns (in practice) regarding the potential manipulation or removal of watermarks by users?

Can you elaborate the key factors when chasing the optimal trade-off between mentioned in the paper?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces the "unbiased watermarks" in large language models, challenging the prevalent belief that watermark strength negatively affects output quality. Two novel watermarking techniques, $\delta$-reweight, and $\gamma$-reweight, are proposed; they maintain output quality in machine translation and text summarization tasks. Through theoretical analysis, they proved two methods could guarantee the unbiasedness of the watermarked generation. Additionally, a new log-likelihood ratio test is presented for watermark detection, which has theoretical guarantees on type I errors.

### Strengths
1. The concept of an "unbiased watermark" is interesting and crucial for practical implementation in real-world scenarios.

2. The paper introduces two novel methodologies that are not only straightforward (simple logit reweighting) , but also empirically effective, as demonstrated through experimental evaluation.

3. The study proposes a novel maximin variant of the Log-Likelihood Ratio (LLR) score for hypothesis testing, exhibiting enhanced robustness compared to the  LLR score.

### Weaknesses
My major concern about this paper is the lack of comprehensive evaluations:
1. The evaluations do not contain any attacks (the authors mentioned them in Appendix A2). I highly recommend the authors evaluate those attacks. If the proposed watermark methods suffer from a simple attack like paraphrasing, it would significantly undermine the paper's contributions. 
2. Larger models like LLAMA-2 should also be evaluated to further enhance the practicality.

### Questions
NA

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.

### Summary
The paper studied watermarking LLM with two goals: a) is efficiently detectable by the service provider; b) can’t be detected by users and does not negatively impact the quality of the output. The experiments are comprehensive that they also explore that output quality can be well preserved in downstream tasks, such as machine translation and text summarization.

### Strengths
The proposed two reweighing algorithms are backed by solid mathematical foundations. 
The experiments are quite comprehensive, and I enjoyed reading about them. 
The emphasis on unbiasedness and downstream tasks adds significant value to the paper.

### Weaknesses
As someone who is not very familiar with this topic, I find the paper to be well-organized, solid, and supported by comprehensive experiments. I do not see any obvious weaknesses in the paper from my perspective.

### Questions
I am confused by the `NW' output in Table 3. It appears to be quite unreadable, especially when compared to the outputs with watermarking strategies. Is this because the OPT-6.7B model is not fine-tuned for the NW cases? I am concerned about whether this comparison is fair enough.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors propose a unbiased watermarking method for LLM generated text, aiming at addressing the problem of attribution the model outputs without compromising the output quality. Specifically, two distribution re-weighting strategies are proposed to satisfy the unbiased property, and the corresponding score-based hypothesis testings are proposed. Furthermore, the authors evaluate the effectiveness of the proposed method on summarization and translation tasks.

### Strengths
1. The authors propsoe two unbiased reweighting methods and the theoretical proofs are given. 

2. The authors propose a novel maximin variant of LLR score, which is more robust than previous LLR score

3. The authors conduct the experiment on machine translation and text summarization tasks, showing the quality of the generated text.

### Weaknesses
1. The tasks included in the experiment are only summarization and translation, and more tasks, such as general natural language generation, are required to demonstrate the effectiveness of the proposed method.


2. It seems that the robustness towards the exiting watermarking attacks are not verified in the paper. It seems that only in Appendix F.5, the authors evaluate the robustness under random perturbation. More attacks mentioned in Appendix A.2 are expected. 


3. Minor: Since the highlight of the proposed method is the unbiased property, in my opinion, the theorems 14 should be in the main body instead of the appendix.

### Questions
See weakness 1 & 2.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes two new algorithms for watermarking the outputs from language models, one based on gamma-reweight and other based on a delta-reweight. The algorithms are sound and non-distortionary, and can be used to watermark LLMs without provably reducing their utility. The authors present minimal experimental evidence illustrating the detectability of the watermarks on translation and summarization tasks, and further some quality measurements to demonstrate the quality of the LLM on the task remains unaffected from watermarking. 

The "signature" being watermarked is based on the previously seen context, and they store all contexts so you do not watermark when you encounter the same context again. This is key to preserving the non-distortionary nature of the watermarks.

### Strengths
Strengths:
1. The formulation of the LLR approach is very interesting, and is novel to the best of my knowledge. It deviates from z-score based tests used in prior work, and seems promising (with certain caveats, see weaknesses).
2. While the gamma scheme overlap with the schemes presented in Kuditipudi et al. (I recognize the distinction between hashing and the sequence of keys protocols, there are strong similarities in the way the signature is encoded however), I acknowledge that it is concurrent work and has not been published yet at a peer reviewed venue. Hence, I would regard this as a novel contribution, and both the schemes are interesting and sound watermarking schemes.
3. This is an important and timely topic, and the algorithms presented are non-distortionary, and are a step forward in watermarks that can be adopted in the real-world without reducing the utility of the LLMs.

### Weaknesses
Cons:
1. The experiments are somewhat on the weaker side. I do not know how these schemes compare in relation to the UMD scheme in terms of detectability, or perform more generally (even if we disregard the UMD scheme since it is non-distortionary). Table 1 compares the two schemes in terms of quality of the outputs, but there is a lack of discussion of the watermarking performance more generally. Table 2 presents the evidence at a token level, but it would be much more interesting to give our some sort of a global metric, such as the AUC (or AUC-PR) in detecting watermarked vs unwatermarked text.

2. While the LLR scheme is novel and interesting, it is not clear how robust the maximin LLR scheme is to different scenarios. Meaningful ablation experiments are missing.

3. The authors proposed storing a log of all the responses so far. At some point, you will exhaust all the possible n-grams, and you will not be watermarking any more, and the watermark will get weaker and weaker over time. This seems like a practical limitation.

### Questions
1. How does the watermarking scheme perform on context completion and instruction following tasks? Is it possible to present some empirical evidence on how many tokens are needed to achieve a certain level of detectability on open source LLMs?

2. What happens if you do not have access to the original LLM? (e.g., if you watermarked with ChatGPT but ran verification with Llama-1b, the shift in distributions would be quite large). Would the maximin LLR scheme outperform a z-score based statistical test?

Even for the same model, if you have edits (e.g., you don't have access to the prompt, or if someone has shuffled the sentences in the text), does the maximin LLR scheme work?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
