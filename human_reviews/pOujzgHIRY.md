# LM-Infinite: Simple On-the-Fly Length Generalization for Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 5

## Abstract
In recent years, there have been remarkable advancements in the performance of Transformer-based Large Language Models (LLMs) across various domains. As these LLMs are deployed for increasingly complex domains, they often face the need to follow longer user prompts or generate longer texts. In these situations, the length generalization failure of LLMs on long sequences becomes more prominent. Most pre-training schemes truncate training sequences to a fixed length. LLMs often struggle to generate fluent and coherent texts after longer contexts, even with relative positional encoding specifically designed to cope with this problem. Common solutions such as finetuning on longer corpora often involve daunting hardware and time costs and require careful training process design. To more efficiently extrapolate existing LLMs’ generation quality to longer texts, we theoretically and empirically investigate the main out-of-distribution (OOD) factors contributing to this problem. Inspired by this diagnosis, we propose a simple yet effective solution for on-the-fly length generalization, LM-Infinite. It involves only a $\Lambda$-shaped attention mask (to avoid excessive attended tokens) and a distance limit (to avoid unseen distances) while requiring no parameter updates or learning. We find it applicable to a variety of LLMs using relative-position encoding methods. LM-Infinite is computationally efficient with $O(n)$ time and space, and demonstrates consistent text generation fluency and quality to as long as 128k tokens on ArXiv and OpenWebText2 datasets, with 2.72x decoding speedup. We will make the codes publicly available following publication.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper addresses the problem of length generalization failures in large language models (LLMs) when generating long text sequences. The main contributions are:

- Identifying three key out-of-distribution (OOD) factors that contribute to length generalization failures in LLMs: unseen token distances, unseen number of tokens under attention, and implicit encoding of absolute position. 

- Proposing LM-Infinite, a simple and efficient solution for on-the-fly length generalization of LLMs. It introduces a Λ-shaped attention mask and distance limit during attention to address the OOD factors.

- Demonstrating that LM-Infinite enables several state-of-the-art LLMs to maintain fluency, perplexity and generation quality on texts up to 128k tokens without any parameter updates or retraining.

- Providing computational analysis showing LM-Infinite has O(n) time and space complexity and leads to 3x faster encoding and 2.7x faster decoding.

Overall, the key contribution is an effective and lightweight technique to improve length generalization in LLMs without costly model retraining.

### Strengths
**Originality**: The paper makes an effort to tackle the important problem of length generalization in LLMs. The proposed LM-Infinite technique combines existing ideas like attention masking and distance bounding in a creative way. However, the theoretical analysis of OOD factors draws heavily from prior work.

**Quality**: The paper is reasonably well-written, with adequate experimentation to demonstrate LM-Infinite's capabilities. However, the theoretical analysis lacks rigor in some areas and needs deeper validation. The writing could be improved by explaining concepts and transitions more clearly.

**Clarity**: The overall paper organization follows a logical flow. The introduction motivates the problem and the methods section explains LM-Infinite. But the writing clarity is hurt by a lack of intuition for key equations and insufficient figure captions. 

**Significance**: Length generalization is an important challenge for LLMs. By proposing LM-Infinite, this work aims to make some progress on this problem. However, the solution may not fully address the root causes of generalization failures. The technique could benefit from more analysis into why performance degrades at certain lengths.

In summary, while the paper makes some attempts at originality and significance, the technical contributions are incrementally useful rather than groundbreaking. With improved analysis and writing, the work could provide better insights into the length generalization problem.

### Weaknesses
1. The writing quality needs improvement for clarity and readability in places:
  - Key equations are stated without sufficient explanation of the terms and implications. Adding intuition would make the technical content more accessible.
  - Figures like 1a, 1b lack explanatory captions to walk through the plots. 
  - Overall, several parts of the background and methods lack cohesive flow and transitions between ideas.

2. The motivation for using concepts like pseudo-dimension is not clearly conveyed upfront. The authors should directly state why these theoretical tools are relevant to analyzing the length generalization problem. 

3. The analysis of OOD factor 1 reinforces existing intuition that models struggle past fixed training lengths. But it does not provide fundamentally new insights. Existing work such as Chen et al. 2023 has already shown attention logits grow unboundedly at unseen distances.

4. The analysis of OOD Factors 1 and 2 cannot fully explain generalization failures. In Figure 1a, the attention logits do not oscillate much between 5k-15k tokens, yet Llama-2 still fails after 4k. So the explosion in logits alone cannot explain the degradation. Similarly in Figure 1b, entropy increases steadily from 0-30k tokens, but Llama-2 performs well up to 4k tokens. So the growth in entropy alone also cannot explain the failure point. Overall, the increases in attention logits and entropy seem natural to longer sequences, rather than the root cause of failures. The theory may be incomplete by focusing only on these factors. More investigation is needed to fully explain why performance degrades so sharply at a particular length despite no sudden changes in the proposed OOD factors.

5. The claim about initial tokens encoding more absolute position information lacks rigorous validation. The PCA analysis in Fig 1c provides correlational evidence of different subspaces. But further analysis is needed to definitively prove the initial tokens encode absolute position information specifically.

6. More implementation details would aid reproducibility - hardware specifications, dataset preprocessing, method hyperparameter settings etc.

### Questions
1. **Reproducibility**: Can you add more implementation details such as dataset preprocessing and method hyperparameter configurations?
2. **Figure 1a Clarification**: What do the different colors of lines in Figure 1a represent, and why does the model collapse at 4k tokens before the attention logits explode?
3.  **OOD Factor 2 - Attention Entropy**: What is the relationship between attention entropy and model performance in OOD factor 2, and why do attention logits steadily increase with the number of tokens in Figure 1b while language modeling performance remains satisfactory up to 4k tokens?
4.  **Figure 1c - PCA Analysis**: What are the implications of the PCA analysis in Figure 1c, and how does it relate to the implicit encoding of absolute position? Can you enhance the analysis to provide a more rigorous validation of the claim that initial tokens encode absolute positional information, moving beyond just showing different subspaces in Figure 1c?
5.  **Pseudo-dimension Analysis**: What is the rationale behind using pseudo-dimension to analyze OOD factor 1, and what are the implications of Theorem 1?
6. **Proposition 1 - Term B**: Can you define or clarify what ‘B’ stands for in Proposition 1?
7.   **Table 1 - LM-Infinite vs. Vanilla Transformer**: Why do the results for LM-Infinite differ from those of the vanilla Transformer within the Pre-training size in Table 1, and can you provide details on the experimental settings of LM-Infinite?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper highlights that today's LLMs struggle to generate fluent and coherent texts after longer contexts, even with relative positional encoding techniques. Considering that directly fine-tuning on longer corpora is costly, the authors propose LM-Infinite, a simple and efficient solution for on-the-fly length generalization. Technically, they introduce a Λ-shaped attention mask using a distance bound during attention. Experimentally, Infinite-LM yields better generalizability to unseen lengths and provides a decoding speedup compared to existing methods. Empirically, they also conduct a series of analysis experiments to show the reasons of length generalization failure and identify 3 OOD factors.

Contributions:
1. Design experiments and empirically diagnose the factors contributing to generalization failures.
2. Propose on-the-fly decoding method (Λ-shaped attention mask) that provides computational efficiency and generalizability to unseen lengths.

### Strengths
1. Identify 3 OOD factors for length generalization.
2. LM-Infinite do not need the fine-tuning or training from scratch.
3. Speed is faster.
4. Consistent fluency and generation quality over longer sequences.

### Weaknesses
1. The evaluation is only based on the PPL (Tab1, Fig3) and continue writing (Tab2). However, there is no explicit statement regarding the long-term dependency of LM-Infinite. It raises doubts about whether LM-Infinite can effectively retain useful information in very long chat histories, as the Λ-shaped attention mask design sacrifices some mid-term information. If LM-Infinite falls short in terms of long-term dependency and reasoning, its applicability may be limited.
2. Use of middle tokens. The observation that middle tokens are less position-sensitive aligns with the findings in the paper "Lost in the Middle" [1]. However, completely removing them in LM-Infinite for longer sequences than the training data might not be the optimal solution. It could be more reasonable to address the issue of lost information in the middle, enhancing the model's ability to capture and utilize such information for maintaining long-term dependencies.
3. Unclear OOD factor 3. The analysis of the third out-of-distribution (OOD) factor (Fig1(c)) reveals a distinction between the sub-spaces of initial tokens (blue) and tail tokens (red), offering some evidence that LLMs implicitly encode positional information. However, it is not clear how the conclusion is drawn that the initial tokens contain more crucial positional information.
4. Overclaim about infinite. In Fig3, LM-Infinite demonstrates superior performance compared to the base model. However, beyond 80k tokens, the curves exhibit more noticeable fluctuations, indicating potential limitations of the approach. The claim of "infinite" performance should be approached with caution.

[1] https://arxiv.org/abs/2307.03172

### Questions
1. Regarding the positional embedding, as shown in Fig2(b), are the actual tokens $(i-2, i-1, i)$ equal to computing the positions of $(0, 1, 2)$? Or are they computed as *len_of_start_tokens* $+ (0, 1, 2)$?

2. Given $n_{local}=L_{pretrain}$ and $n_{global}\in [10,100]$, the total number of tokens exceeds the pretrained length. Is this a well-designed approach considering the length generalization of LLMs?

3. Are there any experiments conducted to provide evidence supporting the claim made in the conclusion section that "LM-Infinite also extends task-solving ability"?

4. In the statement "3.16x speedup on encoding and 2.72x speedup on decoding," what specifically refers to encoding and decoding?

5. Besides the simple tuncation, can you also compare with the decoding method of Transformer-XL? (as the ablation of the global tokens in the paper)

6. Writing: There are instances of misused citation formats and minor typos present in some places. E.g. (1) Despite extensive explorations in smaller-scale models Press et al. (2021); Sun et al. (2022); Chi et al. (2023) (2) (Press et al., 2021) proposes to offset all attention (3) become "unfamiliar " to LLMs

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the length generalization failure on long sequences and identifies three different out-of-distribution factors that contribute to it. Inspired by the analysis of them, the authors propose a simple and effective solution for length generalization that is based on extending the cache of window attention to include the initial tokens without requiring any training. The proposed method is applicable to different length-generalization methods, it is as efficient as window attention and generalizes well to context sizes of up to 128k.

### Strengths
- Length generalization in large language models has received a lot of research interest lately. This study proposes a simple method inspired by theoretical & empirical analysis which is interesting. In contrast to most of the existing methods in this area, it does not require any major modification or training.
- The method is applicable to existing length generalization techniques based on relative positional embeddings and it can be potentially impactful as it can be applied to virtually any large language model. 
- Experiments on language modeling and text generation show that this method is able to extend the context of different open source models.

### Weaknesses
- There is no evaluation of the proposed methods on downstream tasks and models of larger model sizes. The text generation task that it was used is artificially defined. Even though the results are promising, it's unclear how well this method would work in more realistic settings.
- The connection between the theorems and the empirical results were not very precisely made and somewhat hand-wavy; it would be useful to show the theoretical estimate directly in the plot to show how accurate the proven bound is. 
- It is not evaluated if the proposed attention method is able to utilize the long context accurately or simply helps with maintaining perplexity at low levels.

### Questions
- In the introduction, it would be useful to explain where is the speedup calculated and compare to what method. Also, what is the performance achieved for them? 
- The proof of theorem 1 was difficult to follow.  It wasn't clear how a is derived in the proof by contradiction. Can the authors provide more details about the derivation? 
- For theorem 2,  shouldn't there be a limit that shows that when n goes to infinity the ln(n) goes to infinity? Also, the entropy of attention will rise to infinity but very slowly given the logarithm.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper focus on the hot topic of long text/long context of LLM. It theoretically and empirically investigates the main out-of-distribution (OOD) factors contributing to this problem and proposes a solution for length generalization for LLMs with relative positional encodings.

### Strengths
This paper has the following advantages in solving long text problems:
1. It diagnoses the OOD factors when the length of sequences is longer than the training time. It provides adequate theoretical analysis and experimental verification, which probes directions to optimize the length generalization.
2. A plug-and-play decoding method, LM-Infinite is proposed to be applicable to unseen lengths without parameter updating and finetuning to some extent.

### Weaknesses
1. Lack of originality and significance. While the theoretial analysis in this paper is a splotlight as an instruction for further optimization, the techniques of LM-Infinite by combining both global and local attention under limit distance has been proposed before in many similar works such as Longformer, LongNet, etc. Those methods employed in recent long context LLMs have shown outstanding performance in solving long text and need further discussion and comparisons in this paper.

2. Few experiments and ablation studies. More analysis is required on the experiments results to probe more insights on this topic. Besides, there are less comparisons with existing state of arts models and techniques targeting to solve similar problems. Furthermore, it is recommended to assess the models' performance in real multi-task cases with delicately selected corresponding metrics for evaluation instead of automatical n-gram metrics with bias.

3. Dataset used in the experiments are far from adequate for evaluating long context processing and length generalization. Arxiv paper and OpenWebText2 are too domain specific with specific characteristics and cannot applied to evaluate the real capability of LLM in the downstream tasks encountering long text. Please refer to the well acknowledged datasets that are recently proposed for long context evaluations such as ZeroScrolls, LEval, Longbench and etc.

### Questions
See more in Weaknesses sections.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
