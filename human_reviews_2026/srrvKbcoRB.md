# FLEXITOKENS: Flexible Tokenization for Evolving  Multilingual Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
Multilingual language models are challenging to adapt to new data distributions by simple finetuning due to the rigidity of their subword tokenizers, which typically remain unchanged during adaptation. This inflexibility often leads to inefficient tokenization, causing overfragmentation of out-of-distribution domains, unseen languages, or scripts. In this work, we develop byte-level LMs with learnable tokenizers to make tokenization adaptive. Our models include a submodule that learns to predict boundaries between the input byte sequence, encoding it into variable-length segments. Existing tokenizer-free methods train this boundary predictor using an auxiliary loss that enforces a fixed compression rate across the training corpus, introducing a new kind of rigidity. We propose FLEXITOKENS, a simplified training objective that enables significantly greater flexibility during adaptation. Evaluating across multiple multilingual benchmarks, morphologically diverse tasks, and domains, we demonstrate that FLEXITOKENS consistently reduces token over-fragmentation and achieves up to 10% improvements on downstream task performance compared to subword and other gradient-based tokenizers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors build upon prior work on LMs with gradient-based tokenization and address the issue of fixed compression rates by replacing the binomial loss with a new loss function. This modification enables the model to dynamically adjust its compression rate, thereby reducing over-fragmentation and improving downstream task performance.

### Strengths
1.The paper is well structured and highly readable.

2.The motivation is clearly stated, and the methodology is explained in detail.

### Weaknesses
1.The experimental section of the paper includes only two baselines: BPE and Binomial loss, which are insufficient to fully demonstrate the effectiveness of the proposed method.

2.Most of the experiments are conducted on a model with 119M parameters, making it unclear whether the proposed approach can maintain its effectiveness when scaled to larger language models.

### Questions
1.It would be helpful if the paper included a diagram or illustration of the proposed method in the main text, so that readers could more easily grasp the overall framework.

2.The proposed method requires a small n-way parallel corpus to compute per-language parameters. Could the authors clarify whether the choice of this parallel corpus has an impact on the resulting compression rates or model performance?

3.The experiments include only one gradient-based tokenization baseline. By the way,can this method be applied to different model architectures beyond the one used in this paper?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the problem of overfragmentation in subword tokenization. To remedy this, the paper proposes using byte-level LMs with a boundary predictor. (i.e., internally, a model processes a sequence by segment instead of byte.) The proposed method also introduces an additional loss term to make the single boundary predictor effective across different languages. The experiments are conducted on a small-scale setup (119M); the proposed method is compared with the standard BPE-based model and a variant without the additional loss term. For downstream evaluation, each model is fine-tuned on target-domain data. The results show that the proposed method (especially $\lambda$3) outperforms baselines in terms of encoding efficiency and downstream performance.

### Strengths
1. The paper proposes a novel method for encoding and processing multilingual text more effectively and efficiently. The methodology is well-motivated and relatively easy to understand.
2. The results of a small-scale experiment show promising results over the standard BPE baseline and the baseline: Binomial, which does not employ the proposed flexible loss function, in terms of encoding efficiency and fine-tuned downstream performance (when used with $\lambda3$).

### Weaknesses
Overall, the paper proposes a novel method to encode/process multilingual text more effectively and efficiently. Nonetheless, its experimental design and the discussion/analysis require substantial modification.

1. Some expressions in the paper are quite awkward and require a fix. For instance, “Language models are challenging to adapt… (L11)” is grammatically unnatural and should be modified as something like: “Adapting language models to new data distributions by simple finetuning is challenging”. The same applies to “overfragmentation of out-of-distribution domains (L14)”.

2. The major limitation of this paper is a lack of state-of-the-art baselines. This paper deliberately omits a comparison with the closest related work, HNet (Hwang et al., 2025; https://arxiv.org/abs/2507.07955), citing that it is unstable during training (Main, 2025; https://main-horse.github.io/hnet/scaling-laws-byte/). This is a rather weak argument, and citing a blog post to support the claim is not entirely reliable. Also, other studies address a similar challenge (e.g., the Byte Latent Transformer; ACL 2025; https://aclanthology.org/2025.acl-long.453.pdf). The lack of such state-of-the-art baselines prevents us from evaluating the actual impact/effectiveness of the proposed method in comparison to them.

3. Another major limitation of this paper is the use of a small-scale model (119M). Due to this limitation, most of the presented results are from fine-tuning. This is an uncommon practice in the previous literature, where they use zero-shot and few-shot evaluation. Also, this small scale appears to favor the proposed method over the BPE baseline, as the BPE baseline has only five layers compared to 16 for the proposed method. On a larger scale, the observed trend might substantially change.

4. Some results are missing without justification. For instance, the results for the proposed method ($\lambda$1) are missing for CS en-es/CS en-hi in Table 3. Table 2 does not have the results for $\lambda$1 and $\lambda$2. In particular, L323-L358 requires corresponding results for $\lambda$1 and $\lambda$2 to be more convincing. The current statement lacks clear evidence and is not convincing, given that a higher $\lambda$ does not always achieve the best results in Table 3. Furthermore, why does the MT evaluation in Table 2 only cover three language pairs?

5. The mentioned qualitative analysis in L366-369 is missing.

### Questions
1. On Weakness 1, please make sure to make necessary changes regarding the overall writing of the submission.
2. In Section 2.1, how does “the language modelling module” pool the byte-level hidden states? Is it just mean pooling?
3. On Weakness 2, I would suggest including HNet and BLT baselines.
4. On Weakness 3, I would suggest including a larger-scale experiment (e.g., at least 1B).
5. On Weaknesses 4 and 5, please make sure to include the missing results and fix the discussion accordingly.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Tokenization with a fixed vocabulary (e.g., BPE) often suffers from rigidity, leading to over-fragmentation when applied to unseen languages or scripts. Recent work introduces a tokenization module within the language model that segments bytes into patches, learning the desired compression rate through gradients from an auxiliary loss. This is done by incorporating a tokenization module consisting of a Boundary Predictor and an upsampling module, replacing the standard vocabulary-bounded embedding and unembedding layers. The entire model, including these modules, is pretrained jointly.

However, the authors argue that using a predetermined compression rate introduces another form of rigidity.
To address this, the paper proposes a new auxiliary loss function, "FlexiToken" loss, a hinge-like objective that encourages the compression rate to remain within a range (b - a) rather than regressing to a single target value a (referred to as "Binomial" in prior work). In practice, this relaxes the existing training loss by introducing a lower bound and by not penalizing the tokenizer when the compression rate exceeds this threshold.
This is to design a better, flexible adaptation to new domains, languages, and scripts.
Additionally, the authors simplify the model by using a single Boundary Predictor shared across languages, instead of maintaining separate predictors for each language.

Main findings :
1. After pretraining, the proposed method achieves higher compression rates and greater variance in compression compared to the binomial baseline, while maintaining similar perplexity.
2. After fine-tuning on downstream tasks, the model with FlexiToken shows consistent performance improvements over the binomial baseline.

### Strengths
1. Comprehensive overview of current trends:
The paper provides a well-structured summary of recent advances in byte-level tokenization, clearly outlining existing approaches and highlighting their key limitations such as fixed compression rates and lack of adaptability across languages and scripts.
2. Clear motivation for the proposed method:
The motivation behind introducing a new loss function is well articulated, effectively connecting the rigidity of existing compression-based tokenization methods to the need for a more flexible framework.
3. Extensive experimental validation:
The paper has thorough experiments across a wide range of tasks and diverse languages. After fine-tuning on downstream tasks, the model with FlexiToken shows consistent performance improvements over the binomial baseline.
4. Well-formulated and technically sound proposal:
The introduction of the hinge-like FlexiToken loss is conceptually clear and mathematically grounded.

### Weaknesses
1. While the paper's central claim emphasizes improved flexibility and adaptability over the binomial baseline, the empirical evidence supporting this is not entirely convincing. In particular, the variance results discussed in L301 and illustrated in Figure 2 do not clearly demonstrate the reported increase in variance. Moreover, the observed improvements in compression rate may partially stem from the broader allowable range introduced by the proposed loss, rather than from intrinsic flexibility of the model itself.

2. Although the proposed method shows higher performance than the binomial baseline, it remains unclear whether these improvements arise specifically from the relaxation of the compression rate range or from other factors, like single vs separate boundary predictors. Further ablation study would strengthen this claim, for example, by fixing a single language and gradual increase of the lambda (e.g. 0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2) while keeping the same alpha as binomial experiment.

3. The comparison between the binomial and FlexiToken models' compression rates after fine-tuning (Figure 4) is not entirely convincing for demonstrating FlexiToken's advantage over binomial. While the first-row results for en, es, and ru show that the binomial model remains relatively low and stable compared to FlexiToken, the second-row results for hi, te, and uk suggest that the binomial model also adapts to the tasks to a similar extent. This seems like the binomial model, despite its "rigid" compression rate, is still capable of adapting to different tasks and languages.

4. (writings)
- Line 291: Bits per Byte (BPB) may need a brief explanatory phrase clarifying its relation to the original concept, as BPB is not mentioned in the original reference.
- Figure 2 (left, y-axis and titles) uses the Bits per Character (BPC), not BPB.
- Duplicate citations: Lines 702 and 706 contain repeated references that should be consolidated.
- Table 17: The BPE outputs are not readable.
- Figure 4: The x-axis ticks for es and ru are misaligned and should be corrected for consistency.
- It would be helpful to include a pointer to a specific example (if any in Appendix) that substantiates the qualitative observations (Lines 367–372) discussed in this section.

### Questions
1. Figure 1: It may make more sense to compare the proposed method with the binomial baseline rather than BPE. How does the binomial model fragment this text?
2. L156: The calculation of σ is not clearly explained. What exactly does it represent, and what is the tokenization rate before the FlexiToken module is trained?
3. Confused by these two non-penalization descriptions in the intro and methods section:
- L053: "By not penalizing the tokenizer when the compression rate is higher than …" suggests that penalization would push the compression rate lower.
- L158: "This loss will become 0, reducing further incentive to compress by not penalizing the model." implies that penalization would push for compression.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce a tokeniser-free language modelling architecture, FlexiTokens. They present it as more flexible than previously published tokeniser-free architectures. They evaluate their architecture on several tasks, comparing it with BPE and a boundary predictor trained with a binomial loss as in Nawrot et al. (2023).

### Strengths
- A sound tokeniser-free architecture proposal
- Evaluation on multiple tasks, including adaptation to a new domain

### Weaknesses
-The state of the art is incorrectly presented, and, as a result, novelty claims overstated. For instance, in the first paragraph of Section 5, the authors list a number of papers to illustrate architectures that "segment byte sequences into fixed-length [segments]", but this is incorrect for several papers in the list. They then write that "These models are retrained with a fixed target compression rate, which limits their ability to [etc.]", which again is not correct for at least some of the papers in question. This is problematic, because fixed compression rates are mentioned in the abstract and introduction as a main limitation of previous works that the paper overcomes.
- More generally, the authors do not explain how their proposal is novel with respect to previously published tokeniser-free architectures, and their evaluation does not compare their evaluation to any other tokeniser-free architecture.
- The authors write that "there has been little research on adapting tokenizer-dfree LMs to new data distributions". They do not seem to be aware that at least one of the papers they cite in the previous paragraph does exactly that (Godey et al. 2022 do it on on noisy data, both synthetic and naturally occurring).

### Questions
- What are the previously published architectures that are the most similar to yours, how is yours different, and why is it better?

### Soundness
2

### Presentation
1

### Contribution
2
