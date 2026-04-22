# PRO: Enabling Precise and Robust Text Watermark for Open-Source LLMs

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Text watermarking for large language models (LLMs) is important for model owners to verify the origin and protect the intellectual property of AI-generated text. While watermarking methods for closed-source LLMs' text generation are relatively mature, watermarking open-source LLMs' text generation remains challenging. Closed-source model developers typically embed text watermarks during decoding; however, this approach is ineffective for the text generation of open-source models, where developers have no control over how decoding occurs. As a result, owners of open-source LLMs still lack practical methods to verify whether a given piece of AI-generated text originated from their models. The primary challenge lies in embedding watermarks directly into model weights without compromising detection accuracy. One possible solution is first to create a text generation watermark in the closed-source setting, then distill that watermark information into the publicly released model's weights.  However, this approach faces two critical challenges: (i) Reduced detectability due to inconsistency between the watermark patterns learned by the model and the predefined patterns used during detection. This inconsistency arises because existing closed-source watermark patterns are difficult for models to learn effectively. (ii) Vulnerability to modifications by downstream users, such as fine-tuning or model merging, which may weaken or completely remove the embedded watermark. To address these challenges, we propose ***PRO***, a precise and robust text watermarking method for open-source LLMs. First, we introduce a trainable watermark policy model, which is jointly optimized with the LLM during training. This co-optimization helps generate watermark patterns that are easier for the model to learn, significantly reducing inconsistencies between generated patterns and predefined detection criteria. Additionally, we incorporate a regularization term into the watermarking loss, which simulates various perturbations (e.g., fine-tuning, model merging) and penalizes any degradation in watermark detectability under these modifications. This approach ensures that the embedded watermark remains resilient even after downstream model alterations. Our evaluation on mainstream open-source LLMs (e.g., LLaMA-3.2, LLaMA-3, and Phi-2) demonstrates that our approach significantly outperforms prior methods in terms of both watermark detectability and robustness against model modifications. The code is publicly available at https://anonymous.4open.science/r/PRO-DE2A/README.md.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes PRO, a framework for embedding text watermarks directly into the weights of open-source large language models.
It introduces two key components: 1) Co-Adaptive Watermark Policy (CAWP), which jointly trains a learnable watermark mapping and the model to improve generation–detection consistency, and 2) Forgotten Perturbation-aware Learning (FPL), which simulates fine-tuning perturbations during training to enhance robustness against model modifications such as merging, pruning, and quantization.
Experiments on LLaMA and Phi models show that PRO achieves high AUC and low perplexity while maintaining strong robustness compared to prior watermarking methods.

### Strengths
The paper proposes a watermarking for open-source LLMs. The proposed PRO framework—with its co-adaptive watermark policy (CAWP) and forgotten perturbation-aware learning (FPL) is empirically effective, showing strong robustness against fine-tuning, merging, and pruning while maintaining good text quality.

### Weaknesses
1. The convergence analysis in Appendix D essentially restates the standard smoothness and gradient-descent convergence results of KL-based distillation. However, it does not establish any theoretical link between the proposed objectives and the final detection metrics (e.g., AUC or TPR at low FPR), nor does it formally connect to the claimed robustness objective (detectability under fine-tuning, merging, or pruning). For FPL, what is the bound the forgotten perturbation space? In reality, users perform diverse modifications such as LoRA/adapters, parameter averaging or smoothing, and even full re-distillation. It remains unclear how this simplified perturbation approximates such real-world updates.

2. In Table 1, the paper reports AUC which measures overall separability across all thresholds and TPR @ 5%FPR. However, in watermark detection, the low-FPR region is the only practically relevant regime but high AUC does not imply good performance at low FPR. Table 1 should report TPR @ 1% FPR (and ideally TPR @ 0.1\% FPR). This would align with common practice in watermarking papers (e.g., KGW [1]) and make the results interpretable for real-world deployment.

3. The robustness evaluation currently covers only full-parameter fine-tuning, model merging, quantization, and pruning.
However, in the open-source ecosystem, attackers typically use cheaper and more practical operations such as distillation, i.e., training a new student model using an nonwatermarked teacher. This strong modification could completely rewrite the output distribution and is feasible for professional attackers.

[1] A Watermark for Large Language Models. John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein. ICML 2023

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces PRO, a Precise and Robust watermarking framework for open-source large language models (LLMs) that embeds watermarks directly into model weights rather than at the decoding stage, making it effective even when users modify or re-implement decoding. The approach features a trainable watermark policy model jointly optimized with the base LLM to encourage watermark-friendly generation patterns, and a robustness regularization term that simulates downstream perturbations such as fine-tuning and model merging while penalizing any degradation in watermark detectability. Experiments on LLaMA-3.2, LLaMA-3, and Phi-2 show that PRO achieves higher watermark detectability and stronger robustness than prior methods, with publicly released code supporting reproducibility.

### Strengths
- The paper addresses a practical and growing problem: watermarking open-source LLMs where owners lack control over decoding.
- The experiments across multiple open-source models demonstrate that PRO yields higher watermark detectability and improved resistance to post-training modification compared to baseline methods.
- The paper is clearly structured, with intuitive figures.

### Weaknesses
- The paper does not provide a formal analysis or theoretical guarantee on why the joint optimization leads to higher detectability or robustness. A more rigorous treatment (e.g., gradient alignment or mutual information perspective) would strengthen the claims.
- While PRO aims for “precise and robust” watermarking, the authors do not systematically evaluate how the approach affects the model’s general performance (e.g., perplexity, generation quality, reasoning accuracy) on more diverse and broader tasks.
- The paper does not isolate the contributions of the trainable policy model and the robustness regularizer.
- While PRO simulates perturbations from fine-tuning and model merging, it overlooks other crucial real-world perturbations, such as reinforcement learning–based post-training (e.g., RLHF or DPO).

### Questions
In general, the paper is very comprehensive. Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a framework to embed resilient watermarks directly into model weights. It introduces Co-Adaptive Watermark Policy (CAWP) to jointly learn watermark patterns aligned with the model’s behavior, reducing generation–detection inconsistency, and Forgotten Perturbation-aware Learning (FPL) to enhance robustness against fine-tuning and model merging. Experiments on LLaMA3 and Phi2 models show that PRO achieves higher detectability, better text quality, and stronger robustness than existing open-source watermarking methods.

### Strengths
1. Identify the problem of Generation-Detection Inconsistency. The mappings of watermarked tokens are arbitrary.
2. Provide a novel method co-adapting the watmeark model with the real model to better align the watermark with the model's innate performance. And innovatively devise the FPL module to properly solve the weakness of the current open-source model watermark to finetuning.
3. carry out experiment validating the performance of PRO.

### Weaknesses
1. Using model merging as an attack to evaluate learning-based watermarking may be inappropriate, since such attacks assume access to an unwatermarked model. In my opinion model merging shouldn't be considered as a valid attack.
2. Because a key component of CAWP relies on an MLP that extracts semantic information through a BERT encoder, it would be important to include comparisons with prior semantic-invariant distillation method to demonstrate the necessity and contribution of the co-training design.
3. The FPL's loss function seems to only prevent local curvature, lack theoretical and empiraicla exepriment for its efficacy when the model's being trained on more anti-watemrakr token. Does it means the watermakred model itself is hard to be finetuned and gain downstream capability?

### Questions
1. The experiment detail is missing in 3.2's motivation study, which shows the inconsistency between the teacher model and student model, According to prior work [1], it seems to me that the reported AUC drop is unexpectedly large and may require further clarification or replication details.

2. While AUC is reported as the main detection metric, it is not the most interpretable measure for practical watermark detection. It is recommended to include TPR at a given FPR in the main paper rather than relegating it to the appendix, as this provides a clearer assessment of real-world detection performance.
(also see my questions in weakness)

[1] Gu, C., Li, X. L., Liang, P., & Hashimoto, T. (2024). On the Learnability of Watermarks for Language Models. In Proceedings of the Twelfth International Conference on Learning Representations (ICLR 2024).

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
This paper develops a text watermarking method for open-weight LLMs. Because users have control over inference procedures, a watermark for open-weight LLMs must be learned into the weights, instead of relying on decoding-based watermarking. Towards this goal, this paper proposes PRO, a method which jointly trains a watermark policy alongside the LLM, and also includes a loss term that penalizes a decrease in watermark detectability under a weight perturbation. Experiments claim that PRO outperforms baseline methods in terms of text quality, watermark detectability, and robustness against model modifications (e.g., fine-tuning, model merging).

### Strengths
1. Effective and robust watermarking for open-weight LLMs is an important open problem. As open-weight LLMs become more capable and widely used, combating LLM misuse via methods such as watermarking become more important.  
2. The proposed method seems like a natural way to approach the problem. It simultaneously optimizes the watermark policy to increase detectability, along with optimizing against degradation in detectability from a simulated gradient update step on red tokens.  
3. The code is publicly released, enhancing the transparency and reproducibility of the results. Training configurations and hyperparameters are also included in the appendix.

### Weaknesses
1. Watermark detectability still drops significantly in PRO after fine-tuning. The TPR@5 decreases from 0.99 to 0.37 after 1500 fine-tuning steps on OpenMath Instruct, which I’m not sure I would call “robust”.  
2. The numbers reported for the Gloaguen et al. (2025) method in Table 1 do not match up with the numbers they reported, even though the experimental setups seem to be mostly the same. [Gloaguen et al. (2025)](https://arxiv.org/abs/2502.10525) (Table 1\) reports 0.69 TPR@5 after 2,500 fine-tuning steps on OpenMathInstruct (and considers this **not** robust), whereas the PRO paper reports 0.222 TPR@5 after 1,500 fine-tuning steps on OpenMathInstruct. This significant discrepancy is strange.  
3. [Gloaguen et al. (2025)](https://arxiv.org/abs/2502.10525) find that watermark durability can be improved by increasing the distillation dataset size. In their Table 2, they report that this method can achieve 0.91 TPR@5 after 2,500 fine-tuning steps on OpenMathInstruct. The PRO paper does not mention this method at all, and does not run any experiments on how the dataset size/number of training tokens affects durability.  
4. The baseline perplexity of the original model before training is not included. It would be good to have the original model in Table 1, Figure 6, etc. in order to evaluate how much PRO impacts the text quality, compared to no watermarking. It might also be nice to compare with the performance of decoding-based watermarking (perhaps slightly less necessary).  
   * Line 144 claims that PRO “can even match the performance of closed-source counterpart.” But I do not see any experimental comparisons with decoding-based watermarking to support this claim.  
5. Watermark detection is now more expensive, as it requires running an embedding model and MLP. Most existing detectors are non-neural and can be run on CPU only.  
6. It would be ideal to have I.4 Robustness Against Paraphrasing Attack in the main paper, as robustness to text modifications is an important aspect of evaluating watermarking methods. But I understand that the main paper is already at the page limit.  
7. Appendix F: "Watermarking for Closed-Source LLMs" in this paper appears to be mostly taken from Appendix A: "Additional Details on Watermarking Strategies" in [Gu et al. (2023)](https://arxiv.org/abs/2312.04469), yet there is no citation. Some parts are nearly verbatim identical. A citation appropriately indicating the source and extent of reuse should be added to avoid potential plagiarism concerns.  
   * Similarly, some parts of Appendix G: "Model Modification" appear to be paraphrased from Section 3: "Durability of Open-Source LLM Watermarks" of [Gloaguen et al. (2025)](https://arxiv.org/abs/2502.10525), again with no citation. The overall structure/organization of topics is very similar. An appropriate citation should also be added here.

### Questions
1. Why doesn’t the PRO method appear in Figure 2 (right)?  
2. There should be citations for the models used, e.g., Llama 3 (line 139), Phi2 (line 140), BERT (line 264), etc.  
3. Equation 3: KL divergence should be computed on the probabilities, not the raw logits. Also, $\\pi$ represented probabilities earlier, so it is inconsistent to have it denote logits now.  
4. Line 280 describes ensuring half the tokens are more likely and the other half are less likely, i.e., $\\gamma \= 0.5$. But what if the developer wants to use some other value of $\\gamma$, such as 0.25?  
   * Also, zero mean does not necessarily ensure that half are positive and half are negative. If $1/3$ are positive with logits 1 and $2/3$ are negative with logits \-0.5, then the mean is still zero.  
5. The labels in equation (4) for the terms for (i) unbiased token preference and (ii) balanced watermark logits appear to be swapped.  
   * The first term incentivizes zero mean across the vocabulary for each input embedding. (label should be balanced watermark logits)  
   * The second term incentivizes each token to have zero mean across input positions. (label should be unbiased token preference)  
6. Another related work that uses neural networks to generate watermark logits is [Liu et al. (ICLR 2024\)](https://openreview.net/forum?id=gMLQwKDY3N).  
7. What is the perplexity of the original model before any training? And the original model with decoding-based watermarking?  
8. How many tokens are the methods in Table 1 trained on?  
9. What are the training details for the OpenMath/OpenCode fine-tuning experiments, e.g., batch size, learning rate, training time, etc.?  
10. Several references are cited as arXiv preprints, but they have been published in conferences. For example, [Frantar el at. (ICLR 2023\)](https://openreview.net/forum?id=tcbBPnfwxS), [Gu et al. (ICLR 2024\)](https://openreview.net/forum?id=9k0krNzvlV), [Kirchenbauer et al. (ICLR 2024\)](https://openreview.net/forum?id=DEJIDCmWOz), [Kuditipudi et al. (TMLR)](https://openreview.net/forum?id=FpaCL1MO2C), [Sun et al. (ICLR 2024\)](https://openreview.net/forum?id=PxoFut3dWW), [Zhao et al. (ICLR 2024\)](https://openreview.net/forum?id=SsmT8aO45L). Please update these references, and check for any other papers that are incorrectly cited as preprints.

### Minor notes

1. The font size in equations (1) and (2) seem abnormally small.  
2. The absolute values in equation (4) should be made larger so they are the same size as the enclosed expression (can use `\left` and `\right`).  
3. I suggest using $|x|$ instead of $N$ for the sequence length in equation (3), to avoid confusion with $N$ as the number of samples.  
4. Per the [ICLR Author Guide](https://iclr.cc/Conferences/2026/AuthorGuide), the Ethics Statement and Reproducibility Statement should appear before the References section, right after the main paper.

### Soundness
2

### Presentation
2

### Contribution
3
