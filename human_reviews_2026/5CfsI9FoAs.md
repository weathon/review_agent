# Synthetic Bootstrapped Pretraining

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
We introduce Synthetic Bootstrapped Pretraining (SBP), a language model (LM) pretraining procedure that first learns a model of relations between documents from the pretraining dataset and then leverages it to synthesize a vast new corpus for joint training.
While the standard pretraining teaches LMs to learn causal correlations among tokens within a single document, it is not designed to efficiently model the rich, learnable inter-document correlations that can potentially lead to better performance.
We validate SBP by designing a compute-matched pretraining setup and pretrain a 3B-parameter and a 6B-parameter model on up to 1T tokens from scratch.
We find SBP consistently improves upon a strong repetition baseline and delivers up to 60% of performance improvement attainable by an oracle upper bound with access to 20x more unique data.
Qualitative analysis reveals that the synthesized documents go beyond mere paraphrases -- SBP first abstracts a core concept from the seed material and then crafts a new narration on top of it.
Besides strong empirical performance, SBP admits a natural Bayesian interpretation: the synthesizer implicitly learns to abstract the latent concepts shared between related documents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Synthetic Bootstrapped Pretraining (SBP), a framework that improves language model pretraining by generating synthetic documents from existing data.

SBP first finds semantically related document pairs $(d_1, d_2)$, trains a conditional model to predict $d_2$ from $d_1$, and then uses this model to synthesize a large corpus for joint training.

Experiments with a 3B-parameter model under compute-matched setups (200B and 1T tokens) show consistent improvements over repetition baselines and recover up to ~50% of the oracle gain from 20× more unique data.
Quantitative analysis confirms that SBP-generated data maintains diversity and factuality comparable to real text.

The paper also offers a Bayesian view, interpreting SBP as implicitly learning latent “concept” structures across documents, providing a principled explanation for its data-efficiency benefits.

### Strengths
1. SBP introduces a principled way to exploit inter-document relations for self-synthetic pretraining without external teachers.

2. Compute-matched experiments at 200B/1T scales show consistent gains and high-quality synthetic data, recovering about half of the oracle improvement.

### Weaknesses
1. While SBP claims to capture inter-document correlations, the experiments mainly report aggregate metrics (loss, accuracy) without direct analysis of cross-document dependencies or representation changes. The authors should add the following experiments:

    - Evaluate whether the SBP-trained model can generate $\hat{d}_2$ from $d_1$ that semantically aligns with the true $d_2$ , proving it learns document-to-document conditional structure.

    - Compare embedding distances of paired vs. random documents, see if SBP pulls semantically related but stylistically different pairs closer than the baseline, which proves structural alignment in representation space.

2. The repetition baseline may not reflect realistic industry data-mixing or quality-weighted sampling strategies, possibly inflating SBP’s relative gains.

3. Add ablation: use random document pairs to train the synthsizer.

4. The paper does not clearly state whether the synthesizer-tuning and data generation costs are included in the compute-matched budget, leaving uncertainty about true efficiency.

### Questions
1. The paper argues that SBP captures inter-document structure ignored by standard pretraining. 

    - Could the authors provide a more direct analysis or evidence of such cross-document dependencies (e.g., similarity shifts in embedding space, conditional generation tests, or document-pair prediction tasks)?

2. The 200B-scale experiment uses a 37.5% synthetic and 62.5% real data ratio.

    - How sensitive are the results to this ratio?

    - Would a smaller amount of synthetic data (e.g., 10–20%) still bring noticeable gains?

3. The paper presents SBP as compute-matched to repetition baselines, but it remains unclear whether the synthesizer training and data generation costs are included in the total compute budget.

    - Could the authors clarify how these costs were accounted for, and whether SBP remains more efficient when end-to-end compute (including synthesis) is considered?


4. The synthesizer is initialized from a pretrained LM, which may introduce prior knowledge not attributable to SBP itself.

    - Could the authors clarify how much this initialization contributes to the overall improvement?

    - Have they tried training the synthesizer from scratch or using weaker initialization to isolate SBP’s effect?

### Soundness
2

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
This paper introduces Synthetic Bootstrapped Pretraining (SBP), a new large-scale pretraining procedure that aims to capture inter-document correlations that standard LM pretraining overlooks. SBP identifies semantically similar document pairs, trains a conditional “synthesizer” model to generate one document from another, and then uses it to synthesize new text for joint training. The method is evaluated using compute-matched experiments on a 3B model trained up to 1T tokens. Results show consistent improvement over repetition-based baselines, recovering about 40–50% of the performance gains achievable by an oracle model with access to 20× more unique data. The authors also present a Bayesian interpretation of SBP and provide qualitative and quantitative analyses of the generated data.

### Strengths
1. Introduces a new dimension of pretraining that models inter-document relations rather than intra-document token dependencies.
2. Clear structure and logical flow from motivation.

### Weaknesses
1. Can a 1.5B model trained on a 200B dataset generate data that is both relevant to the source documents and diverse? 
2. What is the computational resource consumption across the three stages?  
3. The related evaluation metrics are highly dependent on the LLM used for judgment. What is the judged LLM?
4. The "pair copying" metric should not be compared solely with the source documents but rather with the entire training dataset. This is because the model might output documents that are similar to others in the dataset, which would contribute little to overall improvement. 
5. Comparing Figure 6 and Figure 7 reveals that as the dataset size increases, the proportion of synthetic data required decreases, which to some extent also illustrates this issue.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Synthetic Bootstrapped Pretraining (SBP), a novel framework aimed at improving data efficiency by explicitly modeling inter-document correlations within the pretraining dataset. The authors conduct rigorous compute-matched large-scale experiments (3B model, up to 1T tokens) demonstrating that SBP significantly outperforms the traditional repetition baseline and effectively reduces the performance gap to the "Oracle" upper bound (a model with access to more unique data). The concept is novel, the experiments are solid, the results are convincing, and the inclusion of a theoretical explanation based on a Bayesian hierarchical concept model makes this a valuable and highly relevant research direction for addressing the imminent "data scaling wall."

### Strengths
1. SBP creatively addresses the issue of exhausting high-quality text data. By learning a conditional generator $P(d_2|d_1)$ from document pairs to synthesize new data, it serves as a powerful complement to traditional token-level autoregressive training, leveraging latent structural signals in the data.

2. The experiments utilize a 3B-parameter model, making the scale relevant for frontier LM development. Crucially, the compute-matched setup ensures a fair comparison across SBP, the repetition baseline, and the Oracle.

3. The results (Table 1 in the original paper) show SBP bridges 42% to 48% of the QA accuracy gap to the Oracle, confirming that the synthesized data $S_{pretrain}$ provides a real and effective signal.

### Weaknesses
1. The quantitative analysis (Tables 2 and 7 in the original paper) shows an alarmingly high non-factual error rate of 15.1% for the 200B-scale synthetic data (compared to 1.81% for real data). Although this rate decreases to 8.7% at the 1T-scale, it remains a significant concern. By the way, the caption in Table 2, `are not are' should be `are not'?

2. The paper acknowledges in Section 6 that the initial nearest neighbor search relies on the external Qwen3-0.6B Embedding model. Why choose such a model, while efficiency is a valid reason, this limitation detracts from the "Bootstrapped Pretraining" claim. 

3. The $d_1 \to c \to d_2$ relationship captured by SBP is fundamentally a statistical dependency (or association) based on a shared latent concept $c$. While this process involves an "implicit posterior inference," framing this learned relationship as "Causality" in a statistical context is inaccurate and can be misleading, as it does not imply a strong intervention relationship.

### Questions
See weakness.

### Soundness
2

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
3

### Summary
This paper proposes a method to generate synthetic data for pre-training in constrained data settings. The method relies on a document generation model that learns to generate new diverse documents from given ones. Concretely, an LM is pre-trained on a dataset D_pretrain (standard next token prediction) and then fine-tuned on the document generation task. To fine-tune the document generator, pairs of input-output documents are created by pairing related documents in D_pretrain. The document generator is then used to generate a synthetic corpus of documents D_synthetic. An LM is then pre-trained on D_pretrain+D_synthetic. The authors also discuss a Bayesian interpretation of the standard data pre-training objective versus the document generator (synthesizer)fine-tuning one; the former marginalizes out latent concepts from which documents are generated while the second models the posterior over these concepts.

### Strengths
- An intelligent idea to augment with synthetic data that carries new signals (compared to the repetition baseline).

### Weaknesses
- There are known issues related to biases set up in pre-training (e.g., [1]). The authors should probably report analysis with respect to this. In particular because of the document generator.

[1] https://openreview.net/pdf?id=KQhUEoPmJy

- From the experiments it is not clear how the improvements brought by the data synthesizer would generalize to larger models. From Table 2 it seems to indicate that with scale, the generated documents might be closer to the input documents. Still the idea is interesting and could be useful for the community.

### Questions
- Line 135 DATA-CONTRAINED > CONSTRAINED

- For the Quantitative analysis in Section 4.2. It would make sense to add metrics based on novel n-grams, e.g.,  percentage of new n-grams appearing in the generated documents or those used in [2], to better reflect the deviation of the generated documents from the seed ones.

[2] https://aclanthology.org/N18-1065/

- Statistics about the length of synthetically generated documents?

### Soundness
3

### Presentation
3

### Contribution
3
