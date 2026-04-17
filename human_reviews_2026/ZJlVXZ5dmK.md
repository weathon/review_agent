# Learning multimodal dictionary decompositions with group-sparse autoencoders

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2, 6

## Abstract
The Linear Representation Hypothesis asserts that the embeddings learned by neural networks can be understood as linear combinations of features corresponding to high-level concepts. Based on this ansatz, sparse autoencoders (SAEs) have recently become a popular method for decomposing embeddings into a sparse combination of linear directions, which have been shown empirically to often correspond to human-interpretable semantics. However, recent attempts to apply SAEs to multimodal embedding spaces (such as the popular CLIP embeddings for image/text data) have found that SAEs often learn ``split dictionaries,'' where most of the learned sparse features are essentially unimodal, active only for data of a single modality. In this work, we study how to effectively adapt SAEs for the setting of multimodal embeddings while ensuring multimodal alignment. We first argue that the existence of a split dictionary decomposition on an aligned embedding space implies the existence of a non-split dictionary with improved modality alignment. Then, we propose a new SAE-based approach to multimodal embedding decomposition using cross-modal random masking and group-sparse regularization. We apply our method to popular embeddings for image/text (CLIP) and audio/text (CLAP) data and show that, compared to standard SAEs, our approach learns a more multimodal dictionary while reducing the number of dead neurons and improving feature semanticity. We finally demonstrate how this improvement in alignment of concepts between modalities can enable improvements in the interpretability and control of cross-modal tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper “Learning Multimodal Dictionary Decompositions with Group-Sparse Autoencoders” introduces a new approach to understanding multimodal embeddings (such as CLIP or CLAP) through sparse autoencoders (SAEs). While traditional SAEs have been effective for interpreting single-modality models by decomposing embeddings into sparse, interpretable features, they struggle in multimodal settings—often producing “split dictionaries” where features activate for only one modality. The authors address this limitation by theoretically showing that a non-split, multimodal dictionary with better modality alignment can always exist, motivating their proposed model: the Masked Group-Sparse Autoencoder (MGSAE). This model incorporates a group-sparsity regularizer and cross-modal random masking to encourage paired activations between modalities during training, thereby promoting shared multimodal features.

### Strengths
1. The paper does more than propose an empirical modification—it provides a theoretical result (Theorem 1) demonstrating that a modality-split dictionary can always be improved to a more aligned, multimodal one. This theoretical grounding strengthens the motivation for their method and shows that poor cross-modal alignment in SAEs is not a fundamental limitation, but rather an optimization bias that can be addressed.
2. The introduction of group-sparse regularization combined with cross-modal random masking is elegant and well-motivated. These additions directly tackle the issue of unimodal activation in SAEs, encouraging shared sparse representations across modalities without requiring extra supervision or hand-engineered alignment losses.
3. The paper is the first to systematically analyze and apply sparse autoencoders to audio/text embeddings (CLAP). This extends interpretability research into a new multimodal domain beyond the widely studied CLIP image–text models.
4. The authors introduce new multimodal monosemanticity metrics that quantify neuron-level semantic alignment across modalities. This is an important methodological contribution for future interpretability research, as it offers a way to measure the coherence and cross-modal consistency of learned concepts.
5. The proposed MGSAE not only improves interpretability—by learning more multimodal, semantically coherent features—but also enhances zero-shot cross-modal task performance, showing practical benefits in alignment-sensitive applications.
6. The paper successfully theoretical insights about dictionary decompositions and practical model design choices. The theoretical results directly inform the architecture, and the empirical results validate those theoretical predictions.

### Weaknesses
1. Although the paper evaluates both image–text and audio–text embeddings, these experiments are still confined to two-modality settings. The proposed method’s scalability to more complex multimodal spaces (e.g., video–audio–text or vision–language–action models) remains unexplored. Additionally, the experiments are conducted on relatively small or well-curated datasets (CC3M and JamendoMaxCaps), leaving open questions about generalization to noisier or larger-scale data.
2. The training configurations (e.g., 25k–10k steps with fixed dictionary size and sparsity) seem tuned for feasibility rather than large-scale rigor. There is limited discussion of hyperparameter sensitivity (e.g., λ for group sparsity, masking probability, or K-sparsity), which could significantly affect outcomes. Without broader ablations or scalability studies, it’s difficult to assess robustness or reproducibility across architectures and modalities.
3. While Theorem 1 provides an elegant existence result, it is relatively abstract and does not characterize the conditions under which optimization will find such multimodal dictionaries in practice. The theory does not account for the stochastic and non-convex nature of training neural autoencoders, so the link between theory and empirical convergence remains somewhat heuristic.
4. Most evaluations revolve around interpretability metrics (monosemanticity, dead neurons, activation overlap). While important, the work provides limited evidence of improvements in real downstream tasks such as retrieval, captioning, or multimodal reasoning. Demonstrating such task-level benefits would strengthen claims of practical relevance.
5. The group-sparse and shared masking mechanisms force stronger alignment between modalities, which may inadvertently suppress modality-unique features that are still valuable (e.g., color-specific features in vision but not text). The paper does not deeply analyze whether this trade-off affects the diversity or richness of learned representations.
6. While the paper reports quantitative metrics, it provides few qualitative examples of what specific multimodal concepts the MGSAE learns (e.g., visualizing dictionary atoms corresponding to text–image pairs). Such examples would have made the interpretability claims more concrete and convincing.

### Questions
1. Your theorem shows that a non-split dictionary with improved modality alignment always exists. Do you have any insights into how this theoretical result translates into practice—for example, what properties of the optimization landscape or initialization help the model actually discover such dictionaries?
2. You base your approach on the Linear Representation Hypothesis. Have you observed any systematic deviations from linearity in multimodal embeddings (e.g., nonlinear interactions between visual and textual features) that limit the effectiveness of linear sparse decompositions?
3. How sensitive is the model’s performance to the choice of the group-sparse regularizer (L₂,₁ norm)? Did you consider other structured penalties, such as mixed ℓ₁/ℓ∞ norms or hierarchical sparsity, to encourage shared but flexible activations?
4. The random masking step seems crucial for reducing dead neurons. Did you explore deterministic or learned masking strategies, or investigate how the masking probability affects multimodal feature sharing?
5. You share encoder and decoder weights across modalities, except for biases. Have you tried partially shared architectures (e.g., modality-specific encoders with shared latent layers)? If so, how does this affect multimodal alignment?
6. How does your method scale with more than two modalities (e.g., image–text–audio)? Would group-sparsity naturally extend to this case, or would you need a modified loss to maintain balanced multimodal alignment?
7. Did you observe high sensitivity to hyperparameters such as λ (for group sparsity), K (for sparsity level), or the masking probability p? Are there guidelines or heuristics you recommend for stable training?
8. Could you share specific examples of learned multimodal concepts (e.g., visualizing dictionary elements that align text like “a dog running” with corresponding visual features)? This would help illustrate what kinds of shared semantics are captured.
9. Does improving multimodal alignment ever come at the cost of losing fine-grained, modality-specific concepts (like visual texture or acoustic timbre)? How do you balance these competing objectives?

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
2

### Summary
The authors have tackled the problem of “split dictionary” observed while using SAEs on multimodal models. They propose the existence of a unified dictionary with improved alignment and also a modified architecture of SAE to realise this. This is applied to CLIP and CLAP data. The authors are also the first to use audio as one of the modalities with text. They also propose a paired multimodal monosemanticity metric.

### Strengths
1.	The problem is well motivated
2.	The proposed metric calculation is explained well.
3.	The authors have also considered the problem of dead neurons.

### Weaknesses
My key concern is that the comparisons are only with SAE and GSAE. Can the authors provide empirical experiments comparing their method against the prior approaches described in the literature [1-4]?

Other questions are detailed in the Questions section.

### Questions
1.	The authors propose a loss term with two group sparse codes – are there ablation studies without this loss?
2.	Also is there any way to prevent the $L_{gs}$ term from causing the codes z and w from collapsing to zero vectors? If this happens, the loss term goes back to being like a standard SAE loss.
3.	The authors propose the use of random masking to encourage multimodality of concepts. Are there ablation studies to show this helps? 
4.	My key concern is that the comparisons are only with SAE and GSAE. Can the authors provide empirical experiments comparing their method against the prior approaches described in the literature [1-4]?
5.	The authors mention that they are the first to use audio data in a multimodal setting. Please provide insights on why audio data is hard to include in general. How does their approach compare with audio only methods[5]?
6.	In Figure 3, MGSAE does best consistently, is there any intuition for why it is the best?
7.	The approach has been evaluated on CLIP and CLAP, can SigLIP be added? If not, please explain why.

[1] Isabel Papadimitriou, Huangyuan Su, Thomas Fel, Sham Kakade, and Stephanie Gil. Interpreting the linear structure of vision-language model embedding spaces. arXiv preprint arXiv:2504.11695, 2025.

[2] Mateusz Pach, Shyamgopal Karthik, Quentin Bouniot, Serge Belongie, and Zeynep Akata. Sparse autoencoders learn monosemantic features in vision-language models. arXiv preprint arXiv:2504.02821, 2025.

[3] Hanqi Yan, Xiangxiang Cui, Lu Yin, Paul Pu Liang, Yulan He, and Yifei Wang. Multi-faceted multimodal monosemanticity. arXiv preprint arXiv:2502.14888, 2025.

[4] Vladimir Zaigrajew, Hubert Baniecki, and Przemyslaw Biecek. Interpreting CLIP with hierarchical sparse autoencoders. arXiv preprint arXiv:2502.20578, 2025.

[5] Pluth, D., Zhou, Y., & Gurbani, V. K. (2025, February). Sparse Autoencoder Insights on Voice Embeddings. In 2025 Conference on Artificial Intelligence x Multimedia (AIxMM) (pp. 1-6). IEEE.

### Soundness
2

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
This work studies interpretability of multimodal embeddings using sparse auto-encoder. The main objective is to find aligned sparse multimodal latent vector from sparse autoencoder. To this end, a definition of modality-split dictionary is provided and the author propose multimodal monosemanticity score (MMS). Moreover, with masking approach, their proposed SAE approach shows improved multimodal dictionary from experiments.

### Strengths
In terms of interpretability, understanding aligned multimodal embedding is important. The paper propose important concept of modality split dictionary, which I think a crucial for interpretability of multimodal embeddings. Proposed approach is simple yet effective as the experimental results verify the authors claim. Overall, although I think the contribution seems a bit incremental, problem and the proposed definition are worth looking at.

### Weaknesses
While I enjoyed reading the paper, I think the paper could be more improved if more experimental analysis is conducted. For example, I do not see ablation study in choosing $K$ in TopK step, and there is no intuitive explanation or implication from it. Moreover, there is no ablation study for $p$ for random mask. It is even difficult to see why the random masking is required. Thus, I think the paper needs more evidence and experiments, which would make it more convincing.

### Questions
1. Can the author provide what is the impact on choosing $K$ in interpretability and downstream tasks (e.g., classification)?
2. What is the rationale to use random mask? I saw the random mask gives very marginal gain. Why does this method give such a small gain?
3. Section 5.3 is interesting. Could the author provide more example or similar studies?

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
3

### Summary
This paper is a well-placed venture in expanding interpretability for multimodal domains by using Sparse AutoEncoders. The authors tackle the issue of modality splitting in the “features” captured by SAEs that are trained on just reconstruction loss for multimodal models. Key contributions are expanding the metric of monosemanticity of a given feature to include the modality of activations and introducing an updated training recipe to improve alignment between the sparse encodings of different modalities.

### Strengths
1. Investigates and quantifies the role of modality alignment in feature discovery using SAEs
2. Introduces a simple but seemingly effective method to improve the discovery of interpretable monosemantic features 
3. Expands the application of SAEs to audio/text CLAP models alongside image-text CLIP models
4. Discusses implications on downstream tasks like classification, retrieval, and steering

### Weaknesses
1. Only considers a limited range of test models (one for each set of modalities), which limits the evidence for the applicability of the proposed method. In particular, state of the art models like AiMV2 are shown to have different distribution of feature weights between modalities compared to the studied encoder [1], and the applicability of the proposed MGSAE method in such situations is unclear. 

2. Ablation studies aren’t presented, in particular, different expansion ratios for the SAE (number of features), and different sparsity measures (K) aren’t explored in the body of the text. This leaves unanswered questions about the applicability of MGSAE and if the method would continue to improve alignment and reduce dead neurons in different settings. 

3. Does not fully expand on the distribution of activations / neurons between different modalities. In particular, he authors do not delve into the comparative strength of activations between the modalities, nor the frequency of the activated concepts. A show of how the weight of the features are split between the modalities would be very useful evidence to further justify the paper’s claims of improved alignment between modalities using the presented method. 

4. Does not present baselines / benchmarks comparing the introduced MGSAE method with other known works that present potential improvements in multimodal settings over the TopK SAE method. In particular the methods proposed in [2] which are cited by the work would help position this work in the existing literature. 

5. While the MMS metric is motivated in intuition by the paper, no quantitive comparisons are present to further validate the measure compared to the results in [3]. An understanding of how MMS and raw MS compare, especially between the known methods and those introduced in the paper could further cement the metric’s usefulness.

[1] “Interpreting the Linear Structure of Vision-language Model Embedding Spaces”, COLM 2025

[2] “Interpreting CLIP with Hierarchical Sparse Autoencoders”, ICML 2025

[3] “Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models“, NeurIPS 2025

### Questions
1. In Figure 5, the right side (MGSAE) shows 2 entires of concepts with the name “beautiful blonde” - do the authors understand why this might have happened and how the actual underlying concepts differ ? 

2. Table 1: Do the authors any intuition as to why there seem to be cases where the (M)GSAE methods perform so competitively despite seemingly having many dead neurons ? Would a potential No SAE baseline for these tasks be helpful in building understanding? 

3. Have the authors considered if the MGSAE method learn stable multimodal features even with random masking ? That is to say, could the concepts learned by the dictionaries in differently seeded runs be aligned ? 

Nits: 
* Line 176: The ‘W’ is missing the subscript for ‘W_dec`
* Line 641: Switch up between z and x, y

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose both a metric and a methodology for better understanding and training sparse autoencoders (SAEs) on vision language model spaces. They propose a metric that combines how useful SAE codes are for aligning semantically aligned pairs, combined with how much SAE codes activate for multiple modalities or just one. They propose that instituting a group-sparse (GS) loss for SAEs is the solution for split-modality concepts, and show that their group-sparse SAEs are better along a suite of metrics like fewer dead codes, more semantic alignment, and better cross-modality transfer.

### Strengths
This paper nicely distills some of the recent research on VLM interpretability into a nuanced metric and and interesting new methodology. 

I find the evaluations of the GS-SAEs convincing, with both the intrinsic SAE evaluations of dead codes and MMS, as well as the more extrinsic zero-shot transfer (which comes out within a reasonable distance of the solidskyline of the actual embeddings), being interesting.

### Weaknesses
I think this is a paper that makes a solid contribution, and I don’t consider any of the below weaknesses to be especially strong.

W1 I don’t feel that the paper has analyses or ablations that increase our intuitions in what it is about Sparse Group loss that causes the decrease in dead neurons, and the multimodal concepts (the latter is more clear). This makes the paper weaker, as what we can mostly get from it is that group-sparse SAEs are better, and so for multimodal models we should switch to them. Since as I understand it the paper would be strongest if it is meant to increase our intuitions about what makes good training for linear concepts, I would like to see another analysis helping with that. Possible questions are: How does the group-sparsity play out in the training dynamics? How do the dictionaries differ geometrically when we include or don’t include that

### Questions
Q1 What are the issues that arise from training an SAE on paired data? Might it be a problem that paired data is less broad in domain? Is there a way to integrate these findings about the GS loss with the advantages of using a broader base of data.

Q2 Do you have any idea about why dead neurons are so influenced by the GS loss, and why this seems to be linked also with the multimodality of features?

Q3 It took me some time to understand how the MMS metric differs from some other similar metrics (like the BridgeScore from Papadimitriou et al 2025), so a slightly expanded explanation would be great. After reading through it more carefully, I think I’m convinced that it’s a subtle way of going about measuring split-modality. Could you expand on the metric, and provide some helpful analysis on if there is interference between monosemanticity and multimodality and how this can be quantified?

Q4 It seems that the main methodological contribution of this paper is the group-sparse loss term, but it is not very intuitively explained in the text (eg after line 269). Can you expand slightly on what group-sparsity means and why joint support is encouraged when using this norm? The text mentions this, but as it is the main methodological contribution to the paper I think a slightly more developed theoretical side could be useful. 

Q5 (minor) Is alignment targeting the same evaluation as monosemanticity? I’m not sure that aligned pairs is a measure that can be translated trivially to mean monosemanticity, it might warrant a sentence or two in the rewrite if you think so 

Q6 (minor) I didn’t find Figure 2 very helpful, it seems to emphasize that there are two encoders/decoders, when in fact the main contribution is the loss term which is in quite small font.

### Soundness
3

### Presentation
3

### Contribution
3
