# StyliTruth : Unlocking Stylized yet Truthful LLM Generation via Disentangled Steering

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Generating stylized large language model (LLM) responses via representation editing is a promising way for fine-grained output control. However, there exists an inherent trade-off: imposing a distinctive style often degrades truthfulness. Existing representation editing methods, by naively injecting style signals, overlook this collateral impact and frequently contaminate the model’s core truthfulness representations, resulting in reduced answer correctness. We term this phenomenon stylization-induced truthfulness collapse. We attribute this issue to latent coupling between style and truth directions in certain key attention heads, and propose \textbf{StyliTruth}, a mechanism that preserves stylization while keeping truthfulness intact. StyliTruth separates the style-relevant and truth-relevant subspaces in the model’s representation space via an orthogonal deflation process. This decomposition enables independent control of style and truth in their own subspaces, minimizing interference. By designing adaptive, token-level steering vectors within each subspace, we dynamically and precisely control the generation process to maintain both stylistic fidelity and truthfulness.  We validate our method on multiple styles and languages. Extensive experiments and analyses show that StyliTruth significantly reduces stylization-induced truthfulness collapse and outperforms existing inference-time intervention methods in balancing style adherence with truthfulness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The article discusses steering vectors in the context of **improving honesty when changing the style** of prompts used for language models. For example, we may want a model to respond more empathetically while still remaining truthful.

The authors show that certain layers in a language model are responsible for style and truthfulness (author used only one language model, Qwne-1.5-14B-Chat), and that some layers mix these concepts. They introduce a method to separate these concepts using space decomposition, based on the activations obtained from contrastive prompt pairs (e.g., a negative vs. a positive prompt), which are then analyzed using Singular Value Decomposition.

The authors also introduce a capturable coefficient applied when tokens are generated. This coefficient uses information such as singular values from the previous decomposition to adjust the strength or importance of token activations.

Furthermore, they show that some subsets of model heads exhibit orthogonal concept representations that is, certain stylistic and truthful features are encoded in distinct, "non-overlapping" subspaces.

Finally, the authors demonstrate that their method improves the model’s ability to generate honest responses while maintaining a desired style, outperforming baseline methods such as Contrastive Activation Addition.

### Strengths
The authors' method doesn't require a lot of GPU resources, which is great for researchers with limited computing power. The research problem is well-motivated - style changes can accidentally affect truthfulness, so understanding this connection is important for reliable model editing. Figures 4 and 5 are very clear and visually appealing, making the results easy to understand.

### Weaknesses
The main problem of the article is **clarity**. Several key definitions, explanations, and references are missing or underdeveloped, which makes it difficult to follow the authors' argument and fully appreciate their contributions.

Additionally, the evaluation would be stronger if the authors included experiments on 3–4 additional models, allowing for better generalization and comparison.

**I am willing to increase my score if the following issues are addressed**:
- The clarity and organization of the paper are significantly improved.
- The experiments are extended to include at least 3–4 additional models.

**Suggested Improvements**

I think the paper would benefit from clearer definitions of key terms. Adding explicit explanations for concepts such as relevant heads and irrelevant heads (in Section *Problem Formulation and Analysis*) would help readers follow the authors' approach more easily. I was a bit confused about their definitions.

It would also be helpful to move the evaluation metrics from the appendix into the main text and to clearly state which models the authors are evaluating in the body of the paper (the only place where the evaluated language models appear is in Appendix A3, which feels somewhat *hidden*).

The authors' core claim that *existing representation-editing methods for style transfer often induce a marked collapse in truthfulness* (in the Introduction) is central to the paper's motivation. Supporting this statement with citations or experimental evidence (perhaps in Table 1) would strengthen the argument considerably.

Extending the evaluation to include 3-4 additional language models would make the authors' findings more robust and convincing. This would help demonstrate that their approach generalizes well across different architectures.

A few smaller adjustments would also improve the paper's flow. The reference to the mental health conversational agents paper (lines 2–7) feels somewhat disconnected from the main argument (and I also could not find supporting connections in the provided references). In addition, parameter *K* (the decomposition of subspace using SVD) could use a brief explanation in the main text discussing its role and sensitivity. Finally, Section 5 (*Discussion*) might fit better if integrated into Section 4 or moved to the Appendix.

Figure 6 is difficult to read in its current form. Consider increasing the figure size and using larger fonts to improve readability. Alternatively, the authors could show a representative subset of layers and attention heads in the main text and move the complete figure to the Appendix.

A careful proofread would also help - for example, there is a grammatical error in the question *Does style-relevant subspace and truth-relevant subspace really exit?*, which should read *Do style-relevant subspace and truth-relevant subspace really exist?*

### Questions
All problems and questions are included in the Weaknesses section.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors observe that stylized representation editing undermines truthfulness, resulting in untruthful responses. They analyze this phenomenon and find that it is because of the entanglement of style and truth features in the activation space. To address this, they construct two orthogonal subspaces using an orthogonal deflation method, forming independent bases for style and truth to reduce cross-impact. The empirical results demonstrate that the proposed method achieves good stylization while preserving truthfulness.

### Strengths
1. The paper is well organized with good writing, making it easy to comprehend. In particular, Figure 1 clearly outlines the procedures.
2. The paper targets a challenging problem, i.e., existing representation-editing-based methods for style transfer of ten induce a marked collapse in truthfulness. The authors address this problem following a why-and-how procedure. Specifically, they first observe that such a collapse arises from the entanglement between style- and truth-relevant activation differences. Subsequently, they disentangle this correlation by proposing an orthogonal deflation method. Both the observations and the proposed method are novel.
3. The empirical results demonstrate that the proposed method achieves good stylization while preserving truthfulness.
4. Extensive ablation studies are conducted to validate the effectiveness of various steps.

### Weaknesses
1. This paper claims one of its contributions as introducing a **training-free** editing framework; however, according to the descriptions in Section 4.2, it seems that we have to **train** binary linear classifiers to select relevant heads.

2. This paper targets at truthfulness. The main benchmark for this perspective is TruthfulQA. However, on TruthfulQA, there are two common tasks—generation task and MC task. And in this paper, the authors only compare the proposed method with others on the generation task.

### Questions
1. It would be appreciated if the authors add more descriptions on attention head selection in Section 4.2. For instance, it remain unclear to me if we need to train a binary linear classifier for **each** head. 

2. Is there any comparison between the performance of using the proposed adaptive token-level editing and that of using the constant intervention strengths? which could help validate the effectiveness of this editing strategy.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies a problem termed "stylization-induced truthfulness collapse," where applying inference-time activation steering to make a Large Language Model (LLM) response stylized often causes the response to become factually incorrect. The authors propose StyliTruth to address this. StyliTruth operates by first using linear probes to identify attention heads and activation subspaces relevant to style and truthfulness independently. In heads where these attributes are entangled, it uses an orthogonal deflation technique to disentangle the two subspaces. This allows the method to inject steering vectors for style and truthfulness independently, aiming to produce responses that are both stylistically appropriate and factually accurate.

### Strengths
1. The paper clearly articulates and provides examples for the "stylization-induced truthfulness collapse", a practical problem in controllable text generation.

2. The proposed method is targeted. It attempts to operate on the model's representations by identifying and disentangling the specific subspaces responsible for the conflict.

3. The method demonstrates superior performance outperforming all baselines, and the ablation study (Table 2) effectively shows that both the subspace disentanglement and adaptive token-level editing components contribute to the final performance. The visualizations also provide good intuition that the disentanglement process is working as intended.

### Weaknesses
1. The method's novelty is somewhat limited. It is a direct extension of prior work, seemingly DRESS (Ma et al., 2025), but adapts the framework to also handle truthfulness. The core techniques used—linear probing, SVD for subspace identification, and orthogonal deflation—are all well-established, making this more of a novel combination and application rather than a new fundamental technique. Moreover, the ablation studies, experimental analyses and presentation styles seem very similar to DRESS.

2. The paper motivates the simultaneous control by showing a trade-off. However, is this entanglement fundamental? Could a simpler, sequential approach (e.g., first applying style steering, then applying truth-correcting steering) achieve comparable results? A comparison against such a baseline would strengthen the claim that complex subspace disentanglement is necessary.

3. In Sec.6.1, The metrics are directly introduced by capital letters without any explanations. It is a bad presentation approach. At the very least, the authors should explicitly introduce the full name of the metrics and cite the benchmarks.

4. The method relies on the assumption that style and truth can be cleanly separated into orthogonal subspaces (for Case 2). This may not hold for more complex attributes (e.g., a "formal scientific" style is inherently tied to truthfulness) or in all model architectures.

### Questions
1. Given that this work is based on DRESS (Ma et al., 2025), how much of the "StyliTruth" method is a re-application of the DRESS framework versus a fundamentally new contribution?
 
2. The method relies on selecting the top-H heads and top-K singular vectors. How sensitive is the model's performance to these two hyperparameters? The sensitivity analysis on the number of heads (Fig. 7d) is useful, but a similar analysis for the subspace dimensionality (K) would be insightful.

3. Does a "truth-relevant subspace" identified from TruthfulQA generalize to preserving truthfulness on other domains or topics? Or would the "truth" subspace need to be re-computed for different types of factual knowledge?

### Soundness
3

### Presentation
2

### Contribution
2
