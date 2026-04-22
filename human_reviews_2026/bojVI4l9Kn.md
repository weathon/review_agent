# Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability

- Avg Score: 6.50
- Decision: Accept (Oral)
- Scores: 6, 10, 6, 4

## Abstract
Translating the internal representations and computations of models into concepts that humans can understand is a key goal of interpretability. While recent dictionary learning methods such as Sparse Autoencoders (SAEs) provide a promising route to discover human-interpretable features, they often only recover token-specific, noisy, or highly local concepts. We argue that this limitation stems from neglecting the temporal structure of language, where semantic content typically evolves smoothly over sequences. Building on this insight, we introduce Temporal Sparse Autoencoders (T-SAEs), which incorporate a novel contrastive loss encouraging consistent activations of high-level features over adjacent tokens. This simple yet powerful modification enables SAEs to disentangle semantic from syntactic features in a self-supervised manner. Across multiple datasets and models, T-SAEs recover smoother, more coherent semantic concepts without sacrificing reconstruction quality. Strikingly, they exhibit clear semantic structure despite being trained without explicit semantic signal, offering a new pathway for unsupervised interpretability in language models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Temporal Sparse Autoencoders (Temporal SAEs), an extension of standard SAEs designed to handle sequential data. The key idea is to encourage features to remain active over time, capturing higher-level, temporally consistent concepts while allowing lower-level features to vary more freely. The authors motivate this through an analogy between syntactic (short-term, local) and semantic (long-term, abstract) interpretability, and propose a smoothness-based regularization term to enforce temporal coherence in activations. Experiments on language model activations show that Temporal SAEs produce more stable and interpretable latents, with some evidence of improved long-range consistency and separation between high- and low-level features.

### Strengths
- The distinction between syntactic and semantic interpretability, and how this relates to temporal sequences, is a novel framing. It helps clarify how different kinds of latent dynamics might reflect hierarchical structure in sequential data.

- The conceptual split between high-level, consistent latents and low-level, fast-varying ones is elegant and intuitive. It connects well to both neuroscience analogies and hierarchical representations in modern sequence models.

- Figure 1 effectively communicates the model’s core idea and its temporal behavior—it’s visually clear and supports the text well.

- The modeling assumptions are reasonable, and the paper does a good job of articulating the intuition behind them, making the method easy to follow.

- The experiment on long-range consistency adds depth, showing that the proposed method captures meaningful temporal dependencies beyond local smoothness.

### Weaknesses
- It’s not entirely clear what the practical takeaway is. Should one prefer semantic or syntactic features, and in what contexts? The paper motivates the distinction well, but it stops short of offering guidance on which is preferable or when. Relatedly, the utility of the proposed method remains somewhat vague. Section 4.5 shows an application of Temporal SAEs, but without a comparison to standard SAEs, it’s hard to tell whether the temporal extension is actually necessary or beneficial for that task.

- The smoothness metric feels fragile. Because it depends on the maximum activation change, it can overemphasize outliers or produce unstable values when latent differences are small. A more stable approach (such as comparing relative high-frequency energy or using multi-scale smoothness measures) would likely give a clearer and more reliable picture of temporal consistency.

- The spliced-text experiment (Figure 4) would be much stronger if it directly compared Temporal SAEs to normal SAEs. Figure 1 provides a helpful teaser, but including this comparison within the main experimental section would make the qualitative claims far more convincing.

- Table 1 leaves some gaps. It would be informative to include smoothness scores for the low-level split, since those features are expected to vary more quickly and thus might show even lower smoothness than standard SAEs. Including this would help validate the intended hierarchy between low- and high-level components.

### Questions
- What is the intended takeaway from the distinction between syntactic and semantic features? Should one type be preferred or emphasized in certain contexts? More broadly, what is the practical utility of Temporal SAEs compared to standard SAEs? In what kinds of tasks or analyses do the temporal extensions provide clear advantages?

- Could the authors include a direct comparison between Temporal SAEs and standard SAEs in the spliced-text experiment (Figure 4)? This would help confirm that the observed behavior genuinely depends on the temporal component rather than standard SAE properties.

- How robust is the current smoothness metric to outliers or very small latent changes? And would the authors consider alternative measures—such as frequency-domain smoothness (e.g., high-frequency energy ratio) or multi-scale smoothness—to confirm that the results are not artifacts of the chosen metric?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper introduces a novel variant of the Sparse Autoencoder (SAE) architecture, called the Temporal Sparse Autoencoder (TSAE). This architecture separates features into two groups, where one group of features has a contrastive loss term encouraging similar values in adjacent tokens. This approach is theoretically motivated by considering a text to have "high-level" features like topic and tone (which should remain consistent across the text), and "low-level" features like part of speech (which varies on a token-by-token basis), and the observation that classical SAEs mostly identify the later. The paper measures TSAEs on classical SAE metrics like total variance explained (where they perform similarly) and on new metrics like ability to label MMLU question categories (where TSAEs outperform classical SAEs).

### Strengths
This work addresses a core shortcoming of SAEs as an interpretability technique, which is that the interpretable features they find are often too specific to individual tokens to be useful. For instance, a feature might be "Sentences endings or periods" (Figure 1), which is interpretable and useful to the SAE's reconstruction, but is not useful for downstream applications like steering.

In this regard, the addition of temporal consistency is a natural evolution of the SAE architecture. The discussion in Section 3.1 in particular is excellent at reframing our expectation of how text carries information in order to motivate the temporally-informed architecture.

The authors include clear details to facilitate reproduction of their architecture and experiments. 

The authors demonstrate the advantages of their technique with a range of evidence, including quantitative experiments (Figure 4 and Table 1), a case study (Section 4.5), and visually striking diagrams (Figures 1 and 2). The case study also provides a possible application of this technique.

### Weaknesses
The contrastive loss has a complicated structure, and the authors do not motivate or explain what it has that form. There is insufficient explanation of why this contrastive loss outperforms the naive temporal similarity loss term (Lines 454-458 and Table 2).

The case study in Section 4.5 shows quantitative results, but does not compare them to a classical SAE. This makes it hard to judge whether the TSAE architecture is an improvement over previous methods in this context.

### Questions
Figure 1: What language is this copy of the Bhagavat Gita in? If it is an English translation, please indicate that, since otherwise a reader may be confused as to why the "Non-english text" feature fails to fire.

Line 202: Aren't the two terms in $L_{contr}$ the same? In the second term, if one renames $i$ to $j$ and vice-versa, and uses the symmetry of $s(x,y)=s(y,x)$, doesn't one get the first term?

Line 202: Would it be possible to test a linearized $L_{contr}$? In particular, with modest approximations it is possible to simplify $L_{contr}$ to $- \displaystyle \frac {1} {N} \sum_i s_{i,i} + \frac{1}{N^2} \sum_{i,j} s_{i,j}$, where $s_{i,j}$ denotes $s(z_t^{(i)}, z_{t-1}^{(j)})$. This would make the contrastive loss more immediately understandable, and might speed training.

To achieve this simplification, consider the log term for a single $i$ in $L_{contr}$:

$\log(\frac{e^{s_{i,i}}}{\sum_j exp(s_{i,j})}) = s_{i,i} - log(\sum_j e^{s_{i,j}})$

$≈ s_{i,i} - log(\sum_j (1+s_{i,j}))$

$= s_{i,i} - log(N + \sum_j s_{i,j}))$

$= s_{i,i} - log(N) - log(1+\frac 1 N \sum_j s_{i,j})$ 

$≈s_{i,i} - log(N) - \frac{1}{N} \sum_j s_{i,j}$

where the two approximations are via the taylor series approximations $exp(x) \approx 1+x$ and $log(1+x) \approx x$, respectively.

Now, discarding the $log(N)$ for being constant, averaging over the $i$, and negating, we reach

$L_{contr} ≈ - \frac 1 N \sum_i s_{i,i} + \frac{1}{N^2} \sum_{i,j} s_{i,j}$

If the cosine similarities fall in a tighter range (e.g. they are almost always in the range [0,1]), it might be appropriate to linearize around a point besides x=0.

Line 331: When you calculate AutoInterp scores, how do you handle dead latents? Are the excluded from the average for the autointerp score?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper targets a limitation in standard SAEs used for interpretability: by treating tokens as i.i.d, they disproportionately discover local, token-specific, and often syntactic features  failing to capture higher-level semantic concepts. The authors sensibly argue this is a mismatch with the structure of language, where semantics evolve smoothly while syntax is more local. They propose Temporal Sparse Autoencoders which modify the SAE framework by (a) partitioning the feature dictionary into a "high-level" (semantic) set and a "low-level" (syntactic) set, using a Matryoshka-style residual loss.It introduces a novel temporal contrastive loss that encourages the high-level features to have similar activations for adjacent tokens, while pushing them to be dissimilar from other tokens in the batch. Experiments on Pythia-160m and Gemma-2-2b show that this simple addition works quite well. Qualitative results on spliced text show T-SAEs identify smooth, semantic features that cleanly track topic shifts, unlike the noisy, high-frequency activations of baseline SAEs (like matryoshka SAE). Probing confirms T-SAE high-level features are far better at capturing semantic and contextual information, while the low-level features capture syntax. The authors also noted that improvement in feature quality is achieved without sacrificing standard reconstruction performance.

### Strengths
The presentation of the paper is very clear: the motivation, method, and results are all presented in a way that is easy to follow. The spliced-text visualizations are particularly strong, providing an "it just works" demonstration that is more compelling than the quantitative metrics alone.

The experimental validation is robust. The authors demonstrate their method's contribution through: (a) The smooth, semantic features in the visualizations are neat, (b) probing results confirm the high-level/low-level disentanglementm, and (c) the method maintains competitive performance on standard SAE benchmarks, showing the gains aren't at the cost of reconstruction fidelity.

Despite some novelty concern (detailed below), the results are solid. It confirms that even simple temporal priors can significantly improve the usefulness of SAE features, moving the analysis from token-level only to be useful on a sequence-level as well.

### Weaknesses
A major weakness of this work is the overall lack of proper contextualization of their work against highly relevant works. The central problem—that the i.i.d. assumption for tokens is a flaw and corresponding solution that temporal dynamics should be leveraged for smoother extracted features is not new. For instance, this paper published earlier at iclr 2025 has identified very same problem and proposed a similar temporal modification to SAE (https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ycscpaQAAAAJ&citation_for_view=ycscpaQAAAAJ:2osOgNQ5qMEC). 

Similarly, as an important and related problem (as the authors admit in related works), I believe feature absorption deserves more discussion. Even if not adding some experiments on feature absorption, which should be straightforward given the authors already use SAEBench (https://github.com/adamkarvonen/SAEBench), at least discuss why and contextualize with previous SOTA/prior works on this benchmark, such as https://openreview.net/forum?id=i31cKXiyim and https://www.alignmentforum.org/posts/zbebxYCqsryPALh8C/matryoshka-sparse-autoencoders (both are concurrent with the Matryoshka SAE that the authors cited but still relevant).

Another area that the model can improve upon is the limited scale. With experiments mostly focused on Pythia-160m, and Gemma-2-2b, it is plausible, although eventually unclear, whether the results will generalize mech interp as a field is moving towards larger model.

### Questions
On the feature split: The 20%-80% split between high- and low-level features seems rather arbitrary. How sensitive is the model to this split? And more importantly, are there good rules of thumb when setting this value for a new model/layer without running a sweep?

On the adjacent token assumption: The contrastive loss relies on $x_t$ and $x_{t-1}$. This is a very local definition of semantic smoothness. Have you explored other sampling strategies, such as a small window of recent tokens or a distance-weighted contrastive loss?

My rating is currently marginal towards accept/reject, although I'm willing to raise my rating subject to the weaknesses/questions being sufficiently addressed.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
In this paper, the authors argues that standard SAEs have some problems, e.g., recover mostly shallow, token-local, often syntactic features because they ignore a basic property of language. To address this issue, the authors propose Temporal SAE (T-SAEs), which add a contrastive-style temporal consistency loss that encourages high-level features to stay active across adjacent tokens in a sequence. For experiments, the authors try multiple models and datasets. Results show that the proposed  T-SAEs uncover richer semantic structure than vanilla SAEs.  The authors  also show these semantically cleaner features can help real tasks such as surfacing safety-relevant behaviors. Overall, the topic is interesting, but there are certain concerns that need to be addressed.

### Strengths
1.The topic about temporal SAE is promising and interesting. 

2. The experiments are conducted on multiple models and datasets. 

3.The visualizations of tnse are impressive.

### Weaknesses
[1] In Figure 1 (c, what is the number in x-axis mean? Does it mean the location in a long sentence? 

[2] Could you also discuss the limitations of this study, and potential future directions? 

[3] the experiments are conducted on models Pythia-160m and Gemma2-2b, with small parameter sizes. The reviewer understands this might be constrained by computational resources. I am not asking for additional experiments. However, could you discuss the motivations for choosing these models, and whether the results observed on these models will generalize to bigger models? 

[4] in Line 252, the authors mentioned the training is on layer 8 and layer 12. COuld you explain why other layers are not selected? 


[5] The writing could be further polished. For example, in the Introduction, more references should be added, e.g., reference to SAE in line 34.

[6] In the second paragraph of Intriduction, could you provide some empirical results or references to support the claim in the first sentence? 

[7] In line 77 and 78, authors mention that “This feature should remain active throughout the entire sentence, because the semantic content does not change from word to word” . However, it could be possible, the content change suddenly in a sentence or paragraph. For example, we could first talk about the whether, and then talk about our cats or dogs. 

[8] Could you explain more about Figure 2? This figure seems comprehensive, but missing detailed explanations. 

[9] In section 4.3, are these metrices proposed by this paper or adopted from existing studies?

### Questions
[1] In Figure 1 (c, what is the number in x-axis mean? Does it mean the location in a long sentence? 

[2] Could you also discuss the limitations of this study, and potential future directions? 

[3] the experiments are conducted on models Pythia-160m and Gemma2-2b, with small parameter sizes. The reviewer understands this might be constrained by computational resources. I am not asking for additional experiments. However, could you discuss the motivations for choosing these models, and whether the results observed on these models will generalize to bigger models? 

[4] in Line 252, the authors mentioned the training is on layer 8 and layer 12. COuld you explain why other layers are not selected? 


[5] The writing could be further polished. For example, in the Introduction, more references should be added, e.g., reference to SAE in line 34.

[6] In the second paragraph of Intriduction, could you provide some empirical results or references to support the claim in the first sentence? 

[7] In line 77 and 78, authors mention that “This feature should remain active throughout the entire sentence, because the semantic content does not change from word to word” . However, it could be possible, the content change suddenly in a sentence or paragraph. For example, we could first talk about the whether, and then talk about our cats or dogs. 

[8] Could you explain more about Figure 2? This figure seems comprehensive, but missing detailed explanations. 

[9] In section 4.3, are these metrices proposed by this paper or adopted from existing studies?

### Soundness
2

### Presentation
2

### Contribution
3
