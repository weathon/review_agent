# Latent Structure of Affective Representations in Large Language Models

- Decision: Reject
- Scores: 8, 2, 2, 6

## Abstract
The geometric structure of latent representations in large language models (LLMs) is an active area of research, driven in part by its implications for model transparency and AI safety. Existing literature has focused mainly on general geometric and topological properties of the learnt representations, but due to a lack of ground-truth latent geometry, validating the findings of such approaches is challenging. Emotion processing provides an intriguing testbed for probing representational geometry, as emotions exhibit both categorical organization and continuous affective dimensions, which are well-established in the psychology literature. Moreover, understanding such representations carries safety relevance. In this work, we investigate the latent structure of affective representations in LLMs using geometric data analysis tools. We present three main findings. First, we show that LLMs learn coherent latent representations of affective emotions that align both with widely used valence–arousal models from psychology. Moreover, a comparison with latent structure in human brainwave (EEG) data suggests possible similarities. Second, we find that these representations exhibit nonlinear geometric structure that can nonetheless be well-approximated linearly, providing empirical support for the linear representation hypothesis commonly assumed in model transparency methods. Third, we demonstrate that the learned latent representation space can be leveraged to quantify uncertainty in emotion processing tasks. Our results are based on experiments with the GoEmotions corpus, which contains $\sim$58,000 text comments with manually annotated sentiment.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper explores the geometric structures of latent representations of emotions in large language models, and provides evidence that they show some alignment with valence-arousal models of emotions from psychology and geometric structures observed in human brainwave data.

They employ the GoEmotions dataset of  English-language Reddit comments which have been manually annotated with one of 27 fine-grained emotion categories (or labelled as neutral). These sentences are passed though the Gemma-2-9B and Mistral-7B transformer language models and mean-pooled activations at each layer were recorded. They perform pairwise classification analysis to test the linear separability of emotion representations, finding high test accuracy (>0.9) in Gemma-2-9B across al layers and moderate test accuracy (0.55-0.89) in Mistral-7B.

In addition, they perform dimension reduction using C-MDS and ISOMAP and visualse these activation, observing weak geometric alignment with valence-arousal coordinates (from the ANEW dataset). They also provide numerical quantification of this using the $R^2$ statistic of a scaled orthogonal Procrustes alignment.

They perform similar analysis of neural EEG data and find similar conclusions, and demonstrate how the geometry of misclassified emotion can be leveraged for uncertainty quantification.

### Strengths
This paper has a well-stated goal, a clear and effective methodology, and is very clearly presented and written. The paper provides additional insight into how latent concepts are represented in the internal activations of transformer models, and is a valuable addition to this line of interpretability research

In particular, the paper is well-scoped and sets out a clear hypothesis: "Do LLMs develop coherent internal representations of emotions that align with the valence-arousal model?". The methodology is unambiguous and clearly explained, and the evidence that is provided for the hypothesis is presented in an objective manner that tells a clear story and is not overstated. The graphical displays of the emotion representations are very insightful and nicely presented and quantitive analysis backs up the conclusions which are suggested by the plots.

The analysis of the neural EEG data is also very enlightening, and may suggest additional hypotheses that might be investigated by future researchers. The investigation of the the inherent ambient and intrinsic dimension of the representations, and into the non-linearity of the representations also helped complete by understanding of the authors investigations.

Overall, I really enjoyed this paper and would recommend it for acceptance.

### Weaknesses
I think the authors did a good job of defining their scope and thoroughly investigating their hypothesis within this scope. For this reason, I have no weaknesses to mention.

### Questions
I would be interested to know whether you manually inspected your nearest-neighbour graphs when performing ISOMAP to check for short-circuits? I also think it would be interesting to show the ANEW valence-arousal plots alongside the C-MDS plots, and and the result of applying ISOMAP to it alongside the ISOMAP plots.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper seeks to understand the geometry of emotion representation in LLMs. It does this by using the accuracy score of the pair-logistic classifier over the hidden states of the model when prompted to classify the emotions of a text. It then uses MDS and other latent space techniques on the distance matrix constructed from the pairwise accuracies to get the embeddings space. This method is repeated on EEG data and the resulting 2D map of emotion is compared between the two types of data. Finally a emotion classification technique is proposed that incorporates some of the techniques from earlier in the paper.

### Strengths
The paper is very ambitious in its scope. It tries to find a latent geometry of emotions in LLMs and how it maps to EEG data. Both of these are important and interesting problems that the community needs to tackle. Understanding how LLMs process emotions is important to understanding how they interact with the human-side of language. The method in section 6 is an interesting approach to improve LLMs ability to do emotion classification. The paper acknowledges its limitations well (though despite this, some of them are core weaknesses nonetheless).

### Weaknesses
The paper is very ambitious overall. However, I fear that the paper tends to overclaim results instead of properly explaining what their results are showing.
-	By asking the LLM to classify the emotion of the text, there is the worry that you are “priming” the network to focus on emotion and thus are not looking at how it processes emotions naturally when no attention is focused on that element of text.
-	The paper examines hidden states but not effects of each sub-layer
-	Results are only on two LLMs
-	GoEmotions is on reddit dataset which has one mode of writing. Would prefer to see the results of this analysis on multiple emotion classification datasets to see if the results generalize across different modes of writing.
-	Figure 2 appears to show the geometry of the MDS accuracy-space, not the intrinsic emotional geometry of the LLM’s hidden states. This visualization therefore reflects how easily separable emotions are under linear probes, rather than how emotions are internally organized in the model. Moreover, representing each emotion as a single point, rather than distributions or clouds of examples, obscures whether this organization is consistent within and across emotion labels.
-	It is not clear how VAD scores were produced for the GoEmotions dataset.
-	The comparison with EEG data is again a comparison logistic regression accuracy scores (how separable are the emotions in the EEG data). I am not sure this shows any mapping between how the two types of data represent emotions. At most it seems to show that these emotions are similarly linearly separable. That is an interesting result, but not the same as what is being claimed in the paper.
-	Similarly, from my understanding of reading Section 5, the geometric structure analyzed is of the separability matrix, not of the hidden-states themselves.
-	Section 6 is well executed but it does no mention of how much their method improved emotion classification. The only result (that I saw) was the result of their method with no baseline comparison.

### Questions
Why is it claimed that accuracies in a dissimilarity matrix map represents how emotions are encoded in LLMs? Can you please make this argument clearer?

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work tries to answer whether LLMs (Gemma-2-9B and Mistral-7B) actually encode emotion structures similar to human models of emotion, The authors use a technique using pairwise logistic probing on the GoEmotions dataset, visualize the latent space withgeometric embedding (MDS/Isomap), and statistical alignment (Procrustes analysis) with human data statistically map this geometry onto well-known human psychological models (like ANEW valence arousal maps) using Procrustes analysis, including EEG emotion manifolds. They cap this with a practical demonstration using geometric distance for quantifying uncertainty

### Strengths
- Original Concept: Using affective models as a benchmark for latent geometry is a really neat, original idea for interpretability research. It's meaningful. Using the well-established structure of human affect (valence/arousal) as a geometry benchmark for LLM interpretability is clever and novel. 

- Cross-Modal Insight: The comparison to human EEG data is a creative touch that makes the structural alignment claims much more interesting. The analysis uses appropriate stats like permutation tests and trustworthiness metrics, that's good rigor.

- Interdisciplinary: Comparing LLM structures to EEG emotion manifolds is a creative and refreshing cross-domain perspective.

- Practical Utility: The section on uncertainty quantification is a thoughtful application that proves these geometric insights have promise for safety applications.

- Clear Writing: The paper is easy to follow, and the experimental flow makes sense. The combination of probing, embedding, and Procrustes alignment is technically sound and robust. The use of permutation tests for significance is a strong point.

### Weaknesses
This is a paper with novel methodology that will encourage discussion by linking affective science and LLM geometry. The methodology is strong, but the authors must address the limitations around linear bias, small model scope, and the measurement of manifold of the human-LLM parallels. and the geometric analysis could be expanded, the study provides genuine insight and is methodologically clear. it's likely to generate good discussion.


- Heavy on Linear Probing: Relying only on logistic regression assumes the emotion categories are linearly separable. This might really bias the geometry they end up seeing. It would be stronger to include a non-linear probe (like a kernel method or maybe UMAP embeddings) to check this. to truly validate the latent manifold claims

- Limited Demonstration (evaluation): They only tested two open-source small (7B–9B) models, and they are both relatively small. We don't know if these results will hold up for larger, instruction-tuned, or multimodal systems. Generalizability is a concern.

- Interpretational Overreach: The alignment with EEG data is a correlation and should be framed more cautiously. Structural alignment is not evidence of cognitive isomorphism; this point needs to be softened in the text. Also, The EEG alignment is interesting, but the claim needs check or proof. it's more of an analogy of shared structure, not firm evidence of shared cognitive encoding.

- "Manifold" Claims are Vague: The eigenspectra and Isomap results actually suggest the structure is fairly diffuse and high-rank. Calling this a clean "affective manifold" needs explanation. This papers own eigenspectrum and Isomap results point to a diffuse, high-rank structure, which works against the central idea of a clean, low-dimensional affective manifold. This internal contradiction should be clarified

- Citation needed during discussion of Background and when motivating that distributed embeddings capture affective dimensions. Prior work showing how affect (valence/arousal/dominance) maps onto word embeddings and methods to retrofit standard embeddings to capture affective dimensions. the paper’s claim that latent spaces capture valence–arousal.  Shah, S., Reddy, S., & Bhattacharyya, P. (2022). Affective Retrofitted Word Embeddings (AACL 2022) ACL Anthology


Suggestions

- should ablate or test sensitivity to other alternative distance metrics (e.g., Mahalanobis or just plain cosine) to ensure the geometry isn't picked by author.

- Add a clearer, quantitative summary of the Procrustes alignment quality (like a scatter or correlation plot). Include a simple scatter/correlation plot showing the LLM space vs. the human space to visually assess the Procrustes quality.

- Make the language more clear when discussing human-LLM parallels; focus on structural vs. cognitive similarity. emphasize structural analogy rather than shared representations

- A measurement report with curvature would better support (or refute) the "manifold" interpretation.

### Questions
Please see the concerns raised in the Weaknesses section

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
3

### Summary
This paper investigates how large language models (LLMs) internally represent emotions, using the GoEmotions dataset as a testbed. By analyzing activations from Gemma-2-9B and Mistral-7B, the authors find that emotion representations in these models align with the well-known valence-arousal model from psychology and exhibit a parabolic shaped geometry resembling human affective structure. The study compares these LLM embeddings to human EEG data, showing similar geometric layouts, suggesting parallels between artificial and biological emotion processing. Using linear (MDS) and nonlinear (Isomap) manifold analyses, the authors conclude that affective representations are mostly linear with mild nonlinearity, consistent with the linear representation hypothesis. Finally, they demonstrate that geometric distances in this space can be used to quantify uncertainty in emotion classification, yielding well-calibrated confidence estimates.

### Strengths
- The paper presents a creative, interdisciplinary bridge between computational neuroscience and LLM interpretability, linking artificial and human affective structures.
- The paper provides quantitative and visual evidence that LLMs encode emotion categories in ways consistent with psychological valence-arousal models and EEG data.
- The discovery that LLMs exhibit modestly nonlinear yet largely linear affective geometry helps reconcile competing interpretability hypotheses.
- Figures effectively visualize emotion clusters, parabolic structures, and the connection between emotion axes and valence.

### Weaknesses
- The study is limited to two LLMs (Gemma-2-9B, Mistral-7B); generalization to larger or instruction-tuned models remains untested.
- The datasets (GoEmotions, FACED) are small and culturally specific, possibly limiting the universality of the findings.
- The causal link between latent geometry and model behavior (e.g., emotion reasoning, empathy) is not established.

### Questions
- How stable is the observed valence–arousal alignment across model families (e.g., LLaMA, GPT, Claude)?
- Could fine-tuning on emotion datasets distort or strengthen the parabolic geometry found here?
- Could the geometric features observed be used to steer model emotional tone or improve alignment for safety applications?

### Soundness
3

### Presentation
3

### Contribution
3
