# Sparse Autoencoders Trained on the Same Data Learn Different Features

- Decision: Accept (Poster)
- Scores: 8, 4, 8, 2

## Abstract
Sparse autoencoders (SAEs) are a useful tool for uncovering human-interpretable features in the activations of large language models (LLMs). While some expect SAEs to find the true underlying features used by a model, our research shows that SAEs trained on the same model and data, differing only in the random seed used to initialize their weights, identify different sets of features. For example, in an SAE with 131K latents trained on a feedforward network in Llama 3 8B, only 30\% of the features were shared across different seeds. We observed this phenomenon across multiple layers of three different LLMs, two datasets, and several SAE architectures. While ReLU SAEs trained with the L1 sparsity loss showed greater stability across seeds, SAEs using the state-of-the-art TopK activation function were more seed-dependent, even when controlling for the level of sparsity. Our results suggest that the set of features uncovered by an SAE should be viewed as a pragmatically useful decomposition of activation space, rather than an exhaustive and universal list of features ``truly used'' by the model.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper engages with the question of wether SAEs learn some fundamental set of features, which is a common theme in the interpretability literature. The paper studies this using feature alignment (an approach from the literature), and runs evals across multiple models, saes, and datasets. 
In particular, the paper checks feature alignment across different seeds, and it checks the interpretability of aligned/not features.
It also studies the sensibility of alignment to hyperparameters such as the sae expansion factor and k in topk.

### Strengths
This paper is rigorous and works across different families of models, saes, and datasets.
The paper studies a fundamental question to the development of SAEs: wether they learn a universal set of features or not, and it makes a meaningful contribution to the discussion.
The findings are very interesting (and I believe will be interesting to the community), the highlights were:
- features learned across different seeds are significantly different
- the interpretable features were more aligned
- increasing the number of latents reduced alignment, and so does increasing k in topk

This leads to interesting places, for ex:
- maybe the linearly encoded learnable interpretable features only represent a fraction of the total loss (see dark matter sae paper)
- maybe the SAEs have a massive tail of niche features making up a significant portion of the loss, and pick different ones initially due to the seed, but asymptotically we would see all features aligning

### Weaknesses
Fundamental:
- the paper could use some human interpretability besides fuzzing, as auto interp can be misleading 

Style:
- it is not customary to cite personal communication with another team (about gemma scope in this case) as supporting evidence in an argument
- typo at line 209 "an miss"

### Questions
Have you tried experiments on small models with extremely large values of k or the sae expansion factor (to see if the alignment eventually converges)?

Have you tried human interp?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the stability of SAE features learnt on same model & data but different random seeds. The authors compute a bijective matching of features in two SAEs by maximizing the average cosine similarity of encoder vectors or decoder vectors in Hungarian algorithm. The results show that only 30% of features are shared between two SAEs in Llama 3 8B. Interpretability of non-shared features are statistically worse than shared features.

### Strengths
1. The paper is well-written. All experimental settings are clarified in detail. Results are easy to follow.
2. The conclusion they reached, that SAE trained on the same data learn different features, is verified in various models, including SmolLM, GPT-2 and Llama 3.1 8B, respectively on their training set. The Impact of hyperparameter selection is extensively discussed in Section 4.3.
3. Interpretability of shared features versus non-shared features is measured.

### Weaknesses
1. Although this paper conducts a wide range of experiments on the phenomenon, the research question is still a bit simple. The main text focuses on the phenomenonology and empirical studies. The authors do not discuss 1) the underlying mechanism behind the instability of features, 2) the impact of instability of features (how the unpaired latents will harm the interpretability ability of SAEs), or 3) the potential method to consistently discover a universal set of features.
2. While a gap is showed in the interpretability of shared features versus non-shared features, the potential causes and its meaning is unclear.

### Questions
1. (As in Weaknesses 2) As shown by Wang et al. 2024, shared features may largely be features of low complexity, e.g. single token features. Do the interpretability gap dominants by the complexity gap?
2. What does a single point refer to in Figure 2? Does it refer to a pair of base SAE feature and orther SAE feature, so one base SAE feature will be represented by 8 points in the figure?
3. What is Aligned and Encoder-Decoder equal in Figure 5?
4. The conclusion that L1 is better than TopK contradicts recent work [1] where their experiments show the feature consistency in TopK is significantly higher than standard SAEs. Can you provide any explanation on what possible reason leads to the difference?

[1] Song, X., Muhamed, A., Zheng, Y., Kong, L., Tang, Z., Diab, M., Smith, V. \& Zhang, K. Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs. **Mechanistic Interpretability Workshop At NeurIPS 2025**. (2025), https://openreview.net/forum?id=d9ACURK6bI

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the stability of Sparse Autoencoders (SAEs) by analyzing whether independently trained SAEs (differing only in random seeds) discover the same features. The authors propose using the Hungarian algorithm to compute optimal bijective matchings between latents of different SAEs, allowing for a principled measure of feature correspondence. Experiments across multiple LLMs and SAE variants show that a surprisingly small fraction of SAE features are shared across seeds; overlap depends on SAE size, activation type, training time, and layer position. The paper argues SAEs produce pragmatically useful but non-universal decompositions of activation space.

### Strengths
1. Principled and novel matching approach.
The use of the Hungarian algorithm to compute the optimal matching between SAEs trained with different seeds is both reasonable and novel. The paper also validates its effectiveness by showing that matched pairs under the Hungarian algorithm exhibit consistency with cosine similarity, demonstrating that the method produces meaningful alignments.
2. Interesting empirical discoveries from multi-seed experiments.
The authors conduct matching experiments across nine independently trained SAEs and uncover several intriguing empirical patterns, such as the relationship between latent firing rate and matching frequency, and the counterintuitive finding that a significant number of “misses” (unmatched features) have high firing rates. These observations are fresh and thought-provoking for understanding SAE behavior.
3. Interpretability analysis provides practical insight.
By examining the interpretability of unpaired latents, the paper shows that each SAE training run can miss certain high-quality, interpretable features. This finding is practically important and suggests new directions for improving SAE training stability and coverage.
4. Comprehensive ablation study.
The ablation experiments—covering latent size, active latent number, architecture type, and training time—are thorough and enhance the credibility of the results. The consistent trends across these variations further support the paper’s main claims.

### Weaknesses
I couldn't identify serious weaknesses. However, I think analysis somehow remains surface-level.

While the experiments are extensive and clearly presented, the overall analysis remains somewhat superficial. The paper mainly reports empirical correlations (e.g., overlap rates, firing frequency trends) without delving into why such phenomena occur or what mechanisms underlie the observed variability across seeds. For instance, there is little attempt to characterize which properties of latents (e.g., frequency, selectivity, activation entropy) predict stability. A deeper or more mechanistic investigation would significantly strengthen the contribution.

### Questions
Maybe more theoretical analysis or intuition in SAE training should be explored in future work.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This study studies the stochasticity of which latents are found by an SAE training process. They train many different sets of SAEs which differ only in random seeds, but otherwise vary model, layer, dataset and architecture. They discuss methods for matching features between features and choose the Hungarian algorithm for generating a bijective mapping between features. Their core results show low matching between features, which may be partially mediated by architecture. Smaller SAEs show greater overlap. It is suggested that these results provide further evidence that SAEs do not uncover a "universal set of features".

### Strengths
- Quality: In some respects, this investigation is quite thorough, checking how factors including model, layer, architecture or number of latents effect the size of the common set of features found when seeds are varied. 
- Clarity: The paper is generally well written and fairly clear. Graphs are of a reasonable quality.

### Weaknesses
- Originality: The question is not new, and even very early SAE interpretability work includes examples of training multiple SAEs, varying seeds and inspecting the universality of features found (See Towards Monosemanticity - Bricken et al). The Hungarian algorithm even as applied to this problem is not novel (as mentioned by the paper). 
- Significance: Moreover, the idea that SAEs do not find a single basis has been addressed in the literature such as ("Sparse Autoencoders Do Not Find Canonical Units of Analysis" - Leask et al). 

How could the work be improved?
- Since the work is sound, the areas for improvement are likely around dealing with the "so what?" element or the "but why?" element. 
- For "so what?" - how does the stochasticity / arbitrariness of which features are found effect the usefulness of SAEs? Can multiple SAEs varying only in their seeds be more useful if used in combination? 
- For "but why?" - The use of toy models might be useful. What kinds of toy set-ups result in SAE training procedures that are non-deterministic? Possibly Toy models which include underlying features that are nested or overlap such that there is no clear ideal solution. This line of work would be more interesting if it inspires better SAE design. 

Additionally, the author could consider Matrioshka SAEs which may be more deterministic.

### Questions
- What was the most surprising result in your experiments? Where are you most changing the picture painted by the *recent* literature? 
- Did you expect feature absorption to be driving stochasticity in which features are found? What other factors might be driving this?

### Soundness
3

### Presentation
2

### Contribution
2
