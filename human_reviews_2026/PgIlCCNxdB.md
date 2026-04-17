# The Mind's Transformer: Computational Neuroanatomy of LLM-Brain Alignment

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 8, 8

## Abstract
The alignment of Large Language Models (LLMs) and brain activity provides a powerful framework to advance our understanding of cognitive neuroscience and artificial intelligence. In this work, we zoom into one of the fundamental units of LLMs—the transformer block—to provide the first systematic computational neuroanatomy of its internal operations and human brain acitivity during language processing. Analyzing 21 state-of-the-art LLMs across five model families, we extract and evaluate 13 distinct intermediate states per transformer block—from initial layer normalization through attention mechanisms to feed-forward networks (FFNs). Our analysis reveals three key findings: (1) The commonly used hidden states in LLMs are surprisingly suboptimal, with over 90\% of brain voxels in sensory and language regions better explained by previously unexplored intermediate computations; (2) Different computational stages within a single transformer block map to anatomically distinct brain systems, revealing an intra-block hierarchy where early attention states align with sensory cortices while later FFN states correspond to association areas—mirroring the cortical processing hierarchy; (3) Rotary Positional Embeddings (RoPE) specifically enhance alignment along the brain's auditory processing streams. Per-head queries with RoPE best explain 74\% of auditory cortex activity compared to 8\% without RoPE, providing the first neurobiological validation of this architectural component in LLMs. Building on these insights, we propose MindTransformer, a feature selection framework that learns brain-aligned representations from all intermediate states. MindTransformer achieves significant brain alignment performance, with correlation improvements in primary auditory cortex exceeding gains from 456× model scaling. Our computational neuroanatomy approach opens new directions for understanding both biological intelligence through the lens of transformer computations and artificial intelligence through principles of brain organization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates how well each component of a transformer block (e.g., "per-head query", "per-head query with RoPE", "FFN Activated State", "FFN Output") provides representations to predict brain activity during natural language processing. The work systematically compares these components across a large variety of model sizes and model families. It finally shows that combining them provides better brain scores than simply using the commonly used hidden states.

### Strengths
- The work investigates a large number of models from five model families, with sizes ranging from 270M to 123B parameters. The two largest models, above 100B parameters, are relatively rare in this type of literature, and not easily computationally accessible for many researchers, which could be of interest to the community.

- Another strength is the systematic approach the authors propose in studying all the different components of each transformer block.

### Weaknesses
There are several significant limitations that lead me to reject the paper.

- The work compares different representations that do not have the same number of features, which leads to unfair comparisons. Indeed, all else being equal, the brain predictivity increases with the number of potential regressors. Here, all the components in a transformer block do not have the same number of features, and this can vary considerably: in particular, the hidden size of the MLP block (called "ffn activation state" here) is usually much larger than the hidden size (e.g., 3x for Qwen3 0.6B, 4x for LLama 3.2 1B, or even 5x for Qwen3 32B). Unsurprisingly, it is indeed this particular component that wins most of the times (see Fig 1 and Fig 5b). One solution to provide fair comparisons might be to equate the dimensions of all the vectors, either by applying a PCA or by randomly dropping some features. For the same reason, the fact that combining all the different features (which the authors call the "Mind's Transformer") leads to better results is not very surprising nor insightful at this stage. Moreover, previous work has investigated proper ways to combine various features: see Dupré la Tour et al. (2022).

- Another important limitation of the work is the absence of proper baselines. Previous work (see e.g. Schrimpf et al., 2021; Pasquiou et al., 2022; Bonnasse-Gahot & Pallier, 2024) has shown that random baselines or untrained models can achieve pretty high brain scores. Here, it is necessary to compare the results with proper baselines such as random vectors, random embeddings, or untrained networks. For instance, in the auditory cortex, simply having random embeddings that can be used to track some acoustic property such as speech rate can be enough to actually yield a brain score that is as good as a pretrained LLM, given that the LLMs used here are fed with text rather than speech.
This is notably particularly important for the authors' main claim of improvements in brain predictivity in the primary auditory cortex (see also below). Random baselines should put these results in perspective, notably when comparing the different brain regions.

- The work does not in general show all full values, but presents a winning ratio instead. This is quite underspecified, as one component could "win" but by a very small margin. This is important to assess whether it is really worthy to consider these different components. This issue, combined with the two previous ones, makes it impossible to extract insights from the current results. It would be better to simply (or also) present the raw values (provided the comparisons are fair and use proper baselines): brain scores as a function of size, for the different models, different families, different components, and possibly different brain regions.

- One of the main claims of the authors is that their "framework" achieves improvements in brain predictivity in the primary auditory cortex that "substantially exceed gains from 456$\times$ model scaling". This claim is problematic for several reasons. 
First, although the human participants hear the story, the LLMs are fed with a textual version. The fact that there is not a high brain score in the auditory regions is therefore not surprising. Note that this does not mean that this is an "unresolved challenge in the field" as the authors claim. Acoustic language models are indeed able to well predict brain activity in the auditory cortex (see Tuckute et al., 2024; Millet et al., 2022; Antonello et al., 2023). Moreover, comparing against model scaling in these areas does not make much sense, as low-level features are already well-predicted with small models, larger models providing only marginal gains in these sensory regions (see Antonello et al., 2023, for speech; see also similar findings by Raugel et al., 2025, for vision). Finally, the improvements might be based on using the onsets of the words, which helps modeling the acoustic envelope or speech rate, which might lead to an increase in brain score in the auditory cortex. In this view, it is then not surprising that the higher gain is obtained when considering the component just after RoPE, which allows the brain fit to take advantage of the position of the words. Once again, this underscores the need to compare with a proper baseline. The outcome might not be as important as the authors claim, and may be attributable to unsurprising low-level reasons.

Minor comments

- The abstract states that the fundamental unit of a large language model is the transformer block. However, not all LLMs are based on Transformers: see for instance models based on the Mamba architecture.

- Fig. 4: I assume the size of the marker is related to the size of the LLM, but this information is already provided by the x-axis, so it does not bring anything new and obscures the reading of the figure.

- Given the intention of the authors to systematically investigate all components of the transformer block, it might be interesting to also examine the representation before and after the gating for models such as Gemma with gated activation functions.

### Questions
In brief (see Weaknesses section for more detail):
- What are the results when comparing the different components using an equal number of features?
- What are the raw values of the brain scores, and how do they compare across model sizes, families, and brain regions?
- What are the correlations (and differences between conditions) yielded by proper baselines?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors dive deep into the transformer block, pulling out 13 intermediate states across 21 LLMs to map them onto fMRI brain activity during story listening. They claim hidden states are surprisingly insufficient (>90% of voxels do better with intermediates), and aligning intermediate states to brain activity reveals an intra-block hierarchy. They have also demonstrated the importance of RoPE. They wrap it up with MindTransformer, a two-stage selector that beats 456× scaling in A1.

### Strengths
It’s the first time someone’s looked at *all* intermediate states at this scale, and the intra-block hierarchy is neat. The RoPE-auditory link is interesting if it holds up. The scale is impressive—21 models, 13 intermediate states.

### Weaknesses
1.Some critical statistical rigor is missing. 
2.Averaging brain activities across subject may introduce information loss.

### Questions
1.In the encoding model, statistical control is missing. Brain activity of each voxel is predicted by building a regression model, however, not all the predictions are significantly meaningful. It is expected to perform statistical control to determine the voxels whose neural signal can be successfully predicted by the encoding model. Is the correlation between predicted activity and ground-truth statistically above chance level? It also would be helpful to perform noise ceiling validation.

2.In the experiment, fMRI data were averaged over subjects. This operation may help to decrease noise. However, it is unknown that the findings reported in this study are reproducible in different subject. The authors are encouraged to provide additional results for individual subject.

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
5

### Summary
The paper investigates the alignment of different states in LLMs' (n=21) blocks to brain activity on the *Le Petit Prince fMRI Corpus*.
The main claims are that
(1) the typical hidden states used by the field are suboptimal,
(2) the stages within a block reveal a sensory-to-association hierarchy,
and that (3) RoPE improves alignment with auditory streams.
The authors combine these findings into a brain model termed MindTransformer which uses a learned feature retrieval and for which they claim SOTA brain alignment.

### Strengths
1. The study is systematic in evaluating various models, testing 21 LLMs that range in scale from 270M to 123B parameters and stem from various families including LLaMA, Qwen, Mistral, GPT, and Gemma.

2. Improvements in brain alignment with RoPE seem solid and are novel as far as I am aware.

3. The MindTransformer framework nicely ties together the selection of feature states for potentially improved brain predictivity.


Further strengths:
* Figures are clear, text is straightforward to follow
* The claim for a block-intermediate state hierarchy corresponding to the cortical hierarchy is interesting and, to me, unexpected. But I have concerns about its validity (see below)
* I also highly appreciate papers that improve the state-of-the-art models on brain function, but I think this needs to be more rigorous (also below) 

Source code available which is great for reproduction!

### Weaknesses
### Claiming SOTA via training on test
My biggest concern is that the MindTransformer framework seems to select intermediate states on the test set itself. There is no mention of a separate training and validation split, such that the predictive representations are chosen via the test set. This would also yield high alignment for a random input feature set. There needs to be separate validation *after locking down the selected states*, which I think you could do via additional datasets (below).

### Single dataset
Evaluation on a single dataset is too limited for claiming SOTA. There are several publicly available language brain recordings, e.g. the set of Brain-Score Language benchmarks. What is the alignment of the (frozen) model on those?

### Hierarchical claim not well supported
For Claim 2, I'm not actually sure the hierarchy is actually there. It's hard to tell from Figure 2, but for e.g. GPT-oss it seems that STG has a higher contribution of later block stages than the higher-level IFG and angular gyrus. Please quantify the actual correspondence between within-block hierarchy and cortical hierarchy, e.g. by correlating one with the other per model.


Minor:
* L043 related work should include Toneva et al. 2019, Schrimpf et al. 2021, Caucheteux et al. 2022, and Goldstein et al. 2022.
* L318 "Among all intermediate states examined in our computational neuroanatomy analysis, the per-head query with RoPE provides the most substantial and systematic improvement in brain alignment" -- isn't the corresponding 13.03% winning ratio lower than e.g. the FFN activation at 14.86%?
* Figure 3 / section 4.2: missing stats on RoPE improvements. Effect size seems pretty clear though.

### Questions
See Weaknesses above (Minor not that important).

I am happy to increase my score if the generalization is shown, ideally on new datasets and without selecting states again.

### Soundness
2

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper explores how internal computations of transformer-based LLMs align with brain activity during naturalistic language processing. Rather than using the usual hidden states, the authors decompose each transformer block into 13 intermediate computational stages and test their correlation with fMRI responses from the Le Petit Prince dataset. They show that most brain voxels are better explained by these internal states than by the standard representations, that early attention computations align with low-level sensory areas while later FFN states align with association cortex, and that Rotary Positional Embeddings (RoPE) strongly improve alignment in auditory regions. Building on this, they propose MindTransformer, a framework that automatically selects or integrates the most brain-aligned states to improve prediction. The work is carefully executed and clearly presented. The methodology is solid and the figures are convincing; results seem plausible and the code availability supports reproducibility. The main limitation is the interpretation of correlation as evidence of shared computation remains debatable. Overall, this is a strong and well-written paper with interesting insights connecting transformer architecture and brain organization. While not revolutionary, it represents a meaningful methodological and interpretive step forward.

### Strengths
This work goes beyond previous studies by going deeper into the architecture of Transformers, allowing better results.

### Weaknesses
The inherent method of comparing brain activity with transformer activity by correlation might be debatable.

### Questions
Do you expect MindTransformer to generalise to vision ou multimodal brain datasets ?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors investigated whether internal computations inside transformer blocks, not just the usual layer hidden states, better align with human brain activity during naturalistic language comprehension, and whether specific architectural choices (notably RoPE) have distinct neurobiological correspondences. It introduces a “computational neuroanatomy” mapping from 13 intra-block states to brain voxels and proposes a feature-selection framework (MindTransformer) to exploit these states for brain prediction.

fMRI from Le Petit Prince (English; 49 native speakers; ~100 minutes). Signals are group-averaged; ROIs include Harvard–Oxford parcels and a language localizer set. Twenty-one open-weight LLMs across 5 families are analyzed. For each transformer block, the authors extract 13 states. 

The authors found that >90% of voxels in language/sensory regions are better predicted by previously unused intermediate states (vs. standard hidden/context). Early attention states align with low-level sensory cortex; later FFN states with association areas, revealing a fine-grained hierarchy within blocks (not just across layers).

### Strengths
1. The intra-block, 13-state decomposition is principled and broadly applicable; the “winning ratio” analysis shows strong evidence that representation choice dictates which brain systems one can model. 
2. The sharp auditory-stream improvement for per-head Q+RoPE (and the contrast without RoPE) is novel and interpretable, linking an computational component to neurobiology.
3. MindTransformer is simple (ridge + feature selection) yet yields sizable, ROI-specific improvements that outstrip very large model-scaling gains—important for resource-efficient brain modeling.

### Weaknesses
1. The paper averaged BOLD time series over multiple participants, which boosts SNR but obscures individual variability and may inflate voxel-wise correlations; it’s unclear how robust the intra-block specializations are at the single-subject level. Ability to conduct analysis on individual subject is crucial for future application.

2. The RoPE claim is framed as “neurobiological validation,” but evidence is correlational from encoding models; causal or ablation-style tests on models/representations (beyond contrasting with/without RoPE states) would strengthen the claim.

3. Scope beyond language/auditory. The strongest effects are in low-level auditory cortex; improvements in classical high-level language ROIs are modest, and it’s not fully explored why FFN/attention integration doesn’t translate to larger gains there.

### Questions
1. How do the “winning state” maps and MindTransformer gains look when models are trained/tested per subject (or with mixed-effects statistics) rather than on the group-average? Any meaningful variance across individuals? 
2. Do results persist with subject- or voxel-wise HRFs, or FIR bases capturing variable latencies, especially for early auditory regions where timing is critical? 
3. For the RoPE result, I wonder what would happen if you repeated the analysis with otherwise-identical models differing only in positional encoding (e.g., learned absolute vs RoPE) to isolate architectural effects from family-wise confounds.
4. In the multi-state model, which specific features/states dominate the selected top-k set in different ROIs? A stability analysis (e.g., across folds and models) would clarify interpretability and guard against overfitting. 
5. The language-network average improvement is ~2–3%. Do certain layers/states (e.g., FFN activation vs key/query) dominate there, and can targeted combination rules outperform the generic concatenation + top-k approach?

### Soundness
4

### Presentation
3

### Contribution
3
