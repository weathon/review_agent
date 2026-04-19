# Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models

- Decision: Reject
- Scores: 6, 8, 5, 5

## Abstract
We investigate feature universality in large language models (LLMs), a research field that aims to understand how different models similarly represent concepts in the latent spaces of their intermediate layers. Demonstrating feature universality allows discoveries about latent representations to generalize across several models. However, comparing features across LLMs is challenging due to polysemanticity, in which individual neurons often correspond to multiple features rather than distinct ones. This makes it difficult to disentangle and match features across different models. To address this issue, we employ a method known as dictionary learning by using sparse autoencoders (SAEs) to transform LLM activations into more interpretable spaces spanned by neurons corresponding to individual features. After matching feature neurons across models via activation correlation, we apply representational space similarity metrics like Singular Value Canonical Correlation Analysis to analyze these SAE features across different LLMs. Our experiments reveal significant similarities in SAE feature spaces across various LLMs, providing new evidence for feature universality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to explore whether latent features obtained with sparse autoencoders (SAEs) are shared across different sizes within LLM families (Pythia and Gemma). The authors match individual features and broader feature spaces using rotation invariant methods.

The authors match the most correlated feature pairs and run statistical tests to compare them with random feature pairs to obtain a p-value.

My **recommendation**: Accept: 6/10 (borderline paper).

### Why not lower:
- The correlations authors find are notable.
- This paper’s methodology could be useful in answering many related research questions.

### Why not higher
- Some experiments would benefit from a stronger baseline.
- While interesting, the framing of the research question within the literature is somewhat limited.

### Strengths
## Statistical significance

The correlations found seem very statistically **significant**, and some spaces have somewhat high correlations (around 0.6). This suggests that there are indeed shared latent feature spaces across model sizes. This is a **novel** insight into scaling dynamics. The authors further find interesting correlations between individual features of 0.2 on average, which indicates some individual features are replicated across models.

## Strong correlation between known spaces

The authors include what I understand to be an analysis of the correlation between known feature spaces (ex. time nature emotions) in Table 1. This indicates that the correlations are of high **quality**, as we know the ground truth.

## Lagged correlations

As model size scales and more layers are added, the highest correlations in gemma models are between spaces early in the smaller model and later in the bigger model. This is an interesting insight, as it suggests that the increased number of parameters could be spent in the earlier layers computing (possibly) low level features. This could be an interesting future direction.

## Feature transfer in AI alignment.

The implication that known feature spaces are preserved across model scale is **significant** for broader AI alignment goals, such as the possibility to transfer alignment techniques from a model of generation N to the next generation N+1.

### Weaknesses
## Weak baseline

Authors take random feature pairs as a baseline. This is not ideal as a baseline. Comparing the most correlated pair of activations with random pairs is expected to yield low p-values. If you have 100 random pairs of activations (here there are more), and you take the most correlated pair it will yield a p-value of 0.01. Here there are many more, and some are bound to be (spuriously?) correlated.

Authors may take the most correlated pairs of features in models that have nothing in common as a baseline. If the high correlations still show up, then they are spurious and caused by the sorting itself. An example of models with nothing (or little) in common might be LLMs doing very specialized tasks in different languages. Alternatively, the authors could inspect the relevant pairs to see if the correlations are genuine, which I see is somewhat done by taking the mean correlations of known features in Table 1. This somewhat eases my concerns with the methodology.

## Lack of comparison between families

While this is acknowledged, and comparison between different model scales is very interesting, only comparing models within the same family somewhat limits the scope of the findings. It would be interesting to inspect how similar families are (for ex. a big Pythia vs Gemma 2B).

## Few model families

As far as I have read, the authors compare some layers for 2 models from each of the Pythia/Gemma model families. While it is acknowledged that this is due to there only being 2 model families for which open-source SAEs are available, they are still a somewhat limited sample of SAEs to experiment on. While there are no other model families for which different sizes have SAEs, there are other LLMs for which there are SAEs, and it would be interesting to experiment on them.

### Questions
Have you tried the Llama SAE?

Have you tried more layers?

Have you tried comparing across families, for example, a large Pythia with Gemma 2B?

Have you tried comparing unrelated models as a baseline?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates feature universality in large language models by analysing similarities between sparse autoencoder (SAE) representations across different models. The authors introduce a methodology for comparing SAE feature spaces using correlation-based feature matching followed by rotation-invariant similarity metrics (SVCCA and RSA). The work presents evidence for universal features across model scales within architecture families and finds particular similarity in semantic subspaces and middle layers.

### Strengths
1. Clear methodology for comparing SAE spaces, with careful handling of both permutation and rotation invariance issues
2. Systematic approach to noise reduction through filtering non-concept features and many-to-1 mappings
3. Results show consistent patterns across model scales, particularly in middle layers
4. Findings about middle layer similarity align with existing work on model steering
5. Results on semantic subspace similarity provide interesting insights into how models represent concepts

Overall, I think it's a really important piece of work that was sorely needed. Universal feature spaces provides some good evidence for a weak form of the linear representation hypothesis, which may give us encouragement that SAEs are the right level of abstraction to think about atomic units of model computation.

### Weaknesses
1. Limited comparison to alternative methods. The paper does not consider using Maximum Correlation Coefficient (MCC) with Hungarian matching on either decoder weights or latent activations. This would provide a simpler, provably optimal matching baseline against which to compare their approach. (I know MCC is optimal up to sign flips and rotations, but I'd still like to see how this looks as a very simple approach.)

2. Experimental limitations:
   - Comparisons limited to within-family pairs (Pythia and Gemma)
   - Limited cross-architecture analysis to support broader universality claims (however, I understand they only compared MLPs because they couldn't properly compare residual streams across architectures due to different ways of combining MLP and attention outputs)
   - Small dataset size (100 samples) for statistical testing
   - Limited ablation studies on filtering choices

3. Methodological questions:
   - Selection of non-concept tokens appears somewhat arbitrary
   - Choice of semantic categories lacks rigorous validation
   - Top-5 activation analysis may be insufficient for feature characterisation - typically, people sample a spectrum of activations and analyse these e.g. see Templeton et al. (2023). There is inherent bias in looking at only the top activating samples
   - Statistical significance testing could be more rigorous (multiple hypothesis testing not addressed)

4. Missing details:
   - SAE training parameters
   - Computational requirements and scalability
   - Complete implementation details for similarity metrics
   - Thresholds for filtering decisions
   - You're missing a couple of particularly relevant citations. Others have explored similar feature alignment studies, particularly across different SAEs. For instance, see https://arxiv.org/abs/2408.00657#:~:text=Sparse%20autoencoders%20(SAEs)%20have%20shown,effectiveness%20in%20disentangling%20semantic%20concepts.


MINOR COMMENTS:

1. Abstract claims "rigorous" analysis - let readers judge this
2. "Comprehensive experiments" overstates the scope of testing
3. Some figures could be more clearly labeled
4. Statistical significance discussion needs more detail

### Questions
1. Have you compared your correlation-based matching to MCC with Hungarian matching (on either decoder weights or latent activations)? This would provide a principled baseline.

2. How sensitive are the results to:
   - Choice of non-concept tokens for filtering?
   - Number of top activations used for feature analysis?
   - Feature neuron count in the SAEs?

3. Would the findings generalise across different architecture families? Cross-architecture comparisons would strengthen universality claims. I'd even be interested to see the results repeated with something like Llama, which I understand has a bit of a different setup to Gemma and Pythia, even in things like positional embeddings. If you could replicate these findings, it would be good 

4. Can you include a bit more of a general discussion on the linear representation hypothesis? I know it can become difficult to not be too philosophical, but this is essentially what you're testing. Giving the reader some grounding on why we think SAE features might be able to learn the atomic units of model computation really shapes the contours of your work.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper investigated whether SAEs trained on different SAEs learn similar features. They apply techniques that try to both match SAE features directly, and also techniques which find similar subspaces. The paper compares an SAEs trained on Pythia 70m to SAEs trained on Pythia 160m, as well as comparing Gemma-2b SAEs to a Gemma-2-2b SAEs. The authors find evidence that at middle layers of the model SAEs learn similar representations.

### Strengths
The techniques the paper uses, SVCAA and RSA, seem to be novel when applied to SAE features. The paper compares their results to a baseline of randomly pairing up features and subspaces. Comparing SAEs from different models is also not well studied and is a good domain to explore.

### Weaknesses
The paper claims to find universal representations across SAEs from a wide variety of LLMs, but only looks at SAEs within the same LLM family (Pythia to Pythia, and Gemma to Gemma) which likely share a similar architecture and training corpus. It is also not clear what specific SAEs are used aside from saying they are loaded using SAELens. It is likely that the Gemma-2-2b SAEs are from Gemma Scope, but Gemma Scope is not cited and Gemma Scope has a large number of SAEs for every layer of the model with different L0 values. The techniques only seem to offer statistically significant improvements over randomly selecting SAE features in the middle layers of the model. It is unclear if the techniques failing to work in early and later layers of the model is evidence of problems with the technique or evidence that the SAEs are learning different representations. It seems surprising to me that very similar LLMs like Gemma 1 and Gemma 2 have completely unrelated representations of concepts except in middle layers.

The case the paper makes would be bolstered by comparing different SAEs trained on the same model (e.g. the multiple SAEs per layer in Gemma Scope), or multiple SAEs trained with different seeds, to rule out issues with the technique itself. The claim of universality also requires comparing SAEs across model families, not just within a model family.

### Questions
- In the paper, it says "features may be similar relation-wise across spaces, but not rotation-wise due to differing basis". What does it mean to be similar relation-wise? Is this referring to matching feature neurons directly?
- "layer 0 contains few discernible, meaningful, and comparable features" - why would layer 0 contain few meaningful features? layer 0 is the token embeddings, shouldn't embeddings contain meaningful features?
- In figures 2 and 3, does a score of 0.2 for SVCCA or RSA mean that only 20% of feature subspaces match across SAEs according to the metric?
- Why do RSA and SVCCA give such different values for similarity for the same model and same layers?
- In section 4.3, it says "We find that for all layers and for all concept categories, Test 2 described in §3 is passed. Thus, we only report specific results for Test 1 in Tables 1 and 2". Does this mean test 2 failed? I'm confused what this means.
- Section 4.3 mentions non-semantic subspaces. What are non-semantic subspaces? How do you know if a subspace is semantic or non-semantic?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper conducts exploratory experiments to measure representational similarity between the overcomplete bases found by sparse autoencoders (SAEs) across different LLMs and layers therein. 

Specifically:
- suppose we are given two SAEs trained on residual stream activations (possibly coming from different models and/or layers within the model) which have the same number $d_{hidden}$ of hidden latents
- consider the decoder matrices $D_1,D_2$ of these SAEs, which hold the $d_{hidden}$ vectors in the overcomplete basis the SAE uses to sparsely code residual stream activations
- pair up the SAE latents: given a latent in one SAE, pair it up with the latent from the other SAE that has the highest correlation of activations over some set of texts (there are some technical details on how to clean up this mapping to make it injective);
- after this pairing we have two "aligned" matrices of latents $D_1',D_2'$, to which we apply two representation similarity methods:
	- SVCCA (Singular value canonical correlation analysis), which first takes the left singular vectors of $D_1', D_2'$, and then runs canonical correlation analysis on these two sets of vectors, iteratively finding linear combinations of vectors on either side that maximize correlation (equivalently dot product) and removing the pair. The final measure of correlation is the average of the correlations of all pairs;
	- RSA (Representational similarity analysis): here we first build a distance matrix over the vectors in $D_1'$ (and similarly $D_2'$), where the $ij$ entry is the Euclidean distance between the $i$-th and $j$-th latent vectors. Then, the similarity of the two resulting distance matrices is evaluated via Spearman (rank) correlation.
- beyond evaluating the correlations as above, their statistical significance is measured via p-values against a random pairing of the original SAE latents $D_1,D_2$ designed to simulate a null condition.
- these experiments are ran on pairs of SAEs trained on different Pythia models (70M and 160M) and their layers, as well as pairs of SAEs trained on different Gemma-2B model versions (1 and 2) and their layers.

The authors report some findings, such as achieving statistical significance under their null test for most pairs of SAEs, and finding the middle-layer SAEs are most similar.

### Strengths
- The paper studies an underexplored problem, the extent to which SAE latents are universal across models
- The paper proposes some interesting metrics for SAE similarity, such as representational similarity analysis (RSA)
- The result that pairs of middle-layer SAEs are most similar is mildly interesting

### Weaknesses
- The results are exploratory, and there is no strong takeaway for the interpretability research community. 
	- The overall conclusion seems to be that "SAE latents are kind of similar between layers/models", but it is difficult to contextualize the magnitude of the effect. It would be valuable if there is a comparison to similarity in the neuron basis. Currently, it is unclear what is gained by asking the question of SAE latents as opposed to original neurons in the model.
	- It is unclear whether comparing SAE latents across different layers is an interesting question; the main motivation for studying universality is to assure us that training SAEs on a different model using a different dataset sample will discover "the same" concepts *in aggregate*. In particular, the current consensus in the field is that early, middle and late model layers all have distinct roles in computation and likely represent distinct concepts (also see some recent work for why trying to align SAE latents in faraway layers "naively" may be the wrong approach https://transformer-circuits.pub/2024/crosscoders/index.html)
- The use of p-values to quantify the significance of the results is not very illuminating, because even small effect sizes can often achieve small p-values. 
	- In particular, the results on low p-values of SVCCA are perplexing. SVCCA is permutation-invariant, so whether or not features were paired together beforehand (vs randomly paired as in the null condition) should not matter for the final result. I suspect the low SVCCA p-values may be due to the post-processing step after feature pairing that prunes the mapping to make it better behaved.
	- Regardless of this issue, it is altogether not surprising the p-values are low for the other method (RSA). 
- The Methodology section is not very clearly written

### Questions
- Have you looked at the latents that get paired between SAEs in the same (or similar) layers using your approach based on activation correlations? Do they seem to be interpretable? Do they have similar interpretations? 
- Why should we be interested in measuring the similarity between SAEs trained on distant model layers? In particular, when you look at the paired features across distant layers, do they seem to have similar interpretations?
- What baseline similarity should we expect between different SAEs random seeds for the same training set of activations, the same model and the same layer? Similarly, what should we expect when any of these are made different instead of the same?

### Soundness
3

### Presentation
3

### Contribution
2
