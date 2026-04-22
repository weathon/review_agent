# From Isolation to Entanglement: When Do Interpretability Methods Identify and Disentangle Known Concepts?

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
A central goal of interpretability is to recover representations of causally relevant concepts from the activations of neural networks. The quality of these concept representations is typically evaluated in isolation, and under implicit independence assumptions that may not hold in practice. Thus, it is unclear whether common featurization methods—including sparse autoencoders (SAEs) and sparse probes—recover disentangled representations of these concepts. This study proposes a multi-concept evaluation setting where we control the correlations between textual concepts, such as sentiment, domain, and tense, and analyze performance under increasing correlations between them. We first evaluate the extent to which featurizers can learn disentangled representations of each concept under increasing correlational strengths. We observe a one-to-many relationship from concepts to features: features correspond to no more than one concept, but concepts are distributed across many features. Then, we perform steering experiments, measuring whether each concept is independently manipulable. Even when trained on uniform distributions of concepts, SAE features generally affect many concepts when steered, indicating that they are \emph{not} selective nor independent; nonetheless, features affect disjoint subspaces. These results suggest that correlational metrics for measuring disentanglement are generally not sufficient for establishing independence when steering. This underscores the importance of compositional and out-of-distribution evaluations in interpretability research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper evaluates Sparse Autoencoders (SAEs) on their ability to find features which "disentangle" concepts, as measured by 3 criteria: 

1) independent manipulability, meaning steering with a disentangled feature affects only the one associated concept, where the authors find all SAEs to be poorly disentangled; 
2) sparse prediction, meaning only one (or a few) features are needed to predict the presence of the concept, where the authors find all SAEs predict concepts better with k>1 features than the maximally-sparse k=1 predictor, and ReLU-based SAEs require relatively more features; 
3) disjointedness, meaning that multiple steerings with different features should combine their effects additively, which the authors find across all SAEs and language models. 

These results come from three architectures of SAEs trained by the authors on two publicly-available language models.

### Strengths
The paper addresses an important lacuna in the field of interpretability research. Prior research has focused on evaluating features in isolation, but does not consider their interactions with other features.

The paper has excellent presentation, clearly defining how it seeks to measure disentanglement, setting lower and upper bounds with the "baselines and skylines" section, and helping the reader understand the shape of its figures in the text and the "groundtruth" in Figure 4.

### Weaknesses
The paper is modest in scope. It seeks to evaluate existing techniques via new metrics, but without making the evaluations into a benchmark which could be easily used and built upon by other researchers.

Despite otherwise clear communication, the paper omits otherwise key information to reproduce its results. This includes: 
- Which layer(s) of the language model were used to train the SAEs.
- Hyperparameters of SAE training, such as number of features and amount of sparsity.
- The structure of the sentence data described in lines 126-132 (see question below).
- Which tokens in the sentences were used for training and evaluating the SAE features.

The paper's Section 4.2 is potentially exciting, but the results depicted in Figure 5 are not a sufficient test of the proposed hypothesis. In particular:

- Figure 5 appears to be results from a particular choice of concepts/features z_i and z_j. The reader is not informed which features were used to make Figure 5, and is given no evidence that this behavior generalizes to other features.

- These results are only shown via visual inspection of Figure 5, but the different magnitudes of effects make this hard to assess. For instance, in the top-middle figure, intervening on z_i produces apparently-negligible shifts in the logit for z_i, while intervening on z_j or {z_i and z_j} dramatically suppresses the prediction of z_i. But because the effect of intervening on z_j appears >100x larger than the effect of intervening on z_i, it is difficult to assess whether there is truly additivity for the combined intervention. For instance, if the z_j intervention has a -1000 logit effect, and the z_i intervention has a +1 logit effect, Figure 5 is not able to show whether the combined effect is a -999 logit effect (additive) or a -1010 logit effect (not additive).

Both of these critiques could be addressed by moving Figure 5 to an appendix and creating a new figure in its place. This new figure would be like Figure 4, showing a single numeric measurement for the interaction of each pair of concepts. The numeric measurement would aim to capture the nonlinearity of the interaction, such as LogOdds(z_i| steered by z_i,z_j)+LogOdds(z_i|no steering)-LogOdds(z_i|steered by z_i)-LogOdds(z_i|steered by z_j), with suitable normalization.

### Questions
Can you provide examples of the sentence data, as produced in lines 126-132? Are they just the abstract labels (e.g., "active, past, negative, fantasy"), full sentences (e.g., "the dark lord Sauron deceived them"), or something else? A short appendix of example sentences would make it easier to understand what data was used for these tests.

In Figure 5, some graphs have only the purple+red lines or the purple+blue lines. Is this because the purple line completely covers a red/blue lines? If so, please include a comment to that effect in the caption.

On line 242, the paper says "The gap between SAE architectures is significant and consistent across models. Top-K in particular performs well." This is not reflected in Figure 2. TopK SAEs seem to excel in the left images (domain=science), but are on par or worse than other architectures in the other four images. Is this a mistake, or based on some other analysis?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- This paper studies the ability of sparse autoencoders (SAEs) and supervised probes to identify and disentangle concept representations when concepts are correlated or entangled. 
- The authors create synthetic datasets based on natural language where they can manually control the correlation between concepts.
- They train SAEs and probes on model representations from datasets with varying levels of feature entanglement 
- they then evaluate how well learned features capture the underlying concepts via correlation, sparse probing, and independence/disjointness metrics when steering.
- The results show that interpretability methods generally degrade as concept entanglement increases.

### Strengths
- Evaluating the ability of interpretability methods to disentangle correlated concepts is a very relevant question for the field. The approach of using synthetic datasets with controlled correlations is elegant and allows interpretability methods to be evaluated on actual models with still natural-ish text.
- Diverse evaluation learned representations - feature alignment, sparse probing, steeringto measure via independence and disjointness 
- Diverse set of interpretability methods and baselines evaluated, including "organic" SAEs (trained on normal text), individual neurons, and supervised probes.

### Weaknesses
- Several load-bearing details are omitted from the manuscript. It's unclear how sampled concepts are used to construct the actual natural language sentences in the dataset. For SAEs, critical training details are missing—particularly sparsity levels and reconstruction loss. Without these metrics, it's impossible to determine whether SAE performance differences reflect poor training or actual effects of data correlations.
- The synthetic data setup is compelling, but the experiments are limited to just four categorical features, which may not capture the complexity of real concept entanglement.
- Many experimental results feel poorly presented and would benefit from more curation:
  - The sparse probing results (Figure 4) only show the effect of increasing k (number of features in the probe). It's unsurprising that performance increases with k. To align with the paper's main question, these results should show performance as a function of data correlation in the training dataset.
   - The steering results rely heavily on qualitative heatmap inspection. Aggregate quantitative metrics would be more convincing than requiring visual inspection.
  - The presentation around disjointness is unclear. The delta log odds magnitudes in Figure 5 seem problematically large (values around 4,000)—this would imply steering increases concept probability by e^4000, suggesting possible numerical issues or extremely low base rates.

### Questions
- Could you provide the missing SAE training details—specifically sparsity levels and reconstruction losses across different training conditions? This would help determine whether performance differences are due to training quality or data correlation effects.
- How are natural language sentences constructed from the sampled synthetic features? Could you provide qualitative examples of sentences in your dataset?
- Can you clarify the disjointness results in Figure 5? The delta log odds values (~4000) seem implausibly large—are there numerical stability issues, or is this due to extremely low base rates?

### Soundness
1

### Presentation
1

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
This paper studies the disentanglement of features-extractors like SAE on a variety of language models. 
The authors show, that depending on the method and number of $k$ components used to isolate a single ground-truth concept, SAE-like models tend to vary their disentanglement as measured with MCC, while still relatively underperforming probes competitors. 
Overall, this paper highlights the importance of data through in-distribution correlations between concepts when performing dictionary learning and the number of components in use to extract disentangled information from latent representations.

### Strengths
The idea of measuring disentanglement and so interpretability of SAE-like methods is timely and important to gain insight on what we can expect in practice for these methods. The insight about steering and disjointness are useful and well-explained in the paper.

The paper is well presented, and research analyses are well formulated and investigated with rigor.

Overall, the paper shows in a clear way that SAE-like methods do not come with guarantees for properly disentangling ground-truth concepts on their experimental data, and this analysis complements previous findings that many SAE-like methods do not come with guarantees about recovering interpretable features.

### Weaknesses
This paper misses comparisons to other works that recently appeared in SAE literature and treat similarly related aspects (among which identifiability), e.g. [1,2,3].  For this, I cannot say the paper excels in novelty.

Also, while MCC is a quite popular metric for studying disentanglement of representations, it has some pitfalls (since it only tests correlations) that other disentanglement metrics cover, see e.g. [4,5]. For example, DCI-ES [5] includes a training phase on a probe to detect which SAE components affect the prediction. The final disentanglement score accounts for the sparsity of the probe on SAE features. 
IRS [6] instead considers do operations within a causal framework identical to the one considered in this paper. \
Having these two metrics would be desirable for a much more in-depth analysis of disentanglement, and if not integrated, the authors should at least mention them and explain which similar aspects would be detected by using them (for example, if a feature depends on two concepts it would give a quite low IRS, because of lack of interventional independence). 

Furthermore, I found the details about the data generation vague, the presentation should further explain how data are generated and used to train the SAEs beyond Natural baselines. In this respect, the disentanglement analysis is limited to the synthetic dataset the authors use, and conclusion about disentanglement of SAEs in all, in-distribution test data for language models cannot be easily provided. 

-----

[1] Are Sparse Autoencoders Useful? A Case Study in Sparse Probing, Kantamneni et al. 2025 \
[2] Identifiable Steering via Sparse Autoencoding of Multi-Concept Shifts, Joshi et al. 2025 \
[3]  On the Theoretical Understanding of Identifiable Sparse Autoencoders and Beyond, Cui et al. 2025 \
[4] Measuring Disentanglement: A Review of Metrics, Carbonneau et al. 2020 \
[5] DCI-ES: An Extended Disentanglement Framework with Connections to Identifiability, Eastwood et al, 2023 \
[6] Robustly Disentangled Causal Mechanisms: Validating Deep Representations for Interventional Robustness, Suter et al., 2019

### Questions
I don't have specific questions for the authors, but I hope to see some discussion on the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper's goal is to design a new evaluation of the representation disentanglement properties of modern sparse decomposition interpretability techniques such as sparse autoencoder (SAE) variants. 

The authors use a synthetic dataset with ground-truth concept labels used to generate ~400k sentences of text (with ~1k sentences held out in a test set). The concepts, together with their discrete sets of possible values, are: 
- voice (active, passive)
- tense (present, past)
- sentiment (positive, neutral, negative)
- domain (news, science, fantasy, other)
The authors can also control the degree of correlation between a given pair of concepts. 

The paper proposes and carries out the following evaluations:
- compute the "mean correlation coefficient" (MCC) under increasing concept correlation between:
	- the presence of a binarized version of a ground-truth concept (so for concepts taking $n>2$ values, we create $n$ binarized concepts corresponding to each value) in the activation (we know this because we control the data generation process)
	- the activations of the SAE latent most correlated with the binarized concept
	- this is averaged over concepts, hence the "mean" in the name. Note that MCC would be more appropriately called "mean *max* correlation coefficient", because we first pick the SAE latent with max correlation. 
	- results show that SAEs generally fare worse here compared to linear probes trained with supervision (where we use the probe's logits to compute MCC)
- compute the "mean correlation coefficient" (MCC) between:
	- the presence of the same binarized concepts from above;
	- the logits of a $k$-sparse probe trained on an SAE's latents for small-ish values of $k$ (to allow for concepts encoded in multiple SAE latents)
	- results show SAEs again fall short of probes, but the gap is smaller, and top-K SAEs get to within a few percent MCC with $k\approx10$. 
- edit activations with the goal of changing the concept value encoded *without* changing the values of other concepts. Given an SAE, this works by finding the SAE latent with the highest gradient attribution to the logit of a binary probe trained on the last layer of the model (even though in this experiment the SAE is trained on activations from the middle layer). This latent is then used to do a steering-like manipulation on the activation. The logits of a probe trained to predict the concept in the activation space the SAE operates on are used to evaluate the success of this intervention.
	- Results show that activation edits work to change the targeted concepts, but have many side effects
- test for "disjointness" of concept representations, which is defined as checking if steering with two SAE latents simultaneously will result in the same change of log-odds for a concept (as judged by a probe) as steering one and then the other.
	- Results show almost perfect disjointness. However see weaknesses.

### Strengths
Overall, the paper shows that SAEs struggle compared to supervised methods along several interesting axes, and may need multiple latents to express a single concept. Other strengths include:
- a controlled setup that is nontrivial enough to be interesting while still retaining useful ground truth
- lots of relevant experiments 
- writing is clear for the most part, though there are a lot of details to the experiments.

### Weaknesses
- There is only a synthetic dataset evaluation, and the work feels overall incremental compared to prior works. 
- there are lots of different procedures to assign SAE latent(s) to a concept (at least three by my count), which could make it confusing for readers to navigate.
- In experiment 4.2., my understanding is that both the steering manipulation (equation 2, lines 314-315) and the method to evaluate log-odds (using the output of a linear probe) are linear, so it seems tautological that any features will be "disjoint" here? What am I missing?
- the steering experiment (4.1.) doesn't really involve the usual meaning of "steering" as in "generating text from an LLM while applying a steering vector to activations during the forward pass", but instead is (somewhat loosely) inspired by the idea (because of the use of the last-layer probe and attribution back to the SAE's space). Ideally, there would be a sampling-based evaluation that tells us how well steering with the SAE latents works here.

### Questions
- Is the setup in 4.2. not trivial?
- How do you think about bridging the gap from synthetic controlled datasets to more realistic ones?

### Soundness
3

### Presentation
3

### Contribution
2
