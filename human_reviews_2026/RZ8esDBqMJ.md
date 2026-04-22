# A tale of two tails: Preferred and anti-preferred natural stimuli in visual cortex

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 8, 4, 2, 6, 6

## Abstract
An ongoing quest in neuroscience is to find the preferred stimulus of a sensory neuron. This search lays the foundation for understanding how selectivity emerges in the primate visual stream---from simple edge-detecting neurons to highly-selective face neurons---as well as for the architectures and activation functions of deep neural networks. The prevailing notion is that a visual neuron primarily responds to a single preferred visual feature, like an oriented edge or the shape of an object, resulting in a 'one-tailed' distribution of responses to natural images. However, surprisingly, we instead find 'two-tailed' response distributions of primate visual cortical neurons, suggesting that these neurons have both preferred and anti-preferred stimuli. We experimentally validated anti-preferred stimuli by recording responses from macaque V4 to model-optimized stimuli. We find that these anti-preferred stimuli are important for describing a neuron's tuning, as both preferred and anti-preferred images are needed to predict a neuron's responses to natural images. Moreover, in a psychophysics task, humans rely on anti-preferred images to interpret and predict V4 stimulus tuning; this was not the case for internal units from a deep neural network. Interestingly, we find no discernible differences in image statistics between preferred and anti-preferred images. This suggests that by encoding anti-preferred features, a V4 population seemingly doubles its capacity for feature selectivity, allowing for a more flexible downstream readout. Overall, we establish anti-preferred stimuli as an important encoding property of V4 neurons. Our work embarks on a new quest in neuroscience to search for anti-preferred stimuli along the visual stream and offers a new perspective on how feature selectivity arises in the visual cortex and deep neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the representation of “anti-preferred” visual stimuli—images that suppress neural activity rather than increase it—in the macaque visual system. Using electrophysiological recordings, data-driven modeling, image synthesis, and human psychophysics, the authors report that many macaque V4 neurons show two-tailed response distributions, whereas units in standard deep neural networks (DNNs) display one-tailed, ReLU-like activation patterns. This indicates that cortical neurons can encode information through both excitation and suppression, while DNN units represent information only through positive activations. The study also proposes a modified readout that combines pre-ReLU features before a nonlinearity, improving predictions of neural responses. The manuscript is clearly written, well organized, and presents high-quality figures.

The paper’s main strengths are its clear presentation, strong experimental design, and integration of neuroscience and computational modeling to address an important question at the intersection of biological and artificial vision. The main weaknesses concern the fairness of the readout comparison, the interpretation of the pruning and psychophysics analyses, and limited discussion of prior and related work, as well as the lack of broader connections to artificial vision systems.

### Strengths
The study is technically sound and clearly presented. It addresses an important question at the intersection of neuroscience and machine learning by contrasting biological and artificial visual representations. The results suggest that two-tailed neural tuning may provide a richer representational basis than the strictly positive codes used in DNNs. The experiments are well executed, and the integration of electrophysiology, computational modeling, and psychophysics is well thought out. The figures are clear and effectively illustrate the main results.

### Weaknesses
The comparison between the proposed readout and the standard linear mapping could be made clearer. While the authors compare against standard post-ReLU readouts, it is not fully clear whether a capacity-matched baseline—applying the same operations (1×1 mixing, LayerNorm, ReLU) after the ReLU—was included. Such a control would help determine whether the observed advantage truly reflects the benefit of pre-ReLU mixing, which allows access to both positive and negative features, rather than increased model flexibility due to the additional operations. Including or explicitly discussing this control would make the interpretation more convincing.

The image synthesis procedure is described clearly, but the discussion could address the limitations of optimizing stimuli far from the natural image distribution. Several recent approaches, such as diffusion-guided activation maximization or Fourier-phase–constrained optimization, maintain proximity to the natural image manifold while exploring feature space. Citing and briefly discussing such work would situate the present synthesis method within the broader literature.

The data-pruning analysis requires a more careful interpretation. The authors conclude that including anti-preferred images is essential for estimating neural tuning, but this improvement may result from greater coverage of the response range rather than from the special status of anti-preferred images. For neurons with sparse selectivity, non-preferred images still provide valuable contrast for estimating tuning, even if there are no distinct anti-preferences. Acknowledging this possibility would clarify the mechanism underlying the observed effect.

Similarly, the psychophysics results are informative but should be interpreted with caution. It is unclear whether the benefit of including anti-preferred images was specific to neurons with two-tailed response distributions or whether any inclusion of low-response examples helps learning. The results from DNN simulations suggest that the latter could be true. If so, the human data may demonstrate a general advantage of diverse examples rather than direct evidence for two-tailed encoding. Rewording this section to reflect this distinction would improve clarity.

The work would also benefit from a more complete engagement with prior neuroscience literature. Selective suppression has been described previously in both V1 and V4. In addition, a recent preprint (bioRxiv: 2025.07.16.665209) reports closely related findings on the role of suppressive features in shaping visual representations. Discussing this work and clarifying how the present study differs in methodology and conclusions would strengthen the positioning of the paper within the current literature.

Finally, given that this paper is submitted to ICLR, the discussion could better connect the findings to implications for artificial vision systems. Two-tailed selectivity may suggest useful design principles for artificial models, such as incorporating balanced excitatory and suppressive representations to improve robustness or generalization. A short discussion of such implications would make the paper more relevant to the machine learning community.

### Questions
Was a capacity-matched control readout tested—for example, a variant applying the same mixing and normalization operations after the ReLU—to isolate the specific contribution of pre-ReLU access to negative features?

Can the authors clarify whether the psychophysics effects were stronger for neurons with two-tailed response distributions, or whether similar effects occurred for sparse or one-tailed neurons as well?

How robust are the findings to the specific image synthesis method used? Would on-manifold synthesis (e.g., diffusion-guided optimization) produce comparable preferred and anti-preferred examples?

How do the authors view the implications of their results for representation learning in artificial systems? Could introducing bidirectional or opponent-like feature tuning improve generalization or interpretability in DNNs?

Could the authors comment on how their findings relate to the recent bioRxiv preprint (2025.07.16.665209) reporting suppressive feature selectivity in visual cortex?

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
3

### Summary
This is a very interesting paper that combines unique experimental data with modern computational techniques to fundamentally make the a potentially groundbreaking observation about neurons in the visual cortex - that is their tuning properties cannot just be characterized by what they prefer (i.e., are excited by), but there is an independent set of stimuli that effectively suppress the activity of a neuron and it is important to identify these anti-preferred stimuli. They show various results to support this conclusion and present a new tool for neuroscientist to use in order to efficiently identify preferred/anti-preferred stimuli. I enjoyed the paper and I think it has important scientific contributions to offer. I don't believe, however, the ICLR venue is the right audience. I know that with a liberal interpretation, one can consider this paper in the scope of the conference. I believe the ICLR paper should either push ML forward or show a new application of an ML method. I think this paper might be considered in the latter category, but the main message is squarely in neuroscience.

### Strengths
Solid experimental results. 
Very useful and impactful neuroscientific insights. 
A public tool for other neuroscientists to use in order to effectively find anti-preferred and preferred stimuli.

### Weaknesses
The main weakness in my opinion is that there is little contribution to core or foundational ML. There is also no meaningful innovation on the applied side, where an existing ML tool is applied to a novel problem or a state-of-the-art result is achieved on some standard problem. I also found the style of writing and the format and flow very confusing. It follows the typical general-audience high impact journal format (like Nature), where details of methods and experiments are buried in the back of the paper. To me, a good ICLR paper highlights these up front. In fact, those are supposed to be main contributions. 

I found Section 6 to be unconvincing and confusing. I really counldn't follow how the results amounted to the conclusion that "knowing a neuron’s preferred images gives little information about what a neuron’s anti-preferred images will be"

### Questions
1) Can you spell out what novel insights you can offer for the ML community?

### Soundness
3

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
3

### Summary
This paper overturns the traditional “one-tail” view of visual neuron selectivity by showing that many macaque V4 neurons exhibit two-tailed responses, they react strongly to both preferred and anti-preferred stimuli. These anti-preferred images are rich and structured, not blank, and are essential for accurately modeling neural tuning. Incorporating both stimulus types improves encoding performance and aligns with human perception, unlike ReLU-based DNNs that show only one-tailed selectivity. The authors also introduce ImageBeagle, a scalable tool for discovering preferred and anti-preferred stimuli across millions of natural images.

### Strengths
* This work provides an interesting analysis of the preferred and anti-preferred stimuli from macaque V4 region
* The modeling and analysis are robust. Author use multiple experiments: encoding models, data pruning experiments, and psychophysics tasks to show that both preferred and anti-preferred stimuli are essential to fully capture a neuron’s tuning curve.
* The author released ImageBeagle, an efficient large-scale natural image search tool. Provides practical value to visual neuroscientists interested in optimizing neurons’ responses

### Weaknesses
* The discovery of “two-tailed” response distributions is presented as novel, but prior work has long recognized that neurons exhibit both excitatory (maximizing) and suppressive (inhibitory) responses to different stimuli. The paper does not clearly distinguish how its findings go beyond this established understanding.
* The comparison with linear-nonlinear (LN) models and task-driven DNNs is somewhat misleading. Their one-tailed response patterns largely reflect architectural constraints (e.g., ReLU activation), not an intrinsic inability to model two-tailed selectivity. More recent encoding models [1] can predict both positive and negative response values, so the contrast with “biological two-tailed neurons” is overstated.
* The neuron tuning analysis relies on a single model type and uses only the R² metric to assess performance. This is insufficient to robustly support the claim that including anti-preferred stimuli improves tuning prediction. Additional analyses on more encoding models and metricstrained with the same set of data would strengthen the evidence.	
* Although the paper analyzes anti-preferred images from several perspectives—neural recordings, modeling, and psychophysics—the overall impact of these analyses is limited. Beyond showing that incorporating both preferred and anti-preferred images can improve encoding model performance in low-data regimes, the study provides little insight into how anti-preferences fundamentally contribute to neural computation or representation.

[1] Aria Y Wang, Kendrick Kay, Thomas Naselaris, Michael J Tarr, and Leila Wehbe. Better models of human high-level visual cortex emerge from natural language supervision with a large and diverse dataset. Nature Machine Intelligence, 5 (12):1415–1426, 2023. 2

### Questions
* The improved tuning performance using both preferred and anti-preferred images may mainly result from greater response diversity, instead of the features of the images. Random images yield weaker predictions likely because their responses are less extreme, but outperform when more data are available. The claimed connection between this result and the neuron’s tuning, specifically that preferred and anti-preferred images reveal identifiable visual features learned by the model requires further verification.
* Could the authors clarify what is being predicted in this analysis? Are they training a model to predict the activation responses of individual ResNet-50 units to images? If so, is this model evaluated with actual measured biological responses to validate this comparison? 
* The ImageBeagle algorithm searches through nearest neighbors in image feature space without incorporating neural objectives. How should we interpret the “response curve” produced during this search as reflecting neural response changes? Is the method assuming that smooth interpolation between nearby images in feature space implies a correspondingly smooth interpolation of neural responses? If so, could the authors provide justification or evidence supporting this assumption?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
A large body of work exists on studying the properties of stimuli that strongly (or even 'maximally') excite a neuron in visual cortex. The manuscript at hand extends this approach by exploring anti-preferred stimuli that lead to very weak neural responses. The tuning properties of V4 neurons, as well as those of other areas are shown to exhibit a two-tailed distribution, showing that both highly preferred as well as anti-preferred stimuli exist. This is not mirrored by artificial neural networks. The authors derive an alternative mapping from DNN units to real neurons which yields better predictive performance. It is shown that models perform best when both preferred and anti-preferred stimuli are included during training. The authors describe the results of a psychophysics experiment showing that humans can better predict neural responses to novel stimuli when given both preferred and anti-preferred stimuli as a reference. Finally, the authors propose a method to more efficiently sample images to determine the strongest and weakest activators, yielding large computational savings.

### Strengths
1. Studying anti-preferred stimuli is an important step towards capturing neural tuning properties more globally, beyond exceptionally strong activators.
2. The results of the neural data analysis convincingly demonstrate the two-tailed response distribution.
3. The discussion of ReLU-based DNNs in this context is interesting and shows a clear computational difference between artificial neural networks and the primate visual system. 
4. The psychophysics experiment yields interesting insights, showing that anti-preferred stimuli have additional merit for interpreting a neuron's tuning.
5. ImageBeagle demonstrates substantial benefits over random image sampling. The motivation for the method, as well as the discussion of synthetic MEIs is well-rounded and it is made clear to the reader how ImageBeagle combines the advantages of natural-image sampling with computational efficiency of MEIs.
6. The work is well presented; Figures are clear and intuitive and it is easy to understand the results. The work is well motivated and the main story is clear.

### Weaknesses
I am not fully convinced by the results in sections 3) and 4). I would raise the following points:

1.
1.1 The existence of anti-preferred stimuli suggests that neural responses do not follow a ReLU-like response pattern. Based on this fact, the work argues that a linear map from ReLU-thresholded DNNs units (standard in the field) should be suboptimal. I find this hypothesis to not be well motivated in the text. The manuscript clearly states that anti-preferred images contain some anti-preferred feature which suppresses responses. As the post-ReLU units act as one-tailed detectors for preferred features (demonstrated nicely here!), I see no reason why a map like this should be suboptimal. If a feature is anti-preferred, a ReLU unit detecting that features could simply be assigned a negative weight in the linear map. I would ask the authors to clarify the motivation for this section. 
1.2 Further, the experiment conducted to prove the hypothesis does not seem that convincing to me. Mixing the features linearly, applying a ReLU and then a final linear map increases the capacity of the readout model, which could explain better performance without any real ties to the presented hypothesis. The fact that the control experiment with removed intermediate ReLUs does not increase performance does not seem surprising as that procedure again yields a linear map. I therefore find the claim in line 198f to be very strong given the data. 

2. In Section 4, it seems to me like the experiment is biased towards finding pref+anti-pref to be the best condition. If only (anti-)preferred images are shown, we would expect the average response on the training data to be higher/lower than on the test data, which would artificially reduce R^2. Therefore, it would be expected that 'random' outperforms either of the two biased training sets and I could imagine that the superiority of 'pref+anti-pref' is a methodological artifact. I would strongly suggest that the authors reproduce the results with a mean-independent statistic like correlation between predicted and true responses.

Some further points:

3. Much of the analysis (e.g. Section 5) is based on digital twins rather than real neural data due to noisiness. While this limits the conclusions one can draw a little bit, I realize that this is a difficult issue to address and do not see it as a big problem.
4. ImageBeagle, as a strong contribution of the paper, is not explained thoroughly enough in the main text, in my opinion. I agree it makes sense to put small details in the appendix, but in the current form it is difficult to understand from the main text alone. I would kindly ask the authors to include some more detail in the main text.
5. For ImageBeage specifically, it is unclear how strongly the computational savings are overestimated by evaluating on the digital twin model rather than real neural responses.

Overall, I do like this work and the contributions are strong. However, I would ask the authors to consider the points I've raised regarding some of the methodology. If these are addressed, I would increase my score and argue the paper should be accepted.

### Questions
How does ImageBeagle transfer across brain regions? Does the graph need to be recomputed for different regions as the similarity should be computed at different DNN layers?

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper revisits the classic notion of neuronal selectivity by demonstrating that neurons in the primate visual cortex, particularly area V4, exhibit two-tailed response distributions: responding not only to preferred stimuli that excite activity but also to distinct anti-preferred stimuli that suppress it. Using electrophysiological recordings, modeling, and psychophysical experiments, the authors show that these anti-preferred images are structured and informative, not merely blank or featureless, and that incorporating them significantly improves decoding and predictive models of neural responses. Human participants also rely on both preferred and anti-preferred examples to infer neural tuning, whereas deep neural networks, constrained by ReLU activations, typically capture only one-tailed preferences. The study further introduces ImageBeagle, a tool for efficiently identifying preferred and anti-preferred stimuli across millions of natural images. Overall, the paper establishes anti-preferred stimuli as a key component of neural encoding in visual cortex, expanding the representational capacity of neurons and suggesting new directions for biologically inspired model design.

### Strengths
The manuscript is easy to understand and well-organized. It provides  intuitive figures and explanations that make complex concepts accessible to both neuroscience and machine learning audiences.The paper presents a good amount  of well-controlled experimental results combining neural recordings, computational modeling, and psychophysics, which together provide strong empirical support for the existence and importance of anti-preferred stimuli in visual encoding. The authors also contribute a valuable open-source tool, ImageBeagle, which enables researchers to efficiently identify preferred and anti-preferred stimuli across large natural image datasets, potentially accelerating future neuroscience studies. In addition,

### Weaknesses
Although the paper provides strong evidence for two-tailed neural responses, it mainly examines a single model based on ResNet-50 features and their mapping to V4 activity. It remains unclear whether the same results would generalize to other architectures or visual areas. The finding that combining high- and low-activation images improves decoding is somewhat expected, since training with a broader range of responses typically enhances prediction accuracy. What is more surprising is that anti-preferred stimuli alone do not provide more predictive power than random images (Fig. 3), suggesting that the information they convey is only useful when combined with preferred responses. Finally, while the experimental work is extensive, several analyses rely heavily on model-based predictions rather than fully independent data, which somewhat limits the strength of the causal conclusions.

### Questions
Your analysis focuses on mappings from ResNet-50 features to V4 activity. Have you tested whether similar two-tailed response patterns appear when using other architectures (e.g., self-supervised or non-ReLU models) or in other visual areas such as IT or V1?

In Figure 3, training on anti-preferred stimuli alone performs about as well as random sampling. How should we interpret this in light of your claim that anti-preferred stimuli are an important part of neural encoding? Do they carry independent information, or mainly complementary information when combined with preferred images?

Could you elaborate on the possible biological mechanism that gives rise to these two-tailed response distributions? For example, do you suspect inhibitory circuitry or normalization processes in V4 as the source of the anti-preferred suppression?

You show that combining pre-ReLU features improves prediction. Have you tried using activation functions other than ReLU (e.g., tanh or leaky-ReLU) to see whether two-tailed behavior naturally emerges in artificial networks?

The ImageBeagle tool seems promising. Can you clarify how well it generalizes across different neurons or datasets? For instance, would it work for V1 or IT recordings, or for neurons modeled with non-DNN representations?

In the human experiments, participants performed better when shown both preferred and anti-preferred images. Do you think this reflects an actual perceptual analogy to how V4 represents visual information, or is it more an artifact of the task design?

### Soundness
3

### Presentation
3

### Contribution
3
