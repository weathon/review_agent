## Human Reviewer 1

### Summary
In this work, the authors apply a novel method (Sparse Component Alignment) to the problem of differentiating neural response profiles across three distinct streams of the human ventral visual pathway -- all of which are strongly theorized (after decades of intensive neuropsychological and neuroimaging work) to be functionally and representationally distinct, but which many statistical methods do not so saliently differentiate. The authors also assess the advantage of this method in the domain of brain-to-model mappings -- another area where what seems a priori to be salient functional and representational differences often fail to manifest with the statistical methods most popular for assessing them.

### Strengths
This paper constitutes an important contribution and meaningful step towards resolving a deep and lingering concern in computational neuroscience: How do we reconcile data-driven methods with data-derived theory? Methods developed by machine learning researchers and proponents of "hypothesis-free" (a somewhat unfortunate term) structural decomposition / discovery can sometimes clash or produce results counterintuitive to or directly at odds with observations made by experimental neuroscientists and clinicians in well-documented and amply-replicated experimental scenarios. And the issue of "resolution" tends to be one of the defining attributes of this debate. The finding by the authors that there are as-of-yet under-explored statistical methods that advance the mission to use "less biased" / less error-prone "hypothesis-free" data-driven techniques (with scare quotes galore) in a way that nevertheless accords with theoretical priors seems (in my mind) to open the door to new discovery, and far less "two steps forward, one step back"-kind-of debate along the way.

### Weaknesses
While I do not necessarily fault the authors for this (especially given the importance of the first-order business that involves showing this method to be intuitive on the task of differentiating response profiles across the well-separated "streams" of visual cortex), the use of only a few neural network models does not give particularly strong confidence that this method is relevant yet to the differentiation of otherwise very similar-looking brain models (another problem canonical decomposition have heretofore struggled with.)

### Questions
- There are a number of software packages that allow for the comparison of many different neural network models. Why not try this method on more of them? A graph that shows the differentiability of models more canonical methods versus SCA would (in my mind) be a powerful amplifier to this work.)
- The authors mention that the still-remaining non-differentiability of the lateral and dorsal streams is something that we likely need further / better / different brain data assays to meaningfully address. I am entirely on board with this argument, but one natural question that arises here is whether SCA (or any method) can be used not only for differentiation on current data, but for proposing different kinds stimuli that could maximize differentiation in future data? (Think controversial stimuli / selection). 
- A bit of a follow-up on the last point (perhaps something worth addressing depending on the reaction of other reviewers). If at end of day, certain kinds of differentiation can only be achieved with the right kinds of data... doesn't this somewhat undermine one of the central motivations for using these "hypothesis-free" methods in the first place?

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper introduces a new method, Sparse Component Alignment, for comparing models to brain measurements. The method is used to compare fMRI measurements from the Natural Scenes Dataset with 3 Deep neural network architectures trained on ImageNet-1k. Compared with linear (Euclidean) metrics, or RSA, the proposed method stronger alignment with the ventral stream compared with the dorsal or lateral streams.  The authors also compare to behavioral data (the Meadows dataset - sec 3.4), and find similar aligment with the ventral stream.

### Strengths
Development of methods for brain-model comparison is an important area. The authors have developed a promising new method, and have used it to demonstrate  that DNN recognition networks are more similar to ventral stream than dorsal or lateral. Experiments comparing the three brain regions, for 3 different DNN architectures, and 3 different similarity measures, are extensive.

### Weaknesses
The contributions of the paper are both methododological and scientific.  I think both have potential, but found both to be somewhat limited as presented.  

- Method: The intuitive motivation for the SCA method (that it is sensitive  to changes in the axes of the representation) makes sense, but after reading it 3 times, I still don't understand how the mathematical construction of SCA leverages that, nor the extent to which the results should really be attributed to the BNMF pre-processing.  If the method is to be relied upon, I think practioners will want either a derivation tying it to some objective, or some sort of bounds on senstivity to axis changes, or an analysis of why it exhibits this sensitivity to axis changes.  Alternatively, one could provide a more thorough set of empirical tests showing the conditions under which the method works or fails.  For example, how sensitive are results to the choice of the BNMF dimensionality? The simulations provided (fig 2) are quite limited.

- Scientific result: The main result, that recognition models (across three different architectures) are more aligned with ventral stream than dorsal or lateral streams is nice, albeit expected.  RSA already provides this result, in a slightly weaker form (text on p. 8, fig 5).  It was not clear to me how to fairly compare the r-values achived by the two methods.  Nor (as mentioned above) how much the result arises from the BNMF pre-processing.  What if you compute RSA on the BNMF components?  

Writing/presentation could be improved.  
- please indicate up front (abstract/intro) that you're analyzing fMRI data (it was not apparent until sec 2.1).
- I think you could reduce space spent describing Bayesian NMF, which is a known method 
- Provide a more thorough description of the SCA calculation - I read it several times and am still not fully understanding the construction.
- Provide a more  thorough description of the results shown in Fig 4

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper is an attempt to solve the apparent contradiction between the observation that the ventral, dorsal, and lateral streams in human vision have different functions; and the fact that neural activity in each of these tends to be similarly correlated with deep neural-net (DNN) activity trained on a single task. The authors use Bayesian non-negative matrix factorization (NMF) to identify the different selectivities for the three visual streams from fMRI data, confirming previous results. They next introduce a different way of assessing the similarity between neural-net activations and brain recordings that finds that DNNs correlate much better with the ventral stream than with the dorsal and lateral ones.

### Strengths
**Originality**
* the authors observe that the rotational symmetry that is built into existing measures of representational alignment disregards the fact that biological constraints (such as the cost of neural wiring) would in fact be very sensitive to rotations that could, e.g., convert a very sparse representation (that is easy to read out with few neurons) to a very broad one.

**Quality**
* the work seems carefully done, with controls to check that the proposed methods recover known results and that they work on simulated data.

**Clarity**
* the paper is very well written with overall clear figures and detailed explanations.

**Significance**
* understanding the relation between artificial neural nets and their biological counterparts is important both for using the artificial networks to aid in understand the brain, and conversely, as a potential route to gleam inspiration from biological neural circuits to apply to artificial nets.

### Weaknesses
* Fig. 2 is really confusing
  * Both panels seem to mix together the question of inferring the original components correctly with the question of sensitivity to rotations.
  * I would suggest separating the two parts:
    * First, take out the "rotate" part from panel a, focus only on reconstruction, and show bar plot of similarity for different methods.
    * Second, add explanation of rotation and show its effect on SCA.
      * Might also be useful to add lines to show that NMF and PCA don't change at all when you rotate.
  * It's very confusing to have the bars colored in Fig. 2b, as the shades of gray employed there do not match the ones from Fig. 3.
    * If the shade only indicates the similarity level, then it is redundant with the height of the bars, and I would suggest getting rid of it.
*

### Questions
* Why is NMF worse than BNMF with simulated data (Fig. 2)?
  * ...and why is the similarity between inferred and actual components not perfect? If I understand correctly from eq. (4), the data should exactly conform to the model here.
* I don't understand the $\theta$ colorbar in Fig. 2b: what is the range actually? I see it going from $\pi/2\theta$ to $\theta$, which I can't really parse.
* Why is the NMF bar missing from most conditions in Fig. 3b?
* In section 3.2 (line 319), is it standard NMF or Bayesian that is employed? If standard, why not Bayesian?
* The color palette for the lateral stream (Fig. 4b) should go from light pink to red instead of yellow to red, to avoid a confusing change in hue (and also to be consistent with the other two panels).
* The tick and axis labels in most of the figures (Figs. 2, 3, 4, 5, 6) should be significantly bigger.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper introduces Sparse Component Alignment (SCA) for quantifying alignment between two learning systems through a data-driven approach (non-negative matrix factorization). This is done by identifying dominant components that capture sparse neuronal activity. SCA is specific to a system’s native axes of neural tuning. The authors find components selective for faces, scenes, bodies, food, and words in the ventral stream. Furthermore, representations generated by different deep neural networks are found to have a greater alignment with the ventral over dorsal or lateral representations.

### Strengths
Quantifying model to brain similarity is an important research direction that has seen substantive progress in the last few years, and I appreciate the author’s contributions towards the same.
- The manuscript is written in a very straightforward way, and it is easy to follow.
- SCA is biologically and mathematically grounded without a priori hypotheses on neural response profiles.
- The authors show that DANNs, at least the ones tested, show a greater similarity to the ventral stream under SCA over other methods.

### Weaknesses
- Lines 46-9: “How can the same computational model capture the diversity of function across these pathways?” Could the authors ground this question more concretely given that the Topographic DANN (Margalit et al., 2024) was NOT trained to perform only object recognition but instead also imposed a spatial constraint on model units to find separation of visual processing into distinct streams? It was precisely topography and self-supervision that led to that outcome.
- I would have preferred seeing the TDANN’s similarity to the different streams computed using SCA, given that it was cited in the introduction. More generally, I would have preferred seeing more breadth in the computational models evaluated—what do the authors find when they use DANNs that are trained not for object recognition but instead for object detection (such as on the Pascal VOC or MS-COCO datasets) or action recognition (on the Kinetics dataset). Do these models, under SCA, show a higher (or at least somewhat better, given limitations of available neural data) match to the dorsal and lateral streams?

**Things (manuscript writing) that can be improved but did not impact my score:**
- I was having a really hard time looking at the figures since the particular font used makes visibility really low unless the figure is zoomed in a lot. It would be nice if the authors could use a darker and bolder font like Arial or Calibri for the figures.
- Line 97 (and similar phrases throughout the manuscript): “SCA reveals clear alignment of DNNs to the VVP …”. The authors should make it clearer that only those DNNs that have been (a) evaluated in this work, and (b) trained to perform object categorization. DNNs in general is a very broad term given only 4 trained networks were evaluated.

### Questions
How big is the ViT evaluated—is it base-16?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
8

### Confidence
3