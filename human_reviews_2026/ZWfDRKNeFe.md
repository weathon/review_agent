# GeoBS: Information-Theoretic Quantification of Geographic Bias in AI Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
The widespread adoption of AI models, especially foundation models (FMs), has made a profound impact on numerous domains. However, it also raises significant ethical concerns, including bias issues. 
Although numerous efforts have been made to quantify and mitigate social bias in AI models, 
**geographic bias** (in short, geo-bias) receives much less attention, which presents unique challenges. 
While previous work has explored ways to quantify geo-bias, these measures are *model-specific* (e.g., mean absolute deviation of LLM ratings) or *spatially implicit* (e.g., average fairness scores of all spatial partitions). 
We lack a **model-agnostic, universally applicable, and spatially explicit** geo-bias evaluation framework that allows researchers to fairly compare the geo-bias of different AI models and to understand what spatial factors contribute to the geo-bias. 
In this paper, we establish an **information-theoretic framework for geo-bias evaluation**, called **GeoBS** (**Geo**-**B**ias **S**cores). We demonstrate the generalizability of the proposed framework by showing how to interpret and analyze existing geo-bias measures under this framework. Then, we propose three novel geo-bias scores that explicitly take intricate spatial factors (multi-scalability, distance decay, and anisotropy) into consideration. 
Finally, we conduct extensive experiments on 3 tasks, 8 datasets, and 8 models to demonstrate that both task-specific GeoAI models and general-purpose foundation models may suffer from various types of geo-bias. 
This framework will not only advance the technical understanding of geographic bias but will also establish a foundation for integrating spatial fairness into the design, deployment, and evaluation of AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a mathematical frame-work for measuring the geographic bias (geo-bias) of predictive models. The frame-work asserts that there are three main aspects in the design space of geo-bias metric: the map, the reference pattern, and the difference measure, whose variations can construct different metrics. The paper also proposes new metrics of geo-bias that can capture bias related to multi-scalability, distance decay, and anisotropy, and provides experiments that utilize these metrics to measure the geo-bias of several foundation models.

### Strengths
The new metrics can be valuable for encouraging and enabling design of less geographically biased models. The insights about the existence of geo-bias in foundation models is also important for the users of these models in practice.

### Weaknesses
1. I find the mathematical frame-work uninformative. In particular, it is not clear where and how the frame-work can be useful. For example, the new metrics constructed by the paper do not stem from the theoretical framework, nor rely on it for any mathematical guarantees about their validity.

2. The figures in the paper lack clarity in my opinion, and particularly the captions are very uninformative.

3. The justifications for the use of KL divergence is not convincing. The first point is that it is less computationally expensive, but there are other linear-complexity options such as total variation. The second point about KL having a “physical” interpretation seems incorrect to me, providing reference for this claim is important (note that KL is not bounded and also is not a proper distance). Lastly, it is not clear why KL is chosen compared to reverse KL.

4. In Lines 374-377, the use of a binary metric for measuring continuous regression tasks seems unjustified, and the explanation of the binary metric lacks mathematical rigor.

5. The exclusion of some ROIs from evaluation, as pointed out in Lines 400-401, is not well-justified.

6. The choice of hyperparameters seems very ad-hoc, and it is unclear how changing these hyperparameters will affect the results and ranking of the models. This makes these metrics less useful in practice.

7. The proposed metrics, and the reported results, lack any measure of statistical significance (and a way to compute confidence intervals). Therefore, it is unclear how meaningful the differences between these metrics are.

8. The paper provides no validation for its proposed metrics for use in practical applications. How can a user trust conclusions drawn from the proposed metrics?

9. The paper draws some hypotheses from its analyses (Lines 440-444 and 466-472), but does not explore them any further. Providing some evidence to reject/affirm these hypotheses would be valuable for the development of future models.

Typos:

In Definition 4.3, “#” is undefined.

### Questions
1. Can the authors clarify the practical importance of the theoretical frame-work? Does it provide any guarantees, etc?
2. Can the authors provide any way to validate the metrics, for example, on synthetic data?
3. What are confidence intervals for the metrics?
4. What is the advantage of the proposed metric compared to performing independence test (chi-square on contingency tables) between model-correctness and geographical partitions?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focus on the geo-bias evaluation in AI models, the authors propose GeoBS, an information-theoretic, model-agnostic, and spatially explicit framework for fair geo-bias evaluation. They further introduce three SRE scores considering multi-scalability, distance decay, and anisotropy. Experiments on 3 tasks, 8 datasets, and 8 models show that both GeoAI and several general-purpose models exhibit geo-bias, highlighting the need for spatial fairness in AI design.

### Strengths
1. I like this paper — combining spatial point pattern analysis with information theory for geo-bias evaluation is an innovative perspective that provides a theoretical lens. Although geographic bias might seem like a narrow topic within the broader fairness discussion, this kind of specialized, systematic, and in-depth evaluation has strong practical value.

2. This work successfully demonstrates how existing geo-bias metrics (Unmarked SSI and Marked SSI) can be interpreted within this framework, demonstrating good extensibility. The proposed metrics are model-agnostic and applicable to different types of AI models. 

3. The spatially explicit nature makes evaluation results more interpretable. Experiments cover multiple tasks, datasets, and model types, enhancing the credibility of conclusions

### Weaknesses
1. The definition of "spatial homogeneity" is oversimplified; it's difficult to define what constitutes an "unbiased" reference pattern in practice. 
2. Lacks sensitivity analysis for different reference pattern choices; choice of KL divergence lacks sufficient justification.
3. Only considers first-order statistics, potentially missing important spatial interaction information.
4. Hyperparameter choices (ROI radius, grid size, ...) significantly impact results, but lack systematic selection guidance.
5. Lacks statistical significance testing, making it difficult to judge whether observed differences are statistically meaningful.

### Questions
1. How should appropriate "unbiased" reference patterns be selected for different application scenarios? Can you provide a  guiding discussions?
2. Can you provide some discussions showing how these bias scores could guide practitioners in improving their models?
3.  SRE scores themselves show high hyperparameter  sensitivity. How will you address this issue?

### Soundness
3

### Presentation
3

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
The paper introduces an information-theoretic and model-agnostic framework for quantifying geographical bias. This framework can integrate existing metrics and proposes three new metrics that account for multi-scalability, distance decay, and anisotropy. The authors perform experiments across multiple models to reveal that there exist geobias in both task-specific geoAI models and foundation models.

### Strengths
The proposed framework is model-agnostic, addressing a key limitation of prior work, which tends to be restricted to specific tasks. They include ways to quantify bias by including intricate details about space, distance, and scale. This ensures the quantifying of bias in a granular way.

### Weaknesses
In my opinion, the abstract of the paper is misleading because of the part: “ but will also establish a foundation for integrating spatial fairness into the design, deployment, and evaluation of AI systems”. The framework is purely diagnostic, and no mitigation scheme is mentioned.

Figure 1 is a very important figure that helps motivate the importance of quantifying bias in different ways. But it's hard to understand what the different colored points represent since they are not all explicitly mentioned. This makes the overall message of the figure unclear. Additionally, using the same spatial area to illustrate each metric would make the differences between them easier to compare and understand. The acronym SSI is not defined and explained at this stage (it is done way later in the paper). It would have been to see if the authors described what they meant by geo bias clearly with some easy-to-understand intuitive examples.

The writing of the paper is occasionally difficult to follow. For instance, “Spatial point patterns always involve multiple locations (a single point will not form 'patterns'), so we define the unit to evaluate geo-bias as.”. This is followed by a series of undefined notations (e.g., what i,m,s, and t index). The first paragraph of Section 4 is missing logical flow. It starts off by mentioning the objectives, then explains how it serves the purpose. Where the purpose itself is not mentioned. Additionally, they state “three key factors we need to consider when designing new geo-bias metrics”, what are those key factors? It's very important to clarify the differences between unmarked SSI and marked SSI. Figure 2 can be a little misleading since two different locations are used to visualize the geo-bias scenarios. The main idea is that they would both have the same number of points, except one would just have the points against the unobserved background, and the other would have the same points with different colors representing high and low performance. The placement of Table 1 is rather odd, I would want to have it near the intro, not near the end, when I'm almost done reading. And these acronyms are used throughout the texts, not just in the following section.

For definition 4.3, having #P_k/#N outside the sum function makes no sense as to what the definition intends. They are supposed to be using the weight for each patch, not just one.

Converting regression values to binary might lose valuable information, and the choice of threshold should be justified.

Since SPAD is used as a baseline, details about what exactly it quantifies should be mentioned.

The claim that “The geo-bias scores are significantly lower than the task-specific counterparts” is not supported by the presented tables. While Table 4 shows lower bias values relative to Table 3, the comparison to Table 2 contradicts this assertion: many entries in Table 4 are comparable to, or higher than, those in Table 2. The authors should either (a) clarify exactly which comparisons support the “significantly lower” claim, (b) report formal significance tests (with effect sizes and confidence intervals) for the claimed differences, or (c) revise the statement to accurately reflect the mixed results.

The parameter sensitivity analysis is conducted only for a limited subset of models and datasets, without any justification for their selection. Without explaining why these particular models and datasets were chosen, the analysis cannot be considered a comprehensive or reliable assessment of parameter tuning.

The paper does motivate the importance of a model-agnostic approach and the importance of spatial info like (distance, direction, and scale). The discussion remains at a general level, reporting the presence of bias without interpreting its variation by scale, distance decay, or directional (anisotropic) patterns. Providing such an interpretation would strengthen the link between the theoretical motivation and the empirical findings.

In Figure 3, are the bars on the left for a single patch? The caption itself is not informative enough, and there's no mention of this in the text.
Details about the hyperparameter tuning of the models are also missing.

### Questions
In addition to the weaknesses discussed earlier, I would appreciate the authors' responses on the following:

* In Figure 3, are the bars on the left for a single patch?

* Why was the parameter sensitivity analysis conducted only for a limited subset of models and datasets?

*Why was GPT-4o chosen as one of the fms?

### Soundness
2

### Presentation
2

### Contribution
2
