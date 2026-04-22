# Circuit Insights: Towards Interpretability Beyond Activations

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
The fields of explainable AI and mechanistic interpretability aim to uncover the internal structure of neural networks, with circuit discovery as a central tool for understanding model computations. Existing approaches, however, rely on manual inspection and remain limited to toy tasks. Automated interpretability offers scalability by analyzing isolated features and their activations, but it often misses interactions between features and depends strongly on external LLMs and dataset quality. Transcoders have recently made it possible to separate feature attributions into input-dependent and input-invariant components, providing a foundation for more systematic circuit analysis. Building on this, we propose WeightLens and CircuitLens, two complementary methods that go beyond activation-based analysis. WeightLens interprets features directly from their learned weights, removing the need for explainer models or datasets while matching or exceeding the performance of existing methods on context-independent features. CircuitLens captures how feature activations arise from interactions between components, revealing circuit-level dynamics that activation-only approaches cannot identify. Together, these methods increase interpretability robustness and enhance scalable mechanistic analysis of circuits while maintaining efficiency and quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces two new (families of) transcoder feature interpretation techniques. WeightLens techniques rely solely on the feature weights, and their relations to the un/embedding matrices, while CircuitLens techniques go beyond simple maximum feature activations visualizations, to find tokens that are genuinely to the feature as part of a circuit. The authors measure the quality of the interpretations produced by these methods with respect to existing interpretations / methods, and find that their methods perform better.

### Strengths
That feature interpretation is a time-consuming and manual process is a well-known issue, and attempts to solve it via auto-interp have fallen short in various ways. This paper proposes a method (or set thereof) that seems better than existing methods; by studying the weights, and features' roles in circuits, it manages to provide more useful information for autointerp models, and produce better interpretations. In this respect, I think the techniques are a novel combination of existing ones, and a significant contribution to the sparse feature interpretation literature, at least in the transcoder context. The authors perform reasonable measurements of interpretation quality, although measuring this is an ongoing and open problem, as I see it.

### Weaknesses
**Clarity / Organizational Issues**: It is at times hard to understand which methods are which in this paper. The authors go into great detail re: input-invariant and circuit-based analyses in 3.1 and 3.2, but it's not always obvious how these match up to the methods evaluated in Section 4. You should probably discuss / name the methods in Section 4 in the preceding sections.

Relatedly, it's not always clear which techniques are new to this paper, and which are not. It's clear that generating feature descriptions based on max activations is not new. I also think that "3. Analyze output effects" is a fairly standard application of the logit lens to feature output weights, and also not new. On the other hand, step 1 of the same procedure seems new, as does most of 3.2.

More qualitative examples could help make this paper's methods clearer. I'd love to see examples of the different inputs to the annotator LLM, as well as the resulting labels produced by each method for particular examples. This is important, because it's hard to understand why / how your methods are better without knowing what each method takes in and returns.

In general, more editing of this paper would help. Sentences are often long (233-236), which makes reading hard. I don't think the the bolded-\paragraph-takeaways format works very well for this paper. This is in part because the paper doesn't really telegraph well what claims it's going to make / study. I went into each section not knowing what the authors intended to prove / show, and encountered a collection of findings without a clear narrative to guide them. Similarly, in the methods sections, methods are introduced without it being clear what they would later be used for. This made for a rough reading experience.

**Metrics and Comparisons**: This paper describes the metrics introduced by Puri et al only briefly, while a longer description (with more details about how they are concretely measured) would be appropriate. These metrics are, by my understanding, all powered by LLM-as-a-judge. This is not unreasonable: it's not as though humans can realistically annotate all of these feature descriptions without great effort. However, I think that some sort of control in order to ensure the validity of the LLM judgments (say, a comparison to human annotations) is warranted. More discussion is warranted in general on this front: the challenge of producing better explanations is not just producing good explanations, but also measuring how good they are, which is quite challenging.

Re: comparisons, the authors often compare to Neuronpedia and MaxAct*, but really need to include more details on what these are. How are Neuronpedia's features generated, exactly?

**Usefulness / Scalability**: The authors propose methods for interpreting transcoder features, but right now, there are few transcoders out there. It would be helpful to know if this technique works on other sparse dictionaries, like SAEs. I would also love details on the cost of these techniques (specifically the circuit-based analyses), and if there is an implementation available (as they seem pretty non-trivial to implement).

Despite these weaknesses, I think this paper is near-worth-accepting, and could be convinced to raise my score. I'd love to see the authors improve the clarity of this paper, which I believe will hinder its usefulness for people who are not core mech interp researchers. I also worry that techniques proposed here are a bit too niche and expensive to be practically useful / interesting for a broader audience, and have asked questions about this below.

### Questions
- Can this method be used with SAEs (as opposed to transcoders)?
- What are the computational requirements of each proposed method?
- Why do you (in 3.1) also consider contributions of tokens via earlier features? This kind of seems like a manual, 1-step version of attribution (which tries to avoid the fact that you don't actually have activations for your transcoders, since you're not running them on an input). It seems more natural to me to just use direct effects, or to actually do attribution.
- You seem to believe that sampling from various quantiles makes more sense for sparse dictionaries than just interpreting max acts (which is better for neurons). Do you find any empirical evidence for this?
- I find 3.2 rather hard to read, as its framing is a bit strange. Why do you say that "To account for interference between layers, Ameisen et al. (2025) propose incorporating a Jacobian term into the attribution formulation..."? I don't think they're accounting for interference, per se; they're just computing the direct effects of one feature on another via all possible paths, which can be done using a Jacobian term. More pertinently, you haven't really introduced attribution yet, so this section comes a bit out of the blue.
- What exactly are the inputs you use for the circuit-based methods, which require inputs with respect to which to perform attribution? The authors say "For generating circuit-based descriptions, we use sampling, described in the Subsection 3.2", but I think it'd be good to be clear about the whole pipeline. I think that what happens is that they sample sentences using inverse quantile frequency, then perform attribution on those samples, and then use these to generate descriptions.
- Can you make Figure 4 easier to read?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces WeightLens and CircuitLens, two methods for automatically generating feature descriptions. WeightLens analyzes model weights alone to identify token-level features and their stable, input-invariant connections, while CircuitLens traces dynamic, input-dependent interactions between features, attention heads, and output tokens using Jacobian-based attributions. By clustering using Jaccard similarity and DBSCAN, the approach tries to generate multiple descriptions of a feature to incorporate its polysemantic behavior.

### Strengths
1. This work addresses an important problem of generating automatic feature descriptions without relying on black-box LLMs or large-scale datasets.
2. The proposed WeightLens technique offers a practical solution to these limitations by producing competitive, and in some cases superior, feature descriptions compared to activation-based methods, while remaining relatively simple to implement.
3. Moreover, the integration of both input- and output-centric analyses, combined with the use of Jaccard similarity and DBSCAN for clustering input examples based on circuit components, is a particularly interesting idea.

### Weaknesses
1. The main motivation of this work is to eliminate dependence on black-box explainer LLMs and large datasets. However, the CircuitLens method still requires access to both a secondary LLM and a large dataset. Therefore, only the WeightLens method truly addresses the core problem the paper aims to solve.
2. Figure 1 is not very informative in its current form. It could be significantly improved by expanding it into a larger figure composed of subfigures illustrating each of the three steps. For example, consider a random feature f at layer l, and visually depict all the steps required to obtain its token-level descriptions.
3. It would also help readers if the Circuit-based analysis subsection (Section 3.2) were better contextualized in relation to the previous subsection and the overall goal of the work. You could begin the section with a brief paragraph explaining why circuit-based analysis is necessary to obtain meaningful feature descriptions.
4. Section 3.2 is somewhat difficult to follow. Overall, the process for generating feature descriptions involves the following steps: 
   1. Sample examples with varied activation values using the proposed sampling strategy. 
   2. For each example, perform input- and (optionally) output-centric analyses.
   3. Compute the Jaccard similarity matrix. 
   4. Apply DBSCAN to cluster the input examples.
   5. Use an LLM to generate cluster-level and overall feature descriptions.

   However, this workflow is not clearly conveyed in the current text, particularly due to the ordering of explanations. I recommend reorganizing this section to follow the logical flow outlined above.

5. The evaluation results for both WeightLens and CircuitLens are presented for only four layers. Results across all (or most) layers should be included, preferably in the appendix, to enable a more comprehensive assessment of their effectiveness.
6. (Minor point) It would be easier to compare different configurations of both WeightLens and CircuitLens if all the evaluation criteria could be aggregated into a single one.

### Questions
1. It is mentioned that a z-score is used to identify outlier tokens. What threshold value is applied to filter these tokens?
2. In Step 2 (Validate tokens) of Section 3.1, tokens that actually activate the feature are filtered in. What criterion defines a token as activating a feature, i.e. what activation threshold is used to make this determination?
3. In WeightLens, it is unclear how the lemmatized tokens are transformed into feature descriptions. Is an LLM employed for this step?
4. Can you also clarify whether output-centric analysis is used when computing the Jaccard similarity? The paragraph describing circuit-based clustering mentions collecting transcoder features and token/attention head pairs for each input but does not specify whether the feature’s impact on the final logit is considered.
5. Lastly, do the activation-based baseline methods use the same secondary LLM to generate feature descriptions?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes WeightLens and CircuitLens as two complementary methods for sparse feature interpretation. WeightLens leverages the input-invariant term of the transcoder attribution to explain context-independent features, while CircuitLens use attribution scores to capture how feature activations arise from interactions between components. These methods show higher clarity and responsiveness than traditional activation-based interpretation.

### Strengths
1. Current methods for automated interpretation of SAE features largely depend on observing the feature activation pattern, which is restricted in explaining complex features with deep connection with upstream and downstream features. This paper takes advantage of transcoders and attribution scores to extend the information for interpreting features. It follows a natural direction of the progress of SAE feature interpretation.
2. The sampling strategy in Section 3.2 considers the complete spectrum of feature activation magnitudes, providing comprehensive explanation of the feature's behavior.
3. The WeightLens method eliminates the need for an explainer LLM.
4. The interpretability metrics from the FADE framework gets substantial improvement with CircuitLens.

### Weaknesses
1. The motivation of several parts of the paper remains opaque and the expression is unclear, including Assumption 2 in Section 3.1, Circuit-Based Clustering in Section 3.2. This makes the paper hard to follow.
2. The faithfulness score seems very low using either WeightLens or CircuitLens. Whether the interpretation truly reflects the features' behavior is doubtful.

### Questions
1. In generating feature description in Section 3.1, is the connection between transcoders also results in token descriptions? Why is this necessary instead of directly showing inter-feature behavior in description?
2. Can you provide more example of the results of WeightLens & CircuitLens?

### Soundness
3

### Presentation
2

### Contribution
3
