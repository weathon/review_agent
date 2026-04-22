# ExpoTab: One-Step Mixed-Type Tabular Data Generation using Manifold Learning

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4

## Abstract
Fast generation of high-quality synthetic tabular data is critical for supporting latency-sensitive applications. This includes real-time fraud detection, where rapid model iteration depends on quick data synthesis, and large-scale system testing (e.g., e-commerce load simulation), where generators must produce massive volumes under strict time constraints to assess performance. However, current state-of-the-art (SOTA) tabular generators are inefficient, often requiring numerous denoising steps, and ineffective at handling high-cardinality categorical features and mixed-type data representations common in real-world datasets. We introduce an Exponential map method for Tabular data fast generation(ExpoTab), a geometric-aware flow model that addresses these challenges through two key innovations: (1) the ExpoTab-Encoder, a spherical encoder that projects categorical features onto optimized manifold coordinates, with quantile transformers to unify all features into a bounded representation space for stable optimization; and (2) the ExpoTab-Sampler, presenting a curved-path velocity field that leverages Riemannian geometry to model geodesics as one-step exponential maps, achieving one-step sampling. Extensive experiments across ten benchmarks demonstrate that ExpoTab outperforms existing methods in three key aspects: (a) our manifold-aware encoder achieves superior separability and flow matching performance; (b) the exponential map sampler accelerates generation by 4--10$\times$ while maintaining competitive fidelity; and (c) ExpoTab represents the first one-step generator effectively handling mixed-type tabular data, achieving $\sim$0.1s generation for 1M samples.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Current tabular diffusion models model under either a separate or a unified data representation. ExpoTab builds upon TabRep which tackles the latter by introducing a spherical encoding scheme coupled with a manifold-based flow sampler with exponential maps that can achieve state of the art sampling speeds while maintaining generation quality.

### Strengths
**Originality**. The work builds upon existing tabular diffusion methods such as TabRep by proposing a new spherical representation to encode the data. Existing methods such as TabRep-Flow enables few-step generation. To achieve one-step generation for ExpoTab's spherical representation, ExpoTab:

- first employ conditional flow matching via a separate network trained to model the velocity field

- based on the velocity field, it trains a network to directly approximate the endpoint of the exponential map (mapping an initial point to an endpoint of a geodesic) by computing the geometric displacement.

A "two stage" shortcut generative process, resembling mean flow, consistency and shortcut models.

**Quality**. The paper demonstrates gains in sampling speeds for tabular data generation while maintaining and improving generation quality. This validates the promise and importance of exploring various encoding methods for tabular data.

**Clarity**. The paper is written clearly which makes it easy to understand and follow.

**Significance**. Tackles shared space/unified tabular generation via a geometric lens.

### Weaknesses
**Spherical Encoding**. Further justification is required for the motivation behind the spherical encoding. 1. It is clear that the encoding achieves "separability". However, other three dimensional shapes such as a cylinder could achieve this separability notion too. What other motivations are there behind this specific design choice? 2. Furthermore, TabRep assumes encoding on a unit circle. What if at high dimensions, TabRep enlarges the size of its circular embedding to ensure separability. How does ExpoTab fair off against TabRep there? 3. Additionally, why use a "Generalized Spiral layout" instead of other methods for generating evenly spaced points on a sphere e.g. Fibonacci lattice? 4. The current spiral layout induces an order on the categorical features. How does the ordering affect the performance?

**Model Choice**. The authors primary inspiration for one-step generation stems from mean flow. However, when reading the passage, it is difficult to distinguish the complete difference between mean flow and their proposed sampler. On top of mean flow, there are alternative one-step generation methods such as Shortcut Models [1] and Consistency Models [2]. How does the proposed sampler fair off against these alternatives?

[1] Frans, Kevin, et al. "One step diffusion via shortcut models."

[2] Song, Yang, et al. "Consistency models."

**Privacy Preservation**. Privacy preservation is by far one of the most important applications in tabular data generation. Despite conducting DCR experiments, existing literature in tabular data generation [1] and computational privacy [2] [3] have highlighted the inadequacy of DCR in evaluating privacy preservation. Hence, despite leaving MIAs to future work, its still important to demonstrate privacy preservation abilities with at least MIAs.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a novel method of handling high dimensional categorical data and mixed datatypes in tabular data generation. ExpoTab is a geometric approach that maps categorical features onto manifold coordinates and unifies features into a common representation space and leverages Riemannian geometry to sample via geodesics. Simulations suggest improved sampling efficiency for mixed types tabular data as compared to encoding and generative modeling baselines and 10 datasets from the UCI repository.

### Strengths
Strengths of the paper include:
- Tabular data modeling and sampling are important practical and theoretical areas of research.
- The paper includes a large number of simulations. 
- The proposed approach is compared to a number of baselines.

### Weaknesses
Weaknesses include:
- The exposition is not up to the standard expected for the venue
- The mathematical problem could be stated more clearly and concisely. 
- The main paper omits details necessary to understand the approach. 
- The datasets modeled (UCI repository) could be much more ambitious. 

Minor: 
- Typo in the abstract: generation(ExpoTab),
- mechanism(Fiore et al., 2019).
- The caption for figure 1 is not a complete sentence. Was that intentional?
- Figure 4 is cited on page 1 before any other figure but appears on page 8 after three other figures. Was that intentional? 
- Figure 2 is referred to before Figure 1? 
- I would expect the related work to map onto the contributions, but it does not. For example, 2.2 is about mixed data, but the second contribution (as described at the end of section 1, is about categorical data. 
- The abstract claims two contributions, but the introduction claims 3? 
- Lines 213-215 are not as clear as they could be. There is a two stage generation to make a one step sampler. But there has not yet been a complete description of the problem. Equation 1 describes sampling, but where does z_0, the input to the sampler come from? Presumably the answer is the spherical encoder and quantile transformer, but we haven't been told how. 
- Line 213 states "To provide the necessary supervision for this task..." What is the supervision necessary for? There is no minimization problem posed at this point in the paper. 
- The paragraph starting at 227 should define what D is first then D_c and D_d. I would typically expect D to refer to a data matrix, however here it appears to be a natural number? It is important to define the problem and notation, especially for the most basic objects. 
- " For each of the Dd categorical features, a deterministic mapping function Sj : {1, ..., Kj} → S embeds each of the Kj unique categories onto the surface of a 2-sphere Figure 2A". I am not sure that I understand. This describes what happens for each feature. Are there D_d 2-spheres? How are these integrated to a unified representation? 
- Line 232 refers to " generalized spiral parameterization" which is not defined and Algorithm 1, which doesn't appear in the main text. 
- I see that you are forming a product manifold with the spheres 
- The exposition feels like it is written backwards. I normally expect the goal to come at the beginning. Something like: "If we could express our data in this nice mathematical form, then we could sample efficiently in one step. There are X challenges for doing so..."
- Typo: points(Figure 2B(a)(b)).

### Questions
Please see above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents a new method, ExpoTab, for synthetic tabular data generation with a geometric representation of the data manifold. The proposed method has some efficient properties, such as low generation time, but the benchmarks are not organized, which leads to the question of the performance of the proposed method. Furthermore, the paper is not easy to follow; some of the critical evidence lives only in the appendix (e.g., Distance to Closest Record (DCR) results).

### Strengths
- A rapid new method for the tabular data generation

### Weaknesses
### Major Concerns

- Benchmarks: Organization of the benchmarks is poor; some experiments use different sets of baselines and datasets, which is not consistent and limits the ability to assess the method’s performance comprehensively. For example, why is TABSYN not included in Table 1? According to the TABSYN paper, it has 2x better results than TAB-GEOFLOW in terms of runtime. 

- Many recent works are missed, please see this:  https://github.com/Diffusion-Model-Leiden/awesome-diffusion-models-for-tabular-data

- Critical evaluation results, including benchmarking on important measures, are relegated to the appendix rather than included in the main text. For example,  I could not find the result for a crucial metric: `Discriminator measure` (for all baselines). I found a description in the appendix that you use Classifier Two-Sample Test (C2ST), which is in my opinion a weak approach, it is better to use more powerful methods such as RF or even GBDTs. All this reduces transparency and weakens the argumentation.  

- The manuscript exhibits indicators of LLM-generated text. For example, certain citations are incorrect or inconsistent. Specifically, `Line 1356` references “LLM-based: GReaT Ruan et al. (2024): A method that frames tabular generation as an autoregressive language modeling task,” which suggests automated text insertion. There is also an overuse of the em dash, a common pattern in ChatGPT-generated writing.

### Minor Issues

- The text states “(red) while maintaining competitive α-precision relative to the best-performing methods (blue).” In the appendix, it reads “Best results are highlighted in blue,” but no blue highlighting is present

- CTGAN is cited inconsistently as both Xu et al. (2019) and Zhao et al. (2024), which creates confusion

- Figure 3 is not legible in its current form and requires improved resolution or redesign for clarity

### Questions
- Could you please justify the choice of experiments and baselines that you have included into the main paper?
- Based on the method design, I suspect a lot of copies (data leaks) from real data, have you checked how many samples in the synthetic data are exact copies from real dataset?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ExpoTab, a novel tabular data generative model that integrates manifold learning with flow matching to enable one-step data generation.
Unlike conventional diffusion or multi-step flow models, ExpoTab models the data manifold geometrically using geodesic-based exponential mapping, allowing efficient and structure-preserving sampling.
The method employs a unified encoding scheme that embeds numerical and categorical features into a continuous bounded manifold, aiming to capture both smooth and discrete dependencies within tabular data.
Through experiments on various datasets, the authors claim that ExpoTab achieves competitive or superior fidelity ($\alpha$-precision) and machine learning efficacy (MLE), particularly on low-dimensional tabular datasets, while being significantly faster in generation.

### Strengths
1. The paper presents a novel framework that combines manifold learning with flow matching, enabling one-step data sampling. This design significantly improves sampling efficiency, especially on low-dimensional tabular datasets.

2. The authors provide a comprehensive set of experiments comparing multiple encoding schemes, offering valuable insights into the impact of different tabular data representations.

### Weaknesses
1. The datasets used in the experiments appear inconsistent and insufficiently described, making it difficult to assess the generality of the results.

2. The empirical evidence supporting the claimed contribution is not clearly demonstrated.
The paper states that the proposed method enhances the separability of high-cardinality categorical features, thereby improving performance on tabular data dominated by discrete variables.
However, this claim is not explicitly supported by quantitative results.
For instance, in the Adult dataset (with a maximum cardinality of 42), there is no clear empirical verification showing how much the proposed method improves machine learning efficacy compared to prior models.

3. The presentation of the additional experiments in the appendix (e.g., Appendix H) lacks clarity.
Some reported results (such as “improvement” metrics in Table 18) are ambiguous and require further explanation regarding their reference baselines and calculation method.

### Questions
1. Could the authors elaborate on how the proposed model performs on datasets that are purely categorical or purely numerical? In addition, for the datasets listed in Table 4, please clarify the exact composition of numerical vs. categorical variables used in the experiments.

2. The paper demonstrates that the proposed model achieves faster generation than other approaches when the number of samples is large. However, it remains unclear how the method performs on high-dimensional datasets in terms of sampling time. Could the authors provide further discussion or quantitative evidence on this aspect?

3. For MLE, the comparison between the proposed method and existing baselines primarily focuses on high-dimensional datasets (e.g., Table 18). It would be helpful to see a more detailed comparison with other tabular data generation models on the low-dimensional datasets, so that the results can serve as additional evidence supporting the paper’s main claims.

4. The explanations of the additional experimental results in Appendix H (including Table 18) are not entirely clear. For instance, it is ambiguous what the reported “improvement” values represent -- improvement compared to which baseline? In Table 18, for example, the MLE result of GINA is lower than that of CDCD, yet it is marked as having a 2.58% improvement. Clarifying how this metric is computed and what the reference point is would help readers interpret these results more accurately.

### Soundness
3

### Presentation
2

### Contribution
2
