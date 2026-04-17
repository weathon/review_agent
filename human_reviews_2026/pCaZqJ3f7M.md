# GeoSDF: Plane Geometry Diagram Synthesis via Signed Distance Field

- Decision: Reject
- Scores: 8, 6, 2, 4

## Abstract
Plane Geometry Diagram Synthesis has been a crucial task in computer graphics, with applications ranging from educational tools to AI-driven mathematical reasoning. Traditionally, we rely on manual tools (e.g., Matplotlib and GeoGebra) to generate precise diagrams, but this usually requires huge, complicated calculations. Recently, researchers start to work on model-based methods (e.g., Stable Diffusion and GPT5) to automatically generate diagrams, saving operational cost but usually suffering from limited realism and insufficient accuracy. In this paper, we propose a novel framework GeoSDF, to automatically generate diagrams efficiently and accurately with Signed Distance Field (SDF). Specifically, we first represent geometric elements (e.g., points, segments, and circles) in the SDF, then construct a series of constraint functions to represent geometric relationships. Next, we optimize those constructed constraint functions to get an optimized field of both elements and constraints. Finally, by rendering the optimized field, we can obtain the synthesized diagram. In our GeoSDF, we define a symbolic language to represent geometric elements and constraints, and our synthesized geometry diagrams can be self-verified in the SDF, ensuring both mathematical accuracy and visual plausibility. In experiments, through both qualitative and quantitative analysis, GeoSDF synthesized both normal high-school level and IMO-level geometry diagrams. We achieve 88.67\% synthesis accuracy by human evaluation in the IMO problem set. Furthermore, we obtain a very high accuracy of solving geometry problems (over 95\% while the current SOTA accuracy is around 75\%) by leveraging our self-verification property. All of these demonstrate the advantage of GeoSDF, paving the way for more sophisticated, accurate, and flexible generation of geometric diagrams for a wide array of applications. The accompanying code, datasets, and all synthesized outputs are being released to benefit the research community.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces GeoSDF, a framework for synthesizing plane geometry diagrams directly from natural-language problem statements. The method converts text into symbolic geometric constraints using an LLM, represents geometric primitives as Signed Distance Fields, and jointly optimizes them to satisfy all constraints. The system produces mathematically valid, self-verifiable diagrams and can even solve geometry problems via direct measurement of optimized configurations. Experiments on FormalGeo7k, GeoQA, and IMO problems demonstrate significant gains over prior rule-based and model-based approaches.

### Strengths
One of the most significant contributions of this paper is the adoption of Signed Distance Fields (SDFs) as the foundational representation for plane geometry diagrams. This choice is both clever and original: by expressing points, lines, and circles as differentiable SDFs and formulating geometric constraints as differentiable loss terms, the method transforms symbolic geometric reasoning into a continuous optimization problem. This formulation elegantly unifies geometry synthesis and constraint satisfaction under a single differentiable framework, enabling gradient-based optimization and smooth convergence toward valid configurations. The approach bridges ideas from computer graphics (implicit surfaces) and symbolic reasoning, marking a clear conceptual advance in geometric problem solving.

Besides of differentiability, the SDF formulation inherently supports quantifiability—every geometric element (angle, distance, area) can be measured directly from the optimized field. This allows GeoSDF not only to synthesize diagrams but also to verify their correctness and even solve geometry problems by extracting numerical answers (“solve-by-construction”). This self-verification property is a strong differentiator compared to black-box neural or diffusion-based methods, and it provides a principled link between diagram generation and mathematical reasoning.

Moreover, the integration of a fine-tuned LLM (Qwen2.5-7B-Instruct) for natural-language parsing is a practical and impactful design decision. It enables the system to directly interpret geometry problems written in natural language, translating them into symbolic constraint sets without requiring a domain-specific language or manual encoding. The resulting pipeline—from text parsing to symbolic representation to differentiable optimization—is compact, self-contained, and user-friendly. This end-to-end design represents a major step toward fully automated geometry reasoning, offering accessibility for future applications.

Finally, The method proposes a batch-enabled high efficient optimization tools, and achieves state-of-the-art results on GeoQA (95.9% accuracy, +20% over prior SOTA) and performs well on IMO-level problems (88.7% human-verified accuracy).

### Weaknesses
Overall the paper is great. There are still several places to improve the final quality:

- While the qualitative results in Figures 3–5 effectively show GeoSDF’s synthesized diagrams, the paper would benefit from more side-by-side visual comparisons against competing systems (e.g., rule-based and diffusion-based baselines). Currently, most comparisons are numerical or textual. Presenting visual outputs from alternative methods under the same problem settings would make the superiority of GeoSDF’s precision and stability more intuitive and convincing.

- Sections 4.6 and 4.7 already provide excellent empirical analyses of convergence and regularization. However, a brief theoretical explanation of why the non-convex optimization behaves robustly, or how the regularization weight $\lambda$ balances constraint accuracy and geometric spacing, would make the methodology more principled and broadly reproducible.

### Questions
1. The paper reports strong parsing performance (F1 = 87.74%, Jaccard = 83.53%) when converting natural language to symbolic constraints. Could the authors elaborate on the types of parsing errors that still occur, especially those involving implicit or unstated geometric relationships? In such cases, does GeoSDF attempt any post-processing, constraint repair, or inference to recover missing elements, or are these examples simply discarded during synthesis?

2. How feasible would it be to extend the SDF-based formulation to 3D or analytic geometry? What challenges might arise in terms of representation or optimization efficiency?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces GeoSDF, a framework for automatic plane geometry diagram synthesis using Signed Distance Fields (SDFs). The method parses natural language problem descriptions into a symbolic constraint representation, represents geometric primitives (points, lines, circles) as SDFs, and optimizes them to satisfy geometric relationships via differentiable loss functions. The framework supports self-verification and quantifiable measurement of geometric entities, allowing both diagram generation and problem-solving. Extensive experiments show improvements over existing model- and rule-based systems.

### Strengths
The strengths lie in its novel and unified design: it introduces an SDF-based formulation that encodes geometric constraints in a fully differentiable manner, allowing optimization and verification within the same framework. 
Empirically, GeoSDF achieves strong results across several benchmarks, clearly outperforming prior neural and multimodal LLM-based solvers. 
The outputs are both quantifiable and interpretable, supporting precise visualization and direct geometric reasoning from a shared representation. 
Overall, GeoSDF offers a broadly impactful contribution that bridges symbolic reasoning, geometry education, and visual problem-solving in AI-assisted mathematical systems.

### Weaknesses
The weaknesses mainly center on generality and robustness. While GeoSDF performs impressively on polygonal and circular figures, it lacks discussion or demonstration of more complex composite geometries—such as spline curves, conic sections, or freeform loci—which limits its expressive scope. 
Furthermore, the optimization procedure may struggle with severely underdetermined or overconstrained systems, yet the paper includes few examples analyzing these failure modes. The reliance on a fine-tuned LLM parser also raises concerns about brittleness when faced with ambiguous or partially specified problem statements. 
Finally, there is no explicit ablation or stress test on scalability with respect to the number of geometric elements or constraints—a key factor for evaluating practical usability and extensibility.

### Questions
I appreciate clarification on several technical aspects of the framework. 
In particular, it remains unclear how GeoSDF could be extended to handle curved or parametric primitives, such as splines or general conic sections, and whether the current SDF formulation could accommodate these without major redesign. 

Another point of concern is how the optimization behaves in highly complex or deeply nested constraint graphs with numerous interdependent elements, and whether there exists a practical failure threshold beyond which convergence becomes unreliable. 

Additionally, clarification is needed on whether the introduced crowd regularization might inadvertently hinder synthesis accuracy in dense configurations where geometric elements are naturally close. 

Finally, it would be valuable to discuss the feasibility of extending GeoSDF beyond planar settings to handle non-planar or 3D geometric diagrams, potentially paving the way toward applications in solid geometry.

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
4

### Summary
Given a geometry problem in Natural Language, the problem here is to construct
a 2d planar diagram illustrating the question.  General-purpose models produce
geometrically inconsistent figures.  Rule-based systems too struggle. The approach
presented in this paper does so by standard gradient descent to find the figure
that minimizes some cost. The cost is set up so that minimizing the cost forces
satisfaction of the constraints in the problem description.

The approach is fairly straight forward: the NL description is parsed to
symbolic constraints, which are then turned into an optimization problem
which is solved by gradient descent.

The paper shows that GeoSDF can produce high-quality geometry figures corresponding
to NL problems. 

The paper also contains a "solve-by-construction" approach, where the diagram
constructed by the GeoSDF is used to measure the entity being sought in the problem
and thus, directly answer the question. This method is shown to perform very well.

### Strengths
Strengths:
1. GeoSDF is a very reasonable approach for constructing geometry diagrams for problems.
2. The evaluation shows that the method works.

### Weaknesses
Weaknesses:
1. This is a fairly natural way to create diagrams. Turning symbolic constraints
solving into an optimization problem is also a common technique.
2. The connection to ML is basically through the use of gradient descent for
solving an optimization problem. And then there is the "solve-by-construction" paradigm
discussed next.
3. The construction-based approach, as described here, can only solve certain problems
where one is asked to measure an angle or segment. Furthermore, it is arguably not
"in the spirit". For example, if two angles of a triangle are given and the problem
asks for the third angle; of course, one can draw and measure the third angle, but that
is usually not the intended approach. 

Appendix D line 940 contains a discussion on implicit assumptions, which play a crucial
role in geometry problems. In the early days, algebraic methods were successful in 
geometry theorem proving (Wu's method) precisely because they had the ability to
synthesize the missing assumptions. It will be interesting to see if GeoSDF can be adapted
to synthesize missing constraints.

### Questions
I do not have any questions.

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
This paper proposes a novel framework, GeoSDF, which represents geometric elements (e.g., points, line segments, and circles) using Signed Distance Fields (SDFs) and constructs a set of constraint functions to encode geometric relationships. By optimizing these constraint functions, the framework generates an optimized field of both elements and constraints, from which diagrams can be efficiently and accurately synthesized through rendering. Both qualitative and quantitative experiments demonstrate that GeoSDF can produce high-level geometric diagrams while maintaining very high accuracy in solving geometry problems.

### Strengths
1. The paper introduce GeoSDF, a novel and accurate framework for synthesizing plane geometry diagrams by optimizing SDF representation against symbolic mathematical constraints.
2. The experimental results demonstrate the effectiveness of the proposed method. GeoSDF not only synthesizes geometric diagrams that are consistent with the problem statements but also provides quantitatively accurate results, highlighting its strong quantifiability.

### Weaknesses
1. The ablation study in Section 4.7 is overly simple. More comprehensive ablation studies as well as qualitative results are suggested;
2. The hyper-parameter \tau_r in Equation 2 is not explicitly specified, and this parameter can affect the optimized SDF results, ultimately influencing the rendered geometric diagrams. It is recommended that the authors provide the value of this hyper-parameter and analyze how its selection impacts the experimental outcomes;
3. The hyper-parameter λ mentioned in Equation 3 is not specified in terms of its value, and in Appendix A, λ also appears in the definition of the parallelism constraint. However, the meanings of λ in these two contexts are different, which may cause confusion. Also, how these hyper-parameters will affect the results is unclear;
4. Other presentation issues:
* In Section 4.2 and the caption of Figure 3, it is stated that the first row on the left represents the original images. However, it should be the first column that corresponds to the original images.
* In Section 4.4, the explanation of Figure 4 is inconsistent with the figure itself. It is unclear whether the angle being calculated is ∠CDE or ∠DFB.
* Section 4 lacks an explanation for subsection 4.3, and the line spacing in Section 4.2 is inconsistent with that of the other subsections.

### Questions
1. The paper mentions that points, line segments, and circles are represented using signed distance fields (SDFs), but the specific formulation or implementation details are not provided. Could the authors clarify how exactly each type of geometric element is represented as an SDF? For instance, how is a line segment’s SDF defined, and how is a circle’s SDF formulated with respect to its center and radius?
2. The paper states that the SDF optimization is a non-convex problem and may converge to local minima or degenerate geometric configurations. Could the authors provide statistics on how often such failures occur in practice? Additionally, are there strategies or heuristics employed to improve convergence stability or avoid degenerate results?
3. Given that Section 3.1 fine-tunes Qwen2.5-7B, could you disclose the use of LLMs in the “Use of LLMs/Responsible AI” section and provide the training hyperparameters, data details, and compliance statements?
4. Could you report the error distribution for angle/length measurements and analyze the sensitivity to the “loss < 0.1” threshold? In addition, how does accuracy change when sweeping the threshold over {0.01, 0.03, 0.1}?
5. Could you detail how the 224k synthetic dataset is used across training/validation/test splits, and provide the deduplication scripts as well as leakage audit results for GeoQA/IMO?

### Soundness
3

### Presentation
2

### Contribution
3
