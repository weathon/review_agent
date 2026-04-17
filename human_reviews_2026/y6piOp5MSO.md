# GeoPE:A Unified Geometric Positional  Embedding for Structured Tensors

- Decision: Reject
- Scores: 4, 2, 2, 8

## Abstract
Standard Vision Transformers flatten 2D images into 1D sequences, disrupting the natural spatial topology. While Rotary Positional Embedding (RoPE) excels in 1D, it inherits this limitation, often treating spatially distant patches (e.g., at row edges) as sequence neighbors. Existing 2D approaches typically treat spatial axes independently, failing to decouple this false sequential proximity from true spatial distance. To restore the 2D spatial manifold, we introduce Geometric Positional Embedding (GeoPE), a framework that extends rotations to 3D Euclidean space using quaternions. To overcome non-commutativity and ensure symmetry, GeoPE constructs a unified rotational operator by computing the geometric mean in the Lie algebra. This creates a geometrically coupled encoding that effectively separates spatial dimensions. Extensive experiments on image classification, object detection, and 3D semantic segmentation demonstrate that GeoPE consistently outperforms existing 2D RoPE variants and significantly enhances shape bias, confirming its ability to capture true geometric structure.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper builds on rotary positional embedding, extending rotations to 3D Euclidean space using quaternions. The authors developed a method to avoid the quaternion multiplication non-commutativity, and test the method in various benchmarks.

### Strengths
The idea of defining positional embedding with quaternions and Lie algebra is interesting and valuable.

The method to avoid the non-commutativity of the Hamilton product is also a good idea, appreciated.

Also, it would be good to study the problem in terms of shape bias relation, even though this point is not properly developed.

### Weaknesses
W1) Results are not convincing. Results in plots in fig.3 and 5 do not favor the proposed method over previous methods. While, for results in tables, the improvement is marginal and no standard deviation is reported, so it is difficult to evaluate the performance.

W2) Figure 6 is unclear and the explanation in sec 5.4 does not help. It would be interesting to better develop this point.

W3) Results in figure 4 are interesting, as it is clear that GeoPE activates more patches wrt previous methods that mainly activates the diagonal. However, diagonal elements are not activated as the diagonal is mainly darker. Why? How does it impact the performance? 

W4) Figure 1 is not clear (and of low quality).

W5) in realted works, especially in the shape bias, some discussion on previous methods involving quaternions, lie algebra, or biases due to algebraic representations should be included, such as:
1) Demystifying the Hypercomplex: Inductive biases in hypercomplex deep learning, Signal Processing Magazine
2) Fast Quaternion Product Units for Learning Disentangled Representations in SO(3), Transactions on Pattern Analysis and Machine Intelligence

W6) in Sec 3.1, it is actually not recommended to build a quaternion by simply splitting a vector v as v/3, since quaternions represent precise Mathematical entities and they better work when correlations/relations between the dimensionalities exist. Indeed, quaternions better work in the case of multimodal/multichannel etc data. If we simply split a vector, this is not guaranteed.

### Questions
Q1) I guess that the colors for the plot in figure 3 are wrong? If not, results are inconsistent across the dimensions.

Q2) Same in figure 5?

Q3) Can the authors report standard deviation results over three runs for tables results?

Q4) Can the aauthors provide computational time comparisons among the models? Especially since they mention it talking about Linear GeoPE.

### Soundness
3

### Presentation
2

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
The paper introduces GeoPE, a positional encoding for transformers operating on 2D or 3D data. 
The presented approach offers an extension of the rotary positional embedding (RoPE) widely used in transformers that operate on 1D sequences.
It combines 3D rotations around different axes (each axis encodes one spatial dimension) into one 3D rotation by log-exp averaging. The resulting 3D rotations of different frequencies are applied on 3D-subvectors to apply a positional encoding to keys, queries and values in the attention mechanism. The authors develop a “linear” version of GeoPE that applies a single matrix between key and query subvectors that depends only on the relative position.
(Linear) GeoPE is compared against the competitor RoPE-Mixed and other baseline methods on 2D image classification and object detection. GeoPE is compared against a simple baseline for semantic segmentation of 3D point clouds.

### Strengths
The idea of GeoPE is straightforward and offers a possibility to address positional encodings for transformers operating on 2D or 3D data.
GeoPE can be easily implemented.

### Weaknesses
The presented method lacks motivation and experimental comparison against related work. The overall presentation is not well structured and in several places unclear. The design choices in GeoPE are in my opinion not sufficiently ablated.

**Comparison against related work:**  
Overall, GeoPE is not consistently compared against competitors such as RoPE-Mixed [1]. The comparisons in Table 1 seem unsystematic and arbitrary. Furthermore, GeoPE should be benchmarked against LieRE [2]. LieRE is applicable to 2D and 3D data and seems to consistently outperform RoPE-Mixed. All together, the presented (incomplete) comparison (e.g. Tab. 1 and Fig. 5) does not make a convincing case that GeoPE “consinstently outperforms standard baselines and existing 3D RoPE variants” as claimed in the abstract (l. 022). For instance, l. 411 claims “exceptional zero-shot inference capabilities across multiple resolutions’’ but RoPE-Mixed seems to be superior (cf. Fig. 5).

What are the conceptual differences and advantages of GeoPE over RoPE-Mixed and LieRE?
The authors state that “these approaches remain essentially 1D ROPE, as axes are treated independently, and mixed-frequency schemes only partially capture diagonal dependencies” (l. 046).  To me it is not clear why treating axes individually is inferior. An explanation based on formulae could make the differentiation more precise.
Furthermore, the authors state that LieRE [2] is “computationally expensive” (l. 056). A runtime comparison against LieRE could help to support this statement.

[1] Byeongho Heo, Song Park, Dongyoon Han, and Sangdoo Yun. Rotary position embedding for vision transformer. In European Conference on Computer Vision, pp. 289–305. Springer, 2024.

[2] Sophie Ostmeier, Brian Axelrod, Maya Varma, Michael Moseley, Akshay S Chaudhari, and Curtis Langlotz. Liere: Lie rotational positional encodings. In Forty-second International Conference on Machine Learning.

**Motivation and ablations:**  
The authors claim that averaging 3D rotations around different axes (each axis encoding the position w.r.t. one spatial dimension) is a “natural choice” (l. 158) and “geometrically sound” (l. 255). This claim seems not sufficiently supported by theory or ablation experiments. 1) Why are rotations around different axes a geometrically meaningful way to couple positional encodings from different dimensions? 2) Why is the average of the rotations around different axes geometrically more meaningful than e.g. the composition or e.g. an average of the separately rotated (sub-)features? 

The text claims that a positional encoding that is non-commutative in height and width encodings (for 2D images) is problematic (l. 183) but the authors do not support this claim experimentally. In particular for video data, it might actually be desirable to distinguish between spatial and temporal embeddings. An ablation that compares the averaged rotations against the composition of rotations would help to justify this claim.

The effectiveness and importance of the linear GeoPE is not sufficiently ablated (it seems to appear only partially in Table 1 and in none of the other tables).

**Limitations (of GeoPE) are missing:**  
Appendix F only discusses limitations of linear GeoPE. Limitations seem to be that GeoPE is only applicable for geometric data in Euclidean space up to dimension 3. Furthermore, the feature dimension must be divisible by 3. Are there other limitations of GeoPE?

**Structure of text and presentation:**  
* A background section on RoPE to introduce the reader to the topic and the notation is missing and would really improve the presentation.
* The term “diagonal interactions” in l. 088 is not clearly defined/introduced.
* When reading from top to bottom, the relation of GeoPE to the Shape Bias paragraph in the related work is unclear.
* The captions of Fig. 1 is rather uninformative. Given that figure aims to explain the main method, a more detailed caption would be helpful.
* Adding an appendix on quaternions, the quaterion product, and the relation to rotation matrices would be helpful and could be reference in Sec. 3.1.
* The caption of Fig. 2 is vague. Fig. 2b should rather be placed much later in the text where it is referenced.
* The statement that “GeoPE keep[s] long distance decay” (l. 215) is unclear. If this is a unique selling point of your method, please elaborate more in the main text. 
* The notion of “mean attention distance” (Fig. 3) is not introduced. (I suppose it is the attention-weighted average of distances?)
* In the caption of Fig. 3 the authors state that RoPE-Mixed and GeoPE apply a “more structured” strategy but it is unclear whether this is the “right” structure. For instance, the curve of GeoPE for the resolution of 128x128 looks very similar to the curve of APE for 224x224.
* In Fig. 4 it is unclear which model is used and on which task it has been trained. Please explain in the text why one can see substructure in patches.
* In Fig. 5 it says “training resolution” on top which seems to be a typo. The caption says that the training resolution was fixed to 224x224. Please clarify.
* How are shape vs. texture decisions defined in Figure 6? Please explain this in the text.

**Minor points:**  
* Please give a reference for the following statement:
“A strong shape bias, which prioritizes object structure over texture, is often correlated with better robustness and generalization.” (l. 465)
* “discussed” typo in l. 214

### Questions
* Do other generalizations of RoPE like RoPE-Mixed or LieRE also preserve the “long distance decay” of attention scores over distance or is this a specific feature of GeoPE?
* For 3D point clouds (in particular molecular data), rotational equivariance is very popular. Can GeoPE be modified to satisfy rotational equivariance?
* The caption of Table 2 says that models with GeoPE are “pre-trained on ImageNet-1K”. Is this also the case for the other models? Does that hinder a fair comparison?

### Soundness
1

### Presentation
1

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
The paper introduces Geometric Positional Embedding (GeoPE), a method that extends Rotary Positional Embeddingfrom 1D to higher-dimensional structured data by using quaternions to represent coupled rotations in 3D space. GeoPE constructs a symmetric rotational operator, ensuring consistent multi-axis encoding and offering a linear variant that preserves strict relative positional relationships. Experiments on image classification, object detection, and 3D segmentation show that GeoPE improves performance a tiny bit and enhances models’ spatial reasoning and shape bias

### Strengths
- Evaluation on a diverse set of tasks like image classification, object detection, and 3D semantic segmentation

### Weaknesses
- The extension of RoPE to 3D has been proposed by several other works already like VideoRoPE. Therefore the motivation and novelty could be made clearer. 
- The experimental results are missing statistical significance shown by confidence intervals for example. Words like "significant performance gains" or "exceptional zero-shot inferencecapabilities" are not backed with statistical meaning or quantitativ results. 
- The experimental comparison to prior work in the 2D and 3D space is incomplete. Here only Rope-Mixed and absolute is compared to where other works like STRING, VideoRope or LieRE works have already shown strong performance. How does GeoPE compare to just the 3D version of Rope-Mixed? A 3D version of Rope-Mixed would also have commutativity.
- Missing description of theoretical guarantees. Why is commutativity important in theory and how does that directly translate to practice? Ablations are missing.

### Questions
- Would it be possible to add confidence intervals?
- Would it be possible to add more SOTA baselines? What is LinGeoPE? Why did you choose CPE and not STRING, VideoRoPE idea or LieRE or recent baselines?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Geometric Positional Embedding (GeoPE), a generalization of Rotary Positional Embedding (RoPE) designed for higher-dimensional structured data, specifically demonstrated on 2D and 3D tasks. The core motivation is the challenge of extending RoPE to higher dimensions: direct generalization requires modeling coupled multi-axis rotations, a problem often bypassed in existing work by assuming axis independence or using heuristic methods. The authors propose using quaternions to formulate 3D rotations. To address the issue of non-commutativity in quaternion multiplication (where rotation order affects the result), they leverage Lie Algebra principles and take the geometric mean of rotations in log space, which retains the desirable property of commutativity. A variant, Linear GeoPE, is also proposed. It aims to reintroduce the relative position encoding capability of 1D RoPE by enforcing a linear relationship within the Lie algebra, by approximating rotational composition with vector addition. This comes at the cost of higher memory complexity. The authors evaluate GeoPE and Linear GeoPE on image classification, object detection, and 3D semantic segmentation across various backbones, and compare them to existing positional encoding baselines and 2D rotational embeddings.

### Strengths
- The paper presents a novel generalization of RoPE to higher dimensions that explicitly models coupled multi-axis rotations, addressing a key limitation in existing methods
- The proposed method achieves superior performance across multiple backbones and 2D/3D tasks compared to competing positional encoding methods
- The paper introduces two variants (GeoPE and Linear GeoPE), offering a practical trade-off between enforcing linear inductive bias (relative position encoding) and computational efficiency
- The authors present an interesting analysis on shape-texture bias, showing that GeoPE increases the model’s shape bias, with the motivation of observed correlation between shape bias and better generalization and robustness

### Weaknesses
The acknowledged limitations of GeoPE (does not inherently enforce the desired linear relationship in the parameter space)  and Linear GeoPE (incurring significant memory overhead)

### Questions
A table comparing the runtime (FLOPs) and memory requirements of GeoPE, Linear GeoPE, and all other compared encoding methods would help readers precisely position the two GeoPE variants in terms of the performance/resource trade-off

### Soundness
3

### Presentation
3

### Contribution
4
