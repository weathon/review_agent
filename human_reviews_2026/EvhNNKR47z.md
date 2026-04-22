# GeoFlow: Geo-Aware Modeling of Inter-Area Relationships in OD Flow Prediction and Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Origin–destination (OD) flow modeling underpins urban planning and mobility analysis, but prevailing graph-based methods often neglect salient geographic attributes, limiting their ability to model long-range and multi-area dependencies. In this paper, we introduce GeoFlow, a novel framework that (i) augments area representations with geospatial attributes, including relative positions, $k$-hop and geodesic distances, (ii) employs a specialized geometric-intrinsic fusion encoder design that combines graph attention for intrinsic area signals with coordinate-aware encoders for global structure, and (iii) adopts an axial-global attention decoder to capture OD-specific competitive dependencies. For OD flow generation, GeoFlow is paired with flow matching models to produce more authentic and diverse mobility samples. Empirically, GeoFlow achieves superior performance in predictive accuracy, while substantially improving generative fidelity and diversity. Ablation and analytical studies confirm the contribution of each component. Code is open-source and available at this [URL](https://anonymous.4open.science/r/GeoFlow-C4BD).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces GeoFlow, a novel framework for origin-destination (OD) flow prediction and generation that systematically integrates geospatial attributes into a unified encoder-decoder architecture. The key contributions include:
(1) Geospatial feature augmentation with relative position, k-hop distance, and geodesic distance to capture long-range and multi-area dependencies.
(2) A geometric-intrinsic fusion encoder that combines graph attention for local attributes with coordinate-aware encoding for global structure.
(3) An axial-global attention decoder that efficiently models competitive dependencies among OD pairs while reducing computational complexity.

Experiments on the CommutingODGen dataset show that GeoFlow achieves state-of-the-art performance in both prediction and generation tasks, with significant improvements in CPC, RMSE, MAE, and diversity metrics.

### Strengths
1. The integration of geospatial augmentation, geometric-intrinsic fusion encoding, and axial-global decoding is well-motivated.

2. The paper is exceptionally well-written, with clear explanations of motivations, methods, and results.

3. The model achieves significant performance gains while maintaining computational efficiency, making it suitable for real-world deployment.

### Weaknesses
Need more exploration on interpretability.

### Questions
1. The model is evaluated only on commuting data. How well does it generalize to other types of OD flows (e.g., tourism, logistics)?

2. While the axial attention reduces cost, what is the total training/inference time compared to baselines like TransFlower?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents GeoFlow, a new method for origin-destination flow prediction and generation. The main novelty is 3 folds: 1) GeoFlows uses four different geospatial relationship features to jointly capture the origin-destination relation; 2) The Axial-global attention is proposed to perform message passing; 3) The GeoFlow can perform both prediction and generation tasks.

### Strengths
1. The experiments clearly show the advantages of GeoFlow over other methods;
2. A set of ablation studies are conducted to test each component of this model.

### Weaknesses
1. Theretically speaking, the relative position and the straight-line distance have the same information. Why use both in the framework? Another ablation setting is needed, which drops the straight-line distance and only uses the other 3 features for flow prediction.
2. Compared with the baselines, GeoFlow performs Axial-global attentions, which lead to higher computational complexity. Please compare its computational complexity with baselines.
3. Equation 6 and 7 are hard to understand. What do Z^(2k) and Z^(2k+1) mean? The formulas lack definitions, which makes this paper hard to understand.
4. What is the difference between A_{axial} in Equation 4 and A^(L_(a))_{axial} in Equation 8?

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposed GeoFlow model for origin-destination flow map prediction and generation. The model integrates various geo-spatial information, intrinsic local information, and graph information via multiple attention mechanisms and finally produces a set of informative embeddings for each OD pair. These embeddings are then used for supervised OD-flow prediction as inputs and OD-flow generation via a flow-matching model as conditions. The predicted OD-flows provides the best flow value estimations, and the generated samples present possible realizations according to the underlying distribution dynamics. The method achieve the new SOTA of the related tasks. Systematical ablation studies and analysis are conducted to explore the effectiveness of the main model components. A few issues exist in the paper, including model architecture, computational complexity, baseline model implementation, and explanability.  I may change my score if the concerns are properly addressed.

### Strengths
1. The work develops a novel geo-spatial and local feature integration methodogy based on multiple hierachical attention mechanisms. 
2. It achieves the SOTA performance not only in supervised OD-flow prediction but also in unsupervised flow map generation tasks. For the latter, this works presents the first flow-matching model for OD-flow map generation, which holds good performance, training stability, and sample diversity. 
3. Ablation studies reveal the importance of k-hop distance and geodesic distance, which possess unique information that cannot be represented by coordinates.

### Weaknesses
1. The model architecture. The final OD pair embedding Z is given by Eq. (5), which depends on the area embedding given by Eq.(4). These area embeddings defined in Eq. (4) depends on two types of attentions, A_axial and A_global. However, the problem is A_axial and A_global further depend the final embedding Z, as specified in Eqs. (6), (7), and (9), and that makes the forward propagation a circle. Do the authors actually made a typo that the Z in Eqs. (6), (7), and (9) should be the geographic tensor G? Please clarify. 
2. Computational complexity. The authors claim to have lower computational complexity than O(N^4). For A_axial, I understand it can be reduced to O(N^3). But for A_global, it is not clearly explained how to make it smaller than O(N^4). Please provide better explanation. 
3. The Transflower baseline. Transflower only uses local features and coordinates without the information of k-hop distance and geodesic distance, Is the performance advantage of GeoFlow mainly because it uses more information? In Table2, does ID 2 corresponds to the same amount of information used by Transflower? If yes, then the model actually doesn't show advantage. If not, please provide this ablation. It is important to distinguish the gains from better model design and from more information. 
4. Explanability of attention map. The attention map of Transflower has clear correlation to the points of interests. What about A_axial and A_global of GeoFlow?
5. Explanability to performances. In Fig. 6 in appendix, the first two rows show decent similarity between the ground truth and both the prediction and the generation samples. But the last two rows seem to show the prediction and generation are poor. Do we know in what case (what geo or intrinsic properties of the areas) the method can or can not work well?

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces GeoFlow, a unified framework for flow prediction and generation. GeoFlow integrates geospatial awareness into machine learning models. In particular: (i) it augments area representations with geospatial attributes (e.g., relative positions), (ii) it uses a special encoder module designed to process and combine intrinsic and geometric features, (iii) it relies on an axial–global attention decoder that let the model to understand how trips from the same origin or to the same destination influence each other. GeoFlow is evaluated on existing OD benchmarks and tested against the main SOTA models. GeoFlow outperforms in prediction accuracy over baselines and shows some gain (7.4% relative) in reconstruction accuracy for generation.

### Strengths
The paper presents several novelties respect to previous works. For example, the fusion of geometric and intrinsic attributes with flow matching is new in OD modeling. Results are also convincing and the SOTA baselines are take into consideration for the experimental setup.

### Weaknesses
While the paper is in-depth and clear for prediction, it lacks of analysis for the generation part. There is a modest gain and I would like to see some inspection or better analysis that justify the improvement (even though is statistically significant, it is not clear where this improvement is).

Scalability is underexplored as well as interpretability and generalization. For example, the geography transferability is not inspected, leaving doubts on the generalization abilities of the model (authors can refer to DeepGravity paper to see examples).

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
