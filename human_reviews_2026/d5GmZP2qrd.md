# HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation

- Decision: Accept (Poster)
- Scores: 4, 4, 2, 6

## Abstract
Visual geolocalization, the task of predicting where an image was taken, remains challenging due to global scale, visual ambiguity, and the inherently hierarchical structure of geography. Existing paradigms rely on either large-scale retrieval, which requires storing a large number of image embeddings, grid-based classifiers that ignore geographic continuity, or generative models that diffuse over space but struggle with fine detail. We introduce an entity-centric formulation of geolocation that replaces image-to-image retrieval with a compact hierarchy of geographic entities embedded in Hyperbolic space. Images are aligned directly to country, region, subregion, and city entities through Geo-Weighted Hyperbolic contrastive learning by directly incorporating haversine distance into the contrastive objective. This hierarchical design enables interpretable predictions and efficient inference with 240k entity embeddings instead of over 5 million image embeddings on the OSV5M benchmark, on which our method establishes a new state-of-the-art performance. Compared to the current methods in the literature, it reduces mean geodesic error by 19.5\%, while improving the fine-grained subregion accuracy by 43\%. These results demonstrate that geometry-aware hierarchical embeddings provide a scalable and conceptually new alternative for global image geolocation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper trains on two datasets (OSV5M and MediaEval’16) but does not include an ablation showing how each dataset affects performance.
Previous geolocation works (e.g., PIGEON, RFM S, GeoDecoder) trained and evaluated using only one dataset, ensuring fair comparison.

Because HierLoc uses multiple datasets—OSV5M for one benchmark and MediaEval’16 for others—the reported gains may partly come from additional training data diversity, not just from the proposed hyperbolic embedding method.
Without a controlled experiment (e.g., training only on OSV5M and testing cross-dataset), it is unclear whether improvements are architectural or data-driven.

### Strengths
The paper is clearly written, with a logical flow.
The figures effectively illustrate the hierarchical process.
Equations are well-defined and consistent with hyperbolic geometry notation.
Introduces the first explicit use of hyperbolic embeddings for visual geolocation, representing geographic hierarchies as hyperbolic manifolds.
The Geo-Weighted Hyperbolic InfoNCE (GWH-InfoNCE) loss incorporating haversine distance is novel and well-justified.
Reformulating geolocation as image-to-entity alignment instead of image-to-image retrieval is conceptually new.

### Weaknesses
1. Some hyperparameters are found empirically, more justification or sensitivity plots would be valuable.

2. The paper employs two large-scale datasets (OSV5M and MediaEval’16) for training but does not provide ablations isolating their effects. Prior works (e.g., PIGEON, GeoDecoder, RFM) train and evaluate using a single dataset, ensuring fair comparison. In contrast, HierLoc benefits from additional data diversity, which may contribute to improved cross-dataset generalization. Without an ablation where HierLoc is trained solely on OSV5M or MediaEval’16 and evaluated on the same benchmarks, it is unclear how much of the reported gains are due to the proposed method versus the increased data volume.

I recommend adding an ablation or at least clarifying which datasets were used for each benchmark evaluation, and ensuring fair comparison with prior work under identical training data conditions.

3. The field of visual geolocation is niche within ICLR’s broader ML audience; the paper’s impact may depend on perceived relevance beyond geolocation (e.g., general multimodal hierarchy learning).

### Questions
None

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
The paper presents hierloc, a new framework for worldwide image geolocalization that reformulates the task as aligning images to a compact hierarchy of geographic entities, embedded in hyperbolic spaces. Rather than image-to-image retrieval, hierloc computes and aligns image embeddings to entity prototypes via contrastive learning, directly incorporating haversine distance to weight negatives (GWH-InfoNCE). This allows efficient hierarchical beam search inference, interpretability, and strong performance. Empirical results on the OSV5M benchmark and others demonstrate improved mean geodesic error and classificationn accuracies relative to prior methods, while reducing storage and computation requirements.

### Strengths
1. The paper is clearly presented, and the reader can smoothly follow the authors’ logic.
2. The idea is relatively novel, and alignment itself is a challenging problem.

### Weaknesses
1. The related work section misses several relevant studies (e.g., GeoReasoner [1] , Img2Loc [2]). I recommend that the authors do a more thorough survey.
2. In Table 2, which backbone is used for HierLoc? Is the comparison with other methods fair?
3. The code and data will only be released after acceptance.
4. Similar to the first point, the experimental section also lacks comparisons with strong baselines. The current results still show a gap from the state-of-the-art on some metrics.

Overall, I think this paper is well-presented, and the experiments are generally self-consistent. However, the lack of strong and up-to-date baselines is quite noticeable. If the authors can include comparisons with more recent baselines in this field, I would consider increasing my score.

[1] Li, Ling, et al. "Georeasoner: Geo-localization with reasoning in street views using a large vision-language model." Forty-first International Conference on Machine Learning. 2024.
[2] Zhou, Zhongliang, et al. "Img2Loc: Revisiting image geolocalization using multi-modality foundation models and image-based retrieval-augmented generation." Proceedings of the 47th international acm sigir conference on research and development in information retrieval. 2024.

### Questions
please refer to the weakness part

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
This paper is an incremental work on traditional retrieval-based geolocalization methods. Instead of performing contrastive learning between a query image and a full gallery of candidate locations, the authors propose to aggregate the candidate locations into a hierarchy of entities (country, region, subregion, city) and perform hierarchical contrastive learning in the hyperbolic space. This is an intuitive approach since it mimics the human spatial reasoning patterns -- narrow down from large-scale to fine-grained guesses. It shows some competitive performance against other baselines, while suffers from generalizability problems.

### Strengths
1. A hierarchical representation of locations and a beam search through hierarchical entities is a very natural and geographically sound alternative to traditional single-location-based retrieval geolocation methods.

2. Using hyperbolic embedding to represent hierarchy is very efficient.

3. Geo-Weighted InfoNCE introduces geo-awareness into the contrastive learning loss.

### Weaknesses
1. The key weakness, which severely restricts the generalizability of the method, is that the proposed entity hierarchy relies heavily on the **coverage** of the dataset. That is, if a location in the test set never appears in any neighborhoods of an entity in the training dataset, the model will never be able to predict its location. This is no rare case -- both MP16 and OSV5M are highly spatially biased, i.e. most data concentrate in North America and Western Europe. This is already a known problem in traditional retrieval based methods such as GeoCLIP (see a recent NeurIPS paper https://openreview.net/pdf/c2d943add9cd78700f9acc1101286c2082105a70.pdf, Section 5.2 and Appendix A.7), and the entity hierarchy which simplifies 4 million candidate locations into 240k entities only makes the problem worse. In other words, the proposed method, on its very basis, can not handle spatially out-of-distribution cases. This is a huge weak point compared to recently developed generative geolocation models (RFM, LocDiff) which naturally generalizes to arbitrary locations on Earth.

This problem is covered in the paper because the authors only performed testing on the same datasets the models are trained -- i.e., if a model is trained on the OSV5M training dataset, it is evaluated on the OSV5M test dataset. It "hides" the generalizability problem.

2. Errors in citations. Too many citations are in wrong formats -- e.g. throughout the paper, most citations are not properly put into parentheses.

3. From Appendix A.2, $\exp_0$ and $\log_0$ should be $\exp_o$ and $\log_o$, because here the $o$ represents the origin *point*, not the number $0$.

### Questions
1. Have you ever tested with cross-dataset experiments? For example, in GeoCLIP, the model trained on MP16 is evaluated on Im2GPS3k and YFCC, and obviously YFCC has much weaker performance because its data distribution is different from MP16. In your experiments, all test data seem to be a left-out of the training data. This will cover your generalizability issue, since the test locations are likely to align with the entity hierarchy you build from the training data.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes to solve **geolocation as a hierarchical retrieval problem**. By leveraging different levels of administrative segmentation, they assign one embedding per entity and then perform **hierarchical retrieval to achieve efficient retrieval** while achieving **SOTA performance**.

### Strengths
1. The paper showcases **extensive experiments** to back up its claims.

2. The method's inference structure allows for **very efficient inference**.

3. The method is showcased on two datasets: **OSV-5M** (street view focused) and **MediaEval** (more generalist).

4. The authors achieve **SOTA performances on OSV-5M**.

### Weaknesses
1. **Data Curation:** It is not clear, but **entities seem to be learned *across* datasets** (per Section 3.2). If true, this **makes the results not comparable**, as the model is trained on significantly more data. This also potentially **breaks the data decontamination** for OSV-5M (1km exclusion zone). I would like to see **results with entities computed separately for each dataset**.

2. **Mean Embeddings for Large Regions:** Taking the mean embedding may be suitable for fine-grained regions, but it **likely doesn't make sense for wider regions** like countries. I would like to see an **ablation study sequentially removing these mean embeddings** for countries, regions, sub-regions, and cities.

3. **Importance of the Encoder:** The paper **fails to ablate the importance of the feature extractor**. For all we know, it’s **DinoV3 that is responsible for the improvement**. **Networks used by other SOTA methods should be ablated** as well (e.g., StreetCLIP on OSV, DinoV2 on YFCC for RFM S2).

4. **Lack of Qualitative Samples:** I would have loved to see some **visualisation of the inference process**. (Perhaps drawing the beam search trees on a map with colors for scores?) It would also beD; good to be able to visualise some final predictions.

5. **Efficiency Comparison:** I would like to see the **inference efficiency compared to non-retrieval methods** as well.

### Questions
**Motivation for Hyperbolic Embeddings:** The paper **struggles to clearly explain the intuition for using hyperbolic embeddings**. I understand that it works better, but what is the **underlying motivation**?

### Soundness
4

### Presentation
3

### Contribution
3
