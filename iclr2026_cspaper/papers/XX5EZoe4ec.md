# Retrievalformer: A Dual-Encoder Trans- Former For Efficient Approximate Nearest Neighbor Retrieval And Cold-Item Recommen- Dation

Anonymous authors Paper under double-blind review 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 We propose RetrievalFormer, a transformer-based dual-encoder recommender architecture that combines competitive accuracy with strong transformer-based sequential baselines, efficient Approximate Nearest Neighbor (ANN) retrieval, and the ability to score feature-described items that are unseen during training. Our architecture uses an attention-based heterogeneous feature encoder that aggregates item and user attributes via shared embedding tables and an AttentionFusion module, so that the resulting user and item representations lie in a shared embedding space suitable for ANN search. On Amazon and MovieLens benchmarks, RetrievalFormer achieves competitive recommendation accuracy, reaching 86–91%
of the Recall@20 of strong transformer-based sequential baselines while enabling up to 288× lower latency at a 10M-item scale via ANN retrieval. On MovieLens1M, RetrievalFormer attains Recall@20 of 0.337. In cold-start experiments where entire items and all of their interactions are held out during training, Retrieval- Former successfully recommends completely unseen items from their features in a leave-one-out cold (LOOC) protocol with zero item leakage between training and evaluation, in which ID-softmax transformer baselines cannot produce scores for such items at all, and it outperforms a strong content-based baseline on a 100% cold-start production dataset. Our approach enables practical deployment of efficient recommendations at scale, offering a compelling trade-off between model accuracy and serving efficiency.

## 1 Introduction

Transformer-based sequential recommenders (e.g., SASRec, BERT4Rec) have achieved state-ofthe-art accuracy in next-item prediction by leveraging self-attention over user behavior sequences (Kang & McAuley, 2018; Sun et al., 2019). These models treat recommendation as a classification over all items in the catalog: given a sequence of past items, the transformer produces a probability distribution over the entire item vocabulary for the next item (Vaswani et al., 2017). While effective, this approach has two key shortcomings in real-world settings. This ID-softmax formulation simultaneously causes two related issues. First, the output layer must compute scores for all N items in the catalog, incurring an O(N d) cost per prediction that dominates the O(L
2d) cost of self-attention once N is large. Second, because the output layer contains one parameter vector per training-time item ID, items whose IDs never appear in training cannot be scored at all, even if rich item features are available at inference time.

First, scoring all items via a full softmax is computationally expensive for large catalogs. Serving such models in production requires scanning through millions of item embeddings for each prediction, leading to high latency and resource costs (Su et al., 2023). For example, Kersbergen et al. (2024) report that a transformer model with a 20-million item catalog required multiple high-end GPU servers to meet a 50ms p90 latency, incurring thousands of dollars per month in deployment cost (Kersbergen et al., 2024).

1

## Abstract

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Second, classical transformers struggle with item cold-start: new items cannot be effectively recommended until the model is retrained or updated (Volkovs et al., 2017; Schein et al., 2002). In dynamic domains with rapid item churn (e.g., news or ephemeral marketing content), the delay in recommending new items is problematic. We propose RetrievalFormer, a dual-encoder sequential recommender that directly addresses both of these issues by reframing next-item recommendation as a retrieval problem. A transformer-based user tower encodes the interaction history into a user embedding, and a feature-based item tower encodes items from their attributes into item embeddings, with both towers trained jointly so that recommendations are produced by dot-product similarity in a shared embedding space rather than a softmax over item IDs. By decoupling user and item representations, our model enables efficient retrieval: at serving time, a user's latest interaction sequence is encoded into a query embedding, and the top-K candidate items are retrieved by Approximate Nearest Neighbor (ANN) search in the item embedding space, instead of computing a full softmax over all items. Concretely, the relevance score between a user u and an item i is

## S(U, I) = Fu(Historyu ) ⊤Fi(Featuresi),

and at serving time we retrieve the top-K items for a given user by performing ANN search over the pre-computed item embeddings fi(·). This retrieval paradigm leverages highly optimized ANN indexes (e.g., HNSW graphs or vector quantization) to find top candidates in sub-linear time (Johnson et al., 2019; Malkov & Yashunin, 2018), circumventing the costly softmax over the entire catalog. In essence, RetrievalFormer achieves transformer-like recommendation quality while operating at the speed of ANN retrieval. Moreover, the item tower directly computes representations from item content and attributes, so new items can be recommended zero-shot, addressing the item cold-start problem without any retraining or extension of the model's vocabulary. Our approach also introduces an attention-based heterogeneous feature encoder to enrich both user and item representations. Modern recommender data is heterogeneous, with information such as item text descriptions, categories, images, and contextual tags, as well as user profile features. Rather than using only IDs or simple feature concatenation, we apply a self-attention fusion mechanism to each set of features describing an entity (an item or an interaction). This allows the model to learn complex interactions between different feature modalities in a data-driven way. For example, an item's textual description and its category label can attend to each other to produce a more informative item embedding. This design draws inspiration from Set Transformer architectures (Lee et al., 2019) and feature interaction learning (Song et al., 2019), enabling permutation-invariant aggregation of arbitrary feature sets. Importantly, this attention fusion mechanism is used throughout our architecture, in the item tower for combining item metadata, in the user interaction history for fusing features of historical items, and in the user tower for processing the resulting token sequence. We further share embedding lookup tables for features across the user and item towers, so that a feature (e.g., a brand ID or a word embedding) has a consistent representation regardless of where it is used. This weight sharing improves training efficiency and alignment between the two towers, as the model can leverage the same semantic signal in multiple contexts. We validate RetrievalFormer on standard benchmarks, finding competitive accuracy versus transformers while achieving 288× speedup at 10M items. Our contributions: (1) a two-tower architecture achieving competitive accuracy with efficient ANN retrieval, (2) attention fusion for heterogeneous features outperforming simple pooling, (3) zero-shot cold-start capability through feature-based encoding, and (4) rigorous evaluation demonstrating practical trade-offs between accuracy and efficiency.

## 2 Related Work

Sequential Recommendation and Transformers. Sequential recommenders model the dynamic sequence of user-item interactions to predict a user's next interest. Early approaches used Markov Chains or RNNs (Hidasi et al., 2015; Li et al., 2017; Tang & Wang, 2018; Wu et al., 2017), but recent advances are dominated by self-attention mechanisms. SASRec (Kang & McAuley, 2018) introduced the use of unidirectional Transformer encoder layers to capture which previous items in the sequence are relevant for predicting the next one. Variants like BERT4Rec extended this with 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 We present RetrievalFormer, a dual-encoder architecture that achieves competitive sequential recommendation accuracy compared to strong transformer-based baselines while enabling efficient ANN-based retrieval. Our approach addresses the fundamental scalability limitation of transformer recommenders, the O(N) inference cost of scoring all items, by decoupling item representations from user sequence modeling. This section describes our architecture design, the attention fusion mechanism for heterogeneous features, and the training methodology that enables both accuracy and efficiency. bidirectional transformers and a Cloze task for training (Sun et al., 2019). These models learn item embeddings and position embeddings, and use multi-head attention to capture long-range dependencies in user behavior. While very effective in accuracy, a core limitation is that they produce predictions by a softmax over the entire item vocabulary at each time step (Kang & McAuley, 2018; Sun et al., 2019). This does not scale well to large catalogs due to the computational cost and memory footprint of the output layer. Recent work has noted the inference bottleneck of such models: even with optimizations, serving a transformer sequential model for millions of items can be prohibitively slow or costly (Su et al., 2023). Two-Stage and Retrieval Models in Recommenders. In industry-scale recommender systems, a common solution is a two-stage pipeline: first retrieve a set of candidates, then apply a more precise ranking model (Covington et al., 2016). The candidate retrieval stage often uses lightweight models (e.g., matrix factorization (Krichene & Rendle, 2020) or two-tower neural networks (Yi et al., 2019)) that can handle a very large item pool efficiently (Yi et al., 2019; Huang et al., 2020a; Eksombatchai et al., 2018; Grbovic & Cheng, 2018). Our work follows this paradigm in spirit: RetrievalFormer's user and item towers correspond to a learned retrieval model producing candidate item embeddings. The key difference is that we aim to approach the accuracy of strong transformer-based sequential models in the retrieval stage itself, rather than using a simplistic retriever, effectively collapsing the quality of a powerful sequential model into an ANN-friendly form. Attribute-Enriched and Cold-Start Recommendation. Another line of related work is utilizing content features and attributes to improve recommendations, especially under sparse data or coldstart scenarios (Schein et al., 2002; Zhou et al., 2022; de Souza Pereira Moreira et al., 2021; Pancha et al., 2022). Many recommender models have been extended to incorporate side information such as item descriptions, knowledge graph entities, or user profile data. For sequential recsys, recent methods like AttrFormer explicitly model item attributes alongside IDs in the attention mechanism (Liu et al., 2025). AttrFormer augments an ID-softmax transformer with attribute-aware attention and achieves strong accuracy on Amazon benchmarks, but it still predicts over a fixed item vocabulary and cannot score items whose IDs never appear during training. In contrast, RetrievalFormer decouples item representations into a feature-based item tower within a dual-encoder design, enabling direct scoring of unseen items from their attributes and making the model naturally compatible with ANN retrieval. Approximate Nearest Neighbors for Recommendation. Fast ANN search has seen rapid progress, with algorithms like IVF, HNSW, and PQ enabling vector search on billions of points within milliseconds (Johnson et al., 2019; Malkov & Yashunin, 2018). Our contribution ensures using ANN does not sacrifice recommendation quality, by training to produce a discriminative embedding space, we achieve both high accuracy and low latency. We do not propose a new ANN algorithm in this work; rather, we design a sequential recommender whose dual-encoder architecture and training objective produce an embedding space that is well aligned with standard ANN indexes such as IVF-PQ and HNSW, so that they can be used for serving without sacrificing recommendation quality. A large body of work aims to reduce the computational cost of neural recommendation models through techniques such as sampled or approximate softmax, model compression and distillation, and explicit two-stage candidate-generation plus re-ranking pipelines. These approaches reduce the effective number of items scored or the size of the re-ranking model, but they typically retain an ID-softmax formulation or a separate heavy re-ranker. By contrast, RetrievalFormer reformulates sequential recommendation itself as a dual-encoder retrieval problem so that efficient ANN search over item embeddings becomes the native serving mechanism.

## 3 Methodology

162

![3_image_0.png](3_image_0.png) 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

## 3.2 Attention Fusion For Heterogeneous Features 3.2.1 Feature Fusion Mechanism 3.1 Overall Architecture

RetrievalFormer employs a dual-encoder design with asymmetric towers optimized for their respective roles (Figure 1). The item tower fi(·) encodes each item's heterogeneous features into a dense embedding y ∈ R
dthat can be pre-computed and indexed. The user tower fu(·), implemented as a transformer, processes the user's interaction sequence to produce a query embedding x ∈ R
d.

At serving time, recommendations are generated through approximate nearest neighbor search for items y that maximize x
⊤y, avoiding the computational bottleneck of exhaustive scoring.

The key insight is that by learning a shared embedding space through contrastive training, we can leverage the same representations for both training (via InfoNCE loss) and inference (via ANN retrieval). This design choice changes the effective scaling from O(N) to empirically sub-linear growth in practice when using standard ANN indexes, while maintaining recommendation quality, as we demonstrate in Section 4.

Formally, let xu = fu(hu) denote the user embedding produced by the user tower from the interaction history hu, and let yi = fi(zi) denote the item embedding produced by the item tower from the item features zi. The relevance score is s(*u, i*) = x
⊤ u yi, and the top-K recommendations for user u are the K items with the largest values of x
⊤ u yi, found via ANN search over {yi}.

Modern recommendation systems must handle diverse feature types: text descriptions, categorical attributes, numerical values, and interaction signals. We introduce an attention-based fusion mechanism that learns to dynamically weight and combine these heterogeneous features, moving beyond simple concatenation or averaging approaches.

Given features F = {f1*, ..., f*M} describing an entity, we embed each feature using shared lookup tables and project to a common dimension:
H = [W1Ef1(f1); ...;WMEfM (fM)] ∈ R
M×d(1)

In our experiments, F includes single-valued categorical features (e.g., item category, brand), multivalued categorical features (e.g., tags), and text-derived features (e.g., token IDs from titles or descriptions). Single-valued categorical features are encoded as a single embedding vector per feature, while multi-valued features are encoded by aggregating the embeddings of all values for that feature (either via mean pooling or attention). Text features are treated as multi-valued categorical features over a token vocabulary. AttentionFusion applies multi-head self-attention over the set of feature embeddings for an item or user, followed by pooling, to produce a single fixed-dimensional representation. We apply multi-head self-attention (Vaswani et al., 2017) with residual connections and layer normalization:
$$\begin{array}{l}{{\mathbf{Z}=\mathrm{LayerNorm}(\mathbf{H}+\mathrm{MultiHeadAttn}(\mathbf{H},\mathbf{H},\mathbf{H}))}}\\ {{\mathbf{U}=\mathrm{LayerNorm}(\mathbf{Z}+\mathrm{FFN}(\mathbf{Z}))}}\end{array}$$
z = MeanPool(U) ∈ R
d(4)
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

## 3.4 User Tower: Transformer Over Enriched Sequences

The user tower processes interaction sequences through a transformer encoder, but critically, each sequence element is an enriched representation combining item features with interaction context. 3.4.1 INTERACTION REPRESENTATION

$\eqref{eq:walpha}$. 
$\eqref{eq:walpha}$. 
This mechanism is permutation-invariant and handles variable-length feature sets, learning complex feature interactions through attention weights. The same fusion architecture is applied consistently at three levels: (1) item metadata fusion, (2) interaction context fusion, and (3) user profile fusion.

## 3.2.2 Shared Embedding Design

A critical design choice is sharing embedding tables across towers. When a categorical feature (e.g., "electronics") appears in different contexts, as an item category, user preference, or interaction attribute, it uses the same embedding vector. This parameter sharing reduces parameters by approximately 3× in our implementation, enables knowledge transfer between representations, improves cold-start generalization, and ensures consistent feature semantics. Sharing embedding tables across user profile, item metadata, and interaction history for the same feature types not only reduces the number of parameters, but also encourages a consistent semantic space for these features. This is particularly important for cold-item generalization, since it allows the model to interpret the same attribute (e.g., a brand or category) consistently whether it appears in a user profile, a historical interaction, or a newly introduced item.

## 3.3 Item Tower: Feature-Based Encoding

The item tower computes dense embeddings from item features. For item i with features Fi =
{f
(i)
1
, ..., f(i)
M }:
yi = AttentionFusion(Fi) ∈ R
d(5)
This feature-based design enables zero-shot generalization where new items receive embeddings immediately from their features. The tower leverages shared feature embeddings and fusion weights, providing scalability and robustness through graceful handling of missing features via attention masking. Because the item tower depends only on item-side features and not on user history, we can precompute yi = fi(zi) for all items offline and build an ANN index over these embeddings, so that online serving only needs to compute the user embedding xu and perform an ANN query.

$\mathbf{h}_{i_t}=\text{AttentionFusion}(\text{ItemFactures}(i_t))$. 
hit = AttentionFusion(ItemFeatures(it)) (6)
For each historical interaction et involving item it, we create enriched tokens through two-stage fusion:

## 3.4.2 Sequence Construction

The transformer processes the sequence:
270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 4 Experiments

$$\mathbf{S}=[\mathbf{z}_{1},...,\mathbf{z}_{T},[\mathrm{SEP}],\mathbf{p}_{u},[\mathrm{CLS}]]$$
$$({\mathfrak{s}})$$
S = [z1*, ...,* zT , [SEP], pu, [CLS]] (8)
where pu = AttentionFusion(UserFeatures) encodes static user attributes. The final [CLS] token representation, after transformer processing with causal masking, becomes the user embedding xu.

This design enables the transformer to model sequential patterns over semantically rich tokens, improving both accuracy and generalization.

## 3.5 Training Methodology

We train RetrievalFormer using contrastive learning to learn a shared embedding space where users and their next items are close while being far from non-relevant items. For a batch of B user-item pairs with embeddings {(xi, yi)}
B
i=1, we optimize the InfoNCE loss (Oord et al., 2018):

$${\mathcal{L}}_{\mathrm{InfinANCE}}=-{\frac{1}{B}}\sum_{i=1}^{B}\log{\frac{\exp(\mathbf{x}_{i}^{\top}\mathbf{y}_{i}/\tau)}{\sum_{j=1}^{B}\exp(\mathbf{x}_{i}^{\top}\mathbf{y}_{j}/\tau)}}$$

$$({\mathfrak{g}})$$

where τ is a temperature hyperparameter. This objective treats all other items in the batch as negatives, efficiently approximating the full softmax over the catalog.

To address popularity bias and improve coverage of tail items, we employ Mixed Negative Sampling (MNS) (Yang et al., 2020), augmenting each batch with uniformly sampled items from the catalog. This ensures diverse negative signals across the entire item distribution, preventing the model from over-optimizing on popular items while neglecting rare ones. The combination of InfoNCE and MNS is particularly important for RetrievalFormer's training. The contrastive objective encourages both alignment between user and positive item embeddings and uniformity of the overall embedding distribution on the hypersphere (Wang & Isola, 2020), which helps to avoid representation collapse and makes the learned space more suitable for ANN retrieval. A brief discussion of these alignment and uniformity properties, together with implementation considerations for mixed negative sampling, is provided in Appendix C. We structure our experimental evaluation around four research questions: RQ1 examines whether RetrievalFormer can achieve competitive recommendation accuracy compared to state-of-the-art transformer-based sequential models on standard benchmarks. RQ2 investigates how the heterogeneous feature inputs and architectural choices (attention fusion, shared embeddings, context tokens) contribute to the model's performance. RQ3 evaluates how well RetrievalFormer handles unseen items and whether it can effectively recommend items that were never in the training set. RQ4 measures the inference efficiency of RetrievalFormer using ANN search and compares it to a conventional Transformer model that scores all items. where ⊕ denotes feature concatenation and InteractionContext includes interaction type (click, purchase), explicit feedback (ratings), and contextual signals (device, timestamp). This two-stage process captures not just *what* items were interacted with, but how and *when*. In all of our experiments, we use the same number of transformer layers and hidden dimension as in the corresponding transformer baselines (e.g., SASRec and BERT4Rec) so that differences in accuracy are attributable to the dual-encoder formulation rather than model capacity; detailed hyperparameters are provided in Section 4.1 and Appendix J.

## Zt = Attentionfusion(Hit ⊕ Interactioncontext(Et)) (7) 4.1 Experimental Setup

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Datasets: We evaluate on three public datasets used in prior Transformer recommender research: Amazon Beauty, Amazon Toys & Games, and MovieLens-1M. For Amazon, we use the sequential rating/review data from McAuley et al. (2015), focusing on users with at least 5 interactions. For MovieLens-1M, we use the 1 million movie ratings, treating a rating as an implicit interaction. For RQ1, RQ2, and RQ4, we use the same data splits, features, and preprocessing as Liu et al. (2025) for direct comparability, where a standard leave-one-out (LOO) approach holds out the last item for testing, the second-to-last for validation, and the remainder for training. For RQ3 (coldstart evaluation), we use a separate Leave-One-Out Cold (LOOC) protocol that ensures test items are completely absent from training; see Section 4.4 and Appendix F for details. We also use a proprietary Email Campaign dataset as a case study for extreme item cold-start (each "item" is a marketing email, and new campaigns launch daily with no historical interactions). Baselines: For RQ1, we compare RetrievalFormer to representative sequential recommenders: SASRec (Kang & McAuley, 2018), BERT4Rec (Sun et al., 2019), GRU4Rec (Hidasi et al., 2015), and the recent AttrFormer (Liu et al., 2025). We adopt the experimental protocol and baseline results from Liu et al. (2025) for fair comparison. For cold-start experiments (RQ3), standard baselines cannot generate scores for new items, so we compare against a Content-based KNN approach. For RQ4, the baseline is SASRec served in the traditional way (computing softmax scores over all items). In total, we compare RetrievalFormer against 12 baseline models across three public benchmarks (Amazon Beauty, Amazon Toys & Games, MovieLens-1M), including the recent AttrFormer model introduced at KDD 2025, as well as on a fourth production dataset (email campaigns) described in Appendix G. Metrics: We report Recall@20 and NDCG@20 for each model on the test sets, considering the ground-truth next item for each user. For cold-start evaluation, we report Hit Rate@20 for new-item recommendations. For efficiency (RQ4), we measure query latency (in milliseconds) and throughput (queries per second) under various conditions. Hyperparameters: Unless otherwise noted, we train all models with the Adam optimizer, batch size 512, sequence length truncated to L = 50, and an initial learning rate of 1 × 10−3 with cosine decay. RetrievalFormer and transformer baselines share the same transformer depth and hidden size on each dataset. For dual-encoder models, we use one in-batch negative per positive example unless otherwise noted, and we train for up to 100 epochs with early stopping on validation Recall@20.

Additional hyperparameters and implementation details are provided in Appendix J.

## 4.2 Rq1: Retrievalformer Vs. Transformer Baselines

Table 1 demonstrates RetrievalFormer achieves competitive performance with established transformer baselines while enabling massive efficiency gains. On Amazon Beauty, RetrievalFormer (0.1208) outperforms SASRec (0.1107) and achieves 91.2% of AttrFormer's performance. On Amazon Toys, we achieve comparable results to MT4SR (0.1169 vs 0.1148). On MovieLens-1M, RetrievalFormer attains Recall@20 of 0.337, narrowing the gap to the strongest transformer baselines. This result represents 96.8% of SASRec's performance (0.3483) and is wellaligned with the established baseline cluster. This modest accuracy trade-off enables a transformative 288× speedup at 10M items, making transformer-quality recommendations practical for industrial deployment. Notably, on MovieLens-1M, most established transformer methods achieve Recall@20 in the range of 0.34-0.36 (SASRec: 0.3483, GRU4Rec: 0.3579, LightSANs: 0.3590), with RetrievalFormer at 0.337 achieving 96.7% of SASRec's performance. AttrFormer's result of 0.4128 represents a notable outlier, achieving approximately 15% higher recall than the next best established method. When compared to the established baseline cluster, RetrievalFormer demonstrates competitive performance while enabling dramatic efficiency improvements.

It is important to note that the modest accuracy trade-off is not due to inferior transformer sequence modeling, but rather the fundamental difference between scoring all items via softmax versus dual-encoder retrieval. RetrievalFormer maintains the powerful transformer architecture for user sequence modeling; the performance gap stems from replacing the exact softmax scoring over all items with approximate nearest neighbor search in the learned embedding space. This architectural choice 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

| Lens. Best results in bold. RetrievalFormer results are from our experiments. Dataset Metric Transformer: N.A. for Attribute Transformer: With Attribute Input GRU4Rec DuoRec SASRec BERT4Rec CL4SRec LightSANs FEARec TiSASRec SASRecF MT4SR DIF-SR AttrFormer RetrievalFormer Recall@5 0.0349 0.0642 0.0556 0.0382 0.0392 0.0561 0.0594 0.0576 0.0587 0.0559 0.0578 0.0657 0.0529 Recall@20 0.0817 0.1132 0.1107 0.0783 0.0742 0.1222 0.1239 0.1244 0.1231 0.1169 0.1273 0.1324 0.1208 Amazon Beauty NDCG@5 0.0231 0.0330 0.0343 0.0265 0.0217 0.0342 0.0337 0.0344 0.0413 0.0360 0.0337 0.0446 0.0351 NDCG@20 0.0362 0.0447 0.0540 0.0378 0.0296 0.0528 0.0520 0.0534 0.0594 0.0533 0.0535 0.0639 0.0541 Recall@5 0.0271 0.0651 0.0600 0.0364 0.0324 0.0632 0.0674 0.0666 0.0585 0.0607 0.0675 0.0720 0.0522 Amazon Recall@20 0.0654 0.0860 0.1073 0.0691 0.0595 0.1273 0.1297 0.1325 0.1217 0.1148 0.1342 0.1357 0.1169 Toys & Games NDCG@5 0.0175 0.0339 0.0435 0.0265 0.0183 0.0370 0.0379 0.0379 0.0393 0.0410 0.0380 0.0501 0.0346 NDCG@20 0.0368 0.0392 0.0570 0.0356 0.0244 0.0552 0.0557 0.0566 0.0571 0.0563 0.0569 0.0681 0.0528 Recall@5 0.1752 0.1477 0.1854 0.1341 0.1395 0.1840 0.1372 0.1816 0.1829 0.1854 0.1518 0.2258 0.1312 MovieLens Recall@20 0.3579 0.2538 0.3483 0.2728 0.2284 0.3590 0.3097 0.3558 0.3553 0.3483 0.3195 0.4128 0.337 NDCG@5 0.1172 0.0947 0.1285 0.1120 0.0535 0.1226 0.1285 0.1216 0.1239 0.1285 0.0964 0.1554 0.0823 1M NDCG@20 0.1687 0.1638 0.1745 0.1311 0.0990 0.1725 0.1320 0.1711 0.1726 0.1745 0.1440 0.2088 0.1390   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

enables the dramatic efficiency gains that make transformer-quality recommendations practical at scale. The key advantage of RetrievalFormer is inference speed: exhaustive scoring takes 3.4ms at 100K items and 29.5ms at 1M items, while RetrievalFormer with ANN achieves 0.58ms and 0.69ms respectively, yielding a 43× speedup at 1M items that grows to 288× at 10M items.

## 4.3 Rq2: Ablation Studies Of Model Components

To understand the impact of our design choices and hyperparameters, we conduct comprehensive ablation experiments on the Amazon Toys & Games dataset. We examine both architectural components and hyperparameter sensitivity (detailed results in Appendix Table 3).

## 4.3.1 Architectural Components

Attention Fusion: Self-attention fusion outperforms simple mean pooling, improving Recall@20 from 0.0960 to 0.1057 (+10.1%). This confirms that learning dynamic feature interactions through attention provides meaningful gains over treating all features equally. Shared Embeddings: Using shared embedding tables across towers improves Recall@20 by approximately 3% on MovieLens-1M, validating our hypothesis that semantic consistency between user and item representations enhances learning. Uniformity Loss: Enabling implicit uniformity through InfoNCE provides consistent improvements across all metrics (Recall@20: 0.1022 → 0.1064, +4.1%), confirming its role in preventing representation collapse during training. Hyperparameter sensitivity analysis (history length, batch size, embedding dimensions) is provided in Appendix E. Key findings include a non-monotonic relationship with sequence length where performance jumps at L=25, and larger batch sizes consistently improving InfoNCE training. These ablations demonstrate that attention fusion and appropriate history length are particularly critical for achieving competitive accuracy.

## 4.4 Rq3: Cold-Start Item Recommendation With Leave-One-Out Cold Evaluation

Real-world recommender systems face a fundamental challenge: new items arrive continuously but have no interaction history. Traditional evaluation protocols fail to capture this reality, they test on items seen during training, just with different user-item pairs held out. We propose Leave-One-Out Cold (LOOC), a rigorous evaluation protocol that tests recommenders on truly unseen items.

## 4.4.1 Leave-One-Out Cold (Looc) Protocol

LOOC extends standard leave-one-out evaluation by ensuring test items are completely absent from training, with no item ID leakage between training and evaluation. Our protocol constructs the 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

Table 2: RetrievalFormer performance comparison between standard Leave-One-Out (LOO) and Leave-One-Out Cold (LOOC) evaluation. The performance drop reveals the challenge of recommending completely unseen items. ID-softmax transformer baselines (SASRec, BERT4Rec, Attr- Former) cannot score items whose IDs never appear during training under LOOC and are therefore marked as N/A.

Evaluation Amazon Beauty Amazon Toys MovieLens-1M

Protocol Recall@20 NDCG@20 Recall@20 NDCG@20 Recall@20 NDCG@20 LOO (Standard) 0.1208 0.0541 0.1169 0.0528 0.337 0.1245 LOOC (Cold Items) 0.0804 0.0351 0.0818 0.0369 0.2267 0.0922 Relative Drop -33.4% -35.1% -30.0% -30.1% -25.0% -25.9%

cold item set as follows: (1) select 500 seed users Useed whose final items define the initial cold set I
0 cold, (2) expand evaluation to all users whose final items fall in I
0 cold, maximizing statistical power while maintaining strict cold-start conditions. This yields evaluation sets ranging from 1,542 users (MovieLens-1M) to 4,681 users (Amazon Toys), providing robust assessment of cold-start performance. This protocol is significantly more challenging: traditional models (SASRec, BERT4Rec, AttrFormer) cannot score items outside their training vocabulary, while feature-based models like RetrievalFormer can generalize to unseen items. The formal protocol with complete statistics is detailed in Appendix F. Because ID-softmax transformer baselines such as SASRec, BERT4Rec, and AttrFormer have no output parameters for item IDs that never appear during training, they cannot assign scores to heldout items under this protocol and thus cannot be evaluated here.

## 4.4.2 Comparing Loo Vs Looc Performance

We evaluate RetrievalFormer under both standard Leave-One-Out (LOO) and Leave-One-Out Cold (LOOC) protocols to quantify the impact of cold-start evaluation: Table 2 reveals the substantial challenge of cold-start recommendation. Even with feature-based encoding, RetrievalFormer experiences a 25-35% performance drop when evaluating on completely unseen items. This drop varies by dataset: Amazon Beauty shows the largest drop (-33.4% Recall@20), likely due to sparse feature coverage for niche products, while MovieLens-1M shows the smallest drop (-25.0%), benefiting from rich genre and tag metadata. The consistent NDCG drops across all datasets indicate ranking quality degradation for cold items. Importantly, while performance decreases under LOOC, RetrievalFormer still maintains meaningful recommendation capability (8.0-22.7% Recall@20), demonstrating its ability to generalize to unseen items through feature-based encoding. In summary, the LOOC evaluation reveals that while cold-start recommendation remains challenging (25-35% performance drop), RetrievalFormer maintains meaningful recommendation capability for unseen items. We emphasize that LOOC is used here as a capability diagnostic to illustrate that a feature-based dual encoder can generate non-trivial recommendations for completely unseen items, rather than as a head-to-head accuracy comparison with ID-softmax baselines, which cannot be evaluated under this protocol. On a 100% cold-start production email campaign dataset (Appendix G), RetrievalFormer outperforms a strong content-based baseline, improving AUC from 0.6854 to 0.7770 (a 13.4% relative improvement), validating its practical effectiveness for dynamic catalogs.

## 4.5 Rq4: Serving Efficiency Of Dual-Encoder Retrieval

The fundamental scalability challenge of transformer-based sequential models is their O(N) inference complexity, where every prediction requires scoring all N items in the catalog. For transformerbased sequential recommenders, the inference cost per request can be decomposed into two parts: (i) O(L
2d) self-attention over the interaction sequence of length L, and (ii) O(N d) dense scoring of all N items in the catalog in the output layer. As the catalog grows, the second term dominates, as also observed in recent latency benchmarks. This architectural constraint creates an insurmount-

![9_image_0.png](9_image_0.png)

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 able bottleneck as catalogs grow: the ETUDE benchmark demonstrates that SASRec exceeds the industry-standard 50ms p90 latency threshold at just 10K items on CPU, with performance degrading to 200ms at 1M items (Kersbergen et al., 2024). RetrievalFormer's dual-encoder architecture with ANN retrieval fundamentally changes this scaling behavior from O(N) to O*(log* N), enabling practical deployment at industrial scale with acceptable accuracy trade-offs. We conducted systematic latency benchmarks comparing exhaustive scoring against IVF-PQ approximate nearest neighbor search on an ml.g6.xlarge instance. Figure 2 compares exhaustive dot-product scoring over all items and ANN-based retrieval using an IVF-PQ index for the *same* dual-encoder scoring function s(u, i) = x
⊤
uyi as the catalog size increases from 10K to 10M items.

Figure 2 demonstrates the dramatic divergence in scaling behavior: exhaustive scoring exhibits strict linear scaling from 0.76ms at 10K items to 292ms at 10M items, while IVF-PQ maintains sub-linear growth from 0.55ms to 1.02ms, a 288× speedup at 10M items. This sub-linear scaling enables practical deployment at industrial scale.

We use a FAISS IVF-PQ index with nlist = 4096 coarse clusters, 64-dimensional product quantization codes, and nprobe = 32 during search; item embeddings are trained on 1M items and indexed offline. All latency measurements are taken on a single NVIDIA V100 GPU with 32GB memory and a batch size of 1024 users after a warm-up phase.

## 5 Conclusion

We introduced RetrievalFormer, a two-tower sequential recommender that combines transformer sequence modeling with efficient ANN retrieval. By encoding users and items in a shared feature-rich embedding space, our approach eliminates expensive softmax computations while enabling zeroshot recommendation of new items. Experiments demonstrate RetrievalFormer achieves 86–91% of the Recall@20 of strong transformer baselines while delivering 288× speedup at 10M items. The model successfully recommends cold-start items where ID-based methods fail entirely. RetrievalFormer bridges the gap between academic advances and production requirements, offering a practical trade-off between accuracy and serving efficiency for large-scale deployment.

## References

Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube recommendations.

In *Proceedings of the 10th ACM conference on recommender systems*, pp. 191–198, 2016.

Gabriel de Souza Pereira Moreira, Sara Rabhi, Jeong Min Lee, Ronay Ak, and Even Oldridge.

Transformers4rec: Bridging the gap between nlp and sequential/session-based recommendation. In *Proceedings of the 15th ACM Conference on Recommender Systems*, pp. 143–153, 2021.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics, pp. 4171–4186, 2019.

Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. In International Conference on Machine Learning, pp. 2793–2803. PMLR, 2021.

Chantat Eksombatchai, Pranav Jindal, Jerry Zitao Liu, Yuchen Liu, Rahul Sharma, Charles Sugnet, Mark Ulrich, and Jure Leskovec. Pixie: A system for recommending 3+ billion items to 200+ million users in real-time. In *Proceedings of the 2018 world wide web conference*, pp. 1775–1784, 2018.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Revisiting deep learning models for tabular data. *Advances in Neural Information Processing Systems*, 34:18932–18943, 2021.

Mihajlo Grbovic and Haibin Cheng. Real-time personalization using embeddings for search ranking at airbnb. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 311–320, 2018.

Cheng Guo and Felix Berkhahn. Entity embeddings of categorical variables. In arXiv preprint arXiv:1604.06737, 2016.

Balazs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. Session-based rec- ´
ommendations with recurrent neural networks. In *arXiv preprint arXiv:1511.06939*, 2015.

Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronber, Janani Padmanabhan, Giuseppe Ottaviano, and Linjun Yang. Embedding-based retrieval in facebook search. pp. 2553–2561, 2020a.

Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar Karnin. Tabtransformer: Tabular data modeling using contextual embeddings. *arXiv preprint arXiv:2012.06678*, 2020b.

Jeff Johnson, Matthijs Douze, and Herve J ´ egou. Billion-scale similarity search with gpus. ´ IEEE
Transactions on Big Data, 7(3):535–547, 2019.

Wang-Cheng Kang and Julian McAuley. Self-attentive sequential recommendation. In *2018 IEEE*
international conference on data mining (ICDM), pp. 197–206. IEEE, 2018.

Barrie Kersbergen, Olivier Sprangers, Frank Kootte, Shubha Guha, Maarten de Rijke, and Sebastian Schelter. Etude: Evaluating the inference latency of session-based recommendation models at scale. 2024. URL https://deem.berlin/pdf/etude.pdf. Industry benchmark with latency/throughput results and SLOs.

Walid Krichene and Steffen Rendle. On sampled metrics for item recommendation. In *Proceedings* of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 1748–1757, 2020.

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention-based permutation-invariant neural networks. In Proceedings of the 36th International Conference on Machine Learning, pp. 3744–3753. PMLR, 2019.

Jing Li, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Tao Lian, and Jun Ma. Neural attentive sessionbased recommendation. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, pp. 1419–1428, 2017.

Gang Liu et al. Learning attribute as explicit relation for sequential recommendation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2025.

Yu A Malkov and Dmitry A Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. volume 42, pp. 824–836. IEEE, 2018.

Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton Van Den Hengel. Image-based recommendations on styles and substitutes. In Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval, pp. 43–52, 2015.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*, 2018.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Nikil Pancha, Andrew Zhai, Jure Leskovec, and Charles Rosenberg. Pinnerformer: Sequence modeling for user representation at pinterest. In *Proceedings of the 28th ACM SIGKDD Conference* on Knowledge Discovery and Data Mining, pp. 3702–3712, 2022.

Andrew I Schein, Alexandrin Popescul, Lyle H Ungar, and David M Pennock. Methods and metrics for cold-start recommendations. In *Proceedings of the 25th Annual International ACM SIGIR* Conference on Research and Development in Information Retrieval, pp. 253–260. ACM, 2002.

Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang.

Autoint: Automatic feature interaction learning via self-attentive neural networks. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, pp. 1161–1170, 2019.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.

Dropout: a simple way to prevent neural networks from overfitting. volume 15, pp. 1929–1958, 2014.

Liangcai Su, Fan Yan, Jieming Zhu, Xi Xiao, Haoyi Duan, Zhou Zhao, Zhenhua Dong, and Ruiming Tang. Beyond two-tower matching: Learning sparse retrievable cross-interactions for recommendation. In *Proceedings of the 46th International ACM SIGIR Conference on Research and* Development in Information Retrieval, pp. 1288–1297, 2023.

Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In *Proceedings* of the 28th ACM international conference on information and knowledge management, pp. 1441– 1450, 2019.

Jiaxi Tang and Ke Wang. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pp. 565–573, 2018.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Maksims Volkovs, Guangwei Yu, and Tomi Poutanen. Dropoutnet: Addressing cold start in recommender systems. In *Advances in neural information processing systems*, volume 30, 2017.

Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In Proceedings of the 37th International Conference on Machine Learning, pp. 9929–9939. PMLR, 2020.

Chao-Yuan Wu, Amr Ahmed, Alex Beutel, Alexander J Smola, and How Jing. Recurrent recommender networks. In *Proceedings of the tenth ACM international conference on web search and* data mining, pp. 495–503, 2017.

Ji Yang, Xinyang Yi, Derek Zhiyuan Cheng, Lichan Hong, Yang Li, Simon Xiaoming Wang, Taibai Xu, and Ed H Chi. Mixed negative sampling for learning two-tower neural networks in recommendations. In *Companion Proceedings of the Web Conference 2020*, pp. 441–447, 2020.

Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, and Ed Chi. Sampling-bias-corrected neural modeling for large corpus item recommendations. In *Proceedings of the 13th ACM Conference on Recommender Systems*, pp. 269–277, 2019.

Kun Zhou, Hui Yu, Wayne Xin Zhao, and Ji-Rong Wen. Filter-enhanced mlp is all you need for sequential recommendation. In *Proceedings of the ACM Web Conference 2022*, pp. 2388–2399, 2022.