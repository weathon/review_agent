# One-Embedding-Fits-All: Efficient Zero-Shot Time Series Forecasting by a Model Zoo

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
The proliferation of Time Series Foundation Models (TSFMs) has significantly advanced zero-shot forecasting, enabling predictions for unseen time series without task-specific fine-tuning. Extensive research has confirmed that no single TSFM excels universally, as different models exhibit preferences for distinct temporal patterns. This diversity suggests an opportunity: how to take advantage of the complementary abilities of TSFMs. To this end, we propose ZooCast, which characterizes each model's distinct forecasting strengths. ZooCast can intelligently assemble current TSFMs into a model zoo that dynamically selects optimal models for different forecasting tasks. Our key innovation lies in the One-Embedding-Fits-All paradigm that constructs a unified representation space where each model in the zoo is represented by a single embedding, enabling efficient similarity matching for all tasks. Experiments demonstrate ZooCast's strong performance on the GIFT-Eval zero-shot forecasting benchmark while maintaining the efficiency of a single TSFM. In real-world scenarios with sequential model releases, the framework seamlessly adds new models for progressive accuracy gains with negligible overhead.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ZooCast, which is a framework that enhances zero-shot time series forecasting by combining multiple time series foundation models. By embedding models and task time series into a shared representation space, optimal models can be selected quickly through similarity matching. The models' representations are computed using the advantage subset once, and then for any new forecasting task, the pipeline selects the most suitable models based on this embedding similarity. The contribution of the paper includes: 1. Construct an advantage subset to characterize the model-specific strengths; 2. A Co-embedding extractor to unify the representation; 3. Ensemble predictions based on the proposed ranking mechanism. Experiments on the GIFT-Eval benchmark show that ZooCast achieves higher accuracy and efficiency than individual TSFMs.

### Strengths
1. Good framework and insights about leveraging multiple TSFMs for zero-shot time series forecasting. Transform the model selection problem into an embedding similarity search, which is a nice direction for TSFM applications.
2. The improved complexity indicates the design is suitable for practical and large-scale applications.
3. The framework is extensively compared on the GIFT-Eval, which is a fair comparison that mitigates the data leakage in recent TSFM evaluations.

### Weaknesses
Several places remain unclear after reading:
1. The pipeline introduction is unfinished. 4.3 introduces the similarity and ranking, and how you leverage this ranking information to obtain the final prediction. Whether using a top-1 model prediction or a Top-K Ensemble prediction, you should have more words to clarify it. 
2. The training co-embedding extractor is very unclear, given that it is the most important component in the pipeline. The rationale of the transfer loss is insufficiently explained; if this loss function represents a novel contribution, it should be detailed in the main text rather than hidden in the appendix. 
3. From 4.2.2, the similarity only reflects the data similarity if using the same encoder, and the design of the co-embedding extractor seems redundant, since a randomly initialized MLP can perform the encoding. (What's the architecture of your co-embedding extractor?)

Minor issue: Consider redrawing the figures in scientific style.

### Questions
1. Equation 2 defines two key metrics: performance gain and selection efficiency, but these metrics are never explicitly reported or discussed in the experimental results. 
2. Could you provide more explanation about why the advantage set can reflect the model preference? 
3. Do you gain some insights about which specific TSFMs perform best on particular types of time series data?
4. Could you provide the anonymous repo for the code for a clearer understanding of your pipeline?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces ZooCast, a model-zoo paradigm for zero-shot time-series forecasting. It embeds both tasks and TSFMs into a shared vector space so that selecting models for a new task reduces to fast similarity matching, followed by a small Top-K ensemble. ZooCast has three parts: (1) advantage subset characterization to capture each model’s strengths using high-variance, discriminative samples; (2) a model–task co-embedding extractor trained with reconstruction, masked-view consistency, and transferability losses; and (3) an error-correcting consensus ranking that aggregates per-channel similarities into a stable global rank. On the GIFT-Eval benchmark with 13 public TSFMs, ZooCast’s ensembles outperform the strongest single model and a naive ensemble method, demonstrating the effectiveness of the proposed method.

### Strengths
1. This paper is well-written and the idea is interesting.
2. Co-embedding turns model choice into an O(1) similarity lookup per model; the once-only precompute plus lightweight selection is orders faster than evaluating all models on every task.
3. Top-K ensembles guided by ZooCast beat the best single model and the All-13 average, with stable gains as K grows and as new models arrive.

### Weaknesses
1. Building advantage subsets from pretraining source pools assumes the developers can access (or faithfully approximate) each TSFM’s pretraining distribution. But for some closed-source models, those corpora are proprietary or legally restricted. Even small mismatches (frequency, domain, seasonality, regime) can distort the strength profile, leading the selector to favor the wrong models.
2. Curating a domain-agnostic yet representative 𝐷∗ that doesn’t overlap with any model’s pretraining data is nontrivial and costly. If 𝐷∗ is out-of-domain (e.g., different sampling rates, covariates, nonstationarity), the learned geometry may not transfer, degrading selection quality. This creates an additional maintenance burden: 𝐷∗ may need periodic refresh or reweighting to track distribution shift.

### Questions
If pretraining source pools are unavailable, what’s the best proxy (e.g., public corpora matched by frequency/domain)? How much does performance drop when advantage subsets are built from mismatched sources?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a framework for improving zero-shot time-series forecasting by leveraging a model zoo of diverse time-series foundation models (TSFMs). 
Instead of relying on a single universal model, ZooCast introduces a One-Embedding-Fits-All mechanism that maps both tasks and models into a shared embedding space, enabling efficient similarity-based model selection without costly evaluations. 
The framework incorporates three core innovations: 
(1) Advantage Subset Characterization, which identifies each model’s domain strengths; 
(2) Model–Task Co-Embedding, which learns a unified representation to predict model–task compatibility; and 
(3) Error-Correcting Consensus Ranking, which refines rankings across multi-dimensional signals for stable selection. 
Extensive experiments on the GIFT-Eval benchmark show that ZooCast achieves SOTA zero-shot performance while maintaining computational efficiency and scalability, offering a practical and interpretable strategy for deploying large TSFMs across diverse forecasting domains.

### Strengths
S1.
While prior efforts have focused on designing ever-larger time-series foundation models (TSFMs), this paper instead proposes ZooCast, a meta-framework that leverages the diversity of existing models through a shared embedding space, thereby sidestepping the inefficiency and rigidity of monolithic architectures.

S2.
This paper is technically sound and experimentally thorough. 
The framework is well-motivated by the empirical observation that “no single TSFM dominates all domains,” and the authors back their claims with robust evaluations across multiple datasets and model types. The three proposed modules are each well-justified, ablated, and shown to contribute meaningfully to performance.

S3.
The paper is clearly written and well-structured, with precise motivation, visual diagrams that clarify the model–task embedding process, and intuitive explanations of how each component fits within the broader system.

### Weaknesses
W1.
The paper’s central innovation—the Model–Task Co-Embedding (MTE)—is intuitively motivated but not theoretically grounded. It is unclear how the embedding similarity quantitatively correlates with forecasting performance or uncertainty. 
For example, if cosine similarity in the embedding space is used to rank models, is there a provable or empirical link between similarity and actual predictive error across domains? 
The lack of formal analysis weakens the justification for this approach compared to traditional meta-learning or Bayesian model averaging frameworks.

W2.
Although ZooCast achieves strong results on GIFT-Eval, its robustness under distributional shift and out-of-domain tasks remains unclear. Since time-series tasks vary drastically (e.g., periodicity, noise, granularity), the model zoo’s success depends heavily on how well the learned embeddings generalize beyond the seen task–model pairs.

W3.
While ZooCast’s embeddings enable efficient model retrieval, the learned representations are opaque: we do not know what temporal features (e.g., seasonality, volatility, sparsity) drive similarity between tasks and models. Without interpretability, users cannot reason about why a model is selected for a given task.

W4.
The experiments focus mainly on pre-trained TSFMs from the GIFT benchmark, but the zoo’s diversity and scalability are not systematically evaluated. 
It remains unclear whether ZooCast maintains its advantage as the number of candidate models grows or if performance saturates beyond a certain zoo size. 
Moreover, comparisons against simple but strong alternatives—such as error-based meta-learners or feature-based retrieval models are missing, making it difficult to isolate the incremental gain from the co-embedding approach.

### Questions
Q1.
How well does the cosine similarity (or whichever metric is used) between model and task embeddings correlate with actual forecasting accuracy across diverse domains?
Have the authors conducted any empirical calibration or rank correlation analyses between similarity scores and true performance metrics?

Q2.
What are the supervised signals or objectives used during co-embedding extractor training—do they rely purely on pairwise model–task performance or also on self-supervised structure?
How sensitive is the learned embedding to the specific datasets (e.g., M3, M5, Tourism) used for training?

Q3.
The current zoo composition appears manually curated. How does ZooCast perform when the zoo contains many redundant or low-quality models?
Is there a mechanism to automatically prune or weight models based on their contribution to diversity or historical utility?

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
4

### Summary
This paper proposes ZooCast, a method for efficient zero-shot time series forecasting using a model zoo. It first characterizes each model's strengths via advantage subsets, which consist of data where the model excels. Then, it uses model-task co-embedding to map models and tasks into a shared space, enabling fast, accurate model selection through similarity matching.  And finally employs an error-correcting consensus mechanism that aggregates multi-channel signals into robust recommendations. The approach achieves high performance with minimal computational cost, making it scalable for real-world deployment.

### Strengths
1.	This paper presents a framework for zero-shot time series forecasting by introducing the novel concept of advantage subsets to characterize each model’s unique strengths. It then maps both models and tasks into a shared space, enabling fast and accurate model selection through simple similarity matching.
2.	The authors not only provide a theoretical efficiency analysis of the proposed method, but also demonstrate its practical effectiveness through runtime comparisons, confirming that their approach is indeed efficient and well aligned with the original goal of rapidly selecting optimal models for different forecasting tasks.
3.	ZooCast exhibits exceptional flexibility and scalability, enabling seamless integration of new models into the model zoo without the need to re-evaluate or recompute embeddings for existing models.

### Weaknesses
1.	The results in Table 2 show that ZooCast's sMAPE is higher than that of Sun.B, indicating that the ensemble performs worse than the best individual model. If the ensemble fails to outperform the top single model, what is its practical value?
2.	Since the extractor only supports fixed-length inputs (length 36), but downstream datasets are often longer than this, would the model selected from the embedding of different segments of length t sampled from the same dataset be consistent? Could sampling different segments lead to varying model rankings?
3.	The paper proposes a multi-channel voting mechanism to derive a consensus ranking. Although the authors mention accounting for noisy channel-wise similarities, further analytical experiments could be conducted to validate this design choice. Especially since, in practice, selecting the best model for each channel is a more intuitive and commonly adopted strategy. Moreover, the current approach of choosing different models per channel and then aggregating via voting may ignore intrinsic dependencies among channels, which could result in suboptimal recommendations. A deeper investigation into these aspects would strengthen the methodological justification and overall robustness of the proposed framework.

### Questions
1.	Please explain why ZooCast's sMAPE in Table 2 is higher than that of Sun.B, and clarify the practical value of the ensemble if it underperforms compared to the best individual model.
2.	Please analyze whether the model selected from embeddings of different segments (of length 36) sampled from the same long time series remains consistent, and discuss the potential impact of segment variation on model ranking stability.
3.	Please provide further justification for using a multi-channel voting mechanism to derive a consensus ranking, especially given that selecting the best model per channel is a more intuitive strategy. Additionally, please investigate whether the independence assumption in channel-wise model selection overlooks intrinsic channel dependencies, and assess how this might affect recommendation quality.

### Soundness
3

### Presentation
2

### Contribution
3
