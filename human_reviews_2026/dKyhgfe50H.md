# Discrete Diffusion for Bundle Construction

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
As a central task in product bundling, bundle construction aims to select a subset of items from large item catalogs to build an entire bundle or, more practically, complete a partial bundle. Existing methods often rely on the sequential construction paradigm that predicts items one at a time, nevertheless, this paradigm is fundamentally unsuitable for the essentially unordered bundles. In contrast, non-sequential methods model a bundle as a set, but still face two dimensionality curses: the combinatorial space grows exponentially with both bundle length and catalog size. Accordingly, we identify two technical challenges: 1) how to effectively and efficiently model the higher-order intra-bundle relations with the growth of bundle length; and 2) how to learn item representations that remain discriminative while avoiding search directly over a huge item catalog.

To address these challenges, we propose DDBC, a Discrete Diffusion model for Bundle Construction. DDBC leverages a masked denoising diffusion process to build bundles non-sequentially, capturing joint dependencies among items without relying on a fixed decoding order, thereby partially alleviating the combinatorial challenge introduced by increasing bundle length. To mitigate the curse of large catalog size, we integrate residual vector quantization (RVQ), which compresses item embeddings into discrete codes drawn from a globally shared codebook, enabling more efficient search while retaining semantic granularity.  We evaluate our method on real-world bundle construction datasets of music playlist continuation and fashion outfit completion, and the experimental results show that DDBC can achieve more than 100\% relative performance improvements compared with state-of-the-art baseline methods. Ablation and model analyses further confirm the effectiveness of both the diffusion backbone and the RVQ tokenizer, with gains becoming more pronounced for longer bundles and larger catalogs. Our code is available at https://github.com/241416/DDBC.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Bundle construction suffers from a massive item pool and exponentially growing combinations. 
DDBC addresses these challenges by utilizing a diffusion model to generate an entire bundle. 
DDBC also quantizes item embeddings into discrete codes to reduce the search space.
Extensive experiments show that DDBC outperforms current bundle generation methods and achieves the state-of-the-art performance.

### Strengths
* Tokenizing items through codebook quantization is effective for processing a massive item pool.
* DDBC outperforms existing baselines by a huge margin.

### Weaknesses
* The size of bundle is fixed and not adjustable, unlike other baselines.
* Using the same item features for all baselines is doubtful. Each model may require its own embedding space depending on its architecture.

### Questions
* How does a STE work for a codebook quantization?
* How does the RVQ strategy illustrate an item in coarse-to-fine manner?
Sharing the same codebook for all positions without scaling seems to treat all tokens equally rather than hierarchically.
* Even if each item is quantized into a unique code sequence, it seems permutations of the sequence would represent semantically same features since dequantizing the residual quantization is aggregating all codes of the sequence.
How does this affect the diffusion process?
Is there a way to utilize this during the diffusion process?
* How does the dedup code working? Does it included in the codebook? Should the diffusion model recover the dedup code by denoising?

### Soundness
2

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
3

### Summary
In this paper, the authors propose DDBC (Discrete Diffusion for Bundle Construction), a novel framework that formulates bundle construction as a masked discrete diffusion process. Rather than following traditional step-by-step methods, DDBC generates bundles in an order-independent way by progressively filling in masked item tokens. The framework combines Residual Vector Quantization (RVQ), which represents items with discrete semantic codes to address the challenge of large item catalogs, with a Discrete Diffusion Model (DDM) designed to capture complex relationships within bundles without requiring a fixed item order. Comprehensive experiments on Spotify playlist and fashion outfit datasets show that DDBC achieves substantial performance improvements over both sequential and set-based baselines. Ablation studies further highlight the complementary strengths of RVQ and the discrete diffusion backbone.

### Strengths
1. The paper presents a well-motivated argument against the sequential construction paradigm, emphasizing that bundles are inherently unordered sets rather than sequences.
2. The integration of discrete diffusion and residual vector quantization is technically coherent and aligns with recent advances in generative recommendation.
3. The paper provides extensive experiments with clear ablations and sensitivity analyses, showing that DDBC scales better with longer bundles and larger catalogs.

### Weaknesses
1. While the paper demonstrates robustness across different input-predict ratios, the framework fundamentally requires at least some partial bundle items at inference. The paper does not explore conditional generation from fully-masked states using only user embeddings or contextual features, which limits applicability for cold-start scenarios where no seed items are available.
2. The model enforces a fixed number of items per bundle, whereas real-world bundles are inherently variable in size. Capping bundle lengths during training and testing introduces a methodological limitation that does not fully capture the characteristics of the bundle item collection as in original dataset.
3. While the integration of RVQ with discrete diffusion is technically sound and well-executed, the core novelty lies primarily in combining existing techniques (RVQ from generative retrieval and masked diffusion from language models) rather than introducing fundamentally new algorithms. The ablation study demonstrates that both components are essential and complementary, but the contribution is more engineering-focused than conceptually novel. The paper would benefit from deeper theoretical analysis of why this combination works particularly well for bundle construction.

Minor typos:
-	L122: Citation formatting error
-	L196: Use $z_{j,l}$ instead of $z_{jl}$
-	L244: The reverse process should be conditioned on current noisy tokens $Z^{(t)}$
-	L267: $\hat{E}_j = \sum_{l=1}^{L-1} e^{(l)}_{z^{(l)}(i)}$
-	L235, L287: The notation $\alpha(t) != \alpha_t$; $\alpha(t)$ should be $1-t/T$ to yield deterministic at $t=T$

### Questions
1. At $t=T$, all tokens are fully masked, yet the model is trained to predict original tokens under Equation 7. While this teaches the model unconditional bundle priors, could the authors clarify: 
    - a) What proportion of training loss comes from high-noise timesteps ($t \geq 0.8T$)
    - b) Whether this supervision is necessary given that inference always starts from partially-observed bundles
    - c) Whether alternative training strategies (e.g., restricting $t < T$) were explored?
2. During inference, how are the “known” items in $b_x$ selected? Since bundle semantics could vary depending on which subset is revealed, does the model show robustness to different partial-item configurations?
3. When bundles exceed the capped size $k$, how are items selected for truncation? Does this selection (e.g., first-k, random-k, most-popular-k) introduce bias toward specific bundle patterns? Have the authors analyzed whether truncation affects semantic coherence or diversity of the resulting training data?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes DDBC, which is a discrete diffusion for bundle construction. The core motivation is that pervious bundle construction methods are based on sequential generation, where bundle length grows results in intra-bundle relational explosion and large catalog size makes item search space massive. DDBC treats bundle construction as masked discrete denoising over a compact discrete code space. DDBC quantizes item embeddings via multi-level RVQ, running masked discrete diffusion over the codes, and decoding the codes back to item IDs. The experiments show significant gains (about 100%) on some of the baselines.

### Strengths
The bundles are regarded as sets not sequences, which is . And the combination of the masked discrete diffusion and RVQ for bundle generation is novel in this domain.

The improvements are significant compared to the baselines. DDBC achieves >100% relative gains in Jaccard and F1 on Spotify with k = 60/90. DDBC scales well to have bigger relative improvements on the datasets with larger k.

DDBC is tiny and cost less inference time compared to many baselines like BundleMLLM.

### Weaknesses
The size of bundle seems to be fixed-length. If so, the application of DDBC could be limited. DDBC does not model set invariance. Although the bundles are not designed as serialized sequences and order is randomized, permutation invariance is not considered.

The evaluation includes only playlist and POG. I wonder if DDBC still performs well when bundles are from completely different domain, e.g., shopping sets from different categories.

The way of handling personalization is not explored. The generation is unconditional w.r.t user preferences, e.g., the RVQ codebooks of DDBC are global, potentially flattening user preference patterns. The current framework may hardly be deployed to scenarios where personalization is required.

### Questions
Could the diffusion be modified to support variable-length bundle reconstruction?

Could DDBC extended to personalization scenarios? e.g., change architecture to conditional diffusion and inject the user preference using the conditions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a discrete diffusion based bundle construction method. To address the complexity challenge of bundle construction caused by the bundle size and large item corpus, it (1) uses RVQ to get the code representation of each item by training a shared codebook; and (2) trains a discrete diffusion model to gradually mask items in a given bundle in the forward process and then gradually reconstruct the masked items in the reverse process with the item token learned through RVQ module. Experiments on two real-world dataset partially prove the effectiveness of the proposed method (as from the result, the proposed method only works well on one of the adopted dataset with larger bundle size). Overall, this is a quite solid study on bundle construction.

### Strengths
S1. The paper is generally well written and organized, which makes it easy to follow. 

S2. The investigated research question is quite meaningful and practical in the real-world application. More importantly, the authors identified the limitations of the existing sequential-based bundle construction, as the order of the items within a bundle may not matter a lot when constructing the bundle (it could be a set).

S3. The effectiveness of the model has been verified on the Spotify dataset with the large bundle size, while it does not work on the POG dataset with smaller bundle size. 

S4. Extensive ablation studies and important hyper-parameter analysis further show the efficacy of the important modules of the model

S5. The source code is provided to ensure a better reproducibility.

### Weaknesses
W1. The title does not really reflect the task that has been investigated in the paper. The paper actually investigates the bundle completion task instead of exact bundle construction task. 

W2. It would better if the authors can highlight their technical contribution. 

W3. The investigate the research question, though being practical and meaningful in real-world application, seems to lack generalizability. It relies on partial items in a bundle, so cannot create a bundle from scratch. However, in the real-world applications, there are many scenarios, where creating bundle from scratch is required. 

W4. Some related works are missing, for instance, Adaptive In-Context Learning with Large Language Models for Bundle Generation (SIGIR 2024). The authors are highly suggested to cover a comprehensive literature, especially LLM based method for bundle construction/completion. 

W5. The fixed-length setting may limit the flexibility of the proposed model. In the real-scenario, we may not be able to show too many items for the users in one page, so how to determine the subset of the items that should be displayed to the user?

### Questions
Q1. How to distinguish the tokens of unpopular items using RVQ? Existing study shows that RVQ in the Euclidean space may not be able to well distinguish the unpopular items, which has a large amount in the real-world scenarios. 

Q2. Does the step T affect the performance? If so, how?

### Soundness
3

### Presentation
3

### Contribution
3
