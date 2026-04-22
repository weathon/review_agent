# GoR: A Unified and Extensible Generative Framework for Ordinal Regression

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 4

## Abstract
Ordinal Regression (OR), which predicts the target values with inherent order, underpins a wide spectrum of applications within diverse domains. The intrinsic ordinal structure and non-stationary inter-class boundaries make OR fundamentally more challenging than conventional classification or regression. Existing approaches, predominantly based on Continuous Space Discretization (CSD), struggle to model these ordinal relationships, but are hampered by boundary ambiguity. Alternative rank-based methods, while effective, rely on implicit order dependencies and suffer from the rigidity of fixed binning.

Inspired by the advances of generative language models, we propose **G**enerative **O**rdinal **R**egression (**GoR**), a novel generative paradigm that reframes OR as a sequential generation task. GoR autoregressively predicts ordinal segments until a dynamic ⟨EOS⟩, explicitly capturing ordinal dependencies while enabling adaptive resolution and interpretable step-wise refinement. To support this process, we theoretically establish a bias–variance decomposed error bound and propose the **Co**verage–**Di**stinctiveness Index (**CoDi**), a principled metric for vocabulary construction that balances quantization bias against statistical variance. The GoR framework is model-agnostic, ensuring broad compatibility with arbitrary task-specific architectures. Moreover, it can be seamlessly integrated with established optimization strategies for generative models at a negligible adaptation cost. Extensive experiments on **17** diverse ordinal regression benchmarks across **six** major domains demonstrate GoR's powerful generalization and consistent superiority over state-of-the-art OR methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
GoR reframes OR as a sequential generation task, where the model autoregressively predicts tokens representing ordinal value segments based on Coverage-Distinctiveness Index (CoDi) and Quantile-based Vocabulary. This method is novel and interesting. Although similar methods have been applied in other visual fields such as image generation and depth estimation, the contribution of applying generative models to the general ordinal regression task is commendable. Despite this paper facing a few weaknesses in qualitative analysis and missing some related works, I think the current version of the paper has reached the acceptance bar of ICLR. If the author can address my concerns, I will consider further improving my score.

### Strengths
1. The method is novel and interesting.

2. Experimental results show the effectiveness of the proposed method. 

3. Model-agnostic framework compatible with various architectures makes the model practical and flexible.

### Weaknesses
1. There is a lack of analysis of the distribution of model improvements. In other words, which part of the corrections leads to the performance improvements? boundary samples, small category samples, or most samples in the whole distribution?

2. Medical disease grading is a common OR task. The medical datasets possess boundary ambiguity and long-tail problems. Testing on medical datasets can enhance the influence and persuasion of the proposed method. 

3. Regarding the related work on ordinal regression, the author seems to overlook a recent development—the methods of introducing CLIP and language models, e.g, OrdinalCLIP, L2RCLIP, NumCLIP and so on. 

4. Lack of experimental comparison to some popular or latest OR methods like PoE (Li et al, CVPR2021), Ord2seq (Wang et al, ICCV2023), and NumCLIP (Du et al, ECCV2024).

### Questions
1. This method is based on discrete modeling via vocabulary construction. What if using continuous modeling methods? Existing methods show that continuous modeling may be a better choice than discrete modeling in some fields[1].  Has the author conducted any relevant experiments? 

   [1] Autoregressive Image Generation without Vector Quantization, NIPS2024.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces GoR (Generative Ordinal Regression), a novel generative framework that formulates ordinal regression as an autoregressive token generation task, terminating with a dynamic <EOS> token. Instead of relying on traditional continuous space discretization or rank-based classification approaches, GoR represents the target ordinal value as a sequence of additive segments drawn from a learned vocabulary. To support this framework, the authors derive a bias-variance decomposed MSE error bound and propose Coverage–Distinctiveness index for principled vocabulary construction, addressing the trade-off between quantization bias and statistical variance. The framework is model-agnostic, supporting a wide range of encoders and decoders, and is extensible to standard generative learning techniques. Extensive experiments show that GoR achieves state-of-the-art performance consistently.

### Strengths
1. The proposal to model ordinal regression as sequential generative modeling is a novel paradigm shift.
2. The bias-variance decomposition provides theoretical grounding for vocabulary design.
3. Good potential for future works as it is compatible with standard training techniques used in generative modeling
4. Achieves consistent gains on the benchmarks.

### Weaknesses
1. Decoding time grows linearly with the number of tokens, which itself varies with the resolution and magnitude of the target ordinal value. The author should consider to include efficiency evaluation.
2. No comparisons with other generative-based models like DDPM or Normalizing Flows in continuous ordinal prediction.
3. Longer sequences amplify token-level noise due to accumulating prediction errors across steps. Prediction accuracy may degrade on long sequences or in fine-resolution tasks, especially under greedy decoding. The authors should consider to analyse the trade-off of the model.

### Questions
See Weaknesses Above.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a novel generative framework termed GoR for ordinal regression that reformulates scalar ordinal prediction as autoregressive token generation. By explicitly modeling ordinal dependencies and dynamically controlling prediction granularity via a learnable ⟨EOS⟩ token, the method overcomes the rigidity and boundary ambiguity in conventional discretization- and ranking-based approaches. The proposed CoDi-guided vocabulary construction is theoretically grounded through a bias–variance decomposed MSE bound, ensuring both representational flexibility and cross-domain adaptability. Extensive experiments across diverse tasks demonstrate consistent and substantial improvements over strong baselines, highlighting GoR as a promising and unified paradigm for future research in ordinal modeling.

### Strengths
-The sequential generation with a dynamic ⟨EOS⟩ explicitly models ordinal dependencies and enables coarse-to-fine refinement, which can offer strong interpretability and flexibility.

-The paper delivers rigorous analysis of limitations in rank-based methods and a principled MSE error bound, which provides solid justification for the proposed approach.

-Extensive experiments across diverse domains show clear and consistent improvements over strong baselines, thereby demonstrating strong generalizability.

### Weaknesses
-It remains unclear whether the improvements can be mainly attributed to the proposed paradigm or simply from the stronger autoregressive decoder; further controlled ablations are needed.

-The effectiveness of the proposed method depends on empirical tuning, which may limit robustness across tasks.

-The sequential generation introduces longer prediction paths and potentially higher latency, which poses concerns for efficiency and industrial deployment.

-Autoregressive decoding may suffer from compounding errors. The paper does not sufficiently address exposure bias, nor analyze the trade-off between robustness gains from beam search and increased inference cost.

-The sequence length varies with label range and distribution, potentially affecting stability and efficiency. More evidence is needed to verify consistent performance across tasks with diverse label scales.

### Questions
-It remains unclear whether the improvements can be mainly attributed to the proposed paradigm or simply from the stronger autoregressive decoder; further controlled ablations are needed.

-The effectiveness of the proposed method depends on empirical tuning, which may limit robustness across tasks.

-The sequential generation introduces longer prediction paths and potentially higher latency, which poses concerns for efficiency and industrial deployment.

-Autoregressive decoding may suffer from compounding errors. The paper does not sufficiently address exposure bias, nor analyze the trade-off between robustness gains from beam search and increased inference cost.

-The sequence length varies with label range and distribution, potentially affecting stability and efficiency. More evidence is needed to verify consistent performance across tasks with diverse label scales.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new method for ordinal regression that treats it as a sequence generation task, predicting segments step by step until an END token. It avoids issues in traditional methods like boundary ambiguity and rigid binning, and introduces a way to balance bias and variance using a new metric called CoDI. It works well across many datasets and is easy to plug into existing generative models.

### Strengths
- This paper is well-written and easy to follow. 
- The experiments show consistent improvements over SOTA across 17 datasets in 6 domains

### Weaknesses
- Is it possible that vocabulary pruning may lead to loss of information, especially affecting minority classes in imbalanced datasets?
- Does GoR support control over output sequence length, or does it only rely on detecting <EOS>? If no, how to control the maximum length of the output sequence, and will it take a longer time for inference? If yes, what would be the performance when limiting the maximum length of the output sequence to a certain value, e.g., 1? What are the impacts on inference time and performance if the sequence length is limited? More evaluations should be included to demonstrate. 
- How was the threshold $\epsilon$ and percentage $\beta$ determined and are there any effects on the final results?

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
