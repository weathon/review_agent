# Latent Thinking Optimization: Your Latent Reasoning Language Model Secretly Encodes Reward Signals in Its Latent Thoughts

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Large Language Models (LLMs) excel at problem solving by generating chain of thoughts in natural language, but such verbal thinking is computationally costly and prone to overthinking. A recent work instead proposes a latent thinking architecture, Huginn-3.5B, which represents intermediate reasoning steps as a sequence of latent representations. However, latent thoughts lack interpretability and are difficult to supervise, raising concerns about the correctness and reliability of the model's latent thinking processes. In this paper, we provide a systematic study of how Huginn-3.5B thinks in the latent space and how external supervision signals can improve its latent thinking processes. We show that latent thoughts leading to correct versus incorrect answers exhibit highly distinguishable patterns, and that a latent classifier can reliably predict answer correctness directly from latent thoughts. Leveraging these insights, we propose Latent Thinking Optimization (LTO), a probabilistic algorithm that employs the latent classifier as a Latent Reward Model (LRM) to optimize the latent thinking processes. Extensive experiments across diverse reasoning tasks demonstrate that LRM is highly effective in detecting incorrect latent thinking patterns, and LTO can significantly improve the latent thinking processes. Furthermore, we show that LRM can generalize across diverse domains, and LTO can be seamlessly applied to general LLMs to improve their thinking processes. In contrast to verbal thinking, our method demonstrates that reward modeling and scaling test-time thinking with supervision can be performed directly in the latent space, highlighting its potential as a general, efficient, and domain-agnostic approach to improving the thinking processes of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies latent thinking in the latent-reasoning LM Huggin-3.5B and claims that correctness signals are encoded in latent trajectories. The authors (i) empirically show separable patterns between correct vs. incorrect latent thoughts via visualization and four representation metrics (entropy, effective rank, anisotropy, intrinsic dimension); (ii) train a latent classifier that predicts answer correctness from partial latent trajectories with high ROC-AUC; and (iii) propose Latent Thinking Optimization (LTO): use the classifier as a Latent Reward Model (LRM) and perform KL-regularized probabilistic selection of latent trajectories via an acceptance–rejection sampler with proofs (Thm. 1/2). LTO improves correctness over voting/self-correction and latent heuristics and is further applied to general LLMs by treating intermediate layer states as “latent thoughts,” showing sizable gains and cross-domain transfer of the LRM.

### Strengths
1. neat empirical evidence that correct trajectories differ from incorrect ones, with stepwise analyses and a probing classifier that improves with longer prefixes. 

2. LTO formulates a KL-regularized objective and provides a closed-form reweighting plus an accept-reject sampler with correctness guarantees (Thm. 1/2). 

3. Broad task coverage & signal utility: improvements across math/commonsense/code; LRM used as weights already helps; LTO helps more. 

4. Generality claims with initial evidence: application to general LLMs (OLMo/Llama/Mistral) and cross-dataset LRM transfer; training-data footprint noted as modest.

### Weaknesses
1. Entropy/effective-rank/anisotropy/intrinsic-dimension are scale- and whitening-sensitive; without controlling for per-step activation scaling or layer-norm statistics, a change in metric may reflect variance rescaling rather than “better thinking.” Please specify invariances and show robustness across metric definitions and layers.

2. The LRM is trained from the same model’s latent trajectories using final-answer correctness as labels; if the generator exhibits systematic artifacts (e.g., decoding shortcuts at later steps), the classifier may learn spurious correlates. The strong AUC near the end steps could partially capture answer-formation traces rather than “process quality.” A control where latents are early-stopped and the answer is masked/not decoded would help.

3.  The reward signa is derived from LTO are binary and can only indicate if the latent thinking processes will lead to the correct answer, as you mentioned in appendix

### Questions
1. As mentioned in Author's guidence, you should disclose the use of Large Language Models. I don't find it in your paper. Could you disclose it here? or you may violate the guidence and lead to desk rejection.

2. Please see weaknesses and try to answer 1-2. For 3, I am curious about methods you have tried.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies how reasoning language models internally represent their “thoughts” in latent space. Using the Huggin-3.5B model, the authors show that hidden-state trajectories corresponding to correct and incorrect answers form clearly separable patterns. They train a small latent classifier to detect these patterns and use it as a Latent Reward Model (LRM) in a new method called Latent Thinking Optimization (LTO)—a test-time sampling procedure that reweights latent trajectories toward those predicted to be correct. Without retraining the base model, LTO consistently improves reasoning accuracy across five benchmarks and generalizes to other LLMs (e.g., Llama-2, Mistral), demonstrating that latent representations implicitly encode reward-like signals useful for optimizing reasoning.

### Strengths
* The author conduct in-depth analysis of the latent reasoning trajectories, both qualitatively and quantitatively through metrics including entropy, effective rank, anisotropy, and intrinsic dimension.
* The proposed method is simple that only involves computing evaluation metrics as input features to a small LRM, which is later used to guide the selection of answers through LTO.
* The paper is well organized, balancing theoretical exposition with extensive experiments and visualizations.

### Weaknesses
* Optimization vs. resampling: LTO is effectively a weighted rejection sampling procedure, not true optimization of the model’s latent policy; the optimization problem formulation is confusing. Also, it not clear if the KL divergence constraint is actually needed. 
* Limited interpretability: Despite excellent quantitative separation, there is little qualitative analysis linking specific latent dimensions or manifolds to interpretable reasoning concepts.
* Minor overclaiming: Phrases like “generalist reward model” or “domain-agnostic optimization” are ambitious relative to the experimental evidence.

### Questions
Q1: can author explain the KL divergence term. This is is common in LLM post-training, but LTO is essentially not a optimization method, the distribution won't deviate from base policy a lot.
Q2: can author provide more insight about why correct and incorrect latent trajectories lead to different patterns? For example, why correct latent trajectories tend to be more compact dispersed?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper first empirically investigates the property of latent in latent thinking LLMs, such as Huggin-3.5B to understand their internal reasoning processes. They discover that the latent trajectories leading to correct versus incorrect answers are highly distinguishable and provide both visual and quantitative measurements to support this claim. Based on this finding, they successfully train a Latent Reward Model (LRM) that predicts the correctness of the final answer directly based on the sequence of latent thoughts. Finally, the paper proposes Latent Thinking Optimization (LTO), a probabilistic sampling algorithm that leverages the LRM's reward signal to guide the model's generation at test-time. This LTO algorithm is supported by both theoretical justification and strong empirical results, demonstrating performance improvements on math, coding, and general QA tasks.

In general, this paper validates the separation of correct/incorrect latent trajectories and train a classifier LRM that predicts the trajectory correctness sorely based on the latent. Using LRM as reward model to guide the sampling with LTO yields better results than other test-time scaling method.

### Strengths
1. Clear motivation and empirical insight: convincingly shows correct vs. incorrect latent trajectories are separable, supported by PCA visualizations and four representation metrics (entropy, effective rank, anisotropy, intrinsic dimension). A lightweight sequence classifier (LRM) can achieve strong predictive power.
2. The sampling algorithm LTO is principled and theoretically sound.
3. Consistent empirical gains on math/coding/QA vs. majority voting, self-correction, and latent heuristics (CoE-R/CoE-C).

### Weaknesses
1. Primary Application Targets a Niche Architecture: The paper's methodology is heavily anchored to the Huggin-3.5B model, which features a specific, recurrent "latent thinking" architecture. This model is not a widely adopted or standard foundation model, making it a niche target. 
2. Ambiguous Application to General LLMs. The paper's method for applying LTO to standard transformers (like Llama 2 or Mistral) is conceptually ambiguous. In section 6, the author mentioned that "train LRMs using the latent representations from general LLMs". In the appendix D, the authors state "The latent representations of general LLMs from all the layers are regarded as latent chain of thoughts.". Does that implies that hidden states in all layers are concatenated together as input to LRM? Meanwhile, efficiency of such methods on the application to general LLM is not sufficiently discussed.
3. Limited Evaluation Scope for General LLMs: The experiments on general LLMs are constrained to relatively weak benchmarks such as GSM8k, it is unclear if these gains would translate to more capable models on frontier benchmarks.

### Questions
1. Can the authors specify how multi-layer latents in general LLM are composed for the LRM as well as the efficiency analysis of LRM on general LLMs?
2. Can the authors provide more experiment results for general LLM on recent benchmarks such as MATH/GPQA-Diamond/AIME?

### Soundness
2

### Presentation
3

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
This paper investigates latent thinking in reasoning language models, focusing on the Huginn-3.5B model which generates intermediate reasoning steps as latent representations rather than natural language. The authors analyze differences in latent thought trajectories between correct and incorrect answers using visualizations and representation quality metrics, revealing distinguishable patterns. They train a latent classifier as a Latent Reward Model (LRM) to predict correctness from these trajectories and propose Latent Thinking Optimization (LTO), a probabilistic sampling algorithm that optimizes latent policies to favor trajectories likely to yield correct answers. Experiments on math and programming tasks show improvements, and the approach is extended to general LLMs with cross-domain generalization.

### Strengths
* The exploration of latent thinking patterns provides novel insights into how LLMs encode reasoning internally, bridging cognitive science inspirations with practical LLM analysis, which could inspire future work on interpretable latent spaces.

* LTO is a computationally efficient alternative to verbal chain-of-thought methods, avoiding overthinking and verbosity, with theoretical grounding in reward optimization and empirical gains on benchmarks like SVAMP and MBPP.

* The generalization of LRM and LTO to standard LLMs (beyond Huginn-3.5B) and across domains is promising, suggesting a scalable, domain-agnostic way to enhance LLM reasoning without heavy natural language generation.

### Weaknesses
* The analysis heavily relies on Huginn-3.5B, a specific latent reasoning model; while extensions to general LLMs are claimed, the paper lacks detailed comparisons or ablation on how well LTO performs on diverse LLM model families.

* Evaluation metrics for latent representations (e.g., entropy, anisotropy) are insightful but somewhat indirect; the paper could benefit from more direct interpretability probes or causal interventions to confirm that observed patterns truly reflect "thinking" rather than spurious correlations or memorization artifacts.

* Experimental details on training the LRM (e.g., data scale, hyperparameters) are sparse in the provided sections, and results focus on two datasets; without broader benchmarks or robustness tests (e.g., against adversarial inputs), it's unclear how LTO scales to real-world, noisy reasoning scenarios.

### Questions
* How sensitive is LTO to the number of sampled trajectories N or the KL regularization weight β? Did you observe trade-offs between optimization gains and deviation from the reference policy?

* The paper mentions cross-domain generalization of LRM with small training data; what specific domains were tested for transfer (e.g., beyond math/programming to commonsense or scientific reasoning), and how much data was used for fine-tuning in those cases?

### Soundness
3

### Presentation
3

### Contribution
3
