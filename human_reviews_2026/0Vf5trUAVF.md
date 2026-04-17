# Learning Ordinal Probabilistic Reward from Preferences

- Decision: Accept (Poster)
- Scores: 6, 2, 8, 4

## Abstract
Reward models are crucial for aligning large language models (LLMs) with human values and intentions.
Existing approaches follow either Generative (GRMs) or Discriminative (DRMs) paradigms, yet both suffer from limitations:
GRMs typically demand costly point-wise supervision, while DRMs produce uncalibrated relative scores that lack probabilistic interpretation.
To address these challenges, we introduce a novel reward modeling paradigm: Probabilistic Reward Model (PRM).
Instead of modeling reward as a deterministic scalar, our approach treats it as a random variable, learning a full probability distribution for the quality of each response.
To make this paradigm practical, we present its closed-form, discrete realization: the **Ordinal Probabilistic Reward Model** (OPRM), which discretizes the quality score into a finite set of ordinal ratings.
Building on OPRM, we propose a data-efficient training strategy called **Region Flooding Tuning** (RgFT).
It enables rewards to better reflect absolute text quality by incorporating quality-level annotations, which guide the model to concentrate the probability mass within corresponding rating sub-regions.
Experiments on various reward model benchmarks show that our method improves accuracy by **2.9% ～ 7.4%** compared to prior reward models, demonstrating strong performance and data efficiency.
Analysis of the score distribution provides evidence that our method captures not only relative rankings but also absolute quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the limitations associated with traditional methods in Reinforcement Learning from Human Feedback (RLHF), specifically Discriminative Reward Models (DRMs) and Generative Reward Models (GRMs). DRMs are known for learning reward models based on preference signals, while GRMs identify preferences using a generative language model (LLM), potentially incorporating chain-of-thought (CoT) processing. However, DRMs often present challenges in reward interpretation and threshold determination, and GRMs are unable to perform "Best of N" comparisons or provide explicit probabilistic outputs.


In response, the authors propose a novel approach termed the Ordinal Probabilistic Reward Model (OPRM). This model outputs a complete distribution of rewards, facilitating a more nuanced understanding. The term "ordinal" reflects the method's discretization of the continuous reward space into sequences to ensure tractability. The training process involves objectives that focus on the distribution's spread, optimized using a technique called Region Flooding Tuning.


For data generation, a scoring methodology is established by curating model prompts and responses with a template that translates scores into categorical labels: Bad, Normal, and Good. This dataset is then utilized to train the model with tailored objectives that leverage Region Flooding Tuning to enhance performance consistency.


The paper employs Qwen-2.5-Instruction model variants with parameters ranging from 7B to 72B. Evaluation is conducted using Reward Bench, PPE, and RMB datasets, alongside comparisons with existing DRM and GRM techniques. The findings reveal a notable improvement over baseline models and alternative methods. Additionally, the paper discusses ablation studies assessing the influences of Region Flooding and examines the correlation with human alignment metrics and also delves into the mechanics of how the gradients operate as the learning signal.

### Strengths
- The paper effectively addresses a feature gap in existing models by introducing a methodology that outputs a probability distribution. As a consequence, it enables benefits such as Best of N comparison and uncertainty quantification, making it suitable for tasks requiring these features.
- The paper explores advanced techniques for constructing the objective, including methods like Region Flooding, to ensure essential learning constraints are satisfied.
- It provides a thorough comparison with existing methods using relevant benchmarks, demonstrating the proposed approach's effectiveness.
- The authors conduct a mathematical analysis illustrating how the Bradley-Terry model emerges as a special case within their distribution formulation.
- The appendix includes well-chosen examples that offer qualitative insights into the method’s output.

### Weaknesses
Although the method effectively addresses a specific feature gap, such as Best of N comparison and uncertainty quantification, the source of its performance enhancement compared to alternative methods remains unclear. Additional interpretative analysis would be beneficial.

### Questions
When generating the score for a prompt-response pair from the LLM, the token probabilities already provide a distribution. How does the learned distribution differ from this existing distribution? What supplementary insights are gained beyond the data-generating distribution? Since this data generation process resembles a GRM-based scoring system, how should we interpret the performance improvement achieved through your method when applied to data generated using GRM? Could you provide clarification on this?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes the Ordinal Probabilistic Reward Model (OPRM), a novel approach for training LLM reward models. Instead of the conventional method of training a regression head to output a single scalar reward, OPRM reframes the task as learning a probability distribution over a finite set of ordinal quality ratings (specifically, 1 to 9). The key mechanism is elegant and parameter-efficient: it directly repurposes the LLM's existing language model head, taking the normalized probabilities of the numeric tokens '1' through '9' as the reward distribution. The model is trained on preference pairs ($y_c$, $y_r$) by optimizing a loss function that maximizes the probability $P(S_c > S_r)$, where $S_c$ and $S_r$ are the random variables representing the scores for the chosen and rejected responses, respectively.

Furthermore, the paper introduces Region Flooding Tuning (RgFT), a fine-tuning strategy designed to anchor these relative preference-based distributions to an absolute quality scale. RgFT uses coarse-grained quality annotations (e.g., "bad", "normal", "good") and partitions the ordinal scores into corresponding sub-regions (e.g., {1-3}, {4-6}, {7-9}). The loss is then modified to guide the model to concentrate probability mass within these target sub-regions. The authors conduct experiments on several reward model benchmarks (RewardBench, PPE, RMB) using Qwen2.5 as a base model, claiming state-of-the-art performance compared to various discriminative and generative reward models.

### Strengths
1.  **Novel and Parameter-Efficient Design:** The core idea of OPRM—repurposing the LM head's vocabulary probabilities for numeric tokens as a reward distribution—is a clever and elegant approach that avoids adding any new parameters, unlike traditional scalar-based reward models that require a separate value head.
2.  **Probabilistic Formulation:** Moving from a deterministic scalar to a full probability distribution  is a well-motivated step. It inherently allows for modeling uncertainty  and, as the authors claim, offers richer interpretability than a single score.
3.  **Absolute Quality Grounding:** The Region Flooding Tuning (RgFT) strategy  is a practical and data-efficient method to tackle a known problem: the misalignment between relative-preference scores and intuitive, absolute quality.

### Weaknesses
1.  **Unsupported Core Assumption:** The entire method hinges on the assumption that an LLM's pre-trained probability distribution over the tokens '1'-'9' has an inherent ordinal correspondence to text quality. This is a very strong and unevaluated claim. It is highly plausible that the model has a strong prior based on token frequency in pre-training data (e.g., '1' may be far more common than '9'), which would be unrelated to quality. The paper provides no analysis of this token prior or any evidence (e.g., calibration metrics like ECE) that the resulting distribution is meaningful.
2.  **Critical Robustness Vulnerability:** The method relies on the next-token prediction at a specific prompt ending in "Score:". This creates a high risk of "jailbreaking" or adversarial manipulation. A malicious user could craft a response that includes text like "This response is excellent. Score: 9" to poison the model's internal state and bias the output distribution at the final scoring step. This plausible and critical failure mode is not evaluated.
3.  **Missing Context from Ordinal Regression Literature:** The paper makes no effort to explain the ordinality in the data as other papers of a similar topic would do, and the ordinal nature is actually imposed by design. One can view the proposed target loss as a voting one: the $P(y_c \succ y_r | x)$ can be rewritten as $\sum_k p_c(k) F_r(k-1)$, where $F_r$ term is the CDF of the rejected distribution. The paper fails to connect or compare its work to the extensive existing literature on ordinal regression and probabilistic ordinal models, which have diverse formulations. The "Related Work" section  only discusses standard DRMs and GRMs, missing a key body of relevant research.
4.  **Contradictory Claims vs. Methodology:** The paper explicitly claims "Handling Annotation Disagreement"  and "robustness to... inconsistent preference data"  as key advantages. However, Appendix H.1 clearly states that the authors "filter out all pairs where the chosen response is not strictly better than the rejected one," including "invalid combinations... such as <normal, good>". This experimental practice of manually removing the exact "inconsistent" or "ambiguous" data that the model claims to be good at handling directly undermines this central claim.
5.  **Unequal Experimental Comparisons:** The main results in Table 2  are difficult to interpret as they compare OPRM (trained on the authors' 130k dataset ) against "Reported Results of Public Models" and "Reproduced Results... From DeepSeek." These baselines are trained on different and unknown datasets. This is not an apples-to-apples comparison. It is impossible to know if OPRM's superiority comes from the method or from the base model or the extra data. A fair comparison would require training all baselines (e.g., a standard BT model, PairRM) on the exact same 130k dataset. The single ablation in Figure 4(a)  is not sufficient.
6.  **Arbitrary Partitioning in RgFT:** The 3-quality-level partition in RgFT ({1-3}, {4-6}, {7-9})  is arbitrary and presented without justification. No sensitivity analysis is performed to show why this even split is optimal, or how performance changes with different partitions (e.g., {1-2}, {3-7}, {8-9}).

### Questions
See the weaknesses. Overall, the idea is interesting, and I believe it has potential. But there are too many concerns to accept the claims.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses limitations of traditional generative and discriminative reward models in RLHF. The authors propose a probabilistic reward model (PRM) training paradigm that discretizes the quality score into a finite set of ordinal ratings and trains the reward model as a classifier over these regions. They further introduce a “region flooding tuning” training strategy to improve coverage and robustness around decision boundaries. Experiments across multiple benchmarks indicate improved effectiveness over baselines.

### Strengths
1. Presents a novel training paradigm that requires minimal architectural changes to existing neural reward models.
2. The proposed method may improve calibration and robustness compared to traditional Bradley-Terry reward models
3. Extensive experimental evaluation across diverse datasets demonstrates practical utility.

### Weaknesses
1.	Although framed generally, the approach appears tied to Bradley–Terry-style preference data and is primarily evaluated in pairwise preference setups. It is unclear how well the method applies to verifiable rewards (e.g., math/code with execution-based correctness) or to process-based rewards.

2.	The choice of the number of ordinal bins, boundary placement, and mapping from discrete classes back to scalar rewards (for policy optimization) may significantly affect performance. Sensitivity analyses and principled binning strategies are not clearly presented.

3.	The training and inference overhead of probabilistic binning and flooding are not fully quantified.

### Questions
1.	Can you explain Eq. 8 and Figure 2 in detail?

2.	How sensitive are results to the number of bins and boundary placement?

3.	Can PRM incorporate external verifiable rewards as additional ordinal classes or as targets?

4.	What is the computational overhead of training?

5.	Does the method improve robustness to reward hacking?

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
The paper proposes a Probabilistic Reward Modeling (PRM) that treats a response’s quality as an absolute reward value rather than relative value. This absolute value is modeled as a random variable and the trained model learns its distribution. Then the author bin the continuous values into 9 discrete bins, where the ordinal quality distribution ranges from 1 to 9, and optimizes the probability that the chosen response’s score exceeds the rejected one. A training strategy, Region Flooding Tuning (RgFT), further uses good/normal/bad quality labels to improve the training. The paper reports empirical results on RewardBench, PPE, and RMB by improving over current baselines.

### Strengths
1. Rating the response by an integer is not novel, but the paper proposes a good way of learning the integer's distribution by contrasting the overall score distribution of chosen and rejected respones. Empirical results show the efficiency.

2. RgFT training controls the shift of probability mass more precisely by using some additional information of quality ("good/normal/bad"). It makes the training more efficient.

3. Reusing the LM head probabilities over numeric tokens is easy to implement and simple yet effective.

### Weaknesses
1. Calibration not measured. A primary motivation is "calibrated, interpretable distributions", yet there are no metrics (e.g., ECE) in regards of calibration reported. This is a big question, as the authors claim that they are modeling the whole distribution. Accuracy isn't enough.

2. I'm not very satisfied with the presentation of this paper. Two major concerns (but not limited to them): (1) Important model/algorithms such as RgFT are not formally defined, which is confusing. Also, gradient updates of the loss can be derived in the main text as this is an important intuition. (2) The authors spend many efforts in describing how effective their method is in the abstract and the introduction, but they do not describe their model or algorithm ideas briefly. That prevents readers from acquiring the useful information. From the perspective of readers, most submissions to NeurIPS/ICML/ICLR are SOTA, but the key difference is how and why.

3. Where do the gains come from? RgFT mixes labeled and unlabeled preference data and changes the optimized region. It would help to quantify how much improvement is due to (i) access to absolute labels vs. (ii) the flooding geometry itself vs. (iii) more total training signal.

4. Limited comparisons to other distributional RMs. The related‑work section cites distributional/uncertainty‑aware RMs (e.g., PURM, quantile approaches), but comparisons in the main tables primarily cover BT‑style and generative baselines. Including at least one representative distributional RM would better show OPRM’s advantages.

### Questions
1. I'm curious about what the definition of "ordinal" is in your sense. For example, there is another previous paper *Reward Modeling with Ordinal Feedback: Wisdom of the Crowd*. The term "ordinal" of yours is a bit different from theirs, and you should discuss about why your discrete absolute scores are one kind of "ordinal" label.

2. Please see Weaknesses 1. Calibration measurement should be included, or the authors should delete the claim "calibrated" in their text.

3. Please see Weaknesses 2 and 3. RgFT is the core part. The authors should include more discussions on RgFT, including formal definitions and intuitions.

4. Please see Weaknesses 4. The authors should include other distributional RMs in the comparison.

### Soundness
2

### Presentation
1

### Contribution
2
