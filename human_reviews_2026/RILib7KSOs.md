# Uncritical tokens are 'critical' in pretraining: the implicit regularization effect of Next token prediction

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Next Token Prediction (NTP) is the prevailing pre-training approach for large language models, which have demonstrated remarkable reasoning capabilities. A key characteristic of NTP is its objective to predict every token in a sequence, including tokens that are not directly relevant to the final answer or core logic—often considered training noise. While such "noise" from uncritical tokens is traditionally thought to impair learning by introducing irrelevant information, our research reveals a counterintuitive positive effect. To isolate this phenomenon, we contrast NTP with Critical Token Prediction (CTP), a training paradigm that focuses exclusively on specific tokens such as the final answer.
Our findings show that NTP consistently surpasses CTP in reasoning ability. We hypothesize and substantiate through theoretical analysis that the learning objective on uncritical tokens acts as an implicit regularizer, analogous to explicit $L^2$ regularization. Further empirical analysis across various benchmark reasoning datasets confirms that NTP-trained models exhibit enhanced generalization and robustness, demonstrating greater resilience to perturbations and achieving flatter loss minima. These findings reveal that uncritical tokens are, in fact, 'critical' for developing robust reasoning during pre-training, offering valuable insights into optimizing training strategies for LLM development.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper challenges the traditional view that "uncritical tokens" in NTP are training noise, revealing their critical role as implicit regularizers for reasoning tasks. By contrasting NTP with CTP which only optimizes for key tokens like answers, the authors theoretically prove that NTP’s uncritical token loss acts as an L²-like implicit regularizer via Fisher information matrix analysis, and empirically validate that NTP-trained models exhibit superior reasoning ability, generalization, and robustness across synthetic and real-world reasoning benchmarks.

### Strengths
* **Important insights**: The paper's central finding—that predicting seemingly irrelevant tokens is beneficial for learning robust reasoning—is counterintuitive and significant.

* **Theoretical and Empirical Synergy**: The work is distinguished by its tight integration of theoretical analysis and empirical validation.

* **Clear Mechanistic Explanations**: Beyond accuracy metrics, the paper uses information flow analysis (Fig. 3b,c) to provide a clear, interpretable visualization of how NTP and CTP lead to different internal reasoning strategies.

### Weaknesses
* **Theoretical scope**: The theoretical claims rely on strong assumptions such as vanishing initialization and uniform Fisher information. While these conditions are definitely useful for creating a tractable analytical model, their applicability to real-world, large-scale training scenarios is not fully established.

* **Scalability**: The experiments on GPT-2 inevitably raise concerns about the robustness and generalizability of the conclusions in the current research context, as works such as RHO-1 have demonstrated effectiveness on models ranging from 1B to 7B parameters at a minimum.

* **Clarity**: The paper uses the term "pre-training" in title and main corpus to describe training models from scratch on task-specific datasets. This could be slightly confusing, as "pre-training" typically refers to unsupervised learning on massive, general-domain corpora. Sharpening this terminology to distinguish between "from-scratch task training" and "fine-tuning" would improve clarity.

* **Reasoning gaps**: I understand that the paper’s task setup is carefully designed and helpful for clarification, but I still worry whether there is a genuine need—and whether significant experimental differences would emerge—when applying the NTP and CTP training paradigms to truly challenging mathematical problems, such as real questions from AIME or OlympiadBench.

### Questions
* In fig. 3(a), your theoretical analysis compellingly links NTP to L2 regularization under specific initialization conditions. How does this implicit regularization from NTP interact with explicit regularizers like weight decay or dropout?

* Appendix A.3 notes that CTP is more efficient when fine-tuning an existing pre-trained model. Could you expand on the practical implications of this? Does it suggest a hybrid strategy where NTP is ideal for initial pre-training, while CTP (i.e., standard SFT) is optimal for downstream adaptation?

* The robustness experiments demonstrate NTP's resilience to input noise and label errors. Do you believe this is solely due to the flatter minima it finds, or are there other mechanisms at play, such as learning a more distributed or compositional representation of knowledge that is inherently more robust?

* Do NTP’s advantages persist in larger models (1B+)? If not, what is the threshold where implicit regularization becomes redundant?

* Can you quantify the implicit regularization strength of NTP (e.g., equivalent L² value) for different tasks/datasets?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the difference between Next Token Prediction (NTP) and Critical Token Prediction (CTP) during pretraining of large language models. The authors argue that NTP’s inclusion of “uncritical” tokens implicitly regularizes the model, similar to L2 weight decay, improving generalization and robustness. They support this claim via synthetic “Anchor Function” experiments, theoretical derivations, and several reasoning benchmarks (e.g., PrOntoQA, CLUTRR, RuleTaker), training models from scratch.

### Strengths
- Includes both theoretical reasoning and empirical comparison with CTP.

- Attempts to unify several lines of thought around noise-induced regularization.

### Weaknesses
- The theoretical analysis, while intuitively appealing, is not fully rigorous. The core derivation that  $\[L_{\text{NTP}} = \frac{1}{T}L_{\text{CTP}} + \frac{1}{2}\theta^\top I_0 \theta + O(\|\theta\|^3)\] $ relies on overly strong assumptions—such as *uniform Fisher information* and *small-weight initialization*—that are unlikely to hold in deep transformer architectures. The argument only establishes a **local first-order approximation** near initialization, so the claimed equivalence between NTP and L2 regularization does not necessarily persist during full training. Moreover, since $\(I_0\)$ is not isotropic, the resulting penalty is **not true L2 regularization** but a direction-dependent quadratic form. Hence, while the result provides a useful intuition, it should be interpreted as an *analogy* rather than a formal equivalence.


- The paper's main claim—that "NTP is mathematically equivalent to CTP plus weight decay"—is expressed too strongly relative to the presented evidence. The experiments are carefully executed but limited in scale and diversity: all models are small (GPT-2 125M) and trained on synthetic or narrow reasoning datasets. As such, the empirical results do not fully substantiate the universality of the claim. The observed effects may instead be dataset- or scale-specific, and the paper would benefit from more systematic ablations (e.g., across model sizes, initialization scales, or data noise levels) to convincingly support the general conclusion.


- The connection between token-level noise and implicit regularization echoes prior work on SGD noise and dropout; thus, the conceptual novelty may be limited without stronger theoretical or empirical differentiation.

### Questions
Refer to Weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores why next-token prediction (NTP), which trains on all tokens, outperforms critical-token prediction (CTP), which trains only on the last token. The authors argue that losses on uncritical tokens act as an implicit L2-style regularizer that improves reasoning and generalization. They support this with theory showing NTP equals CTP plus a quadratic regularization term, synthetic experiments where NTP avoids shortcut solutions, and real benchmarks where NTP-trained GPT-2 models show higher reasoning accuracy, better robustness, and flatter minima. Overall, the study concludes that uncritical tokens, though seemingly redundant, play a key role in regularizing models and promoting better reasoning.

### Strengths
1.	The paper claims that uncritical tokens act as implicit regularizers. They help models avoid shortcut learning and improve reasoning. This idea is useful and explains why removing tokens during training can sometimes harm generalization.
2.	The paper presents a formal analysis showing that NTP is linked to a quadratic regularizer under certain assumptions. It then tests this idea on several synthetic and real reasoning datasets. This mix of theory and experiments makes the claim stronger and more convincing.
3.	The information-flow visualizations and ablations, such as the NTP shuffle, help show how NTP avoids shortcut learning. They reveal that NTP encourages step-by-step reasoning instead of early merging. These visual results make the findings easier to understand.
4.	The paper shows that NTP-trained models are more robust to embedding noise and achieve flatter minima. This finding connects NTP’s behavior to generalization and SGD theory, supporting the idea that it acts as a regularizer.

### Weaknesses
1.	The assumption that the last token is always critical is oversimplified. Datasets like ReCOGS (line 1329-1337) have critical tokens in different position. Using explainability tools (e.g., SHAP or LIME) could better identify important tokens.
2.	The paper mentions token selection methods (RHO-1, Phi-4) but does not evaluate them. It is unclear how partial token dropping affects NTP’s regularization. Experiments across a spectrum—from full NTP to partial selection to pure CTP—would guide practitioners. Intermediate strategies like random token drop or top-k scoring are suggested.
3.	The theory relies on strong assumptions: near-uniform outputs at initialization, uniform Fisher matrix, and quasi-uniform logits at convergence. These may not hold for large-scale pretraining. The authors should discuss this and provide empirical checks (Fisher spectrum, logits uniformity).
4.	All experiments use small GPT-2 models (125M) trained from scratch. This limits realism. The paper misses the opportunity to explore larger and latest models, bigger corpora, or pretrained + fine-tuned setups. Comparing pretrained NTP and CTP on a downstream task would improve practical relevance. Alternatively, test whether the theoretical regularization scales with model or vocabulary size.
5.	The effect of vocabulary size, label frequency, and rare vs. frequent answer tokens on NTP is unexplored. Regularization dynamics are unclear. Tracking parameter norm or Hessian trace across training, with plots for NTP vs CTP, would clarify these effects.

### Questions
- Highlight some quantification of the main findings in the abstract and introduction.

- The legend for Figure 6 is confusing and should be clarified.

- Maintain a consistent color scheme for NTP and CTP throughout the paper (check Figures 3, 4, and 6).

- Move a brief explanation of the NTP shuffle experiment (Appendix A.1) into the main text, as it clearly shows structure vs. objective effects.

- Fairness checks showing NTP outperforms CTP under larger token/epoch budgets are only in the appendix. Include key results with actual numbers and a small table for 1×, 2×, and 4× CTP token budgets in the main text.

- Add a short, intuitive sketch in the main text explaining why the second term becomes (1/2) θᵀ I₀θ.

- Appendix D lists compute but omits key hyperparameters: learning rates, schedules, batch sizes, weight decay, and seeds.

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
3

### Summary
This paper analyzes the differences between Next Token Prediction (NTP) and Critical Token Prediction (CTP) training approaches. Through theoretical analysis and experimental validation, it proposes that NTP training approximates CTP training with added L2 regularization. Experimental results demonstrate that NTP models exhibit stronger generalization and reasoning capabilities than CTP models, consistently outperforming them across various tasks.

### Strengths
The paper conducts thorough theoretical analysis, trains all models from scratch, and compares NTP and CTP model performance alongside loss curve across multiple datasets.

### Weaknesses
In practice, NTP serves as the mainstream pretraining approach, while CTP—constrained by challenges like critical token identification—is primarily used for fine-tuning on specific tasks after pretraining. The theoretical approximation proposed in this paper is only applicable to the early stages of training for models trained from scratch. This limitation prevents the paper from adequately addressing the effectiveness of training approaches like CTP following NTP.

### Questions
1, It is recommended to include the CTP+WD training combination for comparison in experiments such as Figure 4. 
2, Provide performance comparisons between NTP and CTP training approaches using pretrained models on some difficult tasks.

### Soundness
2

### Presentation
3

### Contribution
2
