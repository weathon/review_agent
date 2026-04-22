# Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Unlearning in large language models (LLMs) aims to remove specified data, but its efficacy is typically assessed with task-level metrics like accuracy and perplexity. 
We demonstrate that these metrics are often misleading, as models can appear to forget while their original behavior is easily restored through minimal fine-tuning. 
This phenomenon of \emph{reversibility} suggests that information is merely suppressed, not genuinely erased. To address this critical evaluation gap, we introduce a \emph{representation-level analysis framework}. 
Our toolkit comprises PCA-based similarity and shift, centered kernel alignment (CKA), and Fisher information, complemented by a summary metric, the mean PCA distance, to measure representational drift. 
Applying this framework across six unlearning methods, three data domains, and two LLMs, we identify four distinct forgetting regimes based on their \emph{reversibility} and \emph{catastrophicity}. 
Our analysis reveals that achieving the ideal state--irreversible, non-catastrophic forgetting--is exceptionally challenging. 
By probing the limits of unlearning, we identify a case of seemingly irreversible, targeted forgetting, offering new insights for designing more robust erasure algorithms. 
Our findings expose a fundamental gap in current evaluation practices and establish a representation-level foundation for trustworthy unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
* This paper evaluates unlearning in two settings: single (where all unlearning requests are available simultaneously) and continual (where unlearning requests arrive sequentially).
* The paper defines four unlearning regimes that vary along two axes: reversible vs. irreversible (i.e., whether the unlearned knowledge can be recovered by lightweight retraining) and catastrophic vs. non-catastrophic (i.e., whether the unlearning process significantly affects unrelated knowledge).
* Evaluating 3 unlearning methods and 2 LLMs, the paper first shows that unlearning methods cannot remove knowledge irreversibly. 
* Then, the paper describes and applies several methods for measuring representational similarity between the original LLM and its unlearned and retrained variants, finding that representational similarities correlate with reversibility of unlearning. 
* This is complemented by a theoretical analysis and a short case study of achieving irreversible, non-catastrophic unlearning. 
* The paper concludes that achieving irreversible unlearning remains hard, and representation analysis offers a new perspective on unlearning beyond accuracy on retain and forget sets

### Strengths
**(S1)** The paper successfully demonstrates that current methods do not achieve irreversible and non-catastrophic unlearning

**(S2)** The introduced taxonomy may be helpful in future work to better systematize and discuss the achievements of new unlearning methods

**(S3)** The paper makes a convincing argument that accuracy metrics alone give an insufficient impression of unlearning success, and analyzing the model's internal representations can give important insights beyond accuracy

**(S4)** The paper contains one example of successful irreversible and non-catastrophic unlearning, demonstrating that this goal may be achievable.

### Weaknesses
**(W1)** One major concern is originality: The evaluation of unlearning methods successfully confirms that current methods do not achieve irreversible unlearning, but this is a known fact, as the paper also mentions (e.g., [24]). Likewise, the observation that models break down when applying multiple edits in continual learning has been reported before, e.g. [a]. Finally, none of the proposed metrics for representation analysis is novel.

**(W2)** The paper does not contain any actionable insights. The representation analysis confirms that larger representational dissimilarity correlates with irreversible unlearning, but this is expected as the original model becomes harder to reconstruct the further the parameters move away from it. More importantly, the paper does not give practical tools, for example, how representation similarity can reliably predict successful and irreversible unlearning, which would be very helpful in practice. Overall, the takeaways from this analysis remain unclear: What is the reader to conclude beyond the observation that a larger representation shift correlates with irreversible unlearning?

**(W3)** The theoretical analysis mirrors this problem: It mainly shows how a larger distortion of weights leads to greater dissimilarity (which is intuitive). The connection to irreversible unlearning is not formalized but only claimed in the paragraph starting with line 412. Sec. 5.2 additionally discusses that model outputs are not a reliable indicator of unlearning. However, this is also not a significant finding, as keeping all LLM parameters except those of the last layer frozen while randomizing parameters in the last layer will yield a practically random model (from the black-box perspective), while it is very likely that most information and knowledge learned by the model will continue to be accessible from earlier layers.

**(W4)** The paper overall focuses on a narrow setting where unlearning and catastrophic forgetting are measured through fixed forget and retain sets. However, it does not consider recovering unlearned knowledge through prompt attacks (e.g., [b]) or mechanistic interpretability (e.g., [c], only intended as an example, not available before submission deadline).

**(W5)** LLM unlearning is a popular research area with many methods. Claims that are meant to be generalizable to the entire field, such as the one in this paper, need to be either evaluated on a large set of methods or require a motivation for why the chosen set of unlearning methods is representative and will give such generalizable insights. This aspect can be expanded upon in the current paper.

**(W6)** The experiment in 5.3. appears very interesting, because it directly targets the case of irreversible, non-catastrophic unlearning. This experiment could be one starting to inform more successful unlearning methods. Therefore, I think it would be very interesting to expand this perspective. One concern I have is to what extent this observation is due to the "more constrained relearning conditions" vs. actually successful unlearning.

**(W7)** The supplementary material contains a large number of plots showing the representation similarity measures for different LLM layers. These plots are not individually interpreted or put in context. Their role in the paper is therefore doubtful. If they do not add any tangible value to the paper, consider removing them.

### References
[a] Thede et al.: Understanding the limits of lifelong knowledge editing in llms. In arXiv, 2025\
[b] Patil et al.: Can sensitive information be deleted from llms? objectives for defending against extraction attacks. In ICLR, 2024\
[c] Cywinski et al.: Eliciting Secret Knowledge from Language Models. In arXiv, 2025

### Questions
* Which novel perspectives on unlearning does this paper give the community beyond confirming known problems with current methods?
* Which actionable improvements in LLM unlearning are informed by the representation analysis? What are the main insights beyond the expected observation that higher representation dissimilarity correlates with irreversible unlearning?
* How can we motivate the findings in this paper to generalize to most methods for LLM unlearning, even those not evaluated in the paper? How about extraction attacks beyond retraining?

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
This paper studies two aspects of machine unlearning: reversibility and catastrophic forgetting.
Reversibility is probed via relearning, i.e., a one-epoch finetune on the forget set.
The authors show that, in many cases, this simple adjustment of model weights by relearning restores much of the original model’s performance.

From this, they infer that the knowledge was not effectively removed and can be readily recovered. They further support this by analyzing the intermediate representation space of LLMs and measuring how the unlearned-then-relearned model deviates from the original.

Overall, the paper finds that relearning often recovers performance, presenting this as a failure mode of unlearning and evidence that the underlying knowledge persists.

### Strengths
The idea of studying how easily unlearned knowledge can be recovered after unlearning is quite interesting. In particular, applying relearning and then evaluating the model’s recovery is a valuable direction that deserves further exploration.

### Weaknesses
I don’t find the results of this paper particularly surprising. A single step of finetuning on the forget set can naturally bring back the forgotten knowledge. I don’t quite see why the authors expected this not to work.
After all, with more aggressive settings (e.g., two or three additional epochs), one could almost certainly recover the utility on the forget set. Restoring performance through one epoch of finetuning is not unexpected.

In general, unlearning methods that are truly “irreversible” often achieve this by severely degrading the model, seen as a drop in accuracy on the retain set and overall utility.
A more interesting direction, in my view, would be to study the sample efficiency of relearning: can we recover performance using only a few samples or perhaps by providing them as in-context examples instead of full retraining?

Also, I recenlty found a paper on knowledge recovey of machine unlearning [1], I guess this also worth being discussed in this paper.
[1] Rezaei, Keivan, et al. "RESTOR: Knowledge Recovery in Machine Unlearning." arXiv preprint arXiv:2411.00204 (2024).

### Questions
They are discussed in the weakness section.

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
This paper highlights the drawback of relying solely on task-level metrics (e.g., accuracy, perplexity) for evaluating unlearning in LLMs, since these metrics cannot distinguish genuine erasure from superficial forgetting. To bridge this evaluation gap, the paper introduces a **representation-level analysis framework** to measure representational drift and categorize unlearning behavior into four regimes. The study concludes that achieving the ideal state—**irreversible and non-catastrophic forgetting**—is extremely challenging, and further provides a method combination that achieves a *seemingly* irreversible, non-catastrophic form of forgetting.

### Strengths
- The paper is well-written and easy to follow.
- It clearly identifies the limitations of current task-level evaluations and proposes a **representation-level toolkit** that goes beyond surface metrics.
- Provides clear definitions and a systematic taxonomy of forgetting regimes.

### Weaknesses
- Table 2 demonstrates the weakness of task-level metrics, but it would be stronger to include results on the **Qwen2.5-7B** model to further consolidate this finding.
- It remains unclear whether the same observations hold for **smaller (3B) or other model families (Llama)**.
- The framework measures representational drift but does not formally assess **privacy leakage**; the notion of “irreversible forgetting” is still heuristic.
- The proposed solution is interesting, but **cross-model validation** would strengthen its generality.

### Questions
- Does the proposed framework also generalize to **LLaMA** or **Qwen3** models?
- Could the **mean PCA distance** be correlated with formal privacy metrics such as **MIA AUC** in a consistent way?
- In Tables 2 and 3, how relearning is conducted?

### Soundness
3

### Presentation
3

### Contribution
2
