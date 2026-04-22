# Large Language Models do Not Make Complete Use of Math Reasoning Data

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
In deep learning, increasing dataset size has been shown to improve the performance of deep neural networks. However, it is unclear if these models are able to make complete use of the data that they are trained on. Understanding this is especially important in the current large language model era, where data scarcity has become a pressing issue. We discover that when performing fine-tuning on mathematical reasoning tasks, adding more training data causes the model to incorrectly answer a large portion of previously correctly answered test samples. This remains true even with popular test-time scaling techniques, which can iron out inconsistencies in model predictions. To better understand this phenomenon, we show both empirically and theoretically that models trained using Supervised Fine-Tuning and Reinforcement Learning are incapable of making complete use of the data that they are trained on, where models trained on the same data learn very different functions across different random seeds, exhibiting high predictive multiplicity. This work contains novel insights that can aid in improving a model's ability to effectively scale its performance with more data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies per-item generalization when fine-tuning LLMs on math-reasoning tasks. As training data is incrementally increased, many test items that were previously answered correctly flip to incorrect, so net accuracy improves only marginally because “newly correct” and “newly incorrect” items roughly cancel. The finding is demonstrated for SFT (e.g., Llama-3-8B, Gemma-3-4B on GSM8K/MAWPS) and RL (Qwen2.5-0.5B on GSM8K/MATH8K). The authors quantify a “Union vs Final” gap (items solved by any intermediate model vs the final model), show cross-seed “predictive multiplicity” (same data, different seeds → different test items solved), and offer a high-level explanation via a “strategy set” view of reasoning traces. Ablations suggest sample order and LoRA dropout contribute to divergence across seeds. No new training method is proposed; the work is diagnostic/observational.

### Strengths
– Clear empirical phenomenon: as data scales, substantial per-item flips from correct to incorrect persist, limiting net gains.

– Tracked at the item level across both SFT and RL, with multiple base models/datasets; the “Union vs Final” analysis is informative.

– Fixed-set, cross-seed analysis highlights predictive multiplicity (similar accuracy, different solved sets).

– Ablations (sample order, LoRA dropout) begin to isolate contributors to across-seed divergence.

– The problem is practically important for data scaling, curation, and evaluation of math-reasoning LMs.

### Weaknesses
– Limited causal explanation: the paper establishes that flips occur, but the proposed “strategy set/Rashomon” lens is descriptive and not predictive; it does not isolate necessary/sufficient causes of flipping or quantify their contributions (e.g., data conflicts vs optimization noise vs under/over-fitting vs decoding effects).

– Scope is narrow (math-reasoning, modest model sizes); it is unclear how broadly the phenomenon holds (code, multilingual, instruction-following, safety, etc.), or how it scales with much larger base models and longer training.

– No actionable solution: beyond noting order/dropout effects, the work stops short of proposing methods to reduce flips or close the Union-vs-Final gap (e.g., curriculum, data reweighting, conflict detection, checkpoint ensembling, order-invariant updates, strategy-diversity regularizers).

– Experimental controls are thin in places: small number of seeds; limited statistical testing; compute/training-length/early-stopping effects not deeply probed; decoding settings (e.g., temperature/self-consistency) only partially explored.

– The “union” signal naturally suggests simple mitigations (checkpoint ensembling across data-subset steps, EMA over training, mixture-of-checkpoints) that are not tried; without testing such baselines, the practical impact remains unclear.

– Theoretical component does not yield falsifiable predictions (e.g., when flips should increase/decrease given measurable dataset/model properties).

### Questions
•  What fraction of flips can be attributed to measurable data conflicts (near-duplicates with differing rationales/solutions, annotation noise) vs optimization stochasticity? Can you quantify this via conflict detection or per-sample gradient similarity analyses?

•  Can you predict which items will flip when adding data? For instance, are low-margin items (by log-prob gap), longer reasoning chains, or particular operation types more flip-prone?

•  Does the phenomenon persist with substantially larger base models and longer training to convergence? How does it scale with training steps/epochs and gradient noise scale?

•  What is the effect of decoding schemes (temperature, self-consistency, verifier-guided selection, tool-use) on flips beyond majority voting?

•  Do simple mitigations narrow the Union-vs-Final gap: (a) checkpoint ensembling across subset steps, (b) EMA of weights, (c) curriculum or order-invariant batching, (d) removing LoRA dropout and fixing order for all conditions, (e) data deduplication/cluster-balanced sampling?

•  Does full-parameter fine-tuning (no adapters) or different PEFT choices alter the flip rate?

•  Can the strategy-set view be made predictive (e.g., estimating effective strategy entropy per item) and tested against flip rates?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper foucs on one question do LLMs make complete use of the training data and the short answer is no which they show on different smaller models using and number of math datasets from GSM8k to Hendryks math. The models are mostly small with Lamma3 8B or Gemma 4b.

### Strengths
The study has interesting findings and opens a research agenda for smaller models. I find it interesting to know.  Nice focus on one and important big questions.

### Weaknesses
I wonder more about the reason, is it the size and limited parameter so that the model has to make trade-offs. 

The paper would be strong if the authors could show how this scales between models, you do not need to go to bigger models but could also smaller if there is still a signal found.

### Questions
The question seems to be difficult: is it related to size? You use 8b and 4b models, do you observe differences? They have just different capacities and I guess the smaller models have to cramp the knowledge into a few parameters? Does this hypothesis hold or is it something else? They are a bit from different generation of models.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates how large language models (LLMs) improve performance during fine-tuning on mathematical reasoning tasks. Contrary to the assumption that more data monotonically improves performance, the authors find that adding additional training samples often causes models to forget previously capabilities — around 10–15% of correctly answered test samples become incorrect after adding more data. This behavior persists under both supervised fine-tuning (SFT) and reinforcement learning (RLVR) training paradigms, and even with test-time scaling techniques such as majority voting. The authors attribute this phenomenon to high predictive multiplicity (the Rashomon effect), where models trained on the same data but with different random seeds learn significantly different functions, leading to diverse sets of correctly answered questions. The paper argues that LLMs make incomplete use of math reasoning data because of this multiplicity and sensitivity to training randomness.

### Strengths
1. Interesting empirical finding: The paper opens the black-box performance in standard scaling-law assumptions and highlights a new failure mode of data scaling, which is crucial to understanding generalization in LLM finetuning.

2. Comprehensive experimental design: The study spans multiple models (Llama3-8B, Gemma3-4B, Qwen2.5-0.5B) and training paradigms (SFT and RL), making the findings robust across architectures and methods.

3. Clear visualization and metrics: Figures and fixed-set analyses (e.g., intersection analysis across random seeds) effectively communicate the instability in model predictions.

### Weaknesses
1. The claim "Large language models do not make complete use of training data" is inaccurate to summary the paper conclusion and the term "make use" is ambiguous. The main gist of the paper is generalization capacity rather than training data/sample efficiency, and large language model can certainly "make use" of training data by memorizing them correctly.

2. The message from the theoretical calculation of mode density is unclear. While it is unclear why a "strategy set" concept is necessary to introduce into the calculation, there is also a lack of explanation on the result of the calculation and how it implies or adds to the empirical findings.

3. The paper does not distinguish the proposed forgetting phenomenon from the traditional overfitting phenomenon, as the paper neglected another important factor of training epochs (in each "step") that also affect model performances. While for different "steps" the model is trained on different training sets, the model also goes through different epochs on training data and the forgetting phenomenon may be attributed to severe overfitting on small datasets.

Can you provide more details on how many epochs the model went through in each training stage, and how does model performance / newly correct/incorrect number changes in the process? Can you confirm that for every training set respectively, the training epoch is tuned optimally?

4. There is shortage of fine-grained analysis on the training / newly-correct / newly-incorrect problems regarding topics, difficulty levels, solution length, etc.

### Questions
1. Can you elaborate on the weaknesses above?
2. What does "random seed" in the paper control in the training process and how are the data shuffled? This term is blurry in the original paper.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the phenomenon where fine-tuning LLMs on increasing amounts of mathematical reasoning data does not lead to monotonic improvements on a fixed test set. The authors demonstrate that as more training data is added, a significant portion (10-15%) of previously correctly answered test samples become incorrectly answered. This "churn" mitigates the net performance gain. The authors attribute this behavior to high "predictive multiplicity", where models trained on the same data with different random seeds learn vastly different functions. These models achieve similar overall test accuracies but agree on a surprisingly small subset of correctly answered questions. The paper supports these claims with experiments using both SFT and RL on models like Llama3, Gemma3, and Qwen2.5 across datasets such as GSM8K, MAWPS, and MATH.

### Strengths
1.  The paper compellingly demonstrates a non-trivial and counter-intuitive aspect of scaling laws at the sample level. 

2. The experiments are systematically conducted across multiple models, datasets, and training paradigms (SFT and RL). The use of multiple seeds is crucial and well-executed. 

3. The authors proactively address potential confounding factors by showing the phenomenon persists even with majority voting, a standard technique to reduce prediction variance.

### Weaknesses
1. The paper frames its findings as LLMs "not making complete use of data." This is a very strong claim. An alternative, and perhaps more standard, interpretation is that this is a natural consequence of stochastic gradient-based optimization in a high-dimensional, non-convex landscape. The optimizer is finding a new local minimum to accommodate new data, which inevitably shifts the decision boundary for old data. The paper does not sufficiently differentiate its findings from the well-established field of catastrophic forgetting or continual learning.

2.  The analysis in Section 4.2, which introduces the Rashomon set, feels like a post-hoc application of a known concept rather than a deep analysis. The assumptions made are overly simplistic and the calculations for the "number of permissible models" are combinatorial thought experiments. They do not provide a rigorous link between the training dynamics (e.g., optimizer, learning rate) and the size of the observed Rashomon set.

3.  The experiments primarily use PEFT methods. It is unclear if this phenomenon is exacerbated by the constrained updates of PEFT. A comparison with full fine-tuning would be necessary to claim this is a general property of LLM training.

### Questions
1.  How do the authors distinguish their findings from the classic problem of "catastrophic forgetting" in continual learning? While the setting isn't strictly continual, adding batches of data and observing performance degradation on previously learned samples seems highly related.

2.  The "Union" accuracy in Figure 3 is significantly higher than the final model's accuracy. This strongly suggests that ensembling the models from each training step would be a very effective mitigation strategy. Did the authors consider exploring this or any other method to address the information loss? 

3.  Could the authors comment on the role of PEFT in their observations? Is it possible that the low-rank updates inherent to methods like LoRA make the model more susceptible to this "churn," as the capacity to integrate new information without disrupting old knowledge is constrained? How would these results compare to full fine-tuning?

4. The theoretical analysis in Section 4.2 makes a strong simplifying assumption in "Setting 1" that models in the Rashomon set *only* make mistakes on baseline-correct items and never correct baseline-incorrect ones. This seems unrealistic. How does the analysis hold if this assumption is relaxed to a more plausible trade-off?

### Soundness
2

### Presentation
3

### Contribution
2
