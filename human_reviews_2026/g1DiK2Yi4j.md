# Rethinking Data Selection: The Importance of Coverage over Difficulty in Generative Fine-Tuning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Selecting high-quality training data can reduce computation cost for LLM fine-tuning. Prior data selection methods have developed a variety of scores aiming to reflect what kind of information a data instance can provide to the model, in order to subselect instances for fine-tuning---and a majority of this prior work has focused on scores quantifying difficulty.
The intuition in such work is that more difficult examples are more informative, and can therefore lead to more efficient fine-tuning. 
While data selection based on difficulty has shown promise for smaller classification models, in this work we find that such scores are ineffective for fine-tuning LLMs on generative tasks because their narrow focus on ``difficult'' instances fails to capture the necessary diversity of the input data. We find that in generative tasks, such approaches always fall behind random selection, which our analysis reveals
is more representative of the underlying input space—i.e., has better coverage. Motivated by this, we propose a simple clustering-based selection method which selects data that is more representative of the underlying input distribution, enabling selection of smaller subsets of training data for generative tasks. Using a case study on Llama-3-8B (Grattafiori, et al., 2024) and  OLMO2-7B (Walsh, et al., 2025), we find that the coverage-based approach performs well above difficulty scoring, yielding performance at or above that of random selection across a set of generative tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates how well the sample selection strategies that use the sample difficulty perform for training large language models trained for generative tasks (represented as multi-choice quest answering datasets). Overall, the paper finds these methods are outperformed by simple random selection and postulates that it is due to limited coverage of the training data distribution (i.e., the selected samples are representative only of a part of the distribution and cannot achieve strong performance). Based on this, the paper proposes to select samples based on diversity by proposing to use a simple clustering method.

### Strengths
The paper is well-written and easy to understand. The methods that are used in comparison are explained in detail.


Dealing with an interesting problem -- how to select a subset of samples that would lead to more efficient training of LLMs without sacrificing any performance

### Weaknesses
**Main finding is already well-known**

The main finding of the paper is that selecting samples based on diversity (i.e., to cover the whole training dataset distribution) is already well known and utilised in many different selection methods -- although they were used for typical supervised setting with "smaller models" (e.g., BERT fine-tuning) it makes sense the finding would transfer to the use of LLMs as well. An interesting question would be whether this is still the case in the true generative tasks -- such as translation, free-text responses, etc. -- the paper evaluates it only on multi-choice question answering, which is close to the typical supervised/classification setting. In addition, there are already different well-established strategies that utilise diversity for selecting samples (i.e., active learning). As such, I do not believe that the contribution of the paper is sufficient.

**Missing related work/selection strategies**

The paper focuses solely on selection strategies that utilise difficulty for the selection (or a single strategy that combines difficulty with diversity). The strategies are either already well established for LLMs and NLP, or are transferred from image or supervised classification domain. However, there are many existing strategies that are completely missing -- active learning strategies (that include selection based on diversity), core-set selection strategies (that try to select a subset that is the most representative of the full dataset) or other strategies that also implement simple clustering and and often combine different properties (although they may be used for in-context learning) [1-12] (to name a few).

Many of these strategies are well-established and often used (e.g., active learning) so I would expect a more extensive comparison that would include some of these strategies as well, especially when many of them already optimise for the coverage of the dataset and so are very similar to the strategy proposed in the paper.

There are also some inconsistencies in the paper in regards to the selection strategies. First of all, you claim that the Dataset Cartography paper suggests selecting hard samples for fine-tuning -- however, the main finding from the paper is that combination of easy to learn and ambiguous samples often works the best. Second, you argue that you do not consider Forgetting method because it would require large number of epochs -- however, it requires to run the same number of epochs as the Dataset Cartography (and other approaches that utilise training dynamics). I would agree that defining what "forgetting" means for generative tasks (i.e., open ended tasks with free text answers where multiple answers might be correct), but you do not work in such setting -- in my opinion multi-choice QA is more or less a classification tasks as the model is presented with 4+ options and just selects one of them.

**Questionable experimental setup**

The experimental setup for training the LLMs is a bit questionable for me, although there is not many details included (batch size?). For example, the LLMs are trained only for 3 epochs, which in my experience is often insufficient, especially when the most difficult samples are selected. This is further reinforced by the results show in the paper, where often training the LLM with less than 25% of dataset leads to performance that is lower than the same model used without training (which for me indicates specific problems in the training process). At the same time, the exact same setup is used for subsets of different sizes, irrespective of whether 100 samples are used for training, 1000 or 10000. In each setting, the selection of these hyperparameters (combination of epochs and batch size) would lead to completely different models (sometimes undertrained, sometimes overtrained). I would suggest dedicating more time to finding an optimal set of parameters, especially considering that previous works found that selection methods outperform random selection on smaller sizes, but later all of them behave similarly to random selection.

Furthermore, for most of the methods, results from only a single run are presented. However, LLMs were observed to be especially affected by the randomness in training (which is most significant when using low number of samples), which can also be seen in the results, and so this setup could lead to biased results. I would suggest repeating the training multiple times and report the deviation from different samples as that would provide more comprehensive evaluation.

Overall, I believe that many of the findings in the paper can be explained by the experimental setup leading to undertrained models, and if fixed would lead to different findings.

**References**
1. A Survey of Deep Active Learning
1. Deepcore: A comprehensive library for coreset selection in deep learning
1. Datamodels: Predicting Predictions from Training Data
1. Data Curation Alone Can Stabilize In-context Learning
1. Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics
1. Cartography Active Learning
1. Finding Support Examples for In-Context Learning
1. Automatic Combination of Sample Selection Strategies for Few-Shot Learning 
1. EXPLORA: Efficient Exemplar Subset Selection for Complex Reasoning
1. Sample Efficient Demonstration Selection for In-Context Learning
1. MEAL: Stable and Active Learning for Few-Shot Prompting
1. On Training Instance Selection for Few-Shot Neural Text Generation

### Questions
What specific hyperparameters were used for the different models and training subsets?


How does the comparison between methods look like on true generative tasks (translation, summarisation, free-form question answering)?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a clustering-based data selection method for LLM generative finetuning that samples one instance from each cluster in a partitioned embedding space. While this method does not consistently outperform random selection, it demonstrates more stable, monotonically improving performance and robustly outperforms existing difficulty-based methods.

### Strengths
-	Compelling analysis and insight: The benchmarking and analysis of using difficulty-based methods for LLM generative fine-tuning provide valuable insights and well support the motivation of this work.
-	The proposed coverage-based method presents reasonable performance improvements over existing difficulty-based methods.

### Weaknesses
-	Unclear empirical advantage of the proposed method:  The primary weakness is that the proposed coverage-based method, while more stable, does not provide a clear and consistent performance improvement over the random selection baseline. It remains on par with random selection. This raises the question of why not just use random selection, which is simple, efficient and effective.
-	Limited technical contribution: The proposed method simply uses K-means to cluster the training data and selects one sample from each cluster, which is trivial. It does not introduce new techniques for data selection. In my opinion, the authors should try to improve their current method to achieve better performance, at least better than random selection.
-	Narrow model scopes and tasks: The proposed method is only evaluated on Llama 3 8B and OLMo 2 7B, which is limited. The performance on larger models (e.g., 13B and 32B) and other pretrained backbones (e.g., Qwen) is unclear. Furthermore, the experiments cover only multiple-choice QA tasks. Evaluation on a broader range of generation tasks, such as instruction-following and mathematics tasks, can further verify the effectiveness of the proposed method.
-	Clustering hyperparameter sensitivity: The proposed method's performance is likely sensitive to the choice of the number of clusters \$k\$. The paper fixes \$k = mn/100\$, but this is a heuristic. A more thorough ablation or justification for this choice would strengthen the soundness of this paper.

### Questions
See the Weaknesses. The authors should address these limitations.

### Soundness
2

### Presentation
3

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
This paper (1) evaluates some difficulty-based data point selection methods for their utility in selecting small portions of multiple-choice datasets for finetuning language models, (2) shows that the performance of these methods doesn’t beat random sampling, (3) shows that the performance of these methods correlates reasonably well with an intuitive notion of how much their resulting data subsets cover the distribution defined by the full finetuning dataset, and (4) demonstrate that a coverage-based dataset selection method works roughly as well as random sampling, with potentially lower variance.

### Strengths
The MAUVE-inspired coverage metric proposed in this paper is nice and thought-provoking.
I find the experiments well-documented, reasonably convincing for the claims made in the paper (though, see the weaknesses section.) I think that the superiority of random sampling as a subset selection strategy is a nice result, and the notion of coverage over difficulty being important is again thought-provoking.

I think with additional caveats in the introduction that mention how we’re only working with multiple-choice finetuning problems, this paper provides a nice, well-scoped result that demonstrates both the efficacy of random sampling as dataset selection over difficulty-based methods, and how a coverage-based method also does just about as well as random sampling.

### Weaknesses
The paper claims that the generative case of LMs makes for a difficult new aspect to dataset selection, but it only trains and evaluates on multiple-choice question datasets. This means the claim that the LM “retains its generative capabilities” is also questionable – if you finetune enough on multiple-choice datasets, models do not retain (meaningful) generative capabilities, despite technically still defining distributions over strings. I was surprised by the difference between the intro language (focus on generative aspects of LMs) and the experimental setting (multi-choice, even when parameterized through the likelihood of the choices under the LM’s distribution.)

Overall, I’m not sure why we care about finetuning efficiency in pretrained language models and I’d like the authors to motivate this better. Almost all of the FLOPs that go into the resulting finetuned model have already happened – during pretraining. Even though in the modern era RL is consuming a larger and larger fraction of the flops compared to pretraining, this paper doesn’t consider that setting. My understanding is that the datasets considered are pretty tiny – <4M tokens according to table 4 – so the benefits of using only 1% of that data are unclear to me. If we care about, e.g., sample efficiency – as in, the cost is not in the FLOPs in the model but instead in gathering the data – then the experimental setting should change to independently hyperparameter-optimize for each number of samples (instead of fixing the number of epochs, for example.) Also, the motivation provided by the authors for data selection for breaking scaling laws — citing Sorscher et al., 2023 is indeed for pretraining (or large-scale training in general.)

### Questions
- Why fix the number of epochs in training on a number of samples (instead of more epochs for the smaller sample counts?) I realize this may be to have fewer samples mean less compute going into finetuning, but the real cost of finetuning examples seems to be in getting them, not training on them. Even so, one could increase the learning rate in the few-sample case instead to try to learn more from them in the same amount of compute!
- Footnote 3 –  “In practice, we use the model’s hidden state representation at the last layer and last position as our embedding extractor.” I’m not considering this a weakness to be clear, but I can’t help myself saying that I recommend not doing this! This last layer is a really lexically-driven vector because it must be a low-rank representation of the distribution over the next word. Vectors from the middle of the network would probably work much better for a nice clustering! If you tried this already and it works worse, I’d love to know.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores different data selection methods for fine-tuning LLMs and comparing against difficulty based metrics such as Perplexity, Confidence, V-usable information, etc.). This paper defines the notion of coverage as how well a data selection represents the entire distribution of the training set. A key idea of this work is that coverage and a random selection baseline outperform difficulty-based metrics in improving model performance during fine-tuning.  The authors argue that diversity in training data is more important than difficulty scores.

### Strengths
1. The paper argues, with empirical evidence, that coverage (the representation of the entire data distribution) is more important for generative tasks like LLM fine-tuning than relying on difficulty-based metrics. It shows that coverage-based methods outperform methods based on difficulty (e.g., Perplexity, Confidence) in terms of fine-tuning model performance.
2. The experiments are well-executed with careful consideration of removing randomness in data selection and repeated trials - particularly time-consuming with limited resources and larger models.
3. Usage of k-means clustering of model embeddings offers a new and computationally efficient means of data selection in LLMs. It avoids re-training and reduces compute overhead.

### Weaknesses
1. This paper proposes the random baseline as a rightly powerful one. However, it would significantly benefit from comparisons with data-attribution methods [1] that exist with the newly proposed coverage-based approach. It will be interesting to see how coverage compares with existing dataset selection methods (semi-value-based or influence-function-based) for fine-tuning LLMs / generative tasks that implicitly or explicitly use distributional coverage - such as [2], [3], [4] etc. and when it could have a distinct advantage.



[1] A Survey of Data Attribution: Methods, Applications, and Evaluation in the Era of Generative AI
[2] Get more for less: Principled Data Selection for Warming Up Fine-Tuning in LLMs
[3] What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions
[4] DsDm: Model-Aware Dataset Selection with Datamodels

### Questions
1. Can the authors comment on the computational time complexity of this k-means clustering approach - at what point is it faster to train on a large dataset than a careful selection (since LLMs can generalize over the course of training with enough data).
2. Could the authors offer light on when their method is likely to have an advantage over the existing semi-value-based or influence-function-based dataset selection paradigms.
3. If data quality is low to begin with, can this approach sift out clean and high-quality data subsets (which could be a huge contribution). For instance, noisy data or bad quality data can bias coverage towards the long-tailed poor quality samples.

### Soundness
3

### Presentation
2

### Contribution
2
