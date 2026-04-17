# Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 8

## Abstract
Data teams at frontier AI companies routinely train small proxy models to make critical decisions about pretraining data recipes for full-scale training runs. However, the community has a limited understanding of whether and when conclusions drawn from small-scale experiments reliably transfer to full-scale model training. In this work, we uncover a subtle yet critical issue in the standard experimental protocol for data recipe assessment: the use of identical small-scale model training configurations across all data recipes in the name of ``fair" comparison. We show that the experiment conclusions about data quality can flip with even minor adjustments to training hyperparameters, as the optimal training configuration is inherently data-dependent. Moreover, this fixed-configuration protocol diverges from full-scale model development pipelines, where hyperparameter optimization is a standard step. Consequently, we posit that the objective of data recipe assessment should be to identify the recipe that yields the best performance under data-specific tuning. To mitigate the high cost of hyperparameter tuning, we introduce a simple patch to the evaluation protocol: using *reduced* learning rates for proxy model training. We show that this approach yields relative performance that strongly correlates with that of fully tuned large-scale LLM pretraining runs. Theoretically, we prove that for random-feature models, this approach preserves the ordering of datasets according to their optimal achievable loss. Empirically, we validate this approach across 23 data recipes covering four critical dimensions of data curation, demonstrating dramatic improvements in the reliability of small-scale experiments.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In the context of dataset curation, this work examines the situation where small proxy models are used to evaluate different datasets, in order to train a larger target model downstream for the best possible performance. The work finds that proxy models that are not hyperparameter-optimized are poor predictors of downstream performance. The work proposes a simple solution - train with smaller learning rates of 1-2 orders of magnitude less than standard LRs, as this makes the dataset rankings of the proxy models consistent with changes to the target model's performance given each dataset. The intuition and theory behind this solution is that small learning rates cause training to be dominated by the first order component of the gradient which optimizes for alignment of the model between the train and validation data. Using these small learning rates, experiments show that the dataset rankings given by small GPT proxy models are consistent with a) rankings of the proxy models with tuned hyperparameters, and b) rankings of a larger target LLM.

### Strengths
The problem of dataset curation is highly relevant, and the main result (that dataset ranking is sensitive to hyperparameters) is quite intriguing. The proposed method is simple but backed up by a theoretical arguments and experimental evidence. The paper is clearly presented and the intuition is made easy to follow.

### Weaknesses
The main question for me is whether the results are specific to large language transformers, or more troubling, only the GPT and Pythia models in the experiments. Given that the work already covers a lot of ground, I think instead of trying to generalize to other domains (vision especially) it would be best if the authors qualify their language and restrict their findings to the language domain specifically. As for generalizing from the specific models considered, although training is costly, it might be more convincing to have another target model that is somewhat different than the ones considered in the experiments. A weaker but cheaper experiment would be to show correlation between smaller-scale models of different types.

There is another question more for relevance: how large are the performance differences between different datasets? If they are quite small then the significance of the findings is not only diminished, but the findings themselves may again not generalize to real-world situations, e.g. if evaluating datasets with much larger differences in quality. Here I suggest a potential ablation - the authors could add some artificially bad datasets (e.g. by purposefully duplicating data) to demonstrate that their findings continue to hold outside of the narrow performance range of the datasets in this work.

Minor suggestion: figure 5 and 6 could be swapped as it seems more natural that way.

### Questions
How many replicates are in each of the experiments? A concern I have is whether some of the results (e.g. figure 3) are due to random chance from single runs. For some experiments this is clearly not happening, e.g. when computing correlation over multiple models or data recipes, but nevertheless it would be helpful for the authors to clarify when this may be an issue.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses how small "proxy" models can reliably select pretraining data for large-scale models. It identifies a key flaw in standard practice: proxies are trained with fixed hyperparameters, while large models are tuned for the selected data. The authors demonstrate this makes proxy-based dataset rankings highly fragile, even reversing with minor learning rate adjustments. The primary contribution is a new objective—to find the dataset with the best optimally-tuned performance—and a simple solution: train proxy models with a "tiny" learning rate. This "patch" is shown to be effective, achieving >0.95 rank correlation with tuned, large target models across 23 data recipes. The authors provide theoretical intuition based on first-order alignment and a proof for random feature models.

### Strengths
1. Clear Problem, Simple Solution: The paper identifies a relevant flaw in a common workflow (using fixed-HP proxies). The proposed solution ("use a tiny LR") is straightforward and easy to implement.

2. Helpful Empirical Validation: The paper provides a valuable set of experiments across 23 data recipes and 3 model families . Crucially, the authors use the correct (and expensive) ground truth: large models with per-dataset hyperparameter tuning. The results in Figure 2 and 5 clearly show the method's effectiveness .

3. Helpful Conceptual Reframing: The paper's reframing of the data selection objective (Section 3) is a useful perspective for the community. It correctly points out that we should optimize for a dataset's potential after tuning, not its performance in a fixed setting.

4. Supporting Ablation Studies: The appendices show that the tiny-LR method appears robust to other hyperparameter choices like batch size, weight decay, and token-per-parameter ratios, which helps isolate the learning rate as a uniquely sensitive factor .

### Weaknesses
1. Gap Between Theory and Claim: The paper calls its solution "theoretically-grounded", but the provided theory (Section 5.2) does not fully support its main cross-scale claim. The Random Feature Model analysis (Theorem 2) supports a within-scale claim (a proxy's tiny-LR loss predicts its own optimal loss) , but does not prove the paper's central thesis (that a proxy's loss predicts a target model's optimal loss). This disconnect undermines the claim that the solution is well-understood from first principles and makes it feel more like an (albeit effective) empirical heuristic.

2. Significant Scope Limitation (Single-Epoch): The paper's findings are restricted to single-epoch training. While this is a clean setting for a study, it is a major departure from many modern (especially Chinchilla-optimal) multi-epoch training regimes. The authors acknowledge this as future work , but it remains a significant open question whether the "tiny-LR" advantage holds when long-term optimization dynamics, data ordering, and sample repetition are in play. This limits the currently demonstrated applicability of the patch.

3. Lack of Mechanistic Evidence: The paper proposes "gradient alignment" as the intuition for why this works , but this hypothesis is not empirically tested. The paper would be much stronger if it provided direct measurements of the first-order (alignment) and higher-order terms across scales to show the former is stable while the latter is not. Without this, the "why" remains speculative, and the contribution is primarily an empirical observation.

### Questions
1. On the Theory Gap: The theory in Section 5.2 and Theorem 2 appears to only prove a within-scale result (that a tiny-LR RFM's ranking matches an infinite-width RFM's ranking). This supports Figure 4(a), but not the paper's main cross-scale claim (Figure 5a). Can you comment on this gap? Do you have any theoretical argument that actually bridges the proxy-to-target scale gap, or is the RFM analysis purely for intuition about the tiny-LR $\rightarrow$ optimal-loss link?

2. On Gradient Alignment: Your intuition relies heavily on "first-order gradient alignment" being preserved across model scales. Do you have more direct evidence for this? For instance, did you try to measure the gradient alignment term ($\nabla l_{val}(\theta)\cdot\nabla l_{i}(\theta)$) for different datasets at the 125M scale and the 1B scale (at initialization or after short training) and check if their ranking is also preserved? This seems like a more direct test of your hypothesis.

3. On Multi-Epoch Training: Your work is restricted to a single epoch. How do you speculate these results would change in a multi-epoch regime? One could imagine that over multiple epochs, the "higher-order" effects that you claim scramble rankings would have more time to accumulate, even with a tiny LR. Does the "patch" still work when models are trained for 2, 4, or 10 epochs? This seems like a critical question for real-world adoption.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the reliability of using small proxy models that use fix sets of hyperparameters to to guide data curation decisions for large-scale LLM training. The authors identify a critical fragility in current practices: dataset rankings from proxy models that rely on fixed set of hyperparameters to train on each datasets may not reveal correctly the utility of each dataset. Instead, the hyperparameters used to evaluate the utility of each dataset should be optimized for each dataset. The paper also identify that learning rate is the most important hyperparameter affecting the proxy models' dataset selection. They propose training proxy models with "tiny" learning rates (10⁻⁵ to 10⁻⁶) to improve cross-scale transferability, supported by theoretical analysis on random feature models and comprehensive experiments across 23 data recipes.

### Strengths
The paper is well-written. Addresses a real challenge faced by AI labs where data teams must make expensive curation decisions based on small-scale experiments. The demonstration that minor learning rate variations can completely flip dataset rankings (Figure 3) is compelling and well-presented. The tiny learning rate solution is simple to implement without requiring architectural changes or complex procedures.

### Weaknesses
Only single-epoch training (multi-epoch is increasingly important as data becomes scarce). I am not sure if in pratice, people only use single-epoch training to evaluate the utility of datasets. Also, of course, hyperparameter tuning on each dataset is the best to evaluate the utility of a dataset, however, it can be very costly to do in practice. Furthermore, smaller learning rate may make it slower to reach optimum point, which may affect the correct ranking of datasets if trained on single epoch. I feel this point is not addressed.

### Questions
Also, how are other factors such as dataset sizes, models type, and type of proxy model versus the actual model affect this ranking decision. I dont expect you to run too experiments but further discussion is appreciated.

### Soundness
2

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
This paper explores the reliability of small-scale experiments for predicting larger-scale training outcomes, which especially focuses on the data selection problem in the proxy model approach. A key finding is that learning rates can influence dataset rankings. To improve the accuracy of proxy model-based predictions, the paper suggests the use of sufficiently small learning rates. The paper provides both theoretical understanding and experimental evidence to support their findings.

### Strengths
- The paper offers valuable insights into the proxy model practice. The proposed idea (using small enough learning rates) is notably simple, which is a great advantage for real-world applications. 
- The paper provides rigorous experimental results and detailed findings, which contribute to the reliability of the study.
- The paper includes various theoretical analyses that deepens the understanding of the underlying reasons of findings.

### Weaknesses
- While the Appendix includes some ablation studies on other hyperparameters, more explicit and detailed explanations regarding the impacts of these hyperparameters would greatly enhance the clarity and utility of the paper. It would allow readers to better understand the sensitivity and robustness of the model to different configurations.
- Further discussion on how the observed findings might change when applied to different training strategies, such as reinforcement learning vs supervised fine-tuning, could significantly broaden the paper’s scope and strengthen its overall conclusions.

### Questions
The major questions and suggestions are in the weaknesses section.

### Soundness
4

### Presentation
3

### Contribution
3
