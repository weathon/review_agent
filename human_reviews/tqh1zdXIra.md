# Quick-Tune: Quickly Learning Which Pretrained Model to Finetune and How

- Decision: Accept (oral)
- Scores: 8, 8, 8, 8

## Abstract
With the ever-increasing number of pretrained models, machine learning practitioners are continuously faced with which pretrained model to use, and how to finetune it for a new dataset. In this paper, we propose a methodology that jointly searches for the optimal pretrained model and the hyperparameters for finetuning it. Our method transfers knowledge about the performance of many pretrained models with multiple hyperparameter configurations on a series of datasets. To this aim, we evaluated over 20k hyperparameter configurations for finetuning 24 pretrained image classification models on 87 datasets to generate a large-scale meta-dataset. We meta-learn a gray-box performance predictor on the learning curves of this meta-dataset and use it for fast hyperparameter optimization on new datasets. We empirically demonstrate that our resulting approach can quickly select an accurate pretrained model for a new dataset together with its optimal hyperparameters.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Currently there are many models available online but it is challenging to decide which of them to select and how to fine-tune it to the downstream task. The paper focuses on this challenge and introduces a method to find a suitable pretrained model together with suitable hyperparameters for fine-tuning. The authors construct a meta-dataset and use it for meta-learning a performance predictor that can jointly select the model and the hyperparameters.

### Strengths
* The paper studies an interesting problem setting that is practically important and likely to further increase in importance. The trend is that there will be more and more pretrained models available and these will be more commonly used, so a method for deciding how to select a suitable one and fine-tune it is valuable.
* The method seems to be novel and appears to be a well-designed solution to the presented problem. It also offers strong performance compared to the baselines that are evaluated.
* The paper is well-written and easy to read. The figures and tables are well-presented and make it simpler to understand the work and the results. The questions that are studied are clearly stated.
* The paper contributes a meta-dataset that can be interesting for other researchers to study this important problem.
* A variety of relevant research questions are asked and answered reasonably well.

### Weaknesses
* The paper only considers computer vision classification, while related works such as LogMe (You et al., ICML’21) consider various downstream tasks (classification and regression), and modalities (vision and language). It can be impractically expensive though to evaluate the approach on also these other settings.

### Questions
* There are micro, mini and extended versions of the Meta-Album datasets, which poses the question if Quick-Tune could benefit from using the smaller subsets for more quickly finding settings that work well on the larger subsets. It would be interesting to study this to see if it works for Quick-Tune and how large speedups can be obtained.
* How well does the method work on other downstream tasks or data modalities?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work presents and evaluates an up-to-date combined model and hyperparameter selection method for transfer-learning/finetune setup. It works based on observing the learning curves of training runs and iteratively select which run to continue further based on bayesian optimization. The model maintains a performance predictor and a cost predictor, which is being updated as the search proceeds. The model parameters are/can be meta-learned.

The method is evaluated on a search space composed of 24 models on a pareto curve of number of parameters and ImageNet accuracy and on 86 tasks. The hyper parameter search space includes relevant settingslike finetune strategies, regularisation, data augmentation, optimizers, learning rate schedule.

The method is compared against other approaches such as: single model + HPO and model selection + HPO for selected model.

### Strengths
Model selection and hyperparameter optimization are practical problems many ML practitioners encounter. It is welcome to see a method able to tackle both together and that doing so provides benefits from doing it step wise. This reviewer is not aware of model selection papers / transferability estimation doing joint optimization in a recent finetune/transfer setting as in here.

The paper setting with 24 models and 86 datasets from Meta-Album when training for up to 1,4,16 hours seems reasonable and one practitioners can relate to.

### Weaknesses
I do not follow how the normalised regret is calculated. In particular how is y_max and y_min calculated? Is it provided by Meta-Album datasets? Is it the min/max of all runs ever done on this study? How significant is a 10% regret and is there any more expensive way to close that gap when using this approach?

The curves on plots like figure 3, shows a different behaviour between the methods. It is hard to predict if quick-tune always beats them or if that story changes as the wallclock time gets extended. It would be interesting to see if the search approaches regret 0 or stays ~10% above it.

One issue with model selection in particular with small datasets is overfitting, including to the validation set. I expected some discussion around this and also an explicit reference to which splits are used during which phase.

### Questions
What splits are being used to train, guide search and report test performance? It would be good to have that explicit in the text.

How is normalised regret calculated?

How do the curves in figure 3 look like when the wallclock time gets extended?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
* A challenge when in the  pretraining (PT) / finetuning (FT) paradigm is deciding (1) what PT model to use for a given task and (2) what hyperparameters to use when FT it. This paper presents a method to identify the best pretrained model and finetuning hyperparams to use for a new dataset, using bayesian optimization/
* The proposed method first pretrains surrogate models on a large meta-dataset of finetuning pipeline runs, which captures variation in datasets, model architectures.
* These surrogates are then used to define an acquisition function that defines how to select a finetuning pipeline (model specification and hyperparameter set) during each step of Bayesian optimization. Once more data is acquired, the surrogates are also updated.
* The specific acquisition function is a variation of expected improvement, including a term that captures the cost of running the finetuning pipeline (as opposed to being based purely on performance alone).
* In experiments, the surrogates are trained on a large set of learning curves from the timm benchmark. The algorithm is then applied to the Meta-Album dataset, and the results demonstrate that the proposed method outperforms baselines.
* Ablations where the meta-training step and cost-aware acquisition are removed demonstrate that both parts are important.

### Strengths
* Interesting problem choice, and original direction (to the best of my knowledge)
* Presentation, technical detail, and experiments are good quality.
* Proposed method has strong performance  on benchmarks considered
* Ablations of the cost-aware component of the acquisition function and meta-training were informative.

### Weaknesses
Paper is strong and has thorough results. The one thing I was curious about was how the method performs on other standard benchmarks such as Imagenet, and whether any of these results can be validation on different domains (eg text datasets, where finetuning is also very common).

### Questions
See above.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a Bayesian meta-learning approach called Quick-Tune to jointly optimize choice of models and hyperparameters for finetuning, i.e. a Combined Algorithm Selection and Hyperparameter Optimization (CASH) approach.  Quick-Tune builds on top a previous grey-box hyperparameter optimization approach called DyHPO with a cost-aware acquisition function and a meta-learning approach to warmstart loss and cost estimators.  Experiments on the Meta-Album benchmark for few-shot learning optimizing over a search space including models from TIMM shows Quick-Tune to efficiently select models and associated hyperparameters for new tasks, exceeding other standard hyperparameter methods as well as two-stage model selection and hyperparameter tuning baselines.  As part of training QuickTune, the authors also collect a meta-dataset of learning curves with associated hyperparameters and datasets to add to the set of meta-learning benchmarks.

### Strengths
- Paper is well written and clear.
- Quick-Tune addresses an important problem of how to efficiently select and tune models from a model hub for finetuning/transfer learning on a new dataset.
- Experiments appear to be thorough and high quality.

### Weaknesses
- In the ablation study, for the Micro and Mini cases, QT: -M, +C, +G (DyHPO with cost-awareness) performs as well as QT: +M, +C, +G (DyHPO with cost-awareness and meta-learning) which shows most of the benefit coming from the cost-aware aspect of QuickTune and not the meta-learning.
- Novelty is limited since the core of Quick-Tune is DyHPO, a prior HPO method.  Cost-aware acquisition functions have been used in the past and the approach of using meta-features and meta-learning good initializations for the estimators also lack originality.

### Questions
- What is the unnormalized performance of the approaches in Figure 3 and Figure 4?
- Instead of rank, how would Figure 1 look as a heatmap of unnormalized performance?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
