# Compute-Constrained Data Selection

- Decision: Accept (Poster)
- Scores: 5, 5, 5, 8

## Abstract
Data selection can reduce the amount of training data needed to finetune LLMs; however, the efficacy of data selection scales directly with its compute. Motivated by the practical challenge of compute-constrained finetuning, we consider the setting in which both the cost of selecting data and training are budgeted for. We first formalize the problem of data selection with a cost-aware utility function, 
and model the data selection problem as trading off initial-selection cost for training gain. We run a comprehensive sweep of experiments across multiple tasks, varying compute budget by scaling finetuning tokens, model sizes, and data selection compute. Interestingly we find that many powerful data selection methods are almost never compute-optimal, and that cheaper data selection alternatives dominate both from a theoretical and empirical perspective. For compute-optimal training, we find that perplexity and gradient data selection require training-to-selection model size ratios of 5x and 10x, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors present a study on compute constrained data selection for training large language models (LLMs).
Unlike preceding works, they do not constrain the size of the training set, but the compute, which is the sum of computational expenditure for data selection as well as LLM training. 
They compare various existing data selection methods under this setting and come to the conclusion that many powerful data selection methods are almost never compute-optimal due to their computational cost, making cheaper data selection the favorable choice.

### Strengths
- The study is well motivated and its results are of practical importance for finetuning large language models.
- Empirical findings correspond well with theoretical framework

### Weaknesses
- The title of the paper is rather broad, while the study is rather specific. "Compute-Constrained Data Selection" does not indicate which type of data is selected for which type of task.

Minor remarks:
- p.3 line 3: \mathcal{S} \subseteq \mathcal{D}, as \mathcal{X} is not introduced
- p. 6 bottom: methods -> method

### Questions
- Regarding the notion of utility in Section 3: Is utility here something that is to be minimized, i.e. alternatives with lower utility are preferred over alternatives with higher utility? In the remainder of the paper (expected) model performance is considered for which clearly higher values are preferred.

- I am not sure whether I understood the greedy data selection introduced in Sections 3 and 4. I am under the impression that all data points are scored individually and afterwards they are ranked according to their scores and selected until budget K is exhausted. Isn't it neccesary to do this in an interleaved manner, in order to capture effects like redundancy and the submodularity of utility? Consider the extreme case in which the most informative data point x is repeated K in the dataset, then we would end up with a selection that contains the same datapoint K times.   

- In Figure 2, the plot represents performance of Mid-PPL, while the plot in Figure 3 represents performance of Top-PPL. What is the reason for this discreapency?

- In Figure 2, what exactly is the dashed line? Shouldn't the Pareto front contain all solutions, that are dominating on the dimensions of compute (low) and accuracy (high)? The line is straight, is it something like a linear regression applied to the solutions on the Pareto front?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper considers selecting data for finetuning LLMs under a computational budget. The computational cost is divided into two parts: 1) when using the validation set to evaluate the performance, the validation set will incur an initial cost; 2) training on each sample will cost a fixed amount of computation. The authors propose an exponential distribution model to fit the model performance v.s. the training costs for four types of data selection methods: lexicon-based, embedding-based, perplexity-based, and gradient-based. The paper consists of numerical experiments over several models and several tasks.

### Strengths
1. This paper considers an interesting problem, data selection under computational constraints, and has interesting observations that the initial cost cannot be neglected when considering the computational budget.

### Weaknesses
1. (Major) Lack of novelty: although this paper proposes a framework for analyzing the computational cost of each data selection method, it does not provide any new techniques based on this framework. Furthermore, the key observation is not very surprising: the computational cost contains an initial cost when evaluating the validation set, thus the perplexity-based or the gradient-based is clearly not optimal under a limited compute budget.
2. (Major) Lack of soundness: a) the parametric model is selected to be an exponential distribution and the model is fitted to minimize the squared error, but the choice is never justified by any theoretical analysis or numerical results. The fitted curve is also not very convincing (e.g. Figure 3 and Figure 7). b) The Pareto frontier is never formally defined in this paper nor sufficiently discussed. It's very hard for me to believe that the fitted Pareto curve is indeed Pareto optimal as the points in Figure 8 and Figure 10 exceed the Pareto frontier by a large margin. Also, the fitted exponential curves surpass the fitted Pareto curve considerably in Figure 3, implying that the two curves even contradict each other.
3. (Moderate) Lack of insights: this paper is rather an experiment report than a well-motivated paper. The motivation for studying such a computation-constrained data selection problem is not fully supported. The authors just launch a bunch of models, adopt several tasks, and collect all the results without providing sufficient analyses.
4. (Moderate) The style file seems to not follow ICLR 2025 templates: the lines are not numbered.
5. (Minor) Typo(s):
  Figure 1 "using a much larger base model then data selection model": "then" should be "than".

### Questions
See the weaknesses part.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies a framework considering the practical challenges of training and fine-tuning large language models (LLMs) under computational constraints. It has established a trade-off between achieving better performance using larger data and lowering computational costs by selecting smaller subsets of data. A key takeaway is that simpler data selection methods, such as lexicon-based and embedding-based approaches, often provide a more efficient solution compared with more complex, compute-intensive methods like perplexity-based and gradient-based strategies.

### Strengths
This paper addresses compute-efficient fine-tuning, which is an important task in training LLM. Extensive simulations are conducted to provide empirical evidence and support the framework.

### Weaknesses
1. Although the author claims some simple methods such as Lexicon outperform the complex ones such as Perplexity and Gradient, as shown in Figure 1, the complex ones perform quite well especially under medium and large budget situations. It would be more important to study the tipping point, where the performance gains plateau became flat. This is the place where further increases in computing resources yield diminishing returns. 
2. It is not surprising to see the tradeoff between performance and data size. The conclusions in this paper are largely empirical and may not generalize well to other situations. The practical limit of parametric fit is limited, as it mainly fits observed data points without clear guidance on how to *extrapolate* results to new scenarios. For example, can the results from smaller models (e.g., 13B) be used to predict outcomes for larger models (e.g., 70B)? Can the parameters estimated from smaller models be reliably transferred for larger model? If practitioners need to run experiments on 70B models to obtain these insights and fit the parametric model, the results may not be useful.

### Questions
1. Could the author discuss a real-world scenario to demonstrate how the proposed methods could be applied to guide practitioners?
2. Are the studied methods sensitive to the choice of model architecture?
3. How do these methods scale with hardware improvements?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Surveys and experimentally compares different data selection methods for LLM fine-tuning, and reasonably and quantitatively concludes that only rather cheap methods that choose train samples based on some cheap similarity to the validation samples are likely to be worthwhile, but depends (of course) on how much training computation you are going to run.

### Strengths
I found this is an important experimental contribution for practitioners and academics alike, and is likely to be heavily cited in the future. While there will inevitably be some discussion of whether they compared to all the right and best methods, I think that's in the details: they compared good and sufficiently recent example methods from high level strategies and showed significant enough differences that seem endemic to these different strategies.

### Weaknesses
The weaknesses I detail below should all be corrected, but they are all minor, none of them individually or in total would be a good reason to reject the paper. 

SECTION 3 PROBLEMS:

At the beginning of Section 3: “Our goal is to find the optimal subset S ⊆ X” pretty sure you mean subset S ⊆ D there?

I think you are implying that the train set is not necessarily IID with the validation set, but that the validation set is IID with the test set. All I see you say is that the validation set is “correlated” with the test set, which is a really  weak and vague thing to say, but if that’s all you want to say, okay, but I will be curious if in your experiments you actually make the Val and Test sets IID. 

You need to define that $T$ represents a training procedure, you just use it without defining it now. 

“By ranking the data points….” Given a large initial train set D, having to rank the datapoints at cost O(D log D) is not free, hope you guys are taking that into account. Of course, you might argue just touching all D samples is O(D), but that is less relevant if, say, we have an infinite generator of data D (e.g. a real-time reader of the datastream formerly known as Twitter) and an independent (non-ranking) decider of whether each incoming $x$ is worth training on, that is, we shouldn’t have to assume we need to sort at cost O(D log D).


I’m uncomfortable as a reader that in (2) you are still defining your objective in terms of the test set. I agree that’s the ultimate goal, but if you actually implemented (2) it assumes knowledge of the test set. By the time you get to (2), I expected you to have switched to the validation set in the stated objective, which is different than the final metric, which should of course than be on the test set. 


SECTION 4 FEEDBACK: 
You can cut some of the intro to Sec 4, but please add-in that Lexicon-based and Embedding-based are both strategies that try to select train samples that are similar to the validation samples, whereas Perplexity and Gradient solutions are optimizing for the effect on the model loss.

SECTION 5 FEEDBACK:
Why do you assume training on all x is equal? Is that really true (honest question)? My guess is yes due to the very beaurocratic nature of how these models are run, but that’s not always true of machine-learned models, for example, a classic decision tree is much faster to evaluate for some inputs than others (if it has leaves of varying depths). 

In computing C(k), you sum over C_v(x), which I assume is for x \in D? Please be explicit there about which x you are summing over.  And I’m surprised that that cost does depend on $x$ Does C_v(x) really vary so much? Could that not just be \|D\| (= size of D) times some cost per training sample?  


RANDOM: Really appreciate you comparing to just a random sample as a baseline. 

MINOR BUT IMPORTANT QUIBBLES: 
Authors state too unequivocally: “in practice, the total compute budget is
predetermined: the number of accelerators and their usage hours are allocated in advanced”.   That certainly is NOT true in many large companies that are actively training and leading with LLMs. So please hedge and preface that sentence with “In many cases,”. 


This sentence didn’t make sense to me:
“For example work on parameter efficient fine-tuning, targets improving the
memory-usage of this stage (Hu et al., 2021).”

TYPO “create an minimal” -> “a minimal”

Since you are citing John 1975, consider  also citing the famous and foundational Hart 1968 paper on Condensed Nearest Neighbors, the canonical and classic approach for selecting data for training (summarized e.g. in wikipedia: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm). However, nearest neighbors is such a different machine learning paradigm that if you don’t feel that adds value to this work, you can ignore this suggestion, but given the statement in your paper is “Data
selection is a foundational approach in machine learning where the objective is to create an minimal
dataset from a collection of data” the Hart 1968 paper is exactly the classic paper to have done just that.

TYPO “Data selection takes the full training data as input and
chooses a subset to the train” -> “to train”

This sentence doesn’t quite parse, please re-write “Instruction-tuned models can handle a variety of possible inputs for downstream use cases as
either classification or generative model”

This sentence needs some polishing of singular vs plural:
“Therefore
while instruction-tuning is not the direct focus of this work, it provide a real-world applications of
compute-constrained data selection.”


“Assuming we at minimal” -> “at minimum”

Equation (2) can be written all on one line, winning you a bit of precious space. 

In 4.1 you refer to Section 4.1, which is a bit weird and really you can just delete that whole sentence. 

Do figure out how to bold the last column title C_forward in Table 1. It can be done (probably \mathbf{}). 

TYPO: Fit of Compute-Performace Relationship  -> Performance

### Questions
The value of the strategies that depend on similarity of samples to validation samples worry me in that they seem very dependent on the size of the validation set, and that if the validation set is too small one might overfit.  But perhaps it doesn't matter too much since you are always selecting some large number of training samples anyways, and so even if the validation set is as small as Paris (to use a 2D example), you still correctly pick the subset of training samples in Europe and dump the ones in the Americas, and that wouldn't have changed much if the validation set was all of France instead of just Paris. Be great to see some discussion & experiments about this, even if they are tiny, in this paper. 

See also the weaknesses section.

### Soundness
4

### Presentation
4

### Contribution
4
