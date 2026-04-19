# Improving protein optimization with smoothed fitness landscapes

- Decision: Accept (poster)
- Scores: 6, 6, 6, 3

## Abstract
The ability to engineer novel proteins with higher fitness for a desired property would be revolutionary for biotechnology and medicine. Modeling the combinatorially large space of sequences is infeasible; prior methods often constrain optimization to a small mutational radius, but this drastically limits the design space. Instead of heuristics, we propose smoothing the fitness landscape to facilitate protein optimization. First, we formulate protein fitness as a graph signal then use Tikunov regularization to smooth the fitness landscape. We find optimizing in this smoothed landscape leads to improved performance across multiple methods in the GFP and AAV benchmarks. Second, we achieve state-of-the-art results utilizing discrete energy-based models and MCMC in the smoothed landscape. Our method, called Gibbs sampling with Graph-based Smoothing (GGS), demonstrates a unique ability to achieve 2.5 fold fitness improvement (with in-silico evaluation) over its training set. GGS demonstrates potential to optimize proteins in the limited data regime. Code: https://github.com/kirjner/GGS

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The manuscript proposes a set of techniques for protein optimization. The first is a method for smoothing protein fitness landscapes. The second is a technique to optimizing in this landscape using the Gibbs With Gradients procedure, which has previously been shown to provide excellent results for discrete optimization. The authors also design two new optimization tasks based on the GFP and AAV datasets, which are designed to be more difficult than previous variants. Finally, the authors demonstrate empirically that their method performs competitively with the state-of-the-art.

### Strengths
### Originality
Although the GWG optimization procedure has been used in other contexts, the application to protein optimization is novel. To my knowledge, also the specific graph-based formulation of the regression problem itself is new.

### Quality
The paper seems technically sound. Code was provided to ensure reproducibility, and the authors provide additional details about the method in the supporting material. 

### Significance
The paper does not give much insight into why the method outperform earlier approaches (see below for details), but the empirical results are convincing, which by itself could be sufficient to have impact on the growing subcommunity in ICML interested in protein modelling and design.

### Weaknesses
My main concern with the paper is that I - after reading it - do not feel much wiser about promising methodogical directions for protein modelling going forward.
What I lack in the paper is perhaps more of a motivation of why particular modelling choices were made. For instance, why is the Tikhunov regularization a meaningful choice in the context of protein optimization? Intuitively to me, it seems like a fairly crude choice, ignoring much of what we know about proteins already (e.g. that certain amino acids are biochemically similar to others). The paper also provides no biological intuition about why we would expect the smoothness would help. Presumably, the idea must be that there are different length scales to the problem, and that we can ignore the short length scales and focus on the longer ones - but it is not obvious to be why that would be the case for proteins. Is part of the explanation that experimental data is typically quite noisy? But if that's the case, you would assume that you would get similar behavior by using a simple GP with observation noise - just using a kernel based on edit distance - or based on Eucledian distance in one-hot space. The paper would be much more satisfying for me if the smoothing procedure was motivated more clearly, and perhaps even validated independent of the optimization procedure (I assume you would hope that the smoothed regressor would extrapolate better?)

My other serious concern is about the empirical evaluation of the model. As far as I can see, when we evaluate an optimization model against an oracle, there is a risk that we end up optimizing against extrapolation artefacts of the oracle, in particular if we end up evaluating it far away from the data it was trained on. My concern is whether your method has an unfair advantage compared to the baselines, because it uses the same CNN architecture for both the model and the oracle - and could therefore be particularly well suited for exploiting these artefacts. To rule out this concern, it would be interesting to see how the model performs against an oracle trained using a completely different model architecture.

### Questions
Page 4,
*"Edges, E, are constructed with a k-nearest neighbor graph around each node based on the Levenshtein distance 3."*
In real-world cases, the starting point is often a saturation mutagenesis experiment, where a lot of candidates will be generated with the same edit distance from the wild type (e.g. edit distance 1). In such cases, won’t the fixed k-out degree lead to an arbitrary graph structure (I mean, if the actual number of equidistant neighbors is much larger than k)?

Page 6, *"4.1 Benchmark"*
It was difficult to follow exactly what "develop a set of tasks" implies. Since the benchmarks are built from existing datasets, the authors should make it clearer exactly what they are "developing": is it only the starting set, or do they also restrict themselves to a subsample of the entire set? In table 1 and 2, are both *Range*, *|D|*, and *Gap* specifically selected for, or does e.g. *|D|* arise out of a constraint on *Range* and *Gap*?

Page 6. *"Oracle"*
Since you are using a CNN both as your oracle, and as the basis for your smoothed landscape model, isn’t there a risk that your model is just particularly well suited for exploiting the extrapolation artifacts of the oracle? (repetition of concern stated above).

### Minor comments:
Page 1, *"but high quality structures are not available in many cases"*
After AlphaFold, many would consider that high quality structures are now available in most cases.

Page 2, *"mutation is proposed renewed gradient computations."*
Something is wrong in this sentence

Page 3, *"in-silico oracles provides a accessible way for evaluation and is done in all prior works."*
This is not entirely accurate. People have optimized against actual experiments (e.g. Gruver, ..., Gordon-Wilson, 2023) - or optimized to find the optimal candidate in a fixed set of experimentally characterized proteins.

Page 4, eq (2) *"H(x)"*
As far as I can see, H(x) has not been introduced(?)

Page 6. *"we utilize a simpler CNN that achieves superior performance in terms of Spearman correlation and fewer false positives."*
Was this correlation measured on the GFP test set provided by TAPE after fitting on the training set?. If so, it's odd that the original TAPE paper did not find the CNN-based ResNet to outperform the transformer (actually, the transformer performance was dramatically higher). Please clarify.

Page 6. *"Recall the protein optimization task is to use D"*
Perhaps help the reader by rephrasing to "Recall the protein optimization task is to use the starting set D"

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors introduces a method called Gibbs sampling with Graph-based Smoothing (GGS) that uses Tikunov regularization and graph signals to smooth the protein fitness landscape, improving the ability to create diverse, functional sequences.

### Strengths
Figure 1 is very helpful in the understanding of this approach.

My understanding of the section described in Section 3.2 is relatively clear.

I think the Fitness, Diversity, and Novelty scores to be interpretable and helpful.

I think it is encouraging that graph-based smoothing (GS) helps almost all other methods in Table 3. It’s also great that this is a relatively straightforward procedure.

### Weaknesses
“While dWJS is an alternative approach to fitness regularization, it was only demonstrated for antibody optimization. To the best of our knowledge, we are the first to apply discrete regularization using graph-based smoothing techniques for general protein optimization.” - This doesn’t seem justifiably novel. Proteins are proteins.

Generally, I wouldn’t use the term “fitness” when describing protein function. Rather, I would use phenotype or function, as fitness is a broad, poorly defined subset of fitness.

Figure 5 is a reason why these function predictors should not be called “oracles”, because mapping the effect of mutation to function is difficult itself. I’d prefer “protein function approximator”, or something along those lines.

“These were chosen due to their long lengths, 237 and 28 residues” What do you mean here? 28 isn’t that long. I realize it is in the context of a larger protein, but I’d be clear about that.

### Questions
For the smoothing procedure, it’d be great to show the amount of error introduced into the labels of the sequences. For instances where either a reasonable oracle model exists, or sequences with large hamming distances have been measured, and this smoothing procedure is introduced, what is the correlation of function values before and after?

“To control compute bandwidth, we perform hierarchical clustering (Mullner, 2011) on all the se- ¨ quences in a round and take the sequence of each cluster with the highest predicted fitness using fθ.” Why not use the “noisy model” for this, because it is the oracle for the true fitness of a sequence?

“Section 4.1 presents a set of challenging tasks based on the GFP and AAV proteins that emulate starting optimization with a noisy and limited set of proteins.” I would like the authors to be clear by what the mean by “noisy”. Is it experimental noise? Is the landscape too sparsely sampled? Where is this noise coming from, and what relative distribution does it have?

Generally, I feel like Figure 5 is a distraction from the broader utility of the work. I’d just cite Dallago 2021 like you did for the use of CNNs.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This study proposes to smooth the protein fitness landscape to facilitate protein fitness optimization using gradient based techniques. This is motivated by the ruggedness of protein fitness landscape which makes optimization challenging. A graph based smoothing technique for fitness landscape followed by Gibbs with Gradient sampling is used to perform protein fitness optimization. Evaluation of their method has been done on train sets designed from GFP and AAV with two degrees of difficulty defined by the mutational gap between the starting set and the optimum in the dataset (not included in the starting set).  Their method shows better performance than others in the proposed benchmark. The proposed graph smoothing technique has been shown to help with other methods as well.

### Strengths
Designing train sets with varying difficulties for the task of optimization.
Proposing a new method for smoothing the protein fitness landscape before optimization.

### Weaknesses
The proposed method has many hyperparameters to tune. 
Given certain properties of protein fitness landscape, smoothing can hurt if not done properly.

### Questions
1)	Please explain why after smoothing, the diversity and novelty of the final set of sequences decreases.
2)	In defining train sets with varying levels of difficulty only two medium (mutation gap 6) and hard (mutation gap 7) levels have been used. What happens if you make this harder (higher than 7)? Also, should we assume that for less mutational gap all methods perform comparably?
3)	As stated in the paper, single mutations can dramatically change the fitness. In the smoothing performed, similar sequences are enforced to have similar fitness. Have you investigated where smoothing can be detrimental?
4)	How is the number of proposals ($N_{\text{prop}}$) per sequence set?
5)	Have you tried smaller sizes for the starting set? In real world problems the size of the starting set could be much smaller than 2000? 
6)	Was the oracle only used at the end for performance evaluation? In AdaLead, did you use the oracle as the fitness landscape or $f_\theta$?
7)	Mention the augmented graph size (how does it change with the size of the sequence)
8)	Minor: In Eq 5, $X_0$ should be X.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper propose a smoothing method on fitness function given a protein sequence. Assume that the given original data set is small, authors proposed a sampling augmentation method and a TV smoothing regulariser. After which MCMC algorithm is use to further optimise the fit.

### Strengths
Authors presented some good results on benchmark datasets.

### Weaknesses
The paper is hard to read and understand. I itemise areas for improvements.

1. Having one figure to show overall flow of logic could help. Fig1 seems to do the job. There are some confusion between training and sampling. I understand that the author first train f(x) and then use f(x) as a surrogate function for MCMC optimisation. This point does not come out naturally.

2. construction of KNN graph could be described more clearly. (see Eq above Eq(1))

3. Symbols of Eq.(2) are ill defined. The authors should provide in the appendix some details of GWG and reference the appendix in the main text.

4. Eq.(4) should give the acceptance rate. while q are the probability of trial moves. x and x' are two states for jumping in this one MC step. Usual notation is q(x|x') vs q(x'|x), notation of Eq.(4) certainly is not of this form. Instead i^loc and j^sub and being used. The same i^loc and j^sub cannot appear in both numerator and denominator of Eq.(4).

5. Eq.(4) what is the temperature of this move? It seems the temperature is set to 1. Why is the temperature 1? Is there any annealing process?

6. Clustered sampling section should be explained better.

### Questions
Is there a way to test that the surrogate function by itself is good enough? The authors look at the overall performance that could infer to correctness of the surrogate function.


see above section on 'weakness'

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair
