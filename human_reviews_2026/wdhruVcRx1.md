# Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure

- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
Large Language Models (LLMs) are increasingly employed in real-world applications, driving the need to evaluate the trustworthiness of their generated text. To this end, reliable uncertainty estimation is essential. Leading uncertainty estimation methods generate and analyze multiple output sequences, which is computationally expensive and impractical at scale. In this work, we inspect the theoretical foundations of these methods and explore new directions to enhance computational efficiency. Building on the framework of proper scoring rules, we find that the negative log-likelihood of the most likely output sequence constitutes a theoretically principled uncertainty measure. To approximate this alternative measure, we propose G-NLL, obtained using a single output sequence from greedy decoding. This approach streamlines uncertainty estimation while preserving theoretical rigor. Empirical results demonstrate that G-NLL achieves state-of-the-art performance across various scenarios. Our work lays the theoretical foundation for efficient and reliable uncertainty estimation in natural language generation, challenging the necessity of the prevalent methods that are more complex and resource-intensive.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a new uncertainty quantification scheme for LLMs that leverages the probability of the most likely sequence generated for the input prompt. The proposed method is formulated in terms of a proper scoring rule, specifically a zero-one score. This kind of framing comes with certain properties. The experimental evaluation is restricted to QA benchmark datasets and 3 open models from the Llama and Falcon families. The results show indeed improved performance over the competitive approaches, some of them requiring multiple samples.

### Strengths
- Uncertainty quantification for LLM is currently a hot topic and therefore advances in this area are warranted.

- The proposed method is very efficient in the sense that it requires a single greedy decoding.

### Weaknesses
- I found the presentation of sections 2 and 3 not very clear. Perhaps an illustrative example would help to better convey the technical details. Also, it is not clear what is new here, as proper scoring rules have been presented elsewhere.

- The performance improvements appear to be just marginal over the competitive approaches e.g., 0.804 vs 0.824 in Table 3. Standard deviations appear to be missing from the table, and these need to be included to account for the noise. Also, for such mediocre performance bumps, statistical significance tests are mandatory.

- The proposed method assumes access to sequence probabilities and therefore it is not applicable to closed models such as GPT or Gemini. I think this limitation should be made clearer in the presentation.

### Questions
- I may have misunderstood, but Eq 10 looks like summing up the log probabilities of the tokens obtained during greedy decoding. If that's the case, then it looks like your proposed approach is a slight reformulation of previous methods that combine token probabilities in different ways.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes and justifies a G-NLL measure for uncertainty estimation in LLMs. The main contribution is showing the connection to an AU like term while computing the expected score for the zero-one scoring rule. As I understand, the measure itself has been explored before, but this work provides some justification. Experiments are conducted to show the merits of the simple measure over some baselines. I have reviewed a prior version of the paper and find that this revised version is extremely similar. In fact, I’m not sure about the edits that have been made in this version – they seem minor. Overall, I like aspects of this paper and think it is a useful contribution.

### Strengths
The main strength of the paper is its contribution of making a justification for the proposed uncertainty measure G-NLL. It does so by showing the measure to approximate the aleatoric term in the Bayesian decomposition (TU = AU + EU) for the expected score according to a specific scoring rule. Overall, I continue to appreciate the overall simplicity of the final solution.

### Weaknesses
I find the additional contribution to practice to be somewhat limited, as it seems that the measure or related measures have been explored in prior work. 

Also, I find the claims by the paper to be somewhat excessive – the measure is made out to be more useful than I believe it really is. See my comment about this later.

### Questions
Some comments/questions follow:

G-NLL is said to approximate MSP. Could the authors clarify this a bit further or point me to the place in the paper where they explain the relation?

I feel it would help to present a much broader view of UQ; the current scope is heavily biased towards a subset of literature. This is a suggestion that I made previously as well, and I am slightly frustrated that it seems to have been ignored. Please see some recent survey papers for ideas if needed.

It appears to be standard practice for many uncertainty measures in the literature to be used as scores for predicting whether the greedy sample is correct or not, using metrics like AUROC. Could the authors share their thoughts on why this is the standard mode of evaluation? I am curious to hear their perspective.

“Our work challenges the reliance on sampling-based methods and offers a principled way for efficient uncertainty estimation for LLMs.” I consider the first part of this statement to be a stretch. There is a much broader spectrum of sampling-based methods that were not explored. Moreover, only 3 datasets were considered for experiments, and they do not even scratch the surface of breadth of potential applications. Also, the proposed method only works for scoring the greedy sample and not an arbitrary sample, making uncertainty estimation itself somewhat limiting. In short, there are numerous limitations for the scope of the work. I recommend modulating this statement and discussing limitations adequately.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper contributes in two ways:  
1. Using the framework of proper scoring rules, it shows that MSP baseline is a theoretically grounded approximation of the empirical risk.   
2. More importantly, is that this approximation combined with greedy decoding might give better estimate than entropy-based scores, because the number of samples for entropy to be accurate is huge.   

The authors further conduct experiments and demonstrate that fairly that MSP baseline outperforms many other methods in QA and in certain conditions. They also point out that this baselines should not be overlooked from the evaluation, which sadly happened in previous papers.  

After several reviewing cycles, I believe now it is overall in a good shape and should be accepted.

### Strengths
1.	The authors show that MSP for greedy decoding can be shown as a relatively good approximation of the empirical risk.

### Weaknesses
1.	The method itself is already widely-used, though the theoretical justification was lacking before.

### Questions
I would like to discuss the transition to (4), could you please clarify.

### Soundness
4

### Presentation
3

### Contribution
3
