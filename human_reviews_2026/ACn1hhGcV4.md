# Context-free Recognition with Transformers

- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Transformers excel on tasks that process well-formed inputs according to some grammar, such as natural language and code. However, it remains unclear how they can process grammatical syntax internally. In fact, under standard complexity conjectures, standard transformers cannot recognize context-free languages (CFLs), a canonical formalism to describe syntax, or even regular languages, a subclass of CFLs. Merrill & Sabharwal (2025) show that $\mathcal{O}(log(n))$ looping layers (w.r.t. input length $n$) allows transformers to recognize regular languages, but the question of context-free recognition remained open. In this work, we show that looped transformers with $\mathcal{O}(log(n))$ looping layers and $\mathcal{O}(n^6)$ padding tokens can recognize all CFLs. However, training and inference with $\mathcal{O}(n^6)$ padding tokens is potentially impractical. Fortunately, we show that, for natural subclasses such as unambiguous CFLs, the recognition problem on transformers becomes more tractable, requiring $\mathcal{O}(n^3)$ padding.
We empirically validate our results and show that looping helps on languages that require logarithmic depth.
Overall, our results shed light on the intricacy of CFL recognition by transformers: while general recognition may require an intractable amount of padding, natural constraints such as unambiguity yield efficient recognition algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a theoretical analysis of the ability of transformers to recognize Context-Free Languages (CFLs). The main contribution is a constructive proof that transformers augmented with $O(\log n)$ looping layers and $O(n^6)$ padding tokens can recognize all CFLs. Furthermore, the paper shows that for subclasses of CFLs—namely unambiguous and unambiguous linear CFLs—the resource requirements, particularly the number of padding tokens, can be significantly reduced to $O(n^3)$ and $O(n^2)$ respectively. An empirical evaluation is included to validate the utility of looping on languages of varying complexity.

### Strengths
1. Theoretical Soundness: The core theoretical result—CFL recognition with log-depth looping—is a non-trivial and valuable contribution to the understanding of transformer expressivity. It directly addresses an open question in the field.

2. Nuanced Analysis: Moving beyond the general case to analyze subclasses like unambiguous CFLs is a strength. It provides a more refined perspective that aligns with practical observations (e.g., transformers struggling with ambiguity) and connects theory with relevant linguistic properties.

### Weaknesses
Empirical experiments do not provide a strong support for the theoretical analysis. In Table 2, the differences between Fixed-depth and $\log (n)$ looping in BFVP and BFVP ((postfix)) are small. Moreover, even if there are log-depth transformer constructions for D(2) and Marked Palindrome, $log (n)$ looping is still stronger than Fixed-depth. The above two empirical results seem to show that the theoretical analysis in the paper has little impacts on practical applications.

### Questions
Why is Fixed-depth more accurate than log(n) looping in Palindrome and D(1)? Could you provide a possible explanation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper shows that context-free languages are recognizable by O(log n)-looped AHAT transformers with O(n^6)-padded tokens (this means that apart from a constant number of initial and terminal layer, we have O(log n)-time repetition of a constant block of layers). There are also improvements of the number of padded tokens for subclasses of context-free languages.

The context is that O(log n)-looped AHAT transformers coincide with the class of logspace-uniform O(log n)-depth threshold circuits (Merril, Sabharwal, 2025), and this class is subsumed by NC^2. Due to results of Ruzzo, CFL are in NC^2.

### Strengths
The circuit and computational complexity of CFL is an interesting topic, and transformers recently have established useful connections with circuit complexity. It is now known whether CFL is in NL, it seems unlikely that they are in NC^1.

### Weaknesses
The proof of the main result is a bit sloppy, and I could not understand the details (and could not restore them myself). The idea in short is that there are some guessing rules that allow to check in O(log n) recursive steps whether a given non-terminal derives a word of length n. More precisely, there are polyhnomially many ``derivation statements'' (``items'' and ``slashed items''). Each derivation statement is either an easily checkable base case or can be derived from 2 other derivation statements. Corollary 3.1 is the main non-trivial part that supposed to show that for every statement there exists an O(log n)-depth derivation. Then it is easy to see that a looped transformer can compute all derivable statements if it has enough (poly n) tokens to store information for all statements and try all possible 1-step derivations for O(log n) iterations.

The part after Corollary 3.1 is written in some detail in appendix, but it seems straighfordard anyway. The non-trivial part (Corollary 3.1) is written in the hand-waiving manner,  talking about some balanced decompositions of some trees that have not been defined, and it is not clear where all the details of the recursive cases in Lemmas 3.1 and 3.2 play the role (with the proof of Lemma 3.2 omitted).

It is not clear how much the non-transformer part is original in the paper. I asked the authors to clarify this. My current guess is that Ruzzo probably should have been doing something similar to Corollary 3.1 to obtain that CFL are in NC^2, and probably the current paper replaces the arugment that this guessing system can be simulated in NC^2 by the argument that it can be simulated in a looped AHAT.

So far, due the lack of clarity and non-evident originality, I vote for weak reject.

### Questions
Is some analoge of Corollary 3.1 has been established in the literature (by Ruzzo or somebody else?) If not, what new ingredients the current proof has to obtain that CFL are in log(n)-looped AHAT?

### Soundness
2

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
4

### Summary
This paper investigates the context-free language (CFL) recognition capabilities of Transformer models. The authors present a model of Transformers with looped layers and padding tokens. They give an upper bound on the number of padding tokens ($O(n^6)$) and the number of looped layers ($O(\log n)$) required. They discuss how for simplified versions of CFL languages such as unambiguous CFLs (with one derivation tree per valid word), one can obtain models with smaller size/less padding tokens.

The authors validate their claims empirically by training transformer models to recognize different CFLs such as Palindromic languages, Dyck languages and Boolean formula value problems. Their empirical results show that transformers can indeed learn to recognize context free languages with a high degree of accuracy. Moreover, their analysis shows that using looped layers can increase accuracy.

I have given this paper a 6, but if the authors include an Experimental Details section in the appendix, I will raise my score to an 8.

### Strengths
* Paper is very well written. I found the theoretical exposition to ber quite clear. The approach taken by the authors to construct the algorithms and conduct their analysis before discussing relation/implementation with Transformers was very clear and pleasant to read.
* The theoretical results are novel and interesting. As the authors have also stated, this work is the first theoretical contribution (to the best of my knowledge as well) which proves CFL recognition with transformers.
* The theoretical approach is non-trivial and a significant contribution to current research on formal languages and deep learning.
	* The first proof introduces a novel approach (different than CKY) to find a valid derivation tree for a candidate word.
	* The application of path system results to Transformer expressivity (For the proof in the UCFLs case) is also (to the best of my knowledge) a novel contribution
* The analysis of CFLs with different levels of constraints also adds depth to the paper. I find the analysis of the tradeoffs here to be very interesting.
* Assumptions about the model are clear.
* Proofs seem correct and are written in a style which is easy to follow.

### Weaknesses
- The biggest weakness for me is experiments section. I feel alot of methodological details are missing from this section. See the the "Questions" section below. I feel an appendix section with more experimental details could be beneficial to readers who want to better understand/reproduce similar setups.
- I find the model of Transformers presented by the authors to be unrealistic and quite detached from practice. I have no problem with the use of hard attention however: 
	- Padding tokens are unrealistic; no one really uses this in practice to augment model performance
	- Looping layers are also (to the best of my knowledge) not standard practice
	- Could the authors explain how such concepts can be tied to methods of practical relevance? (Maybe here they could also put emphasis on how their approach could inform future model design instead?)
- Although the proofs are easy to read. My only minor complaint is that they are very text heavy. For instance, I feel the discussion of embeddings used in the proof of Theorem 3.3 could have been clearer if simply presented in vector notation.

### Questions
**Theoretical Results**
* What does guessing mean in Algos 1 and 2? sampling uniformly at random?
* What is the "width" of the transformer needed for Theorems 3.3 and 4.1? Reading the proofs it seems like they would be constant w.r.t. the input length. Could this information be added somewhere? For instance in the Theorem definitions given in the appendix.
* "These results imply that, in order to recognize CFLs, transformers require significantly less depth than that which would be needed to implement a serial parsing algorithm like CKY" Can you comment on the order of magnitude (as a function of $n$) needed for a model to implement CKY?

**Experimental Results**
* How many samples were used for training?
* How were the samples generated? Were the models trained on a fixed dataset for some number of epochs or were batches sampled at training time?
* What is the maximum length of strings models are trained on?
* What are the  lengths of strings in the OOD case?
* How are looping layers implemented in practice?
* Do the authors have a hypothesis as to why the difference between using and not using looped layers is so small? I would have expected a bigger margin here given the theoretical results.

**General Questions/Remarks**
* Can the authors comment on how augmenting a Transformer model with a CoT would change the size needed? Do the authors have a sense of how many CoT steps would be needed to recognize a word in a CFL?
* Do the authors see any connections between looped layers and continuous CoT frameworks?
* You cite the Zhao et al. paper "Do Transformers parse while predicting the masked word?", but do not discuss the theoretical results of this paper nor its implication of this to your own theoretical results. Would it be possible to comment on how the approach/results in your work differ/contrast to those in the work of Zhao et al.
* It could be interesting to have interpretability results as well. I am curious to see what kind of solutions models learn through gradient-based training.

**Minor Comments and Typos**
* Is there in error in the production rule definition (3) at line ~092? We would need $\alpha \in (\mathcal{N} \cup \Sigma)^*$.
* line ~162: "exists a rule $X \to YZ$ and index $k$ s.t. $[Y, i, k-1]$ and $[X,k,j]$ are realizable" shouldn't it be $Z$ instead of $X$ here?
*  Line 447 "Our transformers has 1.2 million parameter budget" typo here

### Soundness
3

### Presentation
4

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
This paper studies when looped transformers with padding tokens can recognize CFLs. The authors prove that looped transformers with $O(\log n)$ looping layers and $O(n^6)$ padding tokens are complete for all CFLs. for natural subclasses, the required padding can be reduced to $O(n^3).

### Strengths
The paper establishes a strong theoretical connection between CFL recognition and model size, which may provide useful insights for designing more efficient models.

### Weaknesses
1. The analysis focuses solely on model expressiveness and does not address learnability.

2. Although $O(n^3)$ padding is an improvement over $O(n^6)$, it may still be impractical in real-world settings.

3. The experimental section does not appear to relate clearly to the theoretical results, particularly concerning the number of padding tokens.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
