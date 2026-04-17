# Softmax Transformers are Turing-Complete

- Decision: Accept (Oral)
- Scores: 6, 2, 4, 10

## Abstract
Hard attention Chain-of-Thought (CoT) transformers are known to be Turing-complete. However, it is an open problem whether softmax attention Chain-of-Thought (CoT) transformers are Turing-complete. In this paper, we prove a stronger result that length-generalizable softmax CoT transformers are Turing-complete. 

More precisely, our Turing-completeness proof goes via the CoT extension of the Counting RASP (C-RASP), which correspond to softmax CoT transformers that admit length generalization. We prove Turing-completeness for CoT C-RASP with causal masking over a unary alphabet (more generally, for the letter-bounded languages). While we show that this is actually not Turing-complete for arbitrary languages, we prove that its extension with relative positional encoding is Turing-complete for arbitrary languages. We empirically validate our theoretical results by training transformers for various languages that require complex (non-linear) arithmetic reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proves, for the first time, that soft attention transformers with chain-of-thought (CoT) are Turing-complete. This is more realistic than past results, which assumed hard attention. The authors use C-RASP to demonstrate that soft attention transformers with CoT can simulate Minsky counter machines, which are known to be equivalent to Turing machines. They show that such transformers are Turing-complete for unary and letter-bounded languages, and that adding relative positional encodings (RPEs) extends this result to arbitrary languages. Empirical results on arithmetic reasoning tasks confirm the theoretical claims: transformers with RPEs achieve near-perfect accuracy and have good length generalization.

### Strengths
1. The paper resolves a major open question in the theoretical literature of transformer expressivity. This result brings us closer to understanding what CoT transformers can perform in practice.

2. The proofs and constructions are rigorous and technically sound. Moreover, the main result is proven using a new proof technique---using counter machines---which strengthens the theoretical contribution.

3. The paper includes experiments on arithmetic reasoning tasks, with results that align closely with the theory: unary tasks succeed without RPEs, while binary tasks require RPEs to generalize.

### Weaknesses
1. Contribution (ii) from the introduction claims to provide a guarantee on trainability; however, this seems somewhat misleading, as the paper does not present a formal analysis of learnability.

2. In the current form, the paper is quite technical and, in places, hard to follow unless the reader is very familiar with related work. I believe the paper would benefit from some more high-level intuitive explanations. Additionally, a summary figure outlining the main results and their corresponding sections would improve clarity.

3. I think the experimental setup would benefit from some more detail and polishing. For instance, there is no mention of the test sets.

### Questions
1. How would the results of Merrill and Sabharwal (2024) translate to this setup?

2. Do you think it would be possible to generalize the result to other types of positional encodings?

3. Can you comment on the val_2 accuracy only approaching, but not reaching 100%? What does this imply? Could you say that the models learn a very good approximation but maybe not actually learn the underlying mechanism? Did you perform any ablation studies to see if you can improve the reported results? I think some discussion about this should be included in the paper.


William Merrill and Ashish Sabharwal. 2024. The Expressive Power of Transformers with Chain of Thought.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proves the completeness of softmax Transformers for CoT C-RASP over a unary alphabet. The result also implies that softmax Transformers exhibit length generalization for this language. The authors further state that softmax Transformers are not Turing-complete for arbitrary languages but show that softmax Transformers with relative positional encoding are Turing-complete for arbitrary languages.

### Strengths
The paper consider an important theoretical question, i.e., whether Turing-completeness holds for more realistic models such as softmax Transformers.

### Weaknesses
1. The expression "Turing-completeness for some languages" is conceptually unclear. Turing-completeness has a strict formal definition. The results presented in the paper do not appear to fully establish the claim.

2. Softmax Transformers should, in principle, approximate hard-attention arbitrarily well. Since hard-attention Transformers are known to be Turing-complete, the paper’s results suggesting a gap between softmax Transformers and Turing-completeness raise questions. The paper seems to provide no clear explanation for this discrepancy.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies whether softmax-attention Chain-of-Thought (CoT) transformers are Turing-complete and answers this question positively. To prove this, they build upon previous work on Chain-of-Thought extensions of a declarative language called C-RASP. They first show that CoT C-RASPs with causal masking are not Turing complete over arbitrary languages, and therefore extend them to CoT C-RASPs with Relative Positional Encodings. Lastly, they support their theoretical results by training Softmax Attention Transformers on various arithmetic tasks and find that they learn these tasks very well.

### Strengths
The paper is able to answer the relevant question whether softmax-attention Chain-of-Thought (CoT) transformers are Turing-complete positively, thereby extending previous work that only showed Turing-completeness using hardmax-attention. Furthermore, it is insightful to see that CoT C-RASPs with causal masking are not generally Turing complete, but that they do become Turing complete when adding Relative Positional Encodings.

### Weaknesses
I think the paper could profit from additional clarity in the writing and in the explanations. I was for example slightly confused by phrases like 'to provide _a kind of_ guarantee of trainability' (in the contributions). Furthermore, I found the section on the 'Empirical Experiments' not very clearly written and would have been interested in seeing slightly more details and explanations on the tasks, the architectures used and the training. The corresponding Appendix D was very short and did not help me much. Lastly, the related work section was very short and I would have in particular been interested to see more discussion on the relation between these results and previous results using hardmax-attention and the relation between this paper and the results from Huang et al. (2025), which is cited very often throughout the paper.

### Questions
1. What do you mean by 'to provide _a kind of_ guarantee of trainability' (in the contributions)? Could you explain the usage of 'kind of' better here.
2. Could you discuss the usage of RPEs for providing positional information in Transformers in more detail? How often are these used and what have been previous empirical findings on them?
3. Many of your definitions and proofs seem to build upon results from Huang et al. (2025). Could you discuss the relation to this paper in more detail.
4. Could you describe the tasks, training architecture, and loss function used in the 'Empirical Experiments' in slightly more detail again?
5. Where can I see the results that you are referring to in this sentence:
> These results demonstrate that unary benefits from NoPE, whereas binary requires R for length generalization.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
2

### Summary
Previous work had shown that hardmax attention transformers are Turing-complete. (Hardmax attention usually means that activations are 1 for some token and 0 elsewhere.) This paper proves the results under the more realistic case when the softmax operation is used in the network architecture, rather than the hardmax. The result goes through different techniques than the prior literature on Turing-completeness of transformers.

The proof of Turing-completeness, at a high level, involves two steps:

- Show that a language CoT C-RASP can be implemented by a softmax transformer (this mainly seems to have been done in prior work by Huang et al).
- Show that CoT C-RASP is Turing-complete, under various conditions (if the language is letter-bounded or there are relative positional encodings).

### Strengths
- This paper takes a step towards proving the Turing-completeness of transformers under more realistic architecture definitions. Softmax (rather than hardmax) is differentiable, so this proof is the first one for a transformer architecture definition that is trainable via gradient methods.
- The proof of Turing completeness goes through different techniques than previous proofs for hardmax attention. I am not familiar enough with these techniques to know if they are standard in formal language theory, but in any case bringing them to the attention of the community focused on computability in LLMs may be useful.

### Weaknesses
- Through Section 2, many proofs of the results, definitions etc. seem to taken nearly directly from Huang et al (2025). This makes it hard  to read without first reading Huang et al. There is significant setup assumed from Huang et al, some of which is never described in the paper itself. Generally, it seems worth defining notation before it is used, even if the notation is standard in formal language theory. Notation was not defined for the definition of C-RASP, CoT C-RASP and in many other places.  If this takes too much space, it would be nice to include an appendix with relevant background so that the paper is more self-contained.
- The paper should do more to highlight to conceptual or technical challenges of moving from hardmax to softmax. What about this alternate proof strategy allowed for circumventing the need/convenience of assuming hard attention?

### Questions
N/A

### Soundness
4

### Presentation
2

### Contribution
4
