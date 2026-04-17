# Review

## Summary
This paper proposes a differentially private dataset distillation algorithm that adapts D3S to work with public pre-trained models while privatizing intermediate activation statistics. This paper also develops multistage clipping and grouped pseudo-classes techniques that improve performance in high-privacy regimes.

## Soundness
2

## Presentation
3

## Contribution
3

## Strengths
- This paper is well-written and easy to follow.
- This paper provides a detailed discussion of the background and related work.
- The experimental results are comprehensive and support the main claims of the paper.

## Weaknesses
- The proposed method is not entirely private, as it relies on a public pre-trained model. Although the authors claim that the public pre-trained model can be obtained from a third party, this approach raises several concerns: 
  - How can a third party obtain a high-quality public pre-trained model without violating differential privacy? 
  - Even if a suitable public pre-trained model can be found, the authors do not discuss the potential privacy risks associated with using it. 
  - Furthermore, the proposed method requires the public pre-trained model to assign pseudo labels, which may leak privacy information. The authors do not address these privacy risks.

- The proposed method is computationally expensive, which limits its practicality. The authors should provide a detailed analysis of the computational complexity of their method and compare it with existing approaches.

- The proposed method is limited to image datasets, which restricts its applicability. The authors should discuss the challenges of extending their method to other types of datasets.

## Questions
- How can a third party obtain a high-quality public pre-trained model without violating differential privacy? 
- Even if a suitable public pre-trained model can be found, the authors do not discuss the potential privacy risks associated with using it. 
- Furthermore, the proposed method requires the public pre-trained model to assign pseudo labels, which may leak privacy information. The authors do not address these privacy risks.
- The authors should provide a detailed analysis of the computational complexity of their method and compare it with existing approaches.
- The authors should discuss the challenges of extending their method to other types of datasets.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4