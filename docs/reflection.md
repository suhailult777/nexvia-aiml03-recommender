# Reflection

I built this pipeline end to end on a Windows laptop with no GPU and no
distributed anything, which shaped almost every decision in it.

## What I actually owned

The data split, the features, both models, the index, the evaluation, the
service, the tests and the docs. I wrote the split myself rather than using a
library helper because the "last 20 percent per user" rule is easy to get wrong
in a way that leaks the future into training. Sorting by `(user_id, timestamp)`
and cutting on a per-row rank made the rule explicit and gave a number I could
put in the README: 80,000 and 20,000 rows, with the exact global timestamp
boundaries written down, including the awkward fact that those boundaries
overlap across users.

The two-tower was the part I learned the most from. My first instinct was
in-batch negatives, which is the standard trick, but with 1,682 items a full
softmax over the catalog costs almost nothing and removes the false negative
problem entirely: if two users in the same batch watched the same movie, the
in-batch version punishes the model for retrieving it. Switching to full softmax
also made the loss curve easier to read, because the baseline is now
`ln(1682) = 7.43` instead of a number that depends on the batch composition.
Training ended at 5.96, which is not a great model, but it is a real number and
it beats popularity by 85 percent on Recall@10.

The ranker was the part that humbled me. I expected the second stage to add
several points. It added 2.8 percent relative at NDCG@10 and nothing at 50. I
tried a listwise softmax objective, rebuilt the training groups around it, and
got 0.1835 NDCG@10 against 0.1837 for the pointwise model. I tried blending the
ranker score with the retrieval cosine at three weights, and the best blend was
also inside the same noise band. The reason is not mysterious: without
impression logs, my "negatives" are items the user did not interact with in a
sparse matrix, which is not the same as items the user was shown and skipped.
The ranker mostly relearns the retrieval score plus a popularity prior. I could
have hidden that by reporting only NDCG@10, or by tuning on the test set until
something looked good. I wrote the real table instead, including the column
where the two-stage loses.

## What I would do differently

I would build the ranker's supervision from a nested temporal holdout inside
train from the start. My first version trained the ranker on train positives
while the retriever had already memorized those same positives, so the ranker
was partly learning "this item is in my training set" rather than "this item is
a good next recommendation". I only understood that after looking at why the
gain was so small.

I would also write the evaluation before the models. I ended up rewriting
`evaluate.py` when I realised the two-stage comparison was unfair unless I
over-fetched candidates before masking seen items. Retrieval with `k = 200` and
then masking leaves far fewer than 200 candidates for an active user with 85
seen items, and the ranker would have looked worse for a reason that had nothing
to do with the ranker.

## What I did not do

I did not use Spark, and I am not going to pretend pandas is Spark. 80,000 rows
fit in memory many times over. The README says this plainly.

I did not measure click-through, because MovieLens has no impressions. Recall
and NDCG are what this dataset supports.

I did not get a real cold-start population. The minimum history is 16
interactions, so zero of 943 users hit the fallback. I simulated it with 60
users and a 2 item seed profile and reported those numbers separately, labelled
as a simulation, because a table that implies real cold traffic would be a lie.

The cold-start fallback itself is simple: 0.6 popularity plus 0.4 genre
affinity. It beat popularity alone on Recall@10 and Recall@50 and tied it on
NDCG@10. That is a modest result and it is what the numbers say.

## What this project taught me

Two-stage retrieval is mostly a data problem. The candidate recall ceiling,
59.85 percent, was more informative than any model tweak: no ranker can recover
a relevant item that stage one never returned. If I had another week, I would
spend it on the retriever and on impression data, not on a bigger ranker.

The other lesson is about reporting. The most useful sentences in the README are
the ones about what did not work: the ranker being flat at 50, zero real cold
users, 71 items with no training interactions, the per-user split overlapping in
global time. Those are the parts a reviewer can trust, because they are the parts
that would have been easy to leave out.
