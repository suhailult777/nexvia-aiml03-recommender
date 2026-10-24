# AIML-03: 10 minute talk

Two-stage recommendation on MovieLens 100K. Numbers below come from
`results/metrics.json`, `results/latency.json` and the test run, not from slides
math.

## 0:00 to 1:00, the problem

Netflix and Amazon do not score a million items per request. They retrieve a
few hundred candidates cheaply, then spend the expensive model on those.

This project builds that shape end to end on MovieLens 100K: 100,000 ratings,
943 users, 1,682 movies. Stage one is a two-tower model with FAISS. Stage two is
a wide and deep ranker. It is served by FastAPI on port 8011.

Say the constraint up front: one Windows laptop, CPU only, no Spark, no GPU, no
Triton, no Redis, no Airflow. Every substitution is named in the README.

## 1:00 to 2:00, data and split

Ratings from September 1997 to April 1998. Every rating is an implicit positive
interaction.

The split is per user by time. Sort each user's rows by timestamp, take the last
20 percent as test. That gives exactly 80,000 train and 20,000 test rows.

Two numbers to show the honesty of it: the max train timestamp is 1998-04-22
23:10:38 UTC and the min test timestamp is 1997-09-20 20:18:15 UTC. Those
overlap, because the split is per user and not global. Within a user, test is
always later. The minimum training history is 16 interactions, which matters
later for cold start.

Features are built in pandas: user and item indices, 19 genre multi-hot,
popularity and its percentile, mean ratings, history length, and a 19 dim genre
profile per user. No Spark. 80,000 rows does not need a cluster.

## 2:00 to 3:30, stage one, two-tower

User tower: a 64 dim id embedding plus a dense block over log history length,
mean rating and genre profile, projected to 64 dims and L2 normalized.

Item tower: a 64 dim id embedding plus a dense block over genre multi-hot,
popularity, popularity percentile and rating stats, also normalized.

They are trained with a softmax over the full 1,682 item catalog, cosine logits
at temperature 0.1. That is sampled softmax with every item sampled, which is
affordable here and avoids the false negative problem of in-batch negatives.

25 epochs, batch 1,024, Adam at 0.01, 80,000 pairs per epoch, 73 seconds on CPU.
Loss goes from 6.65 to 5.96.

## 3:30 to 4:30, FAISS

The item matrix is 1,682 by 64 and unit norm. `IndexFlatIP` turns inner product
into cosine similarity. Exact search, no IVF, no HNSW, because at this catalog
size an approximate index only loses recall.

Retrieval plus seen-item masking plus the slice to 200 candidates: 0.653 ms mean,
1.219 ms p95 over 250 users.

The number that matters for stage two: those 200 candidates contain 59.85
percent of all test interactions. That is the ceiling the ranker can reach.

## 4:30 to 6:00, stage two, wide and deep

Input per pair: both frozen embeddings, their dot product, popularity, mean
ratings, rating counts, history length, genre cosine, rating gap, and two
bucketed crosses. Wide part is a linear term plus two cross embedding tables.
Deep part is 256 to 64 to 1 with dropout.

Training data is mined per user: 30 positives from train, 30 hard negatives from
the top retrieved items the user has not seen, 20 random negatives. 72,980 rows
in total. 12 epochs, 19 seconds, validation AUC 0.860.

It scores the 200 candidates and keeps 10. Ranking stage: 2.890 ms mean, 5.488
ms p95. Full pipeline 3.723 ms mean.

## 6:00 to 7:30, results

| System | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 |
|---|---|---|---|---|
| Popularity | 0.0587 | 0.1963 | 0.1128 | 0.1469 |
| Two-tower | 0.1089 | 0.3625 | 0.1787 | 0.2518 |
| Two-stage | 0.1111 | 0.3569 | 0.1837 | 0.2511 |

Two-tower versus popularity is the real win: plus 85.6 percent Recall@10,
plus 71.4 percent NDCG@50.

Two-stage versus two-tower is honest and small: plus 2.8 percent NDCG@10 at the
head, flat to slightly negative at 50. Say why in one sentence: no impression
log, so the ranker is trained on interaction positives and sampled negatives,
not on real shown-but-not-clicked data. A listwise objective was tested and
landed in the same noise band.

## 7:30 to 8:30, cold start

Rule: fewer than 3 training interactions, or an unknown user id, goes to
`0.6 * popularity + 0.4 * genre affinity`.

How many real users hit it: zero of 943. The minimum history is 16. That is the
honest answer, and it is more useful than claiming the branch was exercised.

To measure it anyway, 60 users get a 2 item seed profile and are scored on their
real test items. Recall@10 is 0.0368 versus 0.0339 for popularity alone.
Recall@50 is 0.1420 versus 0.1252. NDCG@10 is flat. Small, but it is a real
measurement of the real code path.

## 8:30 to 9:15, serving

FastAPI on port 8011, two endpoints. `/health` returns 1,682 FAISS vectors and a
cold-start threshold of 3. `/recommend?user_id=1&k=10` returns titles, genres,
scores and latency.

Live response for user 1: Star Trek First Contact at 0.845, Fargo at 0.824,
Contact at 0.758. All unseen items. First call 54.6 ms because of lazy
allocation, warm HTTP median 3.84 ms.

k is validated 1 to 50; k=51 returns 422.

## 9:15 to 10:00, what is not done

No click-through, because there is no impression log. No distributed anything.
Cold start is simulated. The ranker gain is small and only at the head. 100,000
interactions is a teaching scale.

Next step that would matter most: impression data, then a session based user
tower so a user with two clicks gets a real vector instead of a popularity list.
