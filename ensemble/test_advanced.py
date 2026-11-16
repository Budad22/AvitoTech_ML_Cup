import unittest
import numpy as np
import pandas as pd
import pyarrow as pa
from catboost import CatBoostRanker, Pool
from advanced import add_correction, gated_scores, normalize, ranking_metrics


class AdvancedContracts(unittest.TestCase):
    def test_gates_are_per_feed_and_preserve_parent_endpoints(self):
        counts = np.array([3, 2])
        cat = np.array([3., 1., 2., 5., 4.], dtype=np.float32)
        mlp = -cat
        a, b = [normalize(x, counts, 'zscore') for x in [cat, mlp]]
        scores = gated_scores(cat, mlp, a, b, counts, np.array([True, False]), [1., 0.])
        np.testing.assert_array_equal(scores, [3., 1., 2., -5., -4.])
        perm = [2, 0, 1, 4, 3]
        reversed_scores = gated_scores(cat[perm], mlp[perm], a[perm], b[perm], counts,
                                      np.array([True, False]), [1., 0.])
        np.testing.assert_array_equal(scores[perm], reversed_scores)

    def test_native_baseline_is_added_exactly_once(self):
        rng = np.random.default_rng(42)
        x = pd.DataFrame({'x': rng.normal(size=80)})
        groups = np.repeat(np.arange(20), 4)
        target = (x.x.to_numpy() > 0).astype(int)
        baseline = rng.normal(size=80)
        pool = Pool(x, label=target, group_id=groups, baseline=baseline)
        model = CatBoostRanker(iterations=5, depth=2, loss_function='YetiRank',
                               eval_metric='NDCG:type=Base', thread_count=2,
                               allow_writing_files=False, random_seed=42)
        model.fit(pool, eval_set=pool, verbose=False)
        # CatBoost stores baseline as float32, so use the same representation at prediction.
        effective_baseline = baseline.astype(np.float32).astype(np.float64)
        correction = model.predict(x)
        combined = effective_baseline + correction
        np.testing.assert_allclose(model.predict(pool), combined, atol=1e-12, rtol=0)
        metric = ranking_metrics(groups, target, combined)['ndcg']
        native = model.get_evals_result()['validation']['NDCG:type=Base'][model.get_best_iteration()]
        self.assertAlmostEqual(metric, native, places=12)
        np.testing.assert_array_equal(add_correction(effective_baseline, correction), combined.astype(np.float32))


if __name__ == '__main__':
    unittest.main()
