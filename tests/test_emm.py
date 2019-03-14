import exp_mixture_model as emm
import unittest


class BasicTest(unittest.TestCase):
    def setUp(self):
        self.x = emm.generate_emm(1000, 4, pi=[0.8, 0.15, 0.04, 0.01], mu=[5, 50, 500, 5000])

    def test_generate_emm(self):
        self.assertEqual(len(self.x), 1000)

    def test_EMM_fit(self):
        k = 5
        model = emm.EMM(k)
        with self.assertRaises(AttributeError):
            model.aic()
        pi, mu = model.fit(self.x)
        self.assertEqual(model.n, len(self.x))
        self.assertLessEqual(model.k_final, k)
        self.assertEqual(model.k_final, len(pi))
        self.assertEqual(model.k_final, len(mu))
        for j in range(model.k_final-1):
            self.assertLessEqual(mu[j], mu[j+1])

    def test_EMM_fit_contains_zero(self):
        x_contains_zero = self.x.copy()
        x_contains_zero[0] = 0

        model = emm.EMM()
        model.fit(x_contains_zero)
        self.assertEqual(model.n, len(x_contains_zero) - 1)

    def test_EMMs_fit(self):
        k_candidates = [1, 2, 3]
        models = emm.EMMs(k_candidates)
        self.assertEqual(len(k_candidates), len(models.model_candidates))
        with self.assertRaises(AttributeError):
            models.select("AIC")
        models.fit(self.x)
        with self.assertRaises(ValueError):
            models.select("BAD_CRITERION")
        self.assertEqual(models.n, len(self.x))
        self.assertEqual(len(k_candidates), len(models.result_table))
        self.assertListEqual(models.calculated_column, ["k_final", "marginal_log_likelihood",
                                                        "joint_log_likelihood"])

        best_model = models.select("DNML")
        self.assertListEqual(models.calculated_column, ["k_final", "marginal_log_likelihood",
                                                        "joint_log_likelihood", "DNML"])
        print(best_model.dnml())
        print(models.result_table["DNML"])
        for dnml in models.result_table["DNML"]:
            self.assertLessEqual(best_model.dnml(), dnml)


if __name__ == "__main__":
    unittest.main()
