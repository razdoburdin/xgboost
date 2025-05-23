##############
Slicing Models
##############

Slice tree model
----------------

When ``booster`` is set to ``gbtree`` or ``dart``, XGBoost builds a tree model, which is a
list of trees and can be sliced into multiple sub-models.

.. tabs::

    .. code-tab:: py

        import xgboost as xgb
        from sklearn.datasets import make_classification
        num_classes = 3
        X, y = make_classification(n_samples=1000, n_informative=5,
                                   n_classes=num_classes)
        dtrain = xgb.DMatrix(data=X, label=y)
        num_parallel_tree = 4
        num_boost_round = 16
        # total number of built trees is num_parallel_tree * num_classes * num_boost_round

        # We build a boosted random forest for classification here.
        booster = xgb.train({
            'num_parallel_tree': 4, 'subsample': 0.5, 'num_class': 3},
                            num_boost_round=num_boost_round, dtrain=dtrain)

        # This is the sliced model, containing [3, 7) forests
        # step is also supported with some limitations like negative step is invalid.
        sliced: xgb.Booster = booster[3:7]

        # Access individual tree layer
        trees = [_ for _ in booster]
        assert len(trees) == num_boost_round

    .. code-tab:: r R

        library(xgboost)
        data(agaricus.train, package = "xgboost")
        dm <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)

        model <- xgb.train(
          params = xgb.params(objective = "binary:logistic", max_depth = 4),
          data = dm,
          nrounds = 20
        )
        sliced <- model[seq(3, 7)]
        ##### xgb.Booster
        # of features: 126
        # of rounds:  5

The sliced model is a copy of selected trees, that means the model itself is immutable
during slicing. This feature is the basis of ``save_best`` option in early stopping
callback. See :ref:`sphx_glr_python_examples_individual_trees.py` for a worked example on
how to combine prediction with sliced trees.

.. note::

   The returned model slice doesn't contain attributes like
   :py:class:`~xgboost.Booster.best_iteration` and
   :py:class:`~xgboost.Booster.best_score`.
