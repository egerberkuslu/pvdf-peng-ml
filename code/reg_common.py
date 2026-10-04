"""Model factories of the fourteen-regressor benchmark.

model_factories() maps a model name to a zero-argument callable that builds an
unfitted scikit-learn estimator; hyperparameters are fixed (no tuning on the
evaluation folds) and every random model is seeded with SEED. The names follow
the keys of results/baseline/reg_stats.json and reg_models.json.
"""

SEED = 42


def model_factories():
    """name -> callable(). The GP optimizes its kernel internally (self-tuned)."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import (
        RandomForestRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
    )
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        Matern,
        WhiteKernel,
        ConstantKernel as C_,
    )
    from xgboost import XGBRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.compose import TransformedTargetRegressor

    def gp():
        k = C_(1.0) * Matern(length_scale=[1, 1, 1, 5], nu=2.5) + WhiteKernel(1e-3)
        return GaussianProcessRegressor(
            kernel=k,
            normalize_y=True,
            alpha=1e-6,
            n_restarts_optimizer=4,
            random_state=SEED,
        )

    def ttr(reg):
        # standardize y as well: V_rms (~0.05) and the energy proxy (~180) live on very
        # different scales, so scale-sensitive models (SVR/KRR/MLP) need it for a fair comparison
        return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())

    return {
        "Lineer": lambda: LinearRegression(),
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso": lambda: make_pipeline(
            StandardScaler(), Lasso(alpha=0.01, max_iter=10000)
        ),
        "Polinom2-OLS": lambda: make_pipeline(
            PolynomialFeatures(2), LinearRegression()
        ),
        "Polinom2-Ridge": lambda: make_pipeline(
            StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1.0)
        ),
        "SVR-RBF": lambda: ttr(
            make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, gamma="scale"))
        ),
        "KernelRidge": lambda: ttr(
            make_pipeline(
                StandardScaler(), KernelRidge(kernel="rbf", alpha=0.1, gamma=0.1)
            )
        ),
        "KNN": lambda: make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=5)
        ),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=400, random_state=SEED
        ),
        "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=400, random_state=SEED),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=SEED),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=SEED,
            verbosity=0,
        ),
        # ANN/MLP: the most common model in piezoelectric nanogenerator ML studies;
        # L2-regularized and scaled for the small sample size.
        "ANN-MLP": lambda: ttr(
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    alpha=0.05,
                    max_iter=4000,
                    random_state=SEED,
                ),
            )
        ),
        "ARD-GP": gp,
    }


def svr_tuned_estimator():
    """SVR with an inner grid search, for nested cross-validation (honest tuning)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV

    pipe = Pipeline([("s", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    grid = {"svr__C": [1, 10, 100], "svr__gamma": ["scale", 0.05, 0.2, 0.5]}
    return GridSearchCV(pipe, grid, cv=5, scoring="r2")
