import logging
from typing import Optional, Union

import numpy as np

from ray.tune.search import (
    UNDEFINED_METRIC_MODE,
    UNDEFINED_SEARCH_SPACE,
    UNRESOLVED_SEARCH_SPACE,
    Searcher,
)
from ray.tune.search.sample import Domain, Categorical, Float, Integer, Quantized, Uniform
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils import flatten_dict
from ray.tune.experiment import Trial


logger = logging.getLogger(__name__)


class TrialInfo:
    def __init__(self, config: dict, score: float):
        self.config = config
        self.score = score

    def __str__(self):
        return f"TrialInfo(config={self.config}, score={self.score})"

    def __repr__(self):
        return self.__str__()


class DomainInfo:
    def __init__(self,
                 bounds: tuple[float, float], type_: str,
                 encoded_categories: dict[int, float] = None):
        self.bounds = bounds
        self.type = type_
        self.encoding = encoded_categories

    def __str__(self):
        return f"DomainInfo(bounds={self.bounds}, type={self.type}, encoding={self.encoding})"

    def __repr__(self):
        return self.__str__()


class SoFASearch(Searcher):
    """Search Algorithm based on Survival of the Fittest (SoFA) optimization method."""

    def __init__(
        self,
        space: Optional[dict[str, Domain]] = None,
        metric: Optional[str] = None,
        mode: str = "max",
        population_size: int = 20,
        max_iter_num: int = 1000,
        mutation_rate: Union[float, tuple[float, float]] = (0.5, 1.0),
        cross_over_rate: Union[float, tuple[float, float]] = (0.5, 1.0),
        distance_eps: float = 1e-4,
        use_lambda: bool = True,
        use_epsilon: bool = True,
        use_mutation: bool = True,
        categorical_types: dict[str, str] = None,
        points_to_evaluate: Optional[list[dict]] = None,
        max_concurrent: int = 0,
    ):
        """
        Args:
            space: A dictionary mapping parameter names to search spaces.
            metric: The name of the metric to optimize.
            mode: Whether to optimize a minimization or maximization problem.
            population_size: The number of individuals in the population.
            mutation_rate: The probability of mutating an individual.
            cross_over_rate: The probability of crossing over two individuals. 
                Used for nominal categorical parameters.
            distance_eps: The minimum distance between two individuals.
            use_lambda: Whether to use the lambda parameter.
            use_epsilon: Whether to use the epsilon parameter.
            use_mutation: Whether to use mutation.
            categorical_types: A dictionary mapping parameter 
                names to categorical types: 'nominal' or 'ordinal'.
            points_to_evaluate: A list of points to evaluate in case of 
                having a predefined set of points to evaluate.
            max_concurrent: The maximum number of concurrent trials.
        """

        assert mode in [
            "min", "max"], f"`mode` must be 'min' or 'max', got '{mode}'"

        super(SoFASearch, self).__init__(metric=metric, mode=mode)
        self._population_size = population_size
        self._max_iter_num = max_iter_num
        self._distance_eps = distance_eps
        self._use_lambda = use_lambda
        self._use_epsilon = use_epsilon
        self._use_mutation = use_mutation
        self._categorical_types = categorical_types or {}
        self._original_space = space or {}
        self._flattened_original_space = flatten_dict(
            space, prevent_delimiter=True) if space else {}

        # Handle mutation rate
        self._mutation_boundaries = self._handle_rate(mutation_rate)
        self._cross_over_boundaries = self._handle_rate(cross_over_rate)

        if isinstance(space, dict) and space:
            resolved_vars, domain_vars, grid_vars = parse_spec_vars(space)
            if domain_vars or grid_vars:
                logger.warning(
                    UNRESOLVED_SEARCH_SPACE.format(par="space", cls=type(self))
                )
                space = self.convert_search_space(space)  # , join=True)

        self._space = space

        # Internal state
        self._live_trials: dict[str, dict] = {}  # trial_id -> config
        self._trial_scores: dict[str, float] = {}  # trial_id -> score
        self._best_config: Optional[dict] = None
        self._best_score = float('-inf') if mode == "max" else float('inf')

        if points_to_evaluate:
            self._points_to_evaluate = points_to_evaluate
        else:
            self._points_to_evaluate = []

        self._max_concurrent = max_concurrent

    def set_max_concurrency(self, max_concurrent: int) -> bool:
        self._max_concurrent = max_concurrent
        return True

    def sample_config(self):
        config = {}
        for param, domain in self._original_space.items():
            if isinstance(domain, Domain):
                config[param] = domain.sample()
            else:
                config[param] = domain

        return config

    @staticmethod
    def _handle_rate(rate: Union[float, tuple[float, float]]):
        if isinstance(rate, (float, int)):
            return float(rate), float(rate)
        else:
            return rate[0], rate[1]

    @staticmethod
    def _is_monotonic(values: list):
        return (all(x <= y for x, y in zip(values, values[1:])) or
                all(x >= y for x, y in zip(values, values[1:])))

    def _is_normalizing(self, param: str):
        # Currently we normalize all parameters except for nominal categoricals
        return self._space[param].type != "nominal"

    def _check_ordered(self, categories: list):
        return all(isinstance(categories[i], (int, float))
                   for i in range(len(categories))) and self._is_monotonic(categories)

    def convert_search_space(
            self, spec: dict, join: bool = False) -> dict[str, DomainInfo]:

        resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)

        if grid_vars:
            raise ValueError(
                "Grid search parameters cannot be automatically converted "
                "to a SoFA search space."
            )

        # Flatten and resolve again after checking for grid search.
        spec = flatten_dict(spec, prevent_delimiter=True)
        resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)

        def resolve_value(key: str,
                          domain: Domain) -> tuple[tuple[float, float],
                                                   str,
                                                   dict | None]:
            """
            Returns bounds, type and additional dict 
            with index-encoded ordinal categories if applicable
            """
            sampler = domain.get_sampler()
            if isinstance(sampler, Quantized):
                logger.warning(
                    "SoFA search does not support quantization. "
                    "Dropped quantization."
                )
                sampler = sampler.get_sampler()

            if isinstance(domain, Float):
                if domain.sampler is not None and not isinstance(
                    domain.sampler, Uniform
                ):
                    logger.warning(
                        "SoFA does not support specific sampling methods. "
                        "The {} sampler will be dropped.".format(sampler)
                    )
                return (domain.lower, domain.upper), "float", None

            if isinstance(domain, Categorical):
                if domain.sampler is not None and not isinstance(
                    domain.sampler, Uniform
                ):
                    logger.warning(
                        "SoFA does not support specific sampling methods. "
                        "The {} sampler will be dropped.".format(sampler)
                    )

                if key in self._categorical_types:
                    kind = self._categorical_types[key]
                else:
                    kind = "ordinal" if self._check_ordered(
                        domain.categories) else "nominal"

                if kind == "ordinal":
                    encode = {domain.categories[i]: i
                              for i in range(len(domain.categories))}
                else:
                    encode = None

                return (0, len(domain.categories) - 1), kind, encode

            raise ValueError(
                "SoFA does not support parameters of type "
                "`{}`".format(type(domain).__name__)
            )

        # Parameter name is e.g. "a/b/c" for nested dicts
        space: dict[str, DomainInfo] = {}
        for path, domain in domain_vars:
            key = "/".join(path)
            bounds, type_, encode = resolve_value(key, domain)
            space[key] = DomainInfo(bounds, type_, encode)

        if join:
            spec.update(space)
            space = spec

        return space

    def set_search_properties(
        self, metric: Optional[str], mode: Optional[str], config: dict, **spec
    ) -> bool:

        if self._space:
            return False

        # Here config == param_space
        self._original_space = config
        self._flattened_original_space = flatten_dict(
            config, prevent_delimiter=True)
        space = self.convert_search_space(config)
        self._space = space
        if metric:
            self._metric = metric
        if mode:
            self._mode = mode

        if self._mode == "max":
            self._metric_op = 1.0
        elif self._mode == "min":
            self._metric_op = -1.0

        while len(self._points_to_evaluate) < self._population_size:
            # For initial population, sample randomly from space
            self._points_to_evaluate.append(self.sample_config())

        return True

    def _normalize(self, config: dict) -> dict:
        normalized = {}
        for param, value in config.items():
            if param not in self._space:
                normalized[param] = value
                continue

            lower, upper = self._space[param].bounds
            domain = self._flattened_original_space.get(param)

            # Handle numerical parameters
            if isinstance(domain, (Float, Integer)):
                if upper == lower:
                    normalized_val = 0.0
                else:
                    normalized_val = 2.0 * \
                        (value - lower) / (upper - lower) - 1.0
                normalized[param] = normalized_val

            # Handle ordinal categoricals
            elif isinstance(domain, Categorical) and self._space[param].type == "ordinal":
                normalized_val = 2.0 * \
                    (self._space[param].encoding[value] /
                     (len(domain.categories) - 1)) - 1.0
                normalized[param] = normalized_val

            # Handle other parameters without normalization
            else:
                normalized[param] = value
        return normalized

    def _denormalize(self, normalized_config: dict) -> dict:
        denormalized = {}
        for param, norm_val in normalized_config.items():
            if param not in self._space:
                denormalized[param] = norm_val
                continue

            lower, upper = self._space[param].bounds
            domain = self._flattened_original_space.get(param)

            # Handle numerical parameters
            if isinstance(domain, (Float, Integer)):
                original_val = (norm_val + 1.0) * (upper - lower) / 2.0 + lower
                original_val = np.clip(original_val, lower, upper)
                if isinstance(domain, Integer):
                    original_val = int(np.round(original_val))
                denormalized[param] = original_val
            # Handle categorical ordinal parameters
            elif isinstance(domain, Categorical) and self._space[param].type == "ordinal":
                scaled_val = (norm_val + 1.0) * \
                    (len(domain.categories) - 1) / 2.0
                idx = int(
                    np.round(np.clip(scaled_val, 0, len(domain.categories)-1)))
                denormalized[param] = domain.categories[idx]
            # Handle other parameters, assuming they were not normalized
            else:
                denormalized[param] = norm_val
        return denormalized

    @staticmethod
    def _sample_rate(rate: tuple[float, float]) -> float:
        """Sample a new rate using dithering."""
        return np.random.uniform(rate[0], rate[1])

    def _mutate(self, config_probs: dict) -> Optional[dict]:
        """Generate new configuration using mutation operator."""
        mutation_rate = self._sample_rate(self._mutation_boundaries)

        if len(config_probs) < 3:
            return None

        trial_ids = np.random.choice(
            list(config_probs.keys()),
            p=list(config_probs.values()),
            size=3,
            replace=False
        )
        trial_probs = [config_probs[trial_id] for trial_id in trial_ids]

        normalized_configs = [
            self._normalize(self._live_trials[trial_id])
            for trial_id in trial_ids]

        config_keys = self._best_config.keys()
        config_keys = [key for key in config_keys if self._is_normalizing(key)]

        configs_values = [
            np.array([normalized_config[key] for key in config_keys])
            for normalized_config in normalized_configs
        ]

        normalized_best_config = self._normalize(self._best_config)
        best_values = np.array([normalized_best_config[key]
                               for key in config_keys])

        mutated_values = \
            configs_values[0] + \
            mutation_rate * (best_values - configs_values[0]) + \
            mutation_rate * \
            (configs_values[1] - configs_values[2])

        mutated_config = dict(zip(config_keys, mutated_values))

        # Handle non-normalized parameters, choising values
        # from trials according to their probabilities
        for key in normalized_best_config.keys():
            if key not in config_keys:
                trial_id = np.random.choice(
                    trial_ids,
                    p=trial_probs
                )
                mutated_config[key] = self._live_trials[trial_id][key]

        return self._denormalize(mutated_config)

    def _calculate_epsilon(self, iterations_count: int) -> float:
        """Calculate epsilon sequence value."""
        return max(0.1, 1 - iterations_count / self._max_iter_num)

    def _calculate_lambda(self, iterations_count: int) -> float:
        """Calculate adaptive lambda parameter."""
        return (np.sqrt(iterations_count / 100) + 1)

    def _calculate_selection_probabilities(self) -> dict:
        """Calculate selection probabilities using SoFA formula."""
        trials_count = len(self._trial_scores)
        lambda_val = self._calculate_lambda(
            trials_count) if self._use_lambda else 1
        scores = np.array(list(self._trial_scores.values()))

        if self._mode == "max":
            probs = np.power(scores, lambda_val)
        else:
            probs = np.power(1.0 / (scores + 1e-10), lambda_val)

        probs = probs / np.sum(probs)
        return dict(zip(self._trial_scores.keys(), probs))

    def _generate_new_config(self, reference_config: dict, epsilon: float) -> dict:
        """Generate new config using SoFA's probability distribution."""
        normalized_ref = self._normalize(reference_config)
        keys = [key for key in normalized_ref.keys() if self._is_normalizing(key)]

        values = np.array([normalized_ref[key] for key in keys], dtype=float)

        y = np.random.random(size=values.shape)
        new_values = np.tan(
            y * np.arctan((1.0 - values)/epsilon) +
            (1 - y) * np.arctan((-1.0 - values)/epsilon)
        ) * epsilon + values

        new_values = np.clip(new_values, -1.0, 1.0)

        new_config = dict(zip(keys, new_values))

        # Handle non-normalized parameters, sampling values
        # according to cross-over rate
        cross_over_rate = self._sample_rate(self._cross_over_boundaries)
        for key in normalized_ref.keys():
            if key not in keys:
                if np.random.random() <= cross_over_rate:
                    domain = self._original_space.get(key)
                    if isinstance(domain, Domain):
                        new_config[key] = domain.sample()
                    else:
                        new_config[key] = domain
                else:
                    new_config[key] = normalized_ref[key]

        return self._denormalize(new_config)

    def suggest(self, trial_id: str) -> Optional[dict]:
        """Suggest a new configuration to try."""

        print("ID:", trial_id, "LIVE TRIALS:", len(self._live_trials),
              "POINTS TO EVAL:", len(self._points_to_evaluate))
        
        max_concurrent = (
            self._max_concurrent if self._max_concurrent > 0 else float("inf")
        )

        if len(self._live_trials) >= max_concurrent:
            return None

        if self._points_to_evaluate:
            config = self._points_to_evaluate.pop(0)
            self._live_trials[trial_id] = config
            return config

        # Calculate selection probabilities
        selection_probs = self._calculate_selection_probabilities()

        if len(selection_probs) == 0:
            return None

        # Get reference configuration
        reference_config = None
        if self._use_mutation and self._best_config is not None:
            reference_config = self._mutate(selection_probs)

        if not reference_config:
            reference_trial_id = np.random.choice(
                list(selection_probs.keys()),
                p=list(selection_probs.values())
            )
            reference_config = self._live_trials[reference_trial_id]

        # Generate new configuration
        epsilon = self._calculate_epsilon(
            len(self._trial_scores)) if self._use_epsilon else 1
        new_config = self._generate_new_config(reference_config, epsilon)

        self._live_trials[trial_id] = new_config
        return new_config
    
    def on_trial_result(self, trial_id: str, result: dict):
        """Update the searcher with the latest trial result."""
        score = result.get(self._metric)
        if score is None:
            return

        current_score = self._trial_scores.get(trial_id)
        # Update the trial's best score so far
        if current_score is None:
            self._trial_scores[trial_id] = score
        else:
            if (self._mode == "max" and score > current_score) or \
            (self._mode == "min" and score < current_score):
                self._trial_scores[trial_id] = score

        # Update the global best score and config
        if (self._mode == "max" and score > self._best_score) or \
        (self._mode == "min" and score < self._best_score):
            self._best_score = score
            self._best_config = self._live_trials.get(trial_id)

    def on_trial_complete(self, trial_id: str, result: Optional[dict] = None, error: bool = False):
        """Handle completed trial result."""
        if error or result is None:
            self._live_trials.pop(trial_id, None)
            self._trial_scores.pop(trial_id, None)
            return

        score = result.get(self._metric)
        if score is None:
            return

        self._trial_scores[trial_id] = score

        # Update best score
        if self._mode == "max":
            if score > self._best_score:
                self._best_score = score
                self._best_config = self._live_trials[trial_id]
        else:
            if score < self._best_score:
                self._best_score = score
                self._best_config = self._live_trials[trial_id]
