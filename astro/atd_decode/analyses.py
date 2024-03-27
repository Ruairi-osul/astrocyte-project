from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
from astro.atd_decode.runner import ATDDecodeRunner
from trace_minder.surrogates import Rotater, TraceSampler
from astro.transforms.groups import _permute_dict_arrays
from sklearn.base import clone


class ATDExperiment:
    def get_mean_test_scores(self, results: dict) -> np.number:
        return np.mean(results["test_score"])


class AllNeurons(ATDExperiment):
    def __init__(
        self,
        base_runner: ATDDecodeRunner,
        n_rotations: int | None = None,
        n_group_permutations: int | None = None,
    ):
        self.base_runner = base_runner
        self.n_rotations = n_rotations
        self.n_group_permutations = n_group_permutations

        self.obs_results_ = None
        self.rotated_results_ = None
        self.permuted_results_ = None

    def run_single(self, runner: ATDDecodeRunner) -> dict:
        results = runner.run()
        return results

    def make_group_permutation(self, base_runner: ATDDecodeRunner) -> ATDDecodeRunner:
        surrogate_runner = deepcopy(base_runner)
        surrogate_runner.data_loader.data_config.group_splitter.permute_neurons = True
        return surrogate_runner

    def make_rotation(self, base_runner: ATDDecodeRunner) -> ATDDecodeRunner:
        surrogate_runner = deepcopy(base_runner)
        surrogate_runner.preprocessor.preprocess_config.trace_rotator = Rotater(
            time_col="time", copy=True
        )
        return surrogate_runner

    def run_observed(self) -> dict:
        self.obs_results_ = self.run_single(self.base_runner)
        return self.obs_results_

    def run_rotated(self) -> list[dict]:
        self.rotated_results_ = [
            self.run_single(self.make_rotation(self.base_runner))
            for _ in tqdm(range(self.n_rotations), desc="Trace Rotations")
        ]
        return self.rotated_results_

    def run_permuted(self) -> list[dict]:
        self.permuted_results_ = [
            self.run_single(self.make_group_permutation(self.base_runner))
            for _ in tqdm(range(self.n_group_permutations), desc="Group Permutations")
        ]
        return self.permuted_results_

    def run(self) -> dict:
        self.obs_results_ = self.run_observed()

        if self.n_rotations is not None:
            self.rotated_results_ = self.run_rotated()
        if self.n_group_permutations is not None:
            self.permuted_results_ = self.run_permuted()

        results = dict(
            observed=self.obs_results_,
            rotated=self.rotated_results_,
            permuted=self.permuted_results_,
        )
        return results


class IncreasingNeuronNumber(ATDExperiment):
    def __init__(
        self,
        base_runner: ATDDecodeRunner,
        min_neurons: int = 1,
        max_neurons: int = 100,
        n_samples: int = 30,
        run_rot: bool = False,
        run_obs: bool = True,
    ):
        self.base_runner = base_runner
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.neuron_range_ = np.arange(min_neurons, max_neurons + 1)
        self.n_samples = n_samples
        self.run_rot = run_rot
        self.run_obs = run_obs

    def make_runner(self, n_neurons: int, shuffle: bool = False) -> ATDDecodeRunner:
        new_runner = deepcopy(self.base_runner)
        time_col = new_runner.data_loader.data_config.group_splitter.df_traces_time_col

        trace_sampler = TraceSampler(
            time_col=time_col, n_retained=n_neurons, with_replacement=False
        )
        if shuffle:
            new_runner.preprocessor.preprocess_config.trace_rotator = Rotater(
                time_col=time_col, copy=True
            )
        new_runner.preprocessor.preprocess_config.trace_sampler = trace_sampler
        return new_runner

    def run_single(self, runner: ATDDecodeRunner) -> dict:
        return runner.run()

    def run_observed(self) -> pd.DataFrame:
        res = []
        for neuron_number in tqdm(self.neuron_range_, desc="neuron_number"):
            for sample_idx in range(self.n_samples):
                runner = self.make_runner(neuron_number)
                cross_val_res = self.run_single(runner)
                mean_test_score = np.mean(cross_val_res["test_score"])

                res.append(
                    {
                        "neuron_number": neuron_number,
                        "sample_idx": sample_idx,
                        "mean_test_score": mean_test_score,
                    }
                )

        return pd.DataFrame(res)

    def run_rotated(self) -> pd.DataFrame:
        res = []
        for neuron_number in tqdm(self.neuron_range_, desc="neuron_number"):
            for sample_idx in range(self.n_samples):
                runner = self.make_runner(neuron_number, shuffle=True)
                cross_val_res = self.run_single(runner)
                mean_test_score = np.mean(cross_val_res["test_score"])

                res.append(
                    {
                        "neuron_number": neuron_number,
                        "sample_idx": sample_idx,
                        "mean_test_score": mean_test_score,
                    }
                )

        return pd.DataFrame(res)

    def run(self) -> dict:
        out = {}
        if self.run_obs:
            out["observed"] = self.run_observed()
        if self.run_rot:
            out["rotated"] = self.run_rotated()
        return out


class IncreasingPCNumber(ATDExperiment):
    def __init__(
        self,
        base_runner: ATDDecodeRunner,
        min_pc: int = 1,
        max_pc: int = 100,
        n_rotations: int = 75,
        run_rot: bool = False,
        run_obs: bool = True,
        pca_step_name: str = "pca",
    ):
        self.base_runner = base_runner
        self.base_model = clone(self.base_runner.decoder.model_config.model)
        self.min_pc = min_pc
        self.max_pc = max_pc
        self.pc_range_ = np.arange(min_pc, max_pc + 1)
        self.n_rotations = n_rotations
        self.run_rot = run_rot
        self.run_obs = run_obs
        self.pca_step_name = pca_step_name

    def make_runner(self, n_pcs: int, shuffle: bool = False) -> ATDDecodeRunner:
        new_runner = deepcopy(self.base_runner)

        new_model = clone(self.base_model)
        new_model.named_steps[self.pca_step_name].n_components = n_pcs
        new_runner.decoder.model_config.model = new_model

        if shuffle:
            time_col = (
                new_runner.data_loader.data_config.group_splitter.df_traces_time_col
            )
            new_runner.preprocessor.preprocess_config.trace_rotator = Rotater(
                time_col=time_col, copy=True
            )
        return new_runner

    def run_single(self, runner: ATDDecodeRunner) -> dict:
        return runner.run()

    def run_observed(self) -> pd.DataFrame:
        res = []
        for pc_number in tqdm(self.pc_range_, desc="pc_number-observed"):
            runner = self.make_runner(pc_number)
            cross_val_res = self.run_single(runner)
            explained_variance = [
                mod.named_steps[self.pca_step_name].explained_variance_ratio_
                for mod in cross_val_res["estimator"]
            ]

            res = res + [
                {
                    "cv_idx": i,
                    "pc_number": pc_number,
                    "test_scores": score,
                    "explained_variance": exp_var,
                }
                for i, (score, exp_var) in enumerate(
                    zip(cross_val_res["test_score"], explained_variance)
                )
            ]

        return pd.DataFrame(res)

    def run_rotated(self) -> pd.DataFrame:
        res = []
        for rotation_idx in tqdm(range(self.n_rotations), desc="rotation"):
            for pc_number in self.pc_range_:
                runner = self.make_runner(pc_number, shuffle=True)
                cross_val_res = self.run_single(runner)
                mean_test_score = np.mean(cross_val_res["test_score"])
                explained_variance = [
                    mod.named_steps[self.pca_step_name].explained_variance_ratio_
                    for mod in cross_val_res["estimator"]
                ]
                mean_explained_variance = np.mean(explained_variance, axis=0)

                res.append(
                    {
                        "pc_number": pc_number,
                        "rotation_idx": rotation_idx,
                        "mean_test_score": mean_test_score,
                        "mean_explained_variance": mean_explained_variance,
                    }
                )

        return pd.DataFrame(res)

    def run(self) -> dict:
        out = {}
        if self.run_obs:
            out["observed"] = self.run_observed()
        if self.run_rot:
            out["rotated"] = self.run_rotated()
        return out


class NeuronDrop(ATDExperiment):
    def __init__(
        self,
        base_runner: ATDDecodeRunner,
        type_mapper: dict[str, np.ndarray],
        n_permutations: int | None = None,
    ):
        self.base_runner = base_runner
        self.type_mapper = type_mapper
        self.n_permutations = n_permutations

        if self.base_runner.preprocessor.preprocess_config.neuron_dropper is None:
            raise ValueError(
                "Neuron Dropper not set in base_runner. Please set a Neuron Dropper in the Preprocessor."
            )

        self.obs_results_ = None
        self.permuted_results_ = None

    def run_single_drop(
        self,
        runner: ATDDecodeRunner,
        type_mapper: dict[str, np.ndarray],
        drop_type: str,
    ) -> dict:
        drop_neurons = type_mapper[drop_type]

        new_runner = deepcopy(runner)
        new_runner.preprocessor.preprocess_config.neuron_dropper.neuron_to_drop = (
            drop_neurons
        )
        return new_runner.run()

    def run_single(
        self, runner: ATDDecodeRunner, type_mapper: dict[str, np.ndarray]
    ) -> dict[str, dict]:
        results_by_type = {}
        for drop_type in type_mapper.keys():
            results_by_type[drop_type] = self.run_single_drop(
                runner, type_mapper=type_mapper, drop_type=drop_type
            )
        return results_by_type

    def run_observed(self) -> dict[str, dict]:
        self.obs_results_ = self.run_single(self.base_runner, self.type_mapper)
        return self.obs_results_

    def run_permuted(self) -> pd.DataFrame:
        permuted_results = []
        for idx in tqdm(range(self.n_permutations), desc="Neuron Drop Permutations"):
            permuted_type_mapper = _permute_dict_arrays(deepcopy(self.type_mapper))
            permuted_result = self.run_single(self.base_runner, permuted_type_mapper)
            mean_scores_by_type = {
                k: np.mean(v["test_score"]) for k, v in permuted_result.items()
            }
            mean_scores_by_type["idx"] = idx
            permuted_results.append(mean_scores_by_type)
        self.permuted_results_ = pd.DataFrame(permuted_results)
        return self.permuted_results_

    def run(self) -> dict:
        self.obs_results_ = self.run_observed()
        if self.n_permutations is not None:
            self.permuted_results_ = self.run_permuted()
        return dict(observed=self.obs_results_, permuted=self.permuted_results_)


class AllNeuronsGroupCompare(ATDExperiment):
    def __init__(
        self,
        base_experiment: AllNeurons,
        groups: list[str],
    ):
        self.base_experiment = base_experiment
        self.groups = groups

        self.obs_results_: dict[str, dict] | None = None
        self.permuted_diffs_: pd.DataFrame | None = None

    def make_group_experiment(
        self, group: str, base_experiment: AllNeurons
    ) -> AllNeurons:
        group_experiment = deepcopy(base_experiment)
        group_experiment.base_runner.data_loader.data_config.group = group
        return group_experiment

    def run_observed(self) -> dict:
        obs_results = {}
        for group in self.groups:
            group_experiment = self.make_group_experiment(group, self.base_experiment)
            result_dict = group_experiment.run_observed()
            obs_results[group] = self.get_mean_test_scores(result_dict)

        obs_results["diff"] = obs_results[self.groups[0]] - obs_results[self.groups[1]]
        self.obs_results_ = obs_results
        return self.obs_results_

    def run_permuted(self) -> pd.DataFrame:
        scores = {}
        for group in self.groups:
            group_experiment = self.make_group_experiment(group, self.base_experiment)
            results = group_experiment.run_permuted()
            scores[group] = [self.get_mean_test_scores(result) for result in results]

        df_permuted = pd.DataFrame(scores)
        df_permuted["diff"] = df_permuted[self.groups[0]] - df_permuted[self.groups[1]]
        self.permuted_diffs_ = df_permuted
        return self.permuted_diffs_

    def run(self) -> dict:
        self.obs_results_ = self.run_observed()
        self.permuted_diffs_ = self.run_permuted()
        return dict(observed=self.obs_results_, permuted=self.permuted_diffs_)
