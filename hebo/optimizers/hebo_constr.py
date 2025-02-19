# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import sys
import warnings

from typing import Optional

import numpy  as np
import pandas as pd
import torch
from copy import deepcopy
from torch.quasirandom import SobolEngine
from sklearn.preprocessing import power_transform

from hebo.design_space.design_space import DesignSpace
from hebo.models.model_factory import get_model
from hebo.acquisitions.acq import MACEConstr, Mean, Sigma
from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt
from hebo.optimizers.util import ensure_hard_constr

from .abstract_optimizer import AbstractOptimizer

torch.set_num_threads(min(1, torch.get_num_threads()))

def enable_power_transform(y_raw: np.ndarray)->torch.FloatTensor:
    try:
        if y_raw.min() <= 0:
            y = torch.FloatTensor(power_transform(y_raw / y_raw.std(), method = 'yeo-johnson'))
        else:
            y = torch.FloatTensor(power_transform(y_raw / y_raw.std(), method = 'box-cox'))
            if y.std() < 0.5:
                y = torch.FloatTensor(power_transform(y_raw / y_raw.std(), method = 'yeo-johnson'))
        if y.std() < 0.5:
            raise RuntimeError('Power transformation failed')
    except Exception as e:
        print(f'power transformer error: {e}')
        y     = torch.FloatTensor(y_raw).clone()
    return y

class HEBOConstr(AbstractOptimizer):
    support_parallel_opt  = True
    support_combinatorial = True
    support_contextual    = True
    def __init__(self, 
                space, 
                num_model_constr, 
                y_thres = None, 
                model_name = 'multi_task', 
                base_model_name='gpy', 
                rand_sample = None, 
                acq_cls = MACEConstr,
                es = 'nsga2', 
                model_config = None,
                scramble_seed: Optional[int] = None, 
                num_hard_constr: Optional[int] = None, 
                num_hidden_constr: Optional[int] = 1,
                max_sequence_length: Optional[int] = 12):  # max_sequence_length default to 2^12=4096
        """
        model_name  : surrogate model to be used
        rand_sample : iterations to perform random sampling
        scramble_seed : seed used for the sobol sampling of the first initial points
        NOTE: If the thresholds of inequal constraints are not zero, the observed y should shifted by the threshold
        """
        super().__init__(space)
        self.space       = space
        self.es          = es
        self.X           = pd.DataFrame(columns = self.space.para_names)
        self.y           = np.zeros((0, num_model_constr + 1))

        self.model_name  = model_name
        self.rand_sample = 1 + self.space.num_paras if rand_sample is None else max(2, rand_sample)
        self.scramble_seed = scramble_seed
        self.sobol       = SobolEngine(self.space.num_paras, scramble = True, seed = scramble_seed)
        self.acq_cls     = acq_cls
        self.base_model_name = base_model_name
        self._model_config = model_config
        self.num_hard_constr = num_hard_constr
        self.num_hidden_constr = num_hidden_constr
        self.max_sequence_length = max_sequence_length
        if isinstance(y_thres,list):
            self.y_thres = np.array(y_thres).reshape([1, -1])
            assert self.y_thres.shape[1] == num_model_constr, 'If give list, please give each output dim one threshold'
        elif isinstance(y_thres, int):
            self.y_thres = np.array([y_thres]).reshape([1, -1])
            assert num_model_constr == 1, 'If give int, only one ouput dim is supported'
        elif y_thres is None:
            self.y_thres = y_thres
        

    def quasi_sample(self, n, fix_input = None):
        samp    = self.sobol.draw(n)
        samp    = samp * (self.space.opt_ub - self.space.opt_lb) + self.space.opt_lb
        x       = samp[:, :self.space.num_numeric]
        xe      = samp[:, self.space.num_numeric:]
        for i, n in enumerate(self.space.numeric_names):
            if self.space.paras[n].is_discrete_after_transform:
                x[:, i] = x[:, i].round()
        df_samp = self.space.inverse_transform(x, xe)
        if fix_input is not None:
            for k, v in fix_input.items():
                df_samp[k] = v
        return df_samp

    @property
    def model_config(self):
        if self._model_config is None:
            if self.base_model_name == 'gp':
                cfg = {
                        'lr'           : 0.01,
                        'num_epochs'   : 100,
                        'verbose'      : False,
                        'noise_lb'     : 8e-4,
                        'pred_likeli'  : False,
                        'base_model_name': 'gp'
                        }
            elif self.base_model_name == 'gpy':
                cfg = {
                        'verbose' : False,
                        'warp'    : True,
                        'space'   : self.space,
                        'base_model_name': 'gpy'
                        }
            elif self.base_model_name == 'gpy_mlp':
                cfg = {
                        'verbose' : False,
                        'base_model_name': 'gpy_mlp'
                        }
            elif self.base_model_name == 'rf':
                cfg =  {
                        'n_estimators' : 20,
                        'base_model_name': 'rf'
                        }
            else:
                cfg = {}
        else:
            cfg = deepcopy(self._model_config)

        if self.space.num_categorical > 0:
            cfg['num_uniqs'] = [len(self.space.paras[name].categories) for name in self.space.enum_names]
        return cfg

    def get_best_id(self, fix_input : dict = None) -> int:
        if fix_input is None:
            if self.y_thres is not None:
                valid_row = (self.y[:,1:]<=self.y_thres).all(axis=1)
                if not valid_row.any():
                    warnings.warn('No feasible rec has been delivered!! Consider relax the y_thres')
                    return None
                else:
                    min_val = self.y[valid_row,:1].min()
                    valid_rows_indices = np.where(valid_row)[0]
                    min_val_index_in_valid = np.where(self.y[valid_row, 0] == min_val)[0]
                    return valid_rows_indices[min_val_index_in_valid][0]
            elif self.y_thres is None:
                return np.argmin(self.y[:,:1].reshape(-1))
        
        X = self.X.copy()
        y = self.y.copy()
        for k, v in fix_input.items():
            if X[k].dtype != 'float':
                crit = (X[k] != v).values
            else:
                crit = ((X[k] - v).abs() > np.finfo(float).eps).values
            y[crit]  = np.inf
        if np.isfinite(y).any():
            return np.argmin(y[:,:1].reshape(-1))
        else:
            return np.argmin(self.y[:,:1].reshape(-1))

    def suggest(self, n_suggestions=1, fix_input = None, rf_with_thres = None):
        if self.acq_cls != MACEConstr and n_suggestions != 1:
            raise RuntimeError('Parallel optimization is supported only for MACE acquisition')
        if self.X.shape[0] < self.rand_sample:
            sample = self.quasi_sample(n_suggestions, fix_input)
            if self.num_hard_constr:
                sample = ensure_hard_constr(sample, max_sequence_length=2**self.max_sequence_length)
            return sample
        else:
            X, Xe = self.space.transform(self.X)

            y_obj = self.y[:,0:1]
            y = enable_power_transform(y_obj)

            y_constr = self.y[:, 1:]
            y_thres = None         
            if y_constr.shape[1] > 0:
                y_constr_with_prefer = np.concatenate([y_constr, self.y_thres], axis = 0)
                y_constr_with_prefer = enable_power_transform(y_constr_with_prefer)
                y = torch.cat([y, y_constr_with_prefer[:-1, ]], axis=1)
                y_thres = y_constr_with_prefer[-1: ]

            model = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, y.shape[1], **self.model_config)
            model.fit(X, Xe, y)

            best_id = self.get_best_id(fix_input)
            best_x  = self.X.iloc[[best_id]]
            py_best, ps2_best = model.predict(*self.space.transform(best_x))
            py_best = py_best[:,0].detach().numpy().squeeze()
            ps_best = ps2_best[:,0].sqrt().detach().numpy().squeeze()

            iter  = max(1, self.X.shape[0] // n_suggestions)
            upsi  = 0.5
            delta = 0.01
            # kappa = np.sqrt(upsi * 2 * np.log(iter **  (2.0 + self.X.shape[1] / 2.0) * 3 * np.pi**2 / (3 * delta)))
            kappa = np.sqrt(upsi * 2 * ((2.0 + self.X.shape[1] / 2.0) * np.log(iter) + np.log(3 * np.pi**2 / (3 * delta))))
            acq = self.acq_cls( model,
                                space= self.space,
                                num_model_constr = self.y.shape[1]-1,
                                num_hard_constr = self.num_hard_constr,
                                num_hidden_constr = self.num_hidden_constr,
                                rf_with_thres = rf_with_thres,
                                best_y = py_best,
                                kappa = kappa,
                                y_thres = y_thres,
                                max_sequence_length = self.max_sequence_length)
            mu  = Mean(model.models[0])
            sig = Sigma(model.models[0], linear_a = -1.)
            opt = EvolutionOpt(self.space, acq, pop = 100, iters = 100, verbose = False, es=self.es)
            rec = opt.optimize(initial_suggest = best_x, fix_input = fix_input).drop_duplicates()
            rec = rec[self.check_unique(rec)]

            cnt = 0
            while rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                if self.num_hard_constr:
                    rand_rec = ensure_hard_constr(rand_rec, max_sequence_length=2**self.max_sequence_length)
                rand_rec = rand_rec[self.check_unique(rand_rec)]
                rec      = pd.concat([rec, rand_rec], axis = 0, ignore_index = True)
                cnt +=  1
                if cnt > 3:
                    # sometimes the design space is so small that duplicated sampling is unavoidable
                    break
            if rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                if self.num_hard_constr:
                    rand_rec = ensure_hard_constr(rand_rec, max_sequence_length=2**self.max_sequence_length)
                rec      = pd.concat([rec, rand_rec], axsi = 0, ignore_index = True)

            select_id = np.random.choice(rec.shape[0], n_suggestions, replace = False).tolist()
            x_guess   = []
            with torch.no_grad():
                py_all       = mu(*self.space.transform(rec)).squeeze().numpy()
                ps_all       = -1 * sig(*self.space.transform(rec)).squeeze().numpy()
                best_pred_id = np.argmin(py_all)
                best_unce_id = np.argmax(ps_all)
                if best_unce_id not in select_id and n_suggestions > 2:
                    select_id[0]= best_unce_id
                if best_pred_id not in select_id and n_suggestions > 2:
                    select_id[1]= best_pred_id
                rec_selected = rec.iloc[select_id].copy()
            return rec_selected

    def check_unique(self, rec : pd.DataFrame) -> [bool]:
        return (~pd.concat([self.X, rec], axis = 0).duplicated().tail(rec.shape[0]).values).tolist()

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : pandas DataFrame
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,k)
            Corresponding values where objective has been evaluated
        """
        
        valid_id = np.isfinite(y).all(axis = 1)
        XX       = X.iloc[valid_id]
        yy       = y[valid_id].reshape(-1, y.shape[1])
        self.X   = pd.concat([self.X, XX], axis = 0, ignore_index = True)
        self.y   = np.vstack([self.y, yy])

    @property
    def best_x(self)->pd.DataFrame:
        if self.X.shape[0] == 0:
            raise RuntimeError('No data has been observed!')
        else:
            if self.y_thres is not None:
                valid_row = (self.y[:,1:]<=self.y_thres).all(axis=1)
                if not valid_row.any():
                    warnings.warn('No feasible rec has been delivered!! Consider relax the y_thres')
                    return None
                else:
                    min_val = self.y[valid_row,:1].min()
                    valid_rows_indices = np.where(valid_row)[0]
                    min_val_index_in_valid = np.where(self.y[valid_row, 0] == min_val)[0]
                    min_val_row_number = valid_rows_indices[min_val_index_in_valid][0]
                    return self.X.iloc[[min_val_row_number]]
            else:
                return self.X.iloc[[self.y[:,:1].argmin()]]

    @property
    def best_y(self)->float:
        if self.X.shape[0] == 0:
            raise RuntimeError('No data has been observed!')
        else:
            if self.y_thres is not None:
                valid_row = (self.y[:,1:]<=self.y_thres).all(axis=1)
                if not valid_row.any():
                    warnings.warn('No feasible rec has been delivered!! Consider relax the y_thres')
                    return None
                else:
                    return self.y[valid_row,:1].min()
            else:
                return self.y[:,:1].min()
