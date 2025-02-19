# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import numpy as np
from torch import Tensor
from torch.distributions import Normal
from abc import ABC, abstractmethod
from ..models.base_model import BaseModel


def get_hard_constr(x, max_sequence_length, num_hard_constr, py, space):
    out_hard_constr = torch.zeros(py.shape[0], num_hard_constr)
    if num_hard_constr > 0 and space:
        condition_id_chunked = space.numeric_names.index("enable_chunked_prefill")
        condition_id_prefix = space.numeric_names.index("enable_prefix_caching")
        constr_id_max_seqs = space.numeric_names.index("max_num_seqs")
        constr_id_max_batched_tokens = space.numeric_names.index("max_num_batched_tokens")

        out_hard_constr[:, 0] = (1 - x[:, condition_id_chunked]) * (
                max_sequence_length - x[:, constr_id_max_batched_tokens]) - 1e-5
        out_hard_constr[:, 1] = x[:, constr_id_max_seqs] - x[:, constr_id_max_batched_tokens]
        out_hard_constr[:, 2] = x[:, condition_id_chunked] + x[:, condition_id_prefix] - 1.0
    return out_hard_constr

def get_hidden_constr(x, xe, rf_with_thres, num_hidden_constr, py, space):
    # constraints from rf
    out_hidden_constr = torch.zeros(py.shape[0], num_hidden_constr)
    if num_hidden_constr > 0 and space:
        df_X = space.inverse_transform(x, xe)
        df_X_numpy = df_X.to_numpy()
        out_hidden_constr[:, 0] = rf_with_thres[1] - torch.from_numpy(
            rf_with_thres[0].predict(df_X_numpy)).float()
    return out_hidden_constr

class Acquisition(ABC):
    def __init__(self, model, **conf):
        self.model = model

    @property
    @abstractmethod
    def num_obj(self):
        pass

    @property
    @abstractmethod
    def num_constr(self):
        pass

    @abstractmethod
    def eval(self, x : Tensor,  xe : Tensor) -> Tensor:
        """
        Shape of output tensor: (x.shape[0], self.num_obj + self.num_constr)
        """
        pass

    def __call__(self, x : Tensor,  xe : Tensor):
        return self.eval(x, xe)

class SingleObjectiveAcq(Acquisition):
    """
    Single-objective, unconstrained acquisition
    """
    def __init__(self, model : BaseModel, **conf):
        super().__init__(model, **conf)

    @property
    def num_obj(self):
        return 1

    @property
    def num_constr(self):
        return 0

class LCB(SingleObjectiveAcq):
    def __init__(self, model : BaseModel, **conf):
        super().__init__(model, **conf)
        self.kappa = conf.get('kappa', 3.0)
        assert(model.num_out == 1)
    
    def eval(self, x : Tensor, xe : Tensor) -> Tensor:
        py, ps2 = self.model.predict(x, xe)
        return py - self.kappa * ps2.sqrt()

class Mean(SingleObjectiveAcq):
    def __init__(self, model : BaseModel, **conf):
        super().__init__(model, **conf)
        assert(model.num_out == 1)

    def eval(self, x : Tensor, xe : Tensor) -> Tensor:
        py, _ = self.model.predict(x, xe)
        return py

class Sigma(SingleObjectiveAcq):
    def __init__(self, model : BaseModel, **conf):
        super().__init__(model, **conf)
        assert(model.num_out == 1)

    def eval(self, x : Tensor, xe : Tensor) -> Tensor:
        _, ps2 = self.model.predict(x, xe)
        return -1 * ps2.sqrt()

class EI(SingleObjectiveAcq):
    pass

class logEI(SingleObjectiveAcq):
    pass

class WEI(Acquisition):
    pass

class Log_WEI(Acquisition):
    pass

class MES(SingleObjectiveAcq):
    pass

# class qExpectedImprovement(SingleObjectiveAcq):
#     """
#         qEI utlizes the reparametric trick to achevie the differenctial property 
#     from output of acq function to input of suggorate model:
#                 p(y|x,D)~\mu(x,D)+L(x,D) \dot \epsilon, \epsilon ~ Normal(0, 1)

#                 \sum_m EI(y^m) = \min_q EI(\mu(x_q,D)+L(x_q,D) \dot \epsilon^m), 
#       where \epsilon^m is samples from Normal distribution
#     """
#     def __init__(self, model: BaseModel, best_y: float, **conf):
#         super().__init__(model, **conf)
#         self.tau = best_y
#         self.q = config.get('q', 1)

#     def eval(self, x: Tensor, xe: Tensor) -> Tensor:
#         assert x.shape[0]//self.q == x.shape[0]/

#     @property
#     def num_obj(self):
#         return 1

#     @property
#     def num_constr(self):
#         return 0

    

class MOMeanSigmaLCB(Acquisition):
    def __init__(self, model, best_y, **conf):
        super().__init__(model, **conf)
        self.best_y = best_y
        self.kappa  = conf.get('kappa', 2.0)
        assert(self.model.num_out == 1)

    @property
    def num_obj(self):
        return 2

    @property
    def num_constr(self):
        return 1

    def eval(self, x: Tensor, xe : Tensor) -> Tensor:
        """
        minimize (py, -1 * ps)
        s.t.     LCB  < best_y
        """
        with torch.no_grad():
            out        = torch.zeros(x.shape[0], self.num_obj + self.num_constr)
            py, ps2    = self.model.predict(x, xe)
            noise      = np.sqrt(self.model.noise)
            py        += noise * torch.randn(py.shape)
            ps         = ps2.sqrt()
            lcb        = py - self.kappa * ps
            out[:, 0]  = py.squeeze()
            out[:, 1]  = -1 * ps.squeeze()
            out[:, 2]  = lcb.squeeze() - self.best_y # lcb - best_y < 0
            return out

class MACEConstr(Acquisition):
    def __init__(self, model, best_y, num_model_constr, y_thres, **conf):
        super().__init__(model, **conf)
        self.kappa = conf.get('kappa', 2.0)
        self.eps   = conf.get('eps', 1e-4)
        self._num_hard_constr = conf.get('num_hard_constr', 3)
        self._num_hidden_constr = conf.get('num_hidden_constr', 1)
        self.rf_with_thres = conf.get('rf_with_thres', None)
        self.max_sequence_length = conf.get('max_sequence_length', 12)

        self.space = conf.get('space', None)
        self.tau   = best_y
        self.y_thres = y_thres
        self._num_model_constr= num_model_constr
        if not self.space:
            self._num_hard_constr = 0 
            self._num_hidden_constr = 0

        if self._num_hidden_constr > 0:
            assert isinstance(self.rf_with_thres, tuple), f'num_hiddent_constr is {self._num_hidden_constr}, the rf model and corresponding threshold should be passed'
    @property
    def num_obj(self):
        return 3

    @property
    def num_model_constr(self) -> int:
        return self._num_model_constr
    
    @property
    def num_hidden_constr(self) -> int:
        return self._num_hidden_constr

    @property
    def num_constr(self) -> int:
        return self._num_model_constr + self._num_hard_constr + self._num_hidden_constr
    
    @property
    def num_hard_constr(self) -> int:
        return self._num_hard_constr

    def eval(self, x : torch.FloatTensor, xe : torch.LongTensor) -> torch.FloatTensor:
        """
        minimize (-1 * EI,  -1 * PI, lcb)
        """
        with torch.no_grad():
            py, ps2   = self.model.predict(x, xe)

            ### objectives
            py_obj = py[:,:1]
            ps2_obj = ps2[:, :1]
            noise     = np.sqrt(2.0) * self.model.noise.sqrt()[0]
            ps_obj    = ps2_obj.sqrt().clamp(min = torch.finfo(ps2_obj.dtype).eps)
            lcb_obj   = (py_obj + noise * torch.randn(py_obj.shape)) - self.kappa * ps_obj
            normed    = ((self.tau - self.eps - py_obj - noise * torch.randn(py_obj.shape)) / ps_obj)
            dist      = Normal(0., 1.)
            log_phi   = dist.log_prob(normed)
            Phi       = dist.cdf(normed)
            PI        = Phi
            EI        = ps_obj * (Phi * normed +  log_phi.exp())
            logEIapp  = ps_obj.log() - 0.5 * normed**2 - (normed**2 - 1).log()
            logPIapp  = -0.5 * normed**2 - torch.log(-1 * normed) - torch.log(torch.sqrt(torch.tensor(2 * np.pi)))

            use_app             = ~((normed > -6) & torch.isfinite(EI.log()) & torch.isfinite(PI.log())).reshape(-1)
            out                 = torch.zeros(x.shape[0], 3)
            out[:, 0]           = lcb_obj.reshape(-1)
            out[:, 1][use_app]  = -1 * logEIapp[use_app].reshape(-1)
            out[:, 2][use_app]  = -1 * logPIapp[use_app].reshape(-1)
            out[:, 1][~use_app] = -1 * EI[~use_app].log().reshape(-1)
            out[:, 2][~use_app] = -1 * PI[~use_app].log().reshape(-1)

            ### constraints from suggorate model
            if self.y_thres is not None:
                py_constr = py[:, 1:]
                ps2_constr = ps2[:, 1:]
                ps_constr    = ps2_constr.sqrt().clamp(min = torch.finfo(ps2_constr.dtype).eps)
                out_model_constr = (py_constr + noise * torch.randn(py_constr.shape)) + self.kappa * ps_constr - self.y_thres
            else:
                out_model_constr = torch.zeros([out.size()[0], 0])
            
             # constraints from prior
            out_hard_constr = get_hard_constr(x, self.max_sequence_length,
                                              self.num_hard_constr, py, self.space)
            # constraints from rf
            out_hidden_constr = get_hidden_constr(x, xe, self.rf_with_thres,
                                                  self.num_hidden_constr, py, self.space)
            return torch.concat([out, out_model_constr, out_hard_constr, out_hidden_constr], dim=1)



class MACE(Acquisition):
    def __init__(self, model, best_y, **conf):
        super().__init__(model, **conf)
        self.kappa = conf.get('kappa', 2.0)
        self.eps   = conf.get('eps', 1e-4)
        self.tau   = best_y
    
    @property
    def num_constr(self):
        return 0

    @property
    def num_obj(self):
        return 3

    def eval(self, x : torch.FloatTensor, xe : torch.LongTensor) -> torch.FloatTensor:
        """
        minimize (-1 * EI,  -1 * PI, lcb)
        """
        with torch.no_grad():
            py, ps2   = self.model.predict(x, xe)
            noise     = np.sqrt(2.0) * self.model.noise.sqrt()
            ps        = ps2.sqrt().clamp(min = torch.finfo(ps2.dtype).eps)
            lcb       = (py + noise * torch.randn(py.shape)) - self.kappa * ps
            normed    = ((self.tau - self.eps - py - noise * torch.randn(py.shape)) / ps)
            dist      = Normal(0., 1.)
            log_phi   = dist.log_prob(normed)
            Phi       = dist.cdf(normed)
            PI        = Phi
            EI        = ps * (Phi * normed +  log_phi.exp())
            logEIapp  = ps.log() - 0.5 * normed**2 - (normed**2 - 1).log()
            logPIapp  = -0.5 * normed**2 - torch.log(-1 * normed) - torch.log(torch.sqrt(torch.tensor(2 * np.pi)))

            use_app             = ~((normed > -6) & torch.isfinite(EI.log()) & torch.isfinite(PI.log())).reshape(-1)
            out                 = torch.zeros(x.shape[0], 3)
            out[:, 0]           = lcb.reshape(-1)
            out[:, 1][use_app]  = -1 * logEIapp[use_app].reshape(-1)
            out[:, 2][use_app]  = -1 * logPIapp[use_app].reshape(-1)
            out[:, 1][~use_app] = -1 * EI[~use_app].log().reshape(-1)
            out[:, 2][~use_app] = -1 * PI[~use_app].log().reshape(-1)
            return out


class NoisyAcq(Acquisition):
    def __init__(self, model, num_obj, num_constr):
        super().__init__(model)
        self._num_obj    = num_obj
        self._num_constr = num_constr

    @property
    def num_obj(self) -> int:
        return self._num_obj

    @property
    def num_constr(self) -> int:
        return self._num_constr

    def eval(self, x : torch.FloatTensor, xe : torch.LongTensor) -> torch.FloatTensor:
        with torch.no_grad():
            y_samp = self.model.sample_y(x, xe).reshape(-1, self.num_obj + self.num_constr)
            return y_samp

class GeneralAcq(Acquisition):
    def __init__(self, model, num_obj, num_model_constr, **conf):
        super().__init__(model, **conf)
        self._num_obj    = num_obj
        self._num_model_constr = num_model_constr
        self._num_hard_constr = conf.get('num_hard_constr', 3)
        self._num_hidden_constr = conf.get('num_hidden_constr', 1)
        self.rf_with_thres = conf.get('rf_with_thres', None)
        self.space       = conf.get('space', None)
        self.kappa       = conf.get('kappa', 2.0)
        self.c_kappa     = conf.get('c_kappa', 0.)
        self.use_noise   = conf.get('use_noise', True)
        self.max_sequence_length = conf.get('max_sequence_length', 12)

        if not self.space:
            self._num_hard_constr = 0 
            self._num_hidden_constr = 0
        assert self.model.num_out == self.num_obj + self.num_model_constr
        assert self.num_obj >= 1
        if self._num_hidden_constr > 0:
            assert isinstance(self.rf_with_thres, tuple), f'num_hiddent_constr is {self._num_hidden_constr}, the rf model and corresponding threshold should be passed'

    @property
    def num_obj(self) -> int:
        return self._num_obj

    @property
    def num_model_constr(self) -> int:
        return self._num_model_constr
    
    @property
    def num_hidden_constr(self) -> int:
        return self._num_hidden_constr

    @property
    def num_constr(self) -> int:
        return self._num_model_constr+self._num_hard_constr + self._num_hidden_constr
    
    @property
    def num_hard_constr(self) -> int:
        return self._num_hard_constr

    def eval(self, x : torch.FloatTensor, xe : torch.LongTensor) -> torch.FloatTensor:
        """
        Acquisition function to deal with general constrained, multi-objective optimization problems
        
        Suppose we have $om$ objectives and $cn$ constraints, the problem should has been transformed to :

        Minimize (o1, o1, \dots,  om)
        S.t.     c1 < 0, 
                 c2 < 0, 
                 \dots
                 cb_cn < 0

        In this `GeneralAcq` acquisition function, we calculate lower
        confidence bound of objectives and constraints, and solve the following
        problem:

        Minimize (lcb_o1, lcb_o2, \dots,  lcb_om)
        S.t.     lcb_c1 < 0, 
                 lcb_c2 < 0, 
                 \dots
                 lcb_cn < 0
        """
        with torch.no_grad():
            py, ps2 = self.model.predict(x, xe)
            ps      = ps2.sqrt().clamp(min = torch.finfo(ps2.dtype).eps)
            if self.use_noise:
                noise  = self.model.noise.sqrt()
                py    += noise * torch.randn(py.shape)
            out = torch.ones(py.shape)
            out[:, :self.num_obj] = py[:, :self.num_obj]  - self.kappa   * ps[:, :self.num_obj]
            out[:, self.num_obj:] = py[:, self.num_obj:]  - self.c_kappa * ps[:, self.num_obj:]
            
             # constraints from prior
            out_hard_constr = get_hard_constr(x, self.max_sequence_length,
                                              self.num_hard_constr, py, self.space)
            # constraints from rf
            out_hidden_constr = get_hidden_constr(x, xe, self.rf_with_thres,
                                                  self.num_hidden_constr, py, self.space)
        return torch.concat([out, out_hard_constr, out_hidden_constr],dim=1)