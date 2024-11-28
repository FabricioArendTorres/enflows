import torch

from torch.nn import functional as F
from torch import nn
from torch.nn import init
from torch.func import jacfwd, jacrev
from functools import partial

import numpy as np
import sympytorch

from enflows.transforms import Transform, ConditionalTransform, Sigmoid, ScalarScale, CompositeTransform, ScalarShift
from enflows.transforms.injective.utils import sph_to_cart_jacobian_sympy, spherical_to_cartesian_torch, cartesian_to_spherical_torch, logabsdet_sph_to_car
from enflows.transforms.injective.utils import check_tensor, sherman_morrison_inverse, SimpleNN, SimpleNN_uncnstr, jacobian_sph_to_car, solve_triangular_system

import time
from datetime import timedelta
from torch.utils.benchmark import Timer


from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

import torch.autograd.functional as autograd_F
from gpytorch.utils import linear_cg



class ParamHyperFlow(Transform):
    def __init__(self):
        super().__init__()

    def f_given_x(self, x, context=None):
        raise NotImplementedError()

    def gradient_f_given_x(self, x, context=None):
        raise NotImplementedError()

    def inverse(self, x, context=None):
        f = self.f_given_x(x, context=context)
        x_f = torch.cat([x, f], dim=1)

        grad_f = self.gradient_f_given_x(x, context=context)
        logabsdet = torch.sqrt((1 + grad_f.square().sum(-1)))

        return x_f, logabsdet

    def forward(self, x_f, context=None):
        grad_f = self.gradient_f_given_x(x_f[..., :-1], context=context)
        logabsdet = torch.sqrt((1 + grad_f.square().sum(-1)))

        return x_f[..., :-1], -logabsdet


class LearnableParamHyperFlow(ParamHyperFlow):
    def __init__(self, n):
        super().__init__()

        self.network = SimpleNN_uncnstr(n, hidden_size=128, output_size=1)

    def f_given_x(self, x, context=None):
        f = self.network(x)

        return f

    def gradient_f_given_x(self, x, context=None):
        x.requires_grad_(True)
        f = self.f_given_x(x, context=context)
        grad_f_x = torch.autograd.grad(f,x, grad_outputs=torch.ones_like(f))[0]

        return grad_f_x


class ManifoldFlow(Transform):
    def __init__(self, logabs_jacobian, n_hutchinson_samples=1):
        super().__init__()
        self.n_hutchinson_samples = n_hutchinson_samples
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        assert logabs_jacobian in ["cholesky", "analytical_sm", "analytical_lu", "analytical_gauss", "analytical_block", "fff", "rect"]
        self.logabs_jacobian = logabs_jacobian

    def r_given_theta(self, theta, context=None):
        raise NotImplementedError()

    def gradient_r_given_theta(self, theta, context=None):
        raise NotImplementedError()

    def inverse(self, theta, context=None):
        r = self.r_given_theta(theta, context=context)
        theta_r = torch.cat([theta, r], dim=1)
        # breakpoint()
        # print("theta", theta.min().item(), theta.max().item())
        outputs = spherical_to_cartesian_torch(theta_r)

        # print(outputs.min().item(), outputs.max().item())
        if self.logabs_jacobian == "analytical_sm":
            logabsdet = self.logabs_jacobian_analytical_sm(theta, theta_r, context=context)
        elif self.logabs_jacobian == "analytical_lu":
            logabsdet = self.logabs_jacobian_analytical_lu(theta, theta_r, context=context)
        elif self.logabs_jacobian == "analytical_gauss":
            logabsdet = self.logabs_jacobian_analytical_gauss(theta, theta_r, context=context)
        elif self.logabs_jacobian == "analytical_block":
            logabsdet = self.logabs_jacobian_analytical_block(theta, theta_r, context=context)
        elif self.logabs_jacobian == "cholesky":
            logabsdet = self.logabs_jacobian_cholesky(theta, theta_r, context=context)
        elif self.logabs_jacobian == "fff":
            logabsdet = self.logabs_jacobian_fff_inverse(theta=theta, context=context)
        elif self.logabs_jacobian == "rect":
            logabsdet = self.logabs_jacobian_conjgrad(theta, context=context)
        else:
            raise ValueError(f"logabs_jacobian {self.logabs_jacobian} is not a valid choice")
        # print("logabsdet", logabsdet.min().item(), logabsdet.max().item())

        return outputs, logabsdet

    def forward(self, inputs, context=None):
        # print("forward")
        outputs = cartesian_to_spherical_torch(inputs)

        if self.logabs_jacobian == "analytical_sm":
            logabsdet = self.logabs_jacobian_analytical_sm(outputs[:,:-1], outputs, context=context)
        elif self.logabs_jacobian == "analytical_lu":
            logabsdet = self.logabs_jacobian_analytical_lu(outputs[:,:-1], outputs, context=context)
        elif self.logabs_jacobian == "analytical_gauss":
            logabsdet = self.logabs_jacobian_analytical_gauss(outputs[:,:-1], outputs, context=context)
        elif self.logabs_jacobian == "analytical_block":
            logabsdet = self.logabs_jacobian_analytical_block(outputs[:, :-1], outputs, context=context)
        elif self.logabs_jacobian == "cholesky":
            logabsdet = self.logabs_jacobian_cholesky(outputs[:,:-1], outputs, context=context)
        elif self.logabs_jacobian == "fff":
            logabsdet = self.logabs_jacobian_fff_forward(x=inputs, context=context)
        elif self.logabs_jacobian == "rect":
            logabsdet = self.logabs_jacobian_conjgrad(outputs[:,:-1], context=context)
        else:
            raise ValueError(f"logabs_jacobian {self.logabs_jacobian} is not a valid choice")

        return outputs[..., :-1], -logabsdet

    def compute_jacobian_row(self, output, input):
        assert output.shape[0] == input.shape[0]
        output = output.view(output.shape[0], -1)

        # Compute Jacobian row by row.
        jac = []
        for j in range(output.shape[1]):
            dy_j_dx = torch.autograd.grad(output[:, j], input, torch.ones_like(output[:, j]), retain_graph=True,
                                          create_graph=True)[0].view(input.shape[0], -1)
            jac.append(torch.unsqueeze(dy_j_dx, 1))
        jac = torch.cat(jac, 1)
        return jac

    def logabs_jacobian_analytical_sm(self, theta, theta_r, context=None):
        eps = 1e-8
        # jac = torch.autograd.functional.jacobian(spherical_to_cartesian_torch, theta_r).sum(-2)
        # jac = jacrev(spherical_to_cartesian_torch)(theta_r).sum(-2)
        # jac = jacfwd(spherical_to_cartesian_torch)(theta_r).sum(-2)
        # cartesian = spherical_to_cartesian_torch(theta_r)
        # jac = self.compute_jacobian_row(cartesian, theta_r)
        # jac = jacrev(spherical_to_cartesian_torch)(theta_r).sum(-2)
        cartesian = spherical_to_cartesian_torch(theta_r)
        jac = jacobian_sph_to_car(theta_r, cartesian)


        grad_r = self.gradient_r_given_theta(theta, context=context)

        jac_inv = sherman_morrison_inverse(jac.mT)
        jac_inv_grad = jac_inv @ grad_r

        fro_norm = torch.norm(jac_inv_grad.squeeze(-1), p='fro', dim=1)

        logabsdet_fro_norm = torch.log(fro_norm + eps)
        logabsdet_s_to_c = logabsdet_sph_to_car(theta_r)
        logabsdet = logabsdet_s_to_c + logabsdet_fro_norm

        return logabsdet

    def logabs_jacobian_analytical_gauss(self, theta, theta_r, context=None):
        eps = 1e-8
        cartesian = spherical_to_cartesian_torch(theta_r)
        jac = jacobian_sph_to_car(theta_r, cartesian)

        grad_r = self.gradient_r_given_theta(theta, context=context)

        jac_triang, vector = self.one_step_gaussian_elimination(jac.mT, grad_r)
        # jac_triang = torch.linalg.lu(jac.mT, grad_r)
        jac_inv_grad = torch.linalg.solve_triangular(jac_triang.triu(), grad_r, upper=True)
        fro_norm = torch.norm(jac_inv_grad.squeeze(-1), p='fro', dim=1)

        logabsdet_fro_norm = torch.log(fro_norm + eps)
        logabsdet_s_to_c = logabsdet_sph_to_car(theta_r)
        logabsdet = logabsdet_s_to_c + logabsdet_fro_norm

        return logabsdet

    def one_step_gaussian_elimination(self, matrix, vector):

        for row_idx in range(matrix.shape[-1]-1):
            coeff = matrix[:, -1, row_idx] / matrix[:, row_idx, row_idx]
            matrix[:, -1] -= coeff.unsqueeze(-1) * matrix[:, row_idx]
            vector[:, -1] -= coeff.unsqueeze(-1) * vector[:, row_idx]

        return matrix, vector

    def logabs_jacobian_analytical_lu(self, theta, theta_r, context=None):
        eps = 1e-8
        # jac = torch.autograd.functional.jacobian(spherical_to_cartesian_torch, theta_r).sum(-2)
        # jac = jacrev(spherical_to_cartesian_torch)(theta_r).sum(-2)
        # jac = jacfwd(spherical_to_cartesian_torch)(theta_r).sum(-2)
        # cartesian = spherical_to_cartesian_torch(theta_r)
        # jac = self.compute_jacobian_row(cartesian, theta_r)
        # jac = jacrev(spherical_to_cartesian_torch)(theta_r).sum(-2)
        cartesian = spherical_to_cartesian_torch(theta_r)
        jac = jacobian_sph_to_car(theta_r, cartesian)

        grad_r = self.gradient_r_given_theta(theta, context=context)

        jac = jac + torch.eye(jac.shape[-1], device=theta.device).unsqueeze(0) * eps
        LU, pivots = torch.linalg.lu_factor(jac)
        # LU, pivots = torch.linalg.lu_factor(jac.mT)
        # jac_inv_grad = torch.linalg.lu_solve(LU, pivots, grad_r).squeeze()
        jac_inv_grad = torch.linalg.lu_solve(LU, pivots, grad_r, adjoint=True).squeeze(-1)
        fro_norm = torch.norm(jac_inv_grad.squeeze(-1), p='fro', dim=1)

        logabsdet_fro_norm = torch.log(fro_norm + eps)
        logabsdet_s_to_c = logabsdet_sph_to_car(theta_r)

        logabsdet = logabsdet_s_to_c + logabsdet_fro_norm

        # logabsdet = None
        return logabsdet

    def logabs_jacobian_analytical_block(self, theta, theta_r, context=None):
        eps = 1e-8
        # jac = torch.autograd.functional.jacobian(spherical_to_cartesian_torch, theta_r).sum(-2)
        # jac = jacrev(spherical_to_cartesian_torch)(theta_r).sum(-2)
        # jac = jacfwd(spherical_to_cartesian_torch)(theta_r).sum(-2)
        # cartesian = spherical_to_cartesian_torch(theta_r)
        # jac = self.compute_jacobian_row(cartesian, theta_r)
        # jac = jacrev(spherical_to_cartesian_torch)(theta_r).sum(-2)
        cartesian = spherical_to_cartesian_torch(theta_r)
        jac = jacobian_sph_to_car(theta_r, cartesian)

        grad_r = self.gradient_r_given_theta(theta, context=context)
        jacobian = jac.mT
        y = grad_r[:,:-1]
        A = jacobian[:,:-1,:-1]
        B = jacobian[:,:-1,-1:]
        C = jacobian[:,-1:,:-1]
        D = jacobian[:,-1:,-1:]

        # compute (D - C A^-1 B)
        scalar_DCAB = (D - C @ torch.linalg.solve_triangular(A, B, upper=True))

        # compute top left element (11)
        Ay = torch.linalg.solve_triangular(A, y, upper=True)
        S_11 = Ay * scalar_DCAB + torch.linalg.solve_triangular(A, B @ C @ Ay, upper=True)
        S_12 = - torch.linalg.solve_triangular(A, B, upper=True)
        S_21 = - C @ torch.linalg.solve_triangular(A, y, upper=True)
        S_22 = torch.ones_like(y[:,-1:])

        jac_inv_grad = torch.cat((S_11 + S_12, S_21 + S_22), dim=-2).squeeze() / scalar_DCAB.squeeze(-2)

        fro_norm = torch.norm(jac_inv_grad.squeeze(-1), p='fro', dim=1)
        logabsdet_fro_norm = torch.log(fro_norm + eps)
        logabsdet_s_to_c = logabsdet_sph_to_car(theta_r)

        logabsdet = logabsdet_s_to_c + logabsdet_fro_norm

        # logabsdet = None
        return logabsdet

    # def logabs_jacobian_cholesky(self, theta, theta_r, context=None):
    #
    #     def theta_cartesian(theta, context):
    #         r = self.r_given_theta(theta, context=context)
    #         theta_r = torch.cat([theta, r], dim=1)
    #         outputs = spherical_to_cartesian_torch(theta_r)
    #         return outputs
    #
    #     eps = 1e-8
    #     jac_forward = jacfwd(partial(theta_cartesian, context=context))(theta).sum(-2)
    #     # jac_forward = torch.autograd.functional.jacobian(partial(theta_cartesian, context=context), theta).sum(-2)
    #
    #     jac_full = jac_forward.mT @ jac_forward
    #     logabsdet = 0.5 * torch.logdet(jac_full)
    #     # jac_full_eye = torch.diag_embed(jac_full.new_ones(jac_full.shape[-1]))
    #     # jac_full = jac_full + jac_full_eye * eps
    #     # jac_full_lower = torch.linalg.cholesky(jac_full)
    #     # jac_full_lower_diag = torch.diagonal(jac_full_lower, dim1=1, dim2=2)
    #     # logabsdet = torch.log(jac_full_lower_diag).sum(1) # should be 2* but there is also a 0.5 factor
    #
    #     return logabsdet

    def logabs_jacobian_cholesky(self, theta, theta_r, context=None):
        eps = 1e-8
        # jac_sph_cart = torch.autograd.functional.jacobian(spherical_to_cartesian_torch, theta_r).sum(-2)
        # jac_sph_cart = jacrev(spherical_to_cartesian_torch)(theta_r).sum(-2)
        cartesian = spherical_to_cartesian_torch(theta_r)
        jac_sph_cart = jacobian_sph_to_car(theta_r, cartesian)

        # jac_sph_cart_ = jacfwd(spherical_to_cartesian_torch)(theta_r).sum(-2)

        # jac_r_theta = self.compute_jacobian_row(theta_r, theta)
        eye = torch.eye(theta.shape[-1], device=theta.device).unsqueeze(0).expand(theta.shape[0], -1, -1)
        # r_given_theta = partial(self.r_given_theta, context=context)
        grad_r = self.gradient_r_given_theta(theta, context=context)
        jac_r_theta = torch.cat((eye, grad_r[:, :-1].squeeze(-1).unsqueeze(1)), dim=1)
        # jac_r_theta = torch.autograd.functional.jacobian(r_given_theta, theta).sum(-2)
        # jac_r_theta = jacfwd(r_given_theta)(theta).sum(-2)
        # jac_r_theta = self.compute_jacobian_row(theta_r, theta)
        # grad_r = self.gradient_r_given_theta(theta, context=context)

        # torch.cuda.synchronize()
        # start_time = time.monotonic()


        jac_forward = jac_sph_cart @ jac_r_theta

        # we need to compute 0.5 * sqrt(ac_forward.T @ jac_forward)
        # naively we could compute it as logabsdet2 = 0.5 * torch.logdet(jac_full)
        # instead, we use cholesky decomposition to compute the determinant in O(d^3)
        # alternative: use torch.logdet

        jac_full = jac_forward.mT @ jac_forward

        # lower = torch.rand_like(jac_full).triu()
        # jac_full = lower @ lower.mT

        jac_full_eye = torch.diag_embed(jac_full.new_ones(jac_full.shape[-1]))
        jac_full = jac_full + jac_full_eye * eps
        # jac_full = jac_full + jac_full_eye
        jac_full_lower = torch.linalg.cholesky(jac_full)
        jac_full_lower_diag = torch.diagonal(jac_full_lower, dim1=1, dim2=2)
        logabsdet = torch.log(jac_full_lower_diag).sum(1) # should be 2* but there is also a 0.5 factor

        # torch.cuda.synchronize()
        # end_time = time.monotonic()
        # time_diff = timedelta(seconds=end_time - start_time)
        # time_diff_seconds = str(time_diff).split(":")[-1]
        # print(time_diff)

        return logabsdet

    def sample_v(self, x, hutchinson_samples):
        batch_size, total_dim = x.shape[0], np.prod(x.shape[1:])
        if hutchinson_samples > total_dim:
           raise ValueError("Too many Hutchinson samples: got {hutchinson_samples}, expected <= {total_dim}")

        v = torch.randn(batch_size, total_dim, hutchinson_samples, device=x.device, dtype=x.dtype)
        # v = torch.rand(batch_size, total_dim, hutchinson_samples, device=x.device, dtype=x.dtype) * torch.pi
        q = torch.linalg.qr(v).Q.reshape(*x.shape, hutchinson_samples)
        return q * np.sqrt(total_dim)

    def logabs_jacobian_fff_forward(self, x, hutchinson_samples=1, context=None):

        def sum_except_batch(x):
            """Sum over all dimensions except the first.
            :param x: Input tensor. Shape: (batch_size, ...)
            :return: Sum over all dimensions except the first. Shape: (batch_size,)
            """
            return torch.sum(x.reshape(x.shape[0], -1), dim=1)

        surrogate = 0
        # hutchinson_samples = 10 #x.shape[-1] - 1

        x.requires_grad_()
        theta = cartesian_to_spherical_torch(x)[..., :-1]

        vs = self.sample_v(theta, self.n_hutchinson_samples)

        for k in range(hutchinson_samples):
            v = vs[..., k]

            # $ g'(z) v $ via forward-mode AD
            with dual_level():
                dual_z = make_dual(theta, v)

                radius = self.r_given_theta(dual_z, context=context)

                theta_r = torch.cat([dual_z, radius], dim=1)
                dual_x1 = spherical_to_cartesian_torch(theta_r)
                x1, v1 = unpack_dual(dual_x1)
                # breakpoint()

            # $ v^T f'(x) $ via backward-mode AD
            (v2,) = torch.autograd.grad(theta, x, v, create_graph=True)
            # $ v^T f'(x) stop_grad(g'(z)) v $
            surrogate += sum_except_batch(v2 * v1.detach()) / hutchinson_samples
            # surrogate += sum_except_batch(v2 * v1) / hutchinson_samples
            # print(surrogate)

        return surrogate

    def logabs_jacobian_fff_inverse(self, theta, hutchinson_samples=1, context=None):

        def sum_except_batch(x):
            """Sum over all dimensions except the first.
            :param x: Input tensor. Shape: (batch_size, ...)
            :return: Sum over all dimensions except the first. Shape: (batch_size,)
            """
            return torch.sum(x.reshape(x.shape[0], -1), dim=1)


        surrogate = 0

        theta.requires_grad_()
        r = self.r_given_theta(theta, context=context)
        theta_r = torch.cat([theta, r], dim=1)
        # breakpoint()
        # print("theta", theta.min().item(), theta.max().item())
        x = spherical_to_cartesian_torch(theta_r)

        vs = self.sample_v(x, self.n_hutchinson_samples)

        for k in range(hutchinson_samples):
            v = vs[..., k]

            # $ g'(z) v $ via forward-mode AD
            with dual_level():
                dual_x = make_dual(x, v)

                dual_theta = cartesian_to_spherical_torch(dual_x)[...,:-1]
                x1, v1 = unpack_dual(dual_theta)
                # breakpoint()

            # $ v^T f'(x) $ via backward-mode AD
            (v2,) = torch.autograd.grad(x, theta, v, create_graph=True)
            # $ v^T f'(x) stop_grad(g'(z)) v $
            surrogate += sum_except_batch(v2 * v1.detach()) / hutchinson_samples
            # surrogate += sum_except_batch(v2 * v1) / hutchinson_samples
            # print(surrogate)

        return surrogate


    def logabs_jacobian_conjgrad(self, latent, num_hutchinson_samples=3, context=None):

        sample_shape = (*latent.shape, num_hutchinson_samples)
        hutchinson_distribution = "normal"
        max_cg_iterations = None
        cg_tolerance = None
        training = True

        if hutchinson_distribution == "normal":
            hutchinson_samples = torch.randn(*sample_shape, device=latent.device)
        elif hutchinson_distribution == "rademacher":
            bernoulli_probs = 0.5 * torch.ones(*sample_shape, device=latent.device)
            hutchinson_samples = torch.bernoulli(bernoulli_probs)
            hutchinson_samples.mul_(2.).subtract_(1.)

        repeated_latent = latent.repeat_interleave(num_hutchinson_samples, dim=0)

        def function(theta):
            r = self.r_given_theta(theta, context=context)
            theta_r = torch.cat([theta, r], dim=1)
            # breakpoint()
            # print("theta", theta.min().item(), theta.max().item())
            outputs = spherical_to_cartesian_torch(theta_r)
            return outputs

        def jvp_forward(x, v, function):
            with dual_level():
                inp = make_dual(x, v)
                out = function(inp)
                y, jvp = unpack_dual(out)

            return y, jvp

        def jac_transpose_jac_vec(latent, vec, create_graph, function):
            if not create_graph:
                latent = latent.detach().requires_grad_(False)
                with torch.no_grad():
                    y, jvp = jvp_forward(latent, vec, function)
            else:
                y, jvp = jvp_forward(latent, vec, function)

            flow_forward_flat = lambda x: function(x).flatten(start_dim=1)  # possible mistake here
            _, jtjvp = autograd_F.vjp(flow_forward_flat, latent, jvp.flatten(start_dim=1), create_graph=create_graph)

            return jtjvp, y

        def tensor_to_vector(tensor):
            # Turn a tensor of shape (batch_size x latent_dim x num_hutch_samples)
            # into a vector of shape (batch_size*num_hutch_samples x latent_dim)
            # NOTE: Need to transpose first to get desired stacking from reshape
            vector = tensor.transpose(1, 2).reshape(
                latent.shape[0] * num_hutchinson_samples, latent.shape[1]
            )
            return vector

        def vector_to_tensor(vector):
            # Inverse of `tensor_to_vector` above
            # NOTE: Again need to transpose to correctly unfurl num_hutch_samples as the final dimension
            tensor = vector.reshape(latent.shape[0], num_hutchinson_samples, latent.shape[1])
            return tensor.transpose(1, 2)

        def jac_transpose_jac_closure(tensor, function):
            # NOTE: The CG method available to us expects a method to multiply against
            #       tensors of shape (batch_size x latent_dim x num_hutch_samples).
            #       Thus we need to wrap reshaping around our JtJv method,
            #       which expects v to be of shape (batch_size*num_hutch_samples x latent_dim).
            vec = tensor_to_vector(tensor)
            jtjvp, _ = jac_transpose_jac_vec(repeated_latent, vec, create_graph=False, function=function)
            return vector_to_tensor(jtjvp)

        jtj_inverse_hutchinson = linear_cg(
            partial(jac_transpose_jac_closure, function=function),
            hutchinson_samples,
            max_iter=max_cg_iterations,
            max_tridiag_iter=max_cg_iterations,
            tolerance=cg_tolerance
        ).detach()

        jtj_hutchinson_vec, reconstruction_repeated = jac_transpose_jac_vec(
            repeated_latent, tensor_to_vector(hutchinson_samples), create_graph=training, function=function
        )
        reconstruction = reconstruction_repeated[::num_hutchinson_samples]
        jtj_hutchinson = vector_to_tensor(jtj_hutchinson_vec)

        # NOTE: jtj_inverse does not just cancel out with jtj because the former has a stop gradient applied.
        approx_log_det_jac = torch.mean(torch.sum(jtj_inverse_hutchinson * jtj_hutchinson, dim=1, keepdim=True), dim=2)

        return approx_log_det_jac.squeeze()
    # def _jac_transpose_jac_vec(self, latent, vec, create_graph):
    #     if not create_graph:
    #         latent = latent.detach().requires_grad_(False)
    #         with torch.no_grad():
    #             reconstruction, jvp = self.jvp_forward(latent, vec)
    #     else:
    #         reconstruction, jvp = self.jvp_forward(latent, vec)
    #
    #     flow_forward_flat = lambda x: self.flow_forward(x).flatten(start_dim=1)
    #     _, jtjvp = autograd_F.vjp(flow_forward_flat, latent, jvp.flatten(start_dim=1), create_graph=create_graph)
    #
    #     return jtjvp, reconstruction
    #
    #
    # def jvp_forward(self, x, v):
    #     jvp_stack_copy = self.jvp_stack[:]
    #
    #     while jvp_stack_copy:
    #         jvp_fn = jvp_stack_copy.pop()
    #         jvp_out = jvp_fn(x, v)
    #         x, v = jvp_out["x"], jvp_out["jvp"]
    #     return x, v
    #
    # def _traverse_backward(self, x, prior_dict):
    #     """
    #     This function traverses backward through the transformations defining the flow.
    #     It outputs the low-dim latent variable and its log likelihood.
    #     It also modifies self.transform_stack and self.jvp_stack for self.flow_forward and
    #     self.jvp_forward, respectively.
    #     """
    #     transform_stack = []
    #     jvp_stack = []
    #     prior_pointer = self.prior
    #
    #     while "low-dim-x" not in prior_dict:
    #         prior_dict = prior_dict["prior-dict"]
    #         jvp_stack.append(prior_pointer.jvp)
    #
    #         transform = prior_pointer.bijection.z_to_x
    #         prior_pointer = prior_pointer.prior
    #
    #         transform_stack.append(transform)
    #
    #     jvp_stack.append(prior_pointer.jvp)
    #     transform_stack.append(prior_pointer.low_dim_to_masked)
    #     self._set_flow_and_jvp_stacks(transform_stack, jvp_stack)
    #
    #     low_dim_latent = prior_dict["low-dim-x"]
    #     low_dim_elbo = prior_dict["elbo"]
    #
    #     try:
    #         earliest_latent = prior_pointer.extract_latent(low_dim_latent)
    #     except NotImplementedError:
    #         earliest_latent = ""
    #
    #     return low_dim_latent, low_dim_elbo, earliest_latent
    #
    #
    # def logabs_jacobian_approx(self, latent):
    #
    #     num_hutchinson_samples = 100
    #     hutchinson_distribution = "normal"
    #     max_cg_iterations = 10
    #     cg_tolerance = 1e-8
    #     sample_shape = (*latent.shape, num_hutchinson_samples)
    #
    #     if hutchinson_distribution == "normal":
    #         hutchinson_samples = torch.randn(*sample_shape, device=latent.device)
    #     elif hutchinson_distribution == "rademacher":
    #         bernoulli_probs = 0.5*torch.ones(*sample_shape, device=latent.device)
    #         hutchinson_samples = torch.bernoulli(bernoulli_probs)
    #         hutchinson_samples.mul_(2.).subtract_(1.)
    #     else:
    #         raise ValueError(f"Unknown hutchinson distribution {self.hutchinson_distribution}")
    #
    #     repeated_latent = latent.repeat_interleave(num_hutchinson_samples, dim=0)
    #
    #     def tensor_to_vector(tensor, num_hutchinson_samples):
    #         # Turn a tensor of shape (batch_size x latent_dim x num_hutch_samples)
    #         # into a vector of shape (batch_size*num_hutch_samples x latent_dim)
    #         # NOTE: Need to transpose first to get desired stacking from reshape
    #         vector = tensor.transpose(1,2).reshape(
    #             latent.shape[0]*num_hutchinson_samples, latent.shape[1]
    #         )
    #         return vector
    #
    #     def vector_to_tensor(vector, num_hutchinson_samples):
    #         # Inverse of `tensor_to_vector` above
    #         # NOTE: Again need to transpose to correctly unfurl num_hutch_samples as the final dimension
    #         tensor = vector.reshape(latent.shape[0], num_hutchinson_samples, latent.shape[1])
    #         return tensor.transpose(1,2)
    #
    #     def jac_transpose_jac_closure(tensor, num_hutchinson_samples):
    #         # NOTE: The CG method available to us expects a method to multiply against
    #         #       tensors of shape (batch_size x latent_dim x num_hutch_samples).
    #         #       Thus we need to wrap reshaping around our JtJv method,
    #         #       which expects v to be of shape (batch_size*num_hutch_samples x latent_dim).
    #         vec = tensor_to_vector(tensor, num_hutchinson_samples)
    #         jtjvp, _ = self._jac_transpose_jac_vec(repeated_latent, vec, create_graph=False)
    #         return vector_to_tensor(jtjvp, num_hutchinson_samples)
    #
    #     jtj_inverse_hutchinson = linear_cg(
    #         jac_transpose_jac_closure,
    #         hutchinson_samples,
    #         max_iter=max_cg_iterations,
    #         max_tridiag_iter=max_cg_iterations,
    #         tolerance=cg_tolerance
    #     ).detach()
    #
    #     jtj_hutchinson_vec, reconstruction_repeated = self._jac_transpose_jac_vec(
    #         repeated_latent, tensor_to_vector(hutchinson_samples, num_hutchinson_samples), create_graph=self.training
    #     )
    #     reconstruction = reconstruction_repeated[::num_hutchinson_samples]
    #     jtj_hutchinson = vector_to_tensor(jtj_hutchinson_vec, num_hutchinson_samples)
    #
    #     # NOTE: jtj_inverse does not just cancel out with jtj because the former has a stop gradient applied.
    #     approx_log_det_jac = torch.mean(torch.sum(jtj_inverse_hutchinson*jtj_hutchinson, dim=1, keepdim=True), dim=2)
    #
    #     return approx_log_det_jac, reconstruction

    def _initialize_jacobian(self, inputs):
        spherical_names, jac = sph_to_cart_jacobian_sympy(inputs.shape[1] + 1)
        self.spherical_names = spherical_names
        self.sph_to_cart_jac = sympytorch.SymPyModule(expressions=jac).to(inputs.device)
        self.initialized.data = torch.tensor(True, dtype=torch.bool)

from siren_pytorch import SirenNet
class LearnableManifoldFlow(ManifoldFlow):
    def __init__(self, n, logabs_jacobian, max_radius=2., n_hutchinson_samples=1):
        super().__init__(logabs_jacobian=logabs_jacobian, n_hutchinson_samples=n_hutchinson_samples)

        # self.network = SimpleNN(n, hidden_size=256, output_size=1, max_radius=max_radius)
        # self.network = SirenNet(dim_in=n, dim_hidden=256, dim_out = 1, num_layers = 5,
        #                         final_activation = torch.nn.Sigmoid(), w0_initial = 30.)
        self.network = nn.Sequential(
            nn.Linear(n, 1),
            nn.Sigmoid()
        )

    def r_given_theta(self, theta, context=None):
        r = self.network(theta)

        return r

    def gradient_r_given_theta(self, theta, context=None):
        theta.requires_grad_(True)
        r = self.r_given_theta(theta, context=context)
        grad_r_theta = torch.autograd.grad(r,theta, grad_outputs=torch.ones_like(r), create_graph=True)[0]
        grad_r_theta_aug = torch.cat([- grad_r_theta, torch.ones_like(grad_r_theta[:, :1])], dim=1)

        # check_tensor(grad_r_theta)

        return grad_r_theta_aug.unsqueeze(-1)


class SphereFlow(ManifoldFlow):
    def __init__(self, n, logabs_jacobian, r=1., n_hutchinson_samples=1):
        super().__init__(logabs_jacobian=logabs_jacobian, n_hutchinson_samples=n_hutchinson_samples)
        self.radius = r
        # self.network = SimpleNN(n, hidden_size=50, output_size=1, max_radius=max_radius)

    def r_given_theta(self, theta, context=None):
        r = theta.new_ones(theta.shape[0], 1) * self.radius
        # r = self.network(theta)

        return r

    def gradient_r_given_theta(self, theta, context=None):
        # theta.requires_grad_(True)
        # r = self.r_given_theta(theta, context=context)
        # grad_r_theta = torch.autograd.grad(r,theta, grad_outputs=torch.ones_like(r))[0]
        grad_r_theta = torch.zeros_like(theta)
        grad_r_theta_aug = torch.cat([- grad_r_theta, torch.ones_like(grad_r_theta[:, :1])], dim=1)

        check_tensor(grad_r_theta)

        return grad_r_theta_aug.unsqueeze(-1)

class DeformedSphereFlow(ManifoldFlow):
    def __init__(self, logabs_jacobian, r=1., manifold_type=1, n_hutchinson_samples=1):
        super().__init__(logabs_jacobian=logabs_jacobian, n_hutchinson_samples=n_hutchinson_samples)
        self.radius = r
        self.manifold_type = manifold_type
        # self.network = SimpleNN(n, hidden_size=50, output_size=1, max_radius=max_radius)



    def r_given_theta(self, angles, context=None):
        assert angles.shape[-1] == 2
        theta = angles[:,:1]
        phi = angles[:,1:2]
        r = theta.new_ones(theta.shape[0], 1) * self.radius
        # r = self.network(theta)
        match self.manifold_type:
            case 0:
                #ellipsoid
                a, b, c, = 1, 0.9, 0.8
                r = 1. / torch.sqrt( ( (torch.cos(phi)/a)**2 + (torch.sin(phi)/b)**2 ) * torch.sin(theta)**2 + (torch.cos(theta)/c)**2)
            case 1:
                # heart-like
                r = 1 + 0.05 * torch.sin(5 * theta) ** 2 + 0.2 * torch.sin(theta) * torch.abs(torch.sin(phi)) * torch.cos(phi)
            case 2:
                # weird oscillations
                r = 0.6 + 0.05 * torch.sin(5 * theta) ** 2 + 0.05 * torch.sin(3 * theta) * torch.sin(2 * phi)
            case 3:
                # ridge squash
                focus_equator = 5
                num_ridges = 6
                focus_ridges = 3
                r = 0.6 + 0.5 * torch.exp(- focus_equator * (1.0 - torch.sin(theta))) + 0.3 * torch.sin(
                    num_ridges * phi) * torch.exp(
                    - focus_ridges * (1.0 - torch.sin(theta)))
            case 4:
                # spirals
                spiral_speed = 3
                num_spirals = 3  # this times 2
                spiral_power = 2
                focus_equator = 2
                amplitude = 0.15
                r = 1.0 + amplitude * torch.exp(- focus_equator * (1.0 - torch.sin(theta))) * torch.sin(
                    num_spirals * phi + spiral_speed * theta) ** spiral_power
            case 5:
                # spiral spikes
                spiral_speed = 5
                num_spirals = 4  # this times 2
                spiral_power = 2
                focus_equator = 3
                spike_separation = 7  # this number of spikes per spiral. should be odd
                spike_power = 1.0
                amplitude = 0.4
                r = 1.0 + amplitude * torch.exp(- focus_equator * (1.0 - torch.sin(theta))) * torch.abs(
                    torch.sin(spike_separation * theta)) ** spike_power * torch.sin(
                    num_spirals * phi + spiral_speed * theta) ** spiral_power

        return r

    def gradient_r_given_theta(self, theta, context=None):
        theta.requires_grad_(True)
        r = self.r_given_theta(angles=theta, context=context).squeeze()
        # r = self.r_given_theta(theta, context=context)
        grad_r_theta = torch.autograd.grad(r,theta, grad_outputs=torch.ones_like(r))[0]
        # grad_r_theta = torch.zeros_like(theta)
        grad_r_theta_aug = torch.cat([- grad_r_theta, torch.ones_like(grad_r_theta[:, :1])], dim=1)

        check_tensor(grad_r_theta)
        # breakpoint()

        # return grad_r_theta_aug.unsqueeze(-1)
        return grad_r_theta_aug.unsqueeze(-1)

class LpManifoldFlow(ManifoldFlow):
    def __init__(self, norm, p, logabs_jacobian, given_radius=1.):
        super().__init__(logabs_jacobian=logabs_jacobian)
        self.norm = norm
        self.p = p
        assert given_radius > 0, "radius must be positive"
        self.given_radius = given_radius
        # self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def r_given_theta(self, theta, context=None):
        assert theta.shape[1] >= 2
        eps = 1e-10

        r_theta = torch.cat((theta, torch.ones_like(theta[:,:1])), dim=1)
        cartesian = spherical_to_cartesian_torch(r_theta)
        p_norm = torch.linalg.vector_norm(cartesian, ord=self.p, dim=1)
        r = self.norm / (p_norm + eps) * self.given_radius

        return r.unsqueeze(-1)

    def r_given_theta_(self, theta, context=None):
        if context is None:
            given_norm = self.norm
        else:
            # the first element of context is assumed to be the norm value
            given_norm = context[:, 0]

        assert theta.shape[1] >= 1
        eps = 1e-8
        n = theta.shape[-1]

        sin_thetas = torch.sin(theta)
        cos_thetas = torch.cos(theta)

        norm_1 = (torch.abs(cos_thetas[:, 0]) + eps) ** self.p
        check_tensor(norm_1)
        norm_3 = (torch.abs(torch.prod(sin_thetas, dim=-1)) + eps) ** self.p
        check_tensor(norm_3)


        if theta.shape[1] == 1:
            r = given_norm / ((norm_1 + norm_3 + eps) ** (1. / self.p))
            check_tensor(r)

            return r.unsqueeze(-1)
        else:
            norm_2_ = [(torch.abs(torch.prod(sin_thetas[..., :k - 1], dim=-1) * cos_thetas[..., k - 1]) + eps) ** self.p
                       for k in range(2, n + 1)]
            norm_2 = torch.stack(norm_2_, dim=1).sum(-1)
            check_tensor(norm_2)
            r = given_norm / ((norm_1 + norm_2 + norm_3 + eps) ** (1. / self.p))
            check_tensor(r)

            return r.unsqueeze(-1)

    def gradient_r_given_theta(self, theta, context=None):
        theta.requires_grad_(True)
        r = self.r_given_theta(theta=theta, context=context).squeeze()
        grad_r_theta = torch.autograd.grad(r, theta, grad_outputs=torch.ones_like(r))[0]
        grad_r_theta_aug = torch.cat([- grad_r_theta, torch.ones_like(grad_r_theta[:, :1])], dim=1)

        # check_tensor(grad_r_theta)
        # print("gradient_shape", grad_r_theta.shape, grad_r_theta_aug.unsqueeze(-1).shape)
        return grad_r_theta_aug.unsqueeze(-1)

class CondLpManifoldFlow(LpManifoldFlow):
    def __init__(self, norm, p, logabs_jacobian):
        super().__init__(norm, p, logabs_jacobian)

    def r_given_theta(self, theta, context=None):
        if context is None:
            given_norm = self.norm
        else:
            # the first element of context is assumed to be the norm value
            given_norm = context[:, 0]

        assert theta.shape[1] > 2
        eps = 1e-10

        r_theta = torch.cat((theta, torch.ones_like(theta[:,:1])), dim=1)
        cartesian = spherical_to_cartesian_torch(r_theta)
        p_norm = torch.linalg.vector_norm(cartesian, ord=self.p, dim=1)
        r = given_norm / (p_norm + eps)

        return r.unsqueeze(-1)


class PositiveL1ManifoldFlow(ManifoldFlow):
    def __init__(self, logabs_jacobian):
        super().__init__(logabs_jacobian=logabs_jacobian)

    def r_given_theta(self, theta, context=None):
        assert theta.shape[1] >= 2
        assert torch.all(theta >= 0) and torch.all(theta <= 0.5 * torch.pi)

        r_theta = torch.cat((theta, torch.ones_like(theta[:,:1])), dim=1)
        cartesian = spherical_to_cartesian_torch(r_theta)
        r = 1 / cartesian.sum(-1)

        return r.unsqueeze(-1)

    def gradient_r_given_theta(self, theta, context=None):
        r_theta = torch.cat((theta, torch.ones_like(theta[:, :1])), dim=1)
        cartesian = spherical_to_cartesian_torch(r_theta)


        mb, dim = r_theta.shape
        temp = torch.ones(mb, dim, dim-1, device=r_theta.device).tril()
        cos_sin = torch.cos(theta) / torch.sin(theta)
        temp = temp * cos_sin.reshape(mb, 1, dim-1)
        temp = torch.diagonal_scatter(temp, -torch.reciprocal(cos_sin), dim1=1, dim2=2)

        grad_den = (temp * cartesian.reshape(mb, dim, 1)).sum(-2)
        grad_r_theta = - grad_den / (cartesian.sum(-1).reshape(mb, 1) ** 2)

        # alternatively, grad_r_theta can be computed with autograd
        # radius = self.r_given_theta(theta=theta, context=context).squeeze()
        # grad_r_theta_ = torch.autograd.grad(radius, theta, grad_outputs=torch.ones_like(radius))[0]
        grad_r_theta_aug = torch.cat([- grad_r_theta, torch.ones_like(grad_r_theta[:, :1])], dim=1)

        # check_tensor(grad_r_theta)
        # print("gradient_shape", grad_r_theta.shape, grad_r_theta_aug.unsqueeze(-1).shape)
        return grad_r_theta_aug.unsqueeze(-1)

class PeriodicElementwiseTransform(Transform):
    def __init__(self, elemwise_transform=torch.sin, elemwise_inv_transform=torch.asin, scale=0.5*np.pi):
        super().__init__()
        self.elemwise_transform = elemwise_transform
        self.scale = scale
        self.elemwise_inv_transform = elemwise_inv_transform
        self.eps = 1e-8

    def forward(self, inputs, context=None):
        outputs = (self.elemwise_transform(inputs) + 1) * self.scale
        logabsdet_cos = torch.log(torch.cos(inputs) + self.eps).sum(-1)
        logabsdet_scale = inputs.shape[-1] * np.log(self.scale)

        return outputs, logabsdet_cos + logabsdet_scale

    def inverse(self, inputs, context=None):
        outputs = self.elemwise_inv_transform(inputs / self.scale - 1)
        logabsdet_cos = torch.log(torch.cos(outputs) + self.eps).sum(-1)
        logabsdet_scale = inputs.shape[-1] * np.log(self.scale)

        return outputs, -logabsdet_cos - logabsdet_scale


class ScaleLastDim(Transform):
    def __init__(self, scale=2.):
        super().__init__()
        self.scale = scale

    def forward(self, inputs, context=None):
        mask = torch.zeros_like(inputs)
        mask[...,-1] = torch.ones_like(inputs[..., -1])
        outputs = (1 - mask) * inputs + mask * inputs * self.scale
        logabsdet = inputs.new_ones(inputs.shape[0]) * np.log(self.scale)

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        mask = torch.zeros_like(inputs)
        mask[..., -1] = torch.ones_like(inputs[..., -1])
        outputs = (1 - mask) * inputs + mask * inputs / self.scale
        logabsdet = -inputs.new_ones(inputs.shape[0]) * np.log(self.scale)

        return outputs, logabsdet


class ConstrainedAngles(Transform):
    def __init__(self, elemwise_transform: Transform = Sigmoid()):
        super().__init__()
        self.elemwise_transform = elemwise_transform

    def forward(self, inputs, context=None):
        mask = torch.zeros_like(inputs)
        mask[..., -1] = torch.ones_like(inputs[..., -1])
        transformed_inputs = inputs - 0.5 * mask * inputs
        outputs, logabsdet_elemwise = self.elemwise_transform.inverse(transformed_inputs)
        logabsdet_last_elem = inputs.new_ones(inputs.shape[0]) * torch.log(torch.tensor(0.5))
        # breakpoint()
        return outputs, logabsdet_elemwise + logabsdet_last_elem

    def inverse(self, inputs, context=None):
        mask = torch.zeros_like(inputs)
        mask[...,-1] = torch.ones_like(inputs[..., -1])
        transformed_inputs, logabsdet_elemwise = self.elemwise_transform(inputs)
        outputs = mask * transformed_inputs + transformed_inputs
        logabsdet_last_elem = inputs.new_ones(inputs.shape[0]) * torch.log(torch.tensor(2.))
        # breakpoint()
        return outputs, logabsdet_elemwise + logabsdet_last_elem


class ConstrainedAnglesSigmoid(ConstrainedAngles):
    def __init__(self,temperature=1, learn_temperature=False):
        super().__init__(elemwise_transform=CompositeTransform([Sigmoid(temperature=temperature,
                                                                        learn_temperature=learn_temperature),
                                                                ScalarScale(scale=np.pi, trainable=False)]))

class ClampedAngles(Transform):
    _05PI = 0.5 * np.pi
    _10PI = 1.0 * np.pi
    _15PI = 1.5 * np.pi
    _20PI = 2.0 * np.pi

    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, context):
        self.dtype = inputs.dtype
        thetas = inputs[...,:-1]
        last_theta = inputs[...,-1:]
        _0pi_05pi_mask, _0pi_05pi_clamp = self.compute_mask(arr=thetas, vmin=0., vmax=self._05PI)
        _05pi_10pi_mask, _05pi_10pi_clamp = self.compute_mask(arr=thetas, vmin=self._05PI, vmax=self._10PI, right_included=True)
        clamped_thetas = _0pi_05pi_mask * _0pi_05pi_clamp + _05pi_10pi_mask * _05pi_10pi_clamp


        _0pi_05pi_mask, _0pi_05pi_clamp = self.compute_mask(arr=last_theta, vmin=0., vmax=self._05PI)
        _05pi_10pi_mask, _05pi_10pi_clamp = self.compute_mask(arr=last_theta, vmin=self._05PI, vmax=self._10PI)
        _10pi_15pi_mask, _10pi_15pi_clamp = self.compute_mask(arr=last_theta, vmin=self._10PI, vmax=self._15PI)
        _15pi_20pi_mask, _15pi_20pi_clamp = self.compute_mask(arr=last_theta, vmin=self._15PI, vmax=self._20PI, right_included=True)
        clamped_last_theta = _0pi_05pi_mask * _0pi_05pi_clamp + _05pi_10pi_mask * _05pi_10pi_clamp + \
                         _10pi_15pi_mask * _10pi_15pi_clamp + _15pi_20pi_mask * _15pi_20pi_clamp

        output = torch.cat((clamped_thetas, clamped_last_theta), dim = -1)
        logabsdet = output.new_zeros(inputs.shape[:-1])

        return output, logabsdet

    def compute_mask(self, arr, vmin, vmax, right_included=False):
        if right_included:
            condition = (arr >= vmin) * (arr < vmax)
        else:
            condition = (arr >= vmin) * (arr <= vmax)

        mask = condition.to(self.dtype)
        arr_clamped = torch.clamp(arr, min=vmin + self.eps, max=vmax - self.eps)

        return mask, arr_clamped


    def inverse(self, inputs, context):
        inputs = self.forward(inputs)
        return inputs, torch.zeros_like(inputs[...,0])

class ClampedTheta(Transform):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, context):
        self.dtype = inputs.dtype
        thetas = inputs[...,:-1]
        last_theta = inputs[...,-1:]
        _0_pi_mask, _0_pi_clamp = self.compute_mask(arr=thetas, vmin=0., vmax=np.pi, right_included=True)
        clamped_thetas = _0_pi_mask * _0_pi_clamp

        # _0_2pi_mask, _0_2pi_clamp = self.compute_mask(arr=last_theta, vmin=0., vmax=2*np.pi, right_included=True)
        # clamped_last_theta = _0_2pi_mask * _0_2pi_clamp

        output = torch.cat((clamped_thetas, last_theta), dim = -1)
        logabsdet = output.new_zeros(inputs.shape[:-1])

        return output, logabsdet

    def compute_mask(self, arr, vmin, vmax, right_included=False):
        if right_included:
            condition = (arr >= vmin) * (arr < vmax)
        else:
            condition = (arr >= vmin) * (arr <= vmax)

        mask = condition.to(self.dtype)
        arr_clamped = torch.clamp(arr, min=vmin + self.eps, max=vmax - self.eps)

        return mask, arr_clamped


    def inverse(self, inputs, context):
        outputs, _ = self.forward(inputs, context)
        return outputs, torch.zeros_like(outputs[...,0])

class ClampedThetaPositive(Transform):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, context):
        self.dtype = inputs.dtype
        _0_pi_mask, _0_pi_clamp = self.compute_mask(arr=inputs, vmin=0., vmax=torch.pi*0.5, right_included=True)
        output = _0_pi_mask * _0_pi_clamp
        logabsdet = output.new_zeros(inputs.shape[:-1])

        return output, logabsdet

    def compute_mask(self, arr, vmin, vmax, right_included=False):
        if right_included:
            condition = (arr >= vmin) * (arr < vmax)
        else:
            condition = (arr >= vmin) * (arr <= vmax)

        mask = condition.to(self.dtype)
        arr_clamped = torch.clamp(arr, min=vmin + self.eps, max=vmax - self.eps)

        return mask, arr_clamped


    def inverse(self, inputs, context):
        return inputs, torch.zeros_like(inputs[...,0])

###################################################CONDITIONAL LAYERS###################################################


class ConditionalFixedNorm(Transform):

    def __init__(self,q):
        super().__init__()
        self.q = q

    def forward(self, inputs, context):
        # the first element of context is assumed to be the norm value
        transformer = FixedNorm(norm=context[:, 0], q=self.q)

        output, logabsdet = transformer.forward(inputs, context)

        return output, logabsdet

    def inverse(self, inputs, context):
        raise NotImplementedError
# class ConditionalFixedNorm(ConditionalTransform):
#
#     def __init__(
#             self,
#             features,
#             hidden_features,
#             q,
#             context_features=None,
#             num_blocks=2,
#             use_residual_blocks=True,
#             activation=F.relu,
#             dropout_probability=0.0,
#             use_batch_norm=False,
#     ):
#         self.q = q
#         super().__init__(
#             features=features,
#             hidden_features=hidden_features,
#             context_features=context_features,
#             num_blocks=num_blocks,
#             use_residual_blocks=use_residual_blocks,
#             activation=activation,
#             dropout_probability=dropout_probability,
#             use_batch_norm=use_batch_norm
#         )
#
#     def _forward_given_params(self, inputs, context):
#         # the first element of context is assumed to be the norm value
#         transformer = FixedNorm(norm=context[:,:1], q=self.q)
#
#         output, logabsdet = transformer.forward(inputs)
#
#     def _inverse_given_params(self, inputs, autoregressive_params):
#         NotImplementedError


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class ResidualNetInput(nn.Module):
    """A residual network that . Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features -1)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)

        return torch.cat((inputs[:,:1], outputs), dim=1)




########################################################################################################################
#######################################################DEPRECATED#######################################################
########################################################################################################################



class FixedNorm(Transform):
    def __init__(self, norm, q):
        super().__init__()
        self.norm = norm
        self.q = q

        self.r_given_norm = r_given_norm
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def forward(self, inputs, context=None):

        if self.training and not self.initialized:
            self._initialize_jacobian(inputs)

        theta_r = inflate_radius(inputs, self.norm, self.q)
        outputs = spherical_to_cartesian_torch(theta_r)

        logabsdet = self.logabs_pseudodet(inputs, theta_r)

        return outputs, logabsdet


    def inverse(self, inputs, context=None):
        raise NotImplementedError


    def logabs_pseudodet(self, inputs, theta_r, context=None):
        eps = 1e-8
        spherical_dict = {name: theta_r[:, i].cpu() for i, name in enumerate(self.spherical_names)}
        jac = self.sph_to_cart_jac(**spherical_dict).reshape(-1, theta_r.shape[-1], theta_r.shape[-1]).to(inputs.device)
        # jac = jacobian(spherical_to_cartesian_torch, theta_r).sum(-2)
        check_tensor(jac)

        # assert torch.allclose(jac, jac_)

        jac_inv = sherman_morrison_inverse(jac.mT)
        #jac_inv = torch.inverse(jac.mT)
        check_tensor(jac_inv)


        grad_r = gradient_r(inputs, self.norm, self.q)
        check_tensor(grad_r)
        #grad_r = torch.clamp(grad_r, min=-100, max=100)
        # grad_r_np = grad_r.detach().cpu().numpy().reshape(-1,inputs.shape[-1]+1)[:,:inputs.shape[-1]]
        # inputs_np = inputs.detach().cpu().numpy().reshape(-1,inputs.shape[-1])
        # print("grad_r_np", grad_r_np)
        # print('contains nans', np.any(np.isnan(grad_r_np)))
        # grad_r_np[grad_r_np == 0] = 1e-7
        # log_min = np.log10(np.min(np.abs(grad_r_np.ravel())))
        # log_max = np.log10(np.max(np.abs(grad_r_np.ravel())))
        #
        # print(log_min, log_max)
        # plt.hist(np.abs(grad_r_np).ravel(), bins=np.logspace(log_min, log_max, 100))
        # plt.xscale('log')
        # # plt.scatter(np.linalg.norm(inputs_np, axis=-1), grad_r_np, marker='.')
        # plt.show()

        jac_inv_grad = jac_inv @ grad_r
        check_tensor(jac_inv_grad)
        # print(f"jac inv max: {jac_inv.squeeze().max().item():.3e}, "
        #       f"min: {jac_inv.squeeze().min().item():.3e} ")
        # print(f"grad inv max: {grad_r.squeeze().max().item():.3e}, "
        #       f"min: {grad_r.squeeze().min().item():.3e} ")
        fro_norm = torch.norm(jac_inv_grad.squeeze(), p='fro', dim=1)

        check_tensor(fro_norm)
        logabsdet_fro_norm = torch.log(torch.abs(fro_norm) + eps)
        logabsdet_s_to_c = logabsdet_sph_to_car(theta_r)

        check_tensor(logabsdet_fro_norm)
        check_tensor(logabsdet_s_to_c)

        logabsdet = logabsdet_s_to_c + logabsdet_fro_norm


        return logabsdet

    def _initialize_jacobian(self, inputs):
        spherical_names, jac = sph_to_cart_jacobian_sympy(inputs.shape[1]+1)
        self.spherical_names = spherical_names
        self.sph_to_cart_jac = sympytorch.SymPyModule(expressions=jac).to(inputs.device)


def r_given_norm(thetas, norm, q):
    assert thetas.shape[1] >= 1
    eps = 1e-8
    n = thetas.shape[-1]

    check_tensor(thetas)

    sin_thetas = torch.sin(thetas)
    cos_thetas = torch.cos(thetas)

    norm_1 = (torch.abs(cos_thetas[:, 0]) + eps) ** q
    check_tensor(norm_1)
    norm_3 = (torch.abs(torch.prod(sin_thetas, dim=-1)) + eps) ** q
    check_tensor(norm_3)
    if thetas.shape[1] == 1:
        r = norm / ((norm_1 + norm_3) ** (1. / q))
        check_tensor(r)
        return norm / ((norm_1 + norm_3 + eps) ** (1. / q))
    else:
        norm_2_ = [(torch.abs(torch.prod(sin_thetas[..., :k - 1], dim=-1) * cos_thetas[..., k - 1]) + eps) ** q for k in
                   range(2, n + 1)]
        norm_2 = torch.stack(norm_2_, dim=1).sum(-1)
        check_tensor(norm_2)

        r = norm / ((norm_1 + norm_2 + norm_3 + eps) ** (1. / q))
        check_tensor(r)
        return r


def inflate_radius(inputs, norm, q):
    r = r_given_norm(inputs, norm, q)
    # if context is not None:
    #     theta_r = torch.cat([inputs, r], dim=1)
    # else:
    #     theta_r = torch.cat([inputs, r.unsqueeze(-1)], dim=1)
    theta_r = torch.cat([inputs, r.unsqueeze(-1)], dim=1)
    return theta_r

def gradient_r(inputs, norm, q):
    r = r_given_norm(inputs, norm, q)
    grad_r_theta = torch.autograd.grad(r, inputs, grad_outputs=torch.ones_like(r))[0]
    grad_r_theta_aug = torch.cat([- grad_r_theta, torch.ones_like(grad_r_theta[:, :1])], dim=1)

    check_tensor(grad_r_theta)

    return grad_r_theta_aug.unsqueeze(-1)

def sherman_morrison_inverse_old(M):
    N = M.shape[-1]
    lower_indices = np.tril_indices(n=N, k=-1)
    mask = torch.ones(*M.shape)
    mask[:, lower_indices[0], lower_indices[1]] = 0.
    U = M * mask
    assert torch.all(U.triu() == U)

    v = torch.zeros(M.shape[:2])
    v[:, :-1] = M[:, -1, :-1]
    u = torch.zeros_like(v)
    u[:, -1] = 1.

    uv = torch.einsum("bi,bj->bij", u, v)
    assert torch.all(M == U + torch.einsum("bi,bj->bij", u, v))

    eye = torch.eye(N, N).repeat(M.shape[0], 1, 1)
    U_inv = torch.linalg.solve_triangular(U, eye, upper=True)
    num = U_inv @ uv @ U_inv

    den = 1 + v.unsqueeze(1) @ U_inv @ u.unsqueeze(2)

    return U_inv - num / den