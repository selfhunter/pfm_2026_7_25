#
# Baseline PFM: Original fixed-interval global reinit + uniform density
# Modified from run_2D.py to add metric collection for fair comparison.
# Simulation logic is IDENTICAL to the original — only metrics are added.
#
from hyperparameters import *
from taichi_utils import *
from mgpcg import *
from init_conditions import *
from io_utils import *
import torch
import sys
import shutil
import time
import numpy as np
import os

exp_name = "2D_baseline_1024x256_reinit-20-8"

# ---------------------------------------------------------------------------
# Setup (identical to run_2D.py)
# ---------------------------------------------------------------------------
dx = 1. / res_y
half_dx = 0.5 * dx
upper_boundary = 1 - half_dx
lower_boundary = half_dx
right_boundary = res_x * dx - half_dx
left_boundary = half_dx

ti.init(arch=ti.cuda, device_memory_GB=4.0, debug=False)

particles_per_cell_axis = int(ti.sqrt(particles_per_cell))
dist_between_neighbor = dx / particles_per_cell_axis

boundary_types = ti.Matrix([[2, 2], [2, 2]], ti.i32)
solver = MGPCG_2(boundary_types=boundary_types, N=[res_x, res_y], base_level=3)

X = ti.Vector.field(2, float, shape=(res_x, res_y))
X_horizontal = ti.Vector.field(2, float, shape=(res_x + 1, res_y))
X_vertical = ti.Vector.field(2, float, shape=(res_x, res_y + 1))
center_coords_func(X, dx)
horizontal_coords_func(X_horizontal, dx)
vertical_coords_func(X_vertical, dx)

T_x_grid = ti.Vector.field(2, float, shape=(res_x + 1, res_y))
T_y_grid = ti.Vector.field(2, float, shape=(res_x, res_y + 1))
psi_x_grid = ti.Vector.field(2, float, shape=(res_x + 1, res_y))
psi_y_grid = ti.Vector.field(2, float, shape=(res_x, res_y + 1))

p2g_weight_x = ti.field(float, shape=(res_x + 1, res_y))
p2g_weight_y = ti.field(float, shape=(res_x, res_y + 1))

u = ti.Vector.field(2, float, shape=(res_x, res_y))
w = ti.field(float, shape=(res_x, res_y))
u_x = ti.field(float, shape=(res_x + 1, res_y))
u_y = ti.field(float, shape=(res_x, res_y + 1))

initial_particle_num = res_x * res_y * particles_per_cell
particle_num = initial_particle_num * total_particles_num_ratio
current_particle_num = ti.field(int, shape=1)
particles_pos = ti.Vector.field(2, float, shape=particle_num)
particles_pos_backup = ti.Vector.field(2, float, shape=particle_num)
particles_imp = ti.Vector.field(2, float, shape=particle_num)
particles_init_imp = ti.Vector.field(2, float, shape=particle_num)
particles_init_imp_grad_m = ti.Vector.field(2, float, shape=particle_num)

T_x = ti.Vector.field(2, float, shape=particle_num)
T_y = ti.Vector.field(2, float, shape=particle_num)
T_x_init = ti.Vector.field(2, float, shape=particle_num)
T_y_init = ti.Vector.field(2, float, shape=particle_num)
T_x_grad_m = ti.Vector.field(2, float, shape=particle_num)
T_y_grad_m = ti.Vector.field(2, float, shape=particle_num)
psi = ti.Vector.field(2, float, shape=particle_num)

cell_max_particle_num = int(cell_max_particle_num_ratio * particles_per_cell)
cell_particle_num = ti.field(int, shape=(res_x, res_y))
cell_particles_id = ti.field(int, shape=(res_x, res_y, cell_max_particle_num))

C_x = ti.Vector.field(2, float, shape=particle_num)
C_y = ti.Vector.field(2, float, shape=particle_num)
init_C_x = ti.Vector.field(2, float, shape=particle_num)
init_C_y = ti.Vector.field(2, float, shape=particle_num)

tmp_u_x = ti.field(float, shape=(res_x + 1, res_y))
tmp_u_y = ti.field(float, shape=(res_x, res_y + 1))
count = ti.field(int, shape=particle_num)

max_speed = ti.field(float, shape=())
dts = torch.zeros(1)

init_smoke = ti.Vector.field(3, float, shape=(res_x, res_y))
smoke = ti.Vector.field(3, float, shape=(res_x, res_y))
tmp_smoke = ti.Vector.field(3, float, shape=(res_x, res_y))
err_smoke = ti.Vector.field(3, float, shape=(res_x, res_y))

# ---------------------------------------------------------------------------
# Metric fields (ADDED for comparison)
# ---------------------------------------------------------------------------
kinetic_energy_val = ti.field(float, shape=())
enstrophy_val = ti.field(float, shape=())

# ---------------------------------------------------------------------------
# Standard kernels (IDENTICAL to original run_2D.py)
# ---------------------------------------------------------------------------
@ti.kernel
def calc_max_speed(u_x: ti.template(), u_y: ti.template()):
    max_speed[None] = 1.e-3
    for i, j in ti.ndrange(res_x, res_y):
        u = 0.5 * (u_x[i, j] + u_x[i + 1, j])
        v = 0.5 * (u_y[i, j] + u_y[i, j + 1])
        speed = ti.sqrt(u ** 2 + v ** 2)
        ti.atomic_max(max_speed[None], speed)

@ti.kernel
def calc_max_imp_particles(particles_imp: ti.template()):
    max_speed[None] = 1.e-3
    for i in particles_imp:
        imp = particles_imp[i].norm()
        ti.atomic_max(max_speed[None], imp)

@ti.kernel
def reset_to_identity_grid(psi_x: ti.template(), psi_y: ti.template(), T_x: ti.template(), T_y: ti.template()):
    for i, j in psi_x:
        psi_x[i, j] = X_horizontal[i, j]
    for i, j in psi_y:
        psi_y[i, j] = X_vertical[i, j]
    for i, j in T_x:
        T_x[i, j] = ti.Vector.unit(2, 0)
    for i, j in T_y:
        T_y[i, j] = ti.Vector.unit(2, 1)

@ti.kernel
def reset_to_identity(psi: ti.template(), T_x: ti.template(), T_y: ti.template()):
    for i in psi:
        psi[i] = particles_pos[i]
    for i in T_x:
        T_x[i] = ti.Vector.unit(2, 0)
    for i in T_y:
        T_y[i] = ti.Vector.unit(2, 1)

@ti.kernel
def reset_T_to_identity(T_x: ti.template(), T_y: ti.template()):
    for i in T_x:
        T_x[i] = ti.Vector.unit(2, 0)
    for i in T_y:
        T_y[i] = ti.Vector.unit(2, 1)

@ti.kernel
def check_psi_and_X(curr_step: int, psi: ti.template(), particles_pos_backup: ti.template()):
    different_X_psi_num = 0
    for i in particles_pos_backup:
        diff = psi[i] - particles_pos_backup[i]
        if diff.norm() > 1e-3:
            different_X_psi_num += 1
    print(f'Step {curr_step}: {different_X_psi_num}/{particle_num} different psi and X')

@ti.kernel
def RK4_grid(psi_x: ti.template(), T_x: ti.template(),
             u_x0: ti.template(), u_y0: ti.template(), dt: float):
    neg_dt = -1 * dt
    for i, j in psi_x:
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x[i, j], dx)
        dT_x_dt1 = grad_u_at_psi @ T_x[i, j]
        psi_x1 = psi_x[i, j] + 0.5 * neg_dt * u1
        T_x1 = T_x[i, j] + 0.5 * neg_dt * dT_x_dt1
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi @ T_x1
        psi_x2 = psi_x[i, j] + 0.5 * neg_dt * u2
        T_x2 = T_x[i, j] + 0.5 * neg_dt * dT_x_dt2
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x2, dx)
        dT_x_dt3 = grad_u_at_psi @ T_x2
        psi_x3 = psi_x[i, j] + 1.0 * neg_dt * u3
        T_x3 = T_x[i, j] + 1.0 * neg_dt * dT_x_dt3
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x3, dx)
        dT_x_dt4 = grad_u_at_psi @ T_x3
        psi_x[i, j] = psi_x[i, j] + neg_dt * 1./6 * (u1 + 2*u2 + 2*u3 + u4)
        T_x[i, j] = T_x[i, j] + neg_dt * 1./6 * (dT_x_dt1 + 2*dT_x_dt2 + 2*dT_x_dt3 + dT_x_dt4)

@ti.kernel
def RK4(psi: ti.template(), T_x: ti.template(), T_y: ti.template(),
        u_x0: ti.template(), u_y0: ti.template(), dt: float):
    neg_dt = -1 * dt
    for i in psi:
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi[i], dx)
        dT_x_dt1 = grad_u_at_psi @ T_x[i]
        dT_y_dt1 = grad_u_at_psi @ T_y[i]
        psi_x1 = psi[i] + 0.5 * neg_dt * u1
        T_x1 = T_x[i] + 0.5 * neg_dt * dT_x_dt1
        T_y1 = T_y[i] + 0.5 * neg_dt * dT_y_dt1
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi @ T_x1
        dT_y_dt2 = grad_u_at_psi @ T_y1
        psi_x2 = psi[i] + 0.5 * neg_dt * u2
        T_x2 = T_x[i] + 0.5 * neg_dt * dT_x_dt2
        T_y2 = T_y[i] + 0.5 * neg_dt * dT_y_dt2
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x2, dx)
        dT_x_dt3 = grad_u_at_psi @ T_x2
        dT_y_dt3 = grad_u_at_psi @ T_y2
        psi_x3 = psi[i] + 1.0 * neg_dt * u3
        T_x3 = T_x[i] + 1.0 * neg_dt * dT_x_dt3
        T_y3 = T_y[i] + 1.0 * neg_dt * dT_y_dt3
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x3, dx)
        dT_x_dt4 = grad_u_at_psi @ T_x3
        dT_y_dt4 = grad_u_at_psi @ T_y3
        psi[i] = psi[i] + neg_dt * 1./6 * (u1 + 2*u2 + 2*u3 + u4)
        T_x[i] = T_x[i] + neg_dt * 1./6 * (dT_x_dt1 + 2*dT_x_dt2 + 2*dT_x_dt3 + dT_x_dt4)
        T_y[i] = T_y[i] + neg_dt * 1./6 * (dT_y_dt1 + 2*dT_y_dt2 + 2*dT_y_dt3 + dT_y_dt4)

@ti.kernel
def RK4_T_forward(psi: ti.template(), T_x: ti.template(), T_y: ti.template(),
                  u_x0: ti.template(), u_y0: ti.template(), dt: float, advect_psi: int):
    for i in psi:
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi[i], dx)
        dT_x_dt1 = grad_u_at_psi.transpose() @ T_x[i]
        dT_y_dt1 = grad_u_at_psi.transpose() @ T_y[i]
        psi_x1 = psi[i] + 0.5 * dt * u1
        T_x1 = T_x[i] - 0.5 * dt * dT_x_dt1
        T_y1 = T_y[i] - 0.5 * dt * dT_y_dt1
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi.transpose() @ T_x1
        dT_y_dt2 = grad_u_at_psi.transpose() @ T_y1
        psi_x2 = psi[i] + 0.5 * dt * u2
        T_x2 = T_x[i] - 0.5 * dt * dT_x_dt2
        T_y2 = T_y[i] - 0.5 * dt * dT_y_dt2
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x2, dx)
        dT_x_dt3 = grad_u_at_psi.transpose() @ T_x2
        dT_y_dt3 = grad_u_at_psi.transpose() @ T_y2
        psi_x3 = psi[i] + 1.0 * dt * u3
        T_x3 = T_x[i] - 1.0 * dt * dT_x_dt3
        T_y3 = T_y[i] - 1.0 * dt * dT_y_dt3
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, psi_x3, dx)
        dT_x_dt4 = grad_u_at_psi.transpose() @ T_x3
        dT_y_dt4 = grad_u_at_psi.transpose() @ T_y3
        if advect_psi:
            psi[i] = psi[i] + dt * 1./6 * (u1 + 2*u2 + 2*u3 + u4)
        T_x[i] = T_x[i] - dt * 1./6 * (dT_x_dt1 + 2*dT_x_dt2 + 2*dT_x_dt3 + dT_x_dt4)
        T_y[i] = T_y[i] - dt * 1./6 * (dT_y_dt1 + 2*dT_y_dt2 + 2*dT_y_dt3 + dT_y_dt4)

@ti.kernel
def advect_particles(dt: float):
    for i in particles_pos:
        v1, _ = interp_u_MAC_grad(u_x, u_y, particles_pos[i], dx)
        p2 = particles_pos[i] + v1 * dt * 0.5
        v2, _ = interp_u_MAC_grad(u_x, u_y, p2, dx)
        p3 = particles_pos[i] + v2 * dt * 0.5
        v3, _ = interp_u_MAC_grad(u_x, u_y, p3, dx)
        p4 = particles_pos[i] + v3 * dt
        v4, _ = interp_u_MAC_grad(u_x, u_y, p4, dx)
        particles_pos[i] += (v1 + 2*v2 + 2*v3 + v4) / 6.0 * dt

@ti.kernel
def limit_particles_in_boundary():
    for i in particles_pos:
        if particles_pos[i][0] < left_boundary:
            particles_pos[i][0] = left_boundary
        if particles_pos[i][0] > right_boundary:
            particles_pos[i][0] = right_boundary
        if particles_pos[i][1] < lower_boundary:
            particles_pos[i][1] = lower_boundary
        if particles_pos[i][1] > upper_boundary:
            particles_pos[i][1] = upper_boundary

@ti.kernel
def advect_u(u_x0: ti.template(), u_y0: ti.template(),
             u_x1: ti.template(), u_y1: ti.template(),
             T_x: ti.template(), T_y: ti.template(),
             psi_x: ti.template(), psi_y: ti.template(), dx: float):
    for i, j in u_x1:
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x[i, j], dx)
        u_x1[i, j] = T_x[i, j].dot(u_at_psi)
    for i, j in u_y1:
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, psi_y[i, j], dx)
        u_y1[i, j] = T_y[i, j].dot(u_at_psi)

@ti.kernel
def advect_smoke(smoke0: ti.template(), smoke1: ti.template(),
                 psi_x: ti.template(), psi_y: ti.template(), dx: float):
    for i, j in ti.ndrange(res_x, res_y):
        psi_c = 0.25 * (psi_x[i, j] + psi_x[i+1, j] + psi_y[i, j] + psi_y[i, j+1])
        smoke1[i, j] = interp_1(smoke0, psi_c, dx)

@ti.kernel
def update_particles_imp(particles_imp: ti.template(), particles_init_imp: ti.template(),
                         T_x: ti.template(), T_y: ti.template()):
    for i in particles_imp:
        T = ti.Matrix.cols([T_x[i], T_y[i]])
        particles_imp[i] = T @ particles_init_imp[i]

@ti.kernel
def update_particles_grad_m(C_x: ti.template(), C_y: ti.template(), init_C_x: ti.template(), init_C_y: ti.template(),
                            T_x: ti.template(), T_y: ti.template()):
    for i in C_x:
        T = ti.Matrix.rows([T_x[i], T_y[i]])
        T_transpose = ti.Matrix.cols([T_x[i], T_y[i]])
        init_C = ti.Matrix.rows([init_C_x[i], init_C_y[i]])
        T_init_C_T = T_transpose @ (init_C @ T)
        C_x[i] = T_init_C_T[0, :]
        C_y[i] = T_init_C_T[1, :]

@ti.kernel
def update_T(T_x: ti.template(), T_y: ti.template(), T_x_init: ti.template(), T_y_init: ti.template(),
             T_x_grad_m: ti.template(), T_y_grad_m: ti.template()):
    for i in T_x:
        T_grad_m = ti.Matrix.cols([T_x_grad_m[i], T_y_grad_m[i]])
        T_init = ti.Matrix.cols([T_x_init[i], T_y_init[i]])
        T = T_grad_m @ T_init
        T_x[i] = T[:, 0]
        T_y[i] = T[:, 1]

@ti.kernel
def P2G(particles_imp: ti.template(), particles_pos: ti.template(), u_x: ti.template(), u_y: ti.template(),
        psi: ti.template(), psi_x_grid: ti.template(), psi_y_grid: ti.template(),
        p2g_weight_x: ti.template(), p2g_weight_y: ti.template()):
    u_x.fill(0.0)
    u_y.fill(0.0)
    psi_x_grid.fill(0.0)
    psi_y_grid.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)

    for i in particles_imp:
        pos = particles_pos[i] / dx
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                p2g_weight_x[face_id] += weight
                delta = C_x[i].dot(dpos)
                if use_APIC:
                    u_x[face_id] += (particles_imp[i][0] + delta) * weight
                else:
                    u_x[face_id] += (particles_imp[i][0]) * weight
                psi_x_grid[face_id] += psi[i] * weight

        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                p2g_weight_y[face_id] += weight
                delta = C_y[i].dot(dpos)
                if use_APIC:
                    u_y[face_id] += (particles_imp[i][1] + delta) * weight
                else:
                    u_y[face_id] += (particles_imp[i][1]) * weight
                psi_y_grid[face_id] += psi[i] * weight

    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 0:
            scale = 1. / p2g_weight_x[I]
            u_x[I] *= scale
            psi_x_grid[I] *= scale

    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 0:
            scale = 1. / p2g_weight_y[I]
            u_y[I] *= scale
            psi_y_grid[I] *= scale


def stretch_T_and_advect_particles(particles_pos, T_x, T_y, u_x, u_y, dt):
    RK4_T_forward(particles_pos, T_x, T_y, u_x, u_y, dt, 1)

def stretch_T(particles_pos, T_x, T_y, u_x, u_y, dt):
    RK4_T_forward(particles_pos, T_x, T_y, u_x, u_y, dt, 0)


# ===================================================================
# METRIC KERNELS (ADDED - no impact on simulation)
# ===================================================================

@ti.kernel
def compute_kinetic_energy(u_x: ti.template(), u_y: ti.template(),
                            energy: ti.template()):
    energy[None] = 0.0
    for i, j in ti.ndrange(res_x, res_y):
        u_c = 0.5 * (u_x[i, j] + u_x[i + 1, j])
        v_c = 0.5 * (u_y[i, j] + u_y[i, j + 1])
        energy[None] += 0.5 * (u_c*u_c + v_c*v_c)
    energy[None] /= float(res_x * res_y)


@ti.kernel
def compute_enstrophy(w: ti.template(), enstrophy: ti.template()):
    enstrophy[None] = 0.0
    for i, j in w:
        enstrophy[None] += 0.5 * w[i, j] * w[i, j]
    enstrophy[None] /= float(res_x * res_y)


# ===================================================================
# MAIN (identical logic to original, + metric collection)
# ===================================================================
def main(from_frame=0, testing=False, total_steps_override=None):
    from_frame = max(0, from_frame)
    global total_steps
    if total_steps_override is not None:
        total_steps = total_steps_override

    logsdir = os.path.join('logs', exp_name)
    os.makedirs(logsdir, exist_ok=True)
    if from_frame <= 0:
        remove_everything_in(logsdir)

    vortdir = os.path.join(logsdir, 'vorticity')
    os.makedirs(vortdir, exist_ok=True)
    smokedir = os.path.join(logsdir, 'smoke')
    os.makedirs(smokedir, exist_ok=True)
    ckptdir = os.path.join(logsdir, 'ckpts')
    os.makedirs(ckptdir, exist_ok=True)
    metrics_dir = os.path.join(logsdir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(script_dir, "hyperparameters.py")
    shutil.copyfile(src_path, os.path.join(logsdir, "hyperparameters.py"))

    if testing:
        testdir = os.path.join(logsdir, 'test_buffer')
        os.makedirs(testdir, exist_ok=True)
        remove_everything_in(testdir)
        GTdir = os.path.join(testdir, "GT")
        os.makedirs(GTdir, exist_ok=True)
        preddir = os.path.join(testdir, "pred")
        os.makedirs(preddir, exist_ok=True)

    # initial condition (IDENTICAL to original)
    if from_frame <= 0:
        leapfrog_vel_func(u, X)
        split_central_vector(u, u_x, u_y)
        solver.Poisson(u_x, u_y)
        stripe_func(smoke, X, 0.25, 0.48)

    # Particle init (IDENTICAL to original)
    init_particles_pos_uniform(particles_pos, X, res_x, particles_per_cell, dx,
                               particles_per_cell_axis, dist_between_neighbor)
    init_particles_imp_grad_m(particles_init_imp_grad_m, particles_pos, u_x, u_y,
                              init_C_x, init_C_y, dx)
    reset_T_to_identity(T_x_grad_m, T_y_grad_m)
    backup_particles_pos(particles_pos, particles_pos_backup)
    init_particles_imp(particles_imp, particles_init_imp, particles_pos, u_x, u_y, init_C_x, init_C_y, dx)
    current_particle_num[0] = initial_particle_num

    # Initial visualization
    get_central_vector(u_x, u_y, u)
    curl(u, w, dx)
    w_numpy = w.to_numpy()
    w_max = max(np.abs(w_numpy.max()), np.abs(w_numpy.min()))
    w_min = -1 * w_max
    write_field(w_numpy, vortdir, from_frame, particles_pos.to_numpy() / dx,
                vmin=w_min, vmax=w_max, plot_particles=plot_particles, dpi=dpi_vor)
    write_image(smoke.to_numpy(), smokedir, from_frame)

    if save_ckpt:
        np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame)), u_x.to_numpy())
        np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame)), u_y.to_numpy())
        np.save(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame)), smoke.to_numpy())

    # Simulation state (IDENTICAL to original)
    sub_t = 0.
    frame_idx = from_frame
    last_output_substep = 0
    num_reinits = 0
    i = -1
    ik = 0

    # Metric arrays (ADDED)
    frame_times = np.zeros(total_steps)
    metrics_kinetic_energy = np.zeros(total_steps)
    metrics_enstrophy = np.zeros(total_steps)

    print(f"[Baseline] Running {total_steps} steps (original fixed-interval reinit)")

    while True:
        step_start_time = time.time()
        i += 1
        j = i % reinit_every
        i_next = i + 1
        j_next = i_next % reinit_every

        # Determine dt (IDENTICAL)
        calc_max_speed(u_x, u_y)
        curr_dt = CFL * dx / max_speed[None]

        if save_frame_each_step:
            output_frame = True
            frame_idx += 1
        else:
            if sub_t + curr_dt >= visualize_dt:
                curr_dt = visualize_dt - sub_t
                sub_t = 0.
                frame_idx += 1
                print(f'Visualized frame {frame_idx}')
                output_frame = True
            else:
                sub_t += curr_dt
                output_frame = False

        dts[0] = curr_dt

        # ============================================================
        # GLOBAL REINIT (IDENTICAL to original)
        # ============================================================
        if j == 0:
            print(f"[Simulate] GLOBAL reinit at step {i}")
            if reinit_particle_pos or i == 0:
                init_particles_pos_uniform(particles_pos, X, res_x, particles_per_cell, dx,
                                           particles_per_cell_axis, dist_between_neighbor)
            ik = i
            backup_particles_pos(particles_pos, particles_pos_backup)
            init_particles_imp(particles_imp, particles_init_imp, particles_pos, u_x, u_y, init_C_x, init_C_y, dx)
            reset_to_identity(psi, T_x, T_y)
            reset_T_to_identity(T_x_init, T_y_init)
            copy_to(smoke, init_smoke)
            get_central_vector(u_x, u_y, u)
            num_reinits += 1

        # Grad-m reinit (IDENTICAL)
        k = (i - ik) % reinit_every_grad_m
        if k == 0:
            init_particles_imp_grad_m(particles_init_imp_grad_m, particles_pos, u_x, u_y,
                                      init_C_x, init_C_y, dx)
            reset_T_to_identity(T_x_grad_m, T_y_grad_m)
            copy_to(T_x, T_x_init)
            copy_to(T_y, T_y_init)

        # ============================================================
        # SIMULATION STEP (IDENTICAL to original)
        # ============================================================
        if use_midpoint_vel:
            reset_to_identity_grid(psi_x_grid, psi_y_grid, T_x_grid, T_y_grid)
            RK4_grid(psi_x_grid, T_x_grid, u_x, u_y, 0.5 * curr_dt)
            RK4_grid(psi_y_grid, T_y_grid, u_x, u_y, 0.5 * curr_dt)
            copy_to(u_x, tmp_u_x)
            copy_to(u_y, tmp_u_y)
            advect_u(tmp_u_x, tmp_u_y, u_x, u_y, T_x_grid, T_y_grid, psi_x_grid, psi_y_grid, dx)
            solver.Poisson(u_x, u_y)

        stretch_T_and_advect_particles(particles_pos, T_x_grad_m, T_y_grad_m, u_x, u_y, curr_dt)
        update_particles_grad_m(C_x, C_y, init_C_x, init_C_y, T_x_grad_m, T_y_grad_m)

        update_T(T_x, T_y, T_x_init, T_y_init, T_x_grad_m, T_y_grad_m)
        update_particles_imp(particles_imp, particles_init_imp, T_x, T_y)

        P2G(particles_imp, particles_pos, u_x, u_y, psi, psi_x_grid, psi_y_grid, p2g_weight_x, p2g_weight_y)
        solver.Poisson(u_x, u_y)
        advect_smoke(init_smoke, smoke, psi_x_grid, psi_y_grid, dx)

        # ============================================================
        # METRIC COLLECTION (ADDED)
        # ============================================================
        step_end_time = time.time()
        frame_times[i] = step_end_time - step_start_time

        compute_kinetic_energy(u_x, u_y, kinetic_energy_val)
        metrics_kinetic_energy[i] = kinetic_energy_val[None]

        get_central_vector(u_x, u_y, u)
        curl(u, w, dx)
        compute_enstrophy(w, enstrophy_val)
        metrics_enstrophy[i] = enstrophy_val[None]

        print(f"[Step {i}] time={frame_times[i]:.4f}s KE={metrics_kinetic_energy[i]:.6f} "
              f"Enstrophy={metrics_enstrophy[i]:.6f}")

        if output_frame:
            w_numpy = w.to_numpy()
            write_field(w_numpy, vortdir, frame_idx, particles_pos.to_numpy() / dx,
                        vmin=w_min, vmax=w_max, plot_particles=plot_particles, dpi=dpi_vor)
            write_image(smoke.to_numpy(), smokedir, frame_idx)
            if frame_idx % ckpt_every == 0 and save_ckpt:
                np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(frame_idx)), u_x.to_numpy())
                np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(frame_idx)), u_y.to_numpy())
                np.save(os.path.join(ckptdir, "smoke_numpy_" + str(frame_idx)), smoke.to_numpy())
            print(f"\n[Simulate] Finished frame: {frame_idx} in {i - last_output_substep} substeps\n")
            last_output_substep = i
            if frame_idx >= total_frames:
                break

        if i >= total_steps - 1:
            break

    # ================================================================
    # SAVE METRICS (ADDED)
    # ================================================================
    n_steps = i + 1
    metrics = {
        'frame_times': frame_times[:n_steps],
        'kinetic_energy': metrics_kinetic_energy[:n_steps],
        'enstrophy': metrics_enstrophy[:n_steps],
        'total_cells': res_x * res_y,
        'total_particles': particle_num,
        'exp_name': exp_name,
        'mode': 'baseline',
        'reinit_every': reinit_every,
        'reinit_every_grad_m': reinit_every_grad_m,
    }
    np.savez(os.path.join(metrics_dir, 'metrics.npz'), **metrics)

    frame_time_dir = os.path.join(logsdir, 'frame_time')
    os.makedirs(frame_time_dir, exist_ok=True)
    np.save(os.path.join(frame_time_dir, 'frame_times.npy'), frame_times[:n_steps])

    print(f"[Baseline] Complete. {n_steps} steps simulated.")
    return metrics


if __name__ == '__main__':
    print("[Baseline PFM] Begin")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=None, help='Total steps to run')
    args = parser.parse_args()

    main(from_frame=from_frame, testing=testing, total_steps_override=args.steps)
    print("[Baseline PFM] Complete")
