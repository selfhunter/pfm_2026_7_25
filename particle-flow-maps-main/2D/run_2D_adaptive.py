#
# Adaptive PFM: Deformation-triggered local redistribution + vorticity-guided density
# Based on run_2D.py with the following innovations:
#   1. Deformation factor ||T-I||_F triggers local (not global) reinit
#   2. Vorticity-guided adaptive particle density (16/cell in vortex, 4/cell in calm)
#   3. Full metric collection for before/after comparison
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

# ---------------------------------------------------------------------------
# Adaptive hyperparameters (overridable)
# ---------------------------------------------------------------------------
deformation_threshold = 0.3       # ||T-I||_F threshold for triggering reinit
vorticity_threshold_ratio = 0.25  # fraction of max |ω| for high-density marking
particles_per_cell_high = 16      # vortex / high-deformation regions
particles_per_cell_low = 4        # calm / low-deformation regions
adaptive_reinit_interval = 5      # check deformation every N steps
adaptive_density_interval = 10    # update density marking every N steps
use_adaptive_reinit = True
use_adaptive_density = True
exp_name = "2D_adaptive_1024x256"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
dx = 1. / res_y
half_dx = 0.5 * dx
upper_boundary = 1 - half_dx
lower_boundary = half_dx
right_boundary = res_x * dx - half_dx
left_boundary = half_dx

ti.init(arch=ti.cuda, device_memory_GB=4.0, debug=False)

particles_per_cell_axis_high = int(ti.sqrt(particles_per_cell_high))
particles_per_cell_axis_low = int(ti.sqrt(particles_per_cell_low))
dist_between_neighbor_high = dx / particles_per_cell_axis_high
dist_between_neighbor_low = dx / particles_per_cell_axis_low

# solver
boundary_types = ti.Matrix([[2, 2], [2, 2]], ti.i32)
solver = MGPCG_2(boundary_types=boundary_types, N=[res_x, res_y], base_level=3)

# undeformed coordinates
X = ti.Vector.field(2, float, shape=(res_x, res_y))
X_horizontal = ti.Vector.field(2, float, shape=(res_x + 1, res_y))
X_vertical = ti.Vector.field(2, float, shape=(res_x, res_y + 1))
center_coords_func(X, dx)
horizontal_coords_func(X_horizontal, dx)
vertical_coords_func(X_vertical, dx)

# back flow map (grid)
T_x_grid = ti.Vector.field(2, float, shape=(res_x + 1, res_y))
T_y_grid = ti.Vector.field(2, float, shape=(res_x, res_y + 1))
psi_x_grid = ti.Vector.field(2, float, shape=(res_x + 1, res_y))
psi_y_grid = ti.Vector.field(2, float, shape=(res_x, res_y + 1))

# P2G weight
p2g_weight_x = ti.field(float, shape=(res_x + 1, res_y))
p2g_weight_y = ti.field(float, shape=(res_x, res_y + 1))

# velocity
u = ti.Vector.field(2, float, shape=(res_x, res_y))
w = ti.field(float, shape=(res_x, res_y))
u_x = ti.field(float, shape=(res_x + 1, res_y))
u_y = ti.field(float, shape=(res_x, res_y + 1))

# particle storage
initial_particle_num = res_x * res_y * particles_per_cell
particle_num = initial_particle_num * total_particles_num_ratio
current_particle_num = ti.field(int, shape=1)
particles_pos = ti.Vector.field(2, float, shape=particle_num)
particles_pos_backup = ti.Vector.field(2, float, shape=particle_num)
particles_imp = ti.Vector.field(2, float, shape=particle_num)
particles_init_imp = ti.Vector.field(2, float, shape=particle_num)
particles_init_imp_grad_m = ti.Vector.field(2, float, shape=particle_num)

# back flow map (particle)
T_x = ti.Vector.field(2, float, shape=particle_num)
T_y = ti.Vector.field(2, float, shape=particle_num)
T_x_init = ti.Vector.field(2, float, shape=particle_num)
T_y_init = ti.Vector.field(2, float, shape=particle_num)
T_x_grad_m = ti.Vector.field(2, float, shape=particle_num)
T_y_grad_m = ti.Vector.field(2, float, shape=particle_num)
psi = ti.Vector.field(2, float, shape=particle_num)

# particles in each cell
cell_max_particle_num = int(cell_max_particle_num_ratio * particles_per_cell)
cell_particle_num = ti.field(int, shape=(res_x, res_y))
cell_particles_id = ti.field(int, shape=(res_x, res_y, cell_max_particle_num))

# APIC
C_x = ti.Vector.field(2, float, shape=particle_num)
C_y = ti.Vector.field(2, float, shape=particle_num)
init_C_x = ti.Vector.field(2, float, shape=particle_num)
init_C_y = ti.Vector.field(2, float, shape=particle_num)

# helper
tmp_u_x = ti.field(float, shape=(res_x + 1, res_y))
tmp_u_y = ti.field(float, shape=(res_x, res_y + 1))
count = ti.field(int, shape=particle_num)

# CFL
max_speed = ti.field(float, shape=())
dts = torch.zeros(1)

# smoke
init_smoke = ti.Vector.field(3, float, shape=(res_x, res_y))
smoke = ti.Vector.field(3, float, shape=(res_x, res_y))
tmp_smoke = ti.Vector.field(3, float, shape=(res_x, res_y))
err_smoke = ti.Vector.field(3, float, shape=(res_x, res_y))

# ---------------------------------------------------------------------------
# Adaptive PFM fields
# ---------------------------------------------------------------------------
cell_deformation = ti.field(float, shape=(res_x, res_y))       # avg ||T-I||_F per cell
cell_deformation_count = ti.field(int, shape=(res_x, res_y))    # particle count per cell
cell_vorticity_mag = ti.field(float, shape=(res_x, res_y))      # |ω| per cell
cell_marked_reinit = ti.field(int, shape=(res_x, res_y))        # 1 = needs local reinit
cell_density_level = ti.field(int, shape=(res_x, res_y))        # 1=high, 0=low
particles_active = ti.field(int, shape=particle_num)            # 1 = active
particle_deformation = ti.field(float, shape=particle_num)      # per-particle ||T-I||_F

# reinit statistics
total_marked_cells = ti.field(int, shape=())
total_active_particles = ti.field(int, shape=())

# ---------------------------------------------------------------------------
# Metric collection fields
# ---------------------------------------------------------------------------
kinetic_energy_val = ti.field(float, shape=())
enstrophy_val = ti.field(float, shape=())
avg_deformation_val = ti.field(float, shape=())
max_deformation_val = ti.field(float, shape=())

# ---------------------------------------------------------------------------
# Standard kernels (copied from run_2D.py)
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
        p2g_weight_x: ti.template(), p2g_weight_y: ti.template(),
        particles_active: ti.template()):
    u_x.fill(0.0)
    u_y.fill(0.0)
    psi_x_grid.fill(0.0)
    psi_y_grid.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)

    for i in particles_imp:
        if particles_active[i] == 1:
            pos = particles_pos[i] / dx
            # horizontal impulse
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

            # vertical impulse
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
# ADAPTIVE PFM KERNELS (core innovations)
# ===================================================================

@ti.kernel
def compute_particle_deformation(T_x: ti.template(), T_y: ti.template(),
                                  particle_def: ti.template()):
    """Compute ||T - I||_F for each particle.
    T = [T_x | T_y] as columns (2x2 matrix).
    ||T - I||_F² = (T_x.x-1)² + T_x.y² + T_y.x² + (T_y.y-1)²
    -> 0 for rigid translation, grows with shear/stretch/rotation.
    """
    for i in T_x:
        d00 = T_x[i][0] - 1.0
        d01 = T_x[i][1]
        d10 = T_y[i][0]
        d11 = T_y[i][1] - 1.0
        particle_def[i] = ti.sqrt(d00*d00 + d01*d01 + d10*d10 + d11*d11)


@ti.kernel
def aggregate_cell_deformation(particle_def: ti.template(), particles_pos: ti.template(),
                                cell_def: ti.template(), cell_count: ti.template()):
    """Average deformation factor per grid cell."""
    cell_def.fill(0.0)
    cell_count.fill(0)
    for i in particle_def:
        cell_id = int(particles_pos[i] / dx)
        if 0 <= cell_id[0] < res_x and 0 <= cell_id[1] < res_y:
            cell_def[cell_id] += particle_def[i]
            cell_count[cell_id] += 1
    for i, j in cell_def:
        if cell_count[i, j] > 0:
            cell_def[i, j] /= float(cell_count[i, j])


@ti.kernel
def compute_cell_vorticity_mag(w: ti.template(), cell_vor: ti.template()):
    """Compute |vorticity| per cell from curl field."""
    for i, j in w:
        cell_vor[i, j] = ti.abs(w[i, j])


@ti.kernel
def mark_cells_adaptive(cell_def: ti.template(), cell_vor: ti.template(),
                         cell_marked: ti.template(), cell_density: ti.template(),
                         def_threshold: float, vor_threshold: float):
    """Mark cells for local reinit (deformation > threshold).
    Set density level: 1=high (vortex), 0=low (calm)."""
    for i, j in cell_def:
        # Mark for reinit based on deformation
        if cell_def[i, j] > def_threshold:
            cell_marked[i, j] = 1
        else:
            cell_marked[i, j] = 0

        # Set density based on vorticity magnitude
        if cell_vor[i, j] > vor_threshold:
            cell_density[i, j] = 1  # high density
        else:
            cell_density[i, j] = 0  # low density


@ti.kernel
def count_marked_cells(cell_marked: ti.template(), total: ti.template()):
    """Count total cells marked for reinit."""
    total[None] = 0
    for i, j in cell_marked:
        if cell_marked[i, j] == 1:
            total[None] += 1


@ti.kernel
def adaptive_local_reinit(
    particles_pos: ti.template(), particles_imp: ti.template(),
    particles_init_imp: ti.template(),
    psi: ti.template(), T_x: ti.template(), T_y: ti.template(),
    T_x_init: ti.template(), T_y_init: ti.template(),
    T_x_grad_m: ti.template(), T_y_grad_m: ti.template(),
    particles_active: ti.template(),
    cell_marked: ti.template(),
    u_x: ti.template(), u_y: ti.template(), dx: float):
    """Local reinit: reset flow map to identity ONLY for particles
    in cells marked as high-deformation. Unmarked cells' particles
    retain their accumulated flow maps, preserving continuity."""

    for i in particles_pos:
        cell_id = int(particles_pos[i] / dx)
        if 0 <= cell_id[0] < res_x and 0 <= cell_id[1] < res_y:
            if cell_marked[cell_id] == 1:
                # Reset flow map to identity
                psi[i] = particles_pos[i]
                T_x[i] = ti.Vector.unit(2, 0)
                T_y[i] = ti.Vector.unit(2, 1)
                T_x_init[i] = ti.Vector.unit(2, 0)
                T_y_init[i] = ti.Vector.unit(2, 1)
                T_x_grad_m[i] = ti.Vector.unit(2, 0)
                T_y_grad_m[i] = ti.Vector.unit(2, 1)
                # Re-sample impulse from current grid velocity
                imp, _, _, _ = interp_u_MAC_grad_imp(u_x, u_y, particles_pos[i], dx)
                particles_imp[i] = imp
                particles_init_imp[i] = imp


@ti.kernel
def adaptive_density_redistribute(
    particles_pos: ti.template(), particles_imp: ti.template(),
    particles_init_imp: ti.template(),
    particles_active: ti.template(),
    cell_density: ti.template(),
    u_x: ti.template(), u_y: ti.template(),
    X: ti.template(), dx: float,
    particles_per_cell_high: int, particles_per_cell_low: int):
    """Adjust particle density per cell: deactivate excess particles
    in calm cells, activate particles in vortex cells.
    High-density cells: keep up to particles_per_cell_high active.
    Low-density cells: keep only particles_per_cell_low active."""

    total_active_particles[None] = 0

    for i in particles_pos:
        cell_id = int(particles_pos[i] / dx)
        if 0 <= cell_id[0] < res_x and 0 <= cell_id[1] < res_y:
            if cell_density[cell_id] == 1:
                # High-density cell: activate
                particles_active[i] = 1
            else:
                # Low-density cell: activate only first N particles
                # Use local index within cell
                local_idx = i % particles_per_cell  # approximate
                if local_idx < particles_per_cell_low:
                    particles_active[i] = 1
                else:
                    particles_active[i] = 0
        else:
            particles_active[i] = 0

        if particles_active[i] == 1:
            ti.atomic_add(total_active_particles[None], 1)


@ti.kernel
def count_active_particles(particles_active: ti.template(), total: ti.template()):
    total[None] = 0
    for i in particles_active:
        if particles_active[i] == 1:
            total[None] += 1


# ===================================================================
# METRIC KERNELS
# ===================================================================

@ti.kernel
def compute_kinetic_energy(u_x: ti.template(), u_y: ti.template(),
                            energy: ti.template()):
    """Total kinetic energy: E = 0.5 * Σ|u|² / N_cells"""
    energy[None] = 0.0
    for i, j in ti.ndrange(res_x, res_y):
        u_c = 0.5 * (u_x[i, j] + u_x[i + 1, j])
        v_c = 0.5 * (u_y[i, j] + u_y[i, j + 1])
        energy[None] += 0.5 * (u_c*u_c + v_c*v_c)
    energy[None] /= float(res_x * res_y)


@ti.kernel
def compute_enstrophy(w: ti.template(), enstrophy: ti.template()):
    """Total enstrophy: Ω = 0.5 * Σω² / N_cells"""
    enstrophy[None] = 0.0
    for i, j in w:
        enstrophy[None] += 0.5 * w[i, j] * w[i, j]
    enstrophy[None] /= float(res_x * res_y)


@ti.kernel
def compute_deformation_stats(cell_def: ti.template(),
                               avg_def: ti.template(), max_def: ti.template()):
    """Compute average and max deformation across all cells."""
    avg_def[None] = 0.0
    max_def[None] = 0.0
    cell_count = 0
    for i, j in cell_def:
        avg_def[None] += cell_def[i, j]
        ti.atomic_max(max_def[None], cell_def[i, j])
        cell_count += 1
    if cell_count > 0:
        avg_def[None] /= float(cell_count)


# ===================================================================
# MAIN
# ===================================================================
def main(from_frame=0, testing=False, total_steps_override=None):
    from_frame = max(0, from_frame)

    # Use override or hyperparameter
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

    # initial condition
    if from_frame <= 0:
        leapfrog_vel_func(u, X)
        split_central_vector(u, u_x, u_y)
        solver.Poisson(u_x, u_y)
        stripe_func(smoke, X, 0.25, 0.48)

    # Initialize particles
    init_particles_pos_uniform(particles_pos, X, res_x, particles_per_cell, dx,
                               particles_per_cell_axis, dist_between_neighbor)
    init_particles_imp_grad_m(particles_init_imp_grad_m, particles_pos, u_x, u_y,
                              init_C_x, init_C_y, dx)
    reset_T_to_identity(T_x_grad_m, T_y_grad_m)
    backup_particles_pos(particles_pos, particles_pos_backup)
    init_particles_imp(particles_imp, particles_init_imp, particles_pos, u_x, u_y, init_C_x, init_C_y, dx)
    current_particle_num[0] = initial_particle_num

    # Initialize all particles as active
    particles_active.fill(1)

    # Initial visualization
    get_central_vector(u_x, u_y, u)
    curl(u, w, dx)
    w_numpy = w.to_numpy()
    w_max = max(np.abs(w_numpy.max()), np.abs(w_numpy.min()))
    w_min = -1 * w_max
    write_field(w_numpy, vortdir, from_frame, particles_pos.to_numpy() / dx,
                vmin=w_min, vmax=w_max, plot_particles=plot_particles, dpi=dpi_vor)
    write_image(smoke.to_numpy(), smokedir, from_frame)

    # Compute initial vorticity magnitude for threshold calibration
    compute_cell_vorticity_mag(w, cell_vorticity_mag)
    vor_np = cell_vorticity_mag.to_numpy()
    vor_threshold = np.percentile(vor_np[vor_np > 0],
                                   100 * (1 - vorticity_threshold_ratio)) if vor_np.max() > 0 else 0.01
    print(f"[Adaptive] Vorticity threshold: {vor_threshold:.6f}")

    # Initialize deformation
    compute_particle_deformation(T_x, T_y, particle_deformation)
    aggregate_cell_deformation(particle_deformation, particles_pos,
                                cell_deformation, cell_deformation_count)

    if save_ckpt:
        np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame)), u_x.to_numpy())
        np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame)), u_y.to_numpy())
        np.save(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame)), smoke.to_numpy())

    # Simulation state
    sub_t = 0.
    frame_idx = from_frame
    last_output_substep = 0
    num_reinits = 0
    i = -1
    ik = 0

    # Use total_steps for metric collection
    frame_times = np.zeros(total_steps)
    metrics_kinetic_energy = np.zeros(total_steps)
    metrics_enstrophy = np.zeros(total_steps)
    metrics_active_particles = np.zeros(total_steps)
    metrics_marked_cells = np.zeros(total_steps)
    metrics_avg_deformation = np.zeros(total_steps)
    metrics_max_deformation = np.zeros(total_steps)
    metrics_reinit_type = np.zeros(total_steps)  # 0=none, 1=global, 2=adaptive_local
    metrics_cell_density_high = np.zeros(total_steps)  # number of high-density cells

    print(f"[Adaptive] Running {total_steps} steps with adaptive PFM")
    print(f"[Adaptive] Deformation threshold: {deformation_threshold}")
    print(f"[Adaptive] Adaptive reinit: {use_adaptive_reinit}")
    print(f"[Adaptive] Adaptive density: {use_adaptive_density}")

    while True:
        step_start_time = time.time()
        i += 1
        j = i % reinit_every
        i_next = i + 1
        j_next = i_next % reinit_every

        # Determine dt
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
        # ADAPTIVE REINIT LOGIC (replaces global reinit j==0)
        # ============================================================
        reinit_this_step = False
        reinit_type = 0  # 0=none, 1=global, 2=adaptive_local

        # Compute deformation factor (do this every step for metrics,
        # but only use for reinit decisions periodically)
        compute_particle_deformation(T_x, T_y, particle_deformation)
        aggregate_cell_deformation(particle_deformation, particles_pos,
                                    cell_deformation, cell_deformation_count)

        # Compute vorticity magnitude for density marking
        compute_cell_vorticity_mag(w, cell_vorticity_mag)

        # Periodic adaptive reinit check
        if use_adaptive_reinit and i > 0 and i % adaptive_reinit_interval == 0:
            # Mark cells by deformation
            mark_cells_adaptive(cell_deformation, cell_vorticity_mag,
                                cell_marked_reinit, cell_density_level,
                                deformation_threshold, vor_threshold)
            count_marked_cells(cell_marked_reinit, total_marked_cells)
            marked = total_marked_cells[None]
            total_cells = res_x * res_y

            if marked > 0:
                # Perform local reinit in marked cells only
                adaptive_local_reinit(particles_pos, particles_imp, particles_init_imp,
                                      psi, T_x, T_y, T_x_init, T_y_init,
                                      T_x_grad_m, T_y_grad_m,
                                      particles_active, cell_marked_reinit,
                                      u_x, u_y, dx)
                num_reinits += 1
                reinit_this_step = True
                reinit_type = 2  # adaptive local
                ik = i

                # Also reinit smoke only for marked region (approximate with copy_to)
                copy_to(smoke, init_smoke)

                print(f'[Adaptive] Local reinit: {marked}/{total_cells} cells '
                      f'({100.*marked/total_cells:.1f}%) at step {i}')

        # Fall back to global reinit if adaptive is off or as safety net
        if not use_adaptive_reinit and j == 0:
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
            reinit_this_step = True
            reinit_type = 1  # global

        # ============================================================
        # ADAPTIVE DENSITY (periodic update)
        # ============================================================
        if use_adaptive_density and i > 0 and i % adaptive_density_interval == 0:
            mark_cells_adaptive(cell_deformation, cell_vorticity_mag,
                                cell_marked_reinit, cell_density_level,
                                deformation_threshold, vor_threshold)
            adaptive_density_redistribute(particles_pos, particles_imp,
                                          particles_init_imp, particles_active,
                                          cell_density_level,
                                          u_x, u_y, X, dx,
                                          particles_per_cell_high, particles_per_cell_low)
            count_active_particles(particles_active, total_active_particles)
            print(f'[Adaptive] Density updated: {total_active_particles[None]}/{particle_num} '
                  f'particles active ({100.*total_active_particles[None]/particle_num:.1f}%)')
        elif not use_adaptive_density:
            # All particles always active
            particles_active.fill(1)

        # ============================================================
        # GRAD-M REINIT (short-distance mapping, kept global)
        # ============================================================
        k = (i - ik) % reinit_every_grad_m
        if k == 0:
            init_particles_imp_grad_m(particles_init_imp_grad_m, particles_pos, u_x, u_y,
                                      init_C_x, init_C_y, dx)
            reset_T_to_identity(T_x_grad_m, T_y_grad_m)
            copy_to(T_x, T_x_init)
            copy_to(T_y, T_y_init)
            print(f'reinit short-distance mapping at step {i}')

        # ============================================================
        # STANDARD SIMULATION STEP
        # ============================================================
        # Midpoint velocity
        if use_midpoint_vel:
            reset_to_identity_grid(psi_x_grid, psi_y_grid, T_x_grid, T_y_grid)
            RK4_grid(psi_x_grid, T_x_grid, u_x, u_y, 0.5 * curr_dt)
            RK4_grid(psi_y_grid, T_y_grid, u_x, u_y, 0.5 * curr_dt)
            copy_to(u_x, tmp_u_x)
            copy_to(u_y, tmp_u_y)
            advect_u(tmp_u_x, tmp_u_y, u_x, u_y, T_x_grid, T_y_grid, psi_x_grid, psi_y_grid, dx)
            solver.Poisson(u_x, u_y)

        # Evolve grad_m and particles
        stretch_T_and_advect_particles(particles_pos, T_x_grad_m, T_y_grad_m, u_x, u_y, curr_dt)
        update_particles_grad_m(C_x, C_y, init_C_x, init_C_y, T_x_grad_m, T_y_grad_m)

        # Update composite T and impulses
        update_T(T_x, T_y, T_x_init, T_y_init, T_x_grad_m, T_y_grad_m)
        update_particles_imp(particles_imp, particles_init_imp, T_x, T_y)

        # P2G with active-only particles
        P2G(particles_imp, particles_pos, u_x, u_y, psi, psi_x_grid, psi_y_grid,
            p2g_weight_x, p2g_weight_y, particles_active)
        solver.Poisson(u_x, u_y)
        advect_smoke(init_smoke, smoke, psi_x_grid, psi_y_grid, dx)

        # ============================================================
        # METRIC COLLECTION
        # ============================================================
        step_end_time = time.time()
        frame_times[i] = step_end_time - step_start_time

        # Kinetic energy
        compute_kinetic_energy(u_x, u_y, kinetic_energy_val)
        metrics_kinetic_energy[i] = kinetic_energy_val[None]

        # Enstrophy
        get_central_vector(u_x, u_y, u)
        curl(u, w, dx)
        compute_enstrophy(w, enstrophy_val)
        metrics_enstrophy[i] = enstrophy_val[None]

        # Deformation stats
        compute_deformation_stats(cell_deformation, avg_deformation_val, max_deformation_val)
        metrics_avg_deformation[i] = avg_deformation_val[None]
        metrics_max_deformation[i] = max_deformation_val[None]

        # Active particles
        count_active_particles(particles_active, total_active_particles)
        metrics_active_particles[i] = total_active_particles[None]

        # Marked cells
        count_marked_cells(cell_marked_reinit, total_marked_cells)
        metrics_marked_cells[i] = total_marked_cells[None]

        # Reinit type
        metrics_reinit_type[i] = reinit_type

        # Count high-density cells
        high_count = np.sum(cell_density_level.to_numpy())
        metrics_cell_density_high[i] = high_count

        print(f"[Step {i}] time={frame_times[i]:.4f}s KE={metrics_kinetic_energy[i]:.6f} "
              f"Enstrophy={metrics_enstrophy[i]:.6f} Active={int(metrics_active_particles[i])} "
              f"Marked={int(metrics_marked_cells[i])} Deform_avg={metrics_avg_deformation[i]:.4f} "
              f"Reinit={'adaptive' if reinit_type==2 else 'global' if reinit_type==1 else 'none'}")

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
    # SAVE METRICS
    # ================================================================
    print(f"\n[Adaptive] Saving metrics to {metrics_dir}")

    # Trim to actual steps taken
    n_steps = i + 1
    metrics = {
        'frame_times': frame_times[:n_steps],
        'kinetic_energy': metrics_kinetic_energy[:n_steps],
        'enstrophy': metrics_enstrophy[:n_steps],
        'active_particles': metrics_active_particles[:n_steps],
        'marked_cells': metrics_marked_cells[:n_steps],
        'avg_deformation': metrics_avg_deformation[:n_steps],
        'max_deformation': metrics_max_deformation[:n_steps],
        'reinit_type': metrics_reinit_type[:n_steps],
        'cell_density_high': metrics_cell_density_high[:n_steps],
        'total_cells': res_x * res_y,
        'total_particles': particle_num,
        'deformation_threshold': deformation_threshold,
        'vorticity_threshold': vor_threshold,
        'particles_per_cell_high': particles_per_cell_high,
        'particles_per_cell_low': particles_per_cell_low,
        'exp_name': exp_name,
        'mode': 'adaptive',
    }
    np.savez(os.path.join(metrics_dir, 'metrics.npz'), **metrics)

    # Also save frame times separately (compatible with original format)
    frame_time_dir = os.path.join(logsdir, 'frame_time')
    os.makedirs(frame_time_dir, exist_ok=True)
    np.save(os.path.join(frame_time_dir, 'frame_times.npy'), frame_times[:n_steps])

    print(f"[Adaptive] Complete. {n_steps} steps simulated.")
    print(f"[Adaptive] Metrics saved to {metrics_dir}/metrics.npz")
    return metrics


if __name__ == '__main__':
    print("[Adaptive PFM] Begin")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=None, help='Total steps to run')
    parser.add_argument('--deformation-threshold', type=float, default=deformation_threshold)
    parser.add_argument('--no-adaptive-reinit', action='store_true')
    parser.add_argument('--no-adaptive-density', action='store_true')
    args = parser.parse_args()

    if args.deformation_threshold is not None:
        deformation_threshold = args.deformation_threshold
    if args.no_adaptive_reinit:
        use_adaptive_reinit = False
    if args.no_adaptive_density:
        use_adaptive_density = False

    main(from_frame=from_frame, testing=testing, total_steps_override=args.steps)
    print("[Adaptive PFM] Complete")
