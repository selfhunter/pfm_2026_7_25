# some hyperparameters
dim = 2
testing = False
use_neural = False   #是否使用神经网络辅助模拟
save_ckpt = True
save_frame_each_step = False
use_BFECC = False
use_midpoint_vel = True
use_APIC = True
forward_update_T = True
plot_particles = False
use_reseed_particles = False
reinit_particle_pos = True
dpi_vor = 512 if plot_particles else 512 // 8

# encoder hyperparameters
min_res = (128, 32)
num_levels = 4
feat_dim = 2
activate_threshold = 0.03
# neural buffer hyperparameters
N_iters = 2000
N_batch = 40000 #25000
success_threshold = 3.e-8
# simulation hyperparameters
res_x = 1024
res_y = 256
visualize_dt = 0.1
reinit_every = 20
reinit_every_grad_m = 8
ckpt_every = 1
CFL = 1.0
from_frame = 0
total_frames = 5000
use_total_steps = False
total_steps = 1
exp_name = "2D_leapfrog_1024x256_reinit-20-8"

particles_per_cell = 16
total_particles_num_ratio = 1
cell_max_particle_num_ratio = 2.0
# min_particles_per_cell_ratio = 0.75
# min_particles_per_cell = int(particles_per_cell * min_particles_per_cell_ratio)
min_particles_per_cell_ratio = 1
min_particles_per_cell = int(particles_per_cell * min_particles_per_cell_ratio)
max_particles_per_cell_ratio = 1.6
max_particles_per_cell = int(particles_per_cell * max_particles_per_cell_ratio)

max_delete_particle_try_num = 100000



# 一、基础配置参数
# dim = 2
# 含义：模拟的空间维度
# 用途：指定是 2D（二维）还是 3D（三维）模拟
# 影响：改为 3D 会大幅增加计算量，需要调整网格、粒子数量等参数以适应三维空间
# testing = False
# 含义：是否启用测试模式
# 用途：测试模式通常会简化计算（如减少迭代次数、降低精度），用于快速验证代码正确性
# 影响：设为True时，模拟可能不完整但运行更快，结果不用于正式分析
# use_neural = False
# 含义：是否使用神经网络辅助模拟
# 用途：可能用于替代部分物理模型（如湍流建模）或加速计算
# 影响：设为True时，需要加载神经网络模型，计算逻辑会更复杂，但可能提升模拟精度或效率（取决于网络设计）
# save_ckpt = True
# 含义：是否保存检查点（checkpoint）
# 用途：定期保存模拟状态，便于中断后恢复或后续分析
# 影响：设为False可节省存储空间，但中断后需从头运行；True则会占用更多磁盘空间
# save_frame_each_step = False
# 含义：是否每步都保存可视化帧
# 用途：控制输出频率，避免存储过多冗余帧
# 影响：设为True会生成大量帧文件（占用空间），但可观察每一步的细节；False则按visualize_dt间隔保存
# use_BFECC = False
# 含义：是否使用 BFECC（Boundary-Fitted Error Correction and Correction）算法
# 用途：BFECC 是一种高阶精度的数值格式，用于减少对流项计算的数值耗散
# 影响：启用后模拟精度更高（尤其对界面追踪），但计算量增加
# use_midpoint_vel = True
# 含义：是否使用中点速度（中间时刻的速度）
# 用途：在蛙跳法（leapfrog）等时间积分中，中点速度可提高时间精度（如二阶精度）
# 影响：设为False可能降低时间精度，导致模拟结果更易发散
# use_APIC = True
# 含义：是否使用 APIC（Affine Particle-in-Cell）方法
# 用途：APIC 是一种粒子 - 网格混合方法，比传统 PIC 更精确地传递粒子动量
# 影响：关闭后可能使用 PIC 或 FLIP 方法，精度降低但计算更快
# forward_update_T = True
# 含义：是否向前更新温度（或标量场 T）
# 用途：控制标量场（如温度、浓度）的更新方向
# 影响：改为False可能采用向后差分，稳定性不同但精度可能变化
# plot_particles = False
# 含义：是否可视化粒子
# 用途：控制输出图像中是否显示粒子（而非仅显示网格结果）
# 影响：True时可视化更直观但可能更杂乱，且渲染时间增加
# use_reseed_particles = False
# 含义：是否重新播种粒子（补充新粒子）
# 用途：当粒子分布不均时，补充新粒子以维持模拟精度
# 影响：True可避免粒子稀疏区域的误差，但会增加粒子总数和计算量
# reinit_particle_pos = True
# 含义：是否重新初始化粒子位置
# 用途：定期重置粒子位置以避免累积误差（尤其在长时间模拟中）
# 影响：False可能导致粒子分布逐渐失真，影响模拟稳定性
# dpi_vor = 512 if plot_particles else 512 // 8
# 含义：涡度（vorticity）可视化的分辨率（DPI）
# 用途：控制输出图像的清晰度
# 影响：值越大图像越清晰，但存储和渲染时间增加；plot_particles=True时自动提高分辨率（因粒子细节需要更高清晰度）
# 二、编码器参数（可能用于数据压缩或特征提取）
# min_res = (128, 32)
# 含义：最小分辨率（宽 × 高）
# 用途：可能是编码器输入的最低分辨率，用于多尺度特征提取
# 影响：调大可能增加特征提取精度，但计算量上升
# num_levels = 4
# 含义：多尺度处理的层级数
# 用途：在金字塔式特征提取中，控制尺度层级数量
# 影响：层级越多，能捕捉的尺度范围越广，但计算复杂度显著增加
# feat_dim = 2
# 含义：特征维度
# 用途：编码器输出的特征向量维度
# 影响：增大可保留更多信息，但会增加存储和计算负担
# activate_threshold = 0.03
# 含义：激活阈值
# 用途：控制特征激活的灵敏度（如高于阈值的特征才被保留）
# 影响：值越小保留的特征越多（精度可能更高），但噪声也会增加；值越大则特征更稀疏
# 三、神经缓冲区参数（若使用神经网络）
# N_iters = 2000
# 含义：神经网络训练迭代次数
# 用途：控制训练时长
# 影响：增加迭代次数可能提高网络精度，但可能过拟合且训练时间延长
# N_batch = 40000
# 含义：批处理大小
# 用途：每次训练迭代输入的样本数量
# 影响：增大可提高训练稳定性，但需更多内存；减小则训练更快但收敛可能不稳定
# success_threshold = 3.e-8
# 含义：训练成功的误差阈值
# 用途：当损失函数低于该值时，认为训练达标
# 影响：值越小要求精度越高，但训练时间可能大幅增加
# 四、模拟核心参数
# res_x = 1024，res_y = 256
# 含义：网格分辨率（x 和 y 方向的网格数）
# 用途：决定模拟的空间精度（网格越密，细节越丰富）
# 影响：分辨率提高会显著增加计算量（与网格数成正比），但能捕捉更小尺度的物理现象
# visualize_dt = 0.1
# 含义：可视化输出的时间间隔
# 用途：控制保存帧的频率（单位：模拟时间）
# 影响：值越小输出帧越多，可观察更精细的变化，但占用更多存储空间
# reinit_every = 20
# 含义：粒子位置重新初始化的间隔步数
# 用途：与reinit_particle_pos配合，定期重置粒子以避免误差累积
# 影响：值越小（更频繁重新初始化），稳定性越好但计算量增加；值越大可能导致粒子分布失真
# reinit_every_grad_m = 8
# 含义：梯度相关的重新初始化间隔
# 用途：可能针对梯度较大的区域（如强剪切区）更频繁地重新初始化
# 影响：值越小，对梯度剧烈变化区域的处理越精细，但计算量增加
# ckpt_every = 1
# 含义：保存检查点的间隔步数
# 用途：控制检查点的保存频率
# 影响：值越小保存越频繁，恢复时丢失的数据越少，但占用更多磁盘空间
# CFL = 1.0
# 含义：CFL 条件系数（Courant-Friedrichs-Lewy）
# 用途：控制时间步长dt的上限（dt ≤ CFL × 网格尺寸 / 最大速度），保证数值稳定性
# 影响：值越小，时间步长越小，模拟更稳定但总步数增加（计算变慢）；值接近 1 时效率最高，但可能在高速流动中不稳定
# from_frame = 0
# 含义：从第几帧开始模拟
# 用途：支持从中间帧继续模拟（需对应检查点）
# 影响：修改为非 0 值时，需确保存在对应帧的检查点文件
# total_frames = 5000
# 含义：总模拟帧数
# 用途：控制模拟的总时长（总时间 = total_frames × visualize_dt）
# 影响：增加会延长模拟时间，捕捉更长的物理过程，但计算成本上升
# use_total_steps = False，total_steps = 1
# 含义：是否用总步数替代总帧数控制模拟时长
# 用途：use_total_steps=True时，模拟会运行total_steps步后停止，而非按帧数
# 影响：提供更灵活的终止条件，适合不需要固定帧率输出的场景
# 五、粒子管理参数
# particles_per_cell = 16
# 含义：每个网格单元的目标粒子数
# 用途：控制粒子密度（粒子总数 ≈ 网格数 × 该值）
# 影响：值越大，粒子对流体的采样越密集，精度越高，但计算量和内存占用显著增加
# total_particles_num_ratio = 1
# 含义：总粒子数比例（相对于目标值）
# 用途：整体缩放粒子总数（如=0.5则粒子数减半）
# 影响：减小可降低计算量，但可能影响模拟精度
# cell_max_particle_num_ratio = 2.0，max_particles_per_cell = int(particles_per_cell × max_particles_per_cell_ratio)
# 含义：每个网格允许的最大粒子数比例及计算值
# 用途：限制网格内粒子数量上限，避免局部粒子过于密集
# 影响：比例调大可能允许更多粒子（提高精度），但可能导致局部计算负担过重
# min_particles_per_cell_ratio = 1，min_particles_per_cell = int(particles_per_cell × min_particles_per_cell_ratio)
# 含义：每个网格允许的最小粒子数比例及计算值
# 用途：确保网格内粒子数量不低于阈值，避免信息丢失
# 影响：比例调大（如=1.5）会增加粒子总数，提高稀疏区域的精度，但计算量上升
# max_delete_particle_try_num = 100000
# 含义：删除粒子时的最大尝试次数
# 用途：当粒子超过最大数量时，尝试删除多余粒子的上限次数（避免无限循环）
# 影响：值过小可能导致无法删除足够粒子（粒子数超限），值过大则可能浪费计算时间
# 总结
# 这些参数共同控制了模拟的精度、效率、稳定性和输出方式：

# 精度相关：res_x/res_y（网格）、particles_per_cell（粒子密度）、use_BFECC（高阶格式）等；
# 效率相关：CFL（时间步长）、N_batch（批大小）、dpi_vor（分辨率）等；
# 稳定性相关：reinit_every（粒子重置）、CFL（时间步控制）等；
# 输出控制：save_ckpt、save_frame_each_step、total_frames等。