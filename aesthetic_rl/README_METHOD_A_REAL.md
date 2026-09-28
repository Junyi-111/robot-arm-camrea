# ArtiMuse 特征增强 SAC：真机运行手册

当前离线准备已完成：1009 帧缓存、510 个训练 transition、200 步 BC checkpoint。
恢复本次准备结果请按 [首次真机启动单](METHOD_A_START_TOMORROW.md) 操作，
learner 使用 `--resume`。检查记录见 `method_a_artifacts/readiness.json`。
这些是软件与数据验证结果，真机现场预检仍待连接设备后完成。

这套实现位于 `aesthetic_rl/method_a`，不会修改原来的 `hil-serl`、
`aesthetic_rl_revise` 或原始示范数据。它实现的是已经确认的方法 A：保留
RGB 编码器，将冻结 ArtiMuse 的最后一层多模态隐藏状态（3584 维）经共享、
可训练的 `3584 → 512 → 256` projector 后，与 6 维关节状态、7 维目标框状态
融合，输入 4 维 SAC，动作只映射到 J1/J2/J3/J5。

当前 reward v2 使用 demo 的固定分布参数（均值约32.02、标准差约7.99）：

```text
r_abs   = clip((score_next - 32) / 8, -1, 1)
r_delta = clip((score_next - score_now) / 5, -1, 1)
r_aesthetic = 0.8 * r_abs + 0.2 * r_delta
r_total = r_aesthetic + 0.25 * r_target - 0.02 * r_motion - 0.05 * r_smooth
```

因此当前图片质量是主项；低于28分的 demo 即使相对上一帧提高，也没有正总
reward。`r_target` 的触边值为-1（对总 reward 最多贡献-0.25）；完全无框时
`r_target=-4`（有效贡献-1）并立即终止 episode。

## 已固定的安全行为

- 导入代码、预检和启动 PiPER driver 都不会使能或移动机械臂。
- 真机写入要求同时使用 `--arm` 和环境变量
  `PIPER_ARM_WRITES=I_ACCEPT_REAL_MOTION`。
- PiPER driver 必须以 `auto_enable:=false` 启动。
- `X`、关闭 UI、训练步数用尽、远端视觉失败、相机失败或 learner 断线都会
  停止策略并发送当前位置保持命令；不会自动失能。
- 因机械臂失能会下坠，失能只允许在 `STOPPED_HOLD` 或 `FAULT_HOLD` 状态：
  第一次按 `D` 开始 5 秒警告，倒计时结束、人工托住机械臂后再按一次 `D`
  才调用 `/enable_srv false`。`Esc` 可取消。
- 第一次使能及每个 episode 结束后，六个关节都以 3°/s 慢速返回配置 home；
  J4/J6 固定为 0°以保持腕部相机水平，只有 J1/J2/J3/J5 再随机 ±2°并参与
  SAC/人工动作。复位完成必须按 `Space` 确认后才继续。
- SAC 不控制夹爪。配置会把夹爪固定保持为 demo 采集程序使用的 0.07 m；首次
  运行前确认这个开度与当前腕部相机安装方式一致。PiPER 的使能服务会先发出
  一次零开度，actor 在服务确认后会立即用保持指令恢复 0.07 m。
- 每次新画面显示后保留 0.75 秒人工判定窗口，便于在理想位置按 `O`，把刚才
  到达理想位置的 transition 正确标成终止，而不是再执行一步。
- 人工接管采用离散点动：每次按键最多进入一个动作，必须松开按键并等待当前
  运动和视觉观测完成后才能接受下一次。执行期间的重复键和额外键不会排队；
  当前点动未完成时按 `B` 也不会提前恢复策略。
- Grounding DINO 只要返回通过置信度/面积阈值的框，episode 就继续；框进入
  画面边缘 5% 区域时按边缘距离连续扣分（安全区为0，触边对总 reward
  贡献-0.25），不会仅因靠边终止。动作后完全没有合格框时，该 transition
  记为终止并立即复位。
- 配置中的关节范围是从现有 demo 得到的保守初值，不是机械臂厂家极限。
  首次真机运动前必须对照当前姿态与现场障碍物确认。

## 当前已验证结果

- 原始 demo：10 条、523 个 transition、10 个真实 episode 终点。
- 发现 13 个包含 J4 运动的 transition；构建新数据时会排除，不会改原文件。
- 1009 个唯一画面的远端特征已缓存；按新目标语义重建后的 510 个 demo
  transition 中，191 个框处于 5% 安全区，319 个框获得连续边缘惩罚。
- 本机 RTX 4060 8GB 已安装并识别 JAX CUDA；正式 ResNet、batch 64 的合成
  测试中，首次 BC/SAC 编译分别约 6/7 秒，稳定更新约 0.02 秒。
- actor/learner 的 Agentlace 网络下发、在线数据和接管数据上传已通过回环测试。
- 真机 CAN、D435、远端视觉、复位路径和短 episode 已在现场跑通；旧规则最终
  保存了 49 个 transition，并进行了 49 次 SAC 更新，其中早期 episode 因连续
  三帧“框靠边”被误判为目标丢失而提前结束。
- 修改目标语义后的首次5个 episode 保存了269条 transition，其中127条接管
  transition 受到键盘重复/排队影响。该运行及469步 checkpoint 被完整保留，
  但不会继续用于训练；修复后从纯 BC 200步 checkpoint 重新开始。
- 离散点动修复后的5个 episode 共400条 transition、130条接管数据并训练到
  checkpoint 600；策略仍有动作饱和和低分区域徘徊，因此 reward v2 再次从
  纯 BC 200步 checkpoint 开始，旧运行保留但不混入。

## 一次性远端配置

先关闭 VPN。以下命令均从本地项目根目录执行。

1. 上传服务代码（不会上传 demo、checkpoint 或日志）：

   ```bash
   bash aesthetic_rl/scripts/method_a_upload_remote.sh
   ```

2. 登录服务器并安装 Grounding DINO：

   ```bash
   ssh -p 22 hjy@10.120.17.111
   cd /home/hjy/robot_aesthetic_rl
   bash aesthetic_rl/scripts/method_a_remote_install_grounding.sh
   ```

   源码与权重分开存放：源码默认在
   `/home/hjy/robot_aesthetic_rl/third_party/GroundingDINO`，约 694 MB 的
   checkpoint 默认在
   `/datasets/hjy/robot_aesthetic_models/GroundingDINO/groundingdino_swint_ogc.pth`。
   官方配置依赖的 `bert-base-uncased`（约 440 MB）会保存在
   `/datasets/hjy/robot_aesthetic_models/GroundingDINO/huggingface`；启动服务时
   强制离线读取，不会再次访问 Hugging Face。
   安装脚本固定使用 `transformers==4.41.2`，因为 Grounding DINO 的
   `BertModelWarper` 仍依赖后续版本不再保证的 `BertModel.get_head_mask` API。
   如果服务器 Conda 和系统自带的 `libstdc++.so.6` 都缺少 `GLIBCXX_3.4.29`，脚本会仅在 `groundingdino` 环境中自动升级 `libstdcxx-ng` 和 `libgcc-ng`，不会重新下载已经校验通过的权重。
   已完整下载的 `.part` 会校验后复用，不会重复下载 checkpoint。

3. 每次运行前在服务器执行 `nvidia-smi`。若 7、6 仍空闲，分别开两个远端终端：

   ```bash
   cd /home/hjy/robot_aesthetic_rl
   bash aesthetic_rl/scripts/method_a_remote_start_artimuse.sh 7
   ```

   服务器上的 ArtiMuse 使用已经验证过的 `isaaclab51` Conda 环境；本地电脑才
   使用名为 `artimuse` 的环境。

   ```bash
   cd /home/hjy/robot_aesthetic_rl
   bash aesthetic_rl/scripts/method_a_remote_start_grounding.sh 6
   ```

   如果 GPU 占用变化，把末尾的编号换成当时空闲卡。脚本通过
   `CUDA_VISIBLE_DEVICES` 隔离物理卡，服务进程内部统一使用 `cuda:0`。

4. 本地关闭 VPN，保持第三个终端中的 SSH 隧道运行：

   ```bash
   bash aesthetic_rl/scripts/method_a_open_tunnel.sh
   ```

   `/analyze` 是 ArtiMuse 的单次“评分 + 3584 维特征”接口；`/detect` 是
   Grounding DINO 的花朵框接口。配置里的 18080/18081 是本地隧道端口，不是
   公网地址。

5. 用一张图验证完整返回并写 JSON：

   ```bash
   conda activate artimuse
   cd /home/junyi/robot_aesthetic_rl
   export PYTHONPATH=$PWD
   python -m aesthetic_rl.scripts.method_a_vision_smoke \
     aesthetic_rl/data/portrait/information_test.jpg \
     --output aesthetic_rl/method_a_artifacts/vision_smoke/information_test.json
   ```

## 一次性准备 demo

保持两个远端服务和 SSH 隧道运行。预计算约需 1009 次请求，可能需要较长
时间；中断后原命令重跑会跳过完整缓存。

```bash
conda activate artimuse
cd /home/junyi/robot_aesthetic_rl
export PYTHONPATH=$PWD
python -m aesthetic_rl.scripts.method_a_audit_demos
python -m aesthetic_rl.scripts.method_a_precompute_features
python -m aesthetic_rl.scripts.method_a_prepare_demos
python -m aesthetic_rl.scripts.method_a_doctor --require-learner-gpu
```

最后一条必须显示：`prepared_demo: true`、`resnet_weights: true`、
`learner_gpu: true`。`prepare_demos` 拒绝覆盖已有产物；如果确实需要重建，先
把旧产物移到备份位置。

## 真机只读预检

先启动 driver，但不要使能：

```bash
cd /home/junyi/robot_aesthetic_rl
bash aesthetic_rl/scripts/method_a_start_driver.sh
```

另开终端执行只读预检。该命令只订阅关节反馈、拍一帧并访问远端视觉：

```bash
source /opt/ros/humble/setup.bash
source /home/junyi/handeye/install/setup.bash
source /home/junyi/miniconda3/etc/profile.d/conda.sh
conda activate artimuse
cd /home/junyi/robot_aesthetic_rl
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python -m aesthetic_rl.scripts.method_a_actor --preflight --check-robot-feedback
```

必须检查输出中的相机尺寸、3584 维特征、`target_present` 和
`target_edge_clear`，以及
`controlled_joints_within_startup_bounds: true`。该门禁只检查学习控制的
J1/J2/J3/J5；失能下坠的 J4/J6 不阻止启动，而会在首次使能后慢速归零以调平
相机。失能下坠时，首次检查的
`controlled_joints_within_policy_workspace` 可以为 false，actor 会在第一次 Space 后
慢速返回训练区。若 startup bounds 为 false，不要启动 actor；
先根据现场安全姿态修改 `aesthetic_rl/configs/method_a.yaml`。

## 正式训练启动顺序

保持“远端 ArtiMuse、远端 Grounding DINO、SSH 隧道、PiPER driver”四个进程
运行，再开本地 learner：

```bash
cd /home/junyi/robot_aesthetic_rl
bash aesthetic_rl/scripts/method_a_start_learner.sh
```

看到 `[LEARNER] serving ports 5588/5589` 后，托住机械臂并在最后一个终端
显式解锁 actor：

```bash
cd /home/junyi/robot_aesthetic_rl
export PIPER_ARM_WRITES=I_ACCEPT_REAL_MOTION
bash aesthetic_rl/scripts/method_a_start_actor.sh
```

actor 完成相机、远端视觉、ROS 反馈、learner policy 五项预检后停在
`WAIT_ARM`。此时按 `Space` 才会使能并慢速复位；到达随机初始位后停在
`WAIT_CONFIRM`，再次按 `Space` 才开始 episode。只要 `target_present=true`
即可开始；若 `target_edge_clear=false`，UI 会用橙框提示边缘惩罚已生效。

### UI 操作

| 操作 | 功能 |
|---|---|
| `Space` | 在 `WAIT_ARM` 使能/复位；在 `WAIT_CONFIRM` 开始 episode |
| `P` | 暂停并保持；再次按下恢复策略 |
| `O` | 提前终止当前 episode，复位后等待确认 |
| `A` | 进入人工接管 |
| `B` | 退出人工接管，恢复策略 |
| `W/S` | 接管时 J1 正/负；点按一次后松开并等待完成 |
| `E/D` | 接管时 J2 正/负；点按一次后松开并等待完成 |
| `R/F` | 接管时 J3 正/负；点按一次后松开并等待完成 |
| `T/G` | 接管时 J5 正/负；点按一次后松开并等待完成 |
| `X` 或关闭窗口 | 停止训练，保持使能并保持当前位置 |
| `D` | 仅停止/故障状态下启动或确认失能 |
| `Esc` | 取消失能倒计时 |

按键也做成了可点击按钮；J2 的 `D` 只在人工接管状态解释为负向动作，在停止
状态才解释为失能确认。

## STOP 前的固定姿态评分噪声校准

专用脚本 `scripts/method_a_calibrate_score_noise.sh` 是只读程序：它使用
`PiperRos2(..., allow_writes=False)`，启动脚本还会删除
`PIPER_ARM_WRITES`，因此不会使能、失能、保持或移动机械臂。运行前必须由操作员
确保机械臂已使能并稳定 HOLD 在高分姿态，同时停止 actor 释放 D435；ROS driver、
远端视觉服务和 SSH 隧道需要保持运行。

默认实验先做5次不记录的视觉 warmup，再对同一张参考图片评分30次，然后对固定
姿态下30张新 D435 帧评分。它同时保存关节反馈、目标框几何、推理耗时和每张新
图；任一关节相对基准漂移超过0.1度会立即终止。运行命令：

```bash
cd /home/junyi/robot_aesthetic_rl
bash aesthetic_rl/scripts/method_a_calibrate_score_noise.sh
```

结果位于 `method_a_artifacts/score_noise_calibration/<timestamp>/`。
`summary.json` 给出相同图片噪声、新帧噪声、3帧中位数后的P95相邻差以及建议的
`improvement_epsilon` 和 `deterioration_threshold`。只有
`measurement_valid=true` 时才能直接采用这些建议参数。

## 冻结 checkpoint 1731 的确定性评估

learner 和 actor 必须成对加 `--eval-only`。该模式固定加载
`checkpoint_00001731.msgpack`，强制 actor 使用策略均值动作，不执行 BC/SAC
更新，不上传 transition，也不写 learner 指标、replay 或新 checkpoint。双方会
校验运行模式和 checkpoint 路径，误把训练端与评估端混接时会拒绝运行。评估期间
人工接管仍作为安全手段保留，但对应 episode 应标记为非纯自主评估；即使接管，
transition 也不会进入训练 replay。评估图片和事件单独写入
`aesthetic_rl/method_a_artifacts/evaluation_checkpoint_00001731/`。

J3 的策略/人工动作上限已按真机反馈从 +5° 修正为 0°；+10° 的 startup 上限只
用于接受失能下坠后的初始反馈和安全复位，不允许策略探索到该区域。

### 影子 STOP v2（不会真正回位或停止）

影子 STOP 只允许和冻结 `--eval-only` 模式一起运行。它使用3帧评分中位数、
`epsilon=0.818636`、最低搜索10步、15步 stagnation patience 和连续3次候选，
记录策略“本来会在哪一步结束搜索”。Grounding DINO 只要求目标框存在；框是否靠近
边缘不参与 STOP 门控，构图选择主要交给 ArtiMuse。人工接管会重置当前影子判据。

v2 还会为每一帧保存对应的真实六关节姿态，并缓存目标存在时的最高单帧 ArtiMuse
分数、精确 step 和图像。触发后记录 `SHADOW WOULD_RETURN_TO_BEST`，表示如果进入
真实 STOP，应该返回哪个已经访问过的精确姿态，而不是停在当前姿态。单帧最高分
只用于选择待确认位置；后续真实版本仍必须在返回后固定姿态重新评分3次。

影子模式不会 HOLD、不会执行回位、不会提前结束 episode、不会改变策略动作，也
不会上传 replay。终端出现 `[SHADOW WOULD_RETURN_TO_BEST]` 后，当前 episode 仍会
正常跑到80步、目标丢失或操作员按 `O`。每一步的判据写入 transition 的
`shadow_stop` 字段，每个 episode 另写 `shadow_stop_episode_summary`。actor 启动命令为：

```bash
export PIPER_ARM_WRITES=I_ACCEPT_REAL_MOTION
bash aesthetic_rl/scripts/method_a_start_shadow_eval.sh
```

learner 仍须在另一个终端用 `method_a_start_learner.sh --eval-only` 成对启动。
运行5--10个不接管 episode 后，再统计误停止、漏停止、停止步数和停止后潜在增益；
影子模式通过前不要改成真正自动停止。

### 受控真实 STOP v3（冻结评估专用）

`--active-stop` 与 `--shadow-stop` 相互排斥，而且只能和 `--eval-only` 使用。
触发判据仍与影子 v2 完全相同，目标只要求存在，边缘净空不作为 STOP 门控。
触发后系统冻结历史最高单帧分数对应的六关节反馈姿态，并执行以下保守流程：

1. 确认六关节目标均位于策略工作区；
2. 使用真机已有的 guarded interpolator，以 2 deg/s 返回该姿态；
3. 检查到位后的最大六关节反馈误差不超过 0.75 deg；
4. 保持姿态，重新采集3张独立图像；
5. 只有3张图都检测到目标且 ArtiMuse 分数中位数不低于40，才结束 episode；
6. 确认失败但目标仍存在时恢复策略；目标丢失则结束本 episode；任何运动、反馈或
   视觉异常进入故障 HOLD。整个流程从不自动失能。

首轮每个 episode 最多执行一次真实回位。`X` 在回位/复评期间仍会取消并 HOLD，
`O` 会取消当前回位并结束 episode。启动命令：

```bash
export PIPER_ARM_WRITES=I_ACCEPT_REAL_MOTION
bash aesthetic_rl/scripts/method_a_start_active_stop_eval.sh
```

关键事件包括 `active_stop_triggered`、`active_stop_return_started`、
`active_stop_return_completed`、`active_stop_verification_sample`、
`active_stop_verification_result` 和 `active_stop_episode_summary`。首次只运行3--5个
episode，并全程在机械臂旁观察返回路径；确认回位和复评稳定前不要用于训练模式。

## 产物与恢复

- prepared demo：`aesthetic_rl/method_a_artifacts/demos/method_a_demo_absolute_v2.pkl`
- checkpoint：`aesthetic_rl/method_a_artifacts/checkpoints_absolute_v2/`，从纯 BC
  200步 checkpoint 开始。
- 在线/接管 replay、训练指标与真机事件/图片：
  `aesthetic_rl/method_a_artifacts/runtime_absolute_v2/`。三个旧版本分别保留在
  `runtime/`、`runtime_presence/`、`runtime_discrete_jog/`，不会混入 reward v2。

learner 异常结束后，重新启动可加 `--resume`：

```bash
bash aesthetic_rl/scripts/method_a_start_learner.sh --resume
```

actor 不自动恢复运动，仍需重新通过 `WAIT_ARM` 和两次 `Space`。Reward v2 的
真机评估以累计在线 transition 为主、episode 数为辅（提前终止会让 episode
长度不同）：

- 400 step（最多约 5 个满长 episode）：只作安全与实现检查；若持续撞关节边界、
  丢框或动作明显单向饱和，先停机排查。
- 800 step（最多约 10 个满长 episode）：首次趋势检查，比较前后阶段的 ArtiMuse
  分数、边缘净空、接管率和动作饱和率。
- 1200 step（最多约 15 个满长 episode）：当天主要 go/no-go 检查点；另外运行至少
  3 个不接管的完整 episode，避免把人的纠正误判为策略效果。
- 1600 step（最多约 20 个满长 episode）：若 1200 step 仍不明确才继续；到这里若
  分数/净空无改善、接管率不降或动作仍饱和，则停止本轮，不盲目跑满 2000 step。

由于复位带随机扰动，不能只看单张最高分图。至少同时检查 episode 平均/末步
分数、框的边缘净空、无接管 episode 的表现、接管 transition 比例以及各关节动作
饱和率。训练中 `O` 可以安全提前结束已到达理想位置的 episode；专门评估策略时
不要接管。
