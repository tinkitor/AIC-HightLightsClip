# Video Highlight Pipeline V2

视频高光剪辑推理工程。目前已完成 Stage 1、2、3、3.5、4、5：

- FFprobe 读取视频流、音频流、FPS、时间基、尺寸、旋转和时长；
- PySceneDetect `ContentDetector` 完成硬切镜头检测，可选 `ThresholdDetector` 检测渐变；
- OpenCV 顺序解码并按时间戳完成 2 FPS 粗采样；
- 建立抽样帧、原始帧、时间戳和镜头编号映射；
- 按 24 秒窗口、25% 重叠和镜头边界规划 Stage 2 分析片段；
- FFmpeg 保留 16 kHz 连续音轨，生成基础声学特征、事件和时间线；
- 生成粗高光候选、细化帧级边界，并完成主体跟踪与平滑构图；
- 按原始索引顺序生成和严格校验最终比赛提交 JSONL；
- 每个视频独立持久化、异常隔离，并支持恢复运行。

## 数据位置

原始数据默认位于代码目录之外：

```text
F:/datasets/video-clip/
├─ test_index.json
└─ video/
   ├─ 0.mp4
   └─ ...
```

代码只读取该目录，不复制或修改原始视频。默认路径定义在 `configs/paths.yaml`。

## 环境

```powershell
conda activate video-clip
python -m pip install "scenedetect==0.7.1" "PyYAML==6.0.3"
```

也可以使用 `environment.yml` 更新环境。Stage 1 依赖 FFmpeg/FFprobe 可执行文件在 PATH 中可用。

Grounding DINO + SAM2 后端还需要 GPU 版 PyTorch、官方 SAM2 包和可选依赖：

```powershell
python -m pip install -e ".[grounded-sam2]"
```

SAM2 权重放在 `model_store/sam2/`；将 Hugging Face
`IDEA-Research/grounding-dino-tiny` 的完整模型目录放在
`model_store/grounding-dino-tiny/`。默认配置禁止运行时联网下载。

## 运行 Stage 1

先处理一个视频：

```powershell
python scripts/run_stage1.py --run-id smoke --video-id 0 --strict
```

处理索引中的全部视频：

```powershell
python scripts/run_stage1.py --run-id baseline_stage1
```

断点续跑：

```powershell
python scripts/run_stage1.py --run-id baseline_stage1 --resume
```

输出位于 `runs/<run-id>/stage1/`。每个视频目录包含：

```text
metadata.json
scenes.jsonl
segments.jsonl
sample_map.jsonl
coarse_frames/*.jpg
audio.wav
audio_features.npz
audio_events.jsonl
audio_timeline.jsonl
asr.jsonl
audio_status.json
_SUCCESS.json
```

当前 ASR 后端默认禁用，`asr.jsonl` 为空；这避免在没有明确 ASR 模型时伪造语音结果。连续音频和声学时间线均已保留，后续可接入本地 ASR。

## Stage 2：Qwen3.5-4B 粗高光候选

Stage 2 通过已经部署好的 vLLM OpenAI 兼容接口调用
`QuantTrio/Qwen3.5-4B-AWQ`，项目中不包含模型加载或 vLLM 部署代码。
服务地址、模型名、API Key、超时、生成参数和视频传输模式均位于
`configs/stage2/qwen3_5_4b.yaml`。

当前默认服务配置为：

```text
base_url: http://172.25.254.120:8000/v1
model: QuantTrio/Qwen3.5-4B-AWQ
api_key: EMPTY
```

运行时必须明确指定某次 Stage 1 产物：

```powershell
python scripts/run_stage2.py `
  --stage1-dir runs/baseline_stage1/stage1 `
  --run-id baseline_stage2 `
  --strict
```

可通过命令行临时覆盖部署位置和模型名称：

```powershell
python scripts/run_stage2.py `
  --stage1-dir runs/baseline_stage1/stage1 `
  --base-url http://127.0.0.1:8000/v1 `
  --model QuantTrio/Qwen3.5-4B-AWQ
```

### 视频传输模式

`data_url` 模式是默认模式。它读取 Stage 1 已经生成的 2 FPS JPEG，不重复
解码原视频；客户端只将这些帧临时编码为低帧率 MP4，然后以 Base64
`data:video/mp4` URL 发送：

```yaml
video_input:
  mode: data_url
```

`url` 模式不在客户端读取或编码媒体，只向 vLLM 发送 URL：

```yaml
video_input:
  mode: url
  url_template: "http://media-server/video/{video_id}?start={start_ms}&end={end_ms}"
```

URL 模板支持 `video_id`、`segment_id`、`start_sec`、`end_sec`、`start_ms`、
`end_ms` 和 `source_name`。推荐 URL 本身返回对应片段。若 URL 指向完整视频，
只有在当前 vLLM 媒体解析器明确支持时，才应开启 `include_time_range_fields`。

Stage 2 每个视频输出：

```text
requests.jsonl
raw_responses.jsonl
analyses_segment_results.jsonl
candidates.jsonl
subject_hints.jsonl
_SUCCESS.json
```

Stage 2 只输出主体文本 `subject`，不输出 `subject_point`。低帧率粗采样上的
点定位误差较大，空间点提示将由后续 Stage 3.5 在帧级高光区间确定后单独生成。

响应中的相对时间由确定性代码映射为原视频绝对时间，重叠窗口候选会扩展、
合并并生成稳定的 `candidate_id`。Base64 本体不会写入请求日志。

## Stage 3：高光候选帧级边界定位

Stage 3 读取同批次的 Stage 1 元数据和 Stage 2 候选。默认只对候选区间进行
10 FPS 精解码，提取运动、直方图变化、清晰度、亮度、音频、镜头边界和
Stage 2 粗分数，通过无训练规则后端保守细化边界，并映射成原始视频的左闭
右开帧区间 `[start_frame, end_frame)`。

正常运行：

```powershell
python scripts/run_stage3.py `
  --stage1-dir runs/full/stage1 `
  --stage2-dir runs/full/stage2 `
  --run-id full_stage3 `
  --strict
```

完全跳过 Stage 3 处理，只把 Stage 2 秒区间映射并封装为合法 Stage 3 输出：

```powershell
python scripts/run_stage3.py `
  --stage1-dir runs/full/stage1 `
  --stage2-dir runs/full/stage2 `
  --run-id full_stage3_passthrough `
  --skip-processing `
  --strict
```

`--skip-processing`（别名 `--passthrough`）不会打开或解码源视频，不提取特征，
也不会执行边界细化、无高光门控或区间合并。Stage 2 的 `start_sec/end_sec`
保持原值，只进行 Stage 4 所需的确定性帧号映射。

每个视频输出：

```text
refined_intervals.jsonl
diagnostics.jsonl
_SUCCESS.json
```

Stage 3 只负责时间边界和主体语义透传，`refined_intervals.jsonl` 不包含
`subject_point`；即使读取旧版 Stage 2 产物中的同名字段也会主动丢弃。

## Stage 3.5：逐采样帧多主体观察

Stage 3.5 读取 Stage 3 最终高光区间，并只从 Stage 1 读取源视频路径、FPS 和
总帧数。它不会复用 Stage 1 的粗采样帧，而是在每个 `[start_frame,end_frame)`
内默认按 2 FPS 即时解码原视频，将全部采样 JPEG 和明确的原始帧号/时间戳顺序
一次性发送给 Qwen。提示中同时传入 Stage 3 的 `subject`，并把 `category/reason` 作为
弱上下文。模型为每个 `sample_index` 返回 `group_mode`、`composition_mode`、
`primary_target_ids`、适合开放词汇检测的英文 `grounding_phrases`，以及零到多个目标；
每个目标包含 `target_id`、描述、英文检测短语、用于检测关联的 `subject_point`、用于
构图的 `focus_point`、`role`、`importance`、定位置信度和可见性。`subject` 决定谁是
叙事主主体，不能简单按检测框或 Mask 面积选择。`stage3.5.v3` 还要求每帧输出唯一的
`recommended_crop_center` 和 `recommended_crop_confidence`：前者是结合目标画幅、主次
主体、动作/视线方向和必要留白得到的裁剪框中心，并不等同于任一主体点或多点平均值。

```powershell
python scripts/run_stage3_5.py `
  --stage1-dir runs/full/stage1 `
  --stage3-dir runs/full/stage3 `
  --output-dir runs/full/stage3_5 `
  --strict
```

跳过全部解码和模型调用、但生成相同采样时间轴和空目标契约：

```powershell
python scripts/run_stage3_5.py `
  --stage1-dir runs/full/stage1 `
  --stage3-dir runs/full/stage3 `
  --output-dir runs/full/stage3_5_passthrough `
  --skip-processing `
  --strict
```

每个视频输出 `enriched_intervals.jsonl`、`subject_observations.jsonl`、
`requests.jsonl`、`raw_responses.jsonl`、`diagnostics.jsonl` 和 `_SUCCESS.json`。
请求日志只保存帧号时间线和 JPEG 总字节数，不保存 Base64 图像本体。

### 可视化 Stage 3.5 Qwen 中心点

独立可视化模块只读取 Stage 1 和 Stage 3.5 已完成产物，不会修改上游结果。它会在
每个 Qwen 采样对应的原始视频帧上绘制所有目标点、目标 ID、Grounding 短语、置信度、
多点包围范围和几何组合中心：

```powershell
python scripts/visualize_stage3_5.py `
  --stage1-dir runs/full/stage1 `
  --stage3-5-dir runs/full/stage3_5_grounding `
  --output-dir runs/smoke/visualizations/stage3_5 `
  --video-id 35 `
  --write-video `
  --strict
```

默认输出逐采样帧 JPG；`--write-video` 会为每个高光区间额外生成
`qwen_subject_points.mp4`。使用 `--no-images --write-video` 可只保留视频。每个视频的
`manifest.json` 记录源视频、观察数、有效点数和区间映射，批次根目录还会生成
`summary.json`。解码器支持与 Stage 3.5 相同的 `auto`、`opencv` 和 `ffmpeg` 模式。

若视频携带不完整或异常的色彩元数据（例如视频 97 的 `trc=log316`，但
`colorspace/primaries=unknown`），新版 FFmpeg/swscale 可能拒绝直接转换 BGR。
`sampling.decoder: auto` 会在 OpenCV 解码失败后自动使用 FFmpeg 内存管道，并在
颜色转换前将该异常描述覆盖为 BT.709；整个回退过程同样不会持久化采样帧。排查时
也可以临时设置 `sampling.decoder: ffmpeg` 强制验证该路径。
命令行对应参数为 `--decoder ffmpeg`。

未来获得训练好的 TorchScript TCN 权重后，可用 `--backend tcn --checkpoint ...`
切换到模型推理；默认规则后端不依赖 PyTorch，也不包含训练代码。

## Stage 4：主体构图与轨迹优化

Stage 4 以 Stage 3.5 的 `enriched_intervals.jsonl` 和 `subject_observations.jsonl`
作为高光区间、主体语义、Grounding 短语和逐采样帧多主体点的唯一上游契约；仍可
读取旧版 `subject_points.jsonl` 并自动适配为单目标观察。它只从 Stage 1 读取源视频
路径、原始帧率/尺寸、`targetRatioWH` 和镜头边界；不会直接读取 Stage 2 或 Stage 3。

默认配置使用 Grounding DINO + SAM2：每个镜头起点和 Stage 3.5 锚点先执行开放词汇
检测，多个 Qwen 点分别认领检测框；检测框与上一窗口对象框进行空间关联并复用稳定
`obj_id`，每个对象分别注册给 SAM2。传播阶段只合并当前窗口活跃对象的 Mask，生成
覆盖全部主体的逐帧并集框。Grounding 失败时使用每个 Qwen 点的合成框继续启动 SAM2，
SAM2 窗口失败时才回退到多点包围框插值。

Qwen 的单帧目标数量和提示词可能短暂波动，因此 Stage 4 不再把当前观察直接视为完整
对象集合：Grounding 会合并最近若干锚点的历史短语继续检测旧主体；与历史框匹配的
检测即使不靠近本帧 Qwen 点也可保留。Grounding 仍短暂漏检时，对象在配置的
`object_keepalive_anchors` 内继续复用原 `obj_id` 和最近 SAM2 框，不会立即退出 Mask
并集；历史框按 ID 增量更新，只有超过宽限期才显式过期。该记忆在镜头切换时清空。

在每个锚点，Stage 4 还会把 Qwen 的局部 `target_id` 和主次角色映射到已经稳定的
SAM2 `obj_id`。主主体变化采用迟滞策略：候选主主体需连续
`primary_switch_confirm_anchors` 个锚点成立后才切换；旧主主体已过期时立即切换。
因此单帧漏点、目标数变化或临时角色误判不会直接造成裁剪中心跳变。Qwen 的
`focus_point` 会转换成对象框内相对位置，并随 SAM2 当前帧 Mask 框传播。

SAM2 的逐帧联合 Mask 和各对象 Mask 会压缩为空间面积密度图并交给构图规划，不再只
使用 Mask 最大外接框的几何中心。每个合法尺度都会补充 Mask 覆盖较高的目标比例候选，
并强制加入主主体 `focus_point`、主主体 Mask 质心及多主主体组合中心；该规则对
`fixed_maximum: true/false` 都生效。置信度达到
`crop_candidates.qwen_recommended_min_confidence` 的 Qwen 推荐构图中心也会加入每个尺度
的候选，并通过 `composition.qwen_recommended_center_weight` 提供弱评分偏好。该偏好只用于
区分主体/Mask 质量接近的候选，不会取代覆盖、切边与主主体安全区约束。单帧代价采用不对称主次评分：优先保证主主体完整、
不被切边且位于安全区，辅助主体覆盖是较弱软约束，同时保留联合 Mask IoU/覆盖和尺度；
因此 `crop_candidates.fixed_maximum` 为 `true` 时会在固定尺度下搜索位置，为 `false`
时则同时搜索位置和尺度。无 Mask 的 center、OpenCV 和降级帧仍使用旧的外接框评分。
平滑若让 Mask 覆盖率显著恶化，会退回已经过动态规划时序约束的候选框。

每帧围绕主体生成多尺度、多偏移、运动方向留白的目标比例候选框，使用动态规划选择
低代价轨迹，再对中心和尺度做限速平滑。镜头边界两侧分别优化，不跨硬切镜头平滑。
所有框最后统一取整、再次限界，并输出比赛需要的 `[x, y, w]`。

锚点并集 Mask 与 Qwen 多点或 Grounding 多框不一致时，会把各 Qwen 正点追加到最近
对象进行纠偏；窗口缺帧或失败时只回退该窗口，不丢弃整个高光区间的 SAM2 结果。
SAM2 在传播中出现空 Mask 后不会采用第一帧重新出现的结果：后续 Mask 必须连续达到
`tracking.recovery_confirm_frames`（默认 3 帧）才恢复使用；确认期仍使用 Qwen/center
fallback，期间再次缺失会清零连续计数。`diagnostics.jsonl` 的
`recovery_wait_frame_count` 会记录因此进入恢复等待的帧数。
`crop_candidates.fixed_maximum: true` 时，输出始终采用目标比例下的最大合法裁剪框，
只让跟踪结果决定框中心。纯 `center` 后端及区间级 `center` 降级不再生成主体框、偏移、
Mask 或多尺度候选：每帧只保留以 Qwen 推荐构图中心合法化后的唯一最大框；即使
`fixed_maximum: false` 也保持该语义。采样点之间按镜头内线性插值，镜头内没有有效推荐
时使用画面中心；这条唯一候选轨迹跳过单向 EMA，避免最终中心偏离 Qwen 推荐。SAM2 和
光流路径仍执行多候选 DP 与平滑。

将 `visualization.enabled` 设置为 `true` 后，SAM2 仍逐帧传播，但只按
`visualization.sample_fps` 保存两遍生成的决策诊断图。第一遍在右栏列出 Qwen 检测关联点
`Q` 的坐标，但不再把 `subject_point` 画到主画面；主画面绘制主主体构图焦点 `F`、
Grounding DINO 原始框 `D`、绑定稳定 object_id 后的采用框 `G{id}`、逐对象彩色 Mask 与
Mask 质心 `M{id}`。`D/G` 仅在框左上角显示短标签，不绘制检测框中心点。第二遍在构图
规划完成后补画局部候选中心 `C1..Cn`、动态规划选择中心 `DP`、最终平滑裁剪框和绿色输出
中心，并以紫色 `R` 标记 Qwen 的 `recommended_crop_center`；若 `R` 与 DP/输出中心重合，
改用紫色外环避免遮盖内层决策点。右侧信息栏同步显示 `R` 的像素坐标和
`recommended_crop_confidence`。Grounding 短语、分数、对象身份和坐标集中放在右侧信息栏，
候选排名、最终 local cost、Mask IoU、主主体平均/最低覆盖率、辅助主体覆盖率与切边风险
集中放在底部决策栏。默认只在主画面绘制前三个候选中心，并隐藏冗余的联合框。
相同数据还会写入 `tracks.jsonl` 的 `mask_centers` 和 `composition_decision`，便于脱离图片
做统计分析。锚点帧和窗口回退帧可配置为强制保存；结果位于单视频目录下的
`visualizations/<interval_id>/scene_*/`。`save_images` 和 `write_video` 分别控制 JPG
与每镜头 `propagation.mp4`，两者均关闭时不会创建可视化目录。

字体和标记尺寸全部可在 `configs/stage4/sam2_crop.yaml` 的 `visualization` 中调整：
`tag/sidebar/decision_font_*` 控制三类文字，`candidate/dp/final/mask/qwen/focus/fallback_marker_*`
及 `qwen_recommended_marker_*`
控制各类中心点，`raw/selected_detection_box_thickness` 和
`final_crop_box_thickness` 控制主要框线宽。旧配置不含这些字段时使用当前默认值。
候选点绘制为黑色实心圆，圆内只显示排名数字，例如 `1/2/3` 分别对应底部决策栏
中的 `C1/C2/C3`。其他点保留原有语义填充色并统一使用白色描边，`F1`、`M1`、`DP`
等短标识写在圆内；唯一的最终输出点圆内和圆外均不写文字，只通过绿色圆点与绿色裁剪框
表示。若 `DP` 与最终输出中心重合，绿色点外会显示橙色同心环及一个 `DP` 短标签；右栏的
`COMPOSITION CENTERS` 固定列出二者像素坐标和是否重合，底部决策栏也同步列出 DP 坐标。

正常运行：

```powershell
python scripts/run_stage4.py `
  --stage1-dir runs/full/stage1 `
  --stage3-5-dir runs/full/stage3_5 `
  --output-dir runs/full/stage4 `
  --strict
```

可选后端：

- `--backend opencv`：不需要新权重的光流基线。
- `--backend center`：不解码视频，按同镜头 Qwen 点线性插值；仅在镜头无有效点时
  使用固定中心框，也作为其他后端的区间失败降级。
- `--backend sam2 --sam2-checkpoint <权重路径> --sam2-config <模型配置>`：使用官方
  SAM2 视频预测器传播主体 Mask。默认同时启用 Grounding DINO；可用
  `--grounding-model <本地模型目录>` 覆盖模型，或用 `--no-grounding` 仅运行多 Qwen
  点 + SAM2。只有选择该后端时才加载 SAM2、Grounding DINO 和 PyTorch。

每个视频持久化输出：

```text
crops.jsonl         # Stage 5 直接消费的逐帧 [x,y,w]
tracks.jsonl        # 主体 xyxy、置信度和跟踪来源，便于调试
diagnostics.jsonl   # 区间状态、镜头子段数和降级信息
grounding_anchors.jsonl # 每个锚点的原始候选、评分、obj_id、选择与回退信息
_SUCCESS.json
```

## Stage 5：最终提交 JSONL

Stage 5 只把 Stage 4 的 `crops.jsonl` 作为预测来源。它读取原始 `test_index.json`
以严格保持视频行顺序，并从 Stage 1 元数据复核目标比例、总帧数和画面尺寸；不读取
Stage 2 或 Stage 3。导出时删除 Stage 4 的调试字段，仅保留比赛规定字段。

```powershell
python scripts/run_stage5.py `
  --stage1-dir runs/full/stage1 `
  --stage4-dir runs/full/stage4 `
  --output-dir runs/full/stage5 `
  --overwrite
```

`--input-index` 默认取 `configs/paths.yaml` 中的 `input_index`。最终目录包含：

测试单个视频时可指定 `--video-id`。例如下面只读取并生成视频 20 的一行提交
记录，不要求索引中的其他视频已经产生 Stage 4 结果，也不要求 Stage 1/4 批次级
`_SUCCESS.json` 已经生成；指定视频自身仍须具有成功标记：

```powershell
python scripts/run_stage5.py `
  --stage1-dir runs/full/stage1 `
  --stage4-dir runs/full/stage4 `
  --output-dir runs/test_video_20/stage5 `
  --video-id 20 `
  --overwrite
```

```text
submission.jsonl       # 可直接提交的最终文件
validation_report.json # 视频数、预测帧数、空结果数、文件 SHA-256
resolved_config.json
run_manifest.json
_SUCCESS.json
```

也可脱离生成流程单独复核一个提交文件：

```powershell
python scripts/validate_submission.py `
  --submission runs/full/stage5/submission.jsonl `
  --input-index F:/datasets/video-clip/test_index.json `
  --stage1-dir runs/full/stage1
```

## 测试

```powershell
python -m unittest discover -s tests/stage1 -v
python -m unittest discover -s tests/stage2 -v
python -m unittest discover -s tests/stage3 -v
python -m unittest discover -s tests/stage3_5 -v
python -m unittest discover -s tests/stage4 -v
python -m unittest discover -s tests/stage5 -v
```
