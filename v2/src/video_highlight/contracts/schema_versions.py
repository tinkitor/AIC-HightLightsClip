"""阶段产物 Schema 版本。"""

STAGE1_SCHEMA_VERSION = "stage1.v1"
# v2 明确移除了不可靠的 subject_point；空间点提示改由后续 Stage 3.5 产生。
STAGE2_SCHEMA_VERSION = "stage2.v2"
STAGE3_SCHEMA_VERSION = "stage3.v2"
# v3 在逐目标 subject/focus point 之外增加帧级 Qwen 推荐裁剪中心。
STAGE3_5_SCHEMA_VERSION = "stage3.5.v3"
STAGE4_SCHEMA_VERSION = "stage4.v1"
STAGE5_SCHEMA_VERSION = "stage5.v1"
