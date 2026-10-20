# Qwen-PromptEnhancer

在 ComfyUI 内启动本地 llama.cpp `llama-server` 运行所选 GGUF 模型（默认 Qwen3.8-27B），让本地模型通过 function calling 以**多步 agent** 方式执行任意 Skill，最终输出一段纯文本结果。生图提示词、分镜、文案、方案等文本类 Skill 都能运行。

![Qwen Prompt Enhancer 节点](images/Qwen%20Prompt%20Enhancer.png)

主要特性：

- **Skill 自由选择**：扫描 `skills/`、`custom_skills/` 与 `skill_root` 中的 Skill，可从下拉框直接指定，也可交给模型选 `auto` 自行挑选，还可选 `none` 完全不用 Skill 直接回答。
- **多模型可选**：模型下拉框直接来自 ComfyUI 的 `LLM` 模型目录（`models/LLM` 及 `extra_model_paths.yaml` 注册的目录），可换用任意 GGUF 模型；各种 template 参数按所选模型自己的 chat template 能力下发。
- **多步工具调用**：模型可以列出并读取所选 Skill 目录内的文档（`SKILL.md` 与 `references/`），按 Skill 流程逐步推理，最后调用 `finish` 返回完整结果。
- **全程离线**：节点自带独立的 `llama-server` 进程，推理不访问网络；工具只能读取所选 Skill 目录内的文件。
- **只输出文本**：结果仅通过 `result` 字符串返回，节点不写入任何文件，可直接接字符串/文本节点。
- **Skill 列表免重启**：Skill 目录在每次构建节点定义时重新扫描，新增/改名/删除 Skill 后点「Refresh Node Definitions」或刷新页面即可生效。
- **参考图与参考视频**：可挂载最多 9 张参考图、3 路参考视频帧（需要 mmproj），用途由所选 Skill 与提示词自行决定。
- **资源占用可控**：`max_agent_steps` 限制工具轮次（默认 6），默认在运行结束后终止内置服务并释放显存与内存。
- **界面汉化**：节点名、分类、参数标签与提示均提供中文（`locales/zh/`），跟随 ComfyUI 的语言设置。

本项目是上游项目 [chflame163/ComfyUI_Qwen_H3_Prompt](https://github.com/chflame163/ComfyUI_Qwen_H3_Prompt) 的衍生版本，详见文末[上游与致谢](#上游与致谢)。上游与本节点互不影响，可以同时安装。

## Skill 来源

节点加载时扫描 `skills/` 和 `custom_skills/`，并按 `skill_root` 扫描额外目录，把其中的 Skill 注册进候选列表。

- 每个 Skill 是包含 `SKILL.md` 的子目录。
- Skill id 就是技能目录名（`SKILL.md` 里的 `name` 字段不参与命名，避免两者不一致）；允许大写字母与非 ASCII 字符（例如 `H3-Singularity`、`ai-image-realism-图片真实感`），只有 `auto` 是保留字。
- 技能目录是**每次构建节点定义时重新扫描**的，所以新增/重命名/删除 Skill 后无需重启 ComfyUI：在界面里点「Refresh Node Definitions」或刷新页面即可看到新列表。
- 同名 Skill 先注册者优先：内置 `skills/` > `custom_skills/` > `skill_root`。

## 能力边界

本节点的模型只产出文本（提示词、方案等），不会调用 ComfyUI 队列真正生成图片或视频，也不会执行 Skill 声明的脚本、工具、网络调用、画布流程或审批卡片。第三方 Skill 的内容仅作为本地模型的提示词指导。

**换用其他模型时需要注意**：Skill 执行依赖模型返回工具调用，因此 `skill` 选了具体 Skill 或 `auto` 时，所选模型必须支持 function calling，否则运行会以 “empty turn without calling a tool” 失败；`skill` 选 `none` 时是直接对话，不依赖工具调用。节点会在启动日志里对不支持工具调用的模型给出警告。参考图片/参考视频需要多模态 mmproj。

本节点默认的 Qwen3.8-27B GGUF 模型不具备分析声音的能力，不能准确提取已有音乐的旋律、BPM、歌词、音色或节拍点。有关声音的提示词只是“创作声音的描述”，不是分析或复刻参考音频。

## 安装节点

使用ComfyUI Manager安装， 或使用以下命令手动安装:

```commandline
cd ComfyUI/custom_nodes/
git clone https://github.com/brightests/Qwen-PromptEnhancer.git
```

## 模型目录

GGUF 模型放在 ComfyUI 的 `LLM` 模型目录里，默认是 ```ComfyUI/models/LLM/```。节点通过 ComfyUI 的模型目录系统枚举文件，所以：

* 可以用 `extra_model_paths.yaml` 追加自己的 `LLM` 目录，节点会自动看到；
* 子目录同样会被扫描，例如 `models/LLM/Qwen3.8/` 中的文件以 `Qwen3.8/xxx.gguf` 出现在下拉框里；
* 文件名含 `mmproj` 的进视觉投影模型（`vision_model`）下拉，其余进语言模型（`llm_model`）下拉；
* 新增或删除 GGUF 文件后刷新页面或「Refresh Node Definitions」即可，无需重启。

默认使用 Qwen3.8，权重与投影器如下：

* 视觉权重：从 https://huggingface.co/unsloth/Qwen3.8-27B-GGUF 下载 ```mmproj-F16.gguf```，放进 ```models/LLM/Qwen3.8/```
* GGUF权重：   
  对于20G以上显存的设备，推荐从 https://huggingface.co/unsloth/Qwen3.8-27B-GGUF 下载 ```Qwen3.8-27B-Q4_K_M.gguf```。    
  对于16G显存的设备，推荐 https://huggingface.co/soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf 。

也可以换成其他与 `llama-server` 兼容的 GGUF 模型（不限于 Qwen3.8），选择方式见下面的节点选项说明。

## 安装llama-server runtime

项目只提供一个跨平台安装入口 `install_runtime.py`。它仅使用 Python 标准库，自动识别 Windows、Linux 或 macOS、x64/ARM64 架构及显卡，并为当前机器部署固定的 llama.cpp `b11115` runtime，不需要安装额外的 pip 包。NVIDIA 设备优先使用 `nvidia-smi` 报告的驱动 CUDA 兼容版本，`nvcc` 版本仅作为备用判断。

Windows ComfyUI 官方便携包在节点目录中执行：

```
cd ComfyUI\custom_nodes\Qwen-PromptEnhancer
..\..\..\python_embeded\python.exe install_runtime.py
```

Linux 与 macOS 使用Python执行：

```
python install_runtime.py
```

首次启动前必须先执行安装器。安装器会依次完成硬件检测、后端选择、下载、SHA256 校验、安全解压、`llama-server --version` 与 `--list-devices` 验证，最后生成本机专用的 `runtime_config.json`。下载缓存、源码缓存和安装结果都保存在节点的 `runtime/` 目录中。

如果 Windows 报告 `WinError 5` 拒绝访问，先关闭正在运行的 ComfyUI，再使用当前 Windows 用户在节点目录重新执行 `python install_runtime.py --force`。节点启动 llama-server 时会把工作目录设在自定义节点目录，不依赖 runtime 文件夹作为当前目录；这样可以避开旧安装中 runtime 子目录继承的限制性 ACL。

安装器命令：

```
# 只显示检测结果和计划，不下载或安装
python install_runtime.py --dry-run

# 查看当前系统和架构支持的后端
python install_runtime.py --list-backends

# 手动指定后端
python install_runtime.py --backend vulkan

# 强制重新安装
python install_runtime.py --force

# 只使用 runtime/.downloads/b11115 中已经校验的文件
python install_runtime.py --offline
```

### 自动后端选择

| 系统和设备                         | 自动顺序                                                                 |
| ----------------------------- | -------------------------------------------------------------------- |
| Windows x64 NVIDIA            | 兼容 CUDA 13.4 时：CUDA 13.4 → CUDA 12.4 → Vulkan → CPU；否则从 CUDA 12.4 开始 |
| Windows x64 AMD               | ROCm 10.0 → Vulkan → CPU                                             |
| Windows x64 Intel             | SYCL → Vulkan → CPU                                                  |
| Windows x64 未识别 GPU           | Vulkan → CPU                                                         |
| Windows ARM64 NVIDIA          | 兼容 CUDA 13.4 时：CUDA 13.4 → CPU；否则 CPU                                |
| Windows ARM64 Qualcomm Adreno | OpenCL → CPU                                                         |
| Linux x64 NVIDIA              | 兼容 CUDA 13.4 时：CUDA 13.4 → CUDA 12.8 → Vulkan → CPU；否则从兼容版本开始        |
| Linux x64 AMD                 | ROCm 10.0 → Vulkan → CPU                                             |
| Linux x64 Intel               | SYCL FP16 → Vulkan → CPU                                             |
| Linux x64 未识别 GPU             | Vulkan → CPU                                                         |
| Linux ARM64 NVIDIA            | 兼容 CUDA 13.4 时：CUDA 13.4 → Vulkan → CPU；否则 Vulkan → CPU              |
| Linux ARM64 其他设备              | Vulkan → CPU                                                         |
| macOS Apple Silicon 或 Intel   | Metal → CPU                                                          |

Windows 可用后端取决于架构，包括 `cuda12`、`cuda13`、`rocm`、`sycl`、`vulkan`、`opencl` 和 `cpu`。

Linux 官方预编译后端包括 `cuda12`、`cuda13`、`rocm`、`vulkan`、`sycl`、`sycl-fp32`、`openvino` 和 `cpu`，具体选项取决于架构。官方文件是 Ubuntu 构建，其他发行版需要兼容的 glibc/libstdc++。CUDA、ROCm、Vulkan、Intel SYCL/OpenVINO 还需要系统中已经存在相应的显卡驱动和运行库。

macOS 使用对应架构的官方包。Metal 不需要额外 GPU 计算框架，但 Qwen3.8-27B Q4 和视觉投影器仍需要足够的统一内存。

### Linux CUDA 部署

官方 `b11115` 提供 Ubuntu x64 CUDA 12.8/13.4 和 Ubuntu ARM64 CUDA 13.4 压缩包，自动模式会根据本机驱动兼容版本选择。也可显式要求自动选择 CUDA 预编译包：

```
python install_runtime.py --backend cuda
```

当没有兼容的官方包或需要本机构建时，可显式要求安装器从源码构建：

```
python install_runtime.py --backend cuda --build-from-source
```

源码构建不会调用 `sudo` 或自动修改系统。运行前必须已经具备：

- 可正常运行 `nvidia-smi` 的 NVIDIA 驱动；
- 包含 `nvcc` 的 NVIDIA CUDA Toolkit，`nvcc` 位于 `PATH` 中或已经设置 `CUDA_HOME`；
- Git、CMake、C 编译器和 C++ 编译器，例如 Ubuntu/Debian 的 `git cmake build-essential`。



## 节点

启动ComfyUI，在 `😺Bright/Qwen_PromptEnhancer` 分类中添加 `Qwen Prompt Enhancer · 本地提示词`节点。

`example_workflows/` 内置三个示例工作流（也会出现在 ComfyUI 的「模板浏览」中）：MiniMaxH3 Skill、Moss-Image Skill、Qwen-Image2.1 Skill 的提示词生成示例。

节点界面文案的汉化放在 `locales/zh/`（`nodeDefs.json` 翻译节点名、分类名、输入输出标签与提示，`main.json` 翻译分类）。ComfyUI 会在启动时缓存各节点的 locale 文件，因此**新增或修改 locale 后需要重启 ComfyUI** 才会生效。

节点选项说明：

- `prompt`：用户请求，描述要完成的任务。
- `skill`：下拉框，列出内置 `skills/` 与 `custom_skills/` 中发现的 Skill，另有 `auto` 选项。设为 `auto` 时节点先在全部已发现的 Skill 中自动选择一个。选 `none` 时不加载任何 Skill，跳过工具调用循环直接回答，此时模型不需要支持 function calling 也能使用，`selected_skill` 输出为 `none`。
- `skill_root`：高级选项，默认空。额外 Skill 根目录，其中的 Skill 会被扫描并加入 `auto` 自动选择候选；要让某个 Skill 出现在下拉框中，请把它放进 `custom_skills/`。
- `llm_model`：语言模型，下拉框列出所有已注册 `LLM` 目录中的 GGUF（不含 `mmproj`）。默认 `Qwen3.8/Qwen3.8-27B-Q4_K_M.gguf`，可换成任意支持工具调用的 GGUF 模型。
- `vision_model`：视觉投影模型（mmproj），可选。默认 `Qwen3.8/mmproj-F16.gguf`；选 `none` 时不传 `--mmproj`，按纯文本模型运行，此时不能使用参考图片/参考视频。
- `think_mode`：布尔开关，默认关闭；在 instruct 与 thinking 两套 Qwen 官方推荐采样参数之间切换。所选模型的 chat template 不含 `enable_thinking` 时自动按 instruct 参数运行，并在日志中给出提示。
- `reasoning_effort`：仅在 `think_mode` 开启、且模型模板支持 `reasoning_effort` 时随请求下发，默认 `medium`。
- `seed`：种子。
- `max_tokens`：每个 agent 步骤的最大生成 token 数，范围 256-16384；开启思考模式时思考内容也计入。每步输出会随对话历史重新占用上下文，调大时注意 `context_size` 是否足够。
- `context_size`：高级选项。llama.cpp 上下文窗口（`--ctx-size`），默认 32768，提示词、多步对话历史与输出共享这个窗口。调大会增加 KV cache 显存占用；上下文不足时 llama.cpp 会从开头截断最早的内容而不报错。
- `reference_images`：动态添加 0-9 张独立 IMAGE，作为参考图片随请求一起交给模型。
- `reference_videos`：动态添加 0-3 路 IMAGE 批次，内部固定按 24 fps 计算时长和采样时间，每路不超过 15 秒。
- `video_sample_frames_per_sec`：每路参考视频每秒抽取的帧数，默认值为 2。
- `duration`：提示词时长，默认 `0` 表示不注入（图片类输出无需时长）；大于 0 时在用户请求中追加目标时长（秒，范围 0-30），供视频/声音类 Skill 安排成片节奏，作用与上游 H3 节点的 `duration` 一致。
- `max_agent_steps`：agent 工具调用轮次上限，默认 6，范围 1-12。
- `force unload model`：默认开启。开启时每次完成后终止内置 llama-server服务并强制清理内存和显存；关闭时保留模型供下一次运行复用。发生错误时无论选项状态都会强制释放。

输出：

- `result`：Skill 产出的最终文本。
- `selected_skill`：本次实际使用的 Skill id。

### Agent 工具

本地模型在运行中可以使用以下离线工具：

- `list_skill_files` / `read_skill_reference`：列出并读取所选 Skill 目录内的文档（`SKILL.md` 及其 `references/`）。
- `finish`：返回最终结果并结束运行。

结果只通过节点的 `result` 字符串输出返回，节点不会写入任何文件；把 `result` 接到字符串/文本节点即可使用。读取仅限所选 Skill 目录内。达到 `max_agent_steps` 仍未结束时节点会报错退出。

### 自定义 Skill

在 `custom_skills/` 下为 Skill 建立独立子目录，并至少放入一个 `SKILL.md`；也可以把已有的 Skill 目录通过 `skill_root` 挂载进来。

- Skill id 允许大写字母、数字、点、下划线和连字符，支持非 ASCII 字符；不能使用 `auto`。
- 节点只把 Skill 作为本地模型的提示词指导，不会执行第三方 Skill 声明的脚本、工具、网络调用、画布流程或审批卡片。

### 参考图片与参考视频

参考图片与参考视频都需要多模态投影器：请把 `vision_model` 指向 `mmproj` 文件。`vision_model` 选 `none` 时按纯文本模型运行，此时连接参考图片或参考视频会在运行时直接报错。

`reference_images` 最多连接 9 张独立 IMAGE，每个插槽只接受一张图片；`reference_videos` 最多连接 3 路 IMAGE batch。图片的角色和用途由所选 Skill 与用户提示词决定，节点不做角色判定。

每一路参考视频是按时间顺序排列的 IMAGE batch，至少包含 1 帧，按 24 fps 计算后每路不得超过 15 秒，全部参考视频总时长也不能超过 15 秒。`video_sample_frames_per_sec` 表示从每一路视频的每一个一秒时间窗中均匀抽取多少帧，它不是整段视频的总抽帧数。默认值 `2` 适合大多数动作和运镜；静态内容可用 `1`，快速动作可提高到 `3-8`。数值越高，每秒保留的细节越多，但联系表中单帧面积会变小，同时增加图片编码、内存和推理开销。

当前接口只接收视频画面帧，不会读取视频文件中的音轨。

## 显存和生命周期

若关闭 `force unload model`，内置服务会继续占用显存与内存（默认 Qwen3.8-27B Q4 约 20 GB，换成其他模型时随所选模型变化），模型配置不变时下一次可直接复用；再次开启并运行节点、模型配置发生变化或退出 ComfyUI 时会终止驻留服务。

## 上游与致谢

- 本节点源自上游项目 [`chflame163/ComfyUI_Qwen_H3_Prompt`](https://github.com/chflame163/ComfyUI_Qwen_H3_Prompt)（原作者 [@chflame163](https://github.com/chflame163)）。本项目在它的基础上把原来固定的 MiniMax H3 提示词节点改造成通用的本地 Skill agent，并沿用其跨平台 `llama-server` runtime 安装方案、模型目录约定与部分文档。感谢原作者的开源与整理工作。
- 上游项目与本项目可以同时安装、互不影响；本项目只做新增与改写，不会改动上游仓库的任何文件。
- MiniMax H3 Skill 来自 [`MiniMax-AI/MiniMax-H3`](https://github.com/MiniMax-AI/MiniMax-H3) commit `d21241f0a4b3acbb34c97dae47fa417b7065e438`，官方文件保存在 `skills/`。
- 跨平台runtime安装器固定使用 llama.cpp `b11115`，对应完整 commit `d5f66492e661b63e6c74822c2b72f5146053994e`，并对下载文件执行 SHA256 校验。
- Linux CUDA 源码构建使用同一固定 tag 和 commit。llama.cpp 的 MIT 许可证保存在 `third_party/llama.cpp-LICENSE`。
- 本项目仅为学习研究使用，如果作为商业用途，请查阅相关原项目授权协议。
