# PowerTools

<p align="center">
  <img src="app/ui/resources/images/logo.png" width="120" alt="PowerTools Logo" />
</p>

<p align="center">
  <strong>图像处理工具集</strong> · 水印 · 截图 · OCR · 图像编辑
</p>

<p align="center">
  为创意工作一站式解决图片相关需求，支持 Windows、macOS
</p>

---

## 界面预览

应用左侧导航切换功能，右侧为参数与预览区域。

| 主页 | 水印添加 | 水印移除 | OCR |
|------|----------|----------|----------|
| [![主页](docs/screenshots/home.png)](docs/screenshots/home.png) | [![水印添加](docs/screenshots/watermark-add.png)](docs/screenshots/watermark-add.png) | [![水印移除](docs/screenshots/watermark-remove.png)](docs/screenshots/watermark-remove.png) | [![文字识别](docs/screenshots/ocr.png)](docs/screenshots/ocr.png) |

---
**QQ交流群**：1080076113
## 开发进度与更新计划
### 🛠 进行中
- **图片/视频盲水印去除**

### 📌 未来计划
- ✅ 支持 GPU 加速，提升处理速度（8G显存也可以流程运行。个别场景：针对满屏幕水印，视频分辨率大于720的，需要8-16G显存）
- ✅ 优化视频去水印效果
- ✅ 视频支持字幕提取和去除

### ✅ 已完成
| 功能 | 状态 | 说明 |
|------|------|------|
| 水印添加 | ✅ 已完成 | 可见 + 盲水印，图片/视频，批量 |
| 水印移除 | ✅ 已完成 | AI 去水印，图片/视频，单文件与批量 |
| 文字提取(OCR) | ✅ 已完成 | 图片文字提取与识别，单文件与批量 |


## 安装

### 配置 GPU 环境（可选）
- 安装 [cuda 12.x 版本](https://developer.nvidia.com/cuda/toolkit) + [cudnn 9.x](https://developer.nvidia.com/cudnn) 版本
- ***CUDA*** 配置：将 ***CUDA bin*** 目录添加系统环境变量 ***PATH*** 中
- ***cuDNN*** 配置：将 ***cdDNN bin*** 目录添加系统环境变量 ***PATH*** 中

【**备注**】所配置的路径必须为 ***.dll*** 文件所在的绝对路径。若默认的 ***bin*** 目录下未包含 ***.dll*** 文件，程序将无法正常加载。此时请递归向下查找子目录，直至定位到真正的 ***.dll*** 存放目录，并以此目录作为环境变量的值。

### 下载 Release 安装包（推荐）

在 [Releases](https://github.com/SmileSnail5470/PowerTools/releases) 页面提供各平台的安装包。

| 平台 | 说明 |
|------|------|
| **Windows** | 下载压缩包，解压，双击运行 |
| **macOS** | 下载压缩包，解压，双击运行 |

**步骤简述：**

1. 打开项目 [Releases](https://github.com/SmileSnail5470/PowerTools/releases) 页面  
2. 选择最新版本，下载与您系统对应的安装包  
3. 按上表完成安装或解压  
4. 在解压目录中，双击启动 PowerTools

**下载视频解析软件**
- 下载 [`ffmpeg`](https://github.com/BtbN/FFmpeg-Builds/releases)，**下载包体积最大的**。在软件的设置页面设置【ffmpeg路径】为`ffmpeg`可执行文件所在的路径目录。例如设置`../ffmpeg/bin` 路径。

**可选步骤（也可在设置页面开启对应功能，会自动下载模型文件）：**
- 下载 【[CPU 模型-夸克网盘](https://pan.quark.cn/s/59c585f21160)】
- 下载 【[GPU 模型-夸克网盘](https://pan.quark.cn/s/5e273ec49ecf)】

下载完解压后，启动软件，导航到设置界面，设置【AI模型依赖路径】为解压后的路径父目录。例如下面的 `../deps` 目录下的结构是
```bash
.
├── cpu
│   ├── blind_watermark_addition
│   ├── blind_watermark_remove
│   ├── ocr
│   ├── segment
│   ├── video_inpainting
│   └── visible_watermark_removal
├── GPU
│   ├── blind_watermark_addition
│   ├── blind_watermark_remove
│   ├── ocr
│   ├── segment
│   ├── video_inpainting
│   └── visible_watermark_removal
```
![AI 模型依赖路径设置示例](docs/screenshots/models_path.png)

---

## 结果预览

### 水印添加

| 可见水印添加 | 盲水印添加提取 |
|------|----------|
| [![可见水印添加](docs/results/visible-watermark-add.gif)](docs/results/visible-watermark-add.gif) | [![水盲水印添加提取](docs/results/blind-watermark.gif)](docs/screenshots/blind-watermark.gif) |

---

### 水印移除

| 可见水印移除 |
|------|
| [![可见水印移除](docs/results/visible-watermark-remove.gif)](docs/results/visible-watermark-remove.gif) |

---

### 文字提取（OCR）

| 文字提取 |
|------|
| [![OCR 识别](docs/results/ocr.gif)](docs/results/ocr.gif) |

---

## 反馈与许可

- 使用中如有问题或建议，欢迎通过仓库 **Issues** 或应用内「反馈」提交。

感谢使用 **PowerTools**。
