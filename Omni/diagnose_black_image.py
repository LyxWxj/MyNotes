"""诊断全黑图片: 从 response.json 解码并检查 VAE 输出值分布."""
import base64
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    resp_path = Path(__file__).parent / "response.json"
    with open(resp_path) as f:
        data = json.load(f)

    b64 = data["data"][0]["b64_json"]
    raw = base64.b64decode(b64)

    # 1. 检查已保存的图片
    img = Image.open(Path(__file__).parent / "output.png")
    arr = np.array(img)
    print("=" * 60)
    print("已保存的 output.png 分析")
    print("=" * 60)
    print(f"  Mode: {img.mode}, Size: {img.size}")
    print(f"  Dtype: {arr.dtype}, Shape: {arr.shape}")
    print(f"  Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.4f}")
    pct_zero = 100.0 * np.sum(arr == 0) / arr.size
    print(f"  零值像素占比: {pct_zero:.2f}%")
    print()

    # 2. 模拟正确的 [-1,1] -> [0,255] 转换
    # VAE decode 输出 clamp 到 [-1, 1], 需要 (x + 1) / 2 * 255
    # 但我们拿到的已经是编码后的 PNG, 所以用 response 中的原始数据推算
    print("=" * 60)
    print("根因分析")
    print("=" * 60)
    print("  VAE decode 输出范围: [-1.0, 1.0]")
    print("  当前代码: pixel = (value * 255).clip(0, 255)")
    print("    value=-1.0 -> pixel=0 (黑)")
    print("    value= 0.0 -> pixel=0 (黑)")
    print("    value=+0.5 -> pixel=127")
    print("    value=+1.0 -> pixel=255")
    print()
    print("  正确代码: pixel = ((value + 1) / 2 * 255).clip(0, 255)")
    print("    value=-1.0 -> pixel=0   (黑)")
    print("    value= 0.0 -> pixel=127 (灰)")
    print("    value=+0.5 -> pixel=191")
    print("    value=+1.0 -> pixel=255 (白)")
    print()
    print("  结论: 当前代码把 [-1,1] 的一半映射到了负数然后 clip 成 0,")
    print("  所以约 50% 像素直接变黑, 剩余像素也只有正常亮度的一半.")
    print("  对于典型图片, 这会导致几乎全黑的效果.")
    print()

    # 3. 尝试修正现有图片 (信息有限, 但可以近似反推)
    print("=" * 60)
    print("尝试修正 (近似反推)")
    print("=" * 60)
    # 已编码的值是 v*255 clipped, 其中 v 是 [-1,1] 的值
    # 如果 original_value 在 [0,1], 编码后 = original_value * 255
    # 如果 original_value 在 [-1,0), 编码后 = 0 (信息丢失)
    # 所以我们只能恢复 [0,1] 部分, 无法恢复被 clip 掉的负值
    arr_fixed = arr.astype(np.float32)
    # 近似: 如果原始值是 v, 编码后是 max(v*255, 0)
    # 正确编码应该是 (v+1)/2*255
    # 令 encoded = v*255 (for v>=0), 则 v = encoded/255
    # 正确像素 = (v+1)/2*255 = (encoded/255 + 1)/2 * 255 = encoded/2 + 127.5
    arr_fixed = (arr_fixed / 2.0 + 127.5).clip(0, 255).astype(np.uint8)
    out_path = Path(__file__).parent / "output_approx_fix.png"
    Image.fromarray(arr_fixed).save(out_path)
    print(f"  近似修正图保存到: {out_path}")
    print(f"  Min: {arr_fixed.min()}, Max: {arr_fixed.max()}, Mean: {arr_fixed.mean():.2f}")
    print("  注意: 负值区域的信息已丢失, 这只是近似恢复.")
    print()
    print("=" * 60)
    print("修复方案")
    print("=" * 60)
    print("  文件: multi_instance_launcher.py, _result_to_b64_png 方法 (第 278 行)")
    print("  将:")
    print("    image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)")
    print("  改为:")
    print("    image_np = ((image_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)")


if __name__ == "__main__":
    main()
