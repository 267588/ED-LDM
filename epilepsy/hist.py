import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取源图像和目标图像
source_img = cv2.imread(r'vq-gan_recon/5.png', cv2.IMREAD_GRAYSCALE)
target_img = cv2.imread(r'pv-gan_recon/5.png', cv2.IMREAD_GRAYSCALE)

# 计算源图像和目标图像的直方图
source_hist, bins1 = np.histogram(source_img, bins=256, range=[0, 256])
target_hist, bins2 = np.histogram(target_img, bins=256, range=[0, 256])

# 计算累积直方图
source_cdf = source_hist.cumsum()
source_cdf = (source_cdf / source_cdf[-1]).astype(np.float32)

target_cdf = target_hist.cumsum()

target_cdf = (target_cdf / target_cdf[-1]).astype(np.float32)

# 使用累积直方图进行直方图匹配
matched_cdf = np.interp(source_cdf, target_cdf, range(256)).astype(np.uint8)
matched_img = matched_cdf[source_img]

print(matched_img.shape)
cv2.imwrite("hists/5.png",matched_img)

# 将灰度图像转换为伪彩色图像
# pseudo_color_img = cv2.applyColorMap(matched_img, cv2.COLORMAP_JET)
# matched_rgb = cv2.cvtColor(matched_img, cv2.COLOR_GRAY2RGB)
# 显示图像和直方图
# plt.subplot(2, 2, 1)
# plt.imshow(source_img, cmap='gray')
# plt.title('Source Image')
#
# plt.subplot(2, 2, 2)
# plt.imshow(target_img, cmap='gray')
# plt.title('Target Image')
#
# plt.subplot(2, 2, 3)
# plt.imshow(matched_img, cmap='gray')
# plt.title('Matched Image')
#
# plt.subplot(2, 2, 4)

x_values = np.linspace(0, 1, 256)  # 生成0到1之间的256个均匀分布的数值
plt.plot(x_values, source_cdf, color='r', label='Source CDF')
plt.plot(x_values, target_cdf, color='b', label='Target CDF')
# plt.legend()
plt.savefig("cdf.jpg", bbox_inches='tight')

plt.xlabel('Pixel intensity')
plt.ylabel('Cumulative distribution function (CDF)')
plt.tight_layout()
plt.show()

# plt.imshow(target_hist)
# plt.show()
# plt.subplot(2, 2, 1)
# plt.bar(bins1[:-1], source_hist, width=1, color='blue', edgecolor='black')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.title('Histogram')
# plt.subplot(2, 2, 2)
# plt.bar(bins2[:-1], target_hist, width=1, color='blue', edgecolor='black')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.title('Histogram')
# plt.show()