import cv2  
import gradio as gr  
import numpy as np  
import random
from scipy import ndimage
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2

# 原始图像处理函数
def original_image(input_image):
    return input_image



# 添加高斯噪声
def gaussian_noise(input_image, mean, sigma):
    img = np.array(input_image / 255, dtype=float)
    noise = np.random.normal(float(mean), float(sigma), img.shape)
    out = img + noise
    out = np.clip(out, 0.0, 1.0)
    out = np.uint8(out * 255)
    return out

# 添加椒盐噪声
def salt_and_pepper_noise(input_image, prob, thres):
    output = np.zeros(input_image.shape, np.uint8)
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            rdn = random.random()
            if rdn < float(prob):
                output[i][j] = 0
            elif rdn > (1 - float(thres)):
                output[i][j] = 255
            else:
                output[i][j] = input_image[i][j]
    return output

# 灰度化处理
def grayscale(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    return gray_image

# 反转颜色
def invert_colors(input_image):
    inverted_image = cv2.bitwise_not(input_image)
    return inverted_image

# 二值化处理
def to_binarization(input_image, x, y):
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, int(x), int(y), cv2.THRESH_BINARY)
    return img

# BGR_HSV转换
class BGR_HSV:
    def __init__(self, img, conversion_type):
        self.img = img
        self.conversion_type = conversion_type
    def convert(self):
        function_map = {
            'B': self.toB,
            'G': self.toG,
            'R': self.toR,
            'H': self.toH,
            'S': self.toS,
            'V': self.toV,
        }
        if self.conversion_type in function_map:
            return function_map[self.conversion_type]()  # 调用对应的函数
        else:
            raise ValueError("Invalid filter type")
    # B通道
    def toB(self):
        b = self.img[:, :, 0]
        return b
    # G通道
    def toG(self):
        g = self.img[:, :, 1]
        return g
    # R通道
    def toR(self):
        r = self.img[:, :, 2]
        return r
    # H通道
    def toH(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h = img[:, :, 0]
        return h
    # S通道
    def toS(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        s = img[:, :, 1]
        return s
    # V通道
    def toV(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        v = img[:, :, 2]
        return v
def bgr_hsv_conversion(input_image, conversion_type):
    converter = BGR_HSV(input_image, conversion_type)
    return converter.convert()

# 去噪
# 获取滤波器大小
def get_my_size(s):
    if s == '3*3':
        return (3, 3)
    elif s == '5*5':
        return (5, 5)
    elif s == '7*7':
        return (7, 7)
    else:
        raise Exception("Invalid size")
# 获取列表的中间值的函数
def get_middle(array):
    length = len(array)
    for i in range(length):
        for j in range(i + 1, length):
            if array[j] > array[i]:
                array[j], array[i] = array[i], array[j]
    return array[int(length / 2)]
class Denoise:
    def __init__(self, img, filter_type, size_type, gauss_x=None, gauss_y=None):
        self.img = img
        self.filter_type = filter_type
        self.size_type = size_type
        self.gauss_x = gauss_x
        self.gauss_y = gauss_y

    def convert(self):
        function_map = {
            'mean': self.mean_filtering,
            'median': self.median_filtering,
            'gaussian': self.gaussian_filtering
        }

        if self.filter_type in function_map:
            return function_map[self.filter_type]()  # 调用对应的函数
        else:
            raise ValueError("Invalid filter type")

    # 均值滤波
    def mean_filtering(self):
        size = get_my_size(self.size_type)
        img = cv2.blur(self.img, size)
        return img

    # 中值滤波
    def median_filtering(self):
        size = get_my_size(self.size_type)
        img = cv2.medianBlur(self.img, size[0])
        return img

    # 高斯滤波
    def gaussian_filtering(self):
        size = get_my_size(self.size_type)
        img = cv2.GaussianBlur(self.img, size, float(self.gauss_x), float(self.gauss_y))
        return img
def denoise(input_image, filter_type, size_type, gauss_x=None, gauss_y=None):
    denoiser = Denoise(input_image, filter_type, size_type, gauss_x, gauss_y)
    return denoiser.convert()

# 边缘检测
class EdgeDetection:
    def __init__(self, img, detection_type):
        self.img = img
        self.detection_type = detection_type

    def convert(self):
        function_map = {
            'roberts': self.roberts,
            'prewitt': self.prewitt,
            'sobel': self.sobel,
            'laplacian': self.laplacian,
            'loG': self.loG,
            'canny': self.canny,
        }

        if self.detection_type in function_map:
            return function_map[self.detection_type]()  # 调用对应的函数
        else:
            raise ValueError("Invalid filter type")
    # Roberts算子
    def roberts(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv2.filter2D(img, cv2.CV_16S, kernelx)
        y = cv2.filter2D(img, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return img
    # Prewitt算子
    def prewitt(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x = cv2.filter2D(img, cv2.CV_16S, kernelx)
        y = cv2.filter2D(img, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return img
    # Sobel算子
    def sobel(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return img
    # Laplacian算子
    def laplacian(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        kernel_size = (5, 5)
        sigma = 0
        img = cv2.GaussianBlur(img, kernel_size, sigma)
        img = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
        img = cv2.convertScaleAbs(img)
        return img
    # LoG算子
    def loG(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img = cv2.copyMakeBorder(img, 2, 2, 2, 2, borderType=cv2.BORDER_REPLICATE)
        img = cv2.GaussianBlur(img, (3, 3), 0, 0)
        m1 = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]],
                      dtype=int)
        rows = img.shape[0]
        cols = img.shape[1]
        image1 = np.zeros([rows - 4, cols - 4], dtype=np.int32)
        for i in range(2, rows - 2):
            for j in range(2, cols - 2):
                image1[i - 2, j - 2] = np.sum((m1 * img[i - 2:i + 3, j - 2:j + 3]))
        img = cv2.convertScaleAbs(image1)
        return img
    # Canny算子
    def canny(self):
        img = cv2.GaussianBlur(self.img, (3, 3), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gradx = cv2.Sobel(img, cv2.CV_16SC1, 1, 0)
        grady = cv2.Sobel(img, cv2.CV_16SC1, 0, 1)
        img = cv2.Canny(gradx, grady, 50, 150)
        return img
def edge_detection(input_image, detection_type):
    detector = EdgeDetection(input_image, detection_type)
    return detector.convert()


# 几何变换
def parse_points(*args):
    return [np.array([float(coord) for coord in point.split(',')]) for point in args]
class GeometryTransfer:
    def __init__(self, img, transformation_type, 
                 scale_h=None, scale_w=None, scale_method=None, 
                 translation_x=None, translation_y=None, 
                 rotate_w=None, rotate_h=None, rotate_angle=None, 
                 flip_type=None, 
                 p1=None, p2=None, p3=None, p4=None, p5=None, p6=None, p7=None, p8=None,
                 a1=None, a2=None, a3=None, a4=None, a5=None, a6=None):
        self.img = img
        self.transformation_type = transformation_type
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_method = scale_method
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.rotate_w = rotate_w
        self.rotate_h = rotate_h
        self.rotate_angle = rotate_angle
        self.flip_type = flip_type
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.p7 = p7
        self.p8 = p8
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6

    def convert(self):
        function_map = {
            'scale': self.scale,
            'translation': self.translation,
            'rotate': self.rotate,
            'flip': self.flip,
            'perspective': self.perspective,
            'affine': self.affine,
        }

        if self.transformation_type in function_map:
            return function_map[self.transformation_type]()
        else:
            raise ValueError("Invalid transformation type")

    # 缩放
    def scale(self):
        fx = float(self.scale_h)
        fy = float(self.scale_w)
        d_size = (int(self.img.shape[1] * fx), int(self.img.shape[0] * fy))

        if self.scale_method == "INTER_LINEAR":
            img = cv2.resize(self.img, dsize=d_size, interpolation=cv2.INTER_LINEAR)
        elif self.scale_method == "INTER_NEAREST":
            img = cv2.resize(self.img, dsize=d_size, interpolation=cv2.INTER_NEAREST)
        elif self.scale_method == "INTER_AREA":
            img = cv2.resize(self.img, dsize=d_size, interpolation=cv2.INTER_AREA)
        elif self.scale_method == "INTER_CUBIC":
            img = cv2.resize(self.img, dsize=d_size, interpolation=cv2.INTER_CUBIC)
        elif self.scale_method == "INTER_LANCZOS4":
            img = cv2.resize(self.img, dsize=d_size, interpolation=cv2.INTER_LANCZOS4)
        
        return img
    
    # 平移
    def translation(self):
        height, width = self.img.shape[:2]
        M = np.float32([[1, 0, self.translation_x], [0, 1, self.translation_y]])
        img = cv2.warpAffine(self.img, M, (width, height))
        return img
    
    # 旋转
    def rotate(self):
        height, width = self.img.shape[:2]
        M = cv2.getRotationMatrix2D((width * self.rotate_w, height * self.rotate_h), self.rotate_angle, 1)
        img = cv2.warpAffine(self.img, M, (width, height))
        return img
    
    # 翻转
    def flip(self):
        if self.flip_type == '水平镜像翻转':
            img = cv2.flip(self.img, 1)
        elif self.flip_type == '垂直镜像翻转':
            img = cv2.flip(self.img, 0)
        elif self.flip_type == '对角镜像翻转':
            img = cv2.flip(self.img, -1) 
        return img
    
    # 透视变换
    def perspective(self):
        rows, cols = self.img.shape[:2]
        post1 = np.float32([self.p1, self.p2, self.p3, self.p4])
        post2 = np.float32([self.p5, self.p6, self.p7, self.p8])
        M = cv2.getPerspectiveTransform(post1, post2)
        img = cv2.warpPerspective(self.img, M, (cols, rows))
        return img
    
    # 仿射变换
    def affine(self):
        rows, cols = self.img.shape[:2]
        post1 = np.float32([self.a1, self.a2, self.a3])
        post2 = np.float32([self.a4, self.a5, self.a6])
        M = cv2.getAffineTransform(post1, post2)
        img = cv2.warpAffine(self.img, M, (cols, rows))
        return img
def apply_geometry_transfer(input_image, transformation_type, scale_h=None, scale_w=None, scale_method=None, 
                            translation_x=None, translation_y=None, rotate_w=None, rotate_h=None, rotate_angle=None, 
                            flip_type=None, p1=None, p2=None, p3=None, p4=None, p5=None, p6=None, p7=None, p8=None,
                            a1=None, a2=None, a3=None, a4=None, a5=None, a6=None):
    geometry_transfer = GeometryTransfer(input_image, transformation_type, 
                                         scale_h, scale_w, scale_method, 
                                         translation_x, translation_y, 
                                         rotate_w, rotate_h, rotate_angle, 
                                         flip_type, 
                                         p1, p2, p3, p4, p5, p6, p7, p8,
                                         a1, a2, a3, a4, a5, a6)
    return geometry_transfer.convert()

# 用于空域锐化
def roberts_sharpening(image):
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])
    gradient_x = ndimage.convolve(image, roberts_x)
    gradient_y = ndimage.convolve(image, roberts_y)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    sharpened_image = image + gradient_magnitude
    sharpened_image = np.clip(sharpened_image, 0, 255)
    return sharpened_image.astype(np.uint8)

def sobel_sharpening(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_x = ndimage.convolve(image, sobel_x)
    gradient_y = ndimage.convolve(image, sobel_y)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    sharpened_image = image + gradient_magnitude
    sharpened_image = np.clip(sharpened_image, 0, 255)
    return sharpened_image.astype(np.uint8)

def prewitt_sharpening(image):
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    gradient_x = ndimage.convolve(image, prewitt_x)
    gradient_y = ndimage.convolve(image, prewitt_y)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    sharpened_image = image + gradient_magnitude
    sharpened_image = np.clip(sharpened_image, 0, 255)
    return sharpened_image.astype(np.uint8)

def laplacian_sharpening(image):
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    gradient = ndimage.convolve(image, laplacian)
    sharpened_image = image - gradient
    sharpened_image = np.clip(sharpened_image, 0, 255)
    return sharpened_image.astype(np.uint8)

# 用于频域锐化
def ideal_high_pass_filter(img, cutoff_freq, show):
    f = fftshift(fft2(img))
    rows, cols = img.shape
    x = np.linspace(-cols // 2, cols // 2 - 1, cols)
    y = np.linspace(-rows // 2, rows // 2 - 1, rows)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv ** 2 + yv ** 2)
    mask = np.ones((rows, cols))
    mask[dist <= cutoff_freq] = 0
    filtered_f = f * mask
    sharpened_img = np.real(ifft2(ifftshift(filtered_f)))
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)
    enhanced_img = cv2.addWeighted(img, 1, sharpened_img, 1, 0)
    enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
    return enhanced_img if show == 0 else sharpened_img

def butterworth_high_pass_filter(img, cutoff_freq, order, show):
    f = fftshift(fft2(img))
    rows, cols = img.shape
    x = np.linspace(-cols // 2, cols // 2 - 1, cols)
    y = np.linspace(-rows // 2, rows // 2 - 1, rows)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv ** 2 + yv ** 2)
    H = 1 / (1 + (cutoff_freq / dist) ** (2 * order))
    filtered_f = f * H
    sharpened_img = np.real(ifft2(ifftshift(filtered_f)))
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)
    enhanced_img = cv2.addWeighted(img, 1, sharpened_img, 1, 0)
    enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
    return enhanced_img if show == 0 else sharpened_img

def gaussian_high_pass_filter(img, cutoff_freq, show):
    f = fftshift(fft2(img))
    rows, cols = img.shape
    x = np.linspace(-cols // 2, cols // 2 - 1, cols)
    y = np.linspace(-rows // 2, rows // 2 - 1, rows)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv ** 2 + yv ** 2)
    mask = 1 - np.exp(-(dist ** 2) / (2 * cutoff_freq ** 2))
    filtered_f = f * mask
    sharpened_img = np.real(ifft2(ifftshift(filtered_f)))
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)
    enhanced_img = cv2.addWeighted(img, 1, sharpened_img, 1, 0)
    enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
    return enhanced_img if show == 0 else sharpened_img

# 直方图均衡化，直方图正规化，色彩增强，空域锐化，频域锐化
class ImageEnhancement:
    def __init__(self, img, enhancement_type, airspace_sharpening_method=None, frequency_sharpening_method=None):
        self.img = img
        self.enhancement_type = enhancement_type
        self.airspace_sharpening_method = airspace_sharpening_method
        self.frequency_sharpening_method = frequency_sharpening_method

    def convert(self):
        function_map = {
            '直方图均衡化': self.histogram_equalization,
            '直方图正规化': self.histogram_normalization,
            '色彩增强': self.color_enhance,
            '空域锐化': self.airspace_sharpening,
            '频域锐化': self.frequency_sharpening,
        }

        if self.enhancement_type in function_map:
            return function_map[self.enhancement_type]()
        else:
            raise ValueError("Invalid enhancement type")

    # 直方图均衡化
    def histogram_equalization(self):
        img = self.img.copy()
        for i in range(3):  # 对每个颜色通道进行直方图均衡化
            img[:, :, i] = cv2.equalizeHist(img[:, :, i])
        return img

    # 直方图正规化
    def histogram_normalization(self):
        img = self.img.copy()
        for i in range(3):  # 对每个颜色通道进行处理
            channel = img[:, :, i]
            channel_flat = channel.flatten()
        
            # 计算直方图
            hist, bins = np.histogram(channel_flat, 256, [0, 256])
        
            # 计算累积分布函数（CDF）
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
        
            # 使用CDF进行正规化
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        
            # 应用CDF到原图像
            channel_normalized = cdf[channel_flat]
            img[:, :, i] = np.reshape(channel_normalized, channel.shape)
        
        return img

    # 色彩增强
    def color_enhance(self):
        image = self.img
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = image_lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clipped_l_channel = clahe.apply(l_channel)
        enhanced_image_lab = cv2.merge([clipped_l_channel, image_lab[:, :, 1], image_lab[:, :, 2]])
        enhanced_image = cv2.cvtColor(enhanced_image_lab, cv2.COLOR_LAB2BGR)
        return enhanced_image

    # 空域锐化
    def airspace_sharpening(self):
        img = self.img.copy()
        for i in range(3):  # 对每个颜色通道进行锐化处理
            channel = img[:, :, i]
            if self.airspace_sharpening_method == 'Roberts':
                img[:, :, i] = roberts_sharpening(channel)
            elif self.airspace_sharpening_method == 'Sobel':
                img[:, :, i] = sobel_sharpening(channel)
            elif self.airspace_sharpening_method == 'Prewitt':
                img[:, :, i] = prewitt_sharpening(channel)
            elif self.airspace_sharpening_method == 'Laplacian':
                img[:, :, i] = laplacian_sharpening(channel)
            else:
                raise Exception("Invalid airspace sharpening method")
        return img


    # 频域锐化
    def frequency_sharpening(self):
        img = self.img.copy()
        cutoff_freq = 10
        for i in range(3):  # 对每个颜色通道进行频域锐化处理
            channel = img[:, :, i]
            if self.frequency_sharpening_method == '理想高通滤波':
                img[:, :, i] = ideal_high_pass_filter(channel, cutoff_freq, 0)
            elif self.frequency_sharpening_method == '巴特沃斯高通滤波':
                img[:, :, i] = butterworth_high_pass_filter(channel, cutoff_freq, 2, 0)
            elif self.frequency_sharpening_method == '高斯高通滤波':
                img[:, :, i] = gaussian_high_pass_filter(channel, cutoff_freq, 0)
            else:
                raise Exception("Invalid frequency sharpening method")
        return img
    
def image_enhancement(input_image, enhancement_type, airspace_sharpening_method=None, frequency_sharpening_method=None):
    enhancer = ImageEnhancement(input_image, enhancement_type, airspace_sharpening_method, frequency_sharpening_method)
    return enhancer.convert()

# 线检测
class LineDetection:
    def __init__(self, img, method):
        self.img = img
        self.method = method

    def hough(self):
        img = self.img
        img = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        if self.method == "HoughLines算法":
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            result = img.copy()
            if lines is not None:
                for i_line in lines:
                    for line in i_line:
                        rho = line[0]
                        theta = line[1]
                        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                            pt1 = (int(rho / np.cos(theta)), 0)
                            pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                            cv2.line(result, pt1, pt2, (0, 0, 255))
                        else:
                            pt1 = (0, int(rho / np.sin(theta)))
                            pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                            cv2.line(result, pt1, pt2, (0, 0, 255), 1)
                img = result

        elif self.method == "HoughLinesP算法":
            minLineLength = 150
            maxLineGap = 15
            linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=minLineLength, maxLineGap=maxLineGap)
            result_P = img.copy()
            if linesP is not None:
                for i_P in linesP:
                    for x1, y1, x2, y2 in i_P:
                        cv2.line(result_P, (x1, y1), (x2, y2), (0, 255, 0), 3)
                img = result_P
        return img
def line_detection(input_image, method):
    detector = LineDetection(input_image, method)
    return detector.hough()

def get_my_structuring_element(s):
    if s == '矩形结构元':
        return cv2.MORPH_RECT
    elif s == '交叉结构元':
        return cv2.MORPH_CROSS
    elif s == '椭圆形结构元':
        return cv2.MORPH_ELLIPSE
    else:
        raise Exception("Invalid structuring element type")

# 形态学操作
class Morphology:
    def __init__(self, img, operation_type, structuring_element_type, size_type):
        self.img = img
        self.operation_type = operation_type
        self.structuring_element_type = structuring_element_type
        self.size_type = size_type

    def convert(self):
        function_map = {
            '腐蚀': self.corrosion,
            '膨胀': self.dilatation,
            '开运算': self.open_operation,
            '闭运算': self.close_operation,
            '顶帽运算': self.top_hat_operation,
            '底帽运算': self.black_hat_operation,
            '形态学梯度': self.morphological_gradient,
        }

        if self.operation_type in function_map:
            return function_map[self.operation_type]()
        else:
            raise ValueError("Invalid filter type")
        
    # 腐蚀
    def corrosion(self):
        structuring_element = get_my_structuring_element(self.structuring_element_type)
        size = get_my_size(self.size_type)
        kernel = cv2.getStructuringElement(structuring_element, size)
        img = cv2.erode(self.img, kernel)
        return img

    # 膨胀
    def dilatation(self):
        structuring_element = get_my_structuring_element(self.structuring_element_type)
        size = get_my_size(self.size_type)
        kernel = cv2.getStructuringElement(structuring_element, size)
        img = cv2.dilate(self.img, kernel)
        return img

    # 开运算
    def open_operation(self):
        structuring_element = get_my_structuring_element(self.structuring_element_type)
        size = get_my_size(self.size_type)
        kernel = cv2.getStructuringElement(structuring_element, size)
        img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
        return img

    # 闭运算
    def close_operation(self):
        structuring_element = get_my_structuring_element(self.structuring_element_type)
        size = get_my_size(self.size_type)
        kernel = cv2.getStructuringElement(structuring_element, size)
        img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        return img

    # 顶帽运算
    def top_hat_operation(self):
        structuring_element = get_my_structuring_element(self.structuring_element_type)
        size = get_my_size(self.size_type)
        kernel = cv2.getStructuringElement(structuring_element, size)
        img = cv2.morphologyEx(self.img, cv2.MORPH_TOPHAT, kernel)
        return img

    # 底帽运算
    def black_hat_operation(self):
        structuring_element = get_my_structuring_element(self.structuring_element_type)
        size = get_my_size(self.size_type)
        kernel = cv2.getStructuringElement(structuring_element, size)
        img = cv2.morphologyEx(self.img, cv2.MORPH_BLACKHAT, kernel)
        return img

    # 形态学梯度
    def morphological_gradient(self):
        structuring_element = get_my_structuring_element(self.structuring_element_type)
        size = get_my_size(self.size_type)
        kernel = cv2.getStructuringElement(structuring_element, size)
        img = cv2.morphologyEx(self.img, cv2.MORPH_GRADIENT, kernel)
        return img
def morphology(input_image, operation_type, structuring_element_type, size_type):
    morph = Morphology(input_image, operation_type, structuring_element_type, size_type)
    return morph.convert()

# 自定义卷积核
def custom_filter(input_image, x1y1, x1y2, x1y3, x2y1, x2y2, x2y3, x3y1, x3y2, x3y3):
    kernel = np.array([[x1y1, x1y2, x1y3], [x2y1, x2y2, x2y3], [x3y1, x3y2, x3y3]])
    return cv2.filter2D(input_image, -1, kernel)



def apply_opencv_methods(input_image, method,
                         mean, sigma, 
                         prob, thres,
                         binary_thres1, binary_thres2, 
                         conversion_type, 
                         detection_type, 
                         filter_type, size_type, gauss_x, gauss_y,
                         transformation_type, scale_h, scale_w, scale_method, translation_x, translation_y, rotate_w, rotate_h, rotate_angle, 
                         flip_type, p1, p2, p3, p4, p5, p6, p7, p8, a1, a2, a3, a4, a5, a6,
                         enhancement_type, airspace_sharpening_method, frequency_sharpening_method,
                         line_detection_method,
                         operation_type, structuring_element_type, size_type2, 
                         x1y1, x1y2, x1y3, x2y1, x2y2, x2y3, x3y1, x3y2, x3y3):
    if method == "原图":
        return original_image(input_image)
    elif method == "自定义卷积核":
        return custom_filter(input_image, x1y1, x1y2, x1y3, x2y1, x2y2, x2y3, x3y1, x3y2, x3y3)
    elif method == "高斯噪声":
        return gaussian_noise(input_image, mean, sigma)
    elif method == "椒盐噪声":
        return salt_and_pepper_noise(input_image, prob, thres)
    elif method == "二值化":
        return to_binarization(input_image, binary_thres1, binary_thres2)
    elif method == "BGRHSV转换":
        return bgr_hsv_conversion(input_image, conversion_type)
    elif method == "边缘检测":
        return edge_detection(input_image, detection_type)
    elif method == "去噪":
        return denoise(input_image, filter_type, size_type, gauss_x, gauss_y)
    elif method == "几何变换":
        p1, p2, p3, p4, p5, p6, p7, p8 = parse_points(p1, p2, p3, p4, p5, p6, p7, p8)
        a1, a2, a3, a4, a5, a6 = parse_points(a1, a2, a3, a4, a5, a6)
        return apply_geometry_transfer(input_image, transformation_type, scale_h, scale_w, scale_method, translation_x, translation_y, rotate_w, rotate_h, rotate_angle, flip_type, p1, p2, p3, p4, p5, p6, p7, p8, a1, a2, a3, a4, a5, a6)
    elif method == "图像增强":
        return image_enhancement(input_image, enhancement_type, airspace_sharpening_method, frequency_sharpening_method)
    elif method == "线检测":
        return line_detection(input_image, line_detection_method)
    elif method == "形态学操作":
        return morphology(input_image, operation_type, structuring_element_type, size_type2)
    else:
        methods = {
            "灰度化": grayscale,
            "反转颜色": invert_colors,
        }
        return methods[method](input_image)

# 更新参数的可见性
def update_parameters(method):
    default_visibility = gr.update(visible=False)
    params = {
        "原图": [default_visibility] * 52,
        "高斯噪声": [gr.update(visible=True), gr.update(visible=True)] + [default_visibility] * 50,
        "椒盐噪声": [default_visibility] * 2 + [gr.update(visible=True), gr.update(visible=True)] + [default_visibility] * 48,
        "灰度化": [default_visibility] * 52,
        "反转颜色": [default_visibility] * 52,
        "二值化": [default_visibility] * 4 + [gr.update(visible=True), gr.update(visible=True)] + [default_visibility] * 46,
        "BGRHSV转换": [default_visibility] * 6 + [gr.update(visible=True)] + [default_visibility] * 45,
        "边缘检测": [default_visibility] * 7 + [gr.update(visible=True)] + [default_visibility] * 44,
        "去噪": [default_visibility] * 8 + [gr.update(visible=True)] * 4 + [default_visibility] * 40,
        "几何变换": [default_visibility] * 12 + [gr.update(visible=True)] * 24 + [default_visibility] * 16,
        "图像增强": [default_visibility] * 36 + [gr.update(visible=True)] * 3 + [default_visibility] * 13,
        "线检测": [default_visibility] * 39 + [gr.update(visible=True)] + [default_visibility] * 12,
        "形态学操作": [default_visibility] * 40 + [gr.update(visible=True)] * 3 + [default_visibility] * 9,
        "自定义卷积核": [default_visibility] * 43 + [gr.update(visible=True)] * 9
    }
    return params.get(method, [default_visibility] * 52)

# 创建 Gradio 接口
iface = gr.Blocks()

with iface:
    with gr.Row():
        input_image = gr.Image(type="numpy")
        output_image = gr.Image(type="numpy")
    with gr.Row():
        method = gr.Radio(
            choices=["原图",
                     "高斯噪声", "椒盐噪声",
                     "灰度化", "反转颜色", "二值化",
                     "BGRHSV转换",
                     "边缘检测",
                     "去噪",
                     "几何变换",
                     "图像增强",
                     "线检测",
                     "形态学操作",
                     "自定义卷积核"], value="原图")
    with gr.Row():
        mean = gr.Number(value=0, label="高斯噪声: 平均值", visible=False)
        sigma = gr.Number(value=1, label="高斯噪声: 标准差", visible=False)

        prob = gr.Number(value=0.05, label="椒盐噪声: 概率", visible=False)
        thres = gr.Number(value=0.05, label="椒盐噪声: 阈值", visible=False)

        binary_thres1 = gr.Number(value=127, label="二值化: 阈值1", visible=False)
        binary_thres2 = gr.Number(value=255, label="二值化: 阈值2", visible=False)

        conversion_type = gr.Dropdown(['R', 'G', 'B', 'H', 'S', 'V'], label="BGRHSV转换类型", visible=False)

        detection_type = gr.Dropdown(['roberts', 'prewitt', 'sobel', 'laplacian', 'loG', 'canny'], label="边缘检测类型", visible=False)
        
        filter_type = gr.Dropdown(['mean', 'median', 'gaussian'], label="滤波类型", visible=False)
        size_type = gr.Dropdown(['3*3', '5*5', '7*7'], label="滤波器大小", visible=False)
        gauss_x = gr.Number(value=0, label="高斯滤波X方向", visible=False)
        gauss_y = gr.Number(value=0, label="高斯滤波Y方向", visible=False)
        
        transformation_type = gr.Dropdown(["scale", "translation", "rotate", "flip", "perspective", "affine"], label="几何转换类型", visible=False)
        
        scale_h = gr.Number(value=1.0, label="缩放高度比例", visible=False)
        scale_w = gr.Number(value=1.0, label="缩放宽度比例", visible=False)
        scale_method = gr.Dropdown(["INTER_LINEAR", "INTER_NEAREST", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"], label="缩放方法", visible=False)
        
        translation_x = gr.Number(value=0, label="平移 X", visible=False)
        translation_y = gr.Number(value=0, label="平移 Y", visible=False)
        
        rotate_w = gr.Number(value=0.5, label="旋转中心X", visible=False)
        rotate_h = gr.Number(value=0.5, label="旋转中心Y", visible=False)
        rotate_angle = gr.Number(value=0, label="旋转角度", visible=False)
        
        flip_type = gr.Dropdown(["水平镜像翻转", "垂直镜像翻转", "对角镜像翻转"], label="翻转类型", visible=False)
        
        p1 = gr.Textbox(value="0,0", label="透视变换点1 (x,y)", visible=True)
        p2 = gr.Textbox(value="0,0", label="透视变换点2 (x,y)", visible=True)
        p3 = gr.Textbox(value="0,0", label="透视变换点3 (x,y)", visible=True)
        p4 = gr.Textbox(value="0,0", label="透视变换点4 (x,y)", visible=True)
        p5 = gr.Textbox(value="0,0", label="透视变换点5 (x,y)", visible=True)
        p6 = gr.Textbox(value="0,0", label="透视变换点6 (x,y)", visible=True)
        p7 = gr.Textbox(value="0,0", label="透视变换点7 (x,y)", visible=True)
        p8 = gr.Textbox(value="0,0", label="透视变换点8 (x,y)", visible=True)
        
        a1 = gr.Textbox(value="0,0", label="仿射变换点1 (x,y)", visible=True)
        a2 = gr.Textbox(value="0,0", label="仿射变换点2 (x,y)", visible=True)
        a3 = gr.Textbox(value="0,0", label="仿射变换点3 (x,y)", visible=True)
        a4 = gr.Textbox(value="0,0", label="仿射变换点4 (x,y)", visible=True)
        a5 = gr.Textbox(value="0,0", label="仿射变换点5 (x,y)", visible=True)
        a6 = gr.Textbox(value="0,0", label="仿射变换点6 (x,y)", visible=True)
        
        enhancement_type = gr.Dropdown(['直方图均衡化', '直方图正规化', '色彩增强', '空域锐化', '频域锐化'], label="图像增强类型", visible=False)
        
        airspace_sharpening_method = gr.Dropdown(['Roberts', 'Sobel', 'Prewitt', 'Laplacian'], label="空域锐化方法", visible=False)
        
        frequency_sharpening_method = gr.Dropdown(['理想高通滤波', '巴特沃斯高通滤波', '高斯高通滤波'], label="频域锐化方法", visible=False)
        
        line_detection_method = gr.Dropdown(['HoughLines算法', 'HoughLinesP算法'], label="线检测方法", visible=False)
       
        operation_type = gr.Dropdown(['腐蚀', '膨胀', '开运算', '闭运算', '顶帽运算', '底帽运算', '形态学梯度'], label="形态学操作类型", visible=False)
        structuring_element_type = gr.Dropdown(['矩形结构元', '交叉结构元', '椭圆形结构元'], label="结构元类型", visible=False)
        size_type2 = gr.Dropdown(['3*3', '5*5', '7*7'], label="结构元大小", visible=False)

        x1y1 = gr.Number(value=-1.0, label="自定义卷积核x1y1", visible=False)
        x1y2 = gr.Number(value=-1.0, label="自定义卷积核x1y2", visible=False)
        x1y3 = gr.Number(value=-1.0, label="自定义卷积核x1y3", visible=False)
        x2y1 = gr.Number(value=-1.0, label="自定义卷积核x2y1", visible=False)
        x2y2 = gr.Number(value= 9.0, label="自定义卷积核x2y2", visible=False)
        x2y3 = gr.Number(value=-1.0, label="自定义卷积核x2y3", visible=False)
        x3y1 = gr.Number(value=-1.0, label="自定义卷积核x3y1", visible=False)
        x3y2 = gr.Number(value=-1.0, label="自定义卷积核x3y2", visible=False)
        x3y3 = gr.Number(value=-1.0, label="自定义卷积核x3y3", visible=False)

    
    
    # 参数更新按钮
    method.change(fn=update_parameters, inputs=method, outputs=[
        mean, sigma,
        prob, thres,
        binary_thres1, binary_thres2,
        conversion_type,
        detection_type, filter_type, size_type, gauss_x, gauss_y,
        transformation_type, scale_h, scale_w, scale_method, translation_x, translation_y, rotate_w, rotate_h, rotate_angle, flip_type, p1, p2, p3, p4, p5, p6, p7, p8, a1, a2, a3, a4, a5, a6,
        enhancement_type, airspace_sharpening_method, frequency_sharpening_method,
        line_detection_method,
        operation_type, structuring_element_type, size_type2,
        x1y1, x1y2, x1y3, x2y1, x2y2, x2y3, x3y1, x3y2, x3y3
    ])
    
    # 图像处理按钮
    process_button = gr.Button("处理图像")
    process_button.click(fn=apply_opencv_methods, inputs=[
        input_image, method,
        mean, sigma,
        prob, thres,
        binary_thres1, binary_thres2,
        conversion_type,
        detection_type, filter_type, size_type, gauss_x, gauss_y,
        transformation_type, scale_h, scale_w, scale_method, translation_x, translation_y, rotate_w, rotate_h, rotate_angle, flip_type, p1, p2, p3, p4, p5, p6, p7, p8, a1, a2, a3, a4, a5, a6,
        enhancement_type, airspace_sharpening_method, frequency_sharpening_method,
        line_detection_method,
        operation_type, structuring_element_type, size_type2,
        x1y1, x1y2, x1y3, x2y1, x2y2, x2y3, x3y1, x3y2, x3y3
    ], outputs=output_image)


iface.launch(share=True)