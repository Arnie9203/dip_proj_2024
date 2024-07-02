import cv2
import numpy as np
import gradio as gr
from scipy import sparse
from scipy.sparse.linalg import spsolve
import dlib
import math
import random
from scipy import ndimage
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
import gradio as gr
import subprocess

# 定义各种图像处理函数
def soften_image(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def add_haze(img, haze_factor):
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    return cv2.addWeighted(img, 1 - haze_factor, blurred, haze_factor, 0)

def apply_autumn_filter(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 增加黄色和橙色的饱和度和亮度
    lower_yellow = np.array([20, 50, 50], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    img_hsv[mask_yellow > 0, 1] = np.clip(img_hsv[mask_yellow > 0, 1] * 1.2, 0, 255)
    img_hsv[mask_yellow > 0, 2] = np.clip(img_hsv[mask_yellow > 0, 2] * 1.2, 0, 255)
    
    lower_orange = np.array([10, 50, 50], dtype=np.uint8)
    upper_orange = np.array([20, 255, 255], dtype=np.uint8)
    mask_orange = cv2.inRange(img_hsv, lower_orange, upper_orange)
    img_hsv[mask_orange > 0, 1] = np.clip(img_hsv[mask_orange > 0, 1] * 1.2, 0, 255)
    img_hsv[mask_orange > 0, 2] = np.clip(img_hsv[mask_orange > 0, 2] * 1.2, 0, 255)
    
    # 调整其他颜色的色调和饱和度
    img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0] * 0.9, 0, 255)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 1.1, 0, 255)
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * 1.1, 0, 255)
    
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_rgb

def film_filter(img):
    img[:, :, 0] = np.clip(img[:, :, 0] * 0.5, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1] * 1.1, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(img[:, :, 2] * 1.3, 0, 255).astype(np.uint8)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 0.7, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

def nostalgic_filter(img):
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    newB = 0.272 * R + 0.534 * G + 0.131 * B
    newG = 0.349 * R + 0.686 * G + 0.168 * B
    newR = 0.393 * R + 0.769 * G + 0.189 * B
    newB = np.clip(newB, 0, 255).astype(np.uint8)
    newG = np.clip(newG, 0, 255).astype(np.uint8)
    newR = np.clip(newR, 0, 255).astype(np.uint8)
    return cv2.merge([newB, newG, newR])

def sketch_filter(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray_image = 255 - gray_image
    blurred_inverted_gray_image = cv2.GaussianBlur(inverted_gray_image, (19, 19), 0)
    inverted_blurred_image = 255 - blurred_inverted_gray_image
    # 将灰度图像与反转模糊图像相除，得到素描效果
    sketch = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)  
    # 使用拉普拉斯算子增强边缘
    laplacian = cv2.Laplacian(sketch, cv2.CV_8U, ksize=5)
    sketch = cv2.addWeighted(sketch, 1.5, laplacian, -0.5, 0)
    return sketch

def black_white_filter(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = np.uint8(np.clip((1.2 * img_gray + 0), 0, 255))
    return img_gray

def relief_filter(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    img_relief = np.zeros((h, w), dtype=img_gray.dtype)
    for i in range(h):
        for j in range(w-1):
            a = img_gray[i, j]
            b = img_gray[i, j+1]
            img_relief[i, j] = min(max((int(a) - int(b) + 127), 0), 255)
    return img_relief

def watercolour_filter(img):
    return cv2.stylization(img, sigma_s=55, sigma_r=0.55)

def blur_filter(img):
    return cv2.blur(img, (7, 7))

# 原始图像处理函数
def original_image(input_image):
    return input_image

def apply_opencv_methods1(input_image, method, kernel_size=1, haze_factor=0.0):
    if method == "原图":
        return original_image(input_image)
    elif method == "柔化图像":
        return soften_image(input_image, kernel_size)
    elif method == "秋天滤镜":
        return apply_autumn_filter(input_image)
    elif method == "胶片滤镜":
        return film_filter(input_image)
    elif method == "怀旧滤镜":
        return nostalgic_filter(input_image)
    elif method == "素描滤镜":
        return sketch_filter(input_image)
    elif method == "黑白滤镜":
        return black_white_filter(input_image)
    elif method == "浮雕滤镜":
        return relief_filter(input_image)
    elif method == "水彩滤镜":
        return watercolour_filter(input_image)
    elif method == "模糊滤镜":
        return blur_filter(input_image)
    elif method == "光晕滤镜":
        return add_haze(input_image, haze_factor)
    else:
        raise ValueError("Invalid method")

# 更新参数的可见性
def update_parameters(method):
    default_visibility = gr.update(visible=False)
    params = {
        "原图": [default_visibility] * 3,
        "柔化图像": [gr.update(visible=True)] + [default_visibility] * 2,
        "秋天滤镜": [default_visibility] * 3,
        "胶片滤镜": [default_visibility] * 3,
        "怀旧滤镜": [default_visibility] * 3,
        "素描滤镜": [default_visibility] * 3,
        "黑白滤镜": [default_visibility] * 3,
        "浮雕滤镜": [default_visibility] * 3,
        "水彩滤镜": [default_visibility] * 3,
        "模糊滤镜": [default_visibility] * 3,
        "光晕滤镜": [default_visibility] * 1 + [gr.update(visible=True)]
    }
    return params.get(method, [default_visibility] * 3)

def run_neural_style(style_image, content_image, content_weight, style_weight, num_iterations, output_image):
    # 默认参数值
    default_image_size = 512
    default_gpu = 0
    default_init = "random"
    default_optimizer = "lbfgs"
    default_learning_rate = 1e0

    # 保存上传的图片
    style_image_path = "style_image.jpg"
    content_image_path = "content_image.jpg"
    style_image.save(style_image_path)
    content_image.save(content_image_path)
    
    # 构建命令行参数
    command = [
        "python", "./neural-style-pt/neural_style.py",
        "-style_image", style_image_path,
        "-content_image", content_image_path,
        "-image_size", str(default_image_size),
        "-gpu", str(default_gpu),
        "-content_weight", str(content_weight),
        "-style_weight", str(style_weight),
        "-num_iterations", str(num_iterations),
        "-init", default_init,
        "-optimizer", default_optimizer,
        "-learning_rate", str(default_learning_rate),
        "-model_file", "./neural-style-pt/models/vgg19-d01eb7cb.pth",
        "-output_image", output_image
    ]
    
    # 运行命令
    subprocess.run(command)
    
    # 返回生成的图像
    return output_image





# 原始图像处理函数
def original_image(input_image):
    return input_image


class Eye:
    def __init__(self, img, a):
        self.img = img
        self.a = a

    def landmark_dec_dlib_fun(self, src):
        img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        land_marks = []
        predictor_path = './Face/model/shape_predictor_68_face_landmarks.dat'

        # 使用dlib自带的frontal_face_detector作为我们的特征提取器
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        rects = detector(img_gray, 0)

        for rect in rects:
            land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rect).parts()])
            land_marks.append(land_marks_node)

        return land_marks

    '''
    方法： Interactive Image Warping 局部平移算法
    '''

    def localTranslationWarp(self, srcImg, startX, startY, endX, endY, radius, a):
        ddradius = float(radius * radius)
        copyImg = srcImg.copy()

        H, W, C = srcImg.shape
        for i in range(W):
            for j in range(H):
                if abs(i - startX) > radius and abs(j - startY) > radius:
                    continue

                distance = (i - startX) ** 2 + (j - startY) ** 2

                if distance < ddradius:
                    rnorm = math.sqrt(distance) / radius
                    ratio = 1 - (rnorm - 1) ** 2 * a

                    UX = startX + ratio * (i - startX)
                    UY = startY + ratio * (j - startY)

                    value = self.BilinearInsert(srcImg, UX, UY)
                    copyImg[j, i] = value

        return copyImg

    def BilinearInsert(self, src, ux, uy):
        H, W, C = src.shape
        if ux < 0 or ux >= W - 1 or uy < 0 or uy >= H - 1:
            return np.array([0, 0, 0], dtype=np.uint8)

        x1 = int(ux)
        x2 = x1 + 1
        y1 = int(uy)
        y2 = y1 + 1

        part1 = src[y1, x1].astype(np.float64) * (x2 - ux) * (y2 - uy)
        part2 = src[y1, x2].astype(np.float64) * (ux - x1) * (y2 - uy)
        part3 = src[y2, x1].astype(np.float64) * (x2 - ux) * (uy - y1)
        part4 = src[y2, x2].astype(np.float64) * (ux - x1) * (uy - y1)

        insertValue = part1 + part2 + part3 + part4

        return insertValue.astype(np.uint8)

    def face_big_auto(self, src):
        landmarks = self.landmark_dec_dlib_fun(src)

        if len(landmarks) == 0:
            return src

        for landmarks_node in landmarks:
            left_landmark = landmarks_node[36]
            left_landmark_down = landmarks_node[27]

            right_landmark = landmarks_node[45]
            right_landmark_down = landmarks_node[27]

            endPt = landmarks_node[30]

            r_left = np.linalg.norm(left_landmark - left_landmark_down)
            r_right = np.linalg.norm(right_landmark - right_landmark_down)

            big_image = self.localTranslationWarp(src, left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0], endPt[0, 1], r_left, self.a)
            big_image = self.localTranslationWarp(big_image, right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0], endPt[0, 1], r_right, self.a)

            return big_image

    def eyeBig(self):
        big_image = self.face_big_auto(self.img)
        return big_image

def handle_eye(img, a):
    eye_big = Eye(img, a)
    return eye_big.eyeBig()






class face:
    def __init__(self, img, factor=1.0):
        self.img = img
        self.factor = factor  # 瘦脸程度因子

    def landmark_dec_dlib_fun(self, src):
        img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        land_marks = []
        predictor_path = './Face/model/shape_predictor_68_face_landmarks.dat'

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        rects = detector(img_gray, 0)

        for rect in rects:
            land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rect).parts()])
            land_marks.append(land_marks_node)

        return land_marks

    def localTranslationWarp(self, srcImg, startX, startY, endX, endY, radius):
        ddradius = float(radius * radius)
        copyImg = srcImg.copy()

        s = 120
        ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
        H, W, C = srcImg.shape
        for i in range(W):
            for j in range(H):
                if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                    continue

                distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)

                if distance < ddradius:
                    ratio = (ddradius - distance) / (ddradius - distance + (100 / s) * ddmc)
                    ratio = ratio * ratio * self.factor  # 乘以瘦脸程度因子

                    UX = i - ratio * (endX - startX)
                    UY = j - ratio * (endY - startY)

                    value = self.BilinearInsert(srcImg, UX, UY)
                    copyImg[j, i] = value

        return copyImg

    def BilinearInsert(self, src, ux, uy):
        H, W, C = src.shape

        if ux < 0 or ux >= W - 1 or uy < 0 or uy >= H - 1:
            return np.array([0, 0, 0], dtype=np.uint8)

        x1 = int(ux)
        x2 = min(x1 + 1, W - 1)
        y1 = int(uy)
        y2 = min(y1 + 1, H - 1)

        part1 = src[y1, x1].astype(np.float64) * (x2 - ux) * (y2 - uy)
        part2 = src[y1, x2].astype(np.float64) * (ux - x1) * (y2 - uy)
        part3 = src[y2, x1].astype(np.float64) * (x2 - ux) * (uy - y1)
        part4 = src[y2, x2].astype(np.float64) * (ux - x1) * (uy - y1)

        insertValue = part1 + part2 + part3 + part4

        return insertValue.astype(np.uint8)

    def face_thin_auto(self, src):
        landmarks = self.landmark_dec_dlib_fun(src)

        if len(landmarks) == 0:
            return src

        for landmarks_node in landmarks:
            left_landmark = landmarks_node[3]
            left_landmark_down = landmarks_node[5]

            right_landmark = landmarks_node[13]
            right_landmark_down = landmarks_node[15]

            endPt = landmarks_node[30]

            r_left = np.linalg.norm(left_landmark - left_landmark_down)
            r_right = np.linalg.norm(right_landmark - right_landmark_down)

            thin_image = self.localTranslationWarp(src, left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0], endPt[0, 1], r_left)
            thin_image = self.localTranslationWarp(thin_image, right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0], endPt[0, 1], r_right)
            return thin_image

    def faceThin(self):
        thin_image = self.face_thin_auto(self.img)
        return thin_image

def handle_thin_face(img, factor=1.0):
    face_thin = face(img, factor)
    return face_thin.faceThin()





dlib_path = './Face/model/shape_predictor_81_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)

jaw_point = list(range(0, 17)) + list(range(68, 81))
left_eye = list(range(42, 48))
right_eye = list(range(36, 42))
left_brow = list(range(22, 27))
right_brow = list(range(17, 22))
mouth = list(range(48, 61))
nose = list(range(27, 35))

align = left_brow + right_eye + left_eye + right_brow + nose + mouth

def get_landmark(img):
    faces = detector(img, 1)
    if len(faces) == 0:
        raise ValueError("No face detected")
    shape = predictor(img, faces[0]).parts()
    return np.matrix([[p.x, p.y] for p in shape])

def draw_convex_hull(img, points, color):
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(img, hull, color=color)

def get_skin_mask(img):
    landmarks = get_landmark(img)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    draw_convex_hull(mask, landmarks[jaw_point], color=1)
    for index in [mouth, left_eye, right_eye, left_brow, right_brow, nose]:
        draw_convex_hull(mask, landmarks[index], color=0)
    mask = np.array([mask] * 3).transpose(1, 2, 0)
    return mask

def buff(img, img_skin, value1, value2):
    dx = value1 * 5
    fc = value1 * 12.5
    p = 80
    temp1 = cv2.bilateralFilter(img, dx, fc, fc)
    temp2 = (temp1 - img + 128)
    temp3 = cv2.GaussianBlur(temp2, (2 * value2 - 1, 2 * value2 - 1), 0, 0)
    temp4 = img + 2 * temp3 - 255
    dst = np.uint8(img * ((100 - p) / 100) + temp4 * (p / 100))
    img_skin_c = np.uint8(-(img_skin - 1))
    dst = np.uint8(dst * img_skin + img * img_skin_c)
    return dst

def whitening(img, img_skin, value):
    midtones_add = np.zeros(256)
    for i in range(256):
        midtones_add[i] = 0.667 * (1 - ((i - 127) / 127) * ((i - 127) / 127))
    lookup = np.zeros(256, dtype='uint8')
    for i in range(256):
        red = i
        red += np.uint8(value * midtones_add[red])
        red = max(0, red)
        lookup[i] = np.uint8(red)

    w, h, c = img.shape
    for i in range(w):
        for j in range(h):
            if img_skin[i, j, 0] == 1:
                img[i, j, 0] = lookup[img[i, j, 0]]
                img[i, j, 1] = lookup[img[i, j, 1]]
                img[i, j, 2] = lookup[img[i, j, 2]]
    return img

def handle_whitening(img, value):
    skin_mask = get_skin_mask(img)
    return whitening(img, skin_mask, value)

def handle_buff(img, value1, value2):
    if(value2 <= 0):
        return img
    skin_mask = get_skin_mask(img)
    return buff(img, skin_mask, value1, value2)

def apply_opencv_methods(input_image, method, a, thin_face_factor, whitening_face_factor, buff_factor_1, buff_factor_2):
    if method == "原图":
        return original_image(input_image)
    elif method == "大眼":
        return handle_eye(input_image, a)
    elif method == "瘦脸":
        return handle_thin_face(input_image, thin_face_factor)
    elif method == "美白":
        return handle_whitening(input_image, whitening_face_factor)
    elif method == "磨皮":
        return handle_buff(input_image, buff_factor_1, buff_factor_2)


# 更新参数的可见性
def update_parameters(method):
    default_visibility = gr.update(visible=False)
    params = {
        "原图": [default_visibility] * 5,
        "大眼": [gr.update(visible=True)] + [default_visibility] * 4,
        "瘦脸": [default_visibility] * 1 + [gr.update(visible=True)] + [default_visibility] * 3,
        "美白": [default_visibility] * 2 + [gr.update(visible=True)] + [default_visibility] * 2,
        "磨皮": [default_visibility] * 3 + [gr.update(visible=True), gr.update(visible=True)]
    }
    return params.get(method, [default_visibility] * 43)

def fusion(src, mask, target, pos_x, pos_y, keep_texture=True):
    src = src / 255.
    mask = mask / 255.
    target = target / 255.
    
    assert src.shape == mask.shape
    n, m = src.shape[0], src.shape[1]

    id = {}
    edge = {}
    points = []
    cnt = 0
    edge_cnt = 0
    for i in range(n):
        for j in range(m):
            if mask[i, j, 0] > .5:
                id[(i, j)] = cnt
                points.append((i, j))
                cnt += 1
                if i == 0 or j == 0 or i == n - 1 or j == m - 1:
                    edge[(i, j)] = 1
                    edge_cnt += 1
                elif mask[i - 1, j, 0] < .5 or mask[i, j - 1, 0] < .5 or mask[i + 1, j, 0] < .5 or mask[i, j + 1, 0] < .5:
                    edge[(i, j)] = 1
                    edge_cnt += 1
                else:
                    edge[(i, j)] = 0

    A = sparse.lil_matrix((cnt, cnt), dtype=float)
    b = [np.zeros(shape=(cnt)), np.zeros(shape=(cnt)), np.zeros(shape=(cnt))]
    X = [np.zeros(shape=(cnt)), np.zeros(shape=(cnt)), np.zeros(shape=(cnt))]
    for (x, y) in points:
        k = id[(x, y)]
        if edge[(x, y)] == 1:
            A[k, k] = 1
            for channel in range(3):
                b[channel][k] = target[pos_x + x, pos_y + y, channel]
        else:
            A[k, k] = 4
            A[k, id[(x - 1, y)]] = -1
            A[k, id[(x, y - 1)]] = -1
            A[k, id[(x + 1, y)]] = -1
            A[k, id[(x, y + 1)]] = -1
            grad_src = 0
            grad_target = 0
            for channel in range(3):
                grad_src += (src[x + 1, y, channel] - src[x, y, channel]) ** 2
                grad_src += (src[x, y + 1, channel] - src[x, y, channel]) ** 2
                grad_target += abs(target[pos_x + x + 1, pos_y + y, channel] - target[pos_x + x, pos_y + y, channel])
                grad_target += abs(target[pos_x + x, pos_y + y + 1, channel] - target[pos_x + x, pos_y + y, channel])
            if not keep_texture:
                grad_target = 0
            delta = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for (dx, dy) in delta:
                dsrc = src[x, y] - src[x + dx, y + dy]
                grad_src = np.dot(dsrc, dsrc)
                dtar = target[pos_x + x, pos_y + y] - target[pos_x + x + dx, pos_y + y + dy]
                grad_target = np.dot(dtar, dtar)
                if not keep_texture or grad_src >= grad_target:
                    for channel in range(3):
                        b[channel][k] += dsrc[channel]
                else:
                    for channel in range(3):
                        b[channel][k] += dtar[channel]

    A = A.tocsc()
    for channel in range(3):
        X[channel] = spsolve(A, b[channel])
    for (x, y) in points:
        k = id[(x, y)]
        for channel in range(3):
            target[pos_x + x, pos_y + y, channel] = X[channel][k]
    target = (np.clip(target, 0., 1.) * 255).astype(np.uint8)
    return target

def adjust_image(input_img, exposure, brightness, contrast, saturation, temperature, hue, highlights, shadows, sharpness, denoise):
    img = input_img.astype(np.float32) / 255.0

    img = np.clip(img * exposure + brightness / 100.0, 0, 1)
    img = np.clip((img - 0.5) * contrast + 0.5, 0, 1)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * saturation, 0, 1)
    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    if temperature != 0:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img)
        b = b + temperature
        img = cv2.merge([l, a, b])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue) % 180
    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    img = np.clip(img + highlights / 100.0 - shadows / 100.0, 0, 1)

    if sharpness != 0:
        kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)

    img = cv2.fastNlMeansDenoisingColored((img * 255).astype(np.uint8), None, h=denoise, templateWindowSize=7, searchWindowSize=21)
    img = img.astype(np.float32) / 255.0

    return np.clip(img * 255, 0, 255).astype(np.uint8)

with gr.Blocks() as demo:

    with gr.Tab("图片基础调节功能"):
        with gr.Row():
            input_img = gr.Image(label="Input Image")
            output_img = gr.Image(label="Output Image")
        
        with gr.Row():
            with gr.Column():
                exposure = gr.Slider(minimum=0.1, maximum=3.0, value=1.0, label="曝光度")
                brightness = gr.Slider(minimum=-100.0, maximum=100.0, value=0.0, label="亮度")
            with gr.Column():
                contrast = gr.Slider(minimum=0.1, maximum=3.0, value=1.0, label="对比度")
                saturation = gr.Slider(minimum=0.1, maximum=3.0, value=1.0, label="饱和度")
        
        with gr.Row():
            with gr.Column():
                temperature = gr.Slider(minimum=-100.0, maximum=100.0, value=0.0, label="色温")
                hue = gr.Slider(minimum=-180.0, maximum=180.0, value=0.0, label="色相")
            with gr.Column():
                highlights = gr.Slider(minimum=-100.0, maximum=100.0, value=0.0, label="强光")
                shadows = gr.Slider(minimum=-100.0, maximum=100.0, value=0.0, label="阴影")
        
        with gr.Row():
            with gr.Column():
                sharpness = gr.Slider(minimum=-10.0, maximum=10.0, value=0.0, label="锐化")
                denoise = gr.Slider(minimum=0.0, maximum=100.0, value=0.0, label="降噪")
            with gr.Column():
                pass
        
        btn = gr.Button("生成图像")
        btn.click(adjust_image, inputs=[input_img, exposure, brightness, contrast, saturation, temperature, hue, highlights, shadows, sharpness, denoise], outputs=output_img)


    with gr.Tab("滤镜功能"):
        with gr.Row():
            input_image = gr.Image(type="numpy")
            output_image = gr.Image(type="numpy")
        with gr.Row():
            method = gr.Radio(
                choices=["原图",
                         "柔化图像",
                         "秋天滤镜",
                         "胶片滤镜",
                         "怀旧滤镜",
                         "素描滤镜",
                         "黑白滤镜",
                         "浮雕滤镜",
                         "水彩滤镜",
                         "模糊滤镜",
                         "光晕滤镜"], 
                value="原图")
        with gr.Row():
            kernel_size = gr.Number(value=1, label="柔化图像: 核大小（仅能为奇数）", visible=False)
            haze_factor = gr.Number(value=0.0, label="光晕: 因子（-10~10）", visible=False)
        
        # 参数更新按钮
        method.change(fn=update_parameters, inputs=method, outputs=[
            kernel_size, haze_factor
        ])
        
        # 图像处理按钮
        process_button = gr.Button("处理图像")
        process_button.click(fn=apply_opencv_methods1, inputs=[
            input_image, method, kernel_size, haze_factor
        ], outputs=output_image)
    
    with gr.Tab("高级人像处理功能"):
        with gr.Row():
            input_image = gr.Image(type="numpy")
            output_image = gr.Image(type="numpy")
        with gr.Row():
            method = gr.Radio(
                choices=["原图","大眼","瘦脸","美白","磨皮"], value="原图")
        with gr.Row():
            a = gr.Number(value=0.3, label="大眼参数(建议设置在-0.5到0.5之间)", visible=False)
            thin_face_factor = gr.Number(value=0.5, label="瘦脸参数(建议设置在-1到1之间)", visible=False)
            whitening_face_factor = gr.Number(value=15, label="美白参数(建议设置在-25到25之间)", visible=False)
            buff_factor_1 = gr.Number(value=5, label="磨皮参数1(双边滤波参数，建议不大于5)", visible=False)
            buff_factor_2 = gr.Number(value=5, label="磨皮参数2(高斯模糊参数，必须是正整数，建议不大于5)", visible=False)

        # 参数更新按钮
        method.change(fn=update_parameters, inputs=method, outputs=[
            a, thin_face_factor, whitening_face_factor, buff_factor_1, buff_factor_2
        ])
        
        # 图像处理按钮
        process_button = gr.Button("处理图像")
        process_button.click(fn=apply_opencv_methods, inputs=[
            input_image, method,
            a,thin_face_factor,whitening_face_factor,buff_factor_1,buff_factor_2
        ], outputs=output_image)


    with gr.Tab("图像融合功能"):
        with gr.Row():
            src_img = gr.Image(label="Source Image")
            mask_img = gr.Image(label="Mask Image")
            target_img = gr.Image(label="Target Image")
            output_img = gr.Image(label="Output Image")
        pos_x = gr.Number(label="Position X", value=0)
        pos_y = gr.Number(label="Position Y", value=0)
        keep_texture = gr.Checkbox(label="保持纹理", value=True)
        btn = gr.Button("图像融合")
        btn.click(fusion, inputs=[src_img, mask_img, target_img, pos_x, pos_y, keep_texture], outputs=output_img)


    with gr.Tab("风格迁移功能"):
        with gr.Row():
            style_image = gr.Image(type="pil", label="风格图像")
            content_image = gr.Image(type="pil", label="内容图像")
            output_image = gr.Image(type="pil", label="生成图像")

        content_weight = gr.Slider(minimum=1e0, maximum=1e2, step=1e-1, value=5e0, label="内容权重")
        style_weight = gr.Slider(minimum=1e0, maximum=1e2, step=1e-1, value=1e2, label="风格权重")
        iterations = gr.Slider(minimum=100, maximum=1000, step=1, value=1000, label="迭代次数")
        output_path = gr.Textbox(value="output.png", label="输出结果")

        run_button = gr.Button("生成图像")
        run_button.click(
            fn=run_neural_style,
            inputs=[style_image, content_image, content_weight, style_weight, iterations, output_path],
            outputs=output_image
        )

demo.launch(share=True)
