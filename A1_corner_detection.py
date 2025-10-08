import numpy as np
import time
import cv2
from A1_image_filtering import get_gaussian_filter_2d, cross_correlation_2d
from utils import save, show, load_image


def compute_corner_response(img):
    # Sobel 필터 정의
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    # 이미지에 Sobel 필터 적용
    derivative_x = cross_correlation_2d(img, sobel_x)
    derivative_y = cross_correlation_2d(img, sobel_y)

    # matrix M 계산
    Ixx = derivative_x ** 2
    Iyy = derivative_y ** 2
    Ixy = derivative_x * derivative_y
    window = np.ones((5, 5))
    
    # 각 성분에 윈도우 함수 적용
    Sxx = cross_correlation_2d(Ixx, window)
    Syy = cross_correlation_2d(Iyy, window)
    Sxy = cross_correlation_2d(Ixy, window)
    
    # matrix M 계산
    M = np.zeros((img.shape[0], img.shape[1], 2, 2))
    M[:, :, 0, 0] = Sxx
    M[:, :, 0, 1] = Sxy
    M[:, :, 1, 0] = Sxy
    M[:, :, 1, 1] = Syy
    
    # R 및 corner response 계산
    eig_val, _ = np.linalg.eig(M)
    R = eig_val[:, :, 0] * eig_val[:, :, 1] - 0.04 * (eig_val[:, :, 0] + eig_val[:, :, 1])**2
    corner = np.where(R > 0, R, 0)
    
    # [0, 1] 정규화
    cmin = np.amin(corner)
    cmax = np.amax(corner)
    
    return (corner - cmin) / (cmax - cmin + 1e-10)
    
    
def thresholding(img, corner):
    # unit8로 변환
    safe = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = cv2.normalize(safe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # BGR 변환 후 코너 표시
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_color[:, :, 0] = np.where(corner > 0.1, 0, img_color[:, :, 0])
    img_color[:, :, 1] = np.where(corner > 0.1, 255, img_color[:, :, 1])
    img_color[:, :, 2] = np.where(corner > 0.1, 0, img_color[:, :, 2])

    return img_color


def non_maximum_suppression_win(R, winSize):
    nms_result = np.zeros_like(R)
    
    # NMS 적용
    for x in range(R.shape[0]):
        for y in range(R.shape[1]):
            # 윈도우 범위
            x_start = max(0, x - winSize // 2)
            x_end = min(R.shape[0], x + winSize // 2 + 1)
            y_start = max(0, y - winSize // 2)
            y_end = min(R.shape[1], y + winSize // 2 + 1)
            window = R[x_start:x_end, y_start:y_end]
            # keep if it's the local maximum (>= to tolerate ties) and above threshold
            if R[x, y] >= np.max(window) and R[x, y] > 0.1:
                nms_result[x, y] = R[x, y]

    return nms_result


def draw_circle(img, corners):
    # unit8로 변환
    safe = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = cv2.normalize(safe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # BGR 변환 후 코너(원) 표시
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for x in range(corners.shape[0]):
        for y in range(corners.shape[1]):
            if corners[x, y] > 0:
                cv2.circle(img_color, (y, x), 5, (0, 255, 0), 2)

    return img_color


def main():
    lenna, shapes = load_image()
    
    
    # 3-1 : Gaussian 필터 적용
    lenna_gaussian = cross_correlation_2d(lenna, get_gaussian_filter_2d(7, 1.5))
    shapes_gaussian = cross_correlation_2d(shapes, get_gaussian_filter_2d(7, 1.5))
    
    
    # 3-2 : corner response 계산
    start_time = time.time()
    corner_response_lenna = compute_corner_response(lenna_gaussian)
    end_time = time.time()
    print(f"Lenna 이미지의 corner response 계산에 걸린 시간: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()
    corner_response_shapes = compute_corner_response(shapes_gaussian)
    end_time = time.time()
    print(f"Shapes 이미지의 corner response 계산에 걸린 시간: {end_time - start_time:.4f} seconds\n")
    
    # corner response 시각화 및 저장
    for img, name in zip([corner_response_lenna, corner_response_shapes], ['lenna', 'shapes']):
        show(f'Corner Response {name.capitalize()}', img)
        save(img, f"part_3_corner_raw_{name}.png")

    
    # 3-3 : Thresholding, Non-maximum suppression 적용
    # Thresholding
    lenna_color = thresholding(lenna, corner_response_lenna)
    shapes_color = thresholding(shapes, corner_response_shapes)
    
    # Thresholding 시각화 및 저장
    for img, name in zip([lenna_color, shapes_color], ['lenna', 'shapes']):
        show(f'Corner Detected {name.capitalize()}', img)
        save(img, f"part_3_corner_bin_{name}.png")
        
    # Non-maximum suppression
    start_time = time.time()
    suppressed_lenna = non_maximum_suppression_win(corner_response_lenna, 11)
    end_time = time.time()
    print(f"Lenna 이미지의 non-maximum suppression에 걸린 시간: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    suppressed_shapes = non_maximum_suppression_win(corner_response_shapes, 11)
    end_time = time.time()
    print(f"Shapes 이미지의 non-maximum suppression에 걸린 시간: {end_time - start_time:.4f} seconds")

    corner_sup_lenna = draw_circle(lenna, suppressed_lenna)
    corner_sup_shapes = draw_circle(shapes, suppressed_shapes)

    # Non-maximum suppression 시각화 및 저장
    for img, name in zip([corner_sup_lenna, corner_sup_shapes], ['lenna', 'shapes']):
        show(f'Non-Maximum Suppression {name.capitalize()}', img)
        save(img, f"part_3_corner_sup_{name}.png")


if __name__ == '__main__':
    main()
    