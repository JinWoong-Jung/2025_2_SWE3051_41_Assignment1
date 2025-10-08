import numpy as np
import time
import cv2
from utils import save, show, load_image
from A1_image_filtering import get_gaussian_filter_2d, cross_correlation_2d


def compute_image_gradient(img):
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

    # direction, magnitude 계산
    dir = np.arctan2(derivative_y, derivative_x)
    mag = np.sqrt(derivative_x**2 + derivative_y**2)

    return mag, dir


def non_maximum_suppression_dir(mag, dir):
    height, width = mag.shape
    nms = mag.copy()
    # 방향을 0-360도로 변환
    dir = np.rad2deg(dir) + 180

    # NMS 수행(가장자리 픽셀 제외)
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            theta = dir[row, col]
            # 방향에 따른 픽셀 값 할당
            # case1. 0도 또는 180도
            if ((0 <= theta <= 22.5) or (337.5 < theta <= 360)) or (157.5 < theta <= 202.5) : 
                if mag[row, col] < mag[row, col+1] or mag[row, col] < mag[row, col-1]:
                    nms[row, col] = 0
            # case2. 45도 또는 225도
            elif (22.5 < theta <= 67.5) or (202.5 < theta <= 247.5): 
                if mag[row, col] < mag[row-1, col-1] or mag[row, col] < mag[row+1, col+1]:
                    nms[row, col] = 0
            # case3. 90도 또는 270도
            elif (67.5 < theta <= 112.5) or (247.5 < theta <= 292.5): 
                if mag[row, col] < mag[row+1, col] or mag[row, col] < mag[row-1, col]:
                    nms[row, col] = 0
            # case4. 135도 또는 315도
            elif (112.5 < theta <= 157.5) or (292.5 < theta <= 337.5): 
                if mag[row, col] < mag[row-1, col+1] or mag[row, col] < mag[row+1, col-1]:
                    nms[row, col] = 0
                    
    # 경계값 처리
    nms = np.where(nms < 0, 0, nms)
    nms = np.where(nms > 255, 255, nms)
    
    return nms


def main():
    lenna, shapes = load_image()
    
    
    # 2-1 : 가우시안 필터 적용
    lenna = cross_correlation_2d(lenna, get_gaussian_filter_2d(7, 1.5))
    shapes = cross_correlation_2d(shapes, get_gaussian_filter_2d(7, 1.5))
    
    
    # 2-2 : Sobel 필터를 이용한 magnitude map 계산
    start_time = time.time()
    lenna_magnitude, lenna_dir = compute_image_gradient(lenna)
    end_time = time.time()
    print(f"Lenna 이미지의 gradient 계산에 걸린 시간: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()
    shapes_magnitude, shapes_dir = compute_image_gradient(shapes)
    end_time = time.time()
    print(f"Shapes 이미지의 gradient 계산에 걸린 시간: {end_time - start_time:.4f} seconds\n")
    
    lenna_magnitude_norm = cv2.normalize(lenna_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    shapes_magnitude_norm = cv2.normalize(shapes_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    show('Lenna Magnitude', lenna_magnitude_norm)
    show('Shapes Magnitude', shapes_magnitude_norm)
    
    # 최종 이미지 파일로 저장
    for img, name in zip([lenna_magnitude_norm, shapes_magnitude_norm], ['lenna', 'shapes']):
        save(img, f"part_2_edge_raw_{name}.png")


    # 2-3 : Non-maximum suppression 적용
    start_time = time.time()
    supressed_lenna = non_maximum_suppression_dir(lenna_magnitude, lenna_dir)
    end_time = time.time()
    print(f"Lenna 이미지의 non-maximum suppression에 걸린 시간: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()
    shapes_nms = non_maximum_suppression_dir(shapes_magnitude, shapes_dir)
    end_time = time.time()
    print(f"Shapes 이미지의 non-maximum suppression에 걸린 시간: {end_time - start_time:.4f} seconds\n")
    
    # Non-maximum suppression 결과 시각화 및 저장
    supressed_lenna = cv2.normalize(supressed_lenna, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    shapes_nms = cv2.normalize(shapes_nms, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    for img, name in zip([supressed_lenna, shapes_nms], ['lenna', 'shapes']):
        show(f'Non-Maximum Suppression {name.capitalize()}', img)
        save(img, f"part_2_edge_sup_{name}.png")


if __name__ == "__main__":
    main()