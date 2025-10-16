import numpy as np
import time
import cv2
from utils import save, show, load_image


def padding(img, pad_h, pad_w):
    H_img, W_img = img.shape
    padded_img = np.zeros((H_img + 2*pad_h, W_img + 2*pad_w), dtype=img.dtype)
    padded_img[pad_h:pad_h + H_img, pad_w:pad_w + W_img] = img

    # 가장자리 복제 방식으로 패딩
    padded_img[:pad_h, pad_w:pad_w + W_img] = np.repeat(img[0:1, :], pad_h, axis=0)
    padded_img[pad_h + H_img:, pad_w:pad_w + W_img] = np.repeat(img[-1:, :], pad_h, axis=0)
    padded_img[:, :pad_w] = np.repeat(padded_img[:, pad_w:pad_w+1], pad_w, axis=1)
    padded_img[:,pad_w + W_img:] = np.repeat(padded_img[:, pad_w+W_img-1:pad_w+W_img], pad_w, axis=1)

    return padded_img


def cross_correlation_2d(img, kernel):
    H_img, W_img = img.shape
    H_kernel, W_kernel = kernel.shape
    pad_h, pad_w = H_kernel // 2, W_kernel // 2

    padded_img = padding(img, pad_h, pad_w)
    
    # 출력 이미지 초기화
    filtered_img = np.zeros_like(img, dtype=np.float64)
    
    for i in range(H_img):
        for j in range(W_img):
            patch = padded_img[i:i + H_kernel, j:j + W_kernel]
            filtered_img[i, j] = np.sum(patch * kernel)
    
    return filtered_img


def cross_correlation_1d(img, kernel):
    kernel = np.asarray(kernel)
    if kernel.ndim == 1:
        kernel = kernel.reshape(1, -1)  # 기본은 horizontal
    # 2D kernel인 경우 shape을 확인하여 vertical/horizontal 구분
    # (N, 1) shape는 vertical, (1, N) shape는 horizontal
    return cross_correlation_2d(img, kernel)


def get_gaussian_filter_2d(size, sigma):
    k = size // 2
    # 2D 좌표 생성
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    # 가우시안 커널 계산
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)

    return kernel
    
    
def get_gaussian_filter_1d(size, sigma):
    k = size // 2
    # 1D 좌표 생성
    x = np.arange(-k, k+1)
    # 가우시안 커널 계산
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)

    return kernel


def two_d_filtering_process(img, size, sigma, caption=False):
    # 2D 가우시안 커널 생성
    gaussian_kernel = get_gaussian_filter_2d(size, sigma)
            
    # 2D cross-correlation 수행
    filtered_img_float = cross_correlation_2d(img, gaussian_kernel)
            
    # 정규화 및 uint8 변환
    normalized_img = cv2.normalize(filtered_img_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 캡션 추가(선택)
    if caption:
        caption = f"{size}x{size} s={sigma}"
        img_with_caption = add_caption(normalized_img, caption)
        return img_with_caption

    return normalized_img


def one_d_filtering_process(img, size, sigma):
    # 1D 가우시안 커널 생성
    k_1d = get_gaussian_filter_1d(size, sigma)
    k_1d_horizontal = k_1d.reshape(1, -1)  # 1xN
    k_1d_vertical = k_1d.reshape(-1, 1)    # Nx1

    # horizontal 1D cross-correlation 수행
    img_only_horizontal = cross_correlation_2d(img, k_1d_horizontal)

    # vertical 1D cross-correlation 수행
    filtered_img_float = cross_correlation_2d(img_only_horizontal, k_1d_vertical)

    # 정규화 및 uint8 변환
    normalized_img = cv2.normalize(filtered_img_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return normalized_img


def add_caption(img, text):
    # 캡션 관련
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 0)
    text_position = (20, 40)

    cv2.putText(img, text, text_position, font, font_scale, text_color, font_thickness)
    return img


def create_tiled_image(images, name, save_file_name):
    # 3x3 타일 이미지 생성
    rows = []
    for i in range(0, 9, 3):
        row_images = images[i:i + 3]
        row_combined = cv2.hconcat(row_images)
        rows.append(row_combined)

    # 모든 행을 수직으로 합치기
    final_combined_img = cv2.vconcat(rows)

    # 최종 이미지 표시 및 저장
    show(name, final_combined_img)
    save(final_combined_img, save_file_name)
    


def compare_images(kernel2d_img, kernel1d_img, image_name="comparison"):
    # 절댓값 차이 계산
    diff = np.abs(kernel2d_img.astype(np.float64) - kernel1d_img.astype(np.float64))

    # 절댓값 차이의 합 계산
    sum_absolute_diff = np.sum(diff)
    
    # 콘솔에 결과 출력
    print(f"\n=== {image_name.upper()}.png 이미지 비교 ===")
    print(f"Maximum difference: {np.max(diff):.6f}")
    print(f"Mean difference: {np.mean(diff):.6f}")
    print(f"Sum of absolute intensity differences: {sum_absolute_diff:.6f}\n")
    
    # 차이 맵을 0-255 범위로 정규화하여 시각화
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 차이 맵 시각화
    show(f'{image_name} - 2D Filtered', kernel2d_img)
    show(f'{image_name} - 1D Sequential Filtered', kernel1d_img)
    show(f'{image_name} - Pixel-wise Difference Map', diff_normalized)


def main():
    lenna, shapes = load_image()
    
    
    # 1-2 : Gaussian 필터링
    print("get_gaussian_filter_1d(5,1):\n", get_gaussian_filter_1d(5, 1), "\n")
    print("get_gaussian_filter_2d(5,1):\n", get_gaussian_filter_2d(5, 1), "\n")

    sizes = [5, 11, 17]
    sigmas = [1, 6, 11]
    
    lenna_filtered_imgs = []
    shapes_filtered_imgs = []
    
    for size in sizes:
        for sigma in sigmas:
            lenna_filtered = two_d_filtering_process(lenna, size, sigma, caption=True)
            shapes_filtered = two_d_filtering_process(shapes, size, sigma, caption=True)

            lenna_filtered_imgs.append(lenna_filtered)
            shapes_filtered_imgs.append(shapes_filtered)

    # 결과 이미지 타일 생성 및 저장
    for img, name in zip([lenna_filtered_imgs, shapes_filtered_imgs], ['lenna', 'shapes']):
        create_tiled_image(img, f"3x3 {name.capitalize()}", f"part_1_gaussian_filtered_{name}.png")
    
    # 2D vs 1D 필터링 비교
    # 17x17 s=6 case 선택
    size = 17
    sigma = 6
    imgs = {}
    print(f"\n=== {size}x{size} s={sigma} case 선택 ===")
    
    # 2D 가우시안 필터링
    for img, name in zip([lenna, shapes], ['lenna', 'shapes']):
        start_time = time.time()
        filtered_img = two_d_filtering_process(img, size, sigma, caption=False)
        end_time = time.time()
        print(f"{name.capitalize()} 이미지의 2D 가우시안 필터링에 걸린 시간: {end_time - start_time:.4f} seconds")
        imgs[f"{name}_2Dgaussian"] = filtered_img
        
    # 1D 가우시안 순차 필터링
    for img, name in zip([lenna, shapes], ['lenna', 'shapes']):
        start_time = time.time()
        filtered_img = one_d_filtering_process(img, size, sigma)
        end_time = time.time()
        print(f"{name.capitalize()} 이미지의 1D 순차 필터링에 걸린 시간: {end_time - start_time:.4f} seconds")
        imgs[f"{name}_1Dgaussian"] = filtered_img
        
    # 이미지 비교 및 pixel-wise difference map 시각화
    compare_images(imgs['lenna_2Dgaussian'], imgs['lenna_1Dgaussian'], image_name="LENNA")
    compare_images(imgs['shapes_2Dgaussian'], imgs['shapes_1Dgaussian'], image_name="SHAPES")
    

if __name__ == "__main__":
    main()