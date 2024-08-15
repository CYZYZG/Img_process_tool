import cv2
import numpy as np

def read_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def divide_image(image, block_size):
    h, w = image.shape[:2]
    blocks = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            blocks.append((block, (x, y)))
    return blocks

def match_block(block, search_image, search_range, block_pos, step=1):
    bh, bw = block.shape[:2]
    x_start, y_start = block_pos
    min_diff = float('inf')
    best_match_pos = (0, 0)

    for y in range(max(0, y_start - search_range), min(search_image.shape[0] - bh, y_start + search_range + 1), step):
        for x in range(max(0, x_start - search_range), min(search_image.shape[1] - bw, x_start + search_range + 1), step):
            search_block = search_image[y:y+bh, x:x+bw]
            diff = np.sum(np.abs(block - search_block))
            if diff < min_diff:
                min_diff = diff
                best_match_pos = (x, y)

    return best_match_pos, min_diff

def mouse_callback(event, x, y, flags, param):
    global block_pos, selected_block
    if event == cv2.EVENT_LBUTTONDOWN:
        block_pos = (x, y)
        x = (x // block_size) * block_size
        y = (y // block_size) * block_size
        selected_block = (x, y, x + block_size, y + block_size)

# 读取图像
image1 = read_image('00095.png')
image2 = read_image('00096.png')

# 设置块大小和搜索范围
block_size = 8
search_range = 100

# 将第一张图像分块
blocks = divide_image(image1, block_size)

# 初始化块位置
block_pos = None
selected_block = None

# 显示第一张图像并设置鼠标回调
cv2.namedWindow('Image1')
cv2.setMouseCallback('Image1', mouse_callback)

while True:
    display_image1 = image1.copy()
    
    if selected_block is not None:
        cv2.rectangle(display_image1, (selected_block[0], selected_block[1]), (selected_block[2], selected_block[3]), (0, 255, 0), 1)

    cv2.imshow('Image1', display_image1)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按下ESC键退出
        break
    if block_pos is not None:
        # 找到鼠标点击位置所在的块
        x, y = block_pos
        x = (x // block_size) * block_size
        y = (y // block_size) * block_size
        block = image1[y:y+block_size, x:x+block_size]

        # 在第二张图像上搜索最匹配的块
        best_match_pos, min_diff = match_block(block, image2, search_range, (x, y))

        # 显示匹配结果
        print(f"Best match position: {best_match_pos}, Minimum difference: {min_diff}")

        # 在第二张图像上显示匹配块
        matched_image = image2.copy()
        cv2.rectangle(matched_image, best_match_pos, (best_match_pos[0] + block_size, best_match_pos[1] + block_size), (0, 255, 0), 1)
        cv2.imshow('Image2', matched_image)

        block_pos = None

cv2.destroyAllWindows()
