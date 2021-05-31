import cv2
import seaborn 
import pandas 

# Tạo histogram từ ảnh
def createHistogramData(image):
    assert len(image.shape) == 2
    histogram = [0] * 256
    for row in range(image.shape[0]): 
        for col in range(image.shape[1]): 
            histogram[img[row, col]] += 1
    return histogram

# Vẽ biểu đồ histogram
def drawHistogram(histogram, output):
    hist_data = pandas.DataFrame({'intensity': list(range(256)), 'frequency': histogram})
    sns_hist = seaborn.barplot(x='intensity', y='frequency', data=hist_data, color='green')
    sns_hist.set(xticks=[]) 
    
    fig = sns_hist.get_figure()
    fig.savefig(output)
    return output

# Cân bằng sáng
def equalizeHistogram(img, histogram):
    H_ = [0] * 256 # H' as H_
    for i in range(0, len(H_)):
        H_[i] = sum(histogram[:i])
    H_ = H_[1:]

    # Cân bằng H' trong khoảng 0->255
    max_value = max(H_)
    min_value = min(H_)
    H_ = [int(((h-min_value)/(max_value-min_value))*255) for h in H_]

    # Thay H' vào ảnh cũ
    for row in range(img.shape[0]):
        for col in range(img.shape[1]): 
            img[row, col] = H_[img[row, col]]
    return img

# Đọc ảnh ở dạng greyscale
img = cv2.imread('input.jpg',cv2.IMREAD_GRAYSCALE)
# Ghi lại ảnh dạng greyscale
cv2.imwrite('greyscale.jpg', img)
# Tạo histogram với ảnh chưa cân bằng sáng
histogram = createHistogramData(img)
# Vẽ histogram
drawHistogram(histogram, output='before_equalize.png')
# Cân bằng sáng
equalized_img = equalizeHistogram(img, histogram)
# Ghi lại ảnh sau khi cân bằng sáng
cv2.imwrite('output.jpg', equalized_img)
# Tạo histogram với ảnh đã cân bằng sáng
H_istogram = createHistogramData(equalized_img)
# Trực quan hóa histogram
drawHistogram(H_istogram, output='after_equalize.png')


