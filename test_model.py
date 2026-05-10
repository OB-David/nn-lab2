import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

#model = nn.models.Model_MLP()
model = nn.models.Model_CNN()
model.load_model(r'.\saved_models\CNN_MSGD.pickle')


test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

trained_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
trained_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()
logits = model(test_imgs)

print(f"Test Accuracy: {nn.metric.accuracy(logits, test_labs)}")

preds = np.argmax(logits, axis=1)       # 预测类别标签



#-----------------------------------------Visualization-----------------------------------------#
# 可视化代码在draw_tools文件夹中


'''绘制混淆矩阵，记得改变保存路径'''
from draw_tools.confusion import plot_confusion_matrix
plot_confusion_matrix(
    y_true=test_labs,
    y_pred=preds,
    classes=[str(i) for i in range(10)], 
    title='Confusion Matrix - CNN on MNIST',
    normalize=False,                      
    save_path=r'.\figs\confusion_CNN.png',
    zero_diag= True                 # 将对角线置零，突出错误分类
)

'''绘制权重热力图，记得改变保存路径'''
from draw_tools.heatmap import plot_weights_heatmap
plot_weights_heatmap(model, figsize_per_layer=(12, 8), save_dir=r'.\figs\heatmap_CNN')


'''绘制top-k错误分类样本，记得改变保存路径'''
from draw_tools.misclassified import plot_top_loss_misclassified
mis_indices, mis_losses = plot_top_loss_misclassified(
    logits=logits,
    y_true=test_labs,
    images=test_imgs,          # 形状 (10000, 784)
    img_shape=(28, 28),
    top_k=12,
    save_path=r'.\figs\top12_misclassified_loss_CNN.png',
    show=True,
    title_prefix="Misclassified Samples - CNN on MNIST"
)

