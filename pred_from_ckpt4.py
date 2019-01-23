import numpy as np
import tensorflow as tf
import yaml
import os
import cv2
import argparse
import gc
from hed_net import HED
# from SIFT_HoughLines import SIFT_HoughLines


# 控制台读参
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, required=False, default='0')
    # parser.add_argument('-img_path', type=str, required=False, default='./output7.0/4.jpg')
    # parser.add_argument('-img_path', type=str, required=True, default=None)
    args = parser.parse_args()
    return args


def sess_config(args=None):
    log_device_placement = True  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto(log_device_placement=log_device_placement,
                            allow_soft_placement=allow_soft_placement,
                            gpu_options=gpu_options)
    return config


def img_pre_process(img, **kwargs):
    def stretch(bands, lower_percent=2, high_percent=98, bits=8):
        if bits not in [8, 16]:
            print('error!dest image must be 8bit or 16bits')
            return
        # 生成模仿原数组结构的0数组
        out = np.zeros_like(bands, dtype=np.float32)
        n = bands.shape[2]
        for i in range(n):
            a = 0
            b = 1
            temp = bands[:, :, i]
            c = np.percentile(bands[:, :, i], lower_percent)
            d = np.percentile(bands[:, :, i], high_percent)
            if d - c == 0:
                out[:, :, i] = 0
                continue
            t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
            out[:, :, i] = np.clip(t, a, b)
        if bits == 8:
            return out.astype(np.float32) * 255
        else:
            return np.uint(out.astype(np.float32) * 65535)

    img = stretch(img)
    img -= kwargs['mean']
    return img


def predict_big_map(img_path, out_shape=(448, 448), inner_shape=(224, 224), out_channel=1, pred_fun=None, **kwargs):
    image = cv2.imread(img_path, )
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        gc.collect()
    pd_up_h, pd_lf_w = np.int64((np.array(out_shape) - np.array(inner_shape)) / 2)

    print(image.shape)
    ori_shape = image.shape
    pd_bm_h = (out_shape[0] - pd_up_h) - (image.shape[0] % inner_shape[0])
    pd_rt_w = (out_shape[1] - pd_lf_w) - (image.shape[1] % inner_shape[1])

    it_h = np.int64(np.ceil(1.0 * image.shape[0] / inner_shape[0]))
    it_w = np.int64(np.ceil(1.0 * image.shape[1] / inner_shape[1]))

    image_pd = np.pad(image, ((pd_up_h, pd_bm_h), (pd_lf_w, pd_rt_w), (0, 0)), mode='reflect').astype(
        np.float32)  # the image is default a color one
    print(image_pd.shape)
    print((pd_up_h, pd_bm_h), (pd_lf_w, pd_rt_w))
    gc.collect()

    tp1 = np.array(inner_shape[0] - ori_shape[0] % inner_shape[0])
    tp2 = np.array(inner_shape[1] - ori_shape[1] % inner_shape[1])
    if ori_shape[0] % inner_shape[0] == 0:
        tp1 = 0
    if ori_shape[1] % inner_shape[0] == 0:
        tp2 = 0
    # 确定output的size
    out_img = np.zeros((ori_shape[0] + tp1, ori_shape[1] + tp2, out_channel), np.float32)

    image = None
    for ith in range(0, it_h):
        h_start = ith * inner_shape[0]
        count = 1
        for itw in range(0, it_w):
            w_start = itw * inner_shape[1]
            tp_img = image_pd[h_start:h_start + out_shape[0], w_start:w_start + out_shape[1], :]

            # image pre-process
            tp_img = img_pre_process(tp_img.copy(), **kwargs)
            # print('tp_img', tp_img.shape)

            tp_out = pred_fun(tp_img[np.newaxis, :])
            tp_out = np.squeeze(tp_out, axis=0)

            # image post-process
            # tp_out = post-process

            out_img[h_start:h_start + inner_shape[0], w_start:w_start + inner_shape[1], :] = tp_out[pd_up_h:pd_up_h +
                                                                                                            inner_shape[
                                                                                                                0],
                                                                                             pd_lf_w:pd_lf_w +
                                                                                                     inner_shape[1], :]

            print('haha!', h_start, w_start, count)
            count += 1
    return out_img[0:ori_shape[0], 0:ori_shape[1], :]


def pred(test_images):
    args = arg_parser()
    config = sess_config(args)
    with open(r'./cfg.yml', 'r') as file:
        cfg = yaml.load(file)
    # 获取执行python是传入的图片路径
    # path = args.img_path
    height = cfg['height']
    width = cfg['width']
    channel = cfg['channel']
    mean = cfg['mean']
    hed_class = HED(height=height, width=width, channel=channel)
    hed_class.vgg_hed()
    # 有fusion层
    # sides = [tf.sigmoid(hed_class.side1),
    #          tf.sigmoid(hed_class.side2),
    #          tf.sigmoid(hed_class.side3),
    #          tf.sigmoid(hed_class.side4),
    #          tf.sigmoid(hed_class.side5),
    #          tf.sigmoid(hed_class.fused_side)]
    # 没有fusion层
    sides = [tf.sigmoid(hed_class.side1),
             tf.sigmoid(hed_class.side2),
             tf.sigmoid(hed_class.side3),
             tf.sigmoid(hed_class.side4),
             tf.sigmoid(hed_class.side5)]
    sides = 1.0 * tf.add_n(sides) / len(sides)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    # load weights
    saver.restore(sess, './model10132/' + 'vgg16_hed-300')
    # 批处理
    # test_images = os.listdir('./hed_test_data_resize')
    # 单个文件测试
    # test_images=['14.jpg']
    for test_image in test_images:
        # path = './hed_test_data_resize/' + test_image
        path=test_image
        ipt_img = cv2.imread(path, )
        output_img = predict_big_map(img_path=path, out_shape=(448, 448), inner_shape=(224, 224), out_channel=1,
                                     pred_fun=(lambda ipt: sess.run(sides, feed_dict={hed_class.x: ipt})),
                                     mean=cfg['mean'])
        hed = np.squeeze((output_img * 255).astype(np.uint8))
        # 高斯滤波
        hed = cv2.GaussianBlur(hed, (7, 7),0)
        # minarea
        edge_points = []
        final = np.zeros(hed.shape, np.uint8)
        [rows, cols] = hed.shape
        for i in range(rows):
            for j in range(cols):
                if hed[i, j] > 127:
                    final[i,j]=255
                    edge_points.append([j,i])
        fileName = test_image.split('/')[2]
        if len(edge_points)!=0 :
            cnt = np.array(edge_points)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            for point in box:
                cv2.circle(final, (point[0], point[1]), 6, (255, 0, 0))
            cv2.imwrite('./hed_test_result_nofusion_minArea/'+fileName,final)
        else:
            cv2.imwrite('./hed_test_result_nofusion_minArea/'+fileName,final)
    sess.close()

# 这个文件是为了检验有没有fusion层时的结果
dir='./hed_test_data_resize'
fileNames=os.listdir(dir)
hed_paths=[]
for fileName in fileNames:
    imgPath=dir+'/'+fileName
    hed_paths.append(imgPath)
# hed算法传的参数是所有需要用hed的文件名数组
pred(hed_paths)


