from anormal_det_video import *

class model_nohist(model):
    def __init__(self,id=-1,
                 # some recommended parameters
                 size=(256, 256),
                 size_reduce=(16,16), # redundant size for registration
                 n_sigma=6,  # 用于计算正常上界，小于这个系数下的方差直接过滤掉
                 n_sigma_=10,  # 用于缩放异常规模，数值越大越线性区间越大越不容易饱和，但是最大异常响应越小
                 mean_std_lower_bound=(6, 5),  # 统计异常视频的最小值得出，理想情况下（覆盖每个视角的多样图片），对每个视角单独设置
                 top_roi_partition=4e-5,
                 max_jitter=1,
                 # modified parameters below for your case
                 ROI = (0.1, 0.9, 0.1, 0.9), # numpy style
                 alarm_th=240,
                 ):
        super().__init__(id=id, size=size, size_reduce=size_reduce, n_hist=0, alarm_len=0, top_roi_partition=top_roi_partition,
                 n_sigma=n_sigma, n_sigma_=n_sigma_, mean_std_lower_bound=mean_std_lower_bound,
                 max_jitter=max_jitter, ROI = ROI, alarm_th=alarm_th,
                 )
        assert self.ref is not None, "you have to set ref image before inference"

    def apply(self, frame, *args):
        if isinstance(frame,str): frame = cv2.imread(frame, 0)
        self.counter += 1

        frame_ = cv2.resize(frame[self.ROI[0]:self.ROI[1], self.ROI[2]:self.ROI[3]], self.size)
        self.q.append(cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)[None])

        data_flatten = self.get_fea()[:, IDX].permute((0, 2, 3, 1)).view((self.N, -1))
        self.mean_fused, self.cov_fused = self._mean, self._cov
        score_map = self.compute_score(data_flatten).cpu().numpy()

        scores = np.minimum(1, np.maximum(0, score_map - self.base) / self.tau)* 255
        scores = scores.astype(np.uint8)
        score = np.partition(scores.flatten(), self.k_th)[self.k_th:].mean()
        res = np.where((scores < self.alarm_th*0.85) * (scores > self.alarm_th*0.6), 1, 0).sum() / (np.where(scores > self.alarm_th*0.9, 1, 0).sum() + 1e-4) # 异常的明确性（信心），真异常的图像中间值少而高值多，正常图像高数值少，假异常中间值多
        all_idx_h, all_idx_w = np.where(scores>scores.max()*0.8) # 找局部极大异常区域上下左右坐标极值，在该区域内可以用于计算异常占比
        sub_scores = scores[all_idx_h.min():all_idx_h.max()+1, all_idx_w.min():all_idx_w.max()+1] if len(all_idx_h)>4 else np.ones_like(scores)*self.alarm_th*0.7 # 按照上面的极值裁剪异常图区域，当异常点数量不足时设置为全self.alarm_th*0.7的图像，后面的步骤不起作用
        res_ = np.where((sub_scores < self.alarm_th*0.8) * (sub_scores > self.alarm_th*0.5), 1, 0).sum() / (np.where(sub_scores > self.alarm_th*0.9, 1, 0).sum() + 1) # 算裁剪图片上的异常的明确性
        area = np.where(scores > self.alarm_th*0.85, 255, 0).mean() # 算明确的异常区域面积

        if score > self.alarm_th and ((area > 5 and res<1.65) or (area > 2.5 and res < 3.1 and res_ < 0.45)):
            curr_time = datetime.datetime.now()
            savepath = os.path.join(self.exp_path, str(curr_time.minute) + "_" + str(curr_time.second) + "_" + str(curr_time.microsecond))
            anopos = np.where(scores > scores.max()*0.8)
            H, W = self.ROI[1]-self.ROI[0], self.ROI[3]-self.ROI[2]
            luh = (H * (anopos[0].min() * self.image_size[1] / self.wh[1] + self.size_redundancy[1]) / self.size[1] + self.ROI[0]).astype(np.int64)
            luw = (W * (anopos[1].min() * self.image_size[0] / self.wh[0] + self.size_redundancy[0]) / self.size[0] + self.ROI[2]).astype(np.int64)
            rdh = (H * ((anopos[0].max() + 1) * self.image_size[1] / self.wh[1] + self.size_redundancy[1]) / self.size[1] + self.ROI[0]).astype(np.int64)
            rdw = (W * ((anopos[1].max() + 1) * self.image_size[0] / self.wh[0] + self.size_redundancy[0]) / self.size[0] + self.ROI[2]).astype(np.int64)
            cv2.imwrite(savepath + str(self.counter) + '_alarm.jpg', cv2.rectangle(frame, (luw, luh), (rdw,rdh), 255, 10))
            return ['abnormal', luh, luw, rdh - luh, rdw - luw]
        else:
            return ['normal', 0, 0, 0, 0]

if __name__=='__main__':
    detector = model_nohist(ID)
    while True:
        cap = cv2.VideoCapture(ID)
        if cap.isOpened():
            print('VideoCapture get')
        else:
            print('VideoCapture fail')
            cap.release()
            continue

    i = 0
    while True:
        flag, frame = cap.read()
        if flag and frame is not None:
            if YOUR_OWN_TRIGGER_FLAG:
                print(detector.apply(frame, ID))
            i += 1