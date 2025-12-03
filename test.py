import cv2
from ultralytics import YOLO

# -------------------------------------------------------
# 关键修改点：把 yolov8n.pt 改成 yolov8n-pose.pt
# -------------------------------------------------------
# 这个模型专门用来识别人的骨架（鼻子、肩膀、手肘、膝盖等17个点）
model = YOLO('yolov8n-pose.pt')

# 你的视频路径
video_path = 'F:/pytorchtest/IntellectualCook/man3.mp4'

# 开始预测
# save=True: 同样会把画了骨架的视频保存下来
results = model.predict(source=video_path,
                        stream=True,
                        save=True,
                        device=0)

print("正在加载骨架模型并运行... 按 'q' 键退出")

for r in results:
    # r.plot() 对于 pose 模型，会自动画出 骨架连线 和 关键点
    im_array = r.plot()

    # 显示画面
    cv2.imshow('YOLOv8 Pose Skeleton', im_array)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("处理完成！")