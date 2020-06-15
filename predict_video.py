import cv2
from os.path import isdir
from os import mkdir
import util_v2 as ut
from numpy import expand_dims
class pred_classe:
    def __init__(self, class_threshold, iou_threshold, labels, model):
        self.util = ut.classe_util(class_threshold, iou_threshold, labels, model)


    def read_video(self, filename):
        # CAPTURER LA VIDÉO.
        capture = cv2.VideoCapture(filename)
        i=0
        frames = []

        while(True):
            flag, frame = capture.read()
            if flag!=1:
                break
            # AJOUTER LA FRAME RETOURNÉE.
            frames.append(frame)
            i+=1
        return frames

    def draw_boxes_on_img(self, img, pred_boxes, pred_classes, pred_scores, thickness = 1, color = (255, 255, 255), fontScale = .55):
        for i in range(len(pred_boxes)):
            vb = pred_boxes[i]
            xmin, ymin = int(vb[0]), int(vb[1])
            xmax, ymax = int(vb[2]+xmin), int(vb[3]+ymin)
            start_point=(xmin, ymin)
            end_point = (xmax, ymax)

            image = cv2.rectangle(img, (start_point), end_point, color, thickness)

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = start_point
            text = self.util.labels[pred_classes[i]] + ' [' + str(pred_scores[i])[:5] +']'
            img = cv2.putText(img, text , org, font, fontScale, color, thickness, cv2.LINE_AA)
        return img



    def predict_and_write_video(self, filename, dim=(416, 416), directory = 'output'):
        frames = self.read_video(filename)
        height , width , layers =  frames[0].shape
        shape = (height, width)
        video = cv2.VideoWriter(filename[:filename.find('.')]+'_OUTPUT.avi',cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
        i=0
        for f in frames:
            f = cv2.resize(f, dim, interpolation = cv2.INTER_AREA)
            f = f.astype('float32')
            f /= 255.0
            # add a dimension so that we have one sample
            f = expand_dims(f, 0)

            pred_boxes, pred_classes, pred_scores = self.util.predict_and_return(f, shape)
            if(pred_boxes is None):
                frame_2_write = frames[i]
            else:
                frame_2_write = self.draw_boxes_on_img(frames[i], pred_boxes, pred_classes, pred_scores)
            video.write(frame_2_write)
            i+=1
        video.release()
        return i
