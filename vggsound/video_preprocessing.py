import pandas as pd
import cv2
import os
import pdb
from tqdm import tqdm

# NOTE: I could probably speed this up with FFMpeg's h264_cuvid decoder using GPU-acceleration, but that requires extra install steps that may not be compatible
# with everyone's machines

class videoReader(object):
    def __init__(self, video_path, frame_interval=1, frame_kept_per_second=1):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.frame_kept_per_second = frame_kept_per_second

        self.vid = cv2.VideoCapture(self.video_path)
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.video_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_len = int(self.video_frames/self.fps)


    def video2frame(self, frame_save_path):
        self.frame_save_path = frame_save_path
        success, image = self.vid.read()
        count = 0
        while success:
            count +=1
            if count % self.frame_interval == 0:
                save_name = '{}/frame_{}_{}.jpg'.format(self.frame_save_path, int(count/self.fps), count)  # filename_second_index
                cv2.imencode('.jpg', image)[1].tofile(save_name)
            success, image = self.vid.read()


    def video2frame_update(self, frame_save_path):
        self.frame_save_path = frame_save_path

        count = 0
        frame_interval = int(self.fps / self.frame_kept_per_second)
        frame_id = 0  # Initialize frame_id outside the loop

        while(count < self.video_frames):
            ret, image = self.vid.read()
            if not ret:
                break

            # This condition ensures we save `frame_kept_per_second` frames per second of video
            if frame_id % frame_interval == 0:
                save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                cv2.imencode('.jpg', image)[1].tofile(save_name)

            frame_id += 1
            # Reset frame_id every second (after counting through fps frames)
            if frame_id == self.fps:
                frame_id = 0

            count += 1


class VGGSound_dataset(object):
    def __init__(self, mode = "train", path_to_dataset = '../data/vggsound/', frame_interval=1, frame_kept_per_second=1):
        self.path_to_video = os.path.join(path_to_dataset, mode)
        self.frame_kept_per_second = frame_kept_per_second
        self.path_to_save = os.path.join(path_to_dataset, mode + '_Image-{:02d}-FPS'.format(self.frame_kept_per_second))
        if not os.path.exists(self.path_to_save):
            os.mkdir(self.path_to_save)
        self.file_list = os.listdir(os.path.join(data_root, mode))

    def extractImage(self):

        for each_video in tqdm(self.file_list):
            video_dir = os.path.join(self.path_to_video, each_video)
            try:
                self.videoReader = videoReader(video_path=video_dir, frame_kept_per_second=self.frame_kept_per_second)

                save_dir = os.path.join(self.path_to_save, each_video)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                self.videoReader.video2frame_update(frame_save_path=save_dir)
            except:
                print('Fail @ {}'.format(each_video))


if __name__ == "__main__": 

    data_root = "../data/vggsound/"

    vggsound_test = VGGSound_dataset(mode="test", path_to_dataset=data_root)
    vggsound_test.extractImage()

    vggsound_train = VGGSound_dataset(mode="train", path_to_dataset=data_root)
    vggsound_train.extractImage()


