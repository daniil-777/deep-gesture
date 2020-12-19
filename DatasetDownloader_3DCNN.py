import os
import json
import shutil
import threading
from pytube import YouTube
from preprocessing_videos import Preprocessing


class DatasetDownloader:
    def __init__(self, json_path, dataset_path, batch_size):
        self.dataset_path = dataset_path
        self.tmp_path = dataset_path + "/tmp"
        self.batch_size = batch_size

        self.json_data = self.read_json(json_path)
        self.current_index = 0
        self.youtube_link = "https://www.youtube.com/watch?v="

        self.first_batch = True
        self.no_more_data = False
        self.download_thread = None

        # Preprocessing
        self.preprocesser = Preprocessing(100, 100, 16, overlap=True)

        # Clear dataset dir
        path = os.path.join(self.dataset_path, "train_set.tfrecord")
        if os.path.isfile(path):
            os.unlink(path)

        # Check Validation Set Exists. If not, download it
        path = os.path.join(self.dataset_path, "validation_set.tfrecord")
        if not os.path.isfile(path):
            self.download_validation()

        # Make tmp dir
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
        else:
            self.clear_dir(self.tmp_path)

    def get_next_batch(self):
        """
        Adds the next batch of data to the dataset dir and launches a background thread to download the next batch.
        Returns -1 when there is no more data
        """

        # First Batch
        if self.first_batch:
            self.download_batch_data()
            self.first_batch = False

        # No More Data
        if self.no_more_data:
            return -1

        # Clear dataset dir
        path = os.path.join(self.dataset_path, "train_set.tfrecord")
        if os.path.isfile(path):
            os.unlink(path)

        # self.clear_dir(self.dataset_path)

        # Wait for Download to finish in case it's still active.
        if self.download_thread != None:
            self.download_thread.join()

        # Move Files to dataset dir
        for f in os.listdir(self.tmp_path):
            if f.endswith(".tfrecord"):
                shutil.move(os.path.join(self.tmp_path, f), self.dataset_path)

        if self.current_index < len(self.json_data):
            self.download_thread = threading.Thread(target=self.download_batch_data)
            self.download_thread.start()
        else:
            self.no_more_data = True

    def download_batch_data(self):
        self.clear_dir(self.tmp_path)

        # Download Code
        label_file = self.tmp_path + "/labels.txt"
        count = 0

        with open(label_file, "w") as f:
            while count < self.batch_size and self.current_index < len(self.json_data):
                datum = self.json_data[self.current_index]
                self.current_index += 1
                videolink = self.youtube_link + datum["id"]
                try:
                    yt = YouTube(videolink)
                    yt.streams.get_lowest_resolution().download(
                        self.tmp_path, filename=datum["id"]
                    )
                    videoname = os.path.join(
                        self.tmp_path, "{:s}.{:s}".format(datum["id"], "mp4")
                    )
                    videolabel = datum["label487"]
                    f.write(
                        "{:s} {:s}\n".format(videoname, " ".join(map(str, videolabel)))
                    )
                    f.flush()

                    count += 1
                except:
                    continue

        train_set = self.preprocesser.get_dataset_of_videos(label_file)
        tfrecordfile_train = self.preprocesser.to_TFRecords(
            train_set, "train_set", self.tmp_path
        )

    def download_validation(self):
        print("Downloading Validation Set")
        # path = os.path.join(self.dataset_path, "validation_set.tfrecord")

        json_data = self.read_json("validation_set.json")
        current_index = 0

        # Download Code
        label_file = self.dataset_path + "/labels.txt"
        count = 0

        with open(label_file, "w") as f:
            while current_index < len(json_data):
                datum = json_data[current_index]
                current_index += 1
                videolink = self.youtube_link + datum["id"]
                try:
                    yt = YouTube(videolink)
                    yt.streams.get_lowest_resolution().download(
                        self.dataset_path, filename=datum["id"]
                    )
                    videoname = os.path.join(
                        self.dataset_path, "{:s}.{:s}".format(datum["id"], "mp4")
                    )
                    videolabel = datum["label487"]
                    f.write(
                        "{:s} {:s}\n".format(videoname, " ".join(map(str, videolabel)))
                    )
                    f.flush()

                    count += 1
                except:
                    continue

        validation_set = self.preprocesser.get_dataset_of_videos(label_file)
        tfrecordfile_train = self.preprocesser.to_TFRecords(
            validation_set, "validation_set", self.dataset_path
        )

        # Clear Video Files
        for f in os.listdir(self.dataset_path):
            path = os.path.join(self.dataset_path, f)
            try:
                if os.path.isfile(path) and not f.endswith(".tfrecord"):
                    os.unlink(path)
            except Exception as e:
                print(e)

    def read_json(self, json_file):
        with open(json_file) as data_file:
            data = json.load(data_file)

        return data

    def clear_dir(self, directory):
        """
        Removes all files in the given directory except folders and sub-files
        """
        for f in os.listdir(directory):
            path = os.path.join(directory, f)
            try:
                if os.path.isfile(path):
                    os.unlink(path)
            except Exception as e:
                print(e)


import time

if __name__ == "__main__":
    loader = DatasetDownloader(
        "C:\\Users\\guitb\\LinuxWorkspace\\data\\sports1m_json\\sports1m_test.json",
        "C:\\Users\\guitb\\LinuxWorkspace\\data\\sports1m_test",
        4,
    )
    loader.get_next_batch()
    print("Waiting 20sec")
    time.sleep(20)
    print("Going again!")
    loader.get_next_batch()
