import os
import csv
import patoolib

submission_files = [
    "config.py",
    "dataset.py",
    "dataset_numpy_to_tfrecord.py",
    "model.py",
    "restore_and_evaluate.py",
    "setup.py",
    "Skeleton.py",
    "training.py",
    "utils.py",
]


def create_zip_code_files(output_file="submission_files.zip"):
    patoolib.create_archive(output_file, submission_files)


def create_submission_csv(labels, output_file="submission.csv"):
    with open(output_file, "w") as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["Id", "y"])

        for i, label in enumerate(labels):
            writer.writerow([i + 1, label + 1])


def create_submission_files(labels, out_dir, out_csv_file, out_code_file):
    create_submission_csv(labels, os.path.join(out_dir, out_csv_file))
    create_zip_code_files(os.path.join(out_dir, out_code_file))
