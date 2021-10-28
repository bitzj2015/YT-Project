# Steps for data preparation
### 1. Get the metadata for all videos
```
python get_metadata.py --log <log_path> --data <sock_puppet_path> --video <video_id_path> --metadata <video_metadata_path>
```
### 2. Get video transcripts
```
python get_videotext.py --version <dataset_version>
```