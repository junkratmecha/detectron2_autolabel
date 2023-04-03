# [detectron2](ttps://github.com/facebookresearch/detectron2)での自動ラベリング?
を利用して自動ラベリングしたいな、yoloやcocoのラベル作成したいなみたいなディレクトリ  
弱弱なのでご指摘やアドバイスは以下へお願いします。  
[Twitterのjunkratmecha](https://twitter.com/junkratmecha)

# 使い方
本家のリポジトリのdemoフォルダやmyprogフォルダなどに置けば動くはず(23/4 windows)スクリプトです。  
私のローカルwindows環境(23/4)　Python 3.9.13/torch →1.13.1+cu116/cuda → 11.6/cudnn → v8.3.0

- 構築
    * git clone https://github.com/facebookresearch/detectron2
    * pip install -e detectron2
    * 行ける人は行ける。行けない人は行けない。自分はtorch周りで若干ガチャガチャした記憶有り。
    * ※cudaとかtorch周りは[colabなどで動かしている記事](https://www.ushiji.online/detectron2)をご参照
- 学習
    * [coco-annotator](https://github.com/jsbroks/coco-annotator)などでアノテーションした画像/classes.txt/coco.jsonをimg_trainフォルダなどに置きます。(imgs.zipはサンプル)
    * detectron2/demo/train.pyを実行。ckptフォルダに重みなどができます。
- 推論
    * detectron2/demo/inference.pyを実行。labelsフォルダにcoco/yolo/pascalvocのラベルができます。

# デモ動かし方(cd detectron2/demo/demo)
- 画像  
    py demo.py --input input.jpg --output output.jpg --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --opts MODEL.WEIGHTS  detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
- 動画  
    py demo.py --video-input input.mp4 --output output.mp4 --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
- webcam  
    py demo.py --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

# その他
 * 学習ログ py -m tensorboard.main --logdir ./demo/ckpt/img_train
 * cc0画像収集 pexels_api_downloader.py/pixabay_api_downloader.py(要api_key。一応CC0集めれるがkeywordで欲しいのが集まらない)

# マイ記事 + 参考になった記事
- [マイ記事](https://zenn.dev/junkratmecha/articles/516e884a3116b6)
- [Detectron2 を使ってみよう：Pre-Trainedモデルの使い方説明](https://www.ushiji.online/detectron2#keypoint-detection)  
- [Detectron2で小銭を数える](https://qiita.com/bear_montblanc/items/5bb1ad3506718120682d)