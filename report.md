# DCTベースのプログレッシブ・テキスト圧縮 — 実験レポート

(実験進行中 — 各Phase完了ごとに更新)

## 1. 実験設定の要約

- TODO: モデル、データ量、学習時間、ハードウェア

## 2. 品質 vs K 曲線

- TODO: results/curves.png

## 3. 定性サンプル

- TODO: results/samples.md 参照

## 4. 仮説の判定

- H1 (プログレッシブ性): TODO
- H2 (グレースフルな劣化): TODO
- H3 (学習デコーダの必要性): TODO
- H4 (DCT順序の意味): TODO

## 5. encoder_layer=0 と 4 の比較

- TODO

## 6. 設計判断ログ

実装中に独断で行った設計判断と理由:

1. **隠れ状態のグローバルスケール正規化 (h_scale)**: エンコーダ隠れ状態のRMSをキャリブレーションバッチ(256件)で測定し、単一スカラーで除算してからDCTをかける。DCT係数(特に第0係数は平均×√nでスケールが大きい)をプロジェクション層に入れる際の数値レンジを抑え、学習初期の不安定を避けるため。スカラー1個なので情報は失われない。
2. **val/testの取得元**: タスクは train split からの分割とも読めるが、リークを避けるため validation 2000件は validation split、test 2000件は test split から取得した。
3. **チャンク長**: 32〜128トークンの一様乱数でストリームを切断。長さ条件埋め込みの全レンジを学習で踏ませるため。
4. **プレフィックスへの注意マスク**: 係数トークン列にも通常の因果マスクを適用(z_kはz_{<k}のみ参照)。双方向化はLM改造が必要なため見送り。
5. **BOS**: Qwen2.5はBOSトークンを持たないため、学習可能なBOSベクトル(d_model次元)を導入。
6. **係数インデックス埋め込みのテーブルサイズ**: B3はランダムなK個の係数(インデックス最大n-1=127)を使うため、テーブルはK_max=64ではなくn_max=128エントリで全バリアント共通とした。
7. **生成終了条件**: 圧縮表現は(Z, n)の組であり nは既知なので、EOSではなく「ちょうどnトークン生成」で打ち切る。
8. **B1の最近傍復号**: コサイン類似度で語彙埋め込みとマッチング。enc0系のB1は生埋め込み(正規化なし)を使用。
9. **エンコーダの部分フォワード**: 第L層までで早期打ち切りする自前実装。起動時にフルフォワードとの一致を自動検証し、不一致なら自動でフルフォワードにフォールバックする。
10. **学習精度**: 重みはfp32で保持し、forward/backwardはbf16 autocast。LoRA学習の数値安定性のため。

## 7. 失敗・予想外の挙動

- datasets 5.0 では `wikitext` が名前空間必須になっており `Salesforce/wikitext` に変更が必要だった。
- TODO: 実験中に発見したことを追記

## 8. 再現方法

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch transformers peft datasets scipy sentence-transformers matplotlib sacrebleu pyyaml accelerate
scripts/run_sanity.sh   # Phase 1 サニティチェック
scripts/run_train.sh    # Phase 2 学習 (main/enc0/b2/b3)
scripts/run_eval.sh     # Phase 3 評価
```
